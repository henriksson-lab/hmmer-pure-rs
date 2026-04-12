//! Search pipeline using SIMD MSV + Viterbi + Forward filters.

use crate::bg::Bg;
use crate::profile::*;
use crate::sequence::Sequence;
#[cfg(target_arch = "x86_64")]
use crate::simd::msv_filter::{msv_filter, MsvResult};
use crate::simd::oprofile::OProfile;
#[cfg(target_arch = "x86_64")]
use crate::simd::vit_filter::{viterbi_filter, VitResult};
use crate::stats;
use crate::tophits::*;

/// Pipeline configuration and state.
pub struct Pipeline {
    // Thresholds
    pub f1: f64,  // MSV filter threshold (default 0.02)
    pub f2: f64,  // Viterbi filter threshold (default 1e-3)
    pub f3: f64,  // Forward filter threshold (default 1e-5)

    // Reporting thresholds
    pub e_value_threshold: f64,      // per-sequence E-value (default 10.0)
    pub dom_e_value_threshold: f64,  // per-domain E-value (default 10.0)
    pub inc_e: f64,                  // inclusion E-value (default 0.01)
    pub inc_dome: f64,               // domain inclusion E-value (default 0.01)

    // Flags
    pub do_biasfilter: bool,
    pub do_max: bool, // if true, skip MSV/Viterbi filters

    // Statistics
    pub n_past_msv: u64,
    pub n_past_bias: u64,
    pub n_past_vit: u64,
    pub n_past_fwd: u64,
    pub n_targets: u64,
    pub z: f64,    // database size (number of sequences) for E-value
    pub domz: f64, // domain database size

    // E-value parameters (from HMM)
    pub evparam: [f32; 6],

}

impl Pipeline {
    pub fn new() -> Self {
        Pipeline {
            f1: 0.02,
            f2: 1e-3,
            f3: 1e-5,
            e_value_threshold: 10.0,
            dom_e_value_threshold: 10.0,
            inc_e: 0.01,
            inc_dome: 0.01,
            do_biasfilter: true,
            do_max: false,
            n_past_msv: 0,
            n_past_bias: 0,
            n_past_vit: 0,
            n_past_fwd: 0,
            n_targets: 0,
            z: 0.0,
            domz: 0.0,
            evparam: [0.0; 6],
        }
    }

    /// Configure the pipeline for a new model.
    pub fn new_model(&mut self, gm: &Profile) {
        self.evparam = gm.evparam;
    }

    /// Run the pipeline on a single sequence.
    /// Uses SIMD MSV filter for the first stage, then SIMD Viterbi and Forward.
    /// `om` is the SIMD-optimized profile, `hmm` is needed for domain definition.
    pub fn run(
        &mut self,
        gm: &Profile,
        om: &OProfile,
        bg: &Bg,
        hmm: &crate::hmm::Hmm,
        sq: &Sequence,
        th: &mut TopHits,
    ) -> bool {
        let l = sq.n;
        if l == 0 {
            return false;
        }

        self.n_targets += 1;

        // Null model score
        let null_sc = bg.null_one(l);

        if !self.do_max {
            // Stage 1: SIMD MSV filter
            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("sse2") {
                    let msv_result = unsafe { msv_filter(&sq.dsq, l, om) };
                    let msv_sc = match msv_result {
                        MsvResult::Ok(sc) => sc,
                        MsvResult::Overflow => f32::INFINITY,
                    };

                    if msv_sc != f32::INFINITY {
                        let msv_pval = msv_pvalue(msv_sc, null_sc, &self.evparam);
                        if msv_pval > self.f1 {
                            return false;
                        }
                    }
                    // Overflow = very high score, always passes
                }
            }
            self.n_past_msv += 1;
            self.n_past_bias += 1;

            // Stage 2: SIMD Viterbi filter
            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("sse2") {
                    let vit_result = unsafe { viterbi_filter(&sq.dsq, l, om) };
                    let vit_sc = match vit_result {
                        VitResult::Ok(sc) => sc,
                        VitResult::Overflow => f32::INFINITY,
                    };

                    if vit_sc != f32::INFINITY {
                        let vit_pval = viterbi_pvalue(vit_sc, null_sc, &self.evparam);
                        if vit_pval > self.f2 {
                            return false;
                        }
                    }
                }
            }
            self.n_past_vit += 1;
        } else {
            self.n_past_msv += 1;
            self.n_past_bias += 1;
            self.n_past_vit += 1;
        }

        // Stage 3: SIMD Forward parser
        let fwd_sc;
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("sse2") {
                fwd_sc = unsafe { crate::simd::fwd_filter::forward_parser(&sq.dsq, l, om) };
            } else {
                let mut gx = crate::dp::gmx::Gmx::new(gm.m, l);
                fwd_sc = crate::dp::generic_fwdback::g_forward(&sq.dsq, l, gm, &mut gx);
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let mut gx = crate::dp::gmx::Gmx::new(gm.m, l);
            fwd_sc = crate::dp::generic_fwdback::g_forward(&sq.dsq, l, gm, &mut gx);
        }

        let fwd_pval = forward_pvalue(fwd_sc, null_sc, &self.evparam);
        if !self.do_max && fwd_pval > self.f3 {
            return false;
        }
        self.n_past_fwd += 1;

        // Sequence passes all filters — run domain definition
        let bitscore = (fwd_sc - null_sc) / std::f32::consts::LN_2;
        let lnp = fwd_pval.ln();

        // Domain definition: find domain envelopes using posterior decoding
        let (domains, nexpected, seq_bias) =
            crate::domaindef::define_domains(&sq.dsq, l, gm, hmm, bg, null_sc);

        let seq_bias_bits = (seq_bias / std::f32::consts::LN_2).max(0.0);

        let hit = th.create_next_hit();
        hit.name = sq.name.clone();
        hit.acc = sq.acc.clone();
        hit.desc = sq.desc.clone();
        hit.score = bitscore;
        hit.bias = seq_bias_bits;
        hit.pre_score = bitscore;
        hit.sum_score = bitscore;
        hit.lnp = lnp;
        hit.pre_lnp = lnp;
        hit.sum_lnp = lnp;
        hit.sortkey = lnp;
        hit.nexpected = nexpected;
        hit.ndom = domains.len();
        hit.dcl = domains;

        true
    }
}

/// Calculate MSV p-value from raw score.
fn msv_pvalue(msv_sc: f32, null_sc: f32, evparam: &[f32; 6]) -> f64 {
    let score = (msv_sc - null_sc) / std::f32::consts::LN_2;
    let mu = evparam[crate::hmm::P7_MMU] as f64;
    let lambda = evparam[crate::hmm::P7_MLAMBDA] as f64;
    stats::gumbel::surv(score as f64, mu, lambda)
}

/// Calculate Viterbi p-value from raw score.
fn viterbi_pvalue(vit_sc: f32, null_sc: f32, evparam: &[f32; 6]) -> f64 {
    let score = (vit_sc - null_sc) / std::f32::consts::LN_2;
    let mu = evparam[crate::hmm::P7_VMU] as f64;
    let lambda = evparam[crate::hmm::P7_VLAMBDA] as f64;
    stats::gumbel::surv(score as f64, mu, lambda)
}

/// Calculate Forward p-value from raw score.
fn forward_pvalue(fwd_sc: f32, null_sc: f32, evparam: &[f32; 6]) -> f64 {
    let score = (fwd_sc - null_sc) / std::f32::consts::LN_2;
    let tau = evparam[crate::hmm::P7_FTAU] as f64;
    let lambda = evparam[crate::hmm::P7_FLAMBDA] as f64;
    stats::exponential::surv(score as f64, tau, lambda)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alphabet::Alphabet;
    use std::path::Path;

    #[test]
    fn test_pipeline_finds_hit() {
        let hmm = crate::hmmfile::read_hmm_file(Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/hmmer/testsuite/20aa.hmm"
        )))
        .unwrap()
        .into_iter()
        .next()
        .unwrap();
        let abc = Alphabet::new(hmm.abc_type);
        let mut bg = Bg::new(&abc);
        let mut gm = Profile::new(hmm.m, &abc);
        profile_config(&hmm, &bg, &mut gm, 20, P7_LOCAL);
        let mut om = OProfile::convert(&gm);

        let mut pli = Pipeline::new();
        pli.new_model(&gm);
        let mut th = TopHits::new();

        let dsq = abc.digitize(b"ACDEFGHIKLMNPQRSTVWY");
        let sq = Sequence {
            name: "test".to_string(),
            acc: String::new(),
            desc: String::new(),
            dsq,
            n: 20,
            l: 20,
        };

        bg.set_length(sq.n);
        reconfig_length(&mut gm, sq.n as i32);
        om.reconfig_length(sq.n as i32);

        // For this short perfectly-matching sequence, use --max to bypass filters
        pli.do_max = true;
        let hit = pli.run(&gm, &om, &bg, &hmm, &sq, &mut th);
        assert!(hit, "Pipeline should find a hit for matching sequence");
        assert_eq!(th.hits.len(), 1);
        assert!(th.hits[0].score > 0.0);
    }
}
