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

/// How Z/domZ was set.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ZSetBy {
    /// Automatically counted from number of targets.
    Ntargets,
    /// Explicitly set by user via -Z / --domZ.
    Option,
}

/// Model-specific bit-score cutoff mode (--cut_ga, --cut_nc, --cut_tc).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BitCutoff {
    None,
    GA,
    TC,
    NC,
}

/// Pipeline configuration and state.
pub struct Pipeline {
    // Thresholds
    pub f1: f64, // MSV filter threshold (default 0.02)
    pub f2: f64, // Viterbi filter threshold (default 1e-3)
    pub f3: f64, // Forward filter threshold (default 1e-5)

    // Reporting thresholds (E-value based)
    pub e_value_threshold: f64,     // per-sequence E-value (default 10.0)
    pub dom_e_value_threshold: f64, // per-domain E-value (default 10.0)
    pub inc_e: f64,                 // inclusion E-value (default 0.01)
    pub inc_dome: f64,              // domain inclusion E-value (default 0.01)

    // Reporting thresholds (score based, set by -T/--domT/--incT/--incdomT or --cut_*)
    pub t: Option<f32>,         // per-sequence score threshold
    pub dom_t: Option<f32>,     // per-domain score threshold
    pub inc_t: Option<f32>,     // inclusion score threshold
    pub inc_dom_t: Option<f32>, // domain inclusion score threshold

    // Whether to use E-value or score thresholds
    pub by_e: bool,        // report by E-value (default true)
    pub dom_by_e: bool,    // report domains by E-value (default true)
    pub inc_by_e: bool,    // include by E-value (default true)
    pub incdom_by_e: bool, // include domains by E-value (default true)

    // Model-specific bit cutoffs
    pub use_bit_cutoffs: BitCutoff,

    // Flags
    pub do_biasfilter: bool,
    pub do_null2: bool,
    pub do_max: bool, // if true, skip MSV/Viterbi filters
    pub do_alignment: bool,
    pub do_alignment_display: bool,
    pub seed: u32, // RNG seed for stochastic traceback (default 42, 0 = arbitrary)

    // Statistics
    pub n_past_msv: u64,
    pub n_past_bias: u64,
    pub n_past_vit: u64,
    pub n_past_fwd: u64,
    pub n_targets: u64,
    pub z: f64,    // database size (number of sequences) for E-value
    pub domz: f64, // domain database size
    pub z_setby: ZSetBy,
    pub domz_setby: ZSetBy,

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
            t: None,
            dom_t: None,
            inc_t: None,
            inc_dom_t: None,
            by_e: true,
            dom_by_e: true,
            inc_by_e: true,
            incdom_by_e: true,
            use_bit_cutoffs: BitCutoff::None,
            do_biasfilter: true,
            do_null2: true,
            do_max: false,
            do_alignment: true,
            do_alignment_display: true,
            seed: 42,
            n_past_msv: 0,
            n_past_bias: 0,
            n_past_vit: 0,
            n_past_fwd: 0,
            n_targets: 0,
            z: 0.0,
            domz: 0.0,
            z_setby: ZSetBy::Ntargets,
            domz_setby: ZSetBy::Ntargets,
            evparam: [0.0; 6],
        }
    }

    /// Configure the pipeline for a new model.
    pub fn new_model(&mut self, gm: &Profile) {
        self.evparam = gm.evparam;
    }

    /// Apply model-specific bit-score cutoffs (--cut_ga, --cut_nc, --cut_tc).
    /// Called after new_model() when use_bit_cutoffs is set.
    /// Returns Err if the model doesn't have the requested cutoff annotation.
    pub fn new_model_thresholds(
        &mut self,
        cutoffs: &[f32; crate::hmm::NCUTOFFS],
    ) -> Result<(), String> {
        use crate::hmm::*;
        match self.use_bit_cutoffs {
            BitCutoff::GA => {
                if cutoffs[P7_GA1] == CUTOFF_UNSET {
                    return Err("GA cutoff not set in model".to_string());
                }
                self.t = Some(cutoffs[P7_GA1]);
                self.inc_t = Some(cutoffs[P7_GA1]);
                self.dom_t = Some(cutoffs[P7_GA2]);
                self.inc_dom_t = Some(cutoffs[P7_GA2]);
            }
            BitCutoff::TC => {
                if cutoffs[P7_TC1] == CUTOFF_UNSET {
                    return Err("TC cutoff not set in model".to_string());
                }
                self.t = Some(cutoffs[P7_TC1]);
                self.inc_t = Some(cutoffs[P7_TC1]);
                self.dom_t = Some(cutoffs[P7_TC2]);
                self.inc_dom_t = Some(cutoffs[P7_TC2]);
            }
            BitCutoff::NC => {
                if cutoffs[P7_NC1] == CUTOFF_UNSET {
                    return Err("NC cutoff not set in model".to_string());
                }
                self.t = Some(cutoffs[P7_NC1]);
                self.inc_t = Some(cutoffs[P7_NC1]);
                self.dom_t = Some(cutoffs[P7_NC2]);
                self.inc_dom_t = Some(cutoffs[P7_NC2]);
            }
            BitCutoff::None => {}
        }

        if self.use_bit_cutoffs != BitCutoff::None {
            self.by_e = false;
            self.dom_by_e = false;
            self.inc_by_e = false;
            self.incdom_by_e = false;
        }
        Ok(())
    }

    /// Convenience wrapper: run the pipeline without requiring mutable profiles.
    /// Clones the profile and oprofile internally (~200KB per call).
    /// For high-throughput use, prefer `run()` with `&mut` profiles.
    pub fn run_cloned(
        &mut self,
        gm: &Profile,
        om: &OProfile,
        bg: &Bg,
        hmm: &crate::hmm::Hmm,
        sq: &Sequence,
        th: &mut TopHits,
    ) -> bool {
        let mut gm = gm.clone();
        let mut om = om.clone();
        self.run(&mut gm, &mut om, bg, hmm, sq, th)
    }

    /// Run the pipeline on a single sequence.
    /// Uses SIMD MSV filter for the first stage, then SIMD Viterbi and Forward.
    /// Reconfigures profile/oprofile length model per sequence (matching C).
    /// `om` is the SIMD-optimized profile, `hmm` is needed for domain definition.
    pub fn run(
        &mut self,
        gm: &mut Profile,
        om: &mut OProfile,
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

        // Reconfigure length model for this sequence (matching C's p7_Pipeline)
        reconfig_length(gm, l as i32);
        om.reconfig_length(l as i32);

        // Null model score
        let null_sc = bg.null_one(l);

        // Compute bias filter score (used as null baseline when bias filter is on)
        let filtersc = if self.do_biasfilter {
            bg.filter_score(&sq.dsq, l)
        } else {
            null_sc
        };

        if !self.do_max {
            // Stage 1: SIMD MSV filter
            let mut usc = f32::INFINITY;
            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("sse2") {
                    usc = match unsafe { msv_filter(&sq.dsq, l, om) } {
                        MsvResult::Ok(sc) => sc,
                        MsvResult::Overflow => f32::INFINITY,
                    };

                    if usc != f32::INFINITY {
                        let msv_pval = msv_pvalue(usc, null_sc, &self.evparam);
                        if msv_pval > self.f1 {
                            return false;
                        }
                    }
                }
            }
            self.n_past_msv += 1;

            // Stage 1b: Bias composition filter
            if self.do_biasfilter && usc != f32::INFINITY {
                let bias_pval = msv_pvalue(usc, filtersc, &self.evparam);
                if bias_pval > self.f1 {
                    return false;
                }
            }
            self.n_past_bias += 1;

            // Stage 2: SIMD Viterbi filter (uses filtersc as baseline, matching C)
            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("sse2") {
                    let vit_result = unsafe { viterbi_filter(&sq.dsq, l, om) };
                    let vit_sc = match vit_result {
                        VitResult::Ok(sc) => sc,
                        VitResult::Overflow => f32::INFINITY,
                    };

                    if vit_sc != f32::INFINITY {
                        let vit_pval = viterbi_pvalue(vit_sc, filtersc, &self.evparam);
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

        // Stage 3: full Forward parser.
        let fwd_sc;
        #[cfg(target_arch = "x86_64")]
        let mut fwd_pmx_for_domains = None;
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("sse2") {
                let mut fwd_pmx = crate::simd::probmx::ProbMx::new(l);
                fwd_sc = unsafe {
                    crate::simd::fwd_filter::forward_parser_pmx(&sq.dsq, l, om, &mut fwd_pmx)
                };
                fwd_pmx_for_domains = Some(fwd_pmx);
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

        let fwd_pval = forward_pvalue(fwd_sc, filtersc, &self.evparam);
        if !self.do_max && fwd_pval > self.f3 {
            return false;
        }
        self.n_past_fwd += 1;

        // Sequence passes all filters — run domain definition
        let (mut domains, nexpected, seq_correction_sum, domain_stats) =
            crate::domaindef::define_domains(
                &sq.dsq,
                l,
                gm,
                Some(om),
                #[cfg(target_arch = "x86_64")]
                fwd_pmx_for_domains.as_ref(),
                #[cfg(not(target_arch = "x86_64"))]
                None,
                Some(fwd_sc),
                hmm,
                bg,
                null_sc,
                self.seed,
                self.do_alignment,
                self.do_alignment_display,
            );
        if domains.is_empty() {
            return false;
        }

        // Sequence-level scoring
        // Match C HMMER's two sequence scores:
        // 1. the full-sequence Forward score corrected by summed n2sc[]
        // 2. a reconstruction score from individually rescored domains
        let omega = 1.0_f32 / 256.0;
        let mut pre_score = (fwd_sc - null_sc) / std::f32::consts::LN_2;
        let seqbias = if self.do_null2 {
            crate::logsum::p7_flogsum(0.0, omega.ln() + seq_correction_sum)
        } else {
            0.0
        };
        let mut seq_score = (fwd_sc - (null_sc + seqbias)) / std::f32::consts::LN_2;

        let mut sum_env = 0.0_f32;
        let mut sum_correction = 0.0_f32;
        let mut ld = 0usize;
        for dom in &domains {
            let use_domain = if self.do_null2 {
                dom.envsc - dom.domcorrection > 0.0
            } else {
                dom.envsc > 0.0
            };
            if use_domain {
                sum_env += dom.envsc;
                sum_correction += dom.domcorrection;
                ld += (dom.jenv - dom.ienv + 1) as usize;
            }
        }
        let sum_score_nats = sum_env + (l - ld) as f32 * (l as f32 / (l as f32 + 3.0)).ln();
        let sum_bias = if self.do_null2 {
            crate::logsum::p7_flogsum(0.0, omega.ln() + sum_correction)
        } else {
            0.0
        };
        let pre2_score = (sum_score_nats - null_sc) / std::f32::consts::LN_2;
        let sum_score = (sum_score_nats - (null_sc + sum_bias)) / std::f32::consts::LN_2;
        if ld > 0 && sum_score > seq_score {
            seq_score = sum_score;
            pre_score = pre2_score;
        }
        let seq_bias_bits = (pre_score - seq_score).max(0.0);

        if !self.do_null2 {
            let tau = gm.evparam[crate::hmm::P7_FTAU] as f64;
            let lambda = gm.evparam[crate::hmm::P7_FLAMBDA] as f64;
            for dom in &mut domains {
                let env_len = (dom.jenv - dom.ienv + 1) as usize;
                let length_correction = (l - env_len) as f32 * (l as f32 / (l as f32 + 3.0)).ln();
                dom.dombias = 0.0;
                dom.bitscore = (dom.envsc + length_correction - null_sc) / std::f32::consts::LN_2;
                dom.lnp = crate::stats::exponential::surv(dom.bitscore as f64, tau, lambda).ln();
            }
        }

        // P-values
        let mu_f = self.evparam[crate::hmm::P7_FTAU] as f64;
        let lam_f = self.evparam[crate::hmm::P7_FLAMBDA] as f64;
        let lnp = stats::exponential::surv(seq_score as f64, mu_f, lam_f).ln();
        let pre_lnp = stats::exponential::surv(pre_score as f64, mu_f, lam_f).ln();

        let hit = th.create_next_hit();
        hit.name = sq.name.clone();
        hit.acc = sq.acc.clone();
        hit.desc = sq.desc.clone();
        hit.n = sq.n;
        hit.score = seq_score;
        hit.bias = seq_bias_bits;
        hit.pre_score = pre_score;
        hit.sum_score = sum_score;
        hit.lnp = lnp;
        hit.pre_lnp = pre_lnp;
        hit.sum_lnp = stats::exponential::surv(sum_score as f64, mu_f, lam_f).ln();
        hit.sortkey = lnp;
        hit.nexpected = nexpected;
        hit.nregions = domain_stats.nregions;
        hit.nclustered = domain_stats.nclustered;
        hit.noverlaps = domain_stats.noverlaps;
        hit.nenvelopes = domain_stats.nenvelopes;
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

        // For this short perfectly-matching sequence, use --max to bypass filters
        pli.do_max = true;
        let hit = pli.run(&mut gm, &mut om, &bg, &hmm, &sq, &mut th);
        assert!(hit, "Pipeline should find a hit for matching sequence");
        assert_eq!(th.hits.len(), 1);
        assert!(th.hits[0].score > 0.0);
    }
}
