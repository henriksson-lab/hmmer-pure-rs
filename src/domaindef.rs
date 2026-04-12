//! Simplified domain definition using posterior probabilities.
//! Partial port of p7_domaindef.c — handles single-domain regions.

use crate::alphabet::{Alphabet, Dsq};
use crate::bg::Bg;
use crate::dp::generic_backward::g_backward;
use crate::dp::generic_decoding::{domain_occupancy, g_decoding};
use crate::dp::generic_fwdback::g_forward;
use crate::dp::generic_stotrace::g_stochastic_trace;
use crate::dp::generic_viterbi::g_viterbi;
use crate::dp::generic_null2;
use crate::dp::gmx::*;
use crate::hmm::Hmm;
use crate::profile::*;
use crate::spensemble::{self, SegmentPair, ClusterParams};
use crate::tophits::{AliDisplay, Domain};
use crate::trace::State;
use crate::util::random::MersenneTwister;

/// Thresholds for domain definition
const RT1: f32 = 0.25; // region trigger
const RT2: f32 = 0.10; // region extent

/// Detect domain regions from posterior decoding.
/// Returns a list of (ienv, jenv) domain envelope coordinates (1-based).
pub fn find_domain_regions(mocc: &[f32], l: usize) -> Vec<(usize, usize)> {
    let mut regions = Vec::new();
    let mut i = 1;

    while i <= l {
        // Look for region start
        if mocc[i] >= RT1 {
            let start = i;
            // Extend region while occupancy stays above threshold
            while i <= l && mocc[i] >= RT2 {
                i += 1;
            }
            let end = i - 1;
            if end >= start {
                regions.push((start, end));
            }
        }
        i += 1;
    }

    regions
}

/// Score a domain envelope using Forward on the isolated region.
/// Returns (bitscore, fwd_score_nats, domain) for the envelope.
pub fn score_domain_envelope(
    dsq: &[Dsq],
    l: usize,
    gm: &Profile,
    hmm: &Hmm,
    ienv: usize,
    jenv: usize,
    null_sc: f32,
) -> Domain {
    debug_assert!(ienv >= 1 && jenv <= l, "envelope must be within sequence");
    debug_assert!(null_sc.is_finite(), "null score must be finite");
    let env_len = jenv - ienv + 1;

    // Compute Forward score on the envelope region
    // Create a sub-profile configured for the envelope length
    let mut env_gm = gm.clone();
    reconfig_unihit(&mut env_gm, env_len as i32);

    let mut gx = Gmx::new(gm.m, env_len);

    // Create a sub-sequence view (still 1-based)
    let mut sub_dsq = vec![crate::alphabet::DSQ_SENTINEL];
    sub_dsq.extend_from_slice(&dsq[ienv..=jenv]);
    sub_dsq.push(crate::alphabet::DSQ_SENTINEL);

    let env_fwd_sc = g_forward(&sub_dsq, env_len, &env_gm, &mut gx);

    // Null score for envelope length
    // Null score: geometric model with p1 = L/(L+1) where L = envelope length
    let p1 = env_len as f32 / (env_len as f32 + 1.0);
    let env_null = env_len as f32 * p1.ln() + (1.0 - p1).ln();

    // Compute per-domain null2 bias from envelope-local posteriors
    let mut env_bck = Gmx::new(gm.m, env_len);
    g_backward(&sub_dsq, env_len, &env_gm, &mut env_bck);
    let mut env_pp = Gmx::new(gm.m, env_len);
    crate::dp::generic_decoding::g_decoding(&env_gm, &gx, &env_bck, &mut env_pp);
    let env_null2 = generic_null2::null2_by_expectation(&env_gm, hmm, &env_pp, &crate::bg::AMINO_FREQUENCIES);
    let dom_bias = generic_null2::null2_score(&env_null2, &sub_dsq, 1, env_len);
    let dom_bias_bits = (dom_bias / std::f32::consts::LN_2).max(0.0);

    let bitscore = (env_fwd_sc - env_null) / std::f32::consts::LN_2;

    let tau = gm.evparam[crate::hmm::P7_FTAU] as f64;
    let lambda = gm.evparam[crate::hmm::P7_FLAMBDA] as f64;
    let lnp = crate::stats::exponential::surv(bitscore as f64, tau, lambda).ln();

    // Run Viterbi on envelope for traceback
    let mut gx_vit = Gmx::new(gm.m, env_len);
    g_viterbi(&sub_dsq, env_len, &env_gm, &mut gx_vit);
    let tr = crate::trace::g_trace(&sub_dsq, env_len, &env_gm, &gx_vit);

    // Generate alignment display
    let abc = Alphabet::new(hmm.abc_type);
    let ad = crate::trace::alignment_display(&tr, &sub_dsq, hmm, &abc).map(|mut ad| {
        // Adjust coordinates from envelope-local to sequence-global
        ad.sqfrom += ienv - 1;
        ad.sqto += ienv - 1;
        AliDisplay {
            model: ad.model,
            mline: ad.mline,
            aseq: ad.aseq,
            hmmfrom: ad.hmmfrom,
            hmmto: ad.hmmto,
            sqfrom: ad.sqfrom,
            sqto: ad.sqto,
                ppline: ad.ppline,
        }
    });

    // Update iali/jali from alignment display if available
    let (iali, jali) = if let Some(ref a) = ad {
        (a.sqfrom as i64, a.sqto as i64)
    } else {
        (ienv as i64, jenv as i64)
    };

    Domain {
        iali,
        jali,
        ienv: ienv as i64,
        jenv: jenv as i64,
        bitscore,
        lnp,
        dombias: dom_bias_bits,
        oasc: 0.0,
        envsc: env_fwd_sc,
        domcorrection: dom_bias,
        is_reported: false,
        is_included: false,
        ad,
    }
}

/// Run domain definition on a sequence that passed Forward filter.
/// Returns (domains, nexpected, seq_bias_nats).
pub fn define_domains(
    dsq: &[Dsq],
    l: usize,
    gm: &Profile,
    hmm: &Hmm,
    bg: &Bg,
    null_sc: f32,
) -> (Vec<Domain>, f32, f32) {
    crate::logsum::p7_flogsuminit();

    // Run Forward + Backward with generic DP
    let mut gx_fwd = Gmx::new(gm.m, l);
    let fwd_sc = g_forward(dsq, l, gm, &mut gx_fwd);

    let mut gx_bck = Gmx::new(gm.m, l);
    g_backward(dsq, l, gm, &mut gx_bck);

    // Posterior decoding
    let mut pp = Gmx::new(gm.m, l);
    g_decoding(gm, &gx_fwd, &gx_bck, &mut pp);

    // Domain occupancy
    let mocc = domain_occupancy(&pp);

    // Compute null2 from full-sequence posteriors
    let null2 = generic_null2::null2_by_expectation(gm, hmm, &pp, &bg.f);
    let seq_bias = generic_null2::null2_score(&null2, dsq, 1, l);
    let seq_bias_bits = seq_bias / std::f32::consts::LN_2;

    // Find domain regions
    let mut regions = find_domain_regions(&mocc, l);

    // Expected number of domains
    let nexpected: f32 = mocc[1..=l].iter().sum::<f32>() / gm.m as f32;
    let nexpected = nexpected.max(0.01);

    // Check for multidomain regions: if nexpected > 1.5 and we have regions,
    // try stochastic trace clustering to split them
    if nexpected > 1.5 && !regions.is_empty() {
        let nsamples = 200;
        let mut rng = MersenneTwister::new(42);
        let mut segments = Vec::new();

        for trace_idx in 0..nsamples {
            let tr = g_stochastic_trace(&mut rng, dsq, l, gm, &gx_fwd);
            // Extract segment pairs from this trace (B..E segments)
            let mut in_domain = false;
            let mut seg_i = 0;
            let mut seg_k = 0;
            let mut seg_j = 0;
            let mut seg_m = 0;

            for z in 0..tr.n {
                match tr.st[z] {
                    State::B => {
                        in_domain = true;
                        seg_i = 0;
                        seg_k = 0;
                    }
                    State::M if in_domain => {
                        if seg_i == 0 {
                            seg_i = tr.i[z];
                            seg_k = tr.k[z];
                        }
                        seg_j = tr.i[z];
                        seg_m = tr.k[z];
                    }
                    State::E if in_domain => {
                        if seg_i > 0 && seg_j > 0 {
                            segments.push(SegmentPair {
                                i: seg_i,
                                j: seg_j,
                                k: seg_k,
                                m: seg_m,
                                trace_idx,
                            });
                        }
                        in_domain = false;
                    }
                    _ => {}
                }
            }
        }

        if !segments.is_empty() {
            let params = ClusterParams::default();
            let envelopes = spensemble::cluster(&segments, nsamples, &params);
            if envelopes.len() > 1 {
                // Replace regions with clustered envelopes
                regions = envelopes.iter().map(|e| (e.ienv, e.jenv)).collect();
            }
        }
    }

    if regions.is_empty() {
        // No regions found by posterior — use the whole sequence as one domain
        // Run Viterbi traceback on full sequence for alignment display
        let mut gx_vit = Gmx::new(gm.m, l);
        g_viterbi(dsq, l, gm, &mut gx_vit);
        let tr = crate::trace::g_trace(dsq, l, gm, &gx_vit);
        let abc = Alphabet::new(hmm.abc_type);
        let ad = crate::trace::alignment_display(&tr, dsq, hmm, &abc).map(|ad| {
            AliDisplay {
                model: ad.model,
                mline: ad.mline,
                aseq: ad.aseq,
                hmmfrom: ad.hmmfrom,
                hmmto: ad.hmmto,
                sqfrom: ad.sqfrom,
                sqto: ad.sqto,
                ppline: ad.ppline,
            }
        });
        let (iali, jali) = if let Some(ref a) = ad { (a.sqfrom as i64, a.sqto as i64) } else { (1, l as i64) };

        let dom = Domain {
            iali,
            jali,
            ienv: 1,
            jenv: l as i64,
            bitscore: (fwd_sc - null_sc) / std::f32::consts::LN_2,
            lnp: crate::stats::exponential::surv(
                ((fwd_sc - null_sc) / std::f32::consts::LN_2) as f64,
                gm.evparam[crate::hmm::P7_FTAU] as f64,
                gm.evparam[crate::hmm::P7_FLAMBDA] as f64,
            )
            .ln(),
            dombias: seq_bias_bits.max(0.0),
            oasc: 0.0,
            envsc: fwd_sc,
            domcorrection: 0.0,
            is_reported: false,
            is_included: false,
            ad,
        };
        return (vec![dom], nexpected, seq_bias);
    }

    // Score each domain region
    let mut domains = Vec::new();
    for (ienv, jenv) in &regions {
        let mut dom = score_domain_envelope(dsq, l, gm, hmm, *ienv, *jenv, null_sc);
        // Compute domain-specific bias from null2
        let dom_bias = generic_null2::null2_score(&null2, dsq, *ienv, *jenv);
        dom.dombias = (dom_bias / std::f32::consts::LN_2).max(0.0);
        dom.domcorrection = dom_bias;
        domains.push(dom);
    }

    (domains, nexpected, seq_bias)
}
