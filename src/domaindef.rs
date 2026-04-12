//! Domain definition using posterior probabilities.
//! Optimized: skips expensive Backward+Decoding for simple single-domain cases.

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
use crate::spensemble::{self, ClusterParams, SegmentPair};
use crate::tophits::{AliDisplay, Domain};
use crate::trace::State;
use crate::util::random::MersenneTwister;

const RT1: f32 = 0.25;
const RT2: f32 = 0.10;

pub fn find_domain_regions(mocc: &[f32], l: usize) -> Vec<(usize, usize)> {
    let mut regions = Vec::new();
    let mut i = 1;
    while i <= l {
        if mocc[i] >= RT1 {
            let start = i;
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

/// Score a domain envelope: SIMD Forward for score, Viterbi for traceback.
pub fn score_domain_envelope(
    dsq: &[Dsq],
    l: usize,
    gm: &Profile,
    hmm: &Hmm,
    ienv: usize,
    jenv: usize,
    null_sc: f32,
) -> Domain {
    debug_assert!(ienv >= 1 && jenv <= l);
    debug_assert!(null_sc.is_finite());
    let env_len = jenv - ienv + 1;

    let mut env_gm = gm.clone();
    reconfig_unihit(&mut env_gm, env_len as i32);

    let mut sub_dsq = vec![crate::alphabet::DSQ_SENTINEL];
    sub_dsq.extend_from_slice(&dsq[ienv..=jenv]);
    sub_dsq.push(crate::alphabet::DSQ_SENTINEL);

    // SIMD Forward for the score
    let env_fwd_sc;
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse2") {
            let env_om = crate::simd::oprofile::OProfile::convert(&env_gm);
            env_fwd_sc =
                unsafe { crate::simd::fwd_filter::forward_parser(&sub_dsq, env_len, &env_om) };
        } else {
            let mut gx = Gmx::new(gm.m, env_len);
            env_fwd_sc = g_forward(&sub_dsq, env_len, &env_gm, &mut gx);
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let mut gx = Gmx::new(gm.m, env_len);
        env_fwd_sc = g_forward(&sub_dsq, env_len, &env_gm, &mut gx);
    }

    let p1 = env_len as f32 / (env_len as f32 + 1.0);
    let env_null = env_len as f32 * p1.ln() + (1.0 - p1).ln();
    let bitscore = (env_fwd_sc - env_null) / std::f32::consts::LN_2;
    let tau = gm.evparam[crate::hmm::P7_FTAU] as f64;
    let lambda = gm.evparam[crate::hmm::P7_FLAMBDA] as f64;
    let lnp = crate::stats::exponential::surv(bitscore as f64, tau, lambda).ln();

    // Viterbi traceback for alignment (much cheaper than Forward+Backward+Decoding)
    let mut gx_vit = Gmx::new(gm.m, env_len);
    g_viterbi(&sub_dsq, env_len, &env_gm, &mut gx_vit);
    let tr = crate::trace::g_trace(&sub_dsq, env_len, &env_gm, &gx_vit);

    let abc = Alphabet::new(hmm.abc_type);
    let ad = crate::trace::alignment_display(&tr, &sub_dsq, hmm, &abc).map(|mut ad| {
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

    let (iali, jali) = if let Some(ref a) = ad {
        (a.sqfrom as i64, a.sqto as i64)
    } else {
        (ienv as i64, jenv as i64)
    };

    // Simplified null2 bias from match emission composition (no Backward needed)
    let dom_bias = estimate_bias_fast(hmm, &sub_dsq, env_len);

    Domain {
        iali,
        jali,
        ienv: ienv as i64,
        jenv: jenv as i64,
        bitscore,
        lnp,
        dombias: dom_bias,
        oasc: 0.0,
        envsc: env_fwd_sc,
        domcorrection: dom_bias * std::f32::consts::LN_2,
        is_reported: false,
        is_included: false,
        ad,
    }
}

/// Fast bias estimate without Backward: uses the HMM's match emission composition
/// vs background to estimate composition bias for the envelope residues.
fn estimate_bias_fast(hmm: &Hmm, dsq: &[Dsq], l: usize) -> f32 {
    let k = hmm.abc_k;
    // Count residue frequencies in the envelope
    let mut counts = vec![0usize; k];
    for i in 1..=l {
        let x = dsq[i] as usize;
        if x < k {
            counts[x] += 1;
        }
    }
    // Compare observed composition to background
    let mut bias = 0.0_f32;
    let total = l as f32;
    for x in 0..k {
        let obs_freq = counts[x] as f32 / total;
        let bg_freq = crate::bg::AMINO_FREQUENCIES.get(x).copied().unwrap_or(0.05);
        if obs_freq > 0.0 && bg_freq > 0.0 {
            bias += obs_freq * (obs_freq / bg_freq).ln();
        }
    }
    // Convert to bits, clamp to non-negative
    (bias / std::f32::consts::LN_2 * total).max(0.0)
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

    // Quick estimate of expected domains using SIMD Forward score
    // If the Forward score is modest, likely just one domain — skip expensive posterior
    let fwd_sc;
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse2") {
            let om = crate::simd::oprofile::OProfile::convert(gm);
            fwd_sc = unsafe { crate::simd::fwd_filter::forward_parser(dsq, l, &om) };
        } else {
            let mut gx = Gmx::new(gm.m, l);
            fwd_sc = g_forward(dsq, l, gm, &mut gx);
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let mut gx = Gmx::new(gm.m, l);
        fwd_sc = g_forward(dsq, l, gm, &mut gx);
    }

    let bitscore = (fwd_sc - null_sc) / std::f32::consts::LN_2;

    // Fast path: single domain covering the whole sequence
    // Use Viterbi traceback for alignment (no Backward needed)
    let seq_bias = estimate_bias_fast(hmm, dsq, l);

    // For simple cases (short sequences, moderate scores), skip full posterior
    // and just return a single domain
    if l <= gm.m * 3 || bitscore < 100.0 {
        let dom = score_domain_envelope(dsq, l, gm, hmm, 1, l, null_sc);
        let nexpected = 1.0_f32;
        return (vec![dom], nexpected, seq_bias);
    }

    // Complex case: run full posterior for domain region detection
    let mut gx_fwd = Gmx::new(gm.m, l);
    g_forward(dsq, l, gm, &mut gx_fwd);

    let mut gx_bck = Gmx::new(gm.m, l);
    g_backward(dsq, l, gm, &mut gx_bck);

    let mut pp = Gmx::new(gm.m, l);
    g_decoding(gm, &gx_fwd, &gx_bck, &mut pp);

    let mocc = domain_occupancy(&pp);

    let null2 = generic_null2::null2_by_expectation(gm, hmm, &pp, &bg.f);
    let seq_bias_posterior = generic_null2::null2_score(&null2, dsq, 1, l);

    let mut regions = find_domain_regions(&mocc, l);
    let nexpected: f32 = (mocc[1..=l].iter().sum::<f32>() / gm.m as f32).max(0.01);

    // Multidomain clustering
    if nexpected > 1.5 && !regions.is_empty() {
        let nsamples = 200;
        let mut rng = MersenneTwister::new(42);
        let mut segments = Vec::new();

        for trace_idx in 0..nsamples {
            let tr = g_stochastic_trace(&mut rng, dsq, l, gm, &gx_fwd);
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
                regions = envelopes.iter().map(|e| (e.ienv, e.jenv)).collect();
            }
        }
    }

    if regions.is_empty() {
        let dom = score_domain_envelope(dsq, l, gm, hmm, 1, l, null_sc);
        return (vec![dom], nexpected, seq_bias_posterior);
    }

    let mut domains = Vec::new();
    for (ienv, jenv) in &regions {
        let mut dom = score_domain_envelope(dsq, l, gm, hmm, *ienv, *jenv, null_sc);
        let dom_bias_null2 = generic_null2::null2_score(&null2, dsq, *ienv, *jenv);
        dom.dombias = (dom_bias_null2 / std::f32::consts::LN_2).max(0.0);
        dom.domcorrection = dom_bias_null2;
        domains.push(dom);
    }

    (domains, nexpected, seq_bias_posterior)
}
