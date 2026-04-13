//! Domain definition using posterior probabilities.
//! Port of p7_domaindef.c p7_domaindef_ByPosteriorHeuristics().

use crate::alphabet::{Alphabet, Dsq};
use crate::bg::Bg;
use crate::dp::generic_backward::g_backward;
use crate::dp::generic_decoding::{domain_decoding, g_decoding};
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

// Thresholds matching C's p7_domaindef defaults
const RT1: f32 = 0.25; // mocc threshold to trigger a domain region
const RT2: f32 = 0.10; // mocc threshold to end a domain region
const RT3: f32 = 0.20; // threshold for multi-domain region detection
const NSAMPLES: usize = 200; // stochastic tracebacks for clustering

/// Region detection state machine matching C's p7_domaindef_ByPosteriorHeuristics().
/// Uses btot/etot/mocc arrays to find domain regions.
fn find_domain_regions(
    btot: &[f32],
    etot: &[f32],
    mocc: &[f32],
    l: usize,
) -> Vec<(usize, usize)> {
    let mut regions = Vec::new();
    let mut i: i64 = -1;
    let mut triggered = false;

    for j in 1..=l {
        if !triggered {
            // Looking for the START of a domain region.
            // Reset i when mocc (minus local B contribution) drops below rt2.
            if mocc[j] - (btot[j] - btot[j - 1]) < RT2 {
                i = j as i64;
            } else if i == -1 {
                i = j as i64;
            }
            // Trigger when mocc rises above rt1
            if mocc[j] >= RT1 {
                triggered = true;
            }
        } else if mocc[j] - (etot[j] - etot[j - 1]) < RT2 {
            // Found the END of a domain region: mocc dropped below rt2
            // (after subtracting local E contribution). Region is i..j.
            if i >= 1 {
                regions.push((i as usize, j));
            }
            i = -1;
            triggered = false;
        }
    }

    regions
}

/// Check if a region contains multiple domains.
/// Matches C's is_multidomain_region().
fn is_multidomain_region(btot: &[f32], etot: &[f32], i: usize, j: usize) -> bool {
    let mut max = -1.0_f32;
    for z in i..=j {
        let expected_n = (etot[z] - etot[i - 1]).min(btot[j] - btot[z - 1]);
        max = max.max(expected_n);
    }
    max >= RT3
}

/// Score a domain envelope: Forward for score, Viterbi for traceback,
/// posterior decoding for PP annotation and null2 bias.
/// `seq_len` is the full sequence length (for null model and length correction).
/// `bg_f` is the background frequencies for the alphabet.
pub fn score_domain_envelope(
    dsq: &[Dsq],
    seq_len: usize,
    gm: &Profile,
    hmm: &Hmm,
    ienv: usize,
    jenv: usize,
    null_sc: f32,
    bg_f: &[f32],
) -> Domain {
    debug_assert!(ienv >= 1 && jenv <= seq_len);
    debug_assert!(null_sc.is_finite());
    let env_len = jenv - ienv + 1;
    let l = seq_len; // full sequence length for length correction

    let mut env_gm = gm.clone();
    reconfig_unihit(&mut env_gm, env_len as i32);

    let mut sub_dsq = vec![crate::alphabet::DSQ_SENTINEL];
    sub_dsq.extend_from_slice(&dsq[ienv..=jenv]);
    sub_dsq.push(crate::alphabet::DSQ_SENTINEL);

    // Forward for the envelope score
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

    // Viterbi traceback for alignment
    let mut gx_vit = Gmx::new(gm.m, env_len);
    g_viterbi(&sub_dsq, env_len, &env_gm, &mut gx_vit);
    let tr = crate::trace::g_trace(&sub_dsq, env_len, &env_gm, &gx_vit);

    // Posterior decoding for PP annotation and null2
    let mut gx_fwd = Gmx::new(gm.m, env_len);
    g_forward(&sub_dsq, env_len, &env_gm, &mut gx_fwd);
    let mut gx_bck = Gmx::new(gm.m, env_len);
    g_backward(&sub_dsq, env_len, &env_gm, &mut gx_bck);
    let mut env_pp = Gmx::new(gm.m, env_len);
    g_decoding(&env_gm, &gx_fwd, &gx_bck, &mut env_pp);

    let abc = Alphabet::new(hmm.abc_type);
    let ad = crate::trace::alignment_display_with_pp(&tr, &sub_dsq, hmm, &abc, Some(&env_pp))
        .map(|mut ad| {
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

    // Null2 bias from envelope posterior (with omega weighting, matching C)
    let env_null2 = generic_null2::null2_by_expectation(
        &env_gm,
        hmm,
        &env_pp,
        bg_f,
    );
    let dom_correction = generic_null2::null2_score(&env_null2, &sub_dsq, 1, env_len);
    let omega = 1.0_f32 / 256.0;
    let dom_bias = crate::logsum::p7_flogsum(0.0, omega.ln() + dom_correction);

    // Domain bitscore matching C's rescore_isolated_domain():
    // C: bitscore = envsc + (L - Ld) * log(L / (L+3))
    //    bitscore = (bitscore - (nullsc + dombias)) / log(2)
    // The first line adds a correction for non-envelope residues under the null model.
    // nullsc is the full-sequence null model score.
    let length_correction = (l - env_len) as f32
        * (l as f32 / (l as f32 + 3.0)).ln();
    let dom_bitscore = (env_fwd_sc + length_correction - (null_sc + dom_bias))
        / std::f32::consts::LN_2;
    let tau = gm.evparam[crate::hmm::P7_FTAU] as f64;
    let lambda = gm.evparam[crate::hmm::P7_FLAMBDA] as f64;
    let dom_lnp = crate::stats::exponential::surv(dom_bitscore as f64, tau, lambda).ln();

    Domain {
        iali,
        jali,
        ienv: ienv as i64,
        jenv: jenv as i64,
        bitscore: dom_bitscore,
        lnp: dom_lnp,
        dombias: (dom_bias / std::f32::consts::LN_2).max(0.0),
        oasc: 0.0,
        envsc: env_fwd_sc,
        domcorrection: dom_correction,
        is_reported: false,
        is_included: false,
        ad,
    }
}

/// Run domain definition on a sequence that passed Forward filter.
/// Port of p7_domaindef_ByPosteriorHeuristics().
/// Returns (domains, nexpected, seq_bias_nats).
pub fn define_domains(
    dsq: &[Dsq],
    l: usize,
    gm: &Profile,
    hmm: &Hmm,
    bg: &Bg,
    null_sc: f32,
    seed: u32,
) -> (Vec<Domain>, f32, f32) {
    crate::logsum::p7_flogsuminit();

    // Phase 1: Domain decoding (btot/etot/mocc) — SIMD when available
    let (btot, etot, mocc);

    #[cfg(target_arch = "x86_64")]
    let use_simd = is_x86_feature_detected!("sse2");
    #[cfg(not(target_arch = "x86_64"))]
    let use_simd = false;

    if use_simd {
        #[cfg(target_arch = "x86_64")]
        {
            use crate::simd::probmx::{ProbMx, p_domain_decoding};
            use crate::simd::oprofile::*;

            let om = OProfile::convert(gm);
            let mut fwd_pmx = ProbMx::new(l);
            let fwd_sc = unsafe {
                crate::simd::fwd_filter::forward_parser_pmx(dsq, l, &om, &mut fwd_pmx)
            };
            let mut bck_pmx = ProbMx::new(l);
            unsafe {
                crate::simd::bck_filter::backward_parser_pmx(dsq, l, &om, fwd_sc, &mut bck_pmx);
            };
            let njc_loop = [
                om.xf[P7O_N][P7O_LOOP],
                om.xf[P7O_J][P7O_LOOP],
                om.xf[P7O_C][P7O_LOOP],
            ];
            let r = p_domain_decoding(&fwd_pmx, &bck_pmx, l, njc_loop);
            btot = r.0;
            etot = r.1;
            mocc = r.2;
        }
        #[cfg(not(target_arch = "x86_64"))]
        unreachable!();
    } else {
        let mut gx_fwd = Gmx::new(gm.m, l);
        g_forward(dsq, l, gm, &mut gx_fwd);
        let mut gx_bck = Gmx::new(gm.m, l);
        g_backward(dsq, l, gm, &mut gx_bck);
        let r = domain_decoding(gm, &gx_fwd, &gx_bck);
        btot = r.0;
        etot = r.1;
        mocc = r.2;
    }

    let nexpected = btot[l].max(0.01);

    // Phase 2: Null2 — still needs generic Forward+Backward for per-M-state posteriors
    let mut gx_fwd = Gmx::new(gm.m, l);
    g_forward(dsq, l, gm, &mut gx_fwd);
    let mut gx_bck = Gmx::new(gm.m, l);
    g_backward(dsq, l, gm, &mut gx_bck);
    let mut pp = Gmx::new(gm.m, l);
    g_decoding(gm, &gx_fwd, &gx_bck, &mut pp);
    let null2_arr = generic_null2::null2_by_expectation(gm, hmm, &pp, &bg.f);
    let seq_bias = generic_null2::null2_score(&null2_arr, dsq, 1, l);

    // Region detection using C's state machine
    let regions = find_domain_regions(&btot, &etot, &mocc, l);

    if regions.is_empty() {
        // No regions found — if nexpected > 0, return single domain covering sequence
        if nexpected >= 0.5 {
            let dom = score_domain_envelope(dsq, l, gm, hmm, 1, l, null_sc, &bg.f);
            return (vec![dom], nexpected, seq_bias);
        }
        return (Vec::new(), nexpected, seq_bias);
    }

    let mut domains = Vec::new();

    for &(ri, rj) in &regions {
        if is_multidomain_region(&btot, &etot, ri, rj) {
            // Multi-domain region: resolve by stochastic traceback clustering
            // Run Forward on the region in multihit mode
            let region_len = rj - ri + 1;
            let mut region_gm = gm.clone();
            reconfig_multihit(&mut region_gm, region_len as i32);

            let mut sub_dsq = vec![crate::alphabet::DSQ_SENTINEL];
            sub_dsq.extend_from_slice(&dsq[ri..=rj]);
            sub_dsq.push(crate::alphabet::DSQ_SENTINEL);

            let mut region_fwd = Gmx::new(gm.m, region_len);
            g_forward(&sub_dsq, region_len, &region_gm, &mut region_fwd);

            let mut rng = MersenneTwister::new(seed);
            let mut segments = Vec::new();

            for trace_idx in 0..NSAMPLES {
                let tr = g_stochastic_trace(&mut rng, &sub_dsq, region_len, &region_gm, &region_fwd);
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
                                    i: seg_i + ri - 1, // convert to full-sequence coords
                                    j: seg_j + ri - 1,
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
                let envelopes = spensemble::cluster(&segments, NSAMPLES, &params);
                for env in &envelopes {
                    let ienv = env.ienv.max(1).min(l);
                    let jenv = env.jenv.max(1).min(l);
                    if jenv >= ienv {
                        domains.push(score_domain_envelope(
                            dsq, l, gm, hmm, ienv, jenv, null_sc, &bg.f,
                        ));
                    }
                }
            } else {
                // Clustering failed — treat as single domain
                domains.push(score_domain_envelope(dsq, l, gm, hmm, ri, rj, null_sc, &bg.f));
            }
        } else {
            // Single-domain region: the region IS the envelope
            domains.push(score_domain_envelope(dsq, l, gm, hmm, ri, rj, null_sc, &bg.f));
        }
    }

    // Sort domains by envelope start position
    domains.sort_by_key(|d| d.ienv);

    (domains, nexpected, seq_bias)
}
