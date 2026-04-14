//! Domain definition using posterior probabilities.
//! Port of p7_domaindef.c p7_domaindef_ByPosteriorHeuristics().

use crate::alphabet::{Alphabet, Dsq};
use crate::bg::Bg;
use crate::dp::generic_backward::g_backward;
use crate::dp::generic_decoding::{domain_decoding, g_decoding};
use crate::dp::generic_fwdback::g_forward;
use crate::dp::generic_null2;
use crate::dp::generic_optacc::{g_oa_trace, g_optimal_accuracy_with_deltas, OptAccTDelta};
use crate::dp::generic_stotrace::g_stochastic_trace;
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

#[derive(Debug, Clone, Copy, Default)]
pub struct DomainDefinitionStats {
    pub nregions: usize,
    pub nclustered: usize,
    pub noverlaps: usize,
    pub nenvelopes: usize,
}

#[cfg(target_arch = "x86_64")]
struct DomainSimdScratch {
    fwd_dp: Vec<std::arch::x86_64::__m128>,
    bck_prev: Vec<std::arch::x86_64::__m128>,
    bck_cur: Vec<std::arch::x86_64::__m128>,
    fwd_pmx: crate::simd::probmx::ProbMx,
    bck_pmx: crate::simd::probmx::ProbMx,
    null2: Vec<f32>,
    exp_m: Vec<f32>,
    exp_i: Vec<f32>,
    pp_gmx: Gmx,
    oa_gmx: Gmx,
}

#[cfg(target_arch = "x86_64")]
impl DomainSimdScratch {
    fn new() -> Self {
        Self {
            fwd_dp: Vec::new(),
            bck_prev: Vec::new(),
            bck_cur: Vec::new(),
            fwd_pmx: crate::simd::probmx::ProbMx::new_full(0, 0),
            bck_pmx: crate::simd::probmx::ProbMx::new_full(0, 0),
            null2: Vec::new(),
            exp_m: Vec::new(),
            exp_i: Vec::new(),
            pp_gmx: Gmx::new(0, 0),
            oa_gmx: Gmx::new(0, 0),
        }
    }
}

#[cfg(not(target_arch = "x86_64"))]
struct DomainSimdScratch;

#[cfg(not(target_arch = "x86_64"))]
impl DomainSimdScratch {
    fn new() -> Self {
        Self
    }
}

/// Region detection state machine matching C's p7_domaindef_ByPosteriorHeuristics().
/// Uses btot/etot/mocc arrays to find domain regions.
fn find_domain_regions(btot: &[f32], etot: &[f32], mocc: &[f32], l: usize) -> Vec<(usize, usize)> {
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

fn add_null2_correction(
    null2_arr: &[f32],
    dsq: &[Dsq],
    ienv: usize,
    jenv: usize,
    mut n2sc: Option<&mut [f32]>,
    dom_correction: &mut f32,
) {
    let mut log_null2 = [0.0_f32; 256];
    for (x, val) in null2_arr.iter().enumerate() {
        log_null2[x] = val.ln();
    }
    for pos in ienv..=jenv {
        let x = dsq[pos] as usize;
        let sc = if x < null2_arr.len() {
            log_null2[x]
        } else {
            0.0
        };
        if let Some(ref mut n2sc) = n2sc {
            n2sc[pos] = sc;
        }
        *dom_correction += sc;
    }
}

/// Score a domain envelope: Forward for score, Viterbi for traceback,
/// posterior decoding for PP annotation and null2 bias.
/// `seq_len` is the full sequence length (for null model and length correction).
/// `bg_f` is the background frequencies for the alphabet.
fn score_domain_envelope(
    dsq: &[Dsq],
    seq_len: usize,
    gm: &Profile,
    env_gm: &Profile,
    env_om: Option<&crate::simd::oprofile::OProfile>,
    hmm: &Hmm,
    ienv: usize,
    jenv: usize,
    null_sc: f32,
    _bg_f: &[f32],
    mut n2sc: Option<&mut [f32]>,
    null2_is_done: bool,
    match_odds: Option<&[f32]>,
    optacc_deltas: &OptAccTDelta,
    mut simd_scratch: Option<&mut DomainSimdScratch>,
    make_alignment: bool,
    make_alignment_display: bool,
) -> Domain {
    debug_assert!(ienv >= 1 && jenv <= seq_len);
    debug_assert!(null_sc.is_finite());
    let env_len = jenv - ienv + 1;
    let l = seq_len; // full sequence length for length correction

    // Forward score for the envelope.
    let env_fwd_sc;
    let mut simd_null2_arr: Option<Vec<f32>> = None;
    let mut simd_null2_scratch: Option<&[f32]> = None;
    let mut env_pp_storage: Option<Gmx>;
    let mut env_pp_ptr: *mut Gmx = std::ptr::null_mut();
    if make_alignment {
        if let Some(ref mut scratch) = simd_scratch {
            scratch.pp_gmx.grow_to_zeroed(gm.m, env_len);
            env_pp_ptr = &mut scratch.pp_gmx;
        } else {
            env_pp_storage = Some(Gmx::new(gm.m, env_len));
            env_pp_ptr = env_pp_storage.as_mut().unwrap();
        }
    }

    #[cfg(target_arch = "x86_64")]
    let use_simd = is_x86_feature_detected!("sse2");
    #[cfg(not(target_arch = "x86_64"))]
    let use_simd = false;
    let dsq_offset = ienv - 1;
    let mut sub_dsq = Vec::new();
    if !use_simd || make_alignment_display {
        sub_dsq.push(crate::alphabet::DSQ_SENTINEL);
        sub_dsq.extend_from_slice(&dsq[ienv..=jenv]);
        sub_dsq.push(crate::alphabet::DSQ_SENTINEL);
    }

    if use_simd {
        #[cfg(target_arch = "x86_64")]
        {
            use crate::simd::oprofile::OProfile;
            let env_om_storage;
            let env_om = if let Some(om) = env_om {
                om
            } else {
                env_om_storage = OProfile::convert(env_gm);
                &env_om_storage
            };
            if make_alignment || !null2_is_done {
                use crate::simd::oprofile::{P7O_C, P7O_J, P7O_LOOP, P7O_N};
                use crate::simd::probmx::{
                    match_odds_from_rsc, p_decoding_to_gmx, p_null2_odds_from_gmx,
                    p_null2_odds_from_gmx_reuse, p_null2_odds_from_pmx,
                    p_null2_odds_from_pmx_reuse, ProbMx,
                };

                if let Some(ref mut scratch) = simd_scratch {
                    scratch.fwd_pmx.resize_full(gm.m, env_len);
                    env_fwd_sc = unsafe {
                        crate::simd::fwd_filter::forward_parser_pmx_offset_with_scratch(
                            dsq,
                            dsq_offset,
                            env_len,
                            &env_om,
                            &mut scratch.fwd_pmx,
                            &mut scratch.fwd_dp,
                        )
                    };
                    scratch.bck_pmx.resize_full(gm.m, env_len);
                    unsafe {
                        crate::simd::bck_filter::backward_parser_pmx_offset_with_scratch(
                            dsq,
                            dsq_offset,
                            env_len,
                            &env_om,
                            env_fwd_sc,
                            &mut scratch.bck_pmx,
                            &mut scratch.bck_prev,
                            &mut scratch.bck_cur,
                        );
                    };
                    let njc_loop = [
                        env_om.xf[P7O_N][P7O_LOOP],
                        env_om.xf[P7O_J][P7O_LOOP],
                        env_om.xf[P7O_C][P7O_LOOP],
                    ];
                    if make_alignment {
                        p_decoding_to_gmx(
                            &scratch.fwd_pmx,
                            &scratch.bck_pmx,
                            gm.m,
                            njc_loop,
                            unsafe { &mut *env_pp_ptr },
                        );
                    }
                    if !null2_is_done {
                        let local_match_odds;
                        let match_odds = if let Some(match_odds) = match_odds {
                            match_odds
                        } else {
                            local_match_odds = match_odds_from_rsc(&gm.rsc, gm.abc_k, gm.m);
                            &local_match_odds
                        };
                        if make_alignment {
                            p_null2_odds_from_gmx_reuse(
                                unsafe { &*env_pp_ptr },
                                gm.m,
                                gm.abc_k,
                                match_odds,
                                &mut scratch.null2,
                                &mut scratch.exp_m,
                                &mut scratch.exp_i,
                            );
                            simd_null2_scratch = Some(&scratch.null2);
                        } else {
                            p_null2_odds_from_pmx_reuse(
                                &scratch.fwd_pmx,
                                &scratch.bck_pmx,
                                gm.m,
                                gm.abc_k,
                                match_odds,
                                njc_loop,
                                &mut scratch.null2,
                                &mut scratch.exp_m,
                                &mut scratch.exp_i,
                            );
                            simd_null2_scratch = Some(&scratch.null2);
                        }
                    }
                } else {
                    let mut fwd_pmx = ProbMx::new_full(gm.m, env_len);
                    env_fwd_sc = unsafe {
                        crate::simd::fwd_filter::forward_parser_pmx_offset(
                            dsq,
                            dsq_offset,
                            env_len,
                            &env_om,
                            &mut fwd_pmx,
                        )
                    };
                    let mut bck_pmx = ProbMx::new_full(gm.m, env_len);
                    unsafe {
                        crate::simd::bck_filter::backward_parser_pmx_offset(
                            dsq,
                            dsq_offset,
                            env_len,
                            &env_om,
                            env_fwd_sc,
                            &mut bck_pmx,
                        );
                    }
                    let njc_loop = [
                        env_om.xf[P7O_N][P7O_LOOP],
                        env_om.xf[P7O_J][P7O_LOOP],
                        env_om.xf[P7O_C][P7O_LOOP],
                    ];
                    if make_alignment {
                        p_decoding_to_gmx(
                            &fwd_pmx,
                            &bck_pmx,
                            gm.m,
                            njc_loop,
                            unsafe { &mut *env_pp_ptr },
                        );
                    }
                    if !null2_is_done {
                        let local_match_odds;
                        let match_odds = if let Some(match_odds) = match_odds {
                            match_odds
                        } else {
                            local_match_odds = match_odds_from_rsc(&gm.rsc, gm.abc_k, gm.m);
                            &local_match_odds
                        };
                        simd_null2_arr = Some(if make_alignment {
                            p_null2_odds_from_gmx(
                                unsafe { &*env_pp_ptr },
                                gm.m,
                                gm.abc_k,
                                match_odds,
                            )
                        } else {
                            p_null2_odds_from_pmx(
                                &fwd_pmx, &bck_pmx, gm.m, gm.abc_k, match_odds, njc_loop,
                            )
                        });
                    }
                }
            } else {
                env_fwd_sc = unsafe {
                    crate::simd::fwd_filter::forward_parser_offset(
                        dsq, dsq_offset, env_len, &env_om,
                    )
                };
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        unreachable!();
    } else {
        let mut gx = Gmx::new(gm.m, env_len);
        env_fwd_sc = g_forward(&sub_dsq, env_len, env_gm, &mut gx);
    }

    // Posterior decoding for PP annotation and null2. C HMMER fills n2sc[]
    // from these per-envelope posteriors for simple regions.
    if env_pp_ptr.is_null()
        && simd_null2_arr.is_none()
        && simd_null2_scratch.is_none()
        && !null2_is_done
    {
        env_pp_storage = Some(Gmx::new(gm.m, env_len));
        env_pp_ptr = env_pp_storage.as_mut().unwrap();
    }
    if !env_pp_ptr.is_null()
        && unsafe { (*env_pp_ptr).l == 0 }
        && (make_alignment
            || simd_null2_arr.is_none() && simd_null2_scratch.is_none() && !null2_is_done)
    {
        let mut gx_fwd = Gmx::new(gm.m, env_len);
        g_forward(&sub_dsq, env_len, env_gm, &mut gx_fwd);
        let mut gx_bck = Gmx::new(gm.m, env_len);
        g_backward(&sub_dsq, env_len, env_gm, &mut gx_bck);
        g_decoding(env_gm, &gx_fwd, &gx_bck, unsafe { &mut *env_pp_ptr });
    }
    let mut dom_correction = 0.0_f32;
    if null2_is_done {
        if let Some(ref n2sc) = n2sc {
            for pos in ienv..=jenv {
                dom_correction += n2sc[pos];
            }
        }
    } else {
        if let Some(null2_arr) = simd_null2_scratch {
            add_null2_correction(null2_arr, dsq, ienv, jenv, n2sc.as_deref_mut(), &mut dom_correction);
        } else {
            let null2_arr = simd_null2_arr.unwrap_or_else(|| {
                generic_null2::null2_by_expectation(env_gm, unsafe { &*env_pp_ptr }, 1, env_len)
            });
            add_null2_correction(&null2_arr, dsq, ienv, jenv, n2sc.as_deref_mut(), &mut dom_correction);
        }
    }

    let mut oasc = 0.0_f32;
    let ad = if make_alignment {
        let env_pp = unsafe { &*env_pp_ptr };
        if let Some(ref mut scratch) = simd_scratch {
            scratch.oa_gmx.grow_to_zeroed(gm.m, env_len);
            oasc =
                g_optimal_accuracy_with_deltas(env_gm, env_pp, &mut scratch.oa_gmx, optacc_deltas);
            let tr = g_oa_trace(env_gm, env_pp, &scratch.oa_gmx);
            if make_alignment_display {
                let abc = Alphabet::new(hmm.abc_type);
                crate::trace::alignment_display_with_pp(
                    &tr,
                    &sub_dsq,
                    hmm,
                    &abc,
                    Some(env_pp),
                )
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
                })
            } else {
                crate::trace::alignment_coords(&tr).map(
                    |(hmmfrom, hmmto, mut sqfrom, mut sqto)| {
                        if sqfrom > 0 {
                            sqfrom += ienv - 1;
                        }
                        if sqto > 0 {
                            sqto += ienv - 1;
                        }
                        AliDisplay {
                            model: String::new(),
                            mline: String::new(),
                            aseq: String::new(),
                            hmmfrom,
                            hmmto,
                            sqfrom,
                            sqto,
                            ppline: String::new(),
                        }
                    },
                )
            }
        } else {
            let mut gx_oa = Gmx::new(gm.m, env_len);
            oasc = g_optimal_accuracy_with_deltas(env_gm, env_pp, &mut gx_oa, optacc_deltas);
            let tr = g_oa_trace(env_gm, env_pp, &gx_oa);
            if make_alignment_display {
                let abc = Alphabet::new(hmm.abc_type);
                crate::trace::alignment_display_with_pp(&tr, &sub_dsq, hmm, &abc, Some(env_pp))
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
                    })
            } else {
                crate::trace::alignment_coords(&tr).map(
                    |(hmmfrom, hmmto, mut sqfrom, mut sqto)| {
                        if sqfrom > 0 {
                            sqfrom += ienv - 1;
                        }
                        if sqto > 0 {
                            sqto += ienv - 1;
                        }
                        AliDisplay {
                            model: String::new(),
                            mline: String::new(),
                            aseq: String::new(),
                            hmmfrom,
                            hmmto,
                            sqfrom,
                            sqto,
                            ppline: String::new(),
                        }
                    },
                )
            }
        }
    } else {
        None
    };

    let (iali, jali) = if let Some(ref a) = ad {
        (a.sqfrom as i64, a.sqto as i64)
    } else {
        (ienv as i64, jenv as i64)
    };

    // Omega-weighted null2 bias.
    let omega = 1.0_f32 / 256.0;
    let dom_bias = crate::logsum::p7_flogsum(0.0, omega.ln() + dom_correction);

    // Domain bitscore matching C's rescore_isolated_domain():
    // C: bitscore = envsc + (L - Ld) * log(L / (L+3))
    //    bitscore = (bitscore - (nullsc + dombias)) / log(2)
    // The first line adds a correction for non-envelope residues under the null model.
    // nullsc is the full-sequence null model score.
    let length_correction = (l - env_len) as f32 * (l as f32 / (l as f32 + 3.0)).ln();
    let dom_bitscore =
        (env_fwd_sc + length_correction - (null_sc + dom_bias)) / std::f32::consts::LN_2;
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
        oasc,
        envsc: env_fwd_sc,
        domcorrection: dom_correction,
        is_reported: false,
        is_included: false,
        ad,
    }
}

/// Run domain definition on a sequence that passed Forward filter.
/// Port of p7_domaindef_ByPosteriorHeuristics().
/// Returns (domains, nexpected, seq_bias_nats, stats).
pub fn define_domains(
    dsq: &[Dsq],
    l: usize,
    gm: &Profile,
    om: Option<&crate::simd::oprofile::OProfile>,
    fwd_pmx_input: Option<&crate::simd::probmx::ProbMx>,
    fwd_sc_input: Option<f32>,
    hmm: &Hmm,
    bg: &Bg,
    null_sc: f32,
    seed: u32,
    make_alignment: bool,
    make_alignment_display: bool,
) -> (Vec<Domain>, f32, f32, DomainDefinitionStats) {
    crate::logsum::p7_flogsuminit();

    // Phase 1: Domain decoding (btot/etot/mocc) — SIMD when available
    let (btot, etot, mocc);

    #[cfg(target_arch = "x86_64")]
    let use_simd = is_x86_feature_detected!("sse2");
    #[cfg(not(target_arch = "x86_64"))]
    let use_simd = false;
    let mut simd_scratch = DomainSimdScratch::new();
    if use_simd {
        #[cfg(target_arch = "x86_64")]
        {
            use crate::simd::oprofile::*;
            use crate::simd::probmx::p_domain_decoding;

            let om_storage;
            let om = if let Some(om) = om {
                om
            } else {
                om_storage = OProfile::convert(gm);
                &om_storage
            };
            let mut fwd_pmx_storage = None;
            let (fwd_pmx_ref, fwd_sc) =
                if let (Some(fwd_pmx), Some(fwd_sc)) = (fwd_pmx_input, fwd_sc_input) {
                    (fwd_pmx, fwd_sc)
                } else {
                    let mut fwd_pmx = crate::simd::probmx::ProbMx::new(l);
                    let fwd_sc = unsafe {
                        crate::simd::fwd_filter::forward_parser_pmx_offset_with_scratch(
                            dsq,
                            0,
                            l,
                            om,
                            &mut fwd_pmx,
                            &mut simd_scratch.fwd_dp,
                        )
                    };
                    let fwd_pmx_ref = fwd_pmx_storage.insert(fwd_pmx);
                    (&*fwd_pmx_ref, fwd_sc)
                };
            let mut bck_pmx = crate::simd::probmx::ProbMx::new(l);
            unsafe {
                crate::simd::bck_filter::backward_parser_pmx_offset_with_scratch(
                    dsq,
                    0,
                    l,
                    om,
                    fwd_sc,
                    &mut bck_pmx,
                    &mut simd_scratch.bck_prev,
                    &mut simd_scratch.bck_cur,
                );
            };
            let njc_loop = [
                om.xf[P7O_N][P7O_LOOP],
                om.xf[P7O_J][P7O_LOOP],
                om.xf[P7O_C][P7O_LOOP],
            ];
            let r = p_domain_decoding(fwd_pmx_ref, &bck_pmx, l, njc_loop);
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

    // Region detection using C's state machine
    let regions = find_domain_regions(&btot, &etot, &mocc, l);
    let mut stats = DomainDefinitionStats {
        nregions: regions.len(),
        ..Default::default()
    };
    let mut n2sc = vec![0.0_f32; l + 1];
    let mut env_gm = gm.clone();
    reconfig_unihit(&mut env_gm, l as i32);
    let optacc_deltas = OptAccTDelta::from_profile(&env_gm);
    #[cfg(target_arch = "x86_64")]
    let simd_match_odds = if use_simd {
        Some(crate::simd::probmx::match_odds_from_rsc(
            &gm.rsc, gm.abc_k, gm.m,
        ))
    } else {
        None
    };
    #[cfg(not(target_arch = "x86_64"))]
    let simd_match_odds: Option<Vec<f32>> = None;
    #[cfg(target_arch = "x86_64")]
    let env_om = if use_simd {
        Some(crate::simd::oprofile::OProfile::convert(&env_gm))
    } else {
        None
    };
    #[cfg(not(target_arch = "x86_64"))]
    let env_om: Option<crate::simd::oprofile::OProfile> = None;

    if regions.is_empty() {
        // No regions found — if nexpected > 0, return single domain covering sequence
        if nexpected >= 0.5 {
            let dom = score_domain_envelope(
                dsq,
                l,
                gm,
                &env_gm,
                env_om.as_ref(),
                hmm,
                1,
                l,
                null_sc,
                &bg.f,
                Some(&mut n2sc),
                false,
                simd_match_odds.as_deref(),
                &optacc_deltas,
                Some(&mut simd_scratch),
                make_alignment,
                make_alignment_display,
            );
            stats.nenvelopes += 1;
            let seq_bias = n2sc.iter().sum();
            return (vec![dom], nexpected, seq_bias, stats);
        }
        return (Vec::new(), nexpected, 0.0, stats);
    }

    let mut domains = Vec::new();

    for &(ri, rj) in &regions {
        if is_multidomain_region(&btot, &etot, ri, rj) {
            stats.nclustered += 1;
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
            for pos in ri..=rj {
                n2sc[pos] = 0.0;
            }

            for trace_idx in 0..NSAMPLES {
                let tr =
                    g_stochastic_trace(&mut rng, &sub_dsq, region_len, &region_gm, &region_fwd);
                let mut in_domain = false;
                let mut seg_i = 0;
                let mut seg_k = 0;
                let mut seg_j = 0;
                let mut seg_m = 0;
                let mut zstart = 0usize;
                let mut trace_domains = Vec::new();

                for z in 0..tr.n {
                    match tr.st[z] {
                        State::B => {
                            in_domain = true;
                            seg_i = 0;
                            seg_k = 0;
                            zstart = z;
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
                                let global_i = seg_i + ri - 1;
                                let global_j = seg_j + ri - 1;
                                segments.push(SegmentPair {
                                    i: global_i,
                                    j: global_j,
                                    k: seg_k,
                                    m: seg_m,
                                    trace_idx,
                                });
                                trace_domains.push((seg_i, seg_j, zstart, z));
                            }
                            in_domain = false;
                        }
                        _ => {}
                    }
                }

                let mut pos = 1usize;
                for &(sqfrom, sqto, tfrom, tto) in &trace_domains {
                    let null2 = generic_null2::null2_by_trace(&region_gm, &tr, tfrom, tto);
                    while pos <= sqfrom && pos <= region_len {
                        n2sc[ri + pos - 1] += 1.0;
                        pos += 1;
                    }
                    while pos <= sqto && pos <= region_len {
                        let x = dsq[ri + pos - 1] as usize;
                        n2sc[ri + pos - 1] += if x < null2.len() { null2[x] } else { 1.0 };
                        pos += 1;
                    }
                }
                while pos <= region_len {
                    n2sc[ri + pos - 1] += 1.0;
                    pos += 1;
                }
            }

            for pos in ri..=rj {
                n2sc[pos] = (n2sc[pos] / NSAMPLES as f32).ln();
            }

            if !segments.is_empty() {
                let params = ClusterParams::default();
                let envelopes = spensemble::cluster(&segments, NSAMPLES, &params);
                let mut last_jenv = 0usize;
                for env in &envelopes {
                    let ienv = env.ienv.max(1).min(l);
                    let jenv = env.jenv.max(1).min(l);
                    if jenv >= ienv {
                        if ienv <= last_jenv {
                            stats.noverlaps += 1;
                        }
                        stats.nenvelopes += 1;
                        domains.push(score_domain_envelope(
                            dsq,
                            l,
                            gm,
                            &env_gm,
                            env_om.as_ref(),
                            hmm,
                            ienv,
                            jenv,
                            null_sc,
                            &bg.f,
                            Some(&mut n2sc),
                            true,
                            simd_match_odds.as_deref(),
                            &optacc_deltas,
                            Some(&mut simd_scratch),
                            make_alignment,
                            make_alignment_display,
                        ));
                        last_jenv = jenv;
                    }
                }
            } else {
                // Clustering failed — treat as single domain
                stats.nenvelopes += 1;
                domains.push(score_domain_envelope(
                    dsq,
                    l,
                    gm,
                    &env_gm,
                    env_om.as_ref(),
                    hmm,
                    ri,
                    rj,
                    null_sc,
                    &bg.f,
                    Some(&mut n2sc),
                    false,
                    simd_match_odds.as_deref(),
                    &optacc_deltas,
                    Some(&mut simd_scratch),
                    make_alignment,
                    make_alignment_display,
                ));
            }
        } else {
            // Single-domain region: the region IS the envelope
            stats.nenvelopes += 1;
            domains.push(score_domain_envelope(
                dsq,
                l,
                gm,
                &env_gm,
                env_om.as_ref(),
                hmm,
                ri,
                rj,
                null_sc,
                &bg.f,
                Some(&mut n2sc),
                false,
                simd_match_odds.as_deref(),
                &optacc_deltas,
                Some(&mut simd_scratch),
                make_alignment,
                make_alignment_display,
            ));
        }
    }

    // Sort domains by envelope start position
    domains.sort_by_key(|d| d.ienv);

    let seq_bias = n2sc.iter().sum();
    (domains, nexpected, seq_bias, stats)
}
