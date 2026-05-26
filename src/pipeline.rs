//! Search pipeline using SIMD MSV + Viterbi + Forward filters.

use crate::bg::Bg;
use crate::dp::gmx::Gmx;
use crate::profile::*;
use crate::sequence::Sequence;
#[cfg(target_arch = "x86_64")]
use crate::simd::msv_filter::MsvResult;
use crate::simd::oprofile::OProfile;
#[cfg(target_arch = "x86_64")]
use crate::simd::vit_filter::{viterbi_filter, VitResult};
use crate::stats;
use crate::tophits::*;
use crate::util::cmath::{c_exp_f64, c_log_f64, ESL_CONST_LOG2};

/// Tracehash hook: record a single MSV/bias/Viterbi filter decision for cross-impl validation.
#[cfg(feature = "tracehash")]
fn trace_pipeline_decision(
    function: &'static str,
    dsq: &[u8],
    l: usize,
    m: usize,
    score: f32,
    baseline: f32,
    pvalue: f64,
    ran: bool,
    passed: bool,
) {
    let mut th = match function {
        "pipeline_msv_decision" => tracehash::th_call!("pipeline_msv_decision"),
        "pipeline_bias_decision" => tracehash::th_call!("pipeline_bias_decision"),
        _ => tracehash::th_call!("pipeline_vit_decision"),
    };
    th.input_usize(l);
    th.input_usize(m);
    th.input_bytes(&dsq[1..=l]);
    th.output_u64(ran as u64);
    th.output_f32(score);
    th.output_f32(baseline);
    th.output_f32(pvalue as f32);
    th.output_u64(passed as u64);
    th.finish();
}

/// Tracehash hook: dump the full-sequence bias correction inputs and result.
#[cfg(feature = "tracehash")]
fn trace_pipeline_full_seq_bias_detail(
    dsq: &[u8],
    l: usize,
    m: usize,
    do_null2: bool,
    seq_correction_sum: f32,
    seq_correction_fsum: f32,
    omega: f32,
    log_omega: f64,
    bias_arg: f32,
    flogsum_index: u64,
    seqbias: f32,
) {
    let mut th = tracehash::th_call!("pipeline_full_seq_bias_detail");
    th.input_usize(l);
    th.input_usize(m);
    th.input_bytes(&dsq[1..=l]);
    th.output_u64(do_null2 as u64);
    th.output_f32(seq_correction_sum);
    th.output_f32(seq_correction_fsum);
    th.output_f32(omega);
    th.output_u64(log_omega.to_bits());
    th.output_f32(bias_arg);
    th.output_u64(flogsum_index);
    th.output_f32(seqbias);
    th.finish();
}

/// Tracehash hook: dump intermediate per-sequence score components and emit per-field traces.
#[cfg(feature = "tracehash")]
#[allow(clippy::too_many_arguments)]
fn trace_pipeline_score_components(
    dsq: &[u8],
    l: usize,
    m: usize,
    fwd_sc: f32,
    null_sc: f32,
    full_seq_bias: f32,
    pre_score: f32,
    direct_seq_score: f32,
    sum_score_nats: f32,
    sum_bias: f32,
    pre2_score: f32,
    sum_score: f32,
    final_pre_score: f32,
    final_seq_score: f32,
    ld: usize,
    ndom: usize,
) {
    let mut th = tracehash::th_call!("pipeline_score_components");
    th.input_usize(l);
    th.input_usize(m);
    th.input_bytes(&dsq[1..=l]);
    th.output_f32(fwd_sc);
    th.output_f32(null_sc);
    th.output_f32(full_seq_bias);
    th.output_f32(pre_score);
    th.output_f32(direct_seq_score);
    th.output_f32(sum_score_nats);
    th.output_f32(sum_bias);
    th.output_f32(pre2_score);
    th.output_f32(sum_score);
    th.output_f32(final_pre_score);
    th.output_f32(final_seq_score);
    th.output_u64(ld as u64);
    th.output_u64(ndom as u64);
    th.finish();

    macro_rules! emit_f32 {
        ($name:literal, $value:expr) => {{
            let mut th = tracehash::th_call!($name);
            th.input_usize(l);
            th.input_usize(m);
            th.input_bytes(&dsq[1..=l]);
            th.output_f32($value);
            th.output_f32_quant($value, 1.0e-5);
            th.finish();
        }};
    }
    emit_f32!("pipeline_score_fwd_sc", fwd_sc);
    emit_f32!("pipeline_score_full_seq_bias", full_seq_bias);
    emit_f32!("pipeline_score_direct_seq", direct_seq_score);
    emit_f32!("pipeline_score_sum_nats", sum_score_nats);
    emit_f32!("pipeline_score_sum_bias", sum_bias);
    emit_f32!("pipeline_score_sum_bits", sum_score);
    emit_f32!("pipeline_score_final_seq", final_seq_score);

    let mut th = tracehash::th_call!("pipeline_score_lengths");
    th.input_usize(l);
    th.input_usize(m);
    th.input_bytes(&dsq[1..=l]);
    th.output_u64(ld as u64);
    th.output_u64(ndom as u64);
    th.finish();
}

/// Tracehash helper: bucket a flogsum() input that should round to 0, for hash stability.
#[cfg(feature = "tracehash")]
fn flogsum_index_for_zero_arg(value: f32) -> u64 {
    if value == f32::NEG_INFINITY || value.abs() >= 15.7 {
        u64::MAX
    } else {
        (value.abs() * 1000.0) as u64
    }
}

/// Tracehash hook: dump one candidate domain's envelope and bias correction.
#[cfg(feature = "tracehash")]
fn trace_pipeline_domain_score_candidate(
    dsq: &[u8],
    l: usize,
    m: usize,
    domain_idx: usize,
    ienv: i64,
    jenv: i64,
    envsc: f32,
    domcorrection: f32,
    use_domain: bool,
) {
    let mut th = tracehash::th_call!("pipeline_domain_score_candidate");
    th.input_usize(l);
    th.input_usize(m);
    th.input_usize(domain_idx);
    th.input_bytes(&dsq[1..=l]);
    th.output_u64(ienv as u64);
    th.output_u64(jenv as u64);
    th.output_f32(envsc);
    th.output_f32(domcorrection);
    th.output_u64(use_domain as u64);
    th.finish();
}

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
    // long_target = nhmmer-style DNA genome search. Switches domain-scoring to
    // reconfigure the profile's rest-length per envelope, matching C
    // `rescore_isolated_domain()`'s `if (long_target) ReconfigRestLength(om, j-i+1)`.
    pub long_target: bool,

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

    #[cfg(all(target_arch = "x86_64", not(feature = "tracehash")))]
    msv_dp: Vec<std::arch::x86_64::__m128i>,
    #[cfg(target_arch = "x86_64")]
    fwd_pmx: crate::simd::probmx::ProbMx,
    #[cfg(target_arch = "x86_64")]
    domain_scratch: crate::domaindef::DomainSimdScratch,
}

impl Pipeline {
    /// Create a new accelerated comparison pipeline with default thresholds.
    /// Port of `p7_pipeline_Create()`: sets reporting/inclusion E-values, F1/F2/F3
    /// filter thresholds, RNG seed, and bias/null2 flags to HMMER 3 defaults.
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
            long_target: false,
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
            #[cfg(all(target_arch = "x86_64", not(feature = "tracehash")))]
            msv_dp: Vec::new(),
            #[cfg(target_arch = "x86_64")]
            fwd_pmx: crate::simd::probmx::ProbMx::new(0),
            #[cfg(target_arch = "x86_64")]
            domain_scratch: crate::domaindef::DomainSimdScratch::new(),
        }
    }

    /// Configure the pipeline for a new query/target model.
    /// Port of `p7_pli_NewModel()`: copies the model's E-value calibration
    /// parameters into the pipeline so subsequent filter p-values use them.
    pub fn new_model(&mut self, gm: &Profile) {
        self.evparam = gm.evparam;
    }

    /// True when target ranking should follow bit score instead of P-value.
    /// `TopHits` sorts ascending, so score-based sort keys are stored as
    /// negative scores to keep higher-scoring hits first.
    pub fn score_sort_active(&self) -> bool {
        self.use_bit_cutoffs != BitCutoff::None || (!self.inc_by_e && self.inc_t.is_some())
    }

    pub fn hit_sortkey(&self, score: f32, lnp: f64) -> f64 {
        if self.score_sort_active() {
            -(score as f64)
        } else {
            lnp
        }
    }

    pub fn target_reportable(&self, score: f32, lnp: f64) -> bool {
        if self.by_e {
            if self.long_target {
                c_exp_f64(lnp) <= self.e_value_threshold
            } else {
                c_exp_f64(lnp) * self.current_target_z() <= self.e_value_threshold
            }
        } else if let Some(t) = self.t {
            score >= t
        } else {
            c_exp_f64(lnp) * self.current_target_z() <= self.e_value_threshold
        }
    }

    fn target_includable(&self, score: f32, lnp: f64) -> bool {
        if self.inc_by_e {
            if self.long_target {
                c_exp_f64(lnp) <= self.inc_e
            } else {
                c_exp_f64(lnp) * self.current_target_z() <= self.inc_e
            }
        } else if let Some(t) = self.inc_t {
            score >= t
        } else {
            c_exp_f64(lnp) * self.current_target_z() <= self.inc_e
        }
    }

    fn current_target_z(&self) -> f64 {
        match self.z_setby {
            ZSetBy::Option => self.z.max(1.0),
            ZSetBy::Ntargets => self.z.max(self.n_targets as f64).max(1.0),
        }
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
                if cutoffs[P7_GA1] == CUTOFF_UNSET || cutoffs[P7_GA2] == CUTOFF_UNSET {
                    return Err("GA cutoff not set in model".to_string());
                }
                self.t = Some(cutoffs[P7_GA1]);
                self.inc_t = Some(cutoffs[P7_GA1]);
                self.dom_t = Some(cutoffs[P7_GA2]);
                self.inc_dom_t = Some(cutoffs[P7_GA2]);
            }
            BitCutoff::TC => {
                if cutoffs[P7_TC1] == CUTOFF_UNSET || cutoffs[P7_TC2] == CUTOFF_UNSET {
                    return Err("TC cutoff not set in model".to_string());
                }
                self.t = Some(cutoffs[P7_TC1]);
                self.inc_t = Some(cutoffs[P7_TC1]);
                self.dom_t = Some(cutoffs[P7_TC2]);
                self.inc_dom_t = Some(cutoffs[P7_TC2]);
            }
            BitCutoff::NC => {
                if cutoffs[P7_NC1] == CUTOFF_UNSET || cutoffs[P7_NC2] == CUTOFF_UNSET {
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

        // C HMMER runs the bias filter only after MSV passes; keep it lazy.
        let mut filtersc = null_sc;

        // For long_target, SSV_longtarget and ViterbiFilter_longtarget already
        // ran at the postSSV stage (in nhmmer::search_longtarget); C's
        // postViterbi_LongTarget (p7_pipeline.c:1289) does NOT re-run MSV or
        // Viterbi — it calls p7_ForwardParser directly. Match that here.
        if !self.do_max && !self.long_target {
            let mut filter_p = 0.0_f64;

            // Stage 1: MSV filter
            let usc: f32;
            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("sse2") {
                    #[cfg(feature = "tracehash")]
                    let msv_result = unsafe { crate::simd::msv_filter::msv_filter(&sq.dsq, l, om) };
                    #[cfg(not(feature = "tracehash"))]
                    let msv_result = unsafe {
                        crate::simd::msv_filter::msv_filter_with_scratch(
                            &sq.dsq,
                            l,
                            om,
                            &mut self.msv_dp,
                        )
                    };
                    usc = match msv_result {
                        MsvResult::Ok(sc) => sc,
                        MsvResult::Overflow => f32::INFINITY,
                    };
                } else {
                    let mut gx = Gmx::new(gm.m, l);
                    usc = crate::dp::generic_msv::g_msv(&sq.dsq, l, gm, &mut gx, 2.0);
                }
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                let mut gx = Gmx::new(gm.m, l);
                usc = crate::dp::generic_msv::g_msv(&sq.dsq, l, gm, &mut gx, 2.0);
            }
            if usc != f32::INFINITY {
                let msv_pval = msv_pvalue(usc, null_sc, &self.evparam);
                if msv_pval > self.f1 {
                    #[cfg(feature = "tracehash")]
                    trace_pipeline_decision(
                        "pipeline_msv_decision",
                        &sq.dsq,
                        l,
                        gm.m,
                        usc,
                        null_sc,
                        msv_pval,
                        true,
                        false,
                    );
                    return false;
                }
                filter_p = msv_pval;
            }
            #[cfg(feature = "tracehash")]
            trace_pipeline_decision(
                "pipeline_msv_decision",
                &sq.dsq,
                l,
                gm.m,
                usc,
                null_sc,
                filter_p,
                true,
                true,
            );
            self.n_past_msv += 1;

            // Stage 1b: Bias composition filter
            if self.do_biasfilter {
                filtersc = bg.filter_score(&sq.dsq, l);
                let bias_pval = msv_pvalue(usc, filtersc, &self.evparam);
                if bias_pval > self.f1 {
                    #[cfg(feature = "tracehash")]
                    trace_pipeline_decision(
                        "pipeline_bias_decision",
                        &sq.dsq,
                        l,
                        gm.m,
                        usc,
                        filtersc,
                        bias_pval,
                        true,
                        false,
                    );
                    return false;
                }
                filter_p = bias_pval;
            }
            #[cfg(feature = "tracehash")]
            trace_pipeline_decision(
                "pipeline_bias_decision",
                &sq.dsq,
                l,
                gm.m,
                usc,
                filtersc,
                filter_p,
                self.do_biasfilter,
                true,
            );
            self.n_past_bias += 1;

            // Stage 2: Viterbi filter (uses filtersc as baseline, matching C)
            #[cfg(feature = "tracehash")]
            let mut vit_ran = false;
            #[cfg(feature = "tracehash")]
            let mut vit_sc_for_trace = f32::INFINITY;
            #[cfg(feature = "tracehash")]
            let mut vit_p_for_trace = filter_p;
            if filter_p > self.f2 {
                #[cfg(feature = "tracehash")]
                {
                    vit_ran = true;
                }
                let vit_sc: f32;
                #[cfg(target_arch = "x86_64")]
                {
                    if is_x86_feature_detected!("sse2") {
                        let vit_result = unsafe { viterbi_filter(&sq.dsq, l, om) };
                        vit_sc = match vit_result {
                            VitResult::Ok(sc) => sc,
                            VitResult::Overflow => f32::INFINITY,
                        };
                    } else {
                        let mut gx = Gmx::new(gm.m, l);
                        vit_sc = crate::dp::generic_viterbi::g_viterbi(&sq.dsq, l, gm, &mut gx);
                    }
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    let mut gx = Gmx::new(gm.m, l);
                    vit_sc = crate::dp::generic_viterbi::g_viterbi(&sq.dsq, l, gm, &mut gx);
                }
                #[cfg(feature = "tracehash")]
                {
                    vit_sc_for_trace = vit_sc;
                }

                if vit_sc != f32::INFINITY {
                    let vit_pval = viterbi_pvalue(vit_sc, filtersc, &self.evparam);
                    #[cfg(feature = "tracehash")]
                    {
                        vit_p_for_trace = vit_pval;
                    }
                    if vit_pval > self.f2 {
                        #[cfg(feature = "tracehash")]
                        trace_pipeline_decision(
                            "pipeline_vit_decision",
                            &sq.dsq,
                            l,
                            gm.m,
                            vit_sc_for_trace,
                            filtersc,
                            vit_p_for_trace,
                            vit_ran,
                            false,
                        );
                        return false;
                    }
                }
            }
            #[cfg(feature = "tracehash")]
            trace_pipeline_decision(
                "pipeline_vit_decision",
                &sq.dsq,
                l,
                gm.m,
                vit_sc_for_trace,
                filtersc,
                vit_p_for_trace,
                vit_ran,
                true,
            );
            self.n_past_vit += 1;
        } else {
            self.n_past_msv += 1;
            self.n_past_bias += 1;
            self.n_past_vit += 1;
        }

        // Long-target bias scaling for Forward stage: match C
        // p7_pli_postViterbi_LongTarget (p7_pipeline.c:1330):
        //   F3_L = min(window_len, B3=1000);
        //   filtersc = nullsc + bias_filtersc * (F3_L == window_len ? 1.0 : F3_L/window_len).
        // This caps the bias contribution at B3=1000 residues, making the F3
        // filter more permissive on long windows.
        if self.long_target && self.do_biasfilter {
            let bias_full = bg.filter_score(&sq.dsq, l);
            let bias_delta = bias_full - null_sc;
            const B3: usize = 1000;
            let f3_l = l.min(B3);
            let scale = if f3_l == l {
                1.0_f32
            } else {
                f3_l as f32 / l as f32
            };
            filtersc = null_sc + bias_delta * scale;
        }

        // Stage 3: full Forward parser.
        let fwd_sc;
        #[cfg(target_arch = "x86_64")]
        let mut fwd_pmx_for_domains = None;
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("sse2") {
                self.fwd_pmx.resize_parser(l);
                fwd_sc = unsafe {
                    crate::simd::fwd_filter::forward_parser_pmx(&sq.dsq, l, om, &mut self.fwd_pmx)
                };
                fwd_pmx_for_domains = Some(&self.fwd_pmx);
            } else {
                let mut gx = Gmx::new(gm.m, l);
                fwd_sc = crate::dp::generic_fwdback::g_forward(&sq.dsq, l, gm, &mut gx);
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let mut gx = Gmx::new(gm.m, l);
            fwd_sc = crate::dp::generic_fwdback::g_forward(&sq.dsq, l, gm, &mut gx);
        }

        let fwd_pval = forward_pvalue(fwd_sc, filtersc, &self.evparam);
        if !self.do_max && fwd_pval > self.f3 {
            return false;
        }
        self.n_past_fwd += 1;

        // Sequence passes all filters — run domain definition
        let (mut domains, nexpected, seq_correction_sum, pipeline_seq_correction_sum, domain_stats) =
            crate::domaindef::define_domains(
                &sq.dsq,
                l,
                gm,
                Some(om),
                #[cfg(target_arch = "x86_64")]
                fwd_pmx_for_domains,
                #[cfg(not(target_arch = "x86_64"))]
                None,
                Some(fwd_sc),
                hmm,
                bg,
                null_sc,
                self.seed,
                #[cfg(target_arch = "x86_64")]
                &mut self.domain_scratch,
                self.do_alignment,
                self.do_alignment_display,
                self.long_target,
            );
        #[cfg(not(feature = "tracehash"))]
        let _ = seq_correction_sum;
        if self.long_target {
            domains.retain(|dom| {
                let span = dom
                    .ad
                    .as_ref()
                    .map(|ad| ad.sqfrom.abs_diff(ad.sqto) + 1)
                    .unwrap_or_else(|| (dom.iali - dom.jali).unsigned_abs() as usize + 1);
                span >= 8
            });
        }
        if domains.is_empty() {
            return false;
        }

        // Sequence-level scoring
        // Match C HMMER's two sequence scores:
        // 1. the full-sequence Forward score corrected by summed n2sc[]
        // 2. a reconstruction score from individually rescored domains
        let omega = bg.omega;
        let mut pre_score = nats_to_bits_from_scores(fwd_sc, null_sc);
        #[cfg(feature = "tracehash")]
        let seqbias_log_omega = c_log_f64(omega as f64);
        #[cfg(feature = "tracehash")]
        let seqbias_arg = if self.do_null2 {
            (seqbias_log_omega + pipeline_seq_correction_sum as f64) as f32
        } else {
            0.0
        };
        let seqbias = if self.do_null2 {
            crate::logsum::p7_flogsum(
                0.0,
                (c_log_f64(omega as f64) + pipeline_seq_correction_sum as f64) as f32,
            )
        } else {
            0.0
        };
        #[cfg(feature = "tracehash")]
        trace_pipeline_full_seq_bias_detail(
            &sq.dsq,
            l,
            gm.m,
            self.do_null2,
            seq_correction_sum,
            pipeline_seq_correction_sum,
            omega,
            seqbias_log_omega,
            seqbias_arg,
            flogsum_index_for_zero_arg(seqbias_arg),
            seqbias,
        );
        let mut seq_score = nats_to_bits_from_scores(fwd_sc, null_sc + seqbias);
        let direct_seq_score = seq_score;
        #[cfg(not(feature = "tracehash"))]
        let _ = direct_seq_score;

        let mut sum_env = 0.0_f32;
        let mut sum_correction = 0.0_f32;
        let mut ld = 0usize;
        for (domain_idx, dom) in domains.iter().enumerate() {
            #[cfg(not(feature = "tracehash"))]
            let _ = domain_idx;
            let use_domain = if self.do_null2 {
                dom.envsc - dom.domcorrection > 0.0
            } else {
                dom.envsc > 0.0
            };
            #[cfg(feature = "tracehash")]
            trace_pipeline_domain_score_candidate(
                &sq.dsq,
                l,
                gm.m,
                domain_idx,
                dom.ienv,
                dom.jenv,
                dom.envsc,
                dom.domcorrection,
                use_domain,
            );
            if use_domain {
                sum_env += dom.envsc;
                sum_correction += dom.domcorrection;
                ld += (dom.jenv - dom.ienv + 1) as usize;
            }
        }
        let sum_score_nats = reconstruction_score_nats(sum_env, l, ld);
        let sum_bias = if self.do_null2 {
            crate::logsum::p7_flogsum(
                0.0,
                (c_log_f64(omega as f64) + sum_correction as f64) as f32,
            )
        } else {
            0.0
        };
        let pre2_score = nats_to_bits_from_scores(sum_score_nats, null_sc);
        let sum_score = nats_to_bits_from_scores(sum_score_nats, null_sc + sum_bias);
        if ld > 0 && sum_score > seq_score {
            seq_score = sum_score;
            pre_score = pre2_score;
        }
        let seq_bias_bits = (pre_score - seq_score).max(0.0);
        #[cfg(feature = "tracehash")]
        trace_pipeline_score_components(
            &sq.dsq,
            l,
            gm.m,
            fwd_sc,
            null_sc,
            seqbias,
            nats_to_bits_from_scores(fwd_sc, null_sc),
            direct_seq_score,
            sum_score_nats,
            sum_bias,
            pre2_score,
            sum_score,
            pre_score,
            seq_score,
            ld,
            domains.len(),
        );

        if !self.do_null2 {
            let tau = gm.evparam[crate::hmm::P7_FTAU] as f64;
            let lambda = gm.evparam[crate::hmm::P7_FLAMBDA] as f64;
            for dom in &mut domains {
                let env_len = (dom.jenv - dom.ienv + 1) as usize;
                let length_correction = reconstruction_score_nats(0.0, l, env_len);
                dom.dombias = 0.0;
                dom.bitscore = nats_to_bits_from_scores(dom.envsc + length_correction, null_sc);
                dom.lnp = crate::stats::exponential::logsurv(dom.bitscore as f64, tau, lambda);
            }
        }

        // P-values
        let mu_f = self.evparam[crate::hmm::P7_FTAU] as f64;
        let lam_f = self.evparam[crate::hmm::P7_FLAMBDA] as f64;
        let lnp = stats::exponential::logsurv(seq_score as f64, mu_f, lam_f);
        let pre_lnp = stats::exponential::logsurv(pre_score as f64, mu_f, lam_f);

        // nhmmer (long_target) reports the best per-envelope score as hit.score;
        // it does not aggregate across domains the way hmmsearch does.
        // Mirrors C p7_pipeline.c:1462 `hit->sum_score = hit->score = dom_score`.
        let (hit_score, hit_pre_score, hit_bias, hit_lnp, hit_pre_lnp) = if self.long_target {
            let best = domains
                .iter()
                .min_by(|a, b| a.lnp.total_cmp(&b.lnp))
                .expect("long_target hit must have at least one domain");
            let best_bitscore = best.bitscore;
            // dom.dombias is already in bits (see domaindef.rs).
            let best_bias = best.dombias.max(0.0);
            let pre = best_bitscore + best_bias;
            let pre_lnp_lt = stats::exponential::logsurv(pre as f64, mu_f, lam_f);
            (best_bitscore, pre, best_bias, best.lnp, pre_lnp_lt)
        } else {
            (seq_score, pre_score, seq_bias_bits, lnp, pre_lnp)
        };

        if !self.target_reportable(hit_score, hit_lnp) {
            return false;
        }

        let hit = th.create_next_hit();
        hit.name = sq.name.clone();
        hit.acc = sq.acc.clone();
        hit.desc = sq.desc.clone();
        hit.n = sq.n;
        hit.score = hit_score;
        hit.bias = hit_bias;
        hit.pre_score = hit_pre_score;
        hit.sum_score = if self.long_target {
            hit_score
        } else {
            sum_score
        };
        hit.lnp = hit_lnp;
        hit.pre_lnp = hit_pre_lnp;
        hit.sum_lnp = if self.long_target {
            hit_lnp
        } else {
            stats::exponential::logsurv(sum_score as f64, mu_f, lam_f)
        };
        hit.sortkey = self.hit_sortkey(hit.score, hit.lnp);
        hit.nexpected = nexpected;
        hit.nregions = domain_stats.nregions;
        hit.nclustered = domain_stats.nclustered;
        hit.noverlaps = domain_stats.noverlaps;
        hit.nenvelopes = domain_stats.nenvelopes;
        hit.ndom = domains.len();
        hit.dcl = domains;

        if self.use_bit_cutoffs != BitCutoff::None {
            let target_reported = if self.by_e {
                (self.z.max(1.0)) * c_exp_f64(hit.lnp) <= self.e_value_threshold
            } else if let Some(t) = self.t {
                hit.score >= t
            } else {
                (self.z.max(1.0)) * c_exp_f64(hit.lnp) <= self.e_value_threshold
            };
            if target_reported {
                hit.flags |= P7_IS_REPORTED;
                if self.target_includable(hit.score, hit.lnp) {
                    hit.flags |= P7_IS_INCLUDED;
                }
            }

            for dom in &mut hit.dcl {
                let dom_reported = if self.dom_by_e {
                    (self.domz.max(1.0)) * c_exp_f64(dom.lnp) <= self.dom_e_value_threshold
                } else if let Some(t) = self.dom_t {
                    dom.bitscore >= t
                } else {
                    (self.domz.max(1.0)) * c_exp_f64(dom.lnp) <= self.dom_e_value_threshold
                };
                if dom_reported {
                    dom.is_reported = true;
                    let dom_included = if self.incdom_by_e {
                        (self.domz.max(1.0)) * c_exp_f64(dom.lnp) <= self.inc_dome
                    } else if let Some(t) = self.inc_dom_t {
                        dom.bitscore >= t
                    } else {
                        (self.domz.max(1.0)) * c_exp_f64(dom.lnp) <= self.inc_dome
                    };
                    if dom_included {
                        dom.is_included = true;
                    }
                }
            }
        }

        true
    }
}

/// Calculate MSV p-value from raw score.
pub fn msv_pvalue(msv_sc: f32, null_sc: f32, evparam: &[f32; 6]) -> f64 {
    let score = nats_to_bits_from_scores(msv_sc, null_sc);
    let mu = evparam[crate::hmm::P7_MMU] as f64;
    let lambda = evparam[crate::hmm::P7_MLAMBDA] as f64;
    stats::gumbel::surv(score as f64, mu, lambda)
}

/// Calculate Viterbi p-value from raw score.
fn viterbi_pvalue(vit_sc: f32, null_sc: f32, evparam: &[f32; 6]) -> f64 {
    let score = nats_to_bits_from_scores(vit_sc, null_sc);
    let mu = evparam[crate::hmm::P7_VMU] as f64;
    let lambda = evparam[crate::hmm::P7_VLAMBDA] as f64;
    stats::gumbel::surv(score as f64, mu, lambda)
}

/// Calculate Forward p-value from raw score.
fn forward_pvalue(fwd_sc: f32, null_sc: f32, evparam: &[f32; 6]) -> f64 {
    let score = nats_to_bits_from_scores(fwd_sc, null_sc);
    let tau = evparam[crate::hmm::P7_FTAU] as f64;
    let lambda = evparam[crate::hmm::P7_FLAMBDA] as f64;
    stats::exponential::surv(score as f64, tau, lambda)
}

/// Convert a raw score (nats, relative to `baseline`) into bits via division by ln(2).
#[inline]
fn nats_to_bits_from_scores(score: f32, baseline: f32) -> f32 {
    (((score - baseline) as f64) / ESL_CONST_LOG2) as f32
}

/// Sequence reconstruction score in nats: sum of envelope scores plus a
/// length-correction term for residues not covered by any envelope. Used to
/// build the alternative per-sequence score `sum_score` in `p7_Pipeline()`.
#[inline]
fn reconstruction_score_nats(sum_env: f32, l: usize, ld: usize) -> f32 {
    let len_ratio = l as f32 / (l as f32 + 3.0);
    let uncovered = l as i64 - ld as i64;
    (sum_env as f64 + uncovered as f64 * c_log_f64(len_ratio as f64)) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alphabet::Alphabet;
    use std::path::Path;

    #[test]
    fn target_reportable_uses_explicit_z_before_admitting_hit() {
        let mut pli = Pipeline::new();
        pli.z = 1_000.0;
        pli.z_setby = ZSetBy::Option;
        pli.e_value_threshold = 1.0;

        assert!(!pli.target_reportable(10.0, (0.01_f64).ln()));
        assert!(pli.target_reportable(10.0, (0.0001_f64).ln()));
    }

    #[test]
    fn target_reportable_long_target_uses_prescaled_pvalue() {
        let mut pli = Pipeline::new();
        pli.long_target = true;
        pli.z = 1_000.0;
        pli.z_setby = ZSetBy::Option;
        pli.e_value_threshold = 1.0;

        assert!(pli.target_reportable(10.0, (0.01_f64).ln()));
    }

    /// Smoke test: pipeline produces at least one positive-score hit on a perfectly matching dsq.
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

    /// Regression test: ld > l (envelopes overlap or extend) must not produce NaN/+inf.
    #[test]
    fn reconstruction_score_allows_domain_coverage_to_exceed_sequence_length() {
        let score = reconstruction_score_nats(12.0, 100, 140);
        let expected = 12.0_f64 + (100_i64 - 140_i64) as f64 * c_log_f64(100.0_f64 / 103.0_f64);
        assert!((score as f64 - expected).abs() < 1.0e-6);
        assert!(score.is_finite());
        assert!(score > 12.0);
    }
}
