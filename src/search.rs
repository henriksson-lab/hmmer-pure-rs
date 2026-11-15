//! Checked, reusable local multihit searches over complete target sequences.
//!
//! [`Searcher`] owns scoring buffers and borrows its query HMM. Batch searches
//! and streaming sessions share the same setup and final thresholding. This
//! API performs ordinary sequence searches, not strand-aware genome scanning.

use crate::alphabet::{Alphabet, DSQ_SENTINEL};
use crate::bg::Bg;
use crate::errors::{HmmerError, HmmerResult};
use crate::hmm::{Hmm, COMPO_UNSET, EVPARAM_UNSET, P7_FLAMBDA, P7_MLAMBDA, P7_VLAMBDA};
use crate::pipeline::{Pipeline, ZSetBy};
use crate::profile::{hmm_average_match_composition, Profile, SearchMode};
use crate::sequence::Sequence;
use crate::simd::oprofile::OProfile;
use crate::tophits::{Domain, Hit, TopHits, P7_IS_INCLUDED, P7_IS_REPORTED};
use crate::util::cmath::c_exp_f64;

/// Settings for the convenience search API; defaults match [`Pipeline::new`].
///
/// E-values are linear expected counts, not log values. Settings are checked
/// by [`Searcher::with_options`] and remain fixed for each searcher.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SearchOptions {
    /// Maximum sequence E-value for reporting (default 10).
    pub sequence_evalue: f64,
    /// Maximum conditional domain E-value for reporting (default 10).
    pub domain_evalue: f64,
    /// Maximum sequence E-value for inclusion (default 0.01).
    pub inclusion_evalue: f64,
    /// Maximum conditional domain E-value for inclusion (default 0.01).
    pub domain_inclusion_evalue: f64,
    /// Effective sequence count, or `None` to count every added target,
    /// including empty sequences. Set this when searching a database subset.
    pub database_size: Option<f64>,
    /// Effective conditional-domain search space, or `None` to use the
    /// number of reported sequences after final sequence thresholding.
    pub domain_database_size: Option<f64>,
    /// Bypass MSV, bias, Viterbi, and Forward filter gates (default false).
    /// This is slower and corresponds to the CLI's `--max` option.
    pub max_sensitivity: bool,
    /// Enable the composition bias filter (default true); ignored when
    /// `max_sensitivity` is true. Null2 score correction remains enabled.
    pub bias_filter: bool,
    /// Stochastic traceback seed (default 42); zero selects an arbitrary seed.
    pub seed: u32,
}

impl Default for SearchOptions {
    fn default() -> Self {
        let pipeline = Pipeline::new();
        Self {
            sequence_evalue: pipeline.e_value_threshold,
            domain_evalue: pipeline.dom_e_value_threshold,
            inclusion_evalue: pipeline.inc_e,
            domain_inclusion_evalue: pipeline.inc_dome,
            database_size: None,
            domain_database_size: None,
            max_sensitivity: false,
            bias_filter: true,
            seed: pipeline.seed,
        }
    }
}

impl SearchOptions {
    fn validate(&self) -> HmmerResult<()> {
        for (label, value) in [
            ("sequence_evalue", self.sequence_evalue),
            ("domain_evalue", self.domain_evalue),
            ("inclusion_evalue", self.inclusion_evalue),
            ("domain_inclusion_evalue", self.domain_inclusion_evalue),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(HmmerError::InvalidArg(format!(
                    "{label} must be finite and nonnegative"
                )));
            }
        }
        for (label, value) in [
            ("database_size", self.database_size),
            ("domain_database_size", self.domain_database_size),
        ] {
            if value.is_some_and(|v| !v.is_finite() || v <= 0.0) {
                return Err(HmmerError::InvalidArg(format!(
                    "{label} must be finite and positive"
                )));
            }
        }
        Ok(())
    }
}

/// Reusable scoring state for one calibrated query model.
///
/// Existing low-level APIs remain available for other modes and custom
/// pipelines. This wrapper uses local multihit alignment and HMMER's default
/// background. It borrows `hmm`, so its model cannot change during searches.
/// Construction and scoring initialize HMMER's floating point mode on the
/// calling thread (flush subnormal inputs/results to zero, as in the CLI).
pub struct Searcher<'h> {
    hmm: &'h Hmm,
    alphabet: Alphabet,
    background: Bg,
    profile: Profile,
    optimized: OProfile,
    pipeline: Pipeline,
    options: SearchOptions,
}

impl<'h> Searcher<'h> {
    /// Prepare a local multihit search with default options.
    ///
    /// Validates model inputs as in [`Profile::from_hmm`] and requires finite
    /// E-value calibration parameters with positive lambda parameters.
    /// Uncalibrated models return an error; no implicit calibration is run.
    /// A missing composition annotation is derived from match occupancies.
    pub fn new(hmm: &'h Hmm) -> HmmerResult<Self> {
        Self::with_options(hmm, SearchOptions::default())
    }

    /// Prepare a search with explicit thresholds, database sizes, and filters.
    ///
    /// Model prerequisites follow [`Self::new`]. Thresholds must be finite
    /// and nonnegative; explicit database sizes must be finite and positive.
    /// Invalid settings or model composition return [`HmmerError::InvalidArg`].
    pub fn with_options(hmm: &'h Hmm, options: SearchOptions) -> HmmerResult<Self> {
        options.validate()?;
        if hmm
            .evparam
            .iter()
            .any(|v| !v.is_finite() || *v == EVPARAM_UNSET)
            || [P7_MLAMBDA, P7_VLAMBDA, P7_FLAMBDA]
                .iter()
                .any(|&i| hmm.evparam[i] <= 0.0)
        {
            return Err(HmmerError::InvalidArg(
                "search requires a calibrated HMM; calibrate the model before constructing a Searcher".into(),
            ));
        }
        if hmm.abc_type == crate::alphabet::AlphabetType::Unknown {
            return Err(HmmerError::InvalidArg("model alphabet is unknown".into()));
        }
        let alphabet = Alphabet::new(hmm.abc_type);
        let mut background = Bg::new(&alphabet);
        let profile = Profile::from_hmm(hmm, &background, 0, SearchMode::LocalMultihit)?;
        let composition = if hmm.compo[..alphabet.k].iter().all(|v| *v == COMPO_UNSET) {
            hmm_average_match_composition(hmm)
        } else {
            hmm.compo[..alphabet.k].to_vec()
        };
        let sum: f64 = composition.iter().map(|&p| p as f64).sum();
        if composition
            .iter()
            .any(|p| !p.is_finite() || !(0.0..=1.0).contains(p))
            || (sum - 1.0).abs() > 0.001
        {
            return Err(HmmerError::InvalidArg(
                "model composition must be normalized probabilities".into(),
            ));
        }
        background.set_filter(hmm.m, &composition);
        crate::logsum::p7_flogsuminit();
        crate::util::simd_env::init();
        let optimized = OProfile::convert(&profile);
        let mut pipeline = Pipeline::new();
        pipeline.new_model(&profile);
        pipeline.e_value_threshold = options.sequence_evalue;
        pipeline.dom_e_value_threshold = options.domain_evalue;
        pipeline.inc_e = options.inclusion_evalue;
        pipeline.inc_dome = options.domain_inclusion_evalue;
        pipeline.do_max = options.max_sensitivity;
        pipeline.do_biasfilter = options.bias_filter && !options.max_sensitivity;
        pipeline.seed = options.seed;
        if options.max_sensitivity {
            pipeline.f1 = 1.0;
            pipeline.f2 = 1.0;
            pipeline.f3 = 1.0;
        }
        pipeline.z_setby = if options.database_size.is_some() {
            ZSetBy::Option
        } else {
            ZSetBy::Ntargets
        };
        pipeline.domz_setby = if options.domain_database_size.is_some() {
            ZSetBy::Option
        } else {
            ZSetBy::Ntargets
        };
        Ok(Self {
            hmm,
            alphabet,
            background,
            profile,
            optimized,
            pipeline,
            options,
        })
    }

    /// Alphabet used to digitize every target for this query.
    ///
    /// Digital [`Sequence`] values do not retain their source alphabet. Use
    /// this alphabet with [`Sequence::from_text`] or a sequence-file reader;
    /// numeric residue codes alone cannot reveal an alphabet mismatch.
    pub fn alphabet(&self) -> &Alphabet {
        &self.alphabet
    }

    /// Search borrowed targets, returning sorted hits with final thresholds.
    ///
    /// Starts fresh counters/results and reuses allocated scoring buffers.
    /// Each target is validated as in [`SearchSession::add`]. An error discards
    /// this search's partial results; a later search can reuse the searcher.
    /// Empty input returns empty results with zero automatic database sizes.
    pub fn search<'t>(
        &mut self,
        targets: impl IntoIterator<Item = &'t Sequence>,
    ) -> HmmerResult<SearchResults> {
        let mut session = self.begin_search();
        for target in targets {
            session.add(target)?;
        }
        Ok(session.finish())
    }

    /// Begin an independent streaming database search, reusing scoring buffers.
    ///
    /// Only one session can borrow a searcher at a time. Call
    /// [`SearchSession::finish`] after adding all targets to obtain results
    /// with final E-values. Dropping a session discards its partial results.
    pub fn begin_search(&mut self) -> SearchSession<'_, 'h> {
        self.pipeline.n_targets = 0;
        self.pipeline.n_past_msv = 0;
        self.pipeline.n_past_bias = 0;
        self.pipeline.n_past_vit = 0;
        self.pipeline.n_past_fwd = 0;
        self.pipeline.z = self.options.database_size.unwrap_or(0.0);
        self.pipeline.domz = self.options.domain_database_size.unwrap_or(0.0);
        SearchSession {
            searcher: self,
            hits: TopHits::new(),
            residues: 0,
        }
    }
}

/// An unfinished streaming search; obtain final results with [`Self::finish`].
///
/// Targets are scored immediately and need not remain in memory after `add`.
/// An invalid target leaves the session unchanged, allowing it to be retried.
#[must_use = "add targets and call finish() to obtain final search results"]
pub struct SearchSession<'s, 'h> {
    searcher: &'s mut Searcher<'h>,
    hits: TopHits,
    residues: u64,
}

impl SearchSession<'_, '_> {
    /// Alphabet required by targets added to this session.
    pub fn alphabet(&self) -> &Alphabet {
        self.searcher.alphabet()
    }

    /// Validate and score one complete target, updating its length models.
    ///
    /// Requires `n` residues at `dsq[1..=n]` and boundary sentinels, with no
    /// extra bytes. Residue codes must use [`Self::alphabet`]; gaps, missing
    /// data, and invalid codes are rejected. The nonresidue marker `*` is
    /// accepted, as by [`Sequence::from_text`]. `l` must equal `n`: subsequence
    /// windows and strand-aware genome searches require the low-level APIs.
    /// Lengths must fit `i32`. Empty targets count toward the database size
    /// but produce no hits. Returns an error before scoring invalid input.
    /// Success means the target was processed, whether or not it yielded a hit.
    pub fn add(&mut self, target: &Sequence) -> HmmerResult<()> {
        validate_target(target, self.alphabet())?;
        let residues = self
            .residues
            .checked_add(target.n as u64)
            .ok_or_else(|| HmmerError::Range("total target residue count overflows u64".into()))?;
        let count = self
            .searcher
            .pipeline
            .n_targets
            .checked_add(1)
            .filter(|&n| n <= i64::MAX as u64)
            .ok_or_else(|| HmmerError::Range("target count exceeds i64::MAX".into()))?;
        let s = &mut self.searcher;
        crate::util::simd_env::init();
        s.background.set_length(target.n);
        let before = self.hits.hits.len();
        s.pipeline.run(
            &mut s.profile,
            &mut s.optimized,
            &s.background,
            s.hmm,
            target,
            &mut self.hits,
        );
        // Pipeline::run skips empty targets; the database count includes them.
        s.pipeline.n_targets = count;
        for hit in &mut self.hits.hits[before..] {
            hit.seqidx = count as i64;
        }
        self.residues = residues;
        Ok(())
    }

    /// Finalize database sizes, apply sequence/domain thresholds, and sort hits.
    ///
    /// Automatic sequence search space counts every added target. Automatic
    /// domain search space counts finally reported sequences, matching the
    /// command-line search's two-pass thresholding. Explicit sizes in
    /// [`SearchOptions`] override these counts. Consumes the session and
    /// releases the searcher for another independent search.
    pub fn finish(mut self) -> SearchResults {
        let pipeline = &self.searcher.pipeline;
        let z = self
            .searcher
            .options
            .database_size
            .unwrap_or(pipeline.n_targets as f64);
        self.hits.sort_by_sortkey();
        self.hits.threshold(pipeline, z, z);
        let domz = self
            .searcher
            .options
            .domain_database_size
            .unwrap_or(self.hits.nreported as f64);
        if domz != z {
            self.hits.threshold(pipeline, z, domz);
        }
        SearchResults {
            hits: self.hits,
            database_size: z,
            domain_database_size: domz,
            stats: SearchStats {
                targets: pipeline.n_targets,
                residues: self.residues,
                past_msv: pipeline.n_past_msv,
                past_bias: pipeline.n_past_bias,
                past_viterbi: pipeline.n_past_vit,
                past_forward: pipeline.n_past_fwd,
            },
        }
    }
}

fn validate_target(target: &Sequence, alphabet: &Alphabet) -> HmmerResult<()> {
    if target.n > i32::MAX as usize {
        return Err(HmmerError::Range(
            "target length exceeds i32::MAX residues".into(),
        ));
    }
    if target.l != target.n
        || target.dsq.len() != target.n + 2
        || target.dsq.first() != Some(&DSQ_SENTINEL)
        || target.dsq.last() != Some(&DSQ_SENTINEL)
    {
        return Err(HmmerError::InvalidArg(
            "target must be a complete sequence with matching lengths and boundary sentinels"
                .into(),
        ));
    }
    if target.dsq[1..=target.n]
        .iter()
        .any(|&code| !alphabet.is_residue(code) && code != alphabet.nonresidue_code())
    {
        return Err(HmmerError::InvalidArg(
            "target contains an invalid digital residue code".into(),
        ));
    }
    Ok(())
}

/// Counts for one completed search; filter counts exclude empty targets.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SearchStats {
    /// Number of added targets, including empty sequences.
    pub targets: u64,
    /// Total number of target residues, excluding sentinels.
    pub residues: u64,
    /// Targets passing the MSV filter stage.
    pub past_msv: u64,
    /// Targets passing the bias filter stage.
    pub past_bias: u64,
    /// Targets passing the Viterbi filter stage.
    pub past_viterbi: u64,
    /// Targets passing the Forward filter stage.
    pub past_forward: u64,
}

/// Final ranked hits and the search-space sizes needed to interpret E-values.
///
/// Returned hits use bit scores, natural-log P-values, and one-based inclusive
/// domain coordinates. Stored candidates may include hits rejected by final
/// thresholds; use [`Self::reported_hits`] or [`Self::included_hits`] to filter.
#[derive(Debug)]
pub struct SearchResults {
    hits: TopHits,
    database_size: f64,
    domain_database_size: f64,
    stats: SearchStats,
}

impl SearchResults {
    /// Ranked sequences passing the final reporting thresholds.
    pub fn reported_hits(&self) -> impl Iterator<Item = &Hit> {
        self.hits
            .hits
            .iter()
            .filter(|h| h.flags & P7_IS_REPORTED != 0)
    }

    /// Ranked sequences passing the final inclusion and reporting thresholds.
    pub fn included_hits(&self) -> impl Iterator<Item = &Hit> {
        self.hits
            .hits
            .iter()
            .filter(|h| h.flags & P7_IS_INCLUDED != 0)
    }

    /// Borrow the complete candidate collection, including unreported hits.
    pub fn top_hits(&self) -> &TopHits {
        &self.hits
    }

    /// Consume the results to recover the underlying hit collection.
    /// Retain the search-space sizes separately if computing E-values later.
    pub fn into_top_hits(self) -> TopHits {
        self.hits
    }

    /// Final sequence E-value for a hit from these results (`Z * P`).
    pub fn sequence_evalue(&self, hit: &Hit) -> f64 {
        self.database_size * c_exp_f64(hit.lnp)
    }

    /// Conditional domain E-value (`domZ * P`) for a domain from these results.
    pub fn conditional_domain_evalue(&self, domain: &Domain) -> f64 {
        self.domain_database_size * c_exp_f64(domain.lnp)
    }

    /// Independent domain E-value (`Z * P`) for a domain from these results.
    pub fn independent_domain_evalue(&self, domain: &Domain) -> f64 {
        self.database_size * c_exp_f64(domain.lnp)
    }

    /// Effective sequence count used for final E-values.
    pub fn database_size(&self) -> f64 {
        self.database_size
    }

    /// Effective conditional-domain search space used for final E-values.
    pub fn domain_database_size(&self) -> f64 {
        self.domain_database_size
    }

    /// Target, residue, and filter counts for this search only.
    pub fn stats(&self) -> &SearchStats {
        &self.stats
    }
}
