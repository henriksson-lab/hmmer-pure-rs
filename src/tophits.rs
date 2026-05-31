//! P7_TOPHITS - Ranked hit list for search results.
//! Simplified port focused on what hmmsearch needs.

#![allow(clippy::doc_lazy_continuation)]

use crate::alphabet::Alphabet;
use crate::msa::Msa;
use crate::sequence::Sequence;
use crate::trace::{State, Trace};
use crate::util::cmath::c_exp_f64;

/// Print one alignment display as ASCII blocks, matching C
/// `p7_nontranslated_alidisplay_Print()` (p7_alidisplay.c:1162). Splits the
/// alignment into blocks of `aliwidth = linewidth - namewidth - 2*coordwidth -
/// 5` (bounded below by `min_aliwidth=40`). When `linewidth==0` the whole
/// alignment is printed in one block.
pub fn print_alidisplay_blocks(
    out: &mut dyn std::io::Write,
    hmm_name: &str,
    seq_name: &str,
    ad: &AliDisplay,
    cs_line: Option<&str>,
    linewidth: usize,
) {
    // Back-compat wrapper: never prefers accessions (no --acc handling).
    print_alidisplay_blocks_acc(
        out, hmm_name, "", seq_name, "", ad, cs_line, linewidth, false,
    );
}

/// Like [`print_alidisplay_blocks`] but honors the `--acc` option
/// (C `p7_nontranslated_alidisplay_Print`'s `show_accessions` argument,
/// p7_alidisplay.c:1162,1176-1180): when `show_accessions` is set and the
/// corresponding accession is non-empty, the accession is shown in place of
/// the name and `namewidth` is derived from the chosen strings.
#[allow(clippy::too_many_arguments)]
pub fn print_alidisplay_blocks_acc(
    out: &mut dyn std::io::Write,
    hmm_name: &str,
    hmm_acc: &str,
    seq_name: &str,
    seq_acc: &str,
    ad: &AliDisplay,
    cs_line: Option<&str>,
    linewidth: usize,
    show_accessions: bool,
) {
    let n = ad.model.chars().count();
    if n == 0 {
        return;
    }
    // Mirror C p7_alidisplay.c:1176-1180: with --acc, prefer the accession
    // when one is present; otherwise fall back to the name. namewidth is then
    // derived from whichever string is actually shown.
    let hmm_name = if show_accessions && !hmm_acc.is_empty() {
        hmm_acc
    } else {
        hmm_name
    };
    let seq_name = if show_accessions && !seq_acc.is_empty() {
        seq_acc
    } else {
        seq_name
    };
    let namewidth = hmm_name.len().max(seq_name.len());
    let coordwidth = [ad.hmmfrom, ad.hmmto, ad.sqfrom, ad.sqto]
        .iter()
        .map(|v| v.to_string().len())
        .max()
        .unwrap_or(1);
    let mut aliwidth = if linewidth > 0 {
        linewidth.saturating_sub(namewidth + 2 * coordwidth + 5)
    } else {
        n
    };
    if aliwidth < n && aliwidth < 40 {
        aliwidth = 40;
    }
    if aliwidth == 0 {
        aliwidth = n;
    }

    let model_chars: Vec<char> = ad.model.chars().collect();
    let mline_chars: Vec<char> = ad.mline.chars().collect();
    let aseq_chars: Vec<char> = ad.aseq.chars().collect();
    let pp_chars: Vec<char> = ad.ppline.chars().collect();
    let rf_chars: Vec<char> = ad.rfline.chars().collect();
    let cs_chars: Option<Vec<char>> = cs_line.map(|s| s.chars().collect());

    let reverse = ad.sqfrom > ad.sqto;
    let mut k1 = ad.hmmfrom as i64;
    let mut i1 = ad.sqfrom as i64;

    let indent_w = namewidth + coordwidth + 1;
    let mut first_block = true;
    let mut pos = 0;
    while pos < n {
        if !first_block {
            writeln!(out).unwrap();
        }
        first_block = false;
        let end = (pos + aliwidth).min(n);
        let mut nk = 0i64;
        let mut ni = 0i64;
        for z in pos..end {
            if model_chars[z] != '.' {
                nk += 1;
            }
            if aseq_chars[z] != '-' {
                ni += 1;
            }
        }
        let k2 = k1 + nk - 1;
        let i2 = if reverse { i1 - ni + 1 } else { i1 + ni - 1 };

        if let Some(ref cs) = cs_chars {
            let chunk: String = cs[pos..end.min(cs.len())].iter().collect();
            writeln!(out, "  {:>indent_w$} {} CS", "", chunk, indent_w = indent_w).unwrap();
        }
        if !rf_chars.is_empty() {
            let chunk: String = rf_chars[pos..end.min(rf_chars.len())].iter().collect();
            writeln!(out, "  {:>indent_w$} {} RF", "", chunk, indent_w = indent_w).unwrap();
        }
        {
            let chunk: String = model_chars[pos..end].iter().collect();
            writeln!(
                out,
                "  {:>nw$} {:>cw$} {} {:<cw$}",
                hmm_name,
                k1,
                chunk,
                k2,
                nw = namewidth,
                cw = coordwidth
            )
            .unwrap();
        }
        {
            let chunk: String = mline_chars[pos..end.min(mline_chars.len())]
                .iter()
                .collect();
            writeln!(out, "  {:>indent_w$} {}", "", chunk, indent_w = indent_w).unwrap();
        }
        if ni > 0 {
            let chunk: String = aseq_chars[pos..end].iter().collect();
            writeln!(
                out,
                "  {:>nw$} {:>cw$} {} {:<cw$}",
                seq_name,
                i1,
                chunk,
                i2,
                nw = namewidth,
                cw = coordwidth
            )
            .unwrap();
        } else {
            let chunk: String = aseq_chars[pos..end].iter().collect();
            writeln!(
                out,
                "  {:>nw$} {:>cw$} {} {:>cw$}",
                seq_name,
                "-",
                chunk,
                "-",
                nw = namewidth,
                cw = coordwidth
            )
            .unwrap();
        }
        if !pp_chars.is_empty() {
            let chunk: String = pp_chars[pos..end.min(pp_chars.len())].iter().collect();
            writeln!(out, "  {:>indent_w$} {} PP", "", chunk, indent_w = indent_w).unwrap();
        }

        k1 += nk;
        if reverse {
            i1 -= ni;
        } else {
            i1 += ni;
        }
        pos = end;
    }
}

/// Alignment display data for one domain.
#[derive(Debug, Clone, Default)]
pub struct AliDisplay {
    pub model: String,  // consensus model line
    pub mline: String,  // match/identity line
    pub aseq: String,   // aligned target sequence
    pub ppline: String, // posterior probability annotation
    pub rfline: String, // RF annotation (empty if HMM has no RF)
    pub hmmfrom: usize,
    pub hmmto: usize,
    pub sqfrom: usize,
    pub sqto: usize,
}

/// A single domain within a hit.
#[derive(Debug, Clone)]
pub struct Domain {
    pub iali: i64, // alignment start in seq (1-based)
    pub jali: i64, // alignment end in seq
    pub ienv: i64, // envelope start
    pub jenv: i64, // envelope end
    pub bitscore: f32,
    pub lnp: f64,     // log P-value
    pub dombias: f32, // bias correction
    pub oasc: f32,    // optimal accuracy score
    pub envsc: f32,   // envelope score
    pub domcorrection: f32,
    pub is_reported: bool,
    pub is_included: bool,
    pub ad: Option<AliDisplay>, // alignment display data
}

/// A sequence-level hit.
#[derive(Debug, Clone)]
pub struct Hit {
    pub name: String,
    pub acc: String,
    pub desc: String,
    pub n: usize,       // target sequence length
    pub sortkey: f64,   // primary sort key (lnP, or negative score for score-threshold ranking)
    pub score: f32,     // overall bit score
    pub bias: f32,      // bias correction in bits
    pub pre_score: f32, // pre-bias-correction score
    pub sum_score: f32, // sum score
    pub lnp: f64,       // log P-value
    pub pre_lnp: f64,
    pub sum_lnp: f64,
    pub nexpected: f32, // expected number of domains
    pub nregions: usize,
    pub nclustered: usize,
    pub noverlaps: usize,
    pub nenvelopes: usize,
    pub ndom: usize, // actual number of domains
    pub nreported: usize,
    pub nincluded: usize,
    pub dcl: Vec<Domain>,
    pub flags: u32,
    pub seqidx: i64,
    pub subseq_start: i64,
}

pub const P7_IS_REPORTED: u32 = 1 << 0;
pub const P7_IS_INCLUDED: u32 = 1 << 1;
pub const P7_IS_NEW: u32 = 1 << 2;
pub const P7_IS_DROPPED: u32 = 1 << 3;
pub const P7_IS_DUPLICATE: u32 = 1 << 4;

/// Collection of ranked search results.
#[derive(Debug)]
pub struct TopHits {
    pub hits: Vec<Hit>,
    pub nreported: usize,
    pub nincluded: usize,
    pub is_sorted: bool,
}

impl TopHits {
    /// Allocate a new (empty) ranked hit list.
    /// Port of `p7_tophits_Create()`. Hits are added with `create_next_hit()`.
    pub fn new() -> Self {
        TopHits {
            hits: Vec::new(),
            nreported: 0,
            nincluded: 0,
            is_sorted: false,
        }
    }

    /// Append a fresh zero-initialized `Hit` and return a mutable reference for
    /// the caller to fill in. Port of `p7_tophits_CreateNextHit()`.
    pub fn create_next_hit(&mut self) -> &mut Hit {
        self.hits.push(Hit {
            name: String::new(),
            acc: String::new(),
            desc: String::new(),
            n: 0,
            sortkey: 0.0,
            score: 0.0,
            bias: 0.0,
            pre_score: 0.0,
            sum_score: 0.0,
            lnp: 0.0,
            pre_lnp: 0.0,
            sum_lnp: 0.0,
            nexpected: 0.0,
            nregions: 0,
            nclustered: 0,
            noverlaps: 0,
            nenvelopes: 0,
            ndom: 0,
            nreported: 0,
            nincluded: 0,
            dcl: Vec::new(),
            flags: P7_IS_NEW,
            seqidx: -1,
            subseq_start: 0,
        });
        self.is_sorted = false;
        self.hits.last_mut().unwrap()
    }

    /// Remove duplicate hits produced by overlapping long-target windows
    /// finding the same alignment. Port of `p7_tophits_RemoveDuplicates`
    /// (hmmer/src/p7_tophits.c:823).
    ///
    /// Scans consecutive hits (caller must have sorted the list by
    /// sequence/position). Two hits are duplicates when they target the
    /// same model name and same source sequence on the same strand AND
    /// overlap on the HMM AND at least one of:
    ///   - ali start within 3 positions
    ///   - ali end within 3 positions
    ///   - overlap covers ≥95% of the shorter alignment
    ///   The worse-E hit is marked `P7_IS_DUPLICATE` (and stripped of the
    ///   REPORTED/INCLUDED flags).
    pub fn remove_duplicates(&mut self) {
        if self.hits.len() < 2 {
            return;
        }
        let n = self.hits.len();
        let mut prev = 0usize;
        for i in 1..n {
            // Extract comparison fields without holding a borrow.
            let (p_j, s_j_raw, e_j_raw, hmm_from_j, hmm_to_j) = {
                let h = &self.hits[prev];
                let dom = h.dcl.first();
                let (sj, ej) = dom.map(|d| (d.iali, d.jali)).unwrap_or((0, 0));
                let (hf, ht) = dom
                    .and_then(|d| d.ad.as_ref().map(|a| (a.hmmfrom as i64, a.hmmto as i64)))
                    .unwrap_or((0, 0));
                (h.lnp, sj, ej, hf, ht)
            };
            let (p_i, s_i_raw, e_i_raw, hmm_from_i, hmm_to_i) = {
                let h = &self.hits[i];
                let dom = h.dcl.first();
                let (si, ei) = dom.map(|d| (d.iali, d.jali)).unwrap_or((0, 0));
                let (hf, ht) = dom
                    .and_then(|d| d.ad.as_ref().map(|a| (a.hmmfrom as i64, a.hmmto as i64)))
                    .unwrap_or((0, 0));
                (h.lnp, si, ei, hf, ht)
            };
            // C (p7_tophits.c:861-862) compares the name/seqidx of hit i against
            // the *immediately adjacent* prior hit (i-1), even if i-1 was itself
            // flagged duplicate — distinct from `j` (the last *kept* hit) used for
            // the coordinate/p-value fields above.
            let name_i = self.hits[i].name.clone();
            let seqidx_i = self.hits[i].seqidx;
            let name_j = self.hits[i - 1].name.clone();
            let seqidx_j = self.hits[i - 1].seqidx;
            let dir_j = if s_j_raw < e_j_raw { 1 } else { -1 };
            let dir_i = if s_i_raw < e_i_raw { 1 } else { -1 };
            let (s_j, e_j) = if dir_j == -1 {
                (e_j_raw, s_j_raw)
            } else {
                (s_j_raw, e_j_raw)
            };
            let (s_i, e_i) = if dir_i == -1 {
                (e_i_raw, s_i_raw)
            } else {
                (s_i_raw, e_i_raw)
            };
            let len_j = e_j - s_j + 1;
            let len_i = e_i - s_i + 1;
            let intersect_alistart = s_i.max(s_j);
            let intersect_aliend = e_i.min(e_j);
            let intersect_alilen = intersect_aliend - intersect_alistart + 1;
            let intersect_hmmstart = hmm_from_i.max(hmm_from_j);
            let intersect_hmmend = hmm_to_i.min(hmm_to_j);
            let intersect_hmmlen = intersect_hmmend - intersect_hmmstart + 1;

            let flush_start = (s_i - s_j).abs() <= 3;
            let flush_end = (e_i - e_j).abs() <= 3;
            let mostly_i = len_i > 0 && (intersect_alilen as f64) >= (len_i as f64 * 0.95);
            let mostly_j = len_j > 0 && (intersect_alilen as f64) >= (len_j as f64 * 0.95);

            let is_dup = name_i == name_j
                && seqidx_i == seqidx_j
                && dir_i == dir_j
                && intersect_hmmlen > 0
                && (flush_start || flush_end || mostly_i || mostly_j);

            if is_dup {
                // Keep the one with lower (better) lnp.
                let remove = if p_i < p_j { prev } else { i };
                self.hits[remove].flags |= P7_IS_DUPLICATE;
                self.hits[remove].flags &= !P7_IS_REPORTED;
                self.hits[remove].flags &= !P7_IS_INCLUDED;
                // Prev becomes whichever one we kept.
                if remove == prev {
                    prev = i;
                }
            } else {
                prev = i;
            }
        }
        // Recount reported/included.
        self.nreported = self
            .hits
            .iter()
            .filter(|h| h.flags & P7_IS_REPORTED != 0)
            .count();
        self.nincluded = self
            .hits
            .iter()
            .filter(|h| h.flags & P7_IS_INCLUDED != 0)
            .count();
    }

    /// Sort hits by (seqidx, alignment start position) for duplicate
    /// detection. Mirrors C `p7_tophits_SortBySeqidxAndAlipos` which is
    /// invoked immediately before `p7_tophits_RemoveDuplicates` in
    /// `nhmmer.c`. After sorting here, duplicate hits at the same
    /// position are adjacent.
    pub fn sort_by_seqidx_and_alipos(&mut self) {
        self.hits.sort_by(|a, b| {
            a.seqidx
                .cmp(&b.seqidx)
                .then_with(|| compare_hit_alipos(a, b))
        });
    }

    /// Sort hits by (model name, alignment position) for duplicate detection in
    /// nhmmscan. Mirrors C `p7_tophits_SortByModelnameAndAlipos`.
    pub fn sort_by_modelname_and_alipos(&mut self) {
        self.hits
            .sort_by(|a, b| a.name.cmp(&b.name).then_with(|| compare_hit_alipos(a, b)));
    }

    /// Sort hits by sort key (E-value / score) with C HMMER's tiebreakers
    /// (p7_tophits.c:hit_sorter_by_sortkey):
    ///   1. sortkey ascending (lnP ascending = most-significant first).
    ///   2. name.
    ///   3. strand (positive first).
    ///   4. `dcl[0].iali` ascending.
    ///
    /// Stability note (audit 09 finding, intentional): Rust `sort_by` is a
    /// stable sort, whereas C uses `qsort`, whose ordering on a *full* tie
    /// (every comparator key equal) is unspecified. C's tie order therefore
    /// cannot be reproduced faithfully; a stable sort that preserves input
    /// order is the deterministic, safe choice. This only affects output line
    /// order among exact ties, which is effectively unobservable. The same
    /// reasoning applies to `sort_by_seqidx_and_alipos` and
    /// `sort_by_modelname_and_alipos`.
    pub fn sort_by_sortkey(&mut self) {
        self.hits.sort_by(|a, b| {
            a.sortkey
                .partial_cmp(&b.sortkey)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.name.cmp(&b.name))
                .then_with(|| {
                    let a_dom = a.dcl.first();
                    let b_dom = b.dcl.first();
                    let a_dir = a_dom
                        .map(|d| if d.iali < d.jali { 1 } else { -1 })
                        .unwrap_or(1);
                    let b_dir = b_dom
                        .map(|d| if d.iali < d.jali { 1 } else { -1 })
                        .unwrap_or(1);
                    if a_dir != b_dir {
                        // + strand (dir=1) goes before - strand (dir=-1).
                        b_dir.cmp(&a_dir)
                    } else {
                        let a_i = a_dom.map(|d| d.iali).unwrap_or(0);
                        let b_i = b_dom.map(|d| d.iali).unwrap_or(0);
                        a_i.cmp(&b_i)
                    }
                })
        });
        self.is_sorted = true;
    }

    /// Apply reporting and inclusion thresholds.
    /// Uses E-value thresholds: inc_e for inclusion, report_e for reporting.
    pub fn threshold(&mut self, pli: &super::pipeline::Pipeline, z: f64, domz: f64) {
        self.nreported = 0;
        self.nincluded = 0;

        if pli.use_bit_cutoffs != super::pipeline::BitCutoff::None {
            for hit in &mut self.hits {
                hit.sortkey = pli.hit_sortkey(hit.score, hit.lnp);
                hit.nreported = hit.dcl.iter().filter(|dom| dom.is_reported).count();
                hit.nincluded = hit.dcl.iter().filter(|dom| dom.is_included).count();
                if hit.flags & P7_IS_REPORTED != 0 {
                    self.nreported += 1;
                }
                if hit.flags & P7_IS_INCLUDED != 0 {
                    self.nincluded += 1;
                }
            }
            self.workaround_bug_h74();
            if pli.score_sort_active() {
                self.sort_by_sortkey();
            }
            return;
        }

        for hit in &mut self.hits {
            hit.flags &= !(P7_IS_REPORTED | P7_IS_INCLUDED | P7_IS_DROPPED);
            hit.nreported = 0;
            hit.nincluded = 0;
            hit.sortkey = pli.hit_sortkey(hit.score, hit.lnp);
            for dom in &mut hit.dcl {
                dom.is_reported = false;
                dom.is_included = false;
            }

            // Skip hits already marked duplicate by remove_duplicates.
            if hit.flags & P7_IS_DUPLICATE != 0 {
                continue;
            }

            let evalue = z * c_exp_f64(hit.lnp);

            let reported = if pli.by_e {
                evalue <= pli.e_value_threshold
            } else if let Some(t) = pli.t {
                hit.score as f64 >= t
            } else {
                evalue <= pli.e_value_threshold
            };

            if reported {
                hit.flags |= P7_IS_REPORTED;
                self.nreported += 1;
            } else {
                hit.flags |= P7_IS_DROPPED;
            }

            let included = if pli.inc_by_e {
                evalue <= pli.inc_e
            } else if let Some(t) = pli.inc_t {
                hit.score as f64 >= t
            } else {
                evalue <= pli.inc_e
            };
            if reported && included {
                hit.flags |= P7_IS_INCLUDED;
                self.nincluded += 1;
            }
        }

        for hit in &mut self.hits {
            if hit.flags & P7_IS_REPORTED == 0 {
                continue;
            }

            if pli.long_target {
                if let Some(dom) = hit.dcl.first_mut() {
                    dom.is_reported = true;
                    hit.nreported = 1;
                    if hit.flags & P7_IS_INCLUDED != 0 {
                        dom.is_included = true;
                        hit.nincluded = 1;
                    }
                }
                continue;
            }

            let target_included = hit.flags & P7_IS_INCLUDED != 0;
            for dom in &mut hit.dcl {
                let dom_evalue = domz * c_exp_f64(dom.lnp);

                let dom_reported = if pli.dom_by_e {
                    dom_evalue <= pli.dom_e_value_threshold
                } else if let Some(t) = pli.dom_t {
                    dom.bitscore as f64 >= t
                } else {
                    dom_evalue <= pli.dom_e_value_threshold
                };

                if dom_reported {
                    dom.is_reported = true;
                    hit.nreported += 1;
                }

                let dom_included = target_included
                    && if pli.incdom_by_e {
                        dom_evalue <= pli.inc_dome
                    } else if let Some(t) = pli.inc_dom_t {
                        dom.bitscore as f64 >= t
                    } else {
                        dom_evalue <= pli.inc_dome
                    };
                if dom_included {
                    dom.is_included = true;
                    hit.nincluded += 1;
                }
            }
        }

        self.workaround_bug_h74();

        if pli.score_sort_active() {
            self.sort_by_sortkey();
        }
    }

    /// Workaround for HMMER bug #h74: two distinct envelopes can produce
    /// identical optimal-accuracy alignments (same `iali`/`jali`) because H3's
    /// Forward/Backward cannot be limited to a profile-coordinate range. When
    /// this happens we un-report / un-include the lower-scoring duplicate domain
    /// so only one copy is shown. Faithful port of `workaround_bug_h74()` in
    /// `hmmer/src/p7_tophits.c:756`, called at the end of `p7_tophits_Threshold`.
    ///
    /// Only hits flagged with overlapping envelopes (`noverlaps != 0`) are
    /// examined; this adjusts per-domain `is_reported`/`is_included` flags and
    /// the per-hit `nreported`/`nincluded` domain counts (not the target-level
    /// counts), matching C exactly.
    fn workaround_bug_h74(&mut self) {
        for hit in &mut self.hits {
            if hit.noverlaps == 0 {
                continue;
            }
            let ndom = hit.dcl.len();
            for d1 in 0..ndom {
                for d2 in (d1 + 1)..ndom {
                    if hit.dcl[d1].iali == hit.dcl[d2].iali && hit.dcl[d1].jali == hit.dcl[d2].jali
                    {
                        let dremoved = if hit.dcl[d1].bitscore >= hit.dcl[d2].bitscore {
                            d2
                        } else {
                            d1
                        };
                        if hit.dcl[dremoved].is_reported {
                            hit.dcl[dremoved].is_reported = false;
                            hit.nreported = hit.nreported.saturating_sub(1);
                        }
                        if hit.dcl[dremoved].is_included {
                            hit.dcl[dremoved].is_included = false;
                            hit.nincluded = hit.nincluded.saturating_sub(1);
                        }
                    }
                }
            }
        }
    }
}

impl Default for TopHits {
    fn default() -> Self {
        Self::new()
    }
}

fn compare_hit_alipos(a: &Hit, b: &Hit) -> std::cmp::Ordering {
    let (a_start, a_end, a_dir) = normalized_domain_span(a.dcl.first());
    let (b_start, b_end, b_dir) = normalized_domain_span(b.dcl.first());
    if a_dir != b_dir {
        return b_dir.cmp(&a_dir);
    }
    a_start.cmp(&b_start).then_with(|| b_end.cmp(&a_end))
}

fn normalized_domain_span(dom: Option<&Domain>) -> (i64, i64, i32) {
    let (mut start, mut end) = dom.map(|d| (d.iali, d.jali)).unwrap_or((0, 0));
    let dir = if start < end { 1 } else { -1 };
    if dir == -1 {
        std::mem::swap(&mut start, &mut end);
    }
    (start, end, dir)
}

/// Build a multiple sequence alignment from all included hits/domains in `th`.
/// Each included domain's `AliDisplay` is back-converted to a (sequence, trace)
/// pair, then a faux MSA is assembled mapping each match column of the model.
/// Port of `p7_tophits_Alignment()` (hmmer/src/p7_tophits.c:1478) with the
/// supporting machinery from Easel's `esl_msa_*` helpers. `extra` lets callers
/// (e.g. `jackhmmer`) prepend the query sequence/trace into the alignment.
/// Returns `None` if no included domains and no `extra` were supplied.
pub fn included_alignment(
    th: &TopHits,
    abc: &Alphabet,
    model_len: usize,
    extra: Option<(&Sequence, &Trace)>,
    msa_name: &str,
) -> Option<Msa> {
    let ndom = th
        .hits
        .iter()
        .filter(|hit| hit.flags & P7_IS_INCLUDED != 0)
        .map(|hit| hit.dcl.iter().filter(|dom| dom.is_included).count())
        .sum::<usize>();
    let mut sequences = Vec::new();
    let mut traces = Vec::new();

    if let Some((sq, tr)) = extra {
        sequences.push(sq.clone());
        traces.push(tr.clone());
    }

    if ndom == 0 {
        if sequences.is_empty() {
            return None;
        }
        let (inscount, matuse, matmap, alen) = map_new_msa(model_len, &traces);
        let mut aseq = Vec::with_capacity(sequences.len());
        let mut pp = Vec::with_capacity(sequences.len());
        let mut sqname = Vec::with_capacity(sequences.len());
        let mut pp_totals = vec![0.0_f64; alen];
        let mut pp_counts = vec![0usize; alen];
        for (sq, tr) in sequences.iter().zip(traces.iter()) {
            let mut row = make_text_row(abc, sq, tr, &matuse, &matmap, alen);
            let mut pp_row =
                make_pp_row(tr, &matmap, model_len, alen, &mut pp_totals, &mut pp_counts);
            rejustify_insertions_text(
                &mut row,
                pp_row.as_deref_mut(),
                &inscount,
                &matmap,
                &matuse,
                model_len,
            );
            aseq.push(row);
            pp.push(pp_row);
            sqname.push(sq.name.clone());
        }

        let mut rf = vec![b'.'; alen];
        for k in 1..=model_len {
            if matuse[k] {
                rf[matmap[k] - 1] = b'x';
            }
        }

        return Some(Msa {
            name: msa_name.to_string(),
            acc: None,
            desc: None,
            author: None,
            sqname,
            sqacc: sequences.iter().map(|sq| sq.acc.clone()).collect(),
            sqdesc: sequences.iter().map(|sq| sq.desc.clone()).collect(),
            weights: None,
            pp,
            aseq,
            nseq: sequences.len(),
            alen,
            rf: Some(rf),
            mm: None,
            ss_cons: None,
            sa_cons: None,
            pp_cons: pp_consensus(&pp_totals, &pp_counts),
        });
    }

    for hit in &th.hits {
        if hit.flags & P7_IS_INCLUDED == 0 {
            continue;
        }
        for dom in &hit.dcl {
            if !dom.is_included {
                continue;
            }
            let ad = dom.ad.as_ref()?;
            let (sq, tr) = alidisplay_backconvert(hit, dom, ad, abc);
            sequences.push(sq);
            traces.push(tr);
        }
    }

    if sequences.is_empty() {
        return None;
    }

    let (inscount, matuse, matmap, alen) = map_new_msa(model_len, &traces);
    let mut aseq = Vec::with_capacity(sequences.len());
    let mut pp = Vec::with_capacity(sequences.len());
    let mut sqname = Vec::with_capacity(sequences.len());
    let mut pp_totals = vec![0.0_f64; alen];
    let mut pp_counts = vec![0usize; alen];
    for (sq, tr) in sequences.iter().zip(traces.iter()) {
        let mut row = make_text_row(abc, sq, tr, &matuse, &matmap, alen);
        let mut pp_row = make_pp_row(tr, &matmap, model_len, alen, &mut pp_totals, &mut pp_counts);
        rejustify_insertions_text(
            &mut row,
            pp_row.as_deref_mut(),
            &inscount,
            &matmap,
            &matuse,
            model_len,
        );
        aseq.push(row);
        pp.push(pp_row);
        sqname.push(sq.name.clone());
    }

    let mut rf = vec![b'.'; alen];
    for k in 1..=model_len {
        if matuse[k] {
            rf[matmap[k] - 1] = b'x';
        }
    }

    Some(Msa {
        name: msa_name.to_string(),
        acc: None,
        desc: None,
        author: None,
        sqname,
        sqacc: sequences.iter().map(|sq| sq.acc.clone()).collect(),
        sqdesc: sequences.iter().map(|sq| sq.desc.clone()).collect(),
        weights: None,
        pp,
        aseq,
        nseq: sequences.len(),
        alen,
        rf: Some(rf),
        mm: None,
        ss_cons: None,
        sa_cons: None,
        pp_cons: pp_consensus(&pp_totals, &pp_counts),
    })
}

/// Reconstruct a `(Sequence, Trace)` pair from a printable `AliDisplay`.
/// Walks the alignment columns, dropping gap columns and emitting M/I/D states
/// with sequence indices. Port of `p7_alidisplay_Backconvert()`
/// (hmmer/src/p7_alidisplay.c:1233).
fn alidisplay_backconvert(
    hit: &Hit,
    dom: &Domain,
    ad: &AliDisplay,
    abc: &Alphabet,
) -> (Sequence, Trace) {
    let sub_l = ad.aseq.bytes().filter(|&c| c != b'.' && c != b'-').count();
    let sqfrom = dom.iali.min(dom.jali);
    let sqto = dom.iali.max(dom.jali);

    let mut sq = Sequence::new();
    sq.name = format!("{}/{}-{}", hit.name, sqfrom, sqto);
    sq.desc = format!(
        "[subseq from] {}",
        if hit.desc.is_empty() {
            hit.name.as_str()
        } else {
            hit.desc.as_str()
        }
    );
    sq.acc = hit.acc.clone();
    sq.n = sub_l;
    sq.l = hit.n;
    sq.dsq = Vec::with_capacity(sub_l + 2);
    sq.dsq.push(crate::alphabet::DSQ_SENTINEL);

    let mut tr = Trace::new();
    tr.append(State::S, 0, 0);
    tr.append(State::N, 0, 0);
    tr.append(State::B, 0, 0);

    let model = ad.model.as_bytes();
    let aseq = ad.aseq.as_bytes();
    let mut k = ad.hmmfrom.saturating_sub(1);
    let mut i = 1usize;
    for a in 0..ad.model.len() {
        let model_gap = model[a] == b'.' || model[a] == b'-';
        let aseq_gap = aseq[a] == b'.' || aseq[a] == b'-';
        let state = if !model_gap {
            k += 1;
            if !aseq_gap {
                State::M
            } else {
                State::D
            }
        } else {
            State::I
        };
        let pp = ad
            .ppline
            .as_bytes()
            .get(a)
            .copied()
            .map(decode_postprob)
            .unwrap_or(0.0);
        tr.append_with_pp(state, k, i, pp);
        match state {
            State::M | State::I => {
                sq.dsq
                    .push(abc.digitize_symbol(aseq[a].to_ascii_uppercase()));
                i += 1;
            }
            State::D => {}
            _ => unreachable!(),
        }
    }
    tr.append_with_pp(State::E, 0, 0, 0.0);
    tr.append_with_pp(State::C, 0, 0, 0.0);
    tr.append_with_pp(State::T, 0, 0, 0.0);
    sq.dsq.push(crate::alphabet::DSQ_SENTINEL);

    (sq, tr)
}

/// Encode a posterior probability as a single ASCII character ('0'..'9' or '*').
/// Port of `p7_alidisplay_EncodePostProb()`.
fn encode_postprob(p: f32) -> u8 {
    let shifted = p as f64 + 0.05_f64;
    if shifted >= 1.0 {
        b'*'
    } else {
        (shifted * 10.0) as u8 + b'0'
    }
}

/// Decode a posterior-probability character back to a float in [0, 1].
/// Inverse of `encode_postprob`; matches `p7_alidisplay_DecodePostProb()`.
fn decode_postprob(pc: u8) -> f32 {
    match pc {
        b'0' => 0.01,
        b'*' => 1.0,
        b'.' => 0.0,
        b'1'..=b'9' => (pc - b'0') as f32 / 10.0,
        _ => 0.0,
    }
}

/// Compute the per-column consensus PP line by averaging recorded values and
/// encoding the mean. Returns `None` if no column had any PP-bearing row.
fn pp_consensus(pp_totals: &[f64], pp_counts: &[usize]) -> Option<Vec<u8>> {
    if pp_counts.iter().all(|&count| count == 0) {
        return None;
    }
    Some(
        pp_totals
            .iter()
            .zip(pp_counts.iter())
            .map(|(&total, &count)| {
                if count > 0 {
                    encode_postprob((total / count as f64) as f32)
                } else {
                    b'.'
                }
            })
            .collect(),
    )
}

/// Plan an MSA layout from a set of traces against a model of length `m`.
/// Returns `(inscount[k], matuse[k], matmap[k], alen)`: per-column max insert
/// length, which match columns are used by at least one trace, the mapping
/// from model node 1..M to its alignment column (1-based), and the total
/// alignment width. Mirrors C `map_new_msa()` in `p7_tophits.c`.
fn map_new_msa(m: usize, traces: &[Trace]) -> (Vec<usize>, Vec<bool>, Vec<usize>, usize) {
    let mut inscount = vec![0usize; m + 1];
    let mut matuse = vec![true; m + 1];
    matuse[0] = false;
    let mut insnum = vec![0usize; m + 1];

    for tr in traces {
        insnum.fill(0);
        for z in 1..tr.n {
            match tr.st[z] {
                State::I => insnum[tr.k[z]] += 1,
                State::N if tr.st[z - 1] == State::N => insnum[0] += 1,
                State::C if tr.st[z - 1] == State::C => insnum[m] += 1,
                State::M => matuse[tr.k[z]] = true,
                State::J => panic!("J state unsupported in TopHits alignment"),
                _ => {}
            }
        }
        for k in 0..=m {
            inscount[k] = inscount[k].max(insnum[k]);
        }
    }

    let mut matmap = vec![0usize; m + 1];
    let mut alen = inscount[0];
    for k in 1..=m {
        if matuse[k] {
            matmap[k] = alen + 1;
            alen += 1 + inscount[k];
        } else {
            matmap[k] = alen;
            alen += inscount[k];
        }
    }

    (inscount, matuse, matmap, alen)
}

/// Render one aligned sequence row for `sq`/`tr` into the MSA layout produced
/// by `map_new_msa`. Match states use uppercase, inserts use lowercase, and
/// unused match columns get '-' or '.'. Mirrors `make_text_row()` in C.
fn make_text_row(
    abc: &Alphabet,
    sq: &Sequence,
    tr: &Trace,
    matuse: &[bool],
    matmap: &[usize],
    alen: usize,
) -> Vec<u8> {
    let mut aseq = vec![b'.'; alen];
    for k in 1..matuse.len() {
        if matuse[k] {
            aseq[matmap[k] - 1] = b'-';
        }
    }

    let mut apos = 0usize;
    for z in 0..tr.n {
        match tr.st[z] {
            State::M => {
                let idx = matmap[tr.k[z]] - 1;
                aseq[idx] = (abc.sym[sq.dsq[tr.i[z]] as usize] as char).to_ascii_uppercase() as u8;
                apos = matmap[tr.k[z]];
            }
            State::D => {
                if matuse[tr.k[z]] {
                    aseq[matmap[tr.k[z]] - 1] = b'-';
                }
                apos = matmap[tr.k[z]];
            }
            State::I => {
                if apos < alen {
                    aseq[apos] =
                        (abc.sym[sq.dsq[tr.i[z]] as usize] as char).to_ascii_lowercase() as u8;
                    apos += 1;
                }
            }
            State::N | State::C => {
                if tr.i[z] > 0 && apos < alen {
                    aseq[apos] =
                        (abc.sym[sq.dsq[tr.i[z]] as usize] as char).to_ascii_lowercase() as u8;
                    apos += 1;
                }
            }
            State::E => {
                apos = matmap[matmap.len() - 1];
            }
            _ => {}
        }
    }

    aseq
}

/// Render one PP annotation row aligned to the MSA columns and accumulate the
/// per-column running totals/counts used to build the consensus PP line. Returns
/// `None` if the trace lacks PP annotations.
fn make_pp_row(
    tr: &Trace,
    matmap: &[usize],
    model_len: usize,
    alen: usize,
    pp_totals: &mut [f64],
    pp_counts: &mut [usize],
) -> Option<Vec<u8>> {
    let pp_values = tr.pp.as_ref()?;
    let mut pp = vec![b'.'; alen];
    let mut apos = 0usize;
    for z in 0..tr.n {
        match tr.st[z] {
            State::M => {
                let idx = matmap[tr.k[z]] - 1;
                let value = pp_values[z];
                pp[idx] = encode_postprob(value);
                pp_totals[idx] += value as f64;
                pp_counts[idx] += 1;
                apos = matmap[tr.k[z]];
            }
            State::D => {
                apos = matmap[tr.k[z]];
            }
            State::I => {
                if tr.k[z] != 0 && tr.k[z] != model_len && apos < alen {
                    let value = pp_values[z];
                    pp[apos] = encode_postprob(value);
                    apos += 1;
                }
            }
            State::N | State::C => {
                if tr.i[z] > 0 && apos < alen {
                    let value = pp_values[z];
                    pp[apos] = encode_postprob(value);
                    apos += 1;
                }
            }
            State::E => {
                apos = matmap[model_len];
            }
            _ => {}
        }
    }
    Some(pp)
}

/// Right-justify insert columns so that aligned residues sit flush against the
/// next match column on either side, matching Easel's
/// `rejustify_insertions_text()` used by `esl_msa_FromAliDisplay`-style outputs.
fn rejustify_insertions_text(
    aseq: &mut [u8],
    mut pp: Option<&mut [u8]>,
    inserts: &[usize],
    matmap: &[usize],
    matuse: &[bool],
    m: usize,
) {
    fn is_text_gap(c: u8) -> bool {
        matches!(c, b'.' | b'-' | b'~')
    }

    for k in 0..m {
        if inserts[k] <= 1 {
            continue;
        }

        let start = matmap[k];
        let end = matmap[k + 1] - usize::from(matuse[k + 1]);
        let mut nins = (start..end)
            .filter(|&apos| aseq[apos].is_ascii_alphabetic())
            .count();
        if k == 0 {
            nins = 0;
        } else {
            nins /= 2;
        }

        let floor = (start + nins) as isize;
        let mut opos = end as isize - 1;
        let mut npos = end as isize - 1;
        while opos >= floor {
            if is_text_gap(aseq[opos as usize]) {
                opos -= 1;
                continue;
            }
            aseq[npos as usize] = aseq[opos as usize];
            if let Some(pp) = &mut pp {
                pp[npos as usize] = pp[opos as usize];
            }
            opos -= 1;
            npos -= 1;
        }
        while npos >= floor {
            aseq[npos as usize] = b'.';
            if let Some(pp) = &mut pp {
                pp[npos as usize] = b'.';
            }
            npos -= 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::Pipeline;

    fn test_domain(bitscore: f32, lnp: f64) -> Domain {
        Domain {
            iali: 1,
            jali: 10,
            ienv: 1,
            jenv: 10,
            bitscore,
            lnp,
            dombias: 0.0,
            oasc: 0.0,
            envsc: 0.0,
            domcorrection: 0.0,
            is_reported: true,
            is_included: true,
            ad: None,
        }
    }

    fn test_hit(name: &str, score: f32, lnp: f64, domains: Vec<Domain>) -> Hit {
        Hit {
            name: name.to_string(),
            acc: String::new(),
            desc: String::new(),
            n: 100,
            sortkey: 123.0,
            score,
            bias: 0.0,
            pre_score: score,
            sum_score: score,
            lnp,
            pre_lnp: lnp,
            sum_lnp: lnp,
            nexpected: 1.0,
            nregions: 0,
            nclustered: 0,
            noverlaps: 0,
            nenvelopes: domains.len(),
            ndom: domains.len(),
            nreported: 9,
            nincluded: 9,
            dcl: domains,
            flags: P7_IS_REPORTED | P7_IS_INCLUDED | P7_IS_DROPPED,
            seqidx: 0,
            subseq_start: 0,
        }
    }

    #[test]
    fn threshold_clears_previous_state_and_gates_domains_by_parent() {
        let mut pli = Pipeline::new();
        pli.e_value_threshold = 1.0;
        pli.inc_e = 0.01;
        pli.dom_e_value_threshold = 1.0;
        pli.inc_dome = 1.0;

        let mut th = TopHits::new();
        th.hits.push(test_hit(
            "included",
            20.0,
            -8.0,
            vec![test_domain(10.0, -8.0)],
        ));
        th.hits.push(test_hit(
            "reported-only",
            20.0,
            -2.0,
            vec![test_domain(10.0, -8.0)],
        ));
        th.hits.push(test_hit(
            "dropped",
            20.0,
            1.0,
            vec![test_domain(10.0, -8.0)],
        ));

        th.threshold(&pli, 1.0, 1.0);

        assert_eq!(th.nreported, 2);
        assert_eq!(th.nincluded, 1);

        assert!(th.hits[0].flags & P7_IS_REPORTED != 0);
        assert!(th.hits[0].flags & P7_IS_INCLUDED != 0);
        assert_eq!(th.hits[0].nreported, 1);
        assert_eq!(th.hits[0].nincluded, 1);
        assert!(th.hits[0].dcl[0].is_reported);
        assert!(th.hits[0].dcl[0].is_included);

        assert!(th.hits[1].flags & P7_IS_REPORTED != 0);
        assert_eq!(th.hits[1].flags & P7_IS_INCLUDED, 0);
        assert_eq!(th.hits[1].nreported, 1);
        assert_eq!(th.hits[1].nincluded, 0);
        assert!(th.hits[1].dcl[0].is_reported);
        assert!(!th.hits[1].dcl[0].is_included);

        assert_eq!(th.hits[2].flags & P7_IS_REPORTED, 0);
        assert_eq!(th.hits[2].flags & P7_IS_INCLUDED, 0);
        assert!(th.hits[2].flags & P7_IS_DROPPED != 0);
        assert_eq!(th.hits[2].nreported, 0);
        assert_eq!(th.hits[2].nincluded, 0);
        assert!(!th.hits[2].dcl[0].is_reported);
        assert!(!th.hits[2].dcl[0].is_included);
    }

    #[test]
    fn domain_inclusion_is_independent_from_domain_reporting_threshold() {
        let mut pli = Pipeline::new();
        pli.e_value_threshold = 1.0;
        pli.inc_e = 1.0;
        pli.dom_e_value_threshold = 0.001;
        pli.inc_dome = 0.1;

        let mut th = TopHits::new();
        th.hits.push(test_hit(
            "included-target",
            20.0,
            0.01_f64.ln(),
            vec![test_domain(10.0, 0.01_f64.ln())],
        ));

        th.threshold(&pli, 1.0, 1.0);

        assert_eq!(th.nreported, 1);
        assert_eq!(th.nincluded, 1);
        assert_eq!(th.hits[0].nreported, 0);
        assert_eq!(th.hits[0].nincluded, 1);
        assert!(!th.hits[0].dcl[0].is_reported);
        assert!(th.hits[0].dcl[0].is_included);
    }

    #[test]
    fn workaround_bug_h74_suppresses_lower_scoring_duplicate_domain() {
        let mut pli = Pipeline::new();
        pli.e_value_threshold = 1.0;
        pli.inc_e = 1.0;
        pli.dom_e_value_threshold = 1.0;
        pli.inc_dome = 1.0;

        // Two domains with identical iali/jali (the h74 collision); the hit is
        // flagged with overlapping envelopes (noverlaps != 0).
        let mut dom_hi = test_domain(20.0, 0.001_f64.ln());
        dom_hi.iali = 5;
        dom_hi.jali = 50;
        let mut dom_lo = test_domain(8.0, 0.01_f64.ln());
        dom_lo.iali = 5;
        dom_lo.jali = 50;

        let mut hit = test_hit("dup", 25.0, 0.001_f64.ln(), vec![dom_hi, dom_lo]);
        hit.noverlaps = 1;

        let mut th = TopHits::new();
        th.hits.push(hit);

        th.threshold(&pli, 1.0, 1.0);

        // Both domains pass the thresholds, but h74 suppresses the lower-scoring
        // duplicate (dcl[1], bitscore 8.0). Only the higher-scoring one remains.
        assert!(th.hits[0].dcl[0].is_reported);
        assert!(th.hits[0].dcl[0].is_included);
        assert!(!th.hits[0].dcl[1].is_reported);
        assert!(!th.hits[0].dcl[1].is_included);
        assert_eq!(th.hits[0].nreported, 1);
        assert_eq!(th.hits[0].nincluded, 1);
        // Target-level counts are unchanged by the workaround.
        assert_eq!(th.nreported, 1);
        assert_eq!(th.nincluded, 1);
    }

    #[test]
    fn workaround_bug_h74_ignored_without_overlap_flag() {
        let mut pli = Pipeline::new();
        pli.e_value_threshold = 1.0;
        pli.inc_e = 1.0;
        pli.dom_e_value_threshold = 1.0;
        pli.inc_dome = 1.0;

        // Identical iali/jali but noverlaps == 0: workaround must NOT fire.
        let mut dom_a = test_domain(20.0, 0.001_f64.ln());
        dom_a.iali = 5;
        dom_a.jali = 50;
        let mut dom_b = test_domain(8.0, 0.01_f64.ln());
        dom_b.iali = 5;
        dom_b.jali = 50;

        let mut hit = test_hit("nodup", 25.0, 0.001_f64.ln(), vec![dom_a, dom_b]);
        hit.noverlaps = 0;

        let mut th = TopHits::new();
        th.hits.push(hit);

        th.threshold(&pli, 1.0, 1.0);

        assert!(th.hits[0].dcl[0].is_reported);
        assert!(th.hits[0].dcl[1].is_reported);
        assert_eq!(th.hits[0].nreported, 2);
        assert_eq!(th.hits[0].nincluded, 2);
    }

    #[test]
    fn seqidx_alipos_sort_matches_c_longtarget_order() {
        let mut plus_short = test_hit("z", 10.0, -1.0, vec![test_domain(10.0, -1.0)]);
        plus_short.seqidx = 1;
        plus_short.dcl[0].iali = 10;
        plus_short.dcl[0].jali = 20;
        let mut minus = test_hit("a", 10.0, -1.0, vec![test_domain(10.0, -1.0)]);
        minus.seqidx = 1;
        minus.dcl[0].iali = 25;
        minus.dcl[0].jali = 10;
        let mut plus_long = test_hit("b", 10.0, -1.0, vec![test_domain(10.0, -1.0)]);
        plus_long.seqidx = 1;
        plus_long.dcl[0].iali = 10;
        plus_long.dcl[0].jali = 30;

        let mut th = TopHits::new();
        th.hits = vec![minus, plus_short, plus_long];
        th.sort_by_seqidx_and_alipos();

        assert_eq!(th.hits[0].name, "b");
        assert_eq!(th.hits[1].name, "z");
        assert_eq!(th.hits[2].name, "a");
    }

    #[test]
    fn modelname_alipos_sort_groups_nhmmscan_models_like_c() {
        let mut model_b = test_hit("model-b", 10.0, -1.0, vec![test_domain(10.0, -1.0)]);
        model_b.dcl[0].iali = 5;
        model_b.dcl[0].jali = 20;
        let mut model_a_minus = test_hit("model-a", 10.0, -1.0, vec![test_domain(10.0, -1.0)]);
        model_a_minus.dcl[0].iali = 20;
        model_a_minus.dcl[0].jali = 5;
        let mut model_a_plus = test_hit("model-a", 10.0, -1.0, vec![test_domain(10.0, -1.0)]);
        model_a_plus.dcl[0].iali = 5;
        model_a_plus.dcl[0].jali = 20;

        let mut th = TopHits::new();
        th.hits = vec![model_b, model_a_minus, model_a_plus];
        th.sort_by_modelname_and_alipos();

        assert_eq!(th.hits[0].name, "model-a");
        assert!(th.hits[0].dcl[0].iali < th.hits[0].dcl[0].jali);
        assert_eq!(th.hits[1].name, "model-a");
        assert!(th.hits[1].dcl[0].iali > th.hits[1].dcl[0].jali);
        assert_eq!(th.hits[2].name, "model-b");
    }

    #[test]
    fn inclusion_score_thresholds_use_negative_score_sortkeys_for_ascending_sort() {
        let mut pli = Pipeline::new();
        pli.inc_by_e = false;
        pli.inc_t = Some(0.0);

        let mut th = TopHits::new();
        th.hits
            .push(test_hit("low", 10.0, -20.0, vec![test_domain(10.0, -20.0)]));
        th.hits
            .push(test_hit("high", 50.0, -1.0, vec![test_domain(10.0, -1.0)]));

        th.threshold(&pli, 1.0, 1.0);
        assert_eq!(th.hits[0].name, "high");
        assert_eq!(th.hits[0].sortkey, -50.0);
        assert_eq!(th.hits[1].name, "low");
        assert_eq!(th.hits[1].sortkey, -10.0);
    }

    #[test]
    fn long_target_threshold_uses_caller_supplied_z() {
        let mut pli = Pipeline::new();
        pli.long_target = true;
        pli.e_value_threshold = 0.2;
        pli.inc_e = 0.2;
        pli.dom_e_value_threshold = 0.2;
        pli.inc_dome = 0.2;

        let lnp = 0.1_f64.ln();
        let mut th = TopHits::new();
        th.hits
            .push(test_hit("nhmmer", 20.0, lnp, vec![test_domain(10.0, lnp)]));

        th.threshold(&pli, 1.0, 1.0);
        assert_eq!(th.nreported, 1);
        assert_eq!(th.nincluded, 1);
        assert_eq!(th.hits[0].nreported, 1);
        assert_eq!(th.hits[0].nincluded, 1);

        th.threshold(&pli, 10.0, 10.0);
        assert_eq!(th.nreported, 0);
        assert_eq!(th.nincluded, 0);
        assert_eq!(th.hits[0].nreported, 0);
        assert_eq!(th.hits[0].nincluded, 0);
    }

    #[test]
    fn bit_cutoff_threshold_counts_preassigned_model_specific_flags() {
        let mut pli = Pipeline::new();
        pli.use_bit_cutoffs = crate::pipeline::BitCutoff::GA;
        pli.t = Some(100.0);
        pli.inc_t = Some(100.0);
        pli.dom_t = Some(100.0);
        pli.inc_dom_t = Some(100.0);
        pli.by_e = false;
        pli.inc_by_e = false;
        pli.dom_by_e = false;
        pli.incdom_by_e = false;

        let mut flagged = test_hit("kept", 10.0, -1.0, vec![test_domain(5.0, -1.0)]);
        flagged.flags = P7_IS_REPORTED | P7_IS_INCLUDED;
        flagged.dcl[0].is_reported = true;
        flagged.dcl[0].is_included = true;

        let mut unflagged = test_hit(
            "not-recomputed",
            200.0,
            -100.0,
            vec![test_domain(200.0, -100.0)],
        );
        unflagged.flags = 0;
        unflagged.dcl[0].is_reported = false;
        unflagged.dcl[0].is_included = false;

        let mut th = TopHits::new();
        th.hits.push(unflagged);
        th.hits.push(flagged);

        th.threshold(&pli, 1.0, 1.0);

        assert_eq!(th.nreported, 1);
        assert_eq!(th.nincluded, 1);
        assert_eq!(th.hits[0].name, "not-recomputed");
        assert_eq!(th.hits[0].flags & P7_IS_REPORTED, 0);
        assert_eq!(th.hits[0].nreported, 0);
        assert_eq!(th.hits[1].name, "kept");
        assert!(th.hits[1].flags & P7_IS_REPORTED != 0);
        assert!(th.hits[1].flags & P7_IS_INCLUDED != 0);
        assert_eq!(th.hits[1].nreported, 1);
        assert_eq!(th.hits[1].nincluded, 1);
    }

    #[test]
    fn bit_cutoff_threshold_does_not_fall_back_to_evalues_when_no_hits_preassigned() {
        let mut pli = Pipeline::new();
        pli.use_bit_cutoffs = crate::pipeline::BitCutoff::GA;
        pli.e_value_threshold = 10.0;
        pli.inc_e = 10.0;

        let mut th = TopHits::new();
        let mut hit = test_hit(
            "low-evalue-but-below-cutoff",
            200.0,
            -100.0,
            vec![test_domain(200.0, -100.0)],
        );
        hit.flags = 0;
        hit.dcl[0].is_reported = false;
        hit.dcl[0].is_included = false;
        th.hits.push(hit);

        th.threshold(&pli, 1.0, 1.0);

        assert_eq!(th.nreported, 0);
        assert_eq!(th.nincluded, 0);
        assert_eq!(th.hits[0].flags & P7_IS_REPORTED, 0);
        assert_eq!(th.hits[0].flags & P7_IS_INCLUDED, 0);
        assert_eq!(th.hits[0].nreported, 0);
        assert_eq!(th.hits[0].nincluded, 0);
    }
}
