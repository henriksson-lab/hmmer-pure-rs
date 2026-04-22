//! P7_TOPHITS - Ranked hit list for search results.
//! Simplified port focused on what hmmsearch needs.

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
    let n = ad.model.chars().count();
    if n == 0 {
        return;
    }
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
    pub sortkey: f64,   // primary sort key (negative lnP)
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
    pub fn new() -> Self {
        TopHits {
            hits: Vec::new(),
            nreported: 0,
            nincluded: 0,
            is_sorted: false,
        }
    }

    /// Add a new hit and return a mutable reference to it.
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
    /// The worse-E hit is marked `P7_IS_DUPLICATE` (and stripped of the
    /// REPORTED/INCLUDED flags).
    pub fn remove_duplicates(&mut self) {
        if self.hits.len() < 2 {
            return;
        }
        let n = self.hits.len();
        let mut prev = 0usize;
        for i in 1..n {
            // Extract comparison fields without holding a borrow.
            let (p_j, s_j_raw, e_j_raw, hmm_from_j, hmm_to_j, name_j, seqidx_j) = {
                let h = &self.hits[prev];
                let dom = h.dcl.first();
                let (sj, ej) = dom.map(|d| (d.iali, d.jali)).unwrap_or((0, 0));
                let (hf, ht) = dom
                    .and_then(|d| d.ad.as_ref().map(|a| (a.hmmfrom as i64, a.hmmto as i64)))
                    .unwrap_or((0, 0));
                (h.lnp, sj, ej, hf, ht, h.name.clone(), h.seqidx)
            };
            let (p_i, s_i_raw, e_i_raw, hmm_from_i, hmm_to_i, name_i, seqidx_i) = {
                let h = &self.hits[i];
                let dom = h.dcl.first();
                let (si, ei) = dom.map(|d| (d.iali, d.jali)).unwrap_or((0, 0));
                let (hf, ht) = dom
                    .and_then(|d| d.ad.as_ref().map(|a| (a.hmmfrom as i64, a.hmmto as i64)))
                    .unwrap_or((0, 0));
                (h.lnp, si, ei, hf, ht, h.name.clone(), h.seqidx)
            };
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
            let a_dom = a.dcl.first();
            let b_dom = b.dcl.first();
            let a_idx = a.seqidx;
            let b_idx = b.seqidx;
            a_idx
                .cmp(&b_idx)
                .then_with(|| {
                    let a_name = a.name.as_str();
                    let b_name = b.name.as_str();
                    a_name.cmp(b_name)
                })
                .then_with(|| {
                    let a_pos = a_dom.map(|d| d.iali.min(d.jali)).unwrap_or(0);
                    let b_pos = b_dom.map(|d| d.iali.min(d.jali)).unwrap_or(0);
                    a_pos.cmp(&b_pos)
                })
                .then_with(|| {
                    let a_pos = a_dom.map(|d| d.iali.max(d.jali)).unwrap_or(0);
                    let b_pos = b_dom.map(|d| d.iali.max(d.jali)).unwrap_or(0);
                    a_pos.cmp(&b_pos)
                })
        });
    }

    /// Sort hits by sort key (E-value / score) with C HMMER's tiebreakers
    /// (p7_tophits.c:hit_sorter_by_sortkey):
    ///   1. sortkey ascending (lnP ascending = most-significant first).
    ///   2. name.
    ///   3. strand (positive first).
    ///   4. dcl[0].iali ascending.
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

        for hit in &mut self.hits {
            // Skip hits already marked duplicate by remove_duplicates.
            if hit.flags & P7_IS_DUPLICATE != 0 {
                continue;
            }
            let evalue = z * hit.lnp.exp();

            // Sequence-level reporting
            let reported = if pli.by_e {
                evalue <= pli.e_value_threshold
            } else if let Some(t) = pli.t {
                hit.score >= t
            } else {
                evalue <= pli.e_value_threshold
            };

            if reported {
                hit.flags |= P7_IS_REPORTED;
                self.nreported += 1;

                // Sequence-level inclusion
                let included = if pli.inc_by_e {
                    evalue <= pli.inc_e
                } else if let Some(t) = pli.inc_t {
                    hit.score >= t
                } else {
                    evalue <= pli.inc_e
                };
                if included {
                    hit.flags |= P7_IS_INCLUDED;
                    self.nincluded += 1;
                }
            } else {
                hit.flags |= P7_IS_DROPPED;
            }

            // Domain-level thresholding
            hit.nreported = 0;
            hit.nincluded = 0;
            for dom in &mut hit.dcl {
                let dom_evalue = domz * dom.lnp.exp();

                let dom_reported = if pli.dom_by_e {
                    dom_evalue <= pli.dom_e_value_threshold
                } else if let Some(t) = pli.dom_t {
                    dom.bitscore >= t
                } else {
                    dom_evalue <= pli.dom_e_value_threshold
                };

                if dom_reported {
                    dom.is_reported = true;
                    hit.nreported += 1;

                    let dom_included = if pli.incdom_by_e {
                        dom_evalue <= pli.inc_dome
                    } else if let Some(t) = pli.inc_dom_t {
                        dom.bitscore >= t
                    } else {
                        dom_evalue <= pli.inc_dome
                    };
                    if dom_included {
                        dom.is_included = true;
                        hit.nincluded += 1;
                    }
                }
            }
        }
    }
}
