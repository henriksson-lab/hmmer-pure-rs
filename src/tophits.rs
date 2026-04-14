//! P7_TOPHITS - Ranked hit list for search results.
//! Simplified port focused on what hmmsearch needs.

/// Alignment display data for one domain.
#[derive(Debug, Clone, Default)]
pub struct AliDisplay {
    pub model: String,  // consensus model line
    pub mline: String,  // match/identity line
    pub aseq: String,   // aligned target sequence
    pub ppline: String, // posterior probability annotation
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
    pub ndom: usize,    // actual number of domains
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

    /// Sort hits by sort key (E-value / score).
    pub fn sort_by_sortkey(&mut self) {
        self.hits.sort_by(|a, b| {
            a.sortkey
                .partial_cmp(&b.sortkey)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        self.is_sorted = true;
    }

    /// Apply reporting and inclusion thresholds.
    /// Uses E-value thresholds: inc_e for inclusion, report_e for reporting.
    pub fn threshold(&mut self, pli: &super::pipeline::Pipeline, z: f64, domz: f64) {
        self.nreported = 0;
        self.nincluded = 0;

        for hit in &mut self.hits {
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
