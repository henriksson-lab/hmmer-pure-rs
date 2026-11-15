//! HMMER - Biological sequence analysis using profile hidden Markov models.
//!
//! This is a Rust port of [HMMER 3.4](http://hmmer.org/), a C tool for searching
//! sequence databases for homologous sequences using profile HMMs.
//!
//! # Lengths, modes, and score units
//!
//! An HMM's `m` is its number of match states. A target's length is its
//! number of residues (amino acids or nucleotides). Digital sequences store
//! residues at indices `1..=n`, with sentinels at `0` and `n + 1`; sentinels
//! are excluded from sequence lengths. Use [`Alphabet::digitize_checked`]
//! to reject unsupported symbols in unaligned sequence text.
//!
//! [`profile::profile_config`] converts model probabilities into scoring
//! tables. Its target length configures the unrelated sequence before,
//! between, and after matching domains. [`profile::P7_LOCAL`] allows partial
//! model matches and multiple domains per target; the other mode constants
//! select full-model coverage or a single domain. [`Pipeline::run`] updates
//! profile lengths for each target automatically. Low-level scoring callers
//! must configure lengths themselves as described by their scoring routine.
//!
//! Profile scores and low-level dynamic-programming scores use natural-log
//! units (nats); reported hit and domain bit scores use base-2 units (bits).
//! E-values also depend on the effective search-space size.
//!
//! # Quick Start
//!
//! ```no_run
//! use hmmer_pure_rs::{hmmfile, Searcher, Sequence};
//! use std::path::Path;
//!
//! let hmm = hmmfile::read_first_hmm_file_auto(Path::new("query.hmm"))?;
//! let mut searcher = Searcher::new(&hmm)?;
//! let target = Sequence::from_text(
//!     "target", b"ACDEFGHIKLMNPQRSTVWY", searcher.alphabet(),
//! )?;
//! let results = searcher.search([&target])?;
//! for hit in results.reported_hits() {
//!     println!("{}: {:.1} bits, E={:.2e}",
//!         hit.name, hit.score, results.sequence_evalue(hit));
//! }
//! # Ok::<(), hmmer_pure_rs::errors::HmmerError>(())
//! ```
//!
//! For streaming input, call [`Searcher::begin_search`], add each target with
//! [`SearchSession::add`], then call [`SearchSession::finish`] to obtain final
//! results. Buffers can be reused immediately after `add`. The searcher handles
//! background/profile lengths and final database-size-dependent thresholds.
//! Existing low-level APIs remain available for custom scoring workflows.

// FFI bindings use non-Rust naming conventions
#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

// Let the crate refer to itself by name, so the CLI subcommand modules (moved
// in from the binary) can keep using their existing `hmmer_pure_rs::` paths.
extern crate self as hmmer_pure_rs;

pub mod alphabet;
pub mod bg;
pub mod builder;
pub mod calibrate;
pub mod domaindef;
pub mod dp;
pub mod dsqdata;
pub mod errors;
pub mod eweight;
pub mod fm_index;
pub mod hmm;
pub mod hmmfile;
pub mod hmmfile_binary;
pub mod logsum;
pub mod msa;
pub mod output;
pub mod pipeline;
pub mod pressed;
pub mod prior;
pub mod profile;
pub mod search;
pub mod seqmodel;
pub mod sequence;
pub mod simd;
pub mod spensemble;
pub mod ssi;
pub mod stats;
pub mod subcmd;
pub mod tophits;
pub mod trace;
pub mod util;

// Re-export key types at crate root for convenience
pub use alphabet::Alphabet;
pub use bg::Bg;
pub use hmm::Hmm;
pub use pipeline::Pipeline;
pub use profile::{Profile, SearchMode};
pub use search::{SearchOptions, SearchResults, SearchSession, SearchStats, Searcher};
pub use sequence::Sequence;
pub use simd::oprofile::OProfile;
pub use tophits::TopHits;

// Compile the README's Rust examples as part of cargo test --doc.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
mod readme_examples {}
