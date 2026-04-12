//! HMMER - Biological sequence analysis using profile hidden Markov models.
//!
//! This is a Rust port of [HMMER 3.4](http://hmmer.org/), a C tool for searching
//! sequence databases for homologous sequences using profile HMMs.
//!
//! # Quick Start
//!
//! ```no_run
//! use hmmer::{Alphabet, Bg, Hmm, Profile, Pipeline, TopHits, OProfile};
//! use hmmer::hmmfile;
//! use hmmer::profile::{profile_config, reconfig_length, P7_LOCAL};
//! use hmmer::sequence::Sequence;
//! use std::path::Path;
//!
//! // Load an HMM
//! let hmms = hmmfile::read_hmm_file(Path::new("query.hmm")).unwrap();
//! let hmm = &hmms[0];
//!
//! // Set up alphabet, background model, and scoring profile
//! let abc = Alphabet::new(hmm.abc_type);
//! let bg = Bg::new(&abc);
//! let mut gm = Profile::new(hmm.m, &abc);
//! profile_config(hmm, &bg, &mut gm, 400, P7_LOCAL);
//! let mut om = OProfile::convert(&gm);
//!
//! // Create pipeline and hits collector
//! let mut pli = Pipeline::new();
//! pli.new_model(&gm);
//! let mut th = TopHits::new();
//!
//! // Search a sequence programmatically
//! let dsq = abc.digitize(b"ACDEFGHIKLMNPQRSTVWY");
//! let sq = Sequence { name: "query".into(), acc: String::new(), desc: String::new(), dsq, n: 20, l: 20 };
//! pli.run(&gm, &om, &bg, hmm, &sq, &mut th);
//!
//! // Access results
//! th.sort_by_sortkey();
//! for hit in &th.hits {
//!     println!("{}: score={:.1} bits", hit.name, hit.score);
//! }
//! ```

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(clippy::all)]

/// FFI bindings to the C HMMER library (for testing and gradual migration).
#[cfg(feature = "ffi")]
pub mod ffi {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

pub mod alphabet;
pub mod bg;
pub mod builder;
pub mod calibrate;
pub mod domaindef;
pub mod dp;
pub mod errors;
pub mod hmm;
pub mod hmmfile;
pub mod logsum;
pub mod msa;
pub mod output;
pub mod pipeline;
pub mod profile;
pub mod seqmodel;
pub mod sequence;
pub mod simd;
pub mod stats;
pub mod tophits;
pub mod trace;
pub mod util;

// Re-export key types at crate root for convenience
pub use alphabet::Alphabet;
pub use bg::Bg;
pub use hmm::Hmm;
pub use pipeline::Pipeline;
pub use profile::Profile;
pub use sequence::Sequence;
pub use simd::oprofile::OProfile;
pub use tophits::TopHits;
