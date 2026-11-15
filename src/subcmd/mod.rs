//! Command-line entry points for HMMER subcommands.
//!
//! Each `run`/`run_os` function expects an argument vector including the
//! program name. These entry points can terminate the process and perform
//! file and standard-stream I/O; use the other crate modules for library calls.

pub mod alimask;
pub mod hmmalign;
pub mod hmmbuild;
pub mod hmmconvert;
pub mod hmmemit;
pub mod hmmfetch;
pub mod hmmlogo;
#[cfg(unix)]
pub mod hmmpgmd;
pub mod hmmpress;
pub mod hmmscan;
pub mod hmmsearch;
pub mod hmmsim;
pub mod hmmstat;
pub mod jackhmmer;
pub mod makehmmerdb;
pub mod nhmmer;
pub mod nhmmscan;
pub mod phmmer;
