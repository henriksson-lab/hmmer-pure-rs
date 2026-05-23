//! HMMER error types, mirroring Easel's eslOK/eslFAIL/eslERANGE system.

use thiserror::Error;

/// HMMER/Easel return status codes, mirroring the C enum `eslOK`/`eslFAIL`/...
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum Status {
    /// Success.
    Ok = 0,
    /// Generic failure.
    Fail = 1,
    /// End of line reached.
    Eol = 2,
    /// End of file reached.
    Eof = 3,
    /// Requested item not found.
    Enotfound = 4,
    /// File format error.
    Eformat = 5,
    /// Multiple matches when one was expected.
    Eambiguous = 6,
    /// Duplicate name encountered.
    Edupname = 7,
    /// Incompatible objects (e.g. alphabet mismatch).
    Eincompat = 8,
    /// Invalid input/argument.
    Einval = 9,
    /// Syntax error in parsed input.
    Esyntax = 10,
    /// Numeric range error (e.g. integer overflow).
    Erange = 16,
    /// No result produced (e.g. empty MSA after filtering).
    Enoresult = 17,
}

/// Top-level HMMER error type, used as the `Err` variant of `HmmerResult`.
#[derive(Debug, Error)]
pub enum HmmerError {
    /// Generic/uncategorized error with a free-form message.
    #[error("HMMER error: {0}")]
    General(String),

    /// Filesystem or I/O error from `std::io`.
    #[error("File I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// File or stream format error (e.g. bad HMM file header).
    #[error("Format error: {0}")]
    Format(String),

    /// Requested item (HMM, sequence, key, ...) was not found.
    #[error("Not found: {0}")]
    NotFound(String),

    /// Numeric value outside its allowed range.
    #[error("Value out of range: {0}")]
    Range(String),

    /// Operation produced no result (analog of `eslENORESULT`).
    #[error("No result")]
    NoResult,

    /// Memory allocation failure.
    #[error("Allocation error")]
    Alloc,

    /// Invalid argument (analog of `eslEINVAL`).
    #[error("Invalid argument: {0}")]
    InvalidArg(String),
}

/// Convenience alias: `Result<T, HmmerError>`.
pub type HmmerResult<T> = Result<T, HmmerError>;
