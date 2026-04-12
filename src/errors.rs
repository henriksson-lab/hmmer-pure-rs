//! HMMER error types, mirroring Easel's eslOK/eslFAIL/eslERANGE system.

use thiserror::Error;

/// HMMER/Easel return status codes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum Status {
    Ok = 0,
    Fail = 1,
    Eol = 2,
    Eof = 3,
    Enotfound = 4,
    Eformat = 5,
    Eambiguous = 6,
    Edupname = 7,
    Eincompat = 8,
    Einval = 9,
    Esyntax = 10,
    Erange = 16,
    Enoresult = 17,
}

#[derive(Debug, Error)]
pub enum HmmerError {
    #[error("HMMER error: {0}")]
    General(String),

    #[error("File I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Format error: {0}")]
    Format(String),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Value out of range: {0}")]
    Range(String),

    #[error("No result")]
    NoResult,

    #[error("Allocation error")]
    Alloc,

    #[error("Invalid argument: {0}")]
    InvalidArg(String),
}

pub type HmmerResult<T> = Result<T, HmmerError>;
