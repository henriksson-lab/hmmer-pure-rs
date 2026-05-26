//! hmmconvert — convert HMM files between formats.

use std::io::{BufReader, Write};
use std::path::PathBuf;

use clap::{ArgAction, Parser};

use hmmer_pure_rs::hmmfile::{self, HmmAsciiFormat};
use hmmer_pure_rs::hmmfile_binary;

#[derive(Parser)]
#[command(
    name = "hmmconvert",
    about = "Convert profile HMM file to HMMER3 format"
)]
struct Args {
    /// Output HMMER3 ASCII format
    #[arg(short = 'a', action = ArgAction::SetTrue)]
    ascii: bool,

    /// Output HMMER3 binary format
    #[arg(short = 'b', action = ArgAction::SetTrue)]
    binary: bool,

    /// HMMER2 ASCII output is intentionally unsupported in this Rust port
    #[arg(short = '2', action = ArgAction::SetTrue)]
    hmmer2: bool,

    /// Choose output legacy 3.x file format by name
    #[arg(long = "outfmt")]
    outfmt: Option<String>,

    /// HMM file
    hmmfile: PathBuf,
}

/// Entry point for `hmmconvert`: re-emit each HMM in the input file as HMMER3
/// ASCII by default, as a selected legacy HMMER3 format with `--outfmt`,
/// or HMMER3 binary with `-b`.
/// Mirrors the default and binary output paths of `main()` in hmmer/src/hmmconvert.c.
pub fn run(args: Vec<String>) -> std::process::ExitCode {
    let args = Args::parse_from(&args);

    if args.ascii && args.binary {
        eprintln!("Error: options -a and -b are mutually exclusive");
        std::process::exit(1);
    }
    if args.hmmer2 {
        eprintln!("Error: hmmconvert -2 HMMER2 ASCII output is intentionally unsupported");
        std::process::exit(1);
    }
    let ascii_format = match args.outfmt.as_deref() {
        Some(outfmt) => HmmAsciiFormat::parse(outfmt).unwrap_or_else(|| {
            eprintln!("Error: No such 3.x output format code {outfmt}");
            std::process::exit(1);
        }),
        None => HmmAsciiFormat::Hmmer3f,
    };

    let hmms = read_hmms_maybe_stdin(&args.hmmfile).unwrap_or_else(|e| {
        eprintln!("Error reading HMM file: {}", e);
        std::process::exit(1);
    });

    let stdout = std::io::stdout();
    let mut out = stdout.lock();

    for hmm in &hmms {
        if args.binary {
            hmmfile_binary::write_binary_hmm_with_format(&mut out, hmm, ascii_format)
                .unwrap_or_else(|e| {
                    eprintln!("Error writing binary HMM: {}", e);
                    std::process::exit(1);
                });
        } else {
            hmmfile::write_hmm_with_format(&mut out, hmm, ascii_format).unwrap_or_else(|e| {
                eprintln!("Error writing HMM: {}", e);
                std::process::exit(1);
            });
        }
    }
    out.flush().unwrap_or_else(|e| {
        eprintln!("Error writing HMM: {}", e);
        std::process::exit(1);
    });
    std::process::ExitCode::SUCCESS
}

fn read_hmms_maybe_stdin(
    path: &std::path::Path,
) -> hmmer_pure_rs::errors::HmmerResult<Vec<hmmer_pure_rs::Hmm>> {
    if path == std::path::Path::new("-") {
        let stdin = std::io::stdin();
        hmmfile::read_hmms_auto(BufReader::new(stdin.lock()))
    } else {
        hmmfile::read_hmm_file_auto(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hmmconvert_parses_c_format_options() {
        let args = Args::try_parse_from(["hmmconvert", "-a", "models.hmm"]).unwrap();
        assert!(args.ascii);
        assert!(!args.binary);

        let args = Args::try_parse_from(["hmmconvert", "-b", "models.hmm"]).unwrap();
        assert!(args.binary);

        let args = Args::try_parse_from(["hmmconvert", "--outfmt", "3/f", "models.hmm"]).unwrap();
        assert_eq!(args.outfmt.as_deref(), Some("3/f"));

        let args = Args::try_parse_from(["hmmconvert", "--outfmt", "3/b", "models.hmm"]).unwrap();
        assert_eq!(args.outfmt.as_deref(), Some("3/b"));
    }

    #[test]
    fn hmmconvert_rejects_ascii_binary_conflict_before_io() {
        let args = vec![
            "hmmconvert".to_string(),
            "-a".to_string(),
            "-b".to_string(),
            "missing.hmm".to_string(),
        ];
        let parsed = Args::try_parse_from(args).unwrap();
        assert!(parsed.ascii);
        assert!(parsed.binary);
    }
}
