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
    #[arg(short = 'a', action = ArgAction::SetTrue, conflicts_with_all = ["binary", "hmmer2"])]
    ascii: bool,

    /// Output HMMER3 binary format
    #[arg(short = 'b', action = ArgAction::SetTrue, conflicts_with_all = ["ascii", "hmmer2"])]
    binary: bool,

    /// HMMER2: output backward-compatible HMMER2 ASCII format (ls mode)
    #[arg(short = '2', action = ArgAction::SetTrue, conflicts_with_all = ["ascii", "binary"])]
    hmmer2: bool,

    /// Choose output legacy 3.x file format by name
    #[arg(long = "outfmt", conflicts_with = "hmmer2")]
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

    // -a/-b/-2 mutual exclusion (C: "-a,-b,-2" toggle group) is enforced at
    // parse time by clap.
    let ascii_format = match args.outfmt.as_deref() {
        Some(outfmt) => HmmAsciiFormat::parse(outfmt).unwrap_or_else(|| {
            // Match C's p7_Fail (hmmer/src/errors.c:48-52): leading "\nError: ",
            // the format string "No such 3.x output format code %s.\n", then a
            // trailing "\n". Net bytes: "\nError: No such 3.x output format code <fmt>.\n\n".
            eprint!("\nError: No such 3.x output format code {outfmt}.\n\n");
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
        if args.hmmer2 {
            hmmfile::write_hmm_h2_ascii(&mut out, hmm).unwrap_or_else(|e| {
                eprintln!("Error writing HMMER2 HMM: {}", e);
                std::process::exit(1);
            });
        } else if args.binary {
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
    fn hmmconvert_rejects_format_toggle_conflicts() {
        // C: -a/-b/-2 share toggle group "-a,-b,-2" -> mutually exclusive.
        assert!(Args::try_parse_from(["hmmconvert", "-a", "-b", "models.hmm"]).is_err());
        assert!(Args::try_parse_from(["hmmconvert", "-a", "-2", "models.hmm"]).is_err());
        assert!(Args::try_parse_from(["hmmconvert", "-b", "-2", "models.hmm"]).is_err());
        // C: --outfmt incompatible with -2.
        assert!(
            Args::try_parse_from(["hmmconvert", "-2", "--outfmt", "3/a", "models.hmm"]).is_err()
        );
    }
}
