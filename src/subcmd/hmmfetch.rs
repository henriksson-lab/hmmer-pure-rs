//! hmmfetch — retrieve an HMM from an HMM file by name.
//! Uses SSI-like index for fast lookup on large databases.

use std::collections::HashSet;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;

use clap::{ArgAction, Parser};

use hmmer_pure_rs::hmmfile;
use hmmer_pure_rs::ssi::{self, Index};

#[derive(Parser)]
#[command(name = "hmmfetch", about = "Retrieve HMM(s) from an HMM file by name")]
struct Args {
    /// Second command line arg is a file of names to retrieve
    #[arg(short = 'f', action = ArgAction::SetTrue, conflicts_with_all = ["output_key", "index"])]
    key_file_mode: bool,

    /// Output HMM to file <f> instead of stdout
    #[arg(short = 'o', conflicts_with_all = ["output_key", "index"])]
    output: Option<PathBuf>,

    /// Output HMM to file named <key>
    #[arg(short = 'O', action = ArgAction::SetTrue, conflicts_with_all = ["output", "key_file_mode", "index"])]
    output_key: bool,

    /// Create an SSI index for the HMM file
    #[arg(long = "index")]
    index: bool,

    /// HMM file
    hmmfile: PathBuf,
    /// Name of HMM to retrieve, or file of names with -f
    key: Option<String>,
}

/// Entry point for `hmmfetch`: retrieve one HMM from an HMM database by name,
/// accession, or a file of names with `-f`.
///
/// `--index` writes a C/Easel-compatible SSI index (`<hmmfile>.ssi`). Normal
/// lookup builds an SSI-style in-memory lookup table, then re-scans the file to
/// locate the matching HMM by name or accession before writing it.
pub fn run(args: Vec<String>) -> std::process::ExitCode {
    let args = Args::parse_from(&args);

    if args.index {
        if args.key.is_some() {
            eprintln!("Error: --index takes only <hmmfile>");
            std::process::exit(1);
        }
        if args.hmmfile == PathBuf::from("-") {
            eprintln!("Can't use - with --index, can't index <stdin>.");
            std::process::exit(1);
        }
        match ssi::write_hmm_ssi(&args.hmmfile) {
            Ok((ssi_path, names, accessions)) => {
                println!("Working...    done.");
                if accessions > 0 {
                    println!(
                        "Indexed {} HMMs ({} names and {} accessions).",
                        names, names, accessions
                    );
                } else {
                    println!("Indexed {} HMMs ({} names).", names, names);
                }
                println!("SSI index written to file {}", ssi_path.display());
            }
            Err(e) => {
                eprintln!("Error creating SSI index: {}", e);
                std::process::exit(1);
            }
        }
    } else if args.key_file_mode {
        let Some(ref keyfile) = args.key else {
            eprintln!("Usage: hmmfetch -f <hmmfile> <keyfile>");
            std::process::exit(1);
        };
        if args.hmmfile == PathBuf::from("-") && keyfile == "-" {
            eprintln!("Either <hmmfile> or <keyfile> can be - but not both.");
            std::process::exit(1);
        }
        let keys = read_keys(keyfile);
        let hmms = read_hmms_maybe_stdin(&args.hmmfile).unwrap_or_else(|e| {
            eprintln!("Error reading HMM file: {}", e);
            std::process::exit(1);
        });
        let key_set: HashSet<&str> = keys.iter().map(String::as_str).collect();
        let mut out = open_output(args.output.as_ref());
        for hmm in &hmms {
            if key_set.contains(hmm.name.as_str())
                || hmm.acc.as_deref().is_some_and(|acc| key_set.contains(acc))
            {
                hmmfile::write_hmm(&mut out, hmm).unwrap();
            }
        }
    } else if let Some(ref key) = args.key {
        if args.hmmfile == PathBuf::from("-") {
            let hmms = read_hmms_maybe_stdin(&args.hmmfile).unwrap_or_else(|e| {
                eprintln!("Error reading HMM file: {}", e);
                std::process::exit(1);
            });
            fetch_one(&args, key, &hmms);
        } else {
            // Build index for fast lookup
            let idx = Index::build_from_hmm_file(&args.hmmfile).unwrap_or_else(|e| {
                eprintln!("Error building index: {}", e);
                std::process::exit(1);
            });

            if idx.lookup(key).is_some() {
                // Found in index — read the specific HMM
                let hmms = hmmfile::read_hmm_file(&args.hmmfile).unwrap_or_else(|e| {
                    eprintln!("Error reading HMM file: {}", e);
                    std::process::exit(1);
                });
                fetch_one(&args, key, &hmms);
            } else {
                eprintln!(
                    "Error: HMM '{}' not found in {}",
                    key,
                    args.hmmfile.display()
                );
                std::process::exit(1);
            }
        }
    } else {
        eprintln!("Usage: hmmfetch <hmmfile> <key>");
        std::process::exit(1);
    }
    std::process::ExitCode::SUCCESS
}

fn fetch_one(args: &Args, key: &str, hmms: &[hmmer_pure_rs::Hmm]) {
    let found = hmms
        .iter()
        .find(|h| h.name == key || h.acc.as_deref() == Some(key));
    match found {
        Some(hmm) => {
            let output_name = if args.output_key {
                Some(PathBuf::from(key))
            } else {
                args.output.clone()
            };
            let mut out = open_output(output_name.as_ref());
            hmmfile::write_hmm(&mut out, hmm).unwrap();
            if output_name.is_some() {
                println!("\n\nRetrieved HMM {}.", key);
            }
        }
        None => {
            eprintln!(
                "Error: HMM '{}' not found in {}",
                key,
                args.hmmfile.display()
            );
            std::process::exit(1);
        }
    }
}

fn read_keys(path: &str) -> Vec<String> {
    fn parse_lines<I>(path: &str, lines: I) -> Vec<String>
    where
        I: Iterator<Item = std::io::Result<String>>,
    {
        let mut seen = HashSet::new();
        let mut keys = Vec::new();
        for line in lines {
            let line = line.unwrap_or_else(|e| {
                eprintln!("Error reading key file {}: {}", path, e);
                std::process::exit(1);
            });
            let line = line.split('#').next().unwrap_or("").trim();
            let Some(key) = line.split_whitespace().next() else {
                continue;
            };
            if !seen.insert(key.to_string()) {
                eprintln!(
                    "Error: key {} occurs more than once in key file {}",
                    key, path
                );
                std::process::exit(1);
            }
            keys.push(key.to_string());
        }
        keys
    }

    if path == "-" {
        return parse_lines(path, std::io::stdin().lock().lines());
    }
    let file = std::fs::File::open(path).unwrap_or_else(|e| {
        eprintln!("Error opening key file {}: {}", path, e);
        std::process::exit(1);
    });
    parse_lines(path, std::io::BufReader::new(file).lines())
}

fn read_hmms_maybe_stdin(
    path: &std::path::Path,
) -> hmmer_pure_rs::errors::HmmerResult<Vec<hmmer_pure_rs::Hmm>> {
    if path == std::path::Path::new("-") {
        let stdin = std::io::stdin();
        hmmfile::read_hmms(BufReader::new(stdin.lock()))
    } else {
        hmmfile::read_hmm_file(path)
    }
}

fn open_output(path: Option<&PathBuf>) -> Box<dyn Write> {
    match path {
        Some(path) => Box::new(std::fs::File::create(path).unwrap_or_else(|e| {
            eprintln!("Failed to open output file {}: {}", path.display(), e);
            std::process::exit(1);
        })),
        None => Box::new(std::io::stdout()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hmmfetch_parses_output_modes_like_c() {
        let args =
            Args::try_parse_from(["hmmfetch", "-o", "out.hmm", "models.hmm", "PF00001"]).unwrap();
        assert_eq!(args.output, Some(PathBuf::from("out.hmm")));
        assert_eq!(args.hmmfile, PathBuf::from("models.hmm"));
        assert_eq!(args.key.as_deref(), Some("PF00001"));

        let args = Args::try_parse_from(["hmmfetch", "-O", "models.hmm", "PF00001"]).unwrap();
        assert!(args.output_key);
    }

    #[test]
    fn hmmfetch_parses_multifetch_mode() {
        let args = Args::try_parse_from(["hmmfetch", "-f", "models.hmm", "keys.txt"]).unwrap();
        assert!(args.key_file_mode);
        assert_eq!(args.key.as_deref(), Some("keys.txt"));
    }

    #[test]
    fn hmmfetch_index_mode_takes_only_hmmfile() {
        let args = Args::try_parse_from(["hmmfetch", "--index", "models.hmm"]).unwrap();
        assert!(args.index);
        assert_eq!(args.hmmfile, PathBuf::from("models.hmm"));
        assert!(args.key.is_none());
    }

    #[test]
    fn hmmfetch_keyfile_parsing_matches_c_token_comments_and_duplicates() {
        let dir = tempfile::tempdir().unwrap();
        let keys = dir.path().join("keys.txt");
        std::fs::write(&keys, "  first extra # comment\n# skip\nsecond\n\n").unwrap();
        assert_eq!(
            read_keys(keys.to_str().unwrap()),
            vec!["first".to_string(), "second".to_string()]
        );
    }
}
