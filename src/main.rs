//! hmmer-pure-rs — unified CLI for all HMMER tools.

use std::process::ExitCode;

mod subcmd;

fn main() -> ExitCode {
    hmmer_pure_rs::util::simd_env::init();

    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        print_usage();
        return ExitCode::from(1);
    }

    let cmd = args[1].as_str();
    // Remove the subcommand from args so clap sees the right argv
    let sub_args: Vec<String> = std::iter::once(format!("hmmer {}", cmd))
        .chain(args[2..].iter().cloned())
        .collect();

    match cmd {
        "search" | "hmmsearch" => subcmd::hmmsearch::run(sub_args),
        "build" | "hmmbuild" => subcmd::hmmbuild::run(sub_args),
        "scan" | "hmmscan" => subcmd::hmmscan::run(sub_args),
        "phmmer" => subcmd::phmmer::run(sub_args),
        "jackhmmer" => subcmd::jackhmmer::run(sub_args),
        "nhmmer" => subcmd::nhmmer::run(sub_args),
        "nhmmscan" => subcmd::nhmmscan::run(sub_args),
        "align" | "hmmalign" => subcmd::hmmalign::run(sub_args),
        "stat" | "hmmstat" => subcmd::hmmstat::run(sub_args),
        "emit" | "hmmemit" => subcmd::hmmemit::run(sub_args),
        "convert" | "hmmconvert" => subcmd::hmmconvert::run(sub_args),
        "fetch" | "hmmfetch" => subcmd::hmmfetch::run(sub_args),
        "press" | "hmmpress" => subcmd::hmmpress::run(sub_args),
        "logo" | "hmmlogo" => subcmd::hmmlogo::run(sub_args),
        "alimask" => subcmd::alimask::run(sub_args),
        "makehmmerdb" => subcmd::makehmmerdb::run(sub_args),
        "pgmd" | "hmmpgmd" => subcmd::hmmpgmd::run(sub_args),
        "sim" | "hmmsim" => subcmd::hmmsim::run(sub_args),
        "help" | "--help" | "-h" => { print_usage(); ExitCode::SUCCESS }
        "version" | "--version" | "-V" => { println!("hmmer-pure-rs {}", env!("CARGO_PKG_VERSION")); ExitCode::SUCCESS }
        _ => {
            eprintln!("Unknown command: {}", cmd);
            eprintln!();
            print_usage();
            ExitCode::from(1)
        }
    }
}

fn print_usage() {
    eprintln!("hmmer-pure-rs {} — biological sequence analysis using profile HMMs", env!("CARGO_PKG_VERSION"));
    eprintln!();
    eprintln!("Usage: hmmer <command> [options] [args...]");
    eprintln!();
    eprintln!("Search commands:");
    eprintln!("  search      Search HMM(s) against a sequence database");
    eprintln!("  scan        Search sequence(s) against an HMM database");
    eprintln!("  phmmer      Protein sequence vs protein database search");
    eprintln!("  jackhmmer   Iterative protein sequence search");
    eprintln!("  nhmmer      DNA/RNA HMM vs nucleotide database search");
    eprintln!("  nhmmscan    Nucleotide sequence vs DNA HMM database");
    eprintln!();
    eprintln!("Build commands:");
    eprintln!("  build       Build HMM(s) from multiple sequence alignment(s)");
    eprintln!("  press       Prepare HMM database (binary format)");
    eprintln!("  makehmmerdb Create FM-index database for nhmmer");
    eprintln!();
    eprintln!("Utility commands:");
    eprintln!("  align       Align sequences to a profile HMM");
    eprintln!("  stat        Display HMM statistics");
    eprintln!("  emit        Emit sequences from an HMM");
    eprintln!("  convert     Convert HMM file formats");
    eprintln!("  fetch       Retrieve HMM by name from a file");
    eprintln!("  logo        Generate HMM logo data");
    eprintln!("  alimask     Add mask annotation to alignment");
}
