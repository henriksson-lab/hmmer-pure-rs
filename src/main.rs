//! hmmer-pure-rs — unified CLI for all HMMER tools.

use std::ffi::OsString;
use std::process::ExitCode;

mod subcmd;

/// Unified CLI entry point: initializes SIMD env, parses the subcommand,
/// and dispatches to the matching `subcmd::*::run` module.
fn main() -> ExitCode {
    hmmer_pure_rs::util::simd_env::init();

    let args: Vec<OsString> = std::env::args_os().collect();

    if args.len() < 2 {
        print_usage();
        return ExitCode::from(1);
    }

    let cmd = args[1].to_string_lossy().into_owned();
    // Remove the subcommand from args so clap sees the right argv
    let sub_args_os: Vec<OsString> = std::iter::once(OsString::from(format!("hmmer {}", cmd)))
        .chain(args[2..].iter().cloned())
        .collect();

    match cmd.as_str() {
        "fetch" | "hmmfetch" => subcmd::hmmfetch::run_os(sub_args_os),
        "press" | "hmmpress" => subcmd::hmmpress::run_os(sub_args_os),
        "search" | "hmmsearch" => subcmd::hmmsearch::run(os_args_to_strings(sub_args_os)),
        "build" | "hmmbuild" => subcmd::hmmbuild::run(os_args_to_strings(sub_args_os)),
        "scan" | "hmmscan" => subcmd::hmmscan::run(os_args_to_strings(sub_args_os)),
        "phmmer" => subcmd::phmmer::run(os_args_to_strings(sub_args_os)),
        "jackhmmer" => subcmd::jackhmmer::run(os_args_to_strings(sub_args_os)),
        "nhmmer" => subcmd::nhmmer::run(os_args_to_strings(sub_args_os)),
        "nhmmscan" => subcmd::nhmmscan::run(os_args_to_strings(sub_args_os)),
        "align" | "hmmalign" => subcmd::hmmalign::run(os_args_to_strings(sub_args_os)),
        "stat" | "hmmstat" => subcmd::hmmstat::run(os_args_to_strings(sub_args_os)),
        "emit" | "hmmemit" => subcmd::hmmemit::run(os_args_to_strings(sub_args_os)),
        "convert" | "hmmconvert" => subcmd::hmmconvert::run(os_args_to_strings(sub_args_os)),
        "logo" | "hmmlogo" => subcmd::hmmlogo::run(os_args_to_strings(sub_args_os)),
        "alimask" => subcmd::alimask::run(os_args_to_strings(sub_args_os)),
        "makehmmerdb" => subcmd::makehmmerdb::run(os_args_to_strings(sub_args_os)),
        "pgmd" | "hmmpgmd" => subcmd::hmmpgmd::run(os_args_to_strings(sub_args_os)),
        "sim" | "hmmsim" => subcmd::hmmsim::run(os_args_to_strings(sub_args_os)),
        "help" | "--help" | "-h" => {
            print_usage();
            ExitCode::SUCCESS
        }
        "version" | "--version" | "-V" => {
            println!("hmmer-pure-rs {}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        _ => {
            eprintln!("Unknown command: {}", cmd);
            eprintln!();
            print_usage();
            ExitCode::from(1)
        }
    }
}

fn os_args_to_strings(args: Vec<OsString>) -> Vec<String> {
    args.into_iter()
        .map(|arg| {
            arg.into_string().unwrap_or_else(|arg| {
                eprintln!("Error: non-UTF8 command line argument is unsupported here: {arg:?}");
                std::process::exit(1);
            })
        })
        .collect()
}

/// Print the top-level usage banner listing every supported HMMER subcommand.
fn print_usage() {
    eprintln!(
        "hmmer-pure-rs {} — biological sequence analysis using profile HMMs",
        env!("CARGO_PKG_VERSION")
    );
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
    eprintln!("  press       Prepare an HMM database for hmmscan/nhmmscan");
    eprintln!("  makehmmerdb Create FM-index database for nhmmer");
    eprintln!();
    eprintln!("Utility commands:");
    eprintln!("  align       Align sequences to a profile HMM");
    eprintln!("  stat        Display HMM statistics");
    eprintln!("  emit        Emit sequences from an HMM");
    eprintln!("  sim         Simulate score distributions from an HMM");
    eprintln!("  convert     Convert HMM file formats");
    eprintln!("  fetch       Retrieve HMM by name from a file");
    eprintln!("  logo        Generate HMM logo data");
    eprintln!("  alimask     Add mask annotation to alignment");
    eprintln!("  pgmd        Run minimal HMM database daemon");
}
