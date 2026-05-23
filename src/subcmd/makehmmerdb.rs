//! makehmmerdb — create an FM-index database for nhmmer.

use std::io::{Read, Write};
use std::path::PathBuf;

use clap::Parser;

use hmmer_pure_rs::alphabet::Alphabet;
use hmmer_pure_rs::fm_index::FmIndex;
use hmmer_pure_rs::sequence::{self, Sequence};

#[derive(Parser)]
#[command(name = "makehmmerdb", about = "Create an FM-index database for nhmmer")]
struct Args {
    /// Input sequence file (FASTA)
    seqfile: PathBuf,

    /// Output database file
    binaryfile: PathBuf,

    /// Specify input file format
    #[arg(long = "informat")]
    informat: Option<String>,

    /// Bin length (power of 2; 32<=n<=4096)
    #[arg(long = "bin_length", default_value = "256")]
    bin_length: usize,

    /// Suffix array sample rate (power of 2)
    #[arg(long = "sa_freq", default_value = "8")]
    sa_freq: usize,

    /// Input sequence block size in Mbases
    #[arg(long = "block_size", default_value = "50")]
    block_size: usize,

    /// Assert input alphabet is amino acid
    #[arg(long = "amino", conflicts_with_all = ["dna", "rna"])]
    amino: bool,

    /// Assert input alphabet is DNA
    #[arg(long = "dna", conflicts_with_all = ["amino", "rna"])]
    dna: bool,

    /// Assert input alphabet is RNA
    #[arg(long = "rna", conflicts_with_all = ["amino", "dna"])]
    rna: bool,

    /// Build a forward-strand-only database
    #[arg(long = "fwd_only")]
    fwd_only: bool,
}

/// Entry point for `makehmmerdb`: turn a (DNA) FASTA into an FM-index database
/// consumable by `nhmmer`'s SSV pre-filter.
///
/// Concatenates every sequence into a single `$`-separated text, builds a BWT
/// + suffix array via the `FmIndex` builder, and serializes a simple binary
/// container: `HMMERDB` magic, sequence/name table, then BWT, SA, and C array.
/// Corresponds to `main()` in hmmer/src/makehmmerdb.c; the C version uses
/// HMMER's full FM `meta_data` format that this Rust port simplifies.
pub fn run(args: Vec<String>) -> std::process::ExitCode {
    let args = Args::parse_from(&args);

    if let Some(ref informat) = args.informat {
        if !informat.eq_ignore_ascii_case("fasta") {
            eprintln!("{informat} is not a recognized input sequence file format");
            return std::process::ExitCode::FAILURE;
        }
    }
    if !(32..=4096).contains(&args.bin_length) || !args.bin_length.is_power_of_two() {
        eprintln!("Invalid bin length: --bin_length must be a power of 2 between 32 and 4096");
        return std::process::ExitCode::FAILURE;
    }
    if args.sa_freq == 0 || !args.sa_freq.is_power_of_two() {
        eprintln!("Invalid suffix array sample rate: --sa_freq must be a power of 2");
        return std::process::ExitCode::FAILURE;
    }
    if args.block_size == 0 {
        eprintln!("Invalid block size: --block_size must be > 0");
        return std::process::ExitCode::FAILURE;
    }
    if args.seqfile == PathBuf::from("-") && args.binaryfile == PathBuf::from("-") {
        eprintln!("Either <seqfile> or <binaryfile> can be - but not both.");
        return std::process::ExitCode::FAILURE;
    }
    if args.amino {
        eprintln!("makehmmerdb --amino is not implemented");
        return std::process::ExitCode::FAILURE;
    }
    if args.fwd_only {
        eprintln!("makehmmerdb --fwd_only is not implemented");
        return std::process::ExitCode::FAILURE;
    }

    let abc = if args.rna {
        Alphabet::rna()
    } else {
        Alphabet::dna()
    };

    // Read all sequences
    let mut all_text = Vec::new();
    let mut seq_names = Vec::new();
    let mut seq_starts = Vec::new();

    if args.seqfile == PathBuf::from("-") {
        let stdin = std::io::stdin();
        let sqf = sequence::SeqFile::new(stdin.lock(), abc.clone());
        read_sequences(sqf, &abc, &mut all_text, &mut seq_names, &mut seq_starts);
    } else {
        let sqf = sequence::open_seq_file(&args.seqfile, &abc).unwrap_or_else(|e| {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        });
        read_sequences(sqf, &abc, &mut all_text, &mut seq_names, &mut seq_starts);
    }
    if seq_names.is_empty() {
        eprintln!("Error: no sequences found in {}", args.seqfile.display());
        return std::process::ExitCode::FAILURE;
    }

    let stderr = std::io::stderr();
    let mut err = stderr.lock();

    writeln!(
        err,
        "Read {} sequences ({} residues total)",
        seq_names.len(),
        all_text.len()
    )
    .unwrap();

    // Build FM-index
    writeln!(err, "Building FM-index...").unwrap();
    let fm = FmIndex::build(&all_text);
    writeln!(err, "FM-index built: BWT length = {}", fm.bwt.len()).unwrap();

    // Write database (simple binary format)
    let outpath = args.binaryfile;
    let mut out: Box<dyn Write> = if outpath == PathBuf::from("-") {
        Box::new(std::io::stdout())
    } else {
        Box::new(std::fs::File::create(&outpath).unwrap_or_else(|e| {
            eprintln!("Error creating output: {}", e);
            std::process::exit(1);
        }))
    };

    // Write header
    out.write_all(b"HMMERDB\0").unwrap();
    out.write_all(&(seq_names.len() as u64).to_le_bytes())
        .unwrap();
    out.write_all(&(all_text.len() as u64).to_le_bytes())
        .unwrap();

    // Write sequence names and offsets
    for (i, name) in seq_names.iter().enumerate() {
        out.write_all(&(name.len() as u32).to_le_bytes()).unwrap();
        out.write_all(name.as_bytes()).unwrap();
        out.write_all(&(seq_starts[i] as u64).to_le_bytes())
            .unwrap();
    }

    // Write BWT
    out.write_all(&fm.bwt).unwrap();

    // Write suffix array
    for &sa_val in &fm.sa {
        out.write_all(&sa_val.to_le_bytes()).unwrap();
    }

    // Write C array
    for &c_val in &fm.c {
        out.write_all(&(c_val as u64).to_le_bytes()).unwrap();
    }

    writeln!(err, "Database written to: {}", outpath.display()).unwrap();
    std::process::ExitCode::SUCCESS
}

fn read_sequences<R: Read>(
    mut sqf: sequence::SeqFile<R>,
    abc: &Alphabet,
    all_text: &mut Vec<u8>,
    seq_names: &mut Vec<String>,
    seq_starts: &mut Vec<usize>,
) {
    let mut sq = Sequence::new();
    loop {
        let has_seq = sqf.read(&mut sq).unwrap_or_else(|e| {
            eprintln!("Error reading sequence file: {}", e);
            std::process::exit(1);
        });
        if !has_seq {
            break;
        }
        seq_starts.push(all_text.len());
        seq_names.push(sq.name.clone());
        let text = abc.textize(&sq.dsq, sq.n);
        all_text.extend_from_slice(text.as_bytes());
        all_text.push(b'$');
        sq.reuse();
    }
}
