//! makehmmerdb — create an FM-index database for nhmmer.

use std::io::Write;
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
    #[arg(short = 'o')]
    outfile: Option<PathBuf>,
}

pub fn run(args: Vec<String>) -> std::process::ExitCode {
    let args = Args::parse_from(&args);

    let abc = Alphabet::dna();

    // Read all sequences
    let mut sqf = sequence::open_seq_file(&args.seqfile, &abc).unwrap_or_else(|e| {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    });

    let mut all_text = Vec::new();
    let mut seq_names = Vec::new();
    let mut seq_starts = Vec::new();

    let mut sq = Sequence::new();
    while sqf.read(&mut sq).unwrap() {
        seq_starts.push(all_text.len());
        seq_names.push(sq.name.clone());
        // Convert digital to text for FM-index
        let text = abc.textize(&sq.dsq, sq.n);
        all_text.extend_from_slice(text.as_bytes());
        all_text.push(b'$'); // separator between sequences
        sq.reuse();
    }

    let stderr = std::io::stderr();
    let mut err = stderr.lock();

    writeln!(err, "Read {} sequences ({} residues total)", seq_names.len(), all_text.len()).unwrap();

    // Build FM-index
    writeln!(err, "Building FM-index...").unwrap();
    let fm = FmIndex::build(&all_text);
    writeln!(err, "FM-index built: BWT length = {}", fm.bwt.len()).unwrap();

    // Write database (simple binary format)
    let outpath = args.outfile.unwrap_or_else(|| {
        let mut p = args.seqfile.clone();
        p.set_extension("hmmerdb");
        p
    });

    let mut out = std::fs::File::create(&outpath).unwrap_or_else(|e| {
        eprintln!("Error creating output: {}", e);
        std::process::exit(1);
    });

    // Write header
    out.write_all(b"HMMERDB\0").unwrap();
    out.write_all(&(seq_names.len() as u64).to_le_bytes()).unwrap();
    out.write_all(&(all_text.len() as u64).to_le_bytes()).unwrap();

    // Write sequence names and offsets
    for (i, name) in seq_names.iter().enumerate() {
        out.write_all(&(name.len() as u32).to_le_bytes()).unwrap();
        out.write_all(name.as_bytes()).unwrap();
        out.write_all(&(seq_starts[i] as u64).to_le_bytes()).unwrap();
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
