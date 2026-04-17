//! alimask — add/modify mask annotation on a Stockholm alignment.

use std::io::Write;
use std::path::PathBuf;

use clap::Parser;

use hmmer_pure_rs::msa;

#[derive(Parser)]
#[command(
    name = "alimask",
    about = "Add mask annotation to a Stockholm alignment"
)]
struct Args {
    /// Input alignment file (Stockholm)
    msafile: PathBuf,

    /// Output alignment file
    outfile: PathBuf,

    /// Model mask to apply (string of 'm' and '.' characters)
    #[arg(long = "modelmask")]
    modelmask: Option<String>,
}

pub fn run(args: Vec<String>) -> std::process::ExitCode {
    let args = Args::parse_from(&args);

    let msas = msa::read_stockholm(&args.msafile).unwrap_or_else(|e| {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    });

    let mut out = std::fs::File::create(&args.outfile).unwrap_or_else(|e| {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    });

    for alignment in &msas {
        writeln!(out, "# STOCKHOLM 1.0").unwrap();
        if !alignment.name.is_empty() {
            writeln!(out, "#=GF ID {}", alignment.name).unwrap();
        }

        // Write sequences
        for (i, name) in alignment.sqname.iter().enumerate() {
            let seq: String = alignment.aseq[i].iter().map(|&b| b as char).collect();
            writeln!(out, "{:<20} {}", name, seq).unwrap();
        }

        // Write RF if present
        if let Some(ref rf) = alignment.rf {
            let rf_str: String = rf.iter().map(|&b| b as char).collect();
            writeln!(out, "#=GC RF              {}", rf_str).unwrap();
        }

        // Write model mask if provided
        if let Some(ref mask) = args.modelmask {
            writeln!(out, "#=GC MM              {}", mask).unwrap();
        }

        writeln!(out, "//").unwrap();
    }

    eprintln!(
        "Wrote {} alignment(s) to {}",
        msas.len(),
        args.outfile.display()
    );
    std::process::ExitCode::SUCCESS
}
