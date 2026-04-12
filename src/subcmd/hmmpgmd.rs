//! hmmpgmd — HMMER search daemon.
//! Listens for search requests over TCP.

use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::process::ExitCode;
use std::path::PathBuf;

use clap::Parser;

use hmmer_pure_rs::alphabet::Alphabet;
use hmmer_pure_rs::bg::Bg;
use hmmer_pure_rs::hmmfile;
use hmmer_pure_rs::logsum;
use hmmer_pure_rs::pipeline::Pipeline;
use hmmer_pure_rs::profile::{self, Profile, P7_LOCAL};
use hmmer_pure_rs::sequence::Sequence;
use hmmer_pure_rs::simd::oprofile::OProfile;
use hmmer_pure_rs::tophits::TopHits;

#[derive(Parser)]
#[command(name = "hmmpgmd", about = "HMMER search daemon")]
struct Args {
    /// HMM database file to serve
    #[arg(long = "hmmdb")]
    hmmdb: PathBuf,

    /// Sequence database file
    #[arg(long = "seqdb")]
    seqdb: Option<PathBuf>,

    /// Port to listen on
    #[arg(long = "port", default_value = "51371")]
    port: u16,
}

pub fn run(args: Vec<String>) -> ExitCode {
    let args = Args::parse_from(&args);

    logsum::p7_flogsuminit();

    // Load HMM database
    eprintln!("Loading HMM database: {}", args.hmmdb.display());
    let hmms = hmmfile::read_hmm_file(&args.hmmdb).unwrap_or_else(|e| {
        eprintln!("Error loading HMMs: {}", e);
        std::process::exit(1);
    });
    eprintln!("Loaded {} HMMs", hmms.len());

    let abc = Alphabet::amino();
    let bg = Bg::new(&abc);

    // Pre-build profiles
    let profiles: Vec<(Profile, OProfile)> = hmms.iter().map(|hmm| {
        let mut gm = Profile::new(hmm.m, &abc);
        profile::profile_config(hmm, &bg, &mut gm, 400, P7_LOCAL);
        let om = OProfile::convert(&gm);
        (gm, om)
    }).collect();
    eprintln!("Profiles built");

    // Start listening
    let addr = format!("0.0.0.0:{}", args.port);
    let listener = TcpListener::bind(&addr).unwrap_or_else(|e| {
        eprintln!("Cannot bind to {}: {}", addr, e);
        std::process::exit(1);
    });
    eprintln!("Listening on {}", addr);

    for stream in listener.incoming() {
        match stream {
            Ok(mut stream) => {
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut line = String::new();

                // Simple protocol: client sends a sequence, server responds with hits
                if reader.read_line(&mut line).is_ok() {
                    let seq_text = line.trim();
                    if seq_text.is_empty() || seq_text == "QUIT" {
                        let _ = writeln!(stream, "BYE");
                        continue;
                    }

                    let dsq = abc.digitize(seq_text.as_bytes());
                    let l = dsq.len() - 2;

                    if l == 0 {
                        let _ = writeln!(stream, "ERROR: empty sequence");
                        continue;
                    }

                    // Search against all HMMs
                    let mut results = Vec::new();
                    for (i, (gm, om)) in profiles.iter().enumerate() {
                        let mut local_bg = bg.clone();
                        local_bg.set_length(l);
                        let mut local_gm = gm.clone();
                        let mut local_om = om.clone();
                        profile::reconfig_length(&mut local_gm, l as i32);
                        local_om.reconfig_length(l as i32);

                        let sq = Sequence {
                            name: "query".to_string(),
                            acc: String::new(),
                            desc: String::new(),
                            dsq: dsq.clone(),
                            n: l,
                            l,
                        };

                        let mut pli = Pipeline::new();
                        pli.new_model(&local_gm);
                        let mut th = TopHits::new();
                        if pli.run(&local_gm, &local_om, &local_bg, &hmms[i], &sq, &mut th) {
                            for hit in &th.hits {
                                results.push((hmms[i].name.clone(), hit.score));
                            }
                        }
                    }

                    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    let _ = writeln!(stream, "HITS {}", results.len());
                    for (name, score) in &results {
                        let _ = writeln!(stream, "{}\t{:.1}", name, score);
                    }
                    let _ = writeln!(stream, "//");
                }
            }
            Err(e) => eprintln!("Connection error: {}", e),
        }
    }

    ExitCode::SUCCESS
}
