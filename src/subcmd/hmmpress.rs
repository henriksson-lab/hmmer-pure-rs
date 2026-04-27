//! hmmpress — prepare an HMM database for hmmscan.
//! Creates binary pressed format files (.h3m, .h3f, .h3p, .h3i).

use std::io::Write;
use std::path::PathBuf;

use clap::Parser;

use hmmer_pure_rs::alphabet::Alphabet;
use hmmer_pure_rs::bg::Bg;
use hmmer_pure_rs::profile::{self, Profile, P7_LOCAL};
use hmmer_pure_rs::simd::oprofile::OProfile;
use hmmer_pure_rs::{hmmfile, hmmfile_binary};

#[derive(Parser)]
#[command(name = "hmmpress", about = "Prepare an HMM database for hmmscan")]
struct Args {
    /// HMM file to press
    hmmfile: PathBuf,
}

pub fn run(args: Vec<String>) -> std::process::ExitCode {
    let args = Args::parse_from(&args);

    let hmms = hmmfile::read_hmm_file(&args.hmmfile).unwrap_or_else(|e| {
        eprintln!("Error reading HMM file: {}", e);
        std::process::exit(1);
    });

    let base = args.hmmfile.to_str().unwrap();

    // Create output files
    let h3m_path = format!("{}.h3m", base);
    let h3f_path = format!("{}.h3f", base);
    let h3p_path = format!("{}.h3p", base);
    let h3i_path = format!("{}.h3i", base);

    let mut h3m = std::fs::File::create(&h3m_path).unwrap();
    let mut h3f = std::fs::File::create(&h3f_path).unwrap();
    let mut h3p = std::fs::File::create(&h3p_path).unwrap();
    let mut h3i = std::fs::File::create(&h3i_path).unwrap();

    let stderr = std::io::stderr();
    let mut err = stderr.lock();

    writeln!(err, "Working...    ({} HMMs)", hmms.len()).unwrap();

    // Write magic numbers
    h3m.write_all(&0xE8EDEDBAu32.to_le_bytes()).unwrap(); // HMMER3/f magic
    h3f.write_all(&0xE8EDEDBAu32.to_le_bytes()).unwrap();
    h3p.write_all(&0xE8EDEDBAu32.to_le_bytes()).unwrap();

    // Write SSI index header (simplified)
    h3i.write_all(b"HMMPRESS SSI INDEX\n").unwrap();

    for (idx, hmm) in hmms.iter().enumerate() {
        let abc = Alphabet::new(hmm.abc_type);
        let bg = Bg::new(&abc);

        // Write binary HMM to .h3m
        let h3m_offset = h3m.stream_position().unwrap_or(0);
        hmmfile_binary::write_binary_hmm(&mut h3m, hmm).unwrap_or_else(|e| {
            eprintln!("Error writing binary HMM: {}", e);
            std::process::exit(1);
        });

        // Create profile and optimized profile
        let mut gm = Profile::new(hmm.m, &abc);
        profile::profile_config(hmm, &bg, &mut gm, 400, P7_LOCAL);
        let om = OProfile::convert(&gm);

        // Write MSV filter data to .h3f
        write_msv_filter(&mut h3f, &om);

        // Write profile data to .h3p
        write_profile_data(&mut h3p, &om);

        // Write index entry to .h3i
        writeln!(h3i, "{}\t{}\t{}", hmm.name, idx, h3m_offset).unwrap();
    }

    writeln!(err, "Pressed and calculation complete.").unwrap();
    writeln!(err, "  {}", h3m_path).unwrap();
    writeln!(err, "  {}", h3f_path).unwrap();
    writeln!(err, "  {}", h3p_path).unwrap();
    writeln!(err, "  {}", h3i_path).unwrap();
    std::process::ExitCode::SUCCESS
}

fn write_msv_filter<W: Write>(w: &mut W, om: &OProfile) {
    let m = om.m as u32;
    w.write_all(&m.to_le_bytes()).unwrap();
    // Write byte-precision MSV data
    for x in 0..om.abc_kp {
        for q in 0..om.rbv[x].len() {
            w.write_all(&om.rbv[x][q]).unwrap();
        }
    }
    w.write_all(&[om.tbm_b, om.tec_b, om.tjb_b, om.base_b, om.bias_b])
        .unwrap();
    w.write_all(&om.scale_b.to_le_bytes()).unwrap();
}

fn write_profile_data<W: Write>(w: &mut W, om: &OProfile) {
    let m = om.m as u32;
    w.write_all(&m.to_le_bytes()).unwrap();
    // Write word-precision Viterbi data
    for x in 0..om.abc_kp {
        for q in 0..om.rwv[x].len() {
            for val in &om.rwv[x][q] {
                w.write_all(&val.to_le_bytes()).unwrap();
            }
        }
    }
    // Write float-precision Forward data
    for x in 0..om.abc_kp {
        for q in 0..om.rfv[x].len() {
            for val in &om.rfv[x][q] {
                w.write_all(&val.to_le_bytes()).unwrap();
            }
        }
    }
}

use std::io::Seek;
