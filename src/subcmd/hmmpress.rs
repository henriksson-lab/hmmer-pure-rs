//! hmmpress - prepare an HMM database for hmmscan/nhmmscan.

use std::collections::HashSet;
use std::ffi::OsString;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Seek};
use std::path::PathBuf;

use clap::{ArgAction, Parser};
use hmmer_pure_rs::alphabet::Alphabet;
use hmmer_pure_rs::bg::Bg;
use hmmer_pure_rs::errors::{HmmerError, HmmerResult};
use hmmer_pure_rs::hmmfile;
use hmmer_pure_rs::hmmfile_binary::write_binary_hmm;
use hmmer_pure_rs::pressed::{write_h3f_record, write_h3p_record};
use hmmer_pure_rs::profile::{self, Profile, P7_LOCAL};
use hmmer_pure_rs::simd::oprofile::OProfile;
use hmmer_pure_rs::ssi;

#[derive(Parser)]
#[command(
    name = "hmmpress",
    about = "Prepare an HMM database for hmmscan/nhmmscan"
)]
struct Args {
    /// Force: overwrite any previous pressed sidecars
    #[arg(short = 'f', action = ArgAction::SetTrue)]
    force: bool,

    /// HMM file to press
    hmmfile: PathBuf,
}

pub fn run_os(args: Vec<OsString>) -> std::process::ExitCode {
    run_from(args)
}

fn run_from<I, T>(args: I) -> std::process::ExitCode
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
{
    let args = Args::parse_from(args);

    match press_database(&args) {
        Ok(summary) => {
            println!("Working...    done.");
            if summary.nsecondary > 0 {
                println!(
                    "Pressed and indexed {} HMMs ({} names and {} accessions).",
                    summary.nmodels, summary.nprimary, summary.nsecondary
                );
            } else {
                println!(
                    "Pressed and indexed {} HMMs ({} names).",
                    summary.nmodels, summary.nprimary
                );
            }
            println!(
                "Models pressed into binary file:   {}",
                summary.h3m.display()
            );
            println!(
                "SSI index for binary model file:   {}",
                summary.h3i.display()
            );
            println!(
                "Profiles (MSV part) pressed into:  {}",
                summary.h3f.display()
            );
            println!(
                "Profiles (remainder) pressed into: {}",
                summary.h3p.display()
            );
            std::process::ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::ExitCode::FAILURE
        }
    }
}

struct PressSummary {
    nmodels: usize,
    nprimary: usize,
    nsecondary: usize,
    h3m: PathBuf,
    h3i: PathBuf,
    h3f: PathBuf,
    h3p: PathBuf,
}

fn create_sidecar(path: &PathBuf, force: bool) -> HmmerResult<BufWriter<File>> {
    let mut opts = OpenOptions::new();
    opts.write(true);
    if force {
        opts.create(true).truncate(true);
    } else {
        opts.create_new(true);
    }
    let file = opts.open(path).map_err(|e| {
        if e.kind() == std::io::ErrorKind::AlreadyExists {
            HmmerError::Format(format!(
                "{} already exists; delete old hmmpress sidecars or use -f",
                path.display()
            ))
        } else {
            HmmerError::Io(e)
        }
    })?;
    Ok(BufWriter::new(file))
}

fn ensure_sidecars_creatable(paths: [&PathBuf; 4], force: bool) -> HmmerResult<()> {
    if force {
        return Ok(());
    }

    for path in paths {
        if path.exists() {
            return Err(HmmerError::Format(format!(
                "{} already exists; delete old hmmpress sidecars or use -f",
                path.display()
            )));
        }
    }
    Ok(())
}

fn press_database(args: &Args) -> HmmerResult<PressSummary> {
    if args.hmmfile.as_os_str() == "-" {
        return Err(HmmerError::Format(
            "Can't use - for <hmmfile> argument: can't index standard input".to_string(),
        ));
    }

    let h3m = ssi::path_with_appended_suffix(&args.hmmfile, ".h3m");
    let h3f = ssi::path_with_appended_suffix(&args.hmmfile, ".h3f");
    let h3p = ssi::path_with_appended_suffix(&args.hmmfile, ".h3p");
    let h3i = ssi::path_with_appended_suffix(&args.hmmfile, ".h3i");

    let hmms = hmmfile::read_hmm_file_auto(&args.hmmfile)?;
    prevalidate_index_keys(&hmms)?;
    ensure_sidecars_creatable([&h3m, &h3f, &h3p, &h3i], args.force)?;
    let mut mfp = create_sidecar(&h3m, args.force)?;
    let mut ffp = create_sidecar(&h3f, args.force)?;
    let mut pfp = create_sidecar(&h3p, args.force)?;
    let mut records = Vec::with_capacity(hmms.len());

    for (idx, hmm) in hmms.iter().enumerate() {
        if hmm.name.is_empty() {
            return Err(HmmerError::Format(format!(
                "Every HMM must have a name to be indexed. Failed to find name of HMM #{}",
                idx + 1
            )));
        }

        let abc = Alphabet::new(hmm.abc_type);
        let mut bg = Bg::new(&abc);
        bg.set_length(400);
        let mut gm = Profile::new(hmm.m, &abc);
        profile::profile_config(hmm, &bg, &mut gm, 400, P7_LOCAL);
        let om = OProfile::convert(&gm);

        let moff = mfp.stream_position().map_err(HmmerError::Io)? as i64;
        let foff = ffp.stream_position().map_err(HmmerError::Io)? as i64;
        let poff = pfp.stream_position().map_err(HmmerError::Io)? as i64;

        records.push((hmm.name.clone(), hmm.acc.clone(), moff as u64));
        write_binary_hmm(&mut mfp, hmm)?;
        write_h3f_record(&mut ffp, hmm, &om, [moff, foff, poff])?;
        write_h3p_record(&mut pfp, hmm, &om)?;
    }

    drop(mfp);
    drop(ffp);
    drop(pfp);

    let (_path, nprimary, nsecondary) = ssi::write_hmm_ssi_records_with_stored_path(
        &args.hmmfile,
        &ssi::path_file_name(&args.hmmfile),
        &h3i,
        records,
        args.force,
    )?;

    Ok(PressSummary {
        nmodels: hmms.len(),
        nprimary,
        nsecondary,
        h3m,
        h3i,
        h3f,
        h3p,
    })
}

fn prevalidate_index_keys(hmms: &[hmmer_pure_rs::Hmm]) -> HmmerResult<()> {
    let mut keys = HashSet::new();
    for (idx, hmm) in hmms.iter().enumerate() {
        if hmm.name.is_empty() {
            return Err(HmmerError::Format(format!(
                "Every HMM must have a name to be indexed. Failed to find name of HMM #{}",
                idx + 1
            )));
        }
        if !keys.insert(hmm.name.clone()) {
            return Err(HmmerError::Format(format!(
                "HMM name {} occurs more than once",
                hmm.name
            )));
        }
        if let Some(acc) = &hmm.acc {
            if !acc.is_empty() && !keys.insert(acc.clone()) {
                return Err(HmmerError::Format(format!(
                    "HMM accession {} occurs more than once",
                    acc
                )));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hmmpress_accepts_force_flag_like_c() {
        let args = Args::try_parse_from(["hmmpress", "-f", "models.hmm"]).unwrap();
        assert!(args.force);
        assert_eq!(args.hmmfile, PathBuf::from("models.hmm"));
    }

    #[test]
    fn hmmpress_existing_h3i_does_not_leave_partial_sidecars() {
        let dir = tempfile::tempdir().unwrap();
        let src = PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/hmmer/tutorial/fn3.hmm"
        ));
        let hmm_path = dir.path().join("fn3.hmm");
        std::fs::copy(src, &hmm_path).unwrap();
        std::fs::write(
            ssi::path_with_appended_suffix(&hmm_path, ".h3i"),
            b"old index",
        )
        .unwrap();

        let args = Args {
            force: false,
            hmmfile: hmm_path.clone(),
        };
        let err = match press_database(&args) {
            Ok(_) => panic!("hmmpress unexpectedly succeeded with existing .h3i"),
            Err(err) => err,
        };

        assert!(err.to_string().contains("already exists"));
        assert!(!ssi::path_with_appended_suffix(&hmm_path, ".h3m").exists());
        assert!(!ssi::path_with_appended_suffix(&hmm_path, ".h3f").exists());
        assert!(!ssi::path_with_appended_suffix(&hmm_path, ".h3p").exists());
    }

    #[test]
    fn hmmpress_rejects_duplicate_keys_before_sidecars() {
        let dir = tempfile::tempdir().unwrap();
        let src = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/hmmer/tutorial/fn3.hmm"
        ))
        .unwrap();
        let hmm_path = dir.path().join("dupe.hmm");
        let mut data = src.clone();
        data.extend_from_slice(&src);
        std::fs::write(&hmm_path, data).unwrap();

        let args = Args {
            force: false,
            hmmfile: hmm_path.clone(),
        };
        let err = match press_database(&args) {
            Ok(_) => panic!("hmmpress unexpectedly succeeded with duplicate keys"),
            Err(err) => err,
        };

        assert!(err.to_string().contains("occurs more than once"));
        assert!(!ssi::path_with_appended_suffix(&hmm_path, ".h3m").exists());
        assert!(!ssi::path_with_appended_suffix(&hmm_path, ".h3f").exists());
        assert!(!ssi::path_with_appended_suffix(&hmm_path, ".h3p").exists());
        assert!(!ssi::path_with_appended_suffix(&hmm_path, ".h3i").exists());
    }

    #[cfg(unix)]
    #[test]
    fn hmmpress_preserves_non_utf8_path_bytes_for_sidecars() {
        use std::os::unix::ffi::{OsStrExt, OsStringExt};

        let dir = tempfile::tempdir().unwrap();
        let src = PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/hmmer/tutorial/fn3.hmm"
        ));
        let hmm_path = dir
            .path()
            .join(std::ffi::OsString::from_vec(b"fn3-\xff.hmm".to_vec()));
        std::fs::copy(src, &hmm_path).unwrap();

        let args = Args {
            force: false,
            hmmfile: hmm_path.clone(),
        };
        let summary = press_database(&args).unwrap();

        for (path, suffix) in [
            (&summary.h3m, b".h3m".as_slice()),
            (&summary.h3f, b".h3f".as_slice()),
            (&summary.h3p, b".h3p".as_slice()),
            (&summary.h3i, b".h3i".as_slice()),
        ] {
            assert!(path.exists(), "missing sidecar {}", path.display());
            assert!(path.as_os_str().as_bytes().ends_with(suffix));
            assert!(path.as_os_str().as_bytes().contains(&0xff));
        }
    }

    #[test]
    fn hmmpress_h3i_file_table_matches_easel_basename_behavior() {
        let dir = tempfile::tempdir().unwrap();
        let src = PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/hmmer/tutorial/fn3.hmm"
        ));
        let hmm_path = dir.path().join("fn3.hmm");
        std::fs::copy(src, &hmm_path).unwrap();

        let args = Args {
            force: false,
            hmmfile: hmm_path.clone(),
        };
        let summary = press_database(&args).unwrap();
        let bytes = std::fs::read(&summary.h3i).unwrap();
        let flen = u32::from_be_bytes(bytes[30..34].try_into().unwrap()) as usize;
        let foffset = u64::from_be_bytes(bytes[54..62].try_into().unwrap()) as usize;
        let stored = &bytes[foffset..foffset + flen];

        assert_eq!(flen, hmm_path.as_os_str().len() + 1);
        assert_eq!(
            &stored[..b"fn3.hmm".len() + 1],
            b"fn3.hmm\0",
            "{}",
            String::from_utf8_lossy(stored)
        );
    }
}
