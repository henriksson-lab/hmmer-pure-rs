use std::collections::HashSet;
use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;

use hmmer_pure_rs::hmmfile_binary::{read_binary_hmm_file, write_binary_hmm};

fn main() {
    let mut args = std::env::args().skip(1);
    let input = PathBuf::from(
        args.next()
            .expect("usage: extract_binary_hmms <input.h3m> <output.h3m> <acc>..."),
    );
    let output = PathBuf::from(
        args.next()
            .expect("usage: extract_binary_hmms <input.h3m> <output.h3m> <acc>..."),
    );
    let wanted: HashSet<String> = args
        .map(|acc| acc.split('.').next().unwrap().to_string())
        .collect();
    assert!(!wanted.is_empty(), "at least one accession is required");

    let hmms = read_binary_hmm_file(&input).expect("read binary HMMs");
    let file = File::create(&output).expect("create output");
    let mut writer = BufWriter::new(file);
    let mut found = HashSet::new();

    for hmm in &hmms {
        let acc_base = hmm.acc.as_deref().unwrap_or("").split('.').next().unwrap();
        if wanted.contains(acc_base) || wanted.contains(hmm.name.as_str()) {
            write_binary_hmm(&mut writer, hmm).expect("write binary HMM");
            found.insert(acc_base.to_string());
        }
    }

    assert_eq!(found, wanted, "did not find all requested HMMs");
}
