use std::collections::HashMap;
use std::process::Command;

use hmmer_pure_rs::alphabet::Alphabet;
use hmmer_pure_rs::msa;

fn binary_path(name: &str) -> std::path::PathBuf {
    let mut path = std::env::current_exe()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    path.push(name);
    path
}

fn project_root() -> &'static str {
    env!("CARGO_MANIFEST_DIR")
}

fn test_path(relative: &str) -> String {
    format!("{}/{}", project_root(), relative)
}

fn run_hmmalign(hmm: &str, seqfile: &str) -> String {
    run_hmmalign_with_args(hmm, seqfile, &[])
}

fn run_hmmalign_with_args(hmm: &str, seqfile: &str, extra_args: &[&str]) -> String {
    let output = run_hmmalign_command(hmm, seqfile, extra_args);
    String::from_utf8(output.stdout).unwrap()
}

fn run_hmmalign_command(hmm: &str, seqfile: &str, extra_args: &[&str]) -> std::process::Output {
    let output = Command::new(binary_path("hmmer"))
        .arg("align")
        .args(extra_args)
        .args([hmm, seqfile])
        .output()
        .expect("failed to run hmmer align");

    assert!(
        output.status.success(),
        "hmmer align failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    output
}

fn run_c_hmmalign_command(hmm: &str, seqfile: &str, extra_args: &[&str]) -> std::process::Output {
    let output = Command::new(test_path("hmmer/src/hmmalign"))
        .args(extra_args)
        .args([hmm, seqfile])
        .output()
        .expect("failed to run bundled C hmmalign");

    assert!(
        output.status.success(),
        "bundled C hmmalign failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    output
}

fn normalized_alignment_lines(text: &str) -> Vec<String> {
    text.lines()
        .map(|line| line.split_whitespace().collect::<Vec<_>>().join(" "))
        .collect()
}

fn write_mismatched_mapali(src: &str, dst: &std::path::Path) {
    let original = std::fs::read_to_string(src).unwrap();
    let original_msa = msa::read_stockholm(std::path::Path::new(src)).unwrap();
    assert_eq!(original_msa.len(), 1);
    let abc = Alphabet::amino();
    let original_checksum = msa::checksum(&original_msa[0], &abc);
    let modified = original
        .replacen("ACDEFGHIKLMNPQRSTVWY", "CCDEFGHIKLMNPQRSTVWY", 1)
        .replacen("MY.", "AY.", 1);
    assert_ne!(original, modified);
    std::fs::write(dst, modified).unwrap();
    let modified_msa = msa::read_stockholm(dst).unwrap();
    assert_eq!(modified_msa.len(), 1);
    let modified_checksum = msa::checksum(&modified_msa[0], &abc);
    assert_ne!(original_checksum, modified_checksum);
}

fn parse_fasta_sequences(path: &str) -> HashMap<String, String> {
    let content = std::fs::read_to_string(path).unwrap();
    let mut seqs = HashMap::new();
    let mut current_name = String::new();
    let mut current_seq = String::new();

    for line in content.lines() {
        if let Some(rest) = line.strip_prefix('>') {
            if !current_name.is_empty() {
                seqs.insert(current_name.clone(), current_seq.clone());
            }
            current_name = rest.trim().to_string();
            current_seq.clear();
        } else {
            current_seq.push_str(line.trim());
        }
    }
    if !current_name.is_empty() {
        seqs.insert(current_name, current_seq);
    }

    seqs
}

fn dealign(seq: &str) -> String {
    seq.chars()
        .filter(|ch| ch.is_ascii_alphabetic())
        .map(|ch| ch.to_ascii_uppercase())
        .collect()
}

fn write_dealigned_fasta_from_stockholm(
    msa_path: &str,
    fasta_path: &std::path::Path,
) -> Vec<(String, String)> {
    let mapped = msa::read_stockholm(std::path::Path::new(msa_path))
        .unwrap()
        .remove(0);
    let mut generated_fasta = String::new();
    let mut expected_rows = Vec::new();
    for (name, aseq) in mapped.sqname.iter().zip(mapped.aseq.iter()) {
        let aseq = String::from_utf8(aseq.clone()).unwrap();
        expected_rows.push((name.clone(), aseq.clone()));
        generated_fasta.push('>');
        generated_fasta.push_str(name);
        generated_fasta.push('\n');
        generated_fasta.push_str(&dealign(&aseq));
        generated_fasta.push('\n');
    }
    std::fs::write(fasta_path, generated_fasta).unwrap();
    expected_rows
}

#[test]
fn hmmalign_20aa_preserves_sequences_and_stockholm_shape() {
    let output = run_hmmalign(
        &test_path("hmmer/testsuite/20aa.hmm"),
        &test_path("hmmer/testsuite/20aa-alitest.fa"),
    );
    let original = parse_fasta_sequences(&test_path("hmmer/testsuite/20aa-alitest.fa"));

    let mut aligned = HashMap::new();
    let mut pp_lines = HashMap::new();
    let mut rf = None;
    let mut pp_cons = None;

    for line in output.lines() {
        if line.starts_with("# STOCKHOLM") || line.is_empty() || line == "//" {
            continue;
        }
        if let Some(rest) = line.strip_prefix("#=GR ") {
            let (label, ppline) = rest.split_once(" PP ").unwrap();
            pp_lines.insert(label.to_string(), ppline.to_string());
        } else if let Some(rest) = line.strip_prefix("#=GC RF") {
            rf = Some(rest.trim().to_string());
        } else if let Some(rest) = line.strip_prefix("#=GC PP_cons") {
            pp_cons = Some(rest.trim().to_string());
        } else {
            let (name, seq) = line.split_once(char::is_whitespace).unwrap();
            aligned.insert(name.to_string(), seq.trim().to_string());
        }
    }

    assert_eq!(aligned.len(), 4);
    assert_eq!(pp_lines.len(), 4);

    let alen = aligned.values().next().unwrap().len();
    assert!(aligned.values().all(|seq| seq.len() == alen));
    assert!(pp_lines.values().all(|line| line.len() == alen));
    assert_eq!(rf.as_ref().unwrap().len(), alen);
    assert_eq!(pp_cons.as_ref().unwrap().len(), alen);

    for (name, aseq) in &aligned {
        let recovered: String = aseq
            .chars()
            .filter(|ch| ch.is_ascii_alphabetic())
            .map(|ch| ch.to_ascii_uppercase())
            .collect();
        assert_eq!(recovered, original[name].to_ascii_uppercase(), "{}", name);
    }
}

#[test]
fn hmmalign_20aa_matches_exact_stockholm_output() {
    let output = run_hmmalign(
        &test_path("hmmer/testsuite/20aa.hmm"),
        &test_path("hmmer/testsuite/20aa-alitest.fa"),
    );

    let expected = "\
# STOCKHOLM 1.0

test1         ....ACDEFGHIKLMN.....PQRSTVWY....
#=GR test1 PP ....9***********.....********....
test2         xxxxACDEFGHI--MNx..xxPQRSTVWY....
#=GR test2 PP ****9******9..888..88********....
test3         ....ACDEFGHI-LMNxxxxxPQRSTVWYxxxx
#=GR test3 PP ....9*******.********************
test4         .xxxACDEFGHIKLMN.....PQRSTVWYxxx.
#=GR test4 PP .***9***********.....***********.
#=GC PP_cons  ....9*********99.....********....
#=GC RF       ....xxxxxxxxxxxx.....xxxxxxxx....
//
";

    assert_eq!(output, expected);
}

#[test]
fn hmmalign_globins_matches_exact_first_block_prefix() {
    let output = run_hmmalign(
        &test_path("hmmer/tutorial/globins4.hmm"),
        &test_path("hmmer/tutorial/globins45.fa"),
    );

    let expected_prefix = "\
# STOCKHOLM 1.0

MYG_ESCGI     .-VLSDAEWQLVLNIWAKVEADVAGHGQDILIRLFKGHPETLEKFDKFKHLKTEAEMKASEDLKKHGNTVLTALGGILKK-KGHHEAELKPLAQSHATKHKIPIKYLEFISDAIIHVLHSRHPGDFGADAQAAMNKALELFRKDIAAKYKelgfqg
#=GR MYG_ESCGI PP ..69****************************************************************************.99******************************************************************7******
MYG_HORSE     g--LSDGEWQQVLNVWGKVEADIAGHGQEVLIRLFTGHPETLEKFDKFKHLKTEAEMKASEDLKKHGTVVLTALGGILKK-KGHHEAELKPLAQSHATKHKIPIKYLEFISDAIIHVLHSKHPGNFGADAQGAMTKALELFRNDIAAKYKelgfqg
#=GR MYG_HORSE PP 8..89***************************************************************************.99******************************************************************7******
MYG_PROGU     g--LSDGEWQLVLNVWGKVEGDLSGHGQEVLIRLFKGHPETLEKFDKFKHLKAEDEMRASEELKKHGTTVLTALGGILKK-KGQHAAELAPLAQSHATKHKIPVKYLEFISEAIIQVLQSKHPGDFGADAQGAMSKALELFRNDIAAKYKelgfqg
#=GR MYG_PROGU PP 8..89***************************************************************************.99******************************************************************7******
MYG_SAISC     g--LSDGEWQLVLNIWGKVEADIPSHGQEVLISLFKGHPETLEKFDKFKHLKSEDEMKASEELKKHGTTVLTALGGILKK-KGQHEAELKPLAQSHATKHKIPVKYLELISDAIVHVLQKKHPGDFGADAQGAMKKALELFRNDMAAKYKelgfqg
#=GR MYG_SAISC PP 8..89***************************************************************************.99******************************************************************7******
MYG_LYCPI     g--LSDGEWQIVLNIWGKVETDLAGHGQEVLIRLFKNHPETLDKFDKFKHLKTEDEMKGSEDLKKHGNTVLTALGGILKK-KGHHEAELKPLAQSHATKHKIPVKYLEFISDAIIQVLQNKHSGDFHADTEAAMKKALELFRNDIAAKYKelgfqg
#=GR MYG_LYCPI PP 8..89***************************************************************************.99******************************************************************7******
MYG_MOUSE     g--LSDGEWQLVLNVWGKVEADLAGHGQEVLIGLFKTHPETLDKFDKFKNLKSEEDMKGSEDLKKHGCTVLTALGTILKK-KGQHAAEIQPLAQSHATKHKIPVKYLEFISEIIIEVLKKRHSGDFGADAQGAMSKALELFRNDIAAKYKelgfqg
#=GR MYG_MOUSE PP 8..89***************************************************************************.99******************************************************************7******
MYG_MUSAN     v------DWEKVNSVWSAVESDLTAIGQNILLRLFEQYPESQNHFPKFKNKS-LGELKDTADIKAQADTVLSALGNIVKK-KGSHSQPVKALAATHITTHKIPPHYFTKITTIAVDVLSEMYPSEMNAQVQAAFSGAFKIICSDIEKEYKaanfqg
#=GR MYG_MUSAN PP 7......89***************************************9877.89*************************.99****************************************************************997******
";

    assert!(output.starts_with(expected_prefix));
}

#[test]
fn hmmalign_trim_matches_exact_first_block_prefix() {
    let output = run_hmmalign_with_args(
        &test_path("hmmer/tutorial/globins4.hmm"),
        &test_path("hmmer/tutorial/globins45.fa"),
        &["--trim"],
    );

    let expected_prefix = "\
# STOCKHOLM 1.0

MYG_ESCGI     -VLSDAEWQLVLNIWAKVEADVAGHGQDILIRLFKGHPETLEKFDKFKHLKTEAEMKASEDLKKHGNTVLTALGGILKK-KGHHEAELKPLAQSHATKHKIPIKYLEFISDAIIHVLHSRHPGDFGADAQAAMNKALELFRKDIAAKYK
#=GR MYG_ESCGI PP .69****************************************************************************.99******************************************************************7
MYG_HORSE     --LSDGEWQQVLNVWGKVEADIAGHGQEVLIRLFTGHPETLEKFDKFKHLKTEAEMKASEDLKKHGTVVLTALGGILKK-KGHHEAELKPLAQSHATKHKIPIKYLEFISDAIIHVLHSKHPGNFGADAQGAMTKALELFRNDIAAKYK
#=GR MYG_HORSE PP ..89***************************************************************************.99******************************************************************7
MYG_PROGU     --LSDGEWQLVLNVWGKVEGDLSGHGQEVLIRLFKGHPETLEKFDKFKHLKAEDEMRASEELKKHGTTVLTALGGILKK-KGQHAAELAPLAQSHATKHKIPVKYLEFISEAIIQVLQSKHPGDFGADAQGAMSKALELFRNDIAAKYK
#=GR MYG_PROGU PP ..89***************************************************************************.99******************************************************************7
MYG_SAISC     --LSDGEWQLVLNIWGKVEADIPSHGQEVLISLFKGHPETLEKFDKFKHLKSEDEMKASEELKKHGTTVLTALGGILKK-KGQHEAELKPLAQSHATKHKIPVKYLELISDAIVHVLQKKHPGDFGADAQGAMKKALELFRNDMAAKYK
";

    assert!(output.starts_with(expected_prefix));
}

#[test]
fn hmmalign_o_writes_file_and_suppresses_stdout() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let output = Command::new(binary_path("hmmer"))
        .args([
            "align",
            "-o",
            tmp.path().to_str().unwrap(),
            &test_path("hmmer/testsuite/20aa.hmm"),
            &test_path("hmmer/testsuite/20aa-alitest.fa"),
        ])
        .output()
        .expect("failed to run hmmer align -o");

    assert!(
        output.status.success(),
        "hmmer align -o failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(String::from_utf8(output.stdout).unwrap(), "");

    let file_output = std::fs::read_to_string(tmp.path()).unwrap();
    let expected = run_hmmalign(
        &test_path("hmmer/testsuite/20aa.hmm"),
        &test_path("hmmer/testsuite/20aa-alitest.fa"),
    );
    assert_eq!(file_output, expected);
}

#[test]
fn hmmalign_20aa_a2m_matches_exact_output() {
    let output = run_hmmalign_with_args(
        &test_path("hmmer/testsuite/20aa.hmm"),
        &test_path("hmmer/testsuite/20aa-alitest.fa"),
        &["--outformat", "a2m"],
    );

    let expected = "\
>test1
ACDEFGHIKLMNPQRSTVWY
>test2
xxxxACDEFGHI--MNxxxPQRSTVWY
>test3
ACDEFGHI-LMNxxxxxPQRSTVWYxxxx
>test4
xxxACDEFGHIKLMNPQRSTVWYxxx
";

    assert_eq!(output, expected);
}

#[test]
fn hmmalign_globins_a2m_matches_exact_prefix() {
    let output = run_hmmalign_with_args(
        &test_path("hmmer/tutorial/globins4.hmm"),
        &test_path("hmmer/tutorial/globins45.fa"),
        &["--outformat", "a2m"],
    );

    let expected_prefix = "\
>MYG_ESCGI
-VLSDAEWQLVLNIWAKVEADVAGHGQDILIRLFKGHPETLEKFDKFKHLKTEAEMKASE
DLKKHGNTVLTALGGILKK-KGHHEAELKPLAQSHATKHKIPIKYLEFISDAIIHVLHSR
HPGDFGADAQAAMNKALELFRKDIAAKYKelgfqg
>MYG_HORSE
g--LSDGEWQQVLNVWGKVEADIAGHGQEVLIRLFTGHPETLEKFDKFKHLKTEAEMKAS
EDLKKHGTVVLTALGGILKK-KGHHEAELKPLAQSHATKHKIPIKYLEFISDAIIHVLHS
KHPGNFGADAQGAMTKALELFRNDIAAKYKelgfqg
";

    assert!(output.starts_with(expected_prefix));
}

#[test]
fn hmmalign_stockholm_matches_bundled_c_on_20aa_fixture() {
    let rust = run_hmmalign_command(
        &test_path("hmmer/testsuite/20aa.hmm"),
        &test_path("hmmer/testsuite/20aa-alitest.fa"),
        &[],
    );
    let c = run_c_hmmalign_command(
        &test_path("hmmer/testsuite/20aa.hmm"),
        &test_path("hmmer/testsuite/20aa-alitest.fa"),
        &[],
    );

    assert_eq!(
        normalized_alignment_lines(&String::from_utf8(rust.stdout).unwrap()),
        normalized_alignment_lines(&String::from_utf8(c.stdout).unwrap())
    );
    assert_eq!(
        String::from_utf8(rust.stderr).unwrap(),
        String::from_utf8(c.stderr).unwrap()
    );
}

#[test]
fn hmmalign_a2m_matches_bundled_c_on_20aa_fixture() {
    let rust = run_hmmalign_command(
        &test_path("hmmer/testsuite/20aa.hmm"),
        &test_path("hmmer/testsuite/20aa-alitest.fa"),
        &["--outformat", "a2m"],
    );
    let c = run_c_hmmalign_command(
        &test_path("hmmer/testsuite/20aa.hmm"),
        &test_path("hmmer/testsuite/20aa-alitest.fa"),
        &["--outformat", "A2M"],
    );

    assert_eq!(
        String::from_utf8(rust.stdout).unwrap(),
        String::from_utf8(c.stdout).unwrap()
    );
    assert_eq!(
        String::from_utf8(rust.stderr).unwrap(),
        String::from_utf8(c.stderr).unwrap()
    );
}

#[test]
fn hmmalign_trim_matches_bundled_c_on_globins_fixture() {
    let rust = run_hmmalign_command(
        &test_path("hmmer/tutorial/globins4.hmm"),
        &test_path("hmmer/tutorial/globins45.fa"),
        &["--trim"],
    );
    let c = run_c_hmmalign_command(
        &test_path("hmmer/tutorial/globins4.hmm"),
        &test_path("hmmer/tutorial/globins45.fa"),
        &["--trim"],
    );

    assert_eq!(
        normalized_alignment_lines(&String::from_utf8(rust.stdout).unwrap()),
        normalized_alignment_lines(&String::from_utf8(c.stdout).unwrap())
    );
    assert_eq!(
        String::from_utf8(rust.stderr).unwrap(),
        String::from_utf8(c.stderr).unwrap()
    );
}

#[test]
fn hmmalign_mapali_legacy_20aa_rejects_checksum_mismatch() {
    let dir = tempfile::tempdir().unwrap();
    let seqfile = dir.path().join("test.fa");
    let msafile = dir.path().join("20aa-mismatch.sto");
    std::fs::write(&seqfile, b">test\nACDEFGHIKLMNPQRSTVWY\n").unwrap();
    write_mismatched_mapali(&test_path("hmmer/testsuite/20aa.sto"), &msafile);

    let output = Command::new(binary_path("hmmer"))
        .arg("align")
        .args([
            "--mapali",
            msafile.to_str().unwrap(),
            &test_path("hmmer/testsuite/20aa.hmm"),
            seqfile.to_str().unwrap(),
        ])
        .output()
        .expect("failed to run hmmer align");

    assert!(!output.status.success());
    assert_eq!(String::from_utf8(output.stdout).unwrap(), "");
    assert_eq!(
        String::from_utf8(output.stderr).unwrap(),
        "--mapali MSA isn't same as the one HMM came from (checksum mismatch)\n"
    );
}

#[test]
fn hmmalign_mapali_legacy_20aa_a2m_rejects_checksum_mismatch() {
    let dir = tempfile::tempdir().unwrap();
    let seqfile = dir.path().join("test.fa");
    let msafile = dir.path().join("20aa-mismatch.sto");
    std::fs::write(&seqfile, b">test\nACDEFGHIKLMNPQRSTVWY\n").unwrap();
    write_mismatched_mapali(&test_path("hmmer/testsuite/20aa.sto"), &msafile);

    let output = Command::new(binary_path("hmmer"))
        .arg("align")
        .args([
            "--mapali",
            msafile.to_str().unwrap(),
            "--outformat",
            "a2m",
            &test_path("hmmer/testsuite/20aa.hmm"),
            seqfile.to_str().unwrap(),
        ])
        .output()
        .expect("failed to run hmmer align");

    assert!(!output.status.success());
    assert_eq!(String::from_utf8(output.stdout).unwrap(), "");
    assert_eq!(
        String::from_utf8(output.stderr).unwrap(),
        "--mapali MSA isn't same as the one HMM came from (checksum mismatch)\n"
    );
}

#[test]
fn hmmalign_mapali_custom_built_simple_case_matches_exact_stockholm_output() {
    let dir = tempfile::tempdir().unwrap();
    let msafile = dir.path().join("seed.sto");
    let seqfile = dir.path().join("test.fa");
    let hmmfile = dir.path().join("out.hmm");
    std::fs::write(
        &msafile,
        b"# STOCKHOLM 1.0\n\n#=GF ID test\n\nseq1    ACDEFGHIKLMNPQRSTVWY\nseq2    ACDEFGHIKLMNPQRSTVWY\nseq3    ACDEFGHIKLMNPQRSTVWY\nseq4    ACDEFGHIKLMNPQRSTVWY\nseq5    ACDEFGHIKLMNPQRSTVWY\nseq6    ACDEFGHIKLMNPQRSTVWY\nseq7    ACDEFGHIKLMNPQRSTVWY\nseq8    ACDEFGHIKLMNPQRSTVWY\nseq9    ACDEFGHIKLMNPQRSTVWY\nseq0    ACDEFGHIKLMNPQRSTVWY\n#=GC RF xxxxxxxxxxxxxxxxxxxx\n//\n",
    )
    .unwrap();
    std::fs::write(&seqfile, b">test\nACDEFGHIKLMNPQRSTVWY\n").unwrap();

    let build = Command::new(binary_path("hmmer"))
        .args([
            "build",
            hmmfile.to_str().unwrap(),
            msafile.to_str().unwrap(),
        ])
        .output()
        .expect("failed to run hmmer build");
    assert!(
        build.status.success(),
        "hmmer build failed: {}",
        String::from_utf8_lossy(&build.stderr)
    );

    let output = run_hmmalign_command(
        hmmfile.to_str().unwrap(),
        seqfile.to_str().unwrap(),
        &["--mapali", msafile.to_str().unwrap()],
    );

    let expected_stdout = "\
# STOCKHOLM 1.0

seq1          ACDEFGHIKLMNPQRSTVWY
seq2          ACDEFGHIKLMNPQRSTVWY
seq3          ACDEFGHIKLMNPQRSTVWY
seq4          ACDEFGHIKLMNPQRSTVWY
seq5          ACDEFGHIKLMNPQRSTVWY
seq6          ACDEFGHIKLMNPQRSTVWY
seq7          ACDEFGHIKLMNPQRSTVWY
seq8          ACDEFGHIKLMNPQRSTVWY
seq9          ACDEFGHIKLMNPQRSTVWY
seq0          ACDEFGHIKLMNPQRSTVWY
test          ACDEFGHIKLMNPQRSTVWY
#=GR test PP  9*******************
#=GC PP_cons  9*******************
#=GC RF       xxxxxxxxxxxxxxxxxxxx
//
";

    assert_eq!(String::from_utf8(output.stdout).unwrap(), expected_stdout);
    assert_eq!(String::from_utf8(output.stderr).unwrap(), "");
}

#[test]
fn hmmalign_mapali_rebuilt_20aa_matches_exact_stockholm_output() {
    let output = run_hmmalign_command(
        &test_path("test_data/mapali/20aa-rebuilt.hmm"),
        &test_path("hmmer/testsuite/20aa-alitest.fa"),
        &["--mapali", &test_path("hmmer/testsuite/20aa.sto")],
    );

    let expected_stdout = "\
# STOCKHOLM 1.0

seq1          ....ACDEFGHIKLMN.....PQRSTVWY....
seq2          ....ACDEFGHIKLMN.....PQRSTVWY....
seq3          ....ACDEFGHIKLMN.....PQRSTVWY....
seq4          ....ACDEFGHIKLMN.....PQRSTVWY....
seq5          ....ACDEFGHIKLMN.....PQRSTVWY....
seq6          ....ACDEFGHIKLMN.....PQRSTVWY....
seq7          ....ACDEFGHIKLMN.....PQRSTVWY....
seq8          ....ACDEFGHIKLMN.....PQRSTVWY....
seq9          ....ACDEFGHIKLMN.....PQRSTVWY....
seq0          ....ACDEFGHIKLMN.....PQRSTVWY....
test1         ....ACDEFGHIKLMN.....PQRSTVWY....
#=GR test1 PP ....9***********.....********....
test2         xxxxACDEFGHI--MNx..xxPQRSTVWY....
#=GR test2 PP ****9******9..888..88********....
test3         ....ACDEFGHI-LMNxxxxxPQRSTVWYxxxx
#=GR test3 PP ....9*******.********************
test4         .xxxACDEFGHIKLMN.....PQRSTVWYxxx.
#=GR test4 PP .***9***********.....***********.
#=GC PP_cons  ....9*********99.....********....
#=GC RF       ....xxxxxxxxxxxx.....xxxxxxxx....
//
";

    assert_eq!(String::from_utf8(output.stdout).unwrap(), expected_stdout);
    assert_eq!(String::from_utf8(output.stderr).unwrap(), "");
}

#[test]
fn hmmalign_mapali_rebuilt_20aa_a2m_matches_exact_output() {
    let output = run_hmmalign_command(
        &test_path("test_data/mapali/20aa-rebuilt.hmm"),
        &test_path("hmmer/testsuite/20aa-alitest.fa"),
        &[
            "--mapali",
            &test_path("hmmer/testsuite/20aa.sto"),
            "--outformat",
            "a2m",
        ],
    );

    let expected_stdout = "\
>seq1
ACDEFGHIKLMNPQRSTVWY
>seq2
ACDEFGHIKLMNPQRSTVWY
>seq3
ACDEFGHIKLMNPQRSTVWY
>seq4
ACDEFGHIKLMNPQRSTVWY
>seq5
ACDEFGHIKLMNPQRSTVWY
>seq6
ACDEFGHIKLMNPQRSTVWY
>seq7
ACDEFGHIKLMNPQRSTVWY
>seq8
ACDEFGHIKLMNPQRSTVWY
>seq9
ACDEFGHIKLMNPQRSTVWY
>seq0
ACDEFGHIKLMNPQRSTVWY
>test1
ACDEFGHIKLMNPQRSTVWY
>test2
xxxxACDEFGHI--MNxxxPQRSTVWY
>test3
ACDEFGHI-LMNxxxxxPQRSTVWYxxxx
>test4
xxxACDEFGHIKLMNPQRSTVWYxxx
";

    assert_eq!(String::from_utf8(output.stdout).unwrap(), expected_stdout);
    assert_eq!(String::from_utf8(output.stderr).unwrap(), "");
}

#[test]
fn hmmalign_mapali_custom_built_simple_case_a2m_matches_exact_output() {
    let dir = tempfile::tempdir().unwrap();
    let msafile = dir.path().join("seed.sto");
    let seqfile = dir.path().join("test.fa");
    let hmmfile = dir.path().join("out.hmm");
    std::fs::write(
        &msafile,
        b"# STOCKHOLM 1.0\n\n#=GF ID test\n\nseq1    ACDEFGHIKLMNPQRSTVWY\nseq2    ACDEFGHIKLMNPQRSTVWY\nseq3    ACDEFGHIKLMNPQRSTVWY\nseq4    ACDEFGHIKLMNPQRSTVWY\nseq5    ACDEFGHIKLMNPQRSTVWY\nseq6    ACDEFGHIKLMNPQRSTVWY\nseq7    ACDEFGHIKLMNPQRSTVWY\nseq8    ACDEFGHIKLMNPQRSTVWY\nseq9    ACDEFGHIKLMNPQRSTVWY\nseq0    ACDEFGHIKLMNPQRSTVWY\n#=GC RF xxxxxxxxxxxxxxxxxxxx\n//\n",
    )
    .unwrap();
    std::fs::write(&seqfile, b">test\nACDEFGHIKLMNPQRSTVWY\n").unwrap();
    let build = Command::new(binary_path("hmmer"))
        .args([
            "build",
            hmmfile.to_str().unwrap(),
            msafile.to_str().unwrap(),
        ])
        .output()
        .expect("failed to run hmmer build");
    assert!(
        build.status.success(),
        "hmmer build failed: {}",
        String::from_utf8_lossy(&build.stderr)
    );

    let output = run_hmmalign_command(
        hmmfile.to_str().unwrap(),
        seqfile.to_str().unwrap(),
        &["--mapali", msafile.to_str().unwrap(), "--outformat", "a2m"],
    );

    let expected_stdout = "\
>seq1
ACDEFGHIKLMNPQRSTVWY
>seq2
ACDEFGHIKLMNPQRSTVWY
>seq3
ACDEFGHIKLMNPQRSTVWY
>seq4
ACDEFGHIKLMNPQRSTVWY
>seq5
ACDEFGHIKLMNPQRSTVWY
>seq6
ACDEFGHIKLMNPQRSTVWY
>seq7
ACDEFGHIKLMNPQRSTVWY
>seq8
ACDEFGHIKLMNPQRSTVWY
>seq9
ACDEFGHIKLMNPQRSTVWY
>seq0
ACDEFGHIKLMNPQRSTVWY
>test
ACDEFGHIKLMNPQRSTVWY
";

    assert_eq!(String::from_utf8(output.stdout).unwrap(), expected_stdout);
    assert_eq!(String::from_utf8(output.stderr).unwrap(), "");
}

#[test]
fn hmmalign_mapali_custom_built_fragment_case_matches_exact_output() {
    let dir = tempfile::tempdir().unwrap();
    let msafile = dir.path().join("in.sto");
    let seqfile = dir.path().join("in.fa");
    let hmmfile = dir.path().join("out.hmm");

    std::fs::write(
        &msafile,
        b"# STOCKHOLM 1.0\n\n# s6 is a fragment (by default hmmbuild definition)\n# s7 contains D->I transition\n# s8 contains I->D transition\n\ns1 ACDEFG.HIK.LMNPQRSTVWY\ns2 ACDEFG.HIK.LMNPQRSTVWY\ns3 ACDEFG.HIK.LMNPQRSTVWY\ns4 ACDEFG.HIK.LMNPQRSTVWY\ns5 ACDEFG.HIK.LMNPQRSTVWY\ns6 -----G.HIK.LM---------\ns7 ACDEF-aHIK.LMNPQRSTVWY\ns8 ACDEFG.HIKa.MNPQRSTVWY\n//\n",
    )
    .unwrap();
    std::fs::write(&seqfile, b">test\nCDEFGHIKLMNPQRSTVW\n").unwrap();

    let build = Command::new(binary_path("hmmer"))
        .args([
            "build",
            "-n",
            "test",
            hmmfile.to_str().unwrap(),
            msafile.to_str().unwrap(),
        ])
        .output()
        .expect("failed to run hmmer build");
    assert!(
        build.status.success(),
        "hmmer build failed: {}",
        String::from_utf8_lossy(&build.stderr)
    );

    let output = run_hmmalign_command(
        hmmfile.to_str().unwrap(),
        seqfile.to_str().unwrap(),
        &["--mapali", msafile.to_str().unwrap()],
    );

    let expected_stdout = "\
# STOCKHOLM 1.0

s1            ACDEFG.HIK.LMNPQRSTVWY
s2            ACDEFG.HIK.LMNPQRSTVWY
s3            ACDEFG.HIK.LMNPQRSTVWY
s4            ACDEFG.HIK.LMNPQRSTVWY
s5            ACDEFG.HIK.LMNPQRSTVWY
s6            -----G.HIK.LM---------
s7            ACDEF-aHIK.LMNPQRSTVWY
s8            ACDEFG.HIKa-MNPQRSTVWY
test          -CDEFG.HIK.LMNPQRSTVW-
#=GR test PP  .*****.***.**********.
#=GC PP_cons  .*****.***.**********.
#=GC RF       xxxxxx.xxx.xxxxxxxxxxx
//
";

    assert_eq!(String::from_utf8(output.stdout).unwrap(), expected_stdout);
    assert_eq!(String::from_utf8(output.stderr).unwrap(), "");
}

#[test]
fn hmmalign_mapali_legacy_caudal_act_rejects_checksum_mismatch() {
    let dir = tempfile::tempdir().unwrap();
    let seqfile = dir.path().join("Caudal_act.fa");
    let msafile = dir.path().join("Caudal_act-mismatch.sto");
    write_dealigned_fasta_from_stockholm(&test_path("hmmer/testsuite/Caudal_act.sto"), &seqfile);
    write_mismatched_mapali(&test_path("hmmer/testsuite/Caudal_act.sto"), &msafile);

    let output = Command::new(binary_path("hmmer"))
        .arg("align")
        .args([
            "--mapali",
            msafile.to_str().unwrap(),
            &test_path("hmmer/testsuite/Caudal_act.hmm"),
            seqfile.to_str().unwrap(),
        ])
        .output()
        .expect("failed to run hmmer align");

    assert!(!output.status.success());
    assert_eq!(String::from_utf8(output.stdout).unwrap(), "");
    assert_eq!(
        String::from_utf8(output.stderr).unwrap(),
        "--mapali MSA isn't same as the one HMM came from (checksum mismatch)\n"
    );
}

#[test]
fn hmmalign_mapali_rebuilt_caudal_act_accepts_seed_sequences() {
    let dir = tempfile::tempdir().unwrap();
    let seqfile = dir.path().join("Caudal_act.fa");
    let expected_rows = write_dealigned_fasta_from_stockholm(
        &test_path("hmmer/testsuite/Caudal_act.sto"),
        &seqfile,
    );

    let output = run_hmmalign_command(
        &test_path("test_data/mapali/Caudal_act-rebuilt.hmm"),
        seqfile.to_str().unwrap(),
        &["--mapali", &test_path("hmmer/testsuite/Caudal_act.sto")],
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    let mut observed = HashMap::new();
    for line in stdout.lines() {
        if line.starts_with('#') || line.is_empty() || line == "//" {
            continue;
        }
        let (name, seq) = line.split_once(char::is_whitespace).unwrap();
        observed.insert(name.to_string(), seq.trim().to_string());
    }

    assert!(stdout.starts_with("# STOCKHOLM 1.0\n\n"));
    for (name, aseq) in expected_rows {
        assert!(
            observed.contains_key(&name),
            "missing mapped row name {name}"
        );
        assert!(
            dealign(observed.get(&name).unwrap()) == dealign(&aseq),
            "dealigned sequence mismatch for {name}"
        );
    }
    assert!(stdout.contains("#=GC RF"));
    assert!(stdout.contains("#=GC PP_cons"));
}

#[test]
fn hmmalign_mapali_rebuilt_ecori_matches_exact_stockholm_output() {
    let output = run_hmmalign_command(
        &test_path("test_data/mapali/ecori-rebuilt.hmm"),
        &test_path("test_data/mapali/ecori-query.fa"),
        &["--mapali", &test_path("hmmer/testsuite/ecori.sto")],
    );

    let expected_stdout = "\
# STOCKHOLM 1.0

seq1          GAATTC
seq2          GAATTC
ecori_query   GAATTC
#=GR ecori_query PP 79**97
#=GC PP_cons  79**97
#=GC RF       xxxxxx
//
";

    assert_eq!(String::from_utf8(output.stdout).unwrap(), expected_stdout);
    assert_eq!(String::from_utf8(output.stderr).unwrap(), "");
}
