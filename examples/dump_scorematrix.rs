//! Cross-check harness: dumps lambda and conditional probabilities exactly as
//! the C driver in the parity workflow does, so the two can be diffed.
use hmmer_pure_rs::alphabet::{Alphabet, AlphabetType};
use hmmer_pure_rs::bg::Bg;
use hmmer_pure_rs::util::scorematrix::{joint_to_conditional_on_query, ScoreMatrix};

fn run(abc_type: AlphabetType, mxname: &str) {
    let abc = Alphabet::new(abc_type);
    let bg = Bg::new(&abc);
    let mut s = ScoreMatrix::create(&abc);
    s.set(mxname, &abc).expect("set");
    let f: Vec<f64> = bg.f[..abc.k].iter().map(|&x| x as f64).collect();
    let (lambda, mut q) = s.probify_given_bg(&f, &f, &abc).expect("probify");
    joint_to_conditional_on_query(&abc, &mut q);

    println!(
        "MATRIX {} K={} Kp={} max={} min={} nc={} outorder={}",
        mxname, abc.k, abc.kp, s.max(), s.min(), s.nc, s.outorder
    );
    println!("LAMBDA {mxname} {lambda:.17e}");
    for a in 0..(abc.kp - 2) {
        print!("COND {mxname} {a}");
        for b in 0..abc.k {
            print!(" {:.17e}", q[a][b]);
        }
        println!();
    }
}

fn main() {
    run(AlphabetType::Amino, "BLOSUM62");
    run(AlphabetType::Amino, "PAM30");
    run(AlphabetType::Dna, "DNA1");
    run(AlphabetType::Rna, "DNA1");
}
