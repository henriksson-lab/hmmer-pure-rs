use hmmer_pure_rs::alphabet::{AlphabetType, DSQ_SENTINEL};
use hmmer_pure_rs::hmm::{EVPARAM_UNSET, P7_FLAMBDA};
use hmmer_pure_rs::profile::{profile_config, P7_GLOCAL, P7_LOCAL, P7_UNIGLOCAL, P7_UNILOCAL};
use hmmer_pure_rs::{
    hmmfile, Alphabet, Bg, Hmm, OProfile, Pipeline, Profile, SearchMode, SearchOptions,
    SearchResults, Searcher, Sequence, TopHits,
};
use std::io::{BufReader, Cursor};

fn model() -> Hmm {
    hmmfile::read_hmms(BufReader::new(Cursor::new(include_bytes!(
        "../test_data/Globin_pfam.hmm"
    ))))
    .unwrap()
    .remove(0)
}

fn targets(abc: &Alphabet) -> Vec<Sequence> {
    let globin = b"MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKVKAHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFGKEFTPPVQAAYQKVVAGVANALAHKYH";
    vec![
        Sequence::from_text("globin", globin, abc).unwrap(),
        Sequence::from_text("short", b"ACDEFGHIKLMNPQRSTVWY", abc).unwrap(),
        Sequence::from_text("empty", b"", abc).unwrap(),
    ]
}

#[test]
fn sequence_constructor_validates_and_sets_lengths() {
    for abc in [Alphabet::amino(), Alphabet::dna(), Alphabet::rna()] {
        let seq = Sequence::from_text("target", b"ac \ng", &abc).unwrap();
        assert_eq!(seq.name, "target");
        assert_eq!((seq.n, seq.l), (3, 3));
        assert_eq!(seq.dsq.len(), 5);
        assert_eq!(seq.dsq[0], DSQ_SENTINEL);
        assert_eq!(seq.dsq[4], DSQ_SENTINEL);
        assert!(seq.acc.is_empty() && seq.desc.is_empty());
        assert_eq!(seq.taxid, -1);
        for invalid in [b"A-C".as_slice(), b"A~C", b"A?C", b"A\xffC"] {
            assert!(Sequence::from_text("bad", invalid, &abc).is_err());
        }
        assert_eq!(
            Sequence::from_text("empty", b"", &abc).unwrap().dsq,
            [255, 255]
        );
    }
}

#[test]
fn checked_profile_matches_existing_configuration_in_every_mode() {
    let hmm = model();
    let abc = Alphabet::new(hmm.abc_type);
    let bg = Bg::new(&abc);
    for (mode, legacy) in [
        (SearchMode::LocalMultihit, P7_LOCAL),
        (SearchMode::GlocalMultihit, P7_GLOCAL),
        (SearchMode::LocalUnihit, P7_UNILOCAL),
        (SearchMode::GlocalUnihit, P7_UNIGLOCAL),
    ] {
        let checked = Profile::from_hmm(&hmm, &bg, 123, mode).unwrap();
        let mut old = Profile::new(hmm.m, &abc);
        profile_config(&hmm, &bg, &mut old, 123, legacy);
        assert_eq!(i32::from(mode), legacy);
        assert_eq!(checked.mode, legacy);
        assert_eq!(checked.l, 123);
        assert_eq!(checked.tsc, old.tsc);
        assert_eq!(checked.rsc, old.rsc);
        assert_eq!(checked.xsc, old.xsc);
        assert_eq!(checked.name, old.name);
    }
}

#[test]
fn checked_profile_rejects_mismatches_and_malformed_models() {
    let hmm = model();
    let bg = Bg::new(&Alphabet::amino());
    assert!(Profile::from_hmm(
        &hmm,
        &Bg::new(&Alphabet::dna()),
        10,
        SearchMode::LocalMultihit
    )
    .is_err());
    assert!(
        Profile::from_hmm(&hmm, &bg, i32::MAX as usize + 1, SearchMode::LocalMultihit).is_err()
    );
    let mut bad = hmm.clone();
    bad.mat[1].pop();
    assert!(Profile::from_hmm(&bad, &bg, 10, SearchMode::LocalMultihit).is_err());
    bad = hmm.clone();
    bad.t[0][0] = f32::NAN;
    assert!(Profile::from_hmm(&bad, &bg, 10, SearchMode::LocalMultihit).is_err());
    bad = hmm.clone();
    bad.mat.clear();
    assert!(Profile::from_hmm(&bad, &bg, 10, SearchMode::LocalMultihit).is_err());
    let mut bad_bg = bg.clone();
    bad_bg.f[0] = 0.0;
    assert!(Profile::from_hmm(&hmm, &bad_bg, 10, SearchMode::LocalMultihit).is_err());
    let empty = Hmm::new(0, AlphabetType::Amino, 20);
    assert!(Profile::from_hmm(&empty, &bg, 10, SearchMode::LocalMultihit).is_err());
}

fn assert_same_results(left: &SearchResults, right: &SearchResults) {
    assert_eq!(left.stats(), right.stats());
    assert_eq!(left.database_size(), right.database_size());
    assert_eq!(left.domain_database_size(), right.domain_database_size());
    assert_eq!(left.top_hits().hits.len(), right.top_hits().hits.len());
    for (l, r) in left.top_hits().hits.iter().zip(&right.top_hits().hits) {
        assert_eq!(l.name, r.name);
        assert_eq!(l.seqidx, r.seqidx);
        assert_eq!(l.score, r.score);
        assert_eq!(l.lnp, r.lnp);
        assert_eq!(l.flags, r.flags);
        assert_eq!(l.dcl.len(), r.dcl.len());
        for (ld, rd) in l.dcl.iter().zip(&r.dcl) {
            assert_eq!(ld.bitscore, rd.bitscore);
            assert_eq!(ld.lnp, rd.lnp);
            assert_eq!(
                (ld.is_reported, ld.is_included),
                (rd.is_reported, rd.is_included)
            );
        }
    }
}

#[test]
fn batch_streaming_and_repeated_searches_agree() {
    let hmm = model();
    let mut searcher = Searcher::new(&hmm).unwrap();
    let targets = targets(searcher.alphabet());
    let batch = searcher.search(&targets).unwrap();
    assert!(batch.reported_hits().any(|h| h.name == "globin"));
    assert_eq!(batch.stats().targets, 3);
    assert_eq!(
        batch.stats().residues,
        targets.iter().map(|s| s.n as u64).sum()
    );
    assert_eq!(batch.database_size(), 3.0);
    assert_eq!(
        batch.domain_database_size(),
        batch.reported_hits().count() as f64
    );
    let mut session = searcher.begin_search();
    for target in &targets {
        session.add(target).unwrap();
    }
    let streaming = session.finish();
    assert_same_results(&batch, &streaming);
    assert_same_results(&batch, &searcher.search(&targets).unwrap());
    for hit in batch.reported_hits() {
        assert_eq!(batch.sequence_evalue(hit), 3.0 * hit.lnp.exp());
        for domain in &hit.dcl {
            assert_eq!(
                batch.independent_domain_evalue(domain),
                3.0 * domain.lnp.exp()
            );
            assert_eq!(
                batch.conditional_domain_evalue(domain),
                batch.domain_database_size() * domain.lnp.exp()
            );
        }
    }
}

#[test]
fn streaming_reader_can_reuse_one_sequence_buffer() {
    let hmm = model();
    let mut searcher = Searcher::new(&hmm).unwrap();
    let expected_targets = targets(searcher.alphabet());
    let expected = searcher.search(&expected_targets[..2]).unwrap();
    let mut fasta = Vec::new();
    for target in &expected_targets[..2] {
        fasta.extend_from_slice(format!(">{}\n", target.name).as_bytes());
        fasta.extend_from_slice(
            searcher
                .alphabet()
                .textize(&target.dsq, target.n)
                .as_bytes(),
        );
        fasta.push(b'\n');
    }
    let mut reader =
        hmmer_pure_rs::sequence::SeqFile::new(Cursor::new(fasta), searcher.alphabet().clone());
    let mut target = Sequence::new();
    let mut session = searcher.begin_search();
    while reader.read(&mut target).unwrap() {
        session.add(&target).unwrap();
        target.reuse();
    }
    let actual = session.finish();
    assert_same_results(&actual, &expected);
}

#[test]
fn convenience_search_matches_manual_pipeline_setup() {
    let hmm = model();
    let mut searcher = Searcher::new(&hmm).unwrap();
    let targets = targets(searcher.alphabet());
    let results = searcher.search(&targets).unwrap();
    let abc = Alphabet::new(hmm.abc_type);
    let mut bg = Bg::new(&abc);
    bg.set_filter(hmm.m, &hmm.compo);
    let mut gm = Profile::new(hmm.m, &abc);
    profile_config(&hmm, &bg, &mut gm, 400, P7_LOCAL);
    let mut om = OProfile::convert(&gm);
    let mut pipeline = Pipeline::new();
    pipeline.new_model(&gm);
    let mut hits = TopHits::new();
    for target in &targets {
        bg.set_length(target.n);
        pipeline.run(&mut gm, &mut om, &bg, &hmm, target, &mut hits);
    }
    let z = targets.len() as f64;
    hits.sort_by_sortkey();
    hits.threshold(&pipeline, z, z);
    let domz = hits.nreported as f64;
    hits.threshold(&pipeline, z, domz);
    assert!(!hits.hits.is_empty());
    assert_eq!(results.top_hits().hits.len(), hits.hits.len());
    for (actual, expected) in results.top_hits().hits.iter().zip(&hits.hits) {
        assert_eq!(actual.name, expected.name);
        assert_eq!(actual.score, expected.score);
        assert_eq!(actual.lnp, expected.lnp);
        assert_eq!(actual.flags, expected.flags);
        for (a, e) in actual.dcl.iter().zip(&expected.dcl) {
            assert_eq!((a.iali, a.jali), (e.iali, e.jali));
            assert_eq!(a.bitscore, e.bitscore);
            assert_eq!(
                (a.is_reported, a.is_included),
                (e.is_reported, e.is_included)
            );
        }
    }
}

#[test]
fn finish_reapplies_thresholds_using_the_complete_database() {
    let hmm = model();
    let mut baseline = Searcher::new(&hmm).unwrap();
    let targets = targets(baseline.alphabet());
    let result = baseline.search([&targets[0]]).unwrap();
    let p = result.reported_hits().next().unwrap().lnp.exp();
    assert!(p > 0.0);
    let options = SearchOptions {
        sequence_evalue: 2.0 * p,
        ..SearchOptions::default()
    };
    let mut searcher = Searcher::with_options(&hmm, options).unwrap();
    let mut session = searcher.begin_search();
    session.add(&targets[0]).unwrap();
    for _ in 0..10 {
        session.add(&targets[2]).unwrap();
    }
    let results = session.finish();
    assert_eq!(results.database_size(), 11.0);
    assert!(
        !results.top_hits().hits.is_empty(),
        "candidate should pass the initial storage gate"
    );
    assert_eq!(results.reported_hits().count(), 0);
    assert_eq!(results.included_hits().count(), 0);
    assert_eq!(results.domain_database_size(), 0.0);
}

#[test]
fn sessions_reject_invalid_input_without_counting_it_and_can_be_dropped() {
    let hmm = model();
    let mut searcher = Searcher::new(&hmm).unwrap();
    let targets = targets(searcher.alphabet());
    let mut session = searcher.begin_search();
    for invalid in [
        {
            let mut s = targets[0].clone();
            s.n += 1;
            s
        },
        {
            let mut s = targets[0].clone();
            s.l += 1;
            s
        },
        {
            let mut s = targets[0].clone();
            s.dsq[0] = 0;
            s
        },
        {
            let mut s = targets[0].clone();
            s.dsq[1] = 254;
            s
        },
        {
            let mut s = targets[0].clone();
            s.n = usize::MAX;
            s
        },
    ] {
        assert!(session.add(&invalid).is_err());
    }
    session.add(&targets[0]).unwrap();
    assert_eq!(session.finish().stats().targets, 1);
    {
        let mut abandoned = searcher.begin_search();
        abandoned.add(&targets[0]).unwrap();
    }
    let empty = searcher.search(std::iter::empty()).unwrap();
    assert_eq!(empty.stats().targets, 0);
    assert_eq!(empty.stats().residues, 0);
    assert_eq!(empty.database_size(), 0.0);
    assert_eq!(empty.domain_database_size(), 0.0);
    assert!(empty.top_hits().hits.is_empty());
}

#[test]
fn explicit_search_spaces_and_options_are_checked() {
    let hmm = model();
    let options = SearchOptions {
        database_size: Some(1000.0),
        domain_database_size: Some(25.0),
        max_sensitivity: true,
        ..SearchOptions::default()
    };
    let mut searcher = Searcher::with_options(&hmm, options).unwrap();
    let targets = targets(searcher.alphabet());
    let result = searcher.search([&targets[0]]).unwrap();
    assert_eq!(result.database_size(), 1000.0);
    assert_eq!(result.domain_database_size(), 25.0);
    assert!(result.reported_hits().next().is_some());
    for options in [
        SearchOptions {
            sequence_evalue: f64::NAN,
            ..SearchOptions::default()
        },
        SearchOptions {
            inclusion_evalue: -1.0,
            ..SearchOptions::default()
        },
        SearchOptions {
            database_size: Some(0.0),
            ..SearchOptions::default()
        },
        SearchOptions {
            domain_database_size: Some(f64::INFINITY),
            ..SearchOptions::default()
        },
    ] {
        assert!(Searcher::with_options(&hmm, options).is_err());
    }
}

#[test]
fn search_requires_calibration_and_handles_missing_composition() {
    let mut hmm = model();
    hmm.evparam[0] = EVPARAM_UNSET;
    assert!(Searcher::new(&hmm).is_err());
    hmm = model();
    hmm.evparam[P7_FLAMBDA] = 0.0;
    assert!(Searcher::new(&hmm).is_err());
    hmm = model();
    hmm.compo.fill(-1.0);
    let mut searcher = Searcher::new(&hmm).unwrap();
    let targets = targets(searcher.alphabet());
    assert!(searcher
        .search([&targets[0]])
        .unwrap()
        .reported_hits()
        .next()
        .is_some());
    assert!(hmm.compo.iter().all(|&p| p == -1.0));
}
