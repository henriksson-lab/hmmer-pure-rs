fn main() {
    #[cfg(feature = "ffi")]
    ffi_build();
}

#[cfg(feature = "ffi")]
fn ffi_build() {
    use std::env;
    use std::path::PathBuf;

    let hmmer_dir = PathBuf::from("hmmer");
    let src_dir = hmmer_dir.join("src");
    let easel_dir = hmmer_dir.join("easel");
    let impl_sse_dir = src_dir.join("impl_sse");
    let divsufsort_dir = hmmer_dir.join("libdivsufsort");

    // Common flags for all C builds
    let common_flags = |build: &mut cc::Build| {
        build
            .warnings(false)
            .flag("-O3")
            .flag("-pthread")
            .define("HAVE_CONFIG_H", None);
    };

    // --- Compile libeasel ---
    let easel_sources = [
        "easel", "esl_alloc", "esl_alphabet", "esl_arr2", "esl_arr3",
        "esl_bitfield", "esl_buffer", "esl_cluster", "esl_composition",
        "esl_cpu", "esl_dirichlet", "esl_distance", "esl_dmatrix",
        "esl_dsqdata", "esl_exponential", "esl_fileparser", "esl_gamma",
        "esl_gencode", "esl_getopts", "esl_gev", "esl_graph", "esl_gumbel",
        "esl_heap", "esl_histogram", "esl_hmm", "esl_huffman",
        "esl_hyperexp", "esl_iset", "esl_json", "esl_keyhash",
        "esl_lognormal", "esl_matrixops", "esl_mem", "esl_minimizer",
        "esl_mixdchlet", "esl_mixgev", "esl_mpi", "esl_msa",
        "esl_msacluster", "esl_msafile", "esl_msafile2",
        "esl_msafile_a2m", "esl_msafile_afa", "esl_msafile_clustal",
        "esl_msafile_phylip", "esl_msafile_psiblast", "esl_msafile_selex",
        "esl_msafile_stockholm", "esl_msashuffle", "esl_msaweight",
        "esl_normal", "esl_paml", "esl_quicksort", "esl_random",
        "esl_rand64", "esl_randomseq", "esl_ratematrix", "esl_recorder",
        "esl_red_black", "esl_regexp", "esl_rootfinder", "esl_scorematrix",
        "esl_sq", "esl_sqio", "esl_sqio_ascii", "esl_sqio_ncbi",
        "esl_ssi", "esl_stack", "esl_stats", "esl_stopwatch",
        "esl_stretchexp", "esl_subcmd", "esl_threads", "esl_tree",
        "esl_varint", "esl_vectorops", "esl_weibull", "esl_workqueue",
        "esl_wuss",
        "esl_sse",
    ];

    let mut easel_build = cc::Build::new();
    common_flags(&mut easel_build);
    easel_build.include(&easel_dir).include(&src_dir);
    for src in &easel_sources {
        easel_build.file(easel_dir.join(format!("{}.c", src)));
    }
    easel_build.compile("easel");

    // --- Compile libdivsufsort ---
    let mut divsufsort_build = cc::Build::new();
    divsufsort_build
        .warnings(false)
        .flag("-O3")
        .flag("-pthread")
        .include(&divsufsort_dir)
        .file(divsufsort_dir.join("divsufsort.c"));
    divsufsort_build.compile("divsufsort");

    // --- Compile impl_sse ---
    let impl_sse_sources = [
        "decoding", "fwdback", "io", "ssvfilter", "msvfilter",
        "null2", "optacc", "stotrace", "vitfilter", "p7_omx",
        "p7_oprofile", "mpi",
    ];

    let mut impl_build = cc::Build::new();
    common_flags(&mut impl_build);
    impl_build
        .include(&easel_dir)
        .include(&src_dir)
        .include(&impl_sse_dir)
        .include(&divsufsort_dir);
    for src in &impl_sse_sources {
        impl_build.file(impl_sse_dir.join(format!("{}.c", src)));
    }
    impl_build.compile("hmmer_impl_sse");

    // --- Compile libhmmer (library objects only, NO program mains) ---
    let hmmer_lib_sources = [
        "build", "cachedb", "cachedb_shard", "emit", "errors",
        "evalues", "eweight", "generic_decoding", "generic_fwdback",
        "generic_fwdback_chk", "generic_fwdback_banded", "generic_null2",
        "generic_msv", "generic_optacc", "generic_stotrace",
        "generic_viterbi", "generic_vtrace", "h2_io", "heatmap",
        "hmmlogo", "hmmdmstr", "hmmdmstr_shard", "hmmd_search_status",
        "hmmdwrkr", "hmmdwrkr_shard", "hmmdutils", "hmmer", "logsum",
        "modelconfig", "modelstats", "mpisupport", "seqmodel",
        "tracealign", "p7_alidisplay", "p7_bg", "p7_builder",
        "p7_domain", "p7_domaindef", "p7_gbands", "p7_gmx", "p7_gmxb",
        "p7_gmxchk", "p7_hit", "p7_hmm", "p7_hmmcache",
        "p7_hmmd_search_stats", "p7_hmmfile", "p7_hmmwindow",
        "p7_null3",
        "p7_pipeline", "p7_prior", "p7_profile", "p7_spensemble",
        "p7_tophits", "p7_trace", "p7_scoredata", "hmmpgmd2msa",
        "fm_alphabet", "fm_general", "fm_sse", "fm_ssv",
    ];

    let mut hmmer_build = cc::Build::new();
    common_flags(&mut hmmer_build);
    hmmer_build
        .include(&easel_dir)
        .include(&src_dir)
        .include(&impl_sse_dir)
        .include(&divsufsort_dir);
    for src in &hmmer_lib_sources {
        hmmer_build.file(src_dir.join(format!("{}.c", src)));
    }
    hmmer_build.compile("hmmer");

    // --- Compile program files with main() renamed ---
    // Each program's main() is renamed to <program>_main() so we can call it from Rust.
    let programs = [
        "hmmsearch", "hmmbuild", "phmmer", "nhmmer", "hmmscan",
        "hmmalign", "hmmpress", "hmmfetch", "hmmemit", "hmmconvert",
        "hmmstat", "jackhmmer", "nhmmscan", "makehmmerdb", "alimask",
        "hmmpgmd", "hmmpgmd_shard", "hmmsim",
    ];

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    for prog in &programs {
        // Create a wrapper .c file that renames main
        let wrapper_path = out_dir.join(format!("{}_wrapper.c", prog));
        let wrapper_content = format!(
            "#define main {}_main\n#include \"{}\"\n",
            prog,
            src_dir.join(format!("{}.c", prog)).canonicalize()
                .unwrap_or_else(|_| src_dir.join(format!("{}.c", prog)))
                .display()
        );
        std::fs::write(&wrapper_path, &wrapper_content).unwrap();

        let mut prog_build = cc::Build::new();
        common_flags(&mut prog_build);
        prog_build
            .include(&easel_dir)
            .include(&src_dir)
            .include(&impl_sse_dir)
            .include(&divsufsort_dir)
            .file(&wrapper_path);
        prog_build.compile(&format!("hmmer_prog_{}", prog));
    }

    // --- Link flags ---
    println!("cargo:rustc-link-lib=pthread");
    println!("cargo:rustc-link-lib=m");

    // --- Generate bindings ---
    let bindings = bindgen::Builder::default()
        .header("hmmer/src/hmmer.h")
        .header("hmmer/src/impl_sse/impl_sse.h")
        .header("hmmer/easel/esl_gumbel.h")
        .header("hmmer/easel/esl_exponential.h")
        .clang_args(&[
            "-I", easel_dir.to_str().unwrap(),
            "-I", src_dir.to_str().unwrap(),
            "-I", impl_sse_dir.to_str().unwrap(),
            "-I", divsufsort_dir.to_str().unwrap(),
            "-DHAVE_CONFIG_H",
            "-DeslENABLE_SSE",
            "-DHMMER_THREADS",
        ])
        .allowlist_function("p7_.*")
        .allowlist_function("esl_.*")
        .allowlist_function("fm_.*")
        .allowlist_function("hmmsearch_main")
        .allowlist_function("hmmbuild_main")
        .allowlist_function("phmmer_main")
        .allowlist_function("nhmmer_main")
        .allowlist_function("hmmscan_main")
        .allowlist_function("hmmalign_main")
        .allowlist_function("hmmpress_main")
        .allowlist_function("hmmfetch_main")
        .allowlist_function("hmmemit_main")
        .allowlist_function("hmmconvert_main")
        .allowlist_function("hmmstat_main")
        .allowlist_function("jackhmmer_main")
        .allowlist_function("nhmmscan_main")
        .allowlist_function("makehmmerdb_main")
        .allowlist_function("alimask_main")
        .allowlist_type("P7_.*")
        .allowlist_type("ESL_.*")
        .allowlist_type("FM_.*")
        .allowlist_var("p7_.*")
        .allowlist_var("esl.*")
        .allowlist_var("P7_.*")
        .allowlist_var("ESL_.*")
        .allowlist_var("FM_.*")
        .derive_debug(true)
        .derive_default(true)
        .generate()
        .expect("Unable to generate bindings");

    bindings
        .write_to_file(out_dir.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
