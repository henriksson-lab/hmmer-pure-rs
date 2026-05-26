//! hmmpgmd — HMMER search daemon.
//! Listens for search requests over TCP.

use std::fs::File;
use std::io::{BufRead, BufReader, Cursor, Read, Write};
use std::net::{TcpListener, TcpStream};
#[cfg(unix)]
use std::os::fd::FromRawFd;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use clap::Parser;

use hmmer_pure_rs::alphabet::{Alphabet, AlphabetType};
use hmmer_pure_rs::bg::Bg;
use hmmer_pure_rs::hmm::{Hmm, P7H_CA, P7H_CONS, P7H_CS, P7H_MAP, P7H_MMASK, P7H_RF};
use hmmer_pure_rs::hmmfile;
use hmmer_pure_rs::logsum;
use hmmer_pure_rs::pipeline::{BitCutoff, Pipeline};
use hmmer_pure_rs::profile::{self, Profile, P7_LOCAL};
use hmmer_pure_rs::seqmodel;
use hmmer_pure_rs::sequence::{self, Sequence};
use hmmer_pure_rs::simd::oprofile::OProfile;
use hmmer_pure_rs::tophits::{AliDisplay, Domain, Hit, TopHits};

#[derive(Parser)]
#[command(name = "hmmpgmd", about = "HMMER search daemon")]
struct Args {
    /// Run as the master server
    #[arg(long = "master", conflicts_with = "worker")]
    master: bool,

    /// Run as a worker connected to master host
    #[arg(long = "worker", value_name = "HOST", conflicts_with = "master")]
    worker: Option<String>,

    /// HMM database file to serve
    #[arg(long = "hmmdb", conflicts_with = "seqdb")]
    hmmdb: Option<PathBuf>,

    /// Protein sequence database file to serve
    #[arg(long = "seqdb", conflicts_with = "hmmdb")]
    seqdb: Option<PathBuf>,

    /// Legacy alias for --cport
    #[arg(long = "port")]
    port: Option<u16>,

    /// Port to use for client/server communication
    #[arg(long = "cport", default_value = "51371", conflicts_with = "worker")]
    cport: u16,

    /// Port to use for server/worker communication
    #[arg(long = "wport", default_value = "51372")]
    wport: u16,

    /// File to write the process id to
    #[arg(long = "pid")]
    pid: Option<PathBuf>,

    /// Maximum client-side listen backlog for master mode
    #[arg(long = "ccncts", default_value = "16", value_parser = parse_positive_usize, conflicts_with = "worker")]
    ccncts: usize,

    /// Maximum worker-side listen backlog for master mode
    #[arg(long = "wcncts", default_value = "32", value_parser = parse_positive_usize, conflicts_with = "worker")]
    wcncts: usize,

    /// Number of parallel worker CPU threads
    #[arg(long = "cpu", default_value = "2", value_parser = parse_positive_usize, conflicts_with = "master")]
    cpu: usize,
}

impl Args {
    fn client_port(&self) -> u16 {
        self.port.unwrap_or(self.cport)
    }
}

fn parse_positive_usize(s: &str) -> Result<usize, String> {
    let value = s
        .parse::<usize>()
        .map_err(|e| format!("invalid positive integer: {e}"))?;
    if value > 0 {
        Ok(value)
    } else {
        Err("value must be > 0".to_string())
    }
}

const HMMD_CMD_SEARCH: u32 = 10001;
const HMMD_CMD_SCAN: u32 = 10002;
const HMMD_CMD_INIT: u32 = 10003;
const HMMD_CMD_SHUTDOWN: u32 = 10004;
const HMMD_HEADER_SIZE: usize = 12;
const HMMD_INIT_BODY_SIZE: usize = 88;
const HMMD_INIT_RESPONSE_SIZE: usize = 96;
const HMMD_SEARCH_CMD_FIXED_SIZE: usize = 28;
const C_HMMD_COMMAND_SEARCH_PADDING_SIZE: usize = 60;
const HMMD_SEARCH_STATS_SIZE: usize = 122;
const C_P7_HMM_SHELL_SIZE: usize = 296;
const P7_HIT_BASE_SIZE: usize = 109;
const P7_DOMAIN_BASE_SIZE: usize = 92;
const P7_ALIDISPLAY_BASE_SIZE: usize = 45;
const HMMD_SEQUENCE: u32 = 101;
const HMMD_HMM: u32 = 102;
const P7_HIT_ACC_PRESENT: u8 = 1 << 0;
const P7_HIT_DESC_PRESENT: u8 = 1 << 1;
const P7_ALIDISPLAY_RFLINE_PRESENT: u8 = 1 << 0;
const P7_ALIDISPLAY_MMLINE_PRESENT: u8 = 1 << 1;
const P7_ALIDISPLAY_CSLINE_PRESENT: u8 = 1 << 2;
const P7_ALIDISPLAY_PPLINE_PRESENT: u8 = 1 << 3;
const P7_ALIDISPLAY_ASEQ_PRESENT: u8 = 1 << 4;
const P7_ALIDISPLAY_NTSEQ_PRESENT: u8 = 1 << 5;

#[derive(Clone, Copy, Debug)]
struct HmmdHeader {
    length: u32,
    command: u32,
    status: u32,
}

#[derive(Debug)]
struct ClientCommand {
    db_mode: DbMode,
    query: ClientQuery,
    options: SearchOptions,
    seqdb_ranges: Option<Vec<DbRange>>,
    db_slice: Option<DbSlice>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DbMode {
    Hmm,
    Seq,
}

#[derive(Debug)]
enum ClientQuery {
    Sequence {
        name: String,
        desc: String,
        seq: String,
    },
    Hmm(Hmm),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DbRange {
    start: usize,
    end: usize,
}

#[derive(Clone, Copy, Debug)]
struct DbSlice {
    start: usize,
    count: usize,
}

#[derive(Clone, Copy, Debug)]
struct SearchOptions {
    e_value_threshold: f64,
    dom_e_value_threshold: f64,
    inc_e: f64,
    inc_dome: f64,
    t: Option<f32>,
    dom_t: Option<f32>,
    inc_t: Option<f32>,
    inc_dom_t: Option<f32>,
    bit_cutoff: BitCutoff,
    max: bool,
    f1: f64,
    f2: f64,
    f3: f64,
    nobias: bool,
    nonull2: bool,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            e_value_threshold: 10.0,
            dom_e_value_threshold: 10.0,
            inc_e: 0.01,
            inc_dome: 0.01,
            t: None,
            dom_t: None,
            inc_t: None,
            inc_dom_t: None,
            bit_cutoff: BitCutoff::None,
            max: false,
            f1: 0.02,
            f2: 1e-3,
            f3: 1e-5,
            nobias: false,
            nonull2: false,
        }
    }
}

#[derive(Debug)]
struct SearchHit {
    name: String,
    acc: String,
    desc: String,
    window_length: i32,
    sortkey: f64,
    score: f32,
    pre_score: f32,
    sum_score: f32,
    lnp: f64,
    pre_lnp: f64,
    sum_lnp: f64,
    nexpected: f32,
    nregions: i32,
    nclustered: i32,
    noverlaps: i32,
    nenvelopes: i32,
    flags: u32,
    nreported: i32,
    nincluded: i32,
    best_domain: i32,
    seqidx: i64,
    subseq_start: i64,
    hmm_name: String,
    hmm_acc: String,
    hmm_desc: String,
    seq_name: String,
    seq_acc: String,
    seq_desc: String,
    model_length: i32,
    sequence_length: i64,
    domains: Vec<Domain>,
}

enum ServerState {
    Hmm(HmmDbState),
    Seq(SeqDbState),
}

type WorkerPool = Arc<Mutex<Vec<WorkerConnection>>>;

struct WorkerConnection {
    stream: TcpStream,
}

#[derive(Clone, Copy, Default)]
struct SearchRuntime<'a> {
    thread_pool: Option<&'a rayon::ThreadPool>,
}

impl SearchRuntime<'_> {
    fn is_parallel(self) -> bool {
        self.thread_pool.is_some()
    }
}

struct HmmDbState {
    path: PathBuf,
    hmms: Vec<Hmm>,
    abc: Alphabet,
    bg: Bg,
    profiles: Vec<(Profile, OProfile)>,
}

struct SeqDbState {
    path: PathBuf,
    cache_id: String,
    sequences: Vec<Sequence>,
    abc: Alphabet,
    bg: Bg,
}

/// Entry point for `hmmpgmd`: HMMER search daemon.
///
/// Supports the C daemon's public master/worker flags and wire framing for a
/// compatible subset. The master accepts C-style client command blocks:
/// `@--seqdb 1` or `@--hmmdb 1`, followed by a FASTA query and `//`; `--seqdb`
/// requests may also send one ASCII HMM query record. It replies with
/// `HMMD_SEARCH_STATUS`, serialized `HMMD_SEARCH_STATS`, and C-shaped `P7_HIT`
/// records for sequence-level hits with `P7_DOMAIN`/`P7_ALIDISPLAY` payload
/// shells. Workers can optionally load the same served database and execute
/// framed `HMMD_CMD_SEARCH`/`HMMD_CMD_SCAN` requests, returning the same
/// C-shaped search payload as the master. The master keeps initialized workers
/// in a pool, schedules C-framed sequence-database client searches to them as
/// non-overlapping `--seqdb_ranges` shards, and merges their serialized hit
/// payloads. HMM-database scans are scheduled with C's binary
/// `HMMD_SEARCH_CMD` `(inx,cnt)` body prefix. Sequence queries inside that
/// binary command use C's `name\0desc\0dsq[L+2]` object layout; HMM queries use
/// a C-shaped `P7_HMM` shell followed by transition/emission arrays and
/// optional strings/annotation blocks.
/// The legacy one-line sequence protocol is retained for existing tests and
/// simple manual use.
pub fn run(args: Vec<String>) -> ExitCode {
    let args = Args::parse_from(&args);

    if let Some(pid) = &args.pid {
        if let Err(e) = std::fs::write(pid, format!("{}\n", std::process::id())) {
            eprintln!(
                "Unable to open PID file {} for writing: {}",
                pid.display(),
                e
            );
            std::process::exit(1);
        }
    }

    if let Some(host) = &args.worker {
        let state = match (&args.hmmdb, &args.seqdb) {
            (Some(hmmdb), None) => Some(load_hmmdb(hmmdb)),
            (None, Some(seqdb)) => Some(load_seqdb(seqdb)),
            (None, None) => None,
            (Some(_), Some(_)) => unreachable!("clap rejects conflicting served databases"),
        };
        return run_worker(host, args.wport, args.cpu, state);
    }

    if !args.master {
        eprintln!("hmmpgmd compatibility mode: treating invocation as --master");
    }

    match (&args.hmmdb, &args.seqdb) {
        (None, None) => {
            eprintln!("hmmpgmd requires --hmmdb or --seqdb");
            std::process::exit(1);
        }
        (Some(_), Some(_)) => {
            eprintln!("hmmpgmd accepts one served database at a time: use --hmmdb or --seqdb");
            std::process::exit(1);
        }
        (Some(hmmdb), None) => run_master(
            load_hmmdb(hmmdb),
            args.client_port(),
            args.wport,
            args.ccncts,
            args.wcncts,
        ),
        (None, Some(seqdb)) => run_master(
            load_seqdb(seqdb),
            args.client_port(),
            args.wport,
            args.ccncts,
            args.wcncts,
        ),
    }
}

fn load_hmmdb(hmmdb: &PathBuf) -> ServerState {
    logsum::p7_flogsuminit();

    eprintln!("Loading HMM database: {}", hmmdb.display());
    let hmms = hmmfile::read_hmm_file_auto(hmmdb).unwrap_or_else(|e| {
        eprintln!("Error loading HMMs: {}", e);
        std::process::exit(1);
    });
    eprintln!("Loaded {} HMMs", hmms.len());

    let abc = Alphabet::amino();
    let bg = Bg::new(&abc);

    // Pre-build profiles
    let profiles: Vec<(Profile, OProfile)> = hmms
        .iter()
        .map(|hmm| {
            let mut gm = Profile::new(hmm.m, &abc);
            profile::profile_config(hmm, &bg, &mut gm, 400, P7_LOCAL);
            let om = OProfile::convert(&gm);
            (gm, om)
        })
        .collect();
    eprintln!("Profiles built");

    ServerState::Hmm(HmmDbState {
        path: hmmdb.clone(),
        hmms,
        abc,
        bg,
        profiles,
    })
}

fn load_seqdb(seqdb: &PathBuf) -> ServerState {
    logsum::p7_flogsuminit();

    let abc = Alphabet::amino();
    let bg = Bg::new(&abc);

    eprintln!("Loading sequence database: {}", seqdb.display());
    let mut sqf = sequence::SeqFile::new(
        File::open(seqdb).unwrap_or_else(|e| {
            eprintln!("Error opening sequence database: {}", e);
            std::process::exit(1);
        }),
        abc.clone(),
    )
    .with_fasta_only();
    let mut sequences = Vec::new();
    let mut sq = Sequence::new();
    while sqf.read(&mut sq).unwrap_or_else(|e| {
        eprintln!("Error reading sequence database: {}", e);
        std::process::exit(1);
    }) {
        sequences.push(sq.clone());
        sq.reuse();
    }
    if sequences.is_empty() {
        eprintln!("Error: no sequences found in {}", seqdb.display());
        std::process::exit(1);
    }
    eprintln!("Loaded {} sequences", sequences.len());

    ServerState::Seq(SeqDbState {
        path: seqdb.clone(),
        cache_id: seqdb_cache_id(seqdb),
        sequences,
        abc,
        bg,
    })
}

fn seqdb_cache_id(seqdb: &PathBuf) -> String {
    let Ok(file) = File::open(seqdb) else {
        return String::new();
    };
    let mut reader = BufReader::new(file);
    let mut line = String::new();
    if reader
        .read_line(&mut line)
        .ok()
        .filter(|n| *n > 0)
        .is_none()
    {
        return String::new();
    }
    let Some(rest) = line.strip_prefix('#') else {
        return String::new();
    };

    let mut fields = rest.split_whitespace();
    let _res_count = fields.next();
    let _seq_count = fields.next();
    let db_count = fields
        .next()
        .and_then(|field| field.parse::<usize>().ok())
        .unwrap_or(0);
    for _ in 0..db_count.saturating_mul(2) {
        let _ = fields.next();
    }
    fields.collect::<Vec<_>>().join(" ")
}

fn run_master(
    state: ServerState,
    cport: u16,
    wport: u16,
    ccncts: usize,
    wcncts: usize,
) -> ExitCode {
    let worker_listener = bind_listener("0.0.0.0", wport, "worker", wcncts);
    let worker_pool = Arc::new(Mutex::new(Vec::new()));
    let worker_init_body = Arc::new(master_init_body(&state));
    thread::spawn({
        let worker_pool = Arc::clone(&worker_pool);
        let worker_init_body = Arc::clone(&worker_init_body);
        move || accept_workers(worker_listener, worker_pool, worker_init_body)
    });

    let listener = bind_listener("0.0.0.0", cport, "client", ccncts);

    for stream in listener.incoming() {
        match stream {
            Ok(mut stream) => match handle_client(&state, &worker_pool, &mut stream) {
                Ok(true) => {}
                Ok(false) => break,
                Err(e) => eprintln!("Connection error: {}", e),
            },
            Err(e) => eprintln!("Connection error: {}", e),
        }
    }

    ExitCode::SUCCESS
}

fn handle_client(
    state: &ServerState,
    worker_pool: &WorkerPool,
    stream: &mut TcpStream,
) -> std::io::Result<bool> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let request = read_client_request_from_reader(&mut reader)?;
    if request.trim().is_empty() || request.trim() == "QUIT" {
        writeln!(stream, "BYE")?;
        stream.flush()?;
        return Ok(true);
    }

    if request.trim_start().starts_with('!') {
        return handle_control_client(worker_pool, stream, reader, request);
    }

    if request.trim_start().starts_with('@') {
        match parse_c_client_command(&request) {
            Ok(command) => {
                let db_ok = matches!(
                    (state, command.db_mode),
                    (ServerState::Hmm(_), DbMode::Hmm) | (ServerState::Seq(_), DbMode::Seq)
                );
                if !db_ok {
                    write_c_error(stream, "Requested database is not loaded")?;
                    stream.flush()?;
                    return Ok(true);
                }
                let (nmodels, nseqs) = searched_counts(state, &command.query);
                let payload =
                    match run_worker_pool_search(worker_pool, &request, &command, nmodels, nseqs) {
                        Ok(Some(payload)) => payload,
                        Ok(None) => {
                            let mut results = run_query(state, &command, SearchRuntime::default())?;
                            serialize_c_search_payload(nmodels, nseqs, &mut results)
                        }
                        Err(e) => {
                            write_c_error(stream, &e.to_string())?;
                            stream.flush()?;
                            return Ok(true);
                        }
                    };
                write_c_status(stream, 0, payload.len() as u64)?;
                stream.write_all(&payload)?;
            }
            Err(e) => write_c_error(stream, &e)?,
        }
    } else {
        let query = ClientQuery::Sequence {
            name: "query".to_string(),
            desc: String::new(),
            seq: request.trim().to_string(),
        };
        let command = ClientCommand {
            db_mode: match state {
                ServerState::Hmm(_) => DbMode::Hmm,
                ServerState::Seq(_) => DbMode::Seq,
            },
            query,
            options: SearchOptions::default(),
            seqdb_ranges: None,
            db_slice: None,
        };
        let mut results = run_query(state, &command, SearchRuntime::default())?;
        write_hit_response(stream, &mut results);
    }

    stream.flush()?;
    Ok(true)
}

fn handle_control_client(
    worker_pool: &WorkerPool,
    stream: &mut TcpStream,
    mut reader: BufReader<TcpStream>,
    first_request: String,
) -> std::io::Result<bool> {
    let mut request = first_request;
    loop {
        if request.trim_start().starts_with("!shutdown") {
            shutdown_worker_pool(worker_pool)?;
            write_c_status(stream, 0, HMMD_SEARCH_STATS_SIZE as u64)?;
            write_c_stats(stream, 0, 0, 0)?;
            stream.flush()?;
            return Ok(false);
        }

        write_c_error(stream, "Unknown server command")?;
        stream.flush()?;

        let previous_timeout = stream.read_timeout()?;
        stream.set_read_timeout(Some(Duration::from_millis(50)))?;
        let next_request = match read_client_request_from_reader(&mut reader) {
            Ok(next_request) => next_request,
            Err(e)
                if matches!(
                    e.kind(),
                    std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                ) =>
            {
                stream.set_read_timeout(previous_timeout)?;
                return Ok(true);
            }
            Err(e) => {
                stream.set_read_timeout(previous_timeout)?;
                return Err(e);
            }
        };
        stream.set_read_timeout(previous_timeout)?;

        if next_request.trim().is_empty() {
            return Ok(true);
        }
        if !next_request.trim_start().starts_with('!') {
            write_c_error(stream, "Expected server command after control command")?;
            stream.flush()?;
            return Ok(true);
        }
        request = next_request;
    }
}

fn run_query(
    state: &ServerState,
    command: &ClientCommand,
    runtime: SearchRuntime<'_>,
) -> std::io::Result<Vec<SearchHit>> {
    match (state, &command.query) {
        (ServerState::Hmm(db), ClientQuery::Sequence { name, desc, seq }) => {
            run_hmmdb_sequence_query(
                db,
                name,
                desc,
                seq,
                &command.options,
                command.db_slice,
                runtime,
            )
        }
        (ServerState::Hmm(_), ClientQuery::Hmm(_)) => {
            Err("A HMM cannot be used to search a hmm database".to_string())
        }
        (ServerState::Seq(db), ClientQuery::Sequence { name, seq, .. }) => {
            run_seqdb_sequence_query(
                db,
                name,
                seq,
                &command.options,
                command.seqdb_ranges.as_deref(),
                runtime,
            )
        }
        (ServerState::Seq(db), ClientQuery::Hmm(hmm)) => run_seqdb_hmm_query(
            db,
            hmm,
            &command.options,
            command.seqdb_ranges.as_deref(),
            runtime,
        ),
    }
    .map_err(std::io::Error::other)
}

fn run_hmmdb_sequence_query(
    db: &HmmDbState,
    query_name: &str,
    query_desc: &str,
    seq_text: &str,
    options: &SearchOptions,
    db_slice: Option<DbSlice>,
    runtime: SearchRuntime<'_>,
) -> Result<Vec<SearchHit>, String> {
    let dsq = db
        .abc
        .digitize_checked(seq_text.as_bytes())
        .map_err(|e| e.to_string())?;
    let l = dsq.len() - 2;

    if l == 0 {
        return Err("empty sequence".to_string());
    }

    if runtime.is_parallel() {
        use rayon::prelude::*;
        let partials = runtime.thread_pool.unwrap().install(|| {
            db.profiles
                .par_iter()
                .enumerate()
                .map(|(i, (gm, om))| {
                    if !db_index_is_in_slice(i, db_slice) {
                        return Ok(Vec::new());
                    }
                    score_hmmdb_sequence_model(
                        db, i, gm, om, query_name, query_desc, &dsq, l, options,
                    )
                })
                .collect::<Vec<_>>()
        });
        return flatten_search_results(partials);
    }

    let mut results = Vec::new();
    for (i, (gm, om)) in db.profiles.iter().enumerate() {
        if !db_index_is_in_slice(i, db_slice) {
            continue;
        }
        results.extend(score_hmmdb_sequence_model(
            db, i, gm, om, query_name, query_desc, &dsq, l, options,
        )?);
    }

    Ok(results)
}

fn score_hmmdb_sequence_model(
    db: &HmmDbState,
    index: usize,
    gm: &Profile,
    om: &OProfile,
    query_name: &str,
    query_desc: &str,
    dsq: &[u8],
    l: usize,
    options: &SearchOptions,
) -> Result<Vec<SearchHit>, String> {
    let mut local_bg = db.bg.clone();
    local_bg.set_length(l);
    let mut local_gm = gm.clone();
    let mut local_om = om.clone();
    let sq = Sequence {
        name: query_name.to_string(),
        acc: String::new(),
        desc: query_desc.to_string(),
        dsq: dsq.to_vec(),
        n: l,
        l,
        taxid: -1,
    };

    let mut pli = Pipeline::new();
    configure_hmmpgmd_pipeline(&mut pli, options, &db.hmms[index])?;
    pli.new_model(&local_gm);
    let mut th = TopHits::new();
    if !pli.run(
        &mut local_gm,
        &mut local_om,
        &local_bg,
        &db.hmms[index],
        &sq,
        &mut th,
    ) {
        return Ok(Vec::new());
    }

    Ok(th
        .hits
        .iter()
        .map(|hit| {
            SearchHit::from_pipeline_hit(
                hit,
                db.hmms[index].name.clone(),
                &db.hmms[index],
                &format!("{:09}", index + 1),
                query_name,
                "",
                query_desc,
                l,
            )
        })
        .collect())
}

fn db_index_is_in_slice(index: usize, db_slice: Option<DbSlice>) -> bool {
    db_slice
        .map(|slice| index >= slice.start && index < slice.start.saturating_add(slice.count))
        .unwrap_or(true)
}

fn run_seqdb_sequence_query(
    db: &SeqDbState,
    query_name: &str,
    seq_text: &str,
    options: &SearchOptions,
    seqdb_ranges: Option<&[DbRange]>,
    runtime: SearchRuntime<'_>,
) -> Result<Vec<SearchHit>, String> {
    let dsq = db
        .abc
        .digitize_checked(seq_text.as_bytes())
        .map_err(|e| e.to_string())?;
    let l = dsq.len() - 2;
    if l == 0 {
        return Err("empty sequence".to_string());
    }

    let hmm = seqmodel::build_single_seq_hmm(query_name, &dsq, l, &db.abc, &db.bg, 0.02, 0.4);
    let mut model_bg = db.bg.clone();
    model_bg.set_filter(hmm.m, &hmm.compo);

    let mut gm = Profile::new(hmm.m, &db.abc);
    profile::profile_config(&hmm, &model_bg, &mut gm, 400, P7_LOCAL);
    let om = OProfile::convert(&gm);

    if runtime.is_parallel() {
        use rayon::prelude::*;
        let partials = runtime.thread_pool.unwrap().install(|| {
            db.sequences
                .par_iter()
                .enumerate()
                .map(|(target_idx, target)| {
                    if !db_index_is_in_ranges(target_idx + 1, seqdb_ranges) {
                        return Ok(Vec::new());
                    }
                    score_seqdb_target(db, &hmm, &model_bg, &gm, &om, target, options)
                })
                .collect::<Vec<_>>()
        });
        return flatten_search_results(partials);
    }

    let mut results = Vec::new();
    for (target_idx, target) in db.sequences.iter().enumerate() {
        if !db_index_is_in_ranges(target_idx + 1, seqdb_ranges) {
            continue;
        }
        results.extend(score_seqdb_target(
            db, &hmm, &model_bg, &gm, &om, target, options,
        )?);
    }

    Ok(results)
}

fn run_seqdb_hmm_query(
    db: &SeqDbState,
    hmm: &Hmm,
    options: &SearchOptions,
    seqdb_ranges: Option<&[DbRange]>,
    runtime: SearchRuntime<'_>,
) -> Result<Vec<SearchHit>, String> {
    if hmm.abc_type != AlphabetType::Amino {
        return Err("Only amino HMM queries can search a protein sequence database".to_string());
    }

    let mut model_bg = db.bg.clone();
    model_bg.set_filter(hmm.m, &hmm.compo);

    let mut gm = Profile::new(hmm.m, &db.abc);
    profile::profile_config(hmm, &model_bg, &mut gm, 400, P7_LOCAL);
    let om = OProfile::convert(&gm);

    if runtime.is_parallel() {
        use rayon::prelude::*;
        let partials = runtime.thread_pool.unwrap().install(|| {
            db.sequences
                .par_iter()
                .enumerate()
                .map(|(target_idx, target)| {
                    if !db_index_is_in_ranges(target_idx + 1, seqdb_ranges) {
                        return Ok(Vec::new());
                    }
                    score_seqdb_target(db, hmm, &model_bg, &gm, &om, target, options)
                })
                .collect::<Vec<_>>()
        });
        return flatten_search_results(partials);
    }

    let mut results = Vec::new();
    for (target_idx, target) in db.sequences.iter().enumerate() {
        if !db_index_is_in_ranges(target_idx + 1, seqdb_ranges) {
            continue;
        }
        results.extend(score_seqdb_target(
            db, hmm, &model_bg, &gm, &om, target, options,
        )?);
    }

    Ok(results)
}

fn score_seqdb_target(
    _db: &SeqDbState,
    hmm: &Hmm,
    model_bg: &Bg,
    gm: &Profile,
    om: &OProfile,
    target: &Sequence,
    options: &SearchOptions,
) -> Result<Vec<SearchHit>, String> {
    let mut local_bg = model_bg.clone();
    let mut local_gm = gm.clone();
    let mut local_om = om.clone();
    let mut pli = Pipeline::new();
    configure_hmmpgmd_pipeline(&mut pli, options, hmm)?;
    pli.new_model(&local_gm);
    local_bg.set_length(target.n);

    let mut th = TopHits::new();
    if !pli.run(
        &mut local_gm,
        &mut local_om,
        &local_bg,
        hmm,
        target,
        &mut th,
    ) {
        return Ok(Vec::new());
    }

    Ok(th
        .hits
        .iter()
        .map(|hit| {
            SearchHit::from_pipeline_hit(
                hit,
                target.name.clone(),
                hmm,
                &hmm.name,
                &target.name,
                &target.acc,
                &target.desc,
                target.n,
            )
        })
        .collect())
}

fn flatten_search_results(
    partials: Vec<Result<Vec<SearchHit>, String>>,
) -> Result<Vec<SearchHit>, String> {
    let mut results = Vec::new();
    for partial in partials {
        results.extend(partial?);
    }
    Ok(results)
}

fn configure_hmmpgmd_pipeline(
    pli: &mut Pipeline,
    options: &SearchOptions,
    hmm: &Hmm,
) -> Result<(), String> {
    pli.e_value_threshold = options.e_value_threshold;
    pli.dom_e_value_threshold = options.dom_e_value_threshold;
    pli.inc_e = options.inc_e;
    pli.inc_dome = options.inc_dome;
    pli.f1 = if options.max { 1.0 } else { options.f1 };
    pli.f2 = if options.max { 1.0 } else { options.f2 };
    pli.f3 = if options.max { 1.0 } else { options.f3 };
    pli.do_max = options.max;
    pli.do_biasfilter = !(options.max || options.nobias);
    pli.do_null2 = !options.nonull2;

    if let Some(t) = options.t {
        pli.t = Some(t);
        pli.by_e = false;
    }
    if let Some(t) = options.dom_t {
        pli.dom_t = Some(t);
        pli.dom_by_e = false;
    }
    if let Some(t) = options.inc_t {
        pli.inc_t = Some(t);
        pli.inc_by_e = false;
    }
    if let Some(t) = options.inc_dom_t {
        pli.inc_dom_t = Some(t);
        pli.incdom_by_e = false;
    }

    pli.use_bit_cutoffs = options.bit_cutoff;
    if pli.use_bit_cutoffs != BitCutoff::None {
        pli.new_model_thresholds(&hmm.cutoff)?;
    }
    Ok(())
}

fn db_index_is_in_ranges(index: usize, ranges: Option<&[DbRange]>) -> bool {
    ranges
        .map(|ranges| {
            ranges
                .iter()
                .any(|range| index >= range.start && index <= range.end)
        })
        .unwrap_or(true)
}

fn searched_counts(state: &ServerState, query: &ClientQuery) -> (u64, u64) {
    match (state, query) {
        (ServerState::Hmm(db), ClientQuery::Sequence { .. }) => (db.hmms.len() as u64, 1),
        (ServerState::Hmm(db), ClientQuery::Hmm(_)) => (db.hmms.len() as u64, 0),
        (ServerState::Seq(db), ClientQuery::Sequence { .. })
        | (ServerState::Seq(db), ClientQuery::Hmm(_)) => (1, db.sequences.len() as u64),
    }
}

#[cfg(unix)]
fn bind_listener(host: &str, port: u16, role: &str, backlog: usize) -> TcpListener {
    let addr = format!("{}:{}", host, port);
    let ip = host.parse::<std::net::Ipv4Addr>().unwrap_or_else(|e| {
        eprintln!("Cannot parse bind address {}: {}", host, e);
        std::process::exit(1);
    });
    let backlog = backlog.min(i32::MAX as usize) as libc::c_int;
    unsafe {
        let fd = libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0);
        if fd < 0 {
            eprintln!(
                "Cannot create socket for {}: {}",
                addr,
                std::io::Error::last_os_error()
            );
            std::process::exit(1);
        }

        let yes: libc::c_int = 1;
        let _ = libc::setsockopt(
            fd,
            libc::SOL_SOCKET,
            libc::SO_REUSEADDR,
            (&yes as *const libc::c_int).cast(),
            std::mem::size_of_val(&yes) as libc::socklen_t,
        );

        let sockaddr = libc::sockaddr_in {
            sin_family: libc::AF_INET as libc::sa_family_t,
            sin_port: port.to_be(),
            sin_addr: libc::in_addr {
                s_addr: u32::from(ip).to_be(),
            },
            sin_zero: [0; 8],
        };

        if libc::bind(
            fd,
            (&sockaddr as *const libc::sockaddr_in).cast(),
            std::mem::size_of_val(&sockaddr) as libc::socklen_t,
        ) != 0
        {
            let e = std::io::Error::last_os_error();
            let _ = libc::close(fd);
            eprintln!("Cannot bind to {}: {}", addr, e);
            std::process::exit(1);
        }

        if libc::listen(fd, backlog) != 0 {
            let e = std::io::Error::last_os_error();
            let _ = libc::close(fd);
            eprintln!("Cannot listen on {}: {}", addr, e);
            std::process::exit(1);
        }

        let listener = TcpListener::from_raw_fd(fd);
        eprintln!("Listening for {role} connections on {addr} (backlog {backlog})");
        listener
    }
}

#[cfg(not(unix))]
fn bind_listener(host: &str, port: u16, role: &str, backlog: usize) -> TcpListener {
    let addr = format!("{}:{}", host, port);
    let listener = TcpListener::bind(&addr).unwrap_or_else(|e| {
        eprintln!("Cannot bind to {}: {}", addr, e);
        std::process::exit(1);
    });
    eprintln!(
        "Listening for {role} connections on {addr} (requested backlog {backlog}; std::net fallback)"
    );
    listener
}

fn read_client_request_from_reader(reader: &mut impl BufRead) -> std::io::Result<String> {
    let mut first = String::new();
    reader.read_line(&mut first)?;
    if first.trim_start().starts_with('@') || first.trim_start().starts_with('!') {
        let mut request = first;
        loop {
            let mut line = String::new();
            if reader.read_line(&mut line)? == 0 {
                break;
            }
            let done = line.trim_end() == "//";
            request.push_str(&line);
            if done {
                break;
            }
        }
        Ok(request)
    } else {
        Ok(first.trim().to_string())
    }
}

fn write_hit_response(stream: &mut std::net::TcpStream, results: &mut Vec<SearchHit>) {
    sort_search_hits(results);
    let _ = writeln!(stream, "HITS {}", results.len());
    for hit in results.iter() {
        let _ = writeln!(
            stream,
            "{}\t{}",
            hit.name,
            hmmer_pure_rs::output::fmt_score(hit.score)
        );
    }
    let _ = writeln!(stream, "//");
    let _ = stream.flush();
}

fn sort_search_hits(results: &mut [SearchHit]) {
    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

fn run_worker_pool_search(
    worker_pool: &WorkerPool,
    request: &str,
    command: &ClientCommand,
    total_nmodels: u64,
    total_nseqs: u64,
) -> std::io::Result<Option<Vec<u8>>> {
    let active_workers = {
        let mut workers = worker_pool
            .lock()
            .map_err(|_| std::io::Error::other("worker pool lock poisoned"))?;
        if workers.is_empty() {
            return Ok(None);
        }
        std::mem::take(&mut *workers)
    };

    let sharded_requests = worker_sharded_requests(
        request,
        command,
        active_workers.len(),
        total_nmodels,
        total_nseqs,
    );
    let (partials, mut surviving_workers) =
        run_worker_connections_search(active_workers, &sharded_requests);
    let complete = partials.len() == sharded_requests.len();

    if !surviving_workers.is_empty() {
        let mut workers = worker_pool
            .lock()
            .map_err(|_| std::io::Error::other("worker pool lock poisoned"))?;
        workers.append(&mut surviving_workers);
    }

    if complete {
        return merge_worker_payloads(&partials, total_nmodels, total_nseqs).map(Some);
    }

    if !partials.is_empty() {
        eprintln!(
            "Discarding incomplete worker search result: received {} of {} shards",
            partials.len(),
            sharded_requests.len()
        );
    }

    {
        let active_workers = {
            let mut workers = worker_pool
                .lock()
                .map_err(|_| std::io::Error::other("worker pool lock poisoned"))?;
            if workers.is_empty() {
                return Ok(None);
            }
            std::mem::take(&mut *workers)
        };

        let sharded_requests = worker_sharded_requests(
            request,
            command,
            active_workers.len(),
            total_nmodels,
            total_nseqs,
        );
        let (partials, mut surviving_workers) =
            run_worker_connections_search(active_workers, &sharded_requests);
        let complete = partials.len() == sharded_requests.len();
        if !surviving_workers.is_empty() {
            let mut workers = worker_pool
                .lock()
                .map_err(|_| std::io::Error::other("worker pool lock poisoned"))?;
            workers.append(&mut surviving_workers);
        }

        if complete {
            return merge_worker_payloads(&partials, total_nmodels, total_nseqs).map(Some);
        }
        if !partials.is_empty() {
            eprintln!(
                "Discarding incomplete worker retry result: received {} of {} shards",
                partials.len(),
                sharded_requests.len()
            );
        }
    }

    Ok(None)
}

fn run_worker_connections_search(
    workers: Vec<WorkerConnection>,
    request_bodies: &[WorkerRequest],
) -> (Vec<Vec<u8>>, Vec<WorkerConnection>) {
    let mut partials = Vec::new();
    let mut surviving_workers = Vec::new();
    for (idx, mut worker) in workers.into_iter().enumerate() {
        let Some(request_body) = request_bodies.get(idx) else {
            surviving_workers.push(worker);
            continue;
        };
        match run_worker_connection_search(&mut worker.stream, request_body) {
            Ok(payload) => {
                if let Err(e) = parse_worker_payload(&payload) {
                    eprintln!("Dropping failed worker connection: {}", e);
                } else {
                    partials.push(payload);
                    surviving_workers.push(worker);
                }
            }
            Err(e) => {
                eprintln!("Dropping failed worker connection: {}", e);
            }
        }
    }
    (partials, surviving_workers)
}

fn worker_sharded_requests(
    request: &str,
    command: &ClientCommand,
    worker_count: usize,
    total_nmodels: u64,
    total_nseqs: u64,
) -> Vec<WorkerRequest> {
    if worker_count == 0 {
        return Vec::new();
    }

    if command.db_mode == DbMode::Hmm && command.db_slice.is_none() {
        return sharded_hmmdb_scan_requests(request, command, worker_count, total_nmodels);
    }

    if command.db_mode != DbMode::Seq {
        return (0..worker_count)
            .map(|_| WorkerRequest::ascii(HMMD_CMD_SEARCH, request))
            .collect();
    }

    if worker_count == 1 && command.seqdb_ranges.is_some() {
        return vec![WorkerRequest::ascii(HMMD_CMD_SEARCH, request)];
    }

    if worker_count == 1 {
        return if matches!(command.query, ClientQuery::Hmm(_)) {
            vec![binary_worker_request(
                request,
                command,
                HMMD_CMD_SEARCH,
                0,
                total_nseqs.min(u32::MAX as u64) as usize,
            )]
        } else {
            vec![WorkerRequest::ascii(HMMD_CMD_SEARCH, request)]
        };
    }

    let requested_ranges = coalesce_db_ranges(
        &command
            .seqdb_ranges
            .clone()
            .unwrap_or_else(|| match usize::try_from(total_nseqs) {
                Ok(0) | Err(_) => Vec::new(),
                Ok(total) => vec![DbRange {
                    start: 1,
                    end: total,
                }],
            }),
    );
    if matches!(command.query, ClientQuery::Hmm(_)) && !db_ranges_are_contiguous(&requested_ranges)
    {
        return vec![WorkerRequest::ascii(HMMD_CMD_SEARCH, request)];
    }
    let total_targets = count_db_ranges(&requested_ranges);
    let empty_index = total_nseqs.min(usize::MAX as u64 - 1).saturating_add(1) as usize;

    let mut range_cursor = DbRangeCursor::new(&requested_ranges);
    let mut remaining = total_targets;
    let mut remaining_workers = worker_count;
    let mut requests = Vec::with_capacity(worker_count);
    for _ in 0..worker_count {
        let count = remaining / remaining_workers;
        if count == 0 {
            let ranged = request_with_seqdb_range(request, empty_index, empty_index);
            if matches!(command.query, ClientQuery::Hmm(_)) {
                requests.push(binary_worker_request(
                    &ranged,
                    command,
                    HMMD_CMD_SEARCH,
                    0,
                    count,
                ));
            } else {
                requests.push(WorkerRequest::ascii(HMMD_CMD_SEARCH, &ranged));
            }
        } else {
            let ranges = range_cursor.take(count);
            if ranges.is_empty() {
                let ranged = request_with_seqdb_range(request, empty_index, empty_index);
                if matches!(command.query, ClientQuery::Hmm(_)) {
                    requests.push(binary_worker_request(
                        &ranged,
                        command,
                        HMMD_CMD_SEARCH,
                        0,
                        count,
                    ));
                } else {
                    requests.push(WorkerRequest::ascii(HMMD_CMD_SEARCH, &ranged));
                }
            } else {
                let ranged = request_with_seqdb_ranges(request, &ranges);
                if matches!(command.query, ClientQuery::Hmm(_)) {
                    let (slice_start, slice_count) = covering_binary_slice(&ranges);
                    requests.push(binary_worker_request(
                        &ranged,
                        command,
                        HMMD_CMD_SEARCH,
                        slice_start,
                        slice_count,
                    ));
                } else {
                    requests.push(WorkerRequest::ascii(HMMD_CMD_SEARCH, &ranged));
                }
            }
            remaining -= count;
        }
        remaining_workers -= 1;
    }
    requests
}

fn covering_binary_slice(ranges: &[DbRange]) -> (usize, usize) {
    let Some(first) = ranges.first() else {
        return (0, 0);
    };
    let start = first.start.saturating_sub(1);
    let end = ranges
        .iter()
        .map(|range| range.end)
        .max()
        .unwrap_or(first.end);
    (start, end.saturating_sub(first.start).saturating_add(1))
}

fn db_ranges_are_contiguous(ranges: &[DbRange]) -> bool {
    ranges
        .windows(2)
        .all(|window| window[0].end.saturating_add(1) == window[1].start)
}

// Coalesce a list of (possibly overlapping or unsorted) `--seqdb_ranges` into a
// sorted set of disjoint ranges that together cover exactly the distinct target
// indices the user asked for.
//
// FAITHFULNESS TO C: the C hmmpgmd master (`hmmdmstr.c` lines 346-385) never
// assigns the same target to two workers. It walks the database by *physical
// position* with a monotonically advancing `inx`, testing each position once via
// `hmmpgmd_IsWithinRanges`, and hands each position to exactly one worker
// (`inx += worker->srch_cnt`). The sharded master (`hmmdmstr_shard.c`) likewise
// gives each worker a disjoint modulo-`num_shards` partition. In both variants
// every target is scored exactly once, so `forward_results()` only sorts and runs
// `p7_tophits_Threshold` — it does NOT run `p7_tophits_RemoveDuplicates` (that
// primitive is exclusive to nhmmer/nhmmscan's overlapping-window pipeline). There
// is therefore no "overlapping shard" hit-dedup scheme in C hmmpgmd to reconstruct.
//
// The only way the Rust master could produce overlapping shards is by slicing a
// user-supplied overlapping range string (e.g. `1..5,3..8`) directly. Coalescing
// here reproduces C's "visit each index once" semantics up front, so the existing
// exact, byte-identical worker-payload merge stays correct: each surviving target
// is emitted by exactly one worker and the concatenated hit lists need no dedup.
fn coalesce_db_ranges(ranges: &[DbRange]) -> Vec<DbRange> {
    let mut sorted: Vec<DbRange> = ranges.iter().copied().filter(|r| r.end >= r.start).collect();
    sorted.sort_by(|a, b| a.start.cmp(&b.start).then(a.end.cmp(&b.end)));
    let mut out: Vec<DbRange> = Vec::with_capacity(sorted.len());
    for range in sorted {
        match out.last_mut() {
            // Merge when the next range overlaps or is directly adjacent to the
            // current accumulated range (adjacent so 1..3,4..6 collapses to 1..6,
            // matching one contiguous monotonic walk in C).
            Some(last) if range.start <= last.end.saturating_add(1) => {
                last.end = last.end.max(range.end);
            }
            _ => out.push(range),
        }
    }
    out
}

fn count_db_ranges(ranges: &[DbRange]) -> usize {
    ranges
        .iter()
        .map(|range| range.end.saturating_sub(range.start).saturating_add(1))
        .sum()
}

struct DbRangeCursor<'a> {
    ranges: &'a [DbRange],
    range_index: usize,
    next: usize,
}

impl<'a> DbRangeCursor<'a> {
    fn new(ranges: &'a [DbRange]) -> Self {
        Self {
            ranges,
            range_index: 0,
            next: ranges.first().map(|range| range.start).unwrap_or(0),
        }
    }

    fn take(&mut self, mut count: usize) -> Vec<DbRange> {
        let mut out = Vec::new();
        while count > 0 && self.range_index < self.ranges.len() {
            let range = self.ranges[self.range_index];
            if self.next > range.end {
                self.range_index += 1;
                self.next = self
                    .ranges
                    .get(self.range_index)
                    .map(|range| range.start)
                    .unwrap_or(0);
                continue;
            }

            let take_count = count.min(range.end - self.next + 1);
            let end = self.next + take_count - 1;
            out.push(DbRange {
                start: self.next,
                end,
            });
            count -= take_count;
            self.next = end.saturating_add(1);
        }
        out
    }
}

#[derive(Clone)]
struct WorkerRequest {
    command: u32,
    body: Vec<u8>,
}

impl WorkerRequest {
    fn ascii(command: u32, request: &str) -> Self {
        Self {
            command,
            body: request.as_bytes().to_vec(),
        }
    }
}

fn sharded_hmmdb_scan_requests(
    request: &str,
    command: &ClientCommand,
    worker_count: usize,
    total_nmodels: u64,
) -> Vec<WorkerRequest> {
    let mut remaining = total_nmodels as usize;
    let mut start = 0usize;
    let mut remaining_workers = worker_count;
    let mut requests = Vec::with_capacity(worker_count);
    for _ in 0..worker_count {
        let count = remaining / remaining_workers;
        requests.push(binary_worker_request(
            request,
            command,
            HMMD_CMD_SCAN,
            start,
            count,
        ));
        start = start.saturating_add(count);
        remaining = remaining.saturating_sub(count);
        remaining_workers -= 1;
    }
    requests
}

fn binary_worker_request(
    request: &str,
    command: &ClientCommand,
    worker_command: u32,
    start: usize,
    count: usize,
) -> WorkerRequest {
    let (options, _query_text) = split_c_request(request);
    let opts = options.as_bytes();
    let query_type = match &command.query {
        ClientQuery::Sequence { .. } => HMMD_SEQUENCE,
        ClientQuery::Hmm(_) => HMMD_HMM,
    };
    let query = match &command.query {
        ClientQuery::Sequence { name, desc, seq } => serialize_hmmd_sequence_query(name, desc, seq),
        ClientQuery::Hmm(hmm) => serialize_hmmd_hmm_query(hmm),
    };
    let query_length = match &command.query {
        ClientQuery::Sequence { seq, .. } => seq.as_bytes().len().saturating_add(2),
        ClientQuery::Hmm(hmm) => hmm.m,
    };
    let mut body = Vec::with_capacity(
        HMMD_SEARCH_CMD_FIXED_SIZE
            + opts.len()
            + 1
            + query.len()
            + C_HMMD_COMMAND_SEARCH_PADDING_SIZE,
    );
    for value in [
        0u32,
        0u32,
        start.min(u32::MAX as usize) as u32,
        count.min(u32::MAX as usize) as u32,
        query_type,
        query_length.min(u32::MAX as usize) as u32,
        (opts.len() + 1).min(u32::MAX as usize) as u32,
    ] {
        body.extend_from_slice(&value.to_ne_bytes());
    }
    body.extend_from_slice(opts);
    body.push(0);
    body.extend_from_slice(&query);
    body.extend(std::iter::repeat_n(0, C_HMMD_COMMAND_SEARCH_PADDING_SIZE));
    WorkerRequest {
        command: worker_command,
        body,
    }
}

fn serialize_hmmd_sequence_query(name: &str, desc: &str, seq: &str) -> Vec<u8> {
    let abc = Alphabet::amino();
    let dsq = abc.digitize(seq.as_bytes());
    let mut query = Vec::with_capacity(name.len() + desc.len() + dsq.len() + 2);
    query.extend_from_slice(name.as_bytes());
    query.push(0);
    query.extend_from_slice(desc.as_bytes());
    query.push(0);
    query.extend_from_slice(&dsq);
    query
}

fn serialize_hmmd_hmm_query(hmm: &Hmm) -> Vec<u8> {
    let mut query = vec![0u8; C_P7_HMM_SHELL_SIZE];
    let mut pointer_fixups = Vec::new();
    write_i32_ne(&mut query, 0, hmm.m.min(i32::MAX as usize) as i32);
    write_i32_ne(&mut query, 104, hmm.nseq);
    write_f32_ne(&mut query, 108, hmm.eff_nseq);
    write_i32_ne(&mut query, 112, hmm.max_length);
    write_u32_ne(&mut query, 136, hmm.checksum);
    for (idx, value) in hmm.evparam.iter().enumerate() {
        write_f32_ne(&mut query, 140 + idx * 4, *value);
    }
    for (idx, value) in hmm.cutoff.iter().enumerate() {
        write_f32_ne(&mut query, 164 + idx * 4, *value);
    }
    for (idx, value) in hmm.compo.iter().enumerate() {
        write_f32_ne(&mut query, 188 + idx * 4, *value);
    }
    record_hmmd_hmm_pointer(&mut pointer_fixups, 280, true, 0);
    write_i32_ne(&mut query, 288, hmm.flags as i32);

    let transition_offset = query.len();
    record_hmmd_hmm_pointer(&mut pointer_fixups, 8, !hmm.t.is_empty(), transition_offset);
    for row in &hmm.t {
        for value in row {
            query.extend_from_slice(&value.to_ne_bytes());
        }
    }
    let match_offset = query.len();
    record_hmmd_hmm_pointer(&mut pointer_fixups, 16, !hmm.mat.is_empty(), match_offset);
    for (node, row) in hmm.mat.iter().enumerate() {
        for (sym, value) in row.iter().take(hmm.abc_k).enumerate() {
            let value = if node == 0 {
                if sym == 0 {
                    1.0
                } else {
                    0.0
                }
            } else {
                *value
            };
            query.extend_from_slice(&value.to_ne_bytes());
        }
    }
    let insert_offset = query.len();
    record_hmmd_hmm_pointer(&mut pointer_fixups, 24, !hmm.ins.is_empty(), insert_offset);
    for row in &hmm.ins {
        for value in row.iter().take(hmm.abc_k) {
            query.extend_from_slice(&value.to_ne_bytes());
        }
    }

    let name_offset = query.len();
    record_hmmd_hmm_pointer(&mut pointer_fixups, 32, !hmm.name.is_empty(), name_offset);
    append_c_string(&mut query, Some(&hmm.name));
    let acc_offset = query.len();
    record_hmmd_hmm_pointer(&mut pointer_fixups, 40, hmm.acc.is_some(), acc_offset);
    append_c_string(&mut query, hmm.acc.as_ref());
    let desc_offset = query.len();
    record_hmmd_hmm_pointer(&mut pointer_fixups, 48, hmm.desc.is_some(), desc_offset);
    append_c_string(&mut query, hmm.desc.as_ref());
    let rf_offset = query.len();
    record_hmmd_hmm_pointer(&mut pointer_fixups, 56, hmm.rf.is_some(), rf_offset);
    append_hmm_annotation(&mut query, hmm.rf.as_deref(), hmm.m);
    let mm_offset = query.len();
    record_hmmd_hmm_pointer(&mut pointer_fixups, 64, hmm.mm.is_some(), mm_offset);
    append_hmm_annotation(&mut query, hmm.mm.as_deref(), hmm.m);
    let consensus_offset = query.len();
    record_hmmd_hmm_pointer(
        &mut pointer_fixups,
        72,
        hmm.consensus.is_some(),
        consensus_offset,
    );
    append_hmm_annotation(&mut query, hmm.consensus.as_deref(), hmm.m);
    let cs_offset = query.len();
    record_hmmd_hmm_pointer(&mut pointer_fixups, 80, hmm.cs.is_some(), cs_offset);
    append_hmm_annotation(&mut query, hmm.cs.as_deref(), hmm.m);
    let ca_offset = query.len();
    record_hmmd_hmm_pointer(&mut pointer_fixups, 88, hmm.ca.is_some(), ca_offset);
    append_hmm_annotation(&mut query, hmm.ca.as_deref(), hmm.m);
    let map_offset = query.len();
    record_hmmd_hmm_pointer(&mut pointer_fixups, 128, hmm.map.is_some(), map_offset);
    if let Some(map) = &hmm.map {
        for idx in 0..=hmm.m {
            let value = map.get(idx).copied().unwrap_or_default();
            query.extend_from_slice(&value.to_ne_bytes());
        }
    }
    record_hmmd_hmm_pointer(&mut pointer_fixups, 96, hmm.comlog.is_some(), 0);
    record_hmmd_hmm_pointer(&mut pointer_fixups, 120, hmm.ctime.is_some(), 0);
    patch_hmmd_hmm_pointers(&mut query, &pointer_fixups);
    query
}

fn record_hmmd_hmm_pointer(
    pointer_fixups: &mut Vec<(usize, usize)>,
    shell_offset: usize,
    present: bool,
    data_offset: usize,
) {
    if present {
        pointer_fixups.push((shell_offset, data_offset));
    }
}

fn patch_hmmd_hmm_pointers(query: &mut [u8], pointer_fixups: &[(usize, usize)]) {
    let base = query.as_ptr() as usize;
    for &(shell_offset, data_offset) in pointer_fixups {
        // C serializes raw process-local pointers in this shell. Matching the
        // native address value is impossible across processes, but preserving
        // non-null native-width pointers keeps the wire shape and decoder
        // semantics byte-compatible after pointer-slot normalization.
        write_ptr_value(query, shell_offset, base.wrapping_add(data_offset));
    }
}

fn append_c_string(out: &mut Vec<u8>, value: Option<&String>) {
    if let Some(value) = value {
        out.extend_from_slice(value.as_bytes());
        out.push(0);
    }
}

fn append_hmm_annotation(out: &mut Vec<u8>, value: Option<&[u8]>, m: usize) {
    if let Some(value) = value {
        for idx in 0..m + 2 {
            let byte = if idx == m + 1 {
                0
            } else {
                value.get(idx).copied().unwrap_or(b' ')
            };
            out.push(byte);
        }
    }
}

fn split_c_request(request: &str) -> (&str, &str) {
    if let Some(first_newline) = request.find('\n') {
        let (options, rest) = request.split_at(first_newline);
        (options, rest.trim_start_matches('\n'))
    } else {
        (request, "")
    }
}

fn request_with_seqdb_range(request: &str, start: usize, end: usize) -> String {
    request_with_seqdb_ranges(request, &[DbRange { start, end }])
}

fn request_with_seqdb_ranges(request: &str, ranges: &[DbRange]) -> String {
    let ranges = ranges
        .iter()
        .map(|range| format!("{}..{}", range.start, range.end))
        .collect::<Vec<_>>()
        .join(",");
    let Some(first_newline) = request.find('\n') else {
        let options = options_with_seqdb_ranges(request, &ranges);
        return options;
    };
    let (options, rest) = request.split_at(first_newline);
    format!("{}{rest}", options_with_seqdb_ranges(options, &ranges))
}

fn options_with_seqdb_ranges(options: &str, ranges: &str) -> String {
    let words = options.split_whitespace().collect::<Vec<_>>();
    let mut filtered = Vec::with_capacity(words.len() + 2);
    let mut i = 0;
    while i < words.len() {
        let word = words[i];
        if word == "--seqdb_ranges" {
            i += 2;
            continue;
        }
        if word.starts_with("--seqdb_ranges=") {
            i += 1;
            continue;
        }
        filtered.push(word);
        i += 1;
    }
    filtered.push("--seqdb_ranges");
    filtered.push(ranges);
    filtered.join(" ")
}

fn run_worker_connection_search(
    stream: &mut TcpStream,
    request: &WorkerRequest,
) -> std::io::Result<Vec<u8>> {
    stream.set_read_timeout(Some(Duration::from_secs(30)))?;
    stream.set_write_timeout(Some(Duration::from_secs(30)))?;
    write_header(
        stream,
        HmmdHeader {
            length: request.body.len() as u32,
            command: request.command,
            status: 0,
        },
    )?;
    stream.write_all(&request.body)?;
    stream.flush()?;

    let mut status = [0; 12];
    stream.read_exact(&mut status)?;
    let code = u32::from_be_bytes(status[0..4].try_into().unwrap());
    let msg_size = u64::from_be_bytes(status[4..12].try_into().unwrap());
    let mut payload = vec![0; msg_size as usize];
    stream.read_exact(&mut payload)?;
    if code == 0 {
        Ok(payload)
    } else {
        Err(std::io::Error::other(
            String::from_utf8_lossy(&payload).into_owned(),
        ))
    }
}

fn shutdown_worker_pool(worker_pool: &WorkerPool) -> std::io::Result<()> {
    let mut workers = worker_pool
        .lock()
        .map_err(|_| std::io::Error::other("worker pool lock poisoned"))?;
    while !workers.is_empty() {
        match shutdown_worker_connection(&mut workers[0].stream) {
            Ok(()) => {
                workers.swap_remove(0);
            }
            Err(e) => {
                eprintln!("Dropping failed worker connection during shutdown: {}", e);
                workers.swap_remove(0);
            }
        }
    }
    Ok(())
}

fn shutdown_worker_connection(stream: &mut TcpStream) -> std::io::Result<()> {
    stream.set_read_timeout(Some(Duration::from_secs(30)))?;
    stream.set_write_timeout(Some(Duration::from_secs(30)))?;
    write_header(
        stream,
        HmmdHeader {
            length: 0,
            command: HMMD_CMD_SHUTDOWN,
            status: 0,
        },
    )?;
    stream.flush()?;

    let header = read_header(stream)?;
    if header.command != HMMD_CMD_SHUTDOWN || header.status != 0 {
        return Err(std::io::Error::other("worker rejected SHUTDOWN"));
    }
    if header.length > 0 {
        let mut body = vec![0; header.length as usize];
        stream.read_exact(&mut body)?;
    }
    Ok(())
}

// Master-side merge of per-worker serialized search payloads.
//
// Mirrors C `forward_results()` (hmmer/src/hmmdmstr.c): gather every worker's
// P7_HIT byte records, sort them, and re-emit one HMMD_SEARCH_STATS block with
// a fresh hit_offsets array followed by the concatenated hit payloads.
//
// C re-runs `p7_tophits_Threshold(&th, pli)` over the merged list to recompute
// nreported/nincluded against the global Z. Here each worker is dispatched with
// the *global* nmodels/nseqs (see `searched_counts` / `serialize_c_search_payload`,
// which derive Z from those totals) and only searches a *non-overlapping* DB
// shard, so a hit appears in exactly one worker. Per-hit reporting/inclusion
// thresholds are independent given a fixed Z, hence summing the per-shard
// nreported/nincluded is identical to a single global threshold pass — we keep
// the C-compatible totals without needing access to the pipeline state here.
//
// DISTRIBUTED-COORDINATION GAP: the global stats (Z, domZ, n_past_*) are taken
// from the master's totals plus summed worker filter counts; this file does not
// reconstruct a P7_PIPELINE on the master to re-run Threshold, because the
// pipeline thresholds object is per-query and not part of the serialized wire
// payload. Per-hit reporting/inclusion thresholds are independent given a fixed
// global Z, so summing per-shard counts equals a single global Threshold pass.
//
// NO OVERLAPPING-SHARD DEDUP IS NEEDED: both C hmmpgmd master variants partition
// the database into DISJOINT pieces — by monotonic physical index range
// (`hmmdmstr.c` `inx += srch_cnt`) or by modulo shard (`hmmdmstr_shard.c`
// `my_shard`/`num_shards`). Every target is scored by exactly one worker, so C's
// `forward_results()` only sorts + Thresholds and never calls
// `p7_tophits_RemoveDuplicates` (that is solely an nhmmer/nhmmscan within-process
// concern for overlapping nucleotide windows). The Rust master likewise produces
// disjoint shards: master-generated ranges advance monotonically, and any
// user-supplied `--seqdb_ranges` are coalesced into disjoint ranges by
// `coalesce_db_ranges` before partitioning. Hence the concatenated worker hit
// lists contain each target once and this merge stays exact without dedup.
fn merge_worker_payloads(
    partials: &[Vec<u8>],
    total_nmodels: u64,
    total_nseqs: u64,
) -> std::io::Result<Vec<u8>> {
    let mut hits = Vec::new();
    let mut stats = WorkerSearchStats::default();

    for payload in partials {
        let partial = parse_worker_payload(payload)?;
        stats.add(&partial.stats);
        hits.extend(partial.hits);
    }

    hits.sort_by(compare_serialized_hits);

    let mut hit_payload = Vec::new();
    let mut hit_offsets = Vec::with_capacity(hits.len());
    for hit in hits {
        hit_offsets.push(hit_payload.len() as u64);
        hit_payload.extend_from_slice(&hit.bytes);
    }

    let mut payload = Vec::with_capacity(
        HMMD_SEARCH_STATS_SIZE + hit_offsets.len().saturating_mul(8) + hit_payload.len(),
    );
    write_c_stats_payload(
        &mut payload,
        total_nmodels,
        total_nseqs,
        hit_offsets.len() as u64,
        &stats,
        &hit_offsets,
    );
    payload.extend_from_slice(&hit_payload);
    Ok(payload)
}

struct ParsedWorkerPayload {
    stats: WorkerSearchStats,
    hits: Vec<SerializedHit>,
}

#[derive(Clone, Copy, Default)]
struct WorkerSearchStats {
    n_past_msv: u64,
    n_past_bias: u64,
    n_past_vit: u64,
    n_past_fwd: u64,
    nreported: u64,
    nincluded: u64,
}

impl WorkerSearchStats {
    fn add(&mut self, other: &Self) {
        self.n_past_msv = self.n_past_msv.saturating_add(other.n_past_msv);
        self.n_past_bias = self.n_past_bias.saturating_add(other.n_past_bias);
        self.n_past_vit = self.n_past_vit.saturating_add(other.n_past_vit);
        self.n_past_fwd = self.n_past_fwd.saturating_add(other.n_past_fwd);
        self.nreported = self.nreported.saturating_add(other.nreported);
        self.nincluded = self.nincluded.saturating_add(other.nincluded);
    }
}

struct SerializedHit {
    sortkey: f64,
    name: String,
    first_domain_iali: i64,
    first_domain_jali: i64,
    bytes: Vec<u8>,
}

fn compare_serialized_hits(a: &SerializedHit, b: &SerializedHit) -> std::cmp::Ordering {
    a.sortkey
        .partial_cmp(&b.sortkey)
        .unwrap_or(std::cmp::Ordering::Equal)
        .then_with(|| a.name.cmp(&b.name))
        .then_with(|| {
            let a_dir = if a.first_domain_iali < a.first_domain_jali {
                1
            } else {
                -1
            };
            let b_dir = if b.first_domain_iali < b.first_domain_jali {
                1
            } else {
                -1
            };
            if a_dir != b_dir {
                b_dir.cmp(&a_dir)
            } else {
                a.first_domain_iali.cmp(&b.first_domain_iali)
            }
        })
}

fn parse_worker_payload(payload: &[u8]) -> std::io::Result<ParsedWorkerPayload> {
    if payload.len() < HMMD_SEARCH_STATS_SIZE {
        return Err(std::io::Error::other("short worker search payload"));
    }

    let _nmodels = read_payload_u64(payload, 42)?;
    let _nseqs = read_payload_u64(payload, 50)?;
    let stats = WorkerSearchStats {
        n_past_msv: read_payload_u64(payload, 58)?,
        n_past_bias: read_payload_u64(payload, 66)?,
        n_past_vit: read_payload_u64(payload, 74)?,
        n_past_fwd: read_payload_u64(payload, 82)?,
        nreported: read_payload_u64(payload, 98)?,
        nincluded: read_payload_u64(payload, 106)?,
    };
    let nhits = read_payload_u64(payload, 90)? as usize;
    if nhits == 0 {
        if payload.len() != HMMD_SEARCH_STATS_SIZE {
            return Err(std::io::Error::other(
                "zero-hit worker payload has trailing bytes",
            ));
        }
        let sentinel = read_payload_u64(payload, 114)?;
        if sentinel != u64::MAX {
            return Err(std::io::Error::other(
                "zero-hit worker payload missing offset sentinel",
            ));
        }
        return Ok(ParsedWorkerPayload {
            stats,
            hits: Vec::new(),
        });
    }
    let offsets_start = 114;
    let stats_size = offsets_start + nhits.saturating_mul(8);

    let mut hits = None;
    if payload.len() >= stats_size {
        let mut hit_offsets = Vec::with_capacity(nhits);
        for i in 0..nhits {
            hit_offsets.push(read_payload_u64(payload, offsets_start + i * 8)? as usize);
        }
        hits = parse_worker_hits_with_offsets(payload, stats_size, &hit_offsets).ok();
    }
    let hits = match hits {
        Some(hits) => hits,
        None => parse_worker_hits_sequential(payload, nhits)?,
    };

    Ok(ParsedWorkerPayload { stats, hits })
}

fn parse_worker_hits_with_offsets(
    payload: &[u8],
    hit_base: usize,
    hit_offsets: &[usize],
) -> std::io::Result<Vec<SerializedHit>> {
    if !hit_offsets.windows(2).all(|window| window[0] < window[1]) {
        return Err(std::io::Error::other("invalid worker hit offset order"));
    }

    let mut hits = Vec::with_capacity(hit_offsets.len());
    for (idx, offset) in hit_offsets.iter().copied().enumerate() {
        let start = hit_base
            .checked_add(offset)
            .ok_or_else(|| std::io::Error::other("worker hit offset overflow"))?;
        let end = if let Some(next_offset) = hit_offsets.get(idx + 1).copied() {
            hit_base
                .checked_add(next_offset)
                .ok_or_else(|| std::io::Error::other("worker hit offset overflow"))?
        } else {
            payload.len()
        };
        hits.push(parse_serialized_hit_slice(payload, start, end)?);
    }
    Ok(hits)
}

fn parse_worker_hits_sequential(
    payload: &[u8],
    nhits: usize,
) -> std::io::Result<Vec<SerializedHit>> {
    let mut hits = Vec::with_capacity(nhits);
    let mut start = HMMD_SEARCH_STATS_SIZE;
    for _ in 0..nhits {
        let end = serialized_hit_payload_end(payload, start)?;
        hits.push(parse_serialized_hit_slice(payload, start, end)?);
        start = end;
    }
    if start != payload.len() {
        return Err(std::io::Error::other("invalid worker hit trailing payload"));
    }
    Ok(hits)
}

fn parse_serialized_hit_slice(
    payload: &[u8],
    start: usize,
    end: usize,
) -> std::io::Result<SerializedHit> {
    if start >= end || end > payload.len() {
        return Err(std::io::Error::other("invalid worker hit size"));
    }
    validate_serialized_hit_payload(&payload[start..end])?;
    let sortkey = f64::from_bits(u64::from_be_bytes(
        payload[start + 8..start + 16].try_into().unwrap(),
    ));
    let name = serialized_hit_name(&payload[start..end])?;
    let (first_domain_iali, first_domain_jali) =
        serialized_hit_first_domain_alipos(&payload[start..end])?;
    Ok(SerializedHit {
        sortkey,
        name,
        first_domain_iali,
        first_domain_jali,
        bytes: payload[start..end].to_vec(),
    })
}

fn serialized_hit_name(bytes: &[u8]) -> std::io::Result<String> {
    let hit_size = u32::from_be_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let name_start = P7_HIT_BASE_SIZE;
    let Some(name_len) = bytes[name_start..hit_size].iter().position(|&b| b == 0) else {
        return Err(std::io::Error::other("worker hit name is unterminated"));
    };
    std::str::from_utf8(&bytes[name_start..name_start + name_len])
        .map(|name| name.to_string())
        .map_err(|e| std::io::Error::other(format!("worker hit name is not valid UTF-8: {e}")))
}

fn serialized_hit_first_domain_alipos(bytes: &[u8]) -> std::io::Result<(i64, i64)> {
    let ndom = i32::from_be_bytes(bytes[72..76].try_into().unwrap());
    if ndom <= 0 {
        return Ok((0, 1));
    }
    let hit_size = u32::from_be_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let domain_start = hit_size;
    if domain_start + P7_DOMAIN_BASE_SIZE > bytes.len() {
        return Err(std::io::Error::other("short worker domain payload"));
    }
    let iali = i64::from_be_bytes(
        bytes[domain_start + 20..domain_start + 28]
            .try_into()
            .unwrap(),
    );
    let jali = i64::from_be_bytes(
        bytes[domain_start + 28..domain_start + 36]
            .try_into()
            .unwrap(),
    );
    Ok((iali, jali))
}

fn validate_serialized_hit_payload(bytes: &[u8]) -> std::io::Result<()> {
    let end = serialized_hit_payload_end(bytes, 0)?;
    if end != bytes.len() {
        return Err(std::io::Error::other("invalid worker hit trailing payload"));
    }
    Ok(())
}

fn serialized_hit_payload_end(bytes: &[u8], start: usize) -> std::io::Result<usize> {
    if start > bytes.len() {
        return Err(std::io::Error::other("short worker hit payload"));
    }
    let bytes = &bytes[start..];
    if bytes.len() < P7_HIT_BASE_SIZE {
        return Err(std::io::Error::other("short worker hit payload"));
    }
    let hit_size = u32::from_be_bytes(bytes[0..4].try_into().unwrap()) as usize;
    if hit_size < P7_HIT_BASE_SIZE || hit_size > bytes.len() {
        return Err(std::io::Error::other("invalid worker hit size"));
    }

    let ndom = i32::from_be_bytes(bytes[72..76].try_into().unwrap());
    if ndom < 0 {
        return Err(std::io::Error::other("invalid worker domain count"));
    }

    let presence = bytes[108];
    let mut pos = P7_HIT_BASE_SIZE;
    take_serialized_c_string(bytes, &mut pos, hit_size, "worker hit name")?;
    if presence & P7_HIT_ACC_PRESENT != 0 {
        take_serialized_c_string(bytes, &mut pos, hit_size, "worker hit accession")?;
    }
    if presence & P7_HIT_DESC_PRESENT != 0 {
        take_serialized_c_string(bytes, &mut pos, hit_size, "worker hit description")?;
    }
    if pos != hit_size {
        return Err(std::io::Error::other("invalid worker hit string layout"));
    }

    for _ in 0..ndom {
        pos = validate_serialized_domain_payload(bytes, pos)?;
    }
    Ok(start + pos)
}

fn validate_serialized_domain_payload(bytes: &[u8], start: usize) -> std::io::Result<usize> {
    if start + P7_DOMAIN_BASE_SIZE > bytes.len() {
        return Err(std::io::Error::other("short worker domain payload"));
    }
    let domain_size = u32::from_be_bytes(bytes[start..start + 4].try_into().unwrap()) as usize;
    if domain_size != P7_DOMAIN_BASE_SIZE {
        return Err(std::io::Error::other("invalid worker domain size"));
    }
    validate_serialized_alidisplay_payload(bytes, start + domain_size)
}

fn validate_serialized_alidisplay_payload(bytes: &[u8], start: usize) -> std::io::Result<usize> {
    if start + P7_ALIDISPLAY_BASE_SIZE > bytes.len() {
        return Err(std::io::Error::other("short worker alignment payload"));
    }
    let ad_size = u32::from_be_bytes(bytes[start..start + 4].try_into().unwrap()) as usize;
    if ad_size < P7_ALIDISPLAY_BASE_SIZE || start + ad_size > bytes.len() {
        return Err(std::io::Error::other(
            "invalid worker alignment payload size",
        ));
    }
    let n = i32::from_be_bytes(bytes[start + 4..start + 8].try_into().unwrap());
    if n < 0 {
        return Err(std::io::Error::other("invalid worker alignment length"));
    }
    let n = n as usize;
    let end = start + ad_size;
    let presence = bytes[start + 44];
    let mut pos = start + P7_ALIDISPLAY_BASE_SIZE;
    if presence & P7_ALIDISPLAY_RFLINE_PRESENT != 0 {
        take_fixed_serialized_c_string(bytes, &mut pos, end, n, "worker alignment RF line")?;
    }
    if presence & P7_ALIDISPLAY_MMLINE_PRESENT != 0 {
        take_fixed_serialized_c_string(bytes, &mut pos, end, n, "worker alignment MM line")?;
    }
    if presence & P7_ALIDISPLAY_CSLINE_PRESENT != 0 {
        take_fixed_serialized_c_string(bytes, &mut pos, end, n, "worker alignment CS line")?;
    }
    take_fixed_serialized_c_string(bytes, &mut pos, end, n, "worker alignment model line")?;
    take_fixed_serialized_c_string(bytes, &mut pos, end, n, "worker alignment match line")?;
    if presence & P7_ALIDISPLAY_ASEQ_PRESENT != 0 {
        take_fixed_serialized_c_string(bytes, &mut pos, end, n, "worker alignment sequence line")?;
    }
    if presence & P7_ALIDISPLAY_NTSEQ_PRESENT != 0 {
        take_fixed_serialized_c_string(
            bytes,
            &mut pos,
            end,
            n.saturating_mul(3),
            "worker alignment nucleotide line",
        )?;
    }
    if presence & P7_ALIDISPLAY_PPLINE_PRESENT != 0 {
        take_fixed_serialized_c_string(bytes, &mut pos, end, n, "worker alignment posterior line")?;
    }
    for label in [
        "worker alignment HMM name",
        "worker alignment HMM accession",
        "worker alignment HMM description",
        "worker alignment sequence name",
        "worker alignment sequence accession",
        "worker alignment sequence description",
    ] {
        take_serialized_c_string(bytes, &mut pos, end, label)?;
    }
    if pos != end {
        return Err(std::io::Error::other(
            "invalid worker alignment trailing payload",
        ));
    }
    Ok(end)
}

fn take_fixed_serialized_c_string(
    bytes: &[u8],
    pos: &mut usize,
    end: usize,
    len: usize,
    label: &str,
) -> std::io::Result<()> {
    let next = pos
        .checked_add(len)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| std::io::Error::other(format!("{label} length overflow")))?;
    if next > end || bytes[next - 1] != 0 {
        return Err(std::io::Error::other(format!("{label} is malformed")));
    }
    *pos = next;
    Ok(())
}

fn take_serialized_c_string(
    bytes: &[u8],
    pos: &mut usize,
    end: usize,
    label: &str,
) -> std::io::Result<()> {
    if *pos >= end {
        return Err(std::io::Error::other(format!("{label} is missing")));
    }
    let Some(offset) = bytes[*pos..end].iter().position(|&b| b == 0) else {
        return Err(std::io::Error::other(format!("{label} is unterminated")));
    };
    *pos += offset + 1;
    Ok(())
}

fn read_payload_u64(payload: &[u8], offset: usize) -> std::io::Result<u64> {
    if offset + 8 > payload.len() {
        return Err(std::io::Error::other("short worker stats payload"));
    }
    Ok(u64::from_be_bytes(
        payload[offset..offset + 8].try_into().unwrap(),
    ))
}

fn parse_c_client_command(request: &str) -> Result<ClientCommand, String> {
    let mut lines = request.lines();
    let options = lines
        .next()
        .ok_or_else(|| "Missing options string".to_string())?
        .trim();
    let db_mode = if options.contains("--seqdb") {
        DbMode::Seq
    } else if options.contains("--hmmdb") {
        DbMode::Hmm
    } else {
        return Err("No search database specified, --seqdb or --hmmdb.".to_string());
    };

    let query_text = lines.collect::<Vec<_>>().join("\n");
    let query = parse_client_query(&query_text, db_mode)?;
    let search_options = parse_hmmpgmd_search_options(options)?;
    let seqdb_ranges = parse_seqdb_ranges_option(options)?;
    Ok(ClientCommand {
        db_mode,
        query,
        options: search_options,
        seqdb_ranges,
        db_slice: None,
    })
}

fn parse_worker_command_body(body: &[u8]) -> Result<ClientCommand, String> {
    if body.starts_with(b"@") {
        let request = std::str::from_utf8(body)
            .map_err(|e| format!("worker search request is not valid UTF-8: {e}"))?;
        return parse_c_client_command(request);
    }

    if body.len() < HMMD_SEARCH_CMD_FIXED_SIZE {
        return Err("short HMMD_SEARCH_CMD body".to_string());
    }

    let db_type = read_native_u32(body, 4)?;
    let inx = read_native_u32(body, 8)? as usize;
    let cnt = read_native_u32(body, 12)? as usize;
    let query_type = read_native_u32(body, 16)?;
    let query_length = read_native_u32(body, 20)? as usize;
    let opts_length = read_native_u32(body, 24)? as usize;
    let data = &body[HMMD_SEARCH_CMD_FIXED_SIZE..];
    if opts_length == 0 || data.len() < opts_length {
        return Err("invalid HMMD_SEARCH_CMD options length".to_string());
    }
    let opts_bytes = &data[..opts_length];
    if opts_bytes.last().copied() != Some(0) {
        return Err("HMMD_SEARCH_CMD options string is not NUL-terminated".to_string());
    }
    let options = std::str::from_utf8(&opts_bytes[..opts_bytes.len() - 1])
        .map_err(|e| format!("HMMD_SEARCH_CMD options are not valid UTF-8: {e}"))?;
    let query_bytes = &data[opts_length..];
    let db_mode = if options.contains("--seqdb") {
        DbMode::Seq
    } else if options.contains("--hmmdb") {
        DbMode::Hmm
    } else {
        match db_type {
            1 => DbMode::Seq,
            2 => DbMode::Hmm,
            _ => return Err("No search database specified, --seqdb or --hmmdb.".to_string()),
        }
    };
    let query = match query_type {
        HMMD_SEQUENCE => parse_hmmd_sequence_query(query_bytes, query_length)?,
        HMMD_HMM => {
            if query_bytes.len() < query_length {
                return Err("short HMMD_SEARCH_CMD query data".to_string());
            }
            if query_bytes.starts_with(b"HMM") {
                let query_text = std::str::from_utf8(query_bytes)
                    .map_err(|e| format!("HMMD_SEARCH_CMD query is not valid UTF-8: {e}"))?;
                parse_client_query(query_text, db_mode)?
            } else {
                parse_hmmd_hmm_query(query_bytes, query_length)?
            }
        }
        _ => return Err(format!("unknown HMMD_SEARCH_CMD query type {query_type}")),
    };
    let search_options = parse_hmmpgmd_search_options(options)?;
    let mut seqdb_ranges = parse_seqdb_ranges_option(options)?;
    if db_mode == DbMode::Seq && cnt > 0 {
        let slice_range = DbRange {
            start: inx.saturating_add(1),
            end: inx.saturating_add(cnt),
        };
        seqdb_ranges = Some(match seqdb_ranges {
            Some(ranges) => intersect_db_ranges(&ranges, slice_range),
            None => vec![slice_range],
        });
    }
    Ok(ClientCommand {
        db_mode,
        query,
        options: search_options,
        seqdb_ranges,
        db_slice: Some(DbSlice {
            start: inx,
            count: cnt,
        }),
    })
}

fn intersect_db_ranges(ranges: &[DbRange], slice: DbRange) -> Vec<DbRange> {
    ranges
        .iter()
        .filter_map(|range| {
            let start = range.start.max(slice.start);
            let end = range.end.min(slice.end);
            (start <= end).then_some(DbRange { start, end })
        })
        .collect()
}

fn parse_hmmd_sequence_query(
    query_bytes: &[u8],
    query_length: usize,
) -> Result<ClientQuery, String> {
    let (name, mut pos) = take_hmmd_query_string(query_bytes, 0, "sequence name")?;
    let (desc, next) = take_hmmd_query_string(query_bytes, pos, "sequence description")?;
    pos = next;
    let end = pos
        .checked_add(query_length)
        .ok_or_else(|| "HMMD_SEARCH_CMD query length overflow".to_string())?;
    if query_length < 2 || end > query_bytes.len() {
        return Err("short HMMD_SEARCH_CMD query data".to_string());
    }
    let dsq = &query_bytes[pos..end];
    if dsq.first().copied() != Some(hmmer_pure_rs::alphabet::DSQ_SENTINEL)
        || dsq.last().copied() != Some(hmmer_pure_rs::alphabet::DSQ_SENTINEL)
    {
        return Err("HMMD_SEARCH_CMD sequence query missing digital sentinels".to_string());
    }
    let abc = Alphabet::amino();
    let seq = abc.textize(dsq, query_length - 2);
    Ok(ClientQuery::Sequence { name, desc, seq })
}

fn parse_hmmd_hmm_query(query_bytes: &[u8], query_length: usize) -> Result<ClientQuery, String> {
    if query_bytes.len() < C_P7_HMM_SHELL_SIZE {
        return Err("short HMMD_SEARCH_CMD HMM query shell".to_string());
    }
    let m = query_length;
    let mut hmm = Hmm::new(m, AlphabetType::Amino, Alphabet::amino().k);
    hmm.flags = read_native_i32(query_bytes, 288)? as u32;
    hmm.nseq = read_native_i32(query_bytes, 104)?;
    hmm.eff_nseq = read_native_f32(query_bytes, 108)?;
    hmm.max_length = read_native_i32(query_bytes, 112)?;
    hmm.checksum = read_native_u32(query_bytes, 136)?;
    for idx in 0..hmm.evparam.len() {
        hmm.evparam[idx] = read_native_f32(query_bytes, 140 + idx * 4)?;
    }
    for idx in 0..hmm.cutoff.len() {
        hmm.cutoff[idx] = read_native_f32(query_bytes, 164 + idx * 4)?;
    }
    for idx in 0..hmm.compo.len() {
        hmm.compo[idx] = read_native_f32(query_bytes, 188 + idx * 4)?;
    }

    let mut pos = C_P7_HMM_SHELL_SIZE;
    for node in 0..=m {
        for trans in 0..hmmer_pure_rs::hmm::NTRANSITIONS {
            hmm.t[node][trans] = take_native_f32(query_bytes, &mut pos, "HMM transitions")?;
        }
    }
    for node in 0..=m {
        for sym in 0..hmm.abc_k {
            hmm.mat[node][sym] = take_native_f32(query_bytes, &mut pos, "HMM match emissions")?;
        }
    }
    for node in 0..=m {
        for sym in 0..hmm.abc_k {
            hmm.ins[node][sym] = take_native_f32(query_bytes, &mut pos, "HMM insert emissions")?;
        }
    }

    if read_native_usize(query_bytes, 32)? != 0 {
        let (name, next) = take_hmmd_query_string(query_bytes, pos, "HMM name")?;
        hmm.name = name;
        pos = next;
    }
    if read_native_usize(query_bytes, 40)? != 0 {
        let (acc, next) = take_hmmd_query_string(query_bytes, pos, "HMM accession")?;
        hmm.acc = Some(acc);
        pos = next;
    }
    if read_native_usize(query_bytes, 48)? != 0 {
        let (desc, next) = take_hmmd_query_string(query_bytes, pos, "HMM description")?;
        hmm.desc = Some(desc);
        pos = next;
    }
    if hmm.flags & P7H_RF != 0 {
        hmm.rf = Some(take_hmm_annotation(
            query_bytes,
            &mut pos,
            m,
            "RF annotation",
        )?);
    }
    if hmm.flags & P7H_MMASK != 0 {
        hmm.mm = Some(take_hmm_annotation(
            query_bytes,
            &mut pos,
            m,
            "MM annotation",
        )?);
    }
    if hmm.flags & P7H_CONS != 0 {
        hmm.consensus = Some(take_hmm_annotation(
            query_bytes,
            &mut pos,
            m,
            "consensus annotation",
        )?);
    }
    if hmm.flags & P7H_CS != 0 {
        hmm.cs = Some(take_hmm_annotation(
            query_bytes,
            &mut pos,
            m,
            "CS annotation",
        )?);
    }
    if hmm.flags & P7H_CA != 0 {
        hmm.ca = Some(take_hmm_annotation(
            query_bytes,
            &mut pos,
            m,
            "CA annotation",
        )?);
    }
    if hmm.flags & P7H_MAP != 0 {
        let mut map = Vec::with_capacity(m + 1);
        for _ in 0..=m {
            map.push(take_native_i32(query_bytes, &mut pos, "HMM map")?);
        }
        hmm.map = Some(map);
    }
    Ok(ClientQuery::Hmm(hmm))
}

fn take_hmm_annotation(
    bytes: &[u8],
    pos: &mut usize,
    m: usize,
    label: &str,
) -> Result<Vec<u8>, String> {
    let n = m
        .checked_add(2)
        .ok_or_else(|| format!("HMMD_SEARCH_CMD {label} length overflow"))?;
    let end = pos
        .checked_add(n)
        .ok_or_else(|| format!("HMMD_SEARCH_CMD {label} length overflow"))?;
    if end > bytes.len() {
        return Err(format!("short HMMD_SEARCH_CMD {label}"));
    }
    let value = bytes[*pos..end].to_vec();
    *pos = end;
    Ok(value)
}

fn take_hmmd_query_string(
    bytes: &[u8],
    start: usize,
    label: &str,
) -> Result<(String, usize), String> {
    if start >= bytes.len() {
        return Err(format!("HMMD_SEARCH_CMD {label} is missing"));
    }
    let end = bytes[start..]
        .iter()
        .position(|&b| b == 0)
        .map(|offset| start + offset)
        .ok_or_else(|| format!("HMMD_SEARCH_CMD {label} is unterminated"))?;
    let value = std::str::from_utf8(&bytes[start..end])
        .map_err(|e| format!("HMMD_SEARCH_CMD {label} is not valid UTF-8: {e}"))?;
    Ok((value.to_string(), end + 1))
}

fn read_native_u32(body: &[u8], offset: usize) -> Result<u32, String> {
    if offset + 4 > body.len() {
        return Err("short HMMD_SEARCH_CMD body".to_string());
    }
    Ok(u32::from_ne_bytes(
        body[offset..offset + 4].try_into().unwrap(),
    ))
}

fn read_native_i32(body: &[u8], offset: usize) -> Result<i32, String> {
    if offset + 4 > body.len() {
        return Err("short HMMD_SEARCH_CMD body".to_string());
    }
    Ok(i32::from_ne_bytes(
        body[offset..offset + 4].try_into().unwrap(),
    ))
}

fn read_native_f32(body: &[u8], offset: usize) -> Result<f32, String> {
    if offset + 4 > body.len() {
        return Err("short HMMD_SEARCH_CMD body".to_string());
    }
    Ok(f32::from_ne_bytes(
        body[offset..offset + 4].try_into().unwrap(),
    ))
}

fn read_native_usize(body: &[u8], offset: usize) -> Result<usize, String> {
    let width = std::mem::size_of::<usize>();
    if offset + width > body.len() {
        return Err("short HMMD_SEARCH_CMD body".to_string());
    }
    let mut value = [0u8; std::mem::size_of::<usize>()];
    value.copy_from_slice(&body[offset..offset + width]);
    Ok(usize::from_ne_bytes(value))
}

fn take_native_f32(bytes: &[u8], pos: &mut usize, label: &str) -> Result<f32, String> {
    let value =
        read_native_f32(bytes, *pos).map_err(|_| format!("short HMMD_SEARCH_CMD {label}"))?;
    *pos += 4;
    Ok(value)
}

fn take_native_i32(bytes: &[u8], pos: &mut usize, label: &str) -> Result<i32, String> {
    let value =
        read_native_i32(bytes, *pos).map_err(|_| format!("short HMMD_SEARCH_CMD {label}"))?;
    *pos += 4;
    Ok(value)
}

fn write_i32_ne(out: &mut [u8], offset: usize, value: i32) {
    out[offset..offset + 4].copy_from_slice(&value.to_ne_bytes());
}

fn write_u32_ne(out: &mut [u8], offset: usize, value: u32) {
    out[offset..offset + 4].copy_from_slice(&value.to_ne_bytes());
}

fn write_f32_ne(out: &mut [u8], offset: usize, value: f32) {
    out[offset..offset + 4].copy_from_slice(&value.to_ne_bytes());
}

fn write_ptr_value(out: &mut [u8], offset: usize, value: usize) {
    let width = std::mem::size_of::<usize>();
    out[offset..offset + width].copy_from_slice(&value.to_ne_bytes());
}

fn master_init_body(state: &ServerState) -> Vec<u8> {
    let mut body = vec![0u8; HMMD_INIT_BODY_SIZE];
    let mut data = Vec::new();

    match state {
        ServerState::Seq(db) => {
            let path = db.path.to_string_lossy();
            let sid = db.cache_id.as_bytes();
            let copy_len = sid.len().min(63);
            body[..copy_len].copy_from_slice(&sid[..copy_len]);
            write_u32_ne(&mut body, 64, 0);
            write_u32_ne(&mut body, 72, 1);
            write_u32_ne(
                &mut body,
                76,
                db.sequences.len().min(u32::MAX as usize) as u32,
            );
            data.extend_from_slice(path.as_bytes());
            data.push(0);
        }
        ServerState::Hmm(db) => {
            let path = db.path.to_string_lossy();
            write_u32_ne(&mut body, 68, 0);
            write_u32_ne(&mut body, 80, 1);
            write_u32_ne(&mut body, 84, db.hmms.len().min(u32::MAX as usize) as u32);
            data.extend_from_slice(path.as_bytes());
            data.push(0);
        }
    }

    body.extend_from_slice(&data);
    body
}

fn load_worker_init_state(body: &[u8]) -> Result<Option<ServerState>, String> {
    if body.len() < HMMD_INIT_BODY_SIZE {
        return Err("short HMMD_CMD_INIT body".to_string());
    }

    let seq_cnt = read_native_u32(body, 72)? as usize;
    let hmm_cnt = read_native_u32(body, 80)? as usize;
    if seq_cnt != 0 {
        let seqdb_off = read_native_u32(body, 64)? as usize;
        let path = init_body_path(body, seqdb_off, "sequence database")?;
        return Ok(Some(load_seqdb(&PathBuf::from(path))));
    }
    if hmm_cnt != 0 {
        let hmmdb_off = read_native_u32(body, 68)? as usize;
        let path = init_body_path(body, hmmdb_off, "HMM database")?;
        return Ok(Some(load_hmmdb(&PathBuf::from(path))));
    }
    Ok(None)
}

fn init_body_path(body: &[u8], offset: usize, label: &str) -> Result<String, String> {
    let start = HMMD_INIT_BODY_SIZE
        .checked_add(offset)
        .ok_or_else(|| format!("HMMD_CMD_INIT {label} path offset overflow"))?;
    let (path, _) = take_hmmd_query_string(body, start, label)?;
    if path.is_empty() {
        return Err(format!("HMMD_CMD_INIT {label} path is empty"));
    }
    Ok(path)
}

fn parse_hmmpgmd_search_options(options: &str) -> Result<SearchOptions, String> {
    let words = options
        .split_whitespace()
        .map(|word| word.strip_prefix('@').unwrap_or(word))
        .collect::<Vec<_>>();
    let mut parsed = SearchOptions::default();
    let mut i = 0usize;
    while i < words.len() {
        let word = words[i];
        match word {
            "--hmmdb" | "--seqdb" | "--seqdb_ranges" | "--cpu" => i += 2,
            "--cut_ga" => {
                parsed.bit_cutoff = BitCutoff::GA;
                i += 1;
            }
            "--cut_tc" => {
                parsed.bit_cutoff = BitCutoff::TC;
                i += 1;
            }
            "--cut_nc" => {
                parsed.bit_cutoff = BitCutoff::NC;
                i += 1;
            }
            "--max" => {
                parsed.max = true;
                i += 1;
            }
            "--nobias" => {
                parsed.nobias = true;
                i += 1;
            }
            "--nonull2" => {
                parsed.nonull2 = true;
                i += 1;
            }
            "-E" | "--E" | "-T" | "--T" | "--domE" | "--domT" | "--incE" | "--incT"
            | "--incdomE" | "--incdomT" | "--F1" | "--F2" | "--F3" => {
                let (value, next) = hmmpgmd_option_value(&words, i, word)?;
                apply_hmmpgmd_option(&mut parsed, word, value)?;
                i = next;
            }
            _ if word.starts_with("--hmmdb=")
                || word.starts_with("--seqdb=")
                || word.starts_with("--seqdb_ranges=") =>
            {
                i += 1;
            }
            _ => {
                if let Some((option, value)) = hmmpgmd_split_attached_option(word) {
                    apply_hmmpgmd_option(&mut parsed, option, value)?;
                }
                i += 1;
            }
        }
    }
    Ok(parsed)
}

fn hmmpgmd_split_attached_option(word: &str) -> Option<(&'static str, &str)> {
    for option in [
        "--incdomE",
        "--incdomT",
        "--domE",
        "--domT",
        "--incE",
        "--incT",
        "--F1",
        "--F2",
        "--F3",
        "--E",
        "--T",
    ] {
        if let Some(value) = word.strip_prefix(&format!("{option}=")) {
            return Some((option, value));
        }
    }
    if let Some(value) = word.strip_prefix("-E") {
        if !value.is_empty() {
            return Some(("-E", value));
        }
    }
    if let Some(value) = word.strip_prefix("-T") {
        if !value.is_empty() {
            return Some(("-T", value));
        }
    }
    None
}

fn hmmpgmd_option_value<'a>(
    words: &'a [&str],
    index: usize,
    option: &str,
) -> Result<(&'a str, usize), String> {
    words
        .get(index + 1)
        .copied()
        .map(|value| (value, index + 2))
        .ok_or_else(|| format!("{option} requires a value"))
}

fn apply_hmmpgmd_option(
    parsed: &mut SearchOptions,
    option: &str,
    value: &str,
) -> Result<(), String> {
    match option {
        "-E" | "--E" => parsed.e_value_threshold = parse_hmmpgmd_f64_option(option, value)?,
        "-T" | "--T" => parsed.t = Some(parse_hmmpgmd_f32_option(option, value)?),
        "--domE" => parsed.dom_e_value_threshold = parse_hmmpgmd_f64_option(option, value)?,
        "--domT" => parsed.dom_t = Some(parse_hmmpgmd_f32_option(option, value)?),
        "--incE" => parsed.inc_e = parse_hmmpgmd_f64_option(option, value)?,
        "--incT" => parsed.inc_t = Some(parse_hmmpgmd_f32_option(option, value)?),
        "--incdomE" => parsed.inc_dome = parse_hmmpgmd_f64_option(option, value)?,
        "--incdomT" => parsed.inc_dom_t = Some(parse_hmmpgmd_f32_option(option, value)?),
        "--F1" => parsed.f1 = parse_hmmpgmd_f64_option(option, value)?,
        "--F2" => parsed.f2 = parse_hmmpgmd_f64_option(option, value)?,
        "--F3" => parsed.f3 = parse_hmmpgmd_f64_option(option, value)?,
        _ => {}
    }
    Ok(())
}

fn parse_hmmpgmd_f64_option(option: &str, value: &str) -> Result<f64, String> {
    value
        .parse::<f64>()
        .map_err(|_| format!("{option} requires a numeric value"))
}

fn parse_hmmpgmd_f32_option(option: &str, value: &str) -> Result<f32, String> {
    value
        .parse::<f32>()
        .map_err(|_| format!("{option} requires a numeric value"))
}

fn parse_seqdb_ranges_option(options: &str) -> Result<Option<Vec<DbRange>>, String> {
    let mut words = options.split_whitespace();
    while let Some(word) = words.next() {
        if word == "--seqdb_ranges" {
            let value = words
                .next()
                .ok_or_else(|| "--seqdb_ranges requires coords <from>..<to>".to_string())?;
            return parse_db_ranges(value).map(Some);
        }
        if let Some(value) = word.strip_prefix("--seqdb_ranges=") {
            return parse_db_ranges(value).map(Some);
        }
    }
    Ok(None)
}

fn parse_db_ranges(value: &str) -> Result<Vec<DbRange>, String> {
    let mut ranges = Vec::new();
    for part in value.split(',') {
        let (start, end) = part.split_once("..").ok_or_else(|| {
            format!("--seqdb_ranges takes coords <from>..<to>; {part} not recognized")
        })?;
        let start = start.parse::<usize>().map_err(|_| {
            format!("--seqdb_ranges takes coords <from>..<to>; {part} not recognized")
        })?;
        let end = end.parse::<usize>().map_err(|_| {
            format!("--seqdb_ranges takes coords <from>..<to>; {part} not recognized")
        })?;
        if start == 0 || end < start {
            return Err(format!(
                "--seqdb_ranges takes coords <from>..<to>; {part} not recognized"
            ));
        }
        ranges.push(DbRange { start, end });
    }
    if ranges.is_empty() {
        return Err("--seqdb_ranges requires coords <from>..<to>".to_string());
    }
    Ok(ranges)
}

fn parse_client_query(query: &str, db_mode: DbMode) -> Result<ClientQuery, String> {
    let query = query.trim_start();
    if query.starts_with('>') {
        return parse_fasta_query(query).map(|(name, desc, seq)| ClientQuery::Sequence {
            name,
            desc,
            seq,
        });
    }
    if query.starts_with("HMM") {
        if db_mode == DbMode::Hmm {
            return Err("A HMM cannot be used to search a hmm database".to_string());
        }
        return parse_hmm_query(query).map(ClientQuery::Hmm);
    }
    Err("Unknown query sequence/hmm format".to_string())
}

fn parse_fasta_query(fasta: &str) -> Result<(String, String, String), String> {
    let mut lines = fasta.lines();
    let header = lines
        .next()
        .ok_or_else(|| "Missing search sequence/hmm".to_string())?;
    if !header.starts_with('>') {
        return Err("Unknown query sequence/hmm format".to_string());
    }
    let mut header_parts = header[1..].splitn(2, char::is_whitespace);
    let name = header_parts
        .next()
        .ok_or_else(|| "Error parsing FASTA sequence".to_string())?;
    if name.is_empty() {
        return Err("Error parsing FASTA sequence".to_string());
    }
    let desc = header_parts.next().unwrap_or("").trim().to_string();
    let seq: String = lines
        .take_while(|line| line.trim_end() != "//")
        .map(str::trim)
        .collect();
    if seq.is_empty() {
        return Err("Error zero length FASTA sequence".to_string());
    }
    Ok((name.to_string(), desc, seq))
}

fn parse_hmm_query(query: &str) -> Result<Hmm, String> {
    let mut record = String::new();
    let mut saw_end = false;
    for line in query.lines() {
        record.push_str(line);
        record.push('\n');
        if line.trim_end() == "//" {
            saw_end = true;
            break;
        }
    }
    if !saw_end {
        return Err("Error reading query hmm: missing // terminator".to_string());
    }

    let mut hmms = hmmfile::read_hmms(BufReader::new(Cursor::new(record.into_bytes())))
        .map_err(|e| format!("Error reading query hmm: {e}"))?;
    if hmms.len() != 1 {
        return Err(format!(
            "Error reading query hmm: expected one HMM, found {}",
            hmms.len()
        ));
    }
    Ok(hmms.remove(0))
}

fn write_c_status(stream: &mut TcpStream, status: u32, msg_size: u64) -> std::io::Result<()> {
    stream.write_all(&status.to_be_bytes())?;
    stream.write_all(&msg_size.to_be_bytes())
}

fn write_c_error(stream: &mut TcpStream, message: &str) -> std::io::Result<()> {
    write_c_status(stream, 1, message.len() as u64)?;
    stream.write_all(message.as_bytes())
}

fn write_c_stats(
    stream: &mut TcpStream,
    nmodels: u64,
    nseqs: u64,
    included: u64,
) -> std::io::Result<()> {
    let searched = nseqs.max(nmodels);
    for value in [0.0_f64, 0.0, 0.0, searched as f64, searched as f64] {
        stream.write_all(&value.to_bits().to_be_bytes())?;
    }
    stream.write_all(&[0, 0])?;
    for value in [nmodels, nseqs, 0, 0, 0, 0, 0, 0, included, u64::MAX] {
        stream.write_all(&value.to_be_bytes())?;
    }
    Ok(())
}

fn serialize_c_search_payload(nmodels: u64, nseqs: u64, results: &mut Vec<SearchHit>) -> Vec<u8> {
    sort_search_hits(results);

    let mut hit_payload = Vec::new();
    let mut hit_offsets = Vec::with_capacity(results.len());
    for hit in &mut *results {
        hit_offsets.push(hit_payload.len() as u64);
        serialize_p7_hit(&mut hit_payload, hit);
    }

    let included = results.iter().filter(|hit| hit.nincluded > 0).count() as u64;
    let mut payload = Vec::with_capacity(
        HMMD_SEARCH_STATS_SIZE + hit_offsets.len().saturating_mul(8) + hit_payload.len(),
    );
    write_c_stats_payload(
        &mut payload,
        nmodels,
        nseqs,
        results.len() as u64,
        &WorkerSearchStats {
            nreported: results.len() as u64,
            nincluded: included,
            ..WorkerSearchStats::default()
        },
        &hit_offsets,
    );
    payload.extend_from_slice(&hit_payload);
    payload
}

fn write_c_stats_payload(
    out: &mut Vec<u8>,
    nmodels: u64,
    nseqs: u64,
    nhits: u64,
    stats: &WorkerSearchStats,
    hit_offsets: &[u64],
) {
    let searched = nseqs.max(nmodels);
    for value in [0.0_f64, 0.0, 0.0, searched as f64, searched as f64] {
        out.extend_from_slice(&value.to_bits().to_be_bytes());
    }
    out.extend_from_slice(&[0, 0]);
    for value in [
        nmodels,
        nseqs,
        stats.n_past_msv,
        stats.n_past_bias,
        stats.n_past_vit,
        stats.n_past_fwd,
        nhits,
        stats.nreported,
        stats.nincluded,
    ] {
        out.extend_from_slice(&value.to_be_bytes());
    }
    if hit_offsets.is_empty() {
        out.extend_from_slice(&u64::MAX.to_be_bytes());
    } else {
        for offset in hit_offsets {
            out.extend_from_slice(&offset.to_be_bytes());
        }
    }
}

fn serialize_p7_hit(out: &mut Vec<u8>, hit: &SearchHit) {
    let mut presence = 0;
    if !hit.acc.is_empty() {
        presence |= P7_HIT_ACC_PRESENT;
    }
    if !hit.desc.is_empty() {
        presence |= P7_HIT_DESC_PRESENT;
    }
    let name = hit.name.as_bytes();
    let acc = hit.acc.as_bytes();
    let desc = hit.desc.as_bytes();
    let serialized_size = P7_HIT_BASE_SIZE
        + name.len()
        + 1
        + if presence & P7_HIT_ACC_PRESENT != 0 {
            acc.len() + 1
        } else {
            0
        }
        + if presence & P7_HIT_DESC_PRESENT != 0 {
            desc.len() + 1
        } else {
            0
        };

    write_u32(out, serialized_size as u32);
    write_i32(out, hit.window_length);
    write_f64(out, hit.sortkey);
    write_f32(out, hit.score);
    write_f32(out, hit.pre_score);
    write_f32(out, hit.sum_score);
    write_f64(out, hit.lnp);
    write_f64(out, hit.pre_lnp);
    write_f64(out, hit.sum_lnp);
    write_f32(out, hit.nexpected);
    write_i32(out, hit.nregions);
    write_i32(out, hit.nclustered);
    write_i32(out, hit.noverlaps);
    write_i32(out, hit.nenvelopes);
    write_i32(out, hit.domains.len().min(i32::MAX as usize) as i32);
    write_u32(out, hit.flags);
    write_i32(out, hit.nreported);
    write_i32(out, hit.nincluded);
    write_i32(out, hit.best_domain);
    write_i64(out, hit.seqidx);
    write_i64(out, hit.subseq_start);
    out.push(presence);
    out.extend_from_slice(name);
    out.push(0);
    if presence & P7_HIT_ACC_PRESENT != 0 {
        out.extend_from_slice(acc);
        out.push(0);
    }
    if presence & P7_HIT_DESC_PRESENT != 0 {
        out.extend_from_slice(desc);
        out.push(0);
    }

    for domain in &hit.domains {
        serialize_p7_domain(out, hit, domain);
    }
}

fn serialize_p7_domain(out: &mut Vec<u8>, hit: &SearchHit, domain: &Domain) {
    write_u32(out, P7_DOMAIN_BASE_SIZE as u32);
    write_i64(out, domain.ienv);
    write_i64(out, domain.jenv);
    write_i64(out, domain.iali);
    write_i64(out, domain.jali);
    write_i64(out, 0);
    write_i64(out, 0);
    write_f32(out, domain.envsc);
    write_f32(out, domain.domcorrection);
    write_f32(out, domain.dombias);
    write_f32(out, domain.oasc);
    write_f32(out, domain.bitscore);
    write_f64(out, domain.lnp);
    write_i32(out, i32::from(domain.is_reported));
    write_i32(out, i32::from(domain.is_included));
    write_i32(out, 0);
    serialize_p7_alidisplay(out, hit, domain.ad.as_ref());
}

fn serialize_p7_alidisplay(out: &mut Vec<u8>, hit: &SearchHit, ad: Option<&AliDisplay>) {
    let n = ad.map(|ad| ad.model.len()).unwrap_or(0);
    let empty = "";
    let model = ad.map(|ad| ad.model.as_str()).unwrap_or(empty);
    let mline = ad.map(|ad| ad.mline.as_str()).unwrap_or(empty);
    let aseq = ad.map(|ad| ad.aseq.as_str()).unwrap_or(empty);
    let ppline = ad.map(|ad| ad.ppline.as_str()).unwrap_or(empty);
    let rfline = ad
        .and_then(|ad| (!ad.rfline.is_empty()).then_some(ad.rfline.as_str()))
        .unwrap_or(empty);
    let hmmfrom = ad.map(|ad| ad.hmmfrom as i32).unwrap_or(-1);
    let hmmto = ad.map(|ad| ad.hmmto as i32).unwrap_or(-1);
    let sqfrom = ad.map(|ad| ad.sqfrom as i64).unwrap_or(0);
    let sqto = ad.map(|ad| ad.sqto as i64).unwrap_or(0);

    let mut presence = P7_ALIDISPLAY_ASEQ_PRESENT;
    if !rfline.is_empty() {
        presence |= P7_ALIDISPLAY_RFLINE_PRESENT;
    }
    if !ppline.is_empty() {
        presence |= P7_ALIDISPLAY_PPLINE_PRESENT;
    }

    let mut serialized_size = P7_ALIDISPLAY_BASE_SIZE + model.len() + 1 + mline.len() + 1;
    if presence & P7_ALIDISPLAY_RFLINE_PRESENT != 0 {
        serialized_size += rfline.len() + 1;
    }
    if presence & P7_ALIDISPLAY_ASEQ_PRESENT != 0 {
        serialized_size += aseq.len() + 1;
    }
    if presence & P7_ALIDISPLAY_PPLINE_PRESENT != 0 {
        serialized_size += ppline.len() + 1;
    }
    for value in [
        &hit.hmm_name,
        &hit.hmm_acc,
        &hit.hmm_desc,
        &hit.seq_name,
        &hit.seq_acc,
        &hit.seq_desc,
    ] {
        serialized_size += value.len() + 1;
    }

    write_u32(out, serialized_size as u32);
    write_i32(out, n.min(i32::MAX as usize) as i32);
    write_i32(out, hmmfrom);
    write_i32(out, hmmto);
    write_i32(out, hit.model_length);
    write_i64(out, sqfrom);
    write_i64(out, sqto);
    write_i64(out, hit.sequence_length);
    out.push(presence);
    if presence & P7_ALIDISPLAY_RFLINE_PRESENT != 0 {
        write_c_string(out, rfline);
    }
    write_c_string(out, model);
    write_c_string(out, mline);
    if presence & P7_ALIDISPLAY_ASEQ_PRESENT != 0 {
        write_c_string(out, aseq);
    }
    if presence & P7_ALIDISPLAY_PPLINE_PRESENT != 0 {
        write_c_string(out, ppline);
    }
    write_c_string(out, &hit.hmm_name);
    write_c_string(out, &hit.hmm_acc);
    write_c_string(out, &hit.hmm_desc);
    write_c_string(out, &hit.seq_name);
    write_c_string(out, &hit.seq_acc);
    write_c_string(out, &hit.seq_desc);
}

fn write_c_string(out: &mut Vec<u8>, value: &str) {
    out.extend_from_slice(value.as_bytes());
    out.push(0);
}

fn write_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_be_bytes());
}

fn write_i32(out: &mut Vec<u8>, value: i32) {
    out.extend_from_slice(&value.to_be_bytes());
}

fn write_i64(out: &mut Vec<u8>, value: i64) {
    out.extend_from_slice(&value.to_be_bytes());
}

fn write_f32(out: &mut Vec<u8>, value: f32) {
    out.extend_from_slice(&value.to_bits().to_be_bytes());
}

fn write_f64(out: &mut Vec<u8>, value: f64) {
    out.extend_from_slice(&value.to_bits().to_be_bytes());
}

impl SearchHit {
    fn from_pipeline_hit(
        hit: &Hit,
        name: String,
        hmm: &Hmm,
        alidisplay_hmm_name: &str,
        seq_name: &str,
        seq_acc: &str,
        seq_desc: &str,
        sequence_length: usize,
    ) -> Self {
        let best_domain = hit
            .dcl
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.bitscore
                    .partial_cmp(&b.bitscore)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(idx, _)| idx as i32)
            .unwrap_or(-1);
        SearchHit {
            name,
            acc: hit.acc.clone(),
            desc: hit.desc.clone(),
            window_length: hit.n.min(i32::MAX as usize) as i32,
            sortkey: hit.sortkey,
            score: hit.score,
            pre_score: hit.pre_score,
            sum_score: hit.sum_score,
            lnp: hit.lnp,
            pre_lnp: hit.pre_lnp,
            sum_lnp: hit.sum_lnp,
            nexpected: hit.nexpected,
            nregions: hit.nregions.min(i32::MAX as usize) as i32,
            nclustered: hit.nclustered.min(i32::MAX as usize) as i32,
            noverlaps: hit.noverlaps.min(i32::MAX as usize) as i32,
            nenvelopes: hit.nenvelopes.min(i32::MAX as usize) as i32,
            flags: hit.flags,
            nreported: hit.nreported.min(i32::MAX as usize) as i32,
            nincluded: hit.nincluded.min(i32::MAX as usize) as i32,
            best_domain,
            seqidx: hit.seqidx,
            subseq_start: hit.subseq_start,
            hmm_name: alidisplay_hmm_name.to_string(),
            hmm_acc: hmm.acc.clone().unwrap_or_default(),
            hmm_desc: hmm.desc.clone().unwrap_or_default(),
            seq_name: seq_name.to_string(),
            seq_acc: seq_acc.to_string(),
            seq_desc: seq_desc.to_string(),
            model_length: hmm.m.min(i32::MAX as usize) as i32,
            sequence_length: sequence_length.min(i64::MAX as usize) as i64,
            domains: hit.dcl.clone(),
        }
    }
}

fn accept_workers(listener: TcpListener, worker_pool: WorkerPool, init_body: Arc<Vec<u8>>) {
    for stream in listener.incoming() {
        match stream {
            Ok(mut stream) => {
                let worker_pool = Arc::clone(&worker_pool);
                let init_body = Arc::clone(&init_body);
                thread::spawn(move || {
                    if let Err(e) = initialize_worker(&mut stream, &init_body) {
                        eprintln!("Worker connection error: {}", e);
                        return;
                    }
                    eprintln!("Worker initialized");
                    match worker_pool.lock() {
                        Ok(mut workers) => workers.push(WorkerConnection { stream }),
                        Err(_) => eprintln!("Worker pool lock poisoned"),
                    }
                });
            }
            Err(e) => eprintln!("Worker accept error: {}", e),
        }
    }
}

fn initialize_worker(stream: &mut TcpStream, init_body: &[u8]) -> std::io::Result<()> {
    write_header(
        stream,
        HmmdHeader {
            length: init_body.len() as u32,
            command: HMMD_CMD_INIT,
            status: 0,
        },
    )?;
    stream.write_all(init_body)?;
    stream.flush()?;

    let header = read_header(stream)?;
    if header.command != HMMD_CMD_INIT || header.status != 0 {
        return Err(std::io::Error::other("worker rejected INIT"));
    }
    let mut body = vec![0; header.length as usize];
    stream.read_exact(&mut body)?;
    Ok(())
}

fn run_worker(host: &str, wport: u16, cpu: usize, mut state: Option<ServerState>) -> ExitCode {
    eprintln!("Worker search CPU threads: {cpu}");
    let thread_pool = if cpu > 1 {
        Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(cpu)
                .build()
                .unwrap_or_else(|e| {
                    eprintln!("Cannot initialize worker search thread pool: {e}");
                    std::process::exit(1);
                }),
        )
    } else {
        None
    };
    let addr = format!("{}:{}", host, wport);
    let mut stream = TcpStream::connect(&addr).unwrap_or_else(|e| {
        eprintln!("Cannot connect to master worker port {}: {}", addr, e);
        std::process::exit(1);
    });

    loop {
        let header = match read_header(&mut stream) {
            Ok(header) => header,
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                return ExitCode::SUCCESS;
            }
            Err(e) => {
                eprintln!("Worker read failed: {}", e);
                return ExitCode::FAILURE;
            }
        };
        let mut body = vec![0; header.length as usize];
        if let Err(e) = stream.read_exact(&mut body) {
            eprintln!("Worker command body read failed: {}", e);
            return ExitCode::FAILURE;
        }

        match header.command {
            HMMD_CMD_INIT => {
                if state.is_none() {
                    match load_worker_init_state(&body) {
                        Ok(Some(init_state)) => state = Some(init_state),
                        Ok(None) => {}
                        Err(message) => {
                            eprintln!("Worker INIT failed: {}", message);
                            if let Err(e) = write_header(
                                &mut stream,
                                HmmdHeader {
                                    length: 0,
                                    command: HMMD_CMD_INIT,
                                    status: 1,
                                },
                            )
                            .and_then(|_| stream.flush())
                            {
                                eprintln!("Worker INIT error response failed: {}", e);
                            }
                            return ExitCode::FAILURE;
                        }
                    }
                }
                if let Err(e) = write_header(
                    &mut stream,
                    HmmdHeader {
                        length: HMMD_INIT_RESPONSE_SIZE as u32,
                        command: HMMD_CMD_INIT,
                        status: 0,
                    },
                )
                .and_then(|_| stream.write_all(&vec![0; HMMD_INIT_RESPONSE_SIZE]))
                .and_then(|_| stream.flush())
                {
                    eprintln!("Worker INIT response failed: {}", e);
                    return ExitCode::FAILURE;
                }
            }
            HMMD_CMD_SEARCH | HMMD_CMD_SCAN => {
                let runtime = SearchRuntime {
                    thread_pool: thread_pool.as_ref(),
                };
                let result = match state.as_ref() {
                    Some(state) => run_worker_search(state, &body, runtime),
                    None => Err("worker has no loaded --hmmdb or --seqdb".to_string()),
                };
                let write_result = match result {
                    Ok(payload) => write_c_status(&mut stream, 0, payload.len() as u64)
                        .and_then(|_| stream.write_all(&payload))
                        .and_then(|_| stream.flush()),
                    Err(message) => write_c_status(&mut stream, 1, message.len() as u64)
                        .and_then(|_| stream.write_all(message.as_bytes()))
                        .and_then(|_| stream.flush()),
                };
                if let Err(e) = write_result {
                    eprintln!("Worker search response failed: {}", e);
                    return ExitCode::FAILURE;
                }
            }
            HMMD_CMD_SHUTDOWN => {
                let _ = write_header(
                    &mut stream,
                    HmmdHeader {
                        length: 0,
                        command: HMMD_CMD_SHUTDOWN,
                        status: 0,
                    },
                );
                let _ = stream.flush();
                return ExitCode::SUCCESS;
            }
            _ => {
                eprintln!("Unknown worker command {}", header.command);
                return ExitCode::FAILURE;
            }
        }
    }
}

fn run_worker_search(
    state: &ServerState,
    body: &[u8],
    runtime: SearchRuntime<'_>,
) -> Result<Vec<u8>, String> {
    let command = parse_worker_command_body(body)?;
    let db_ok = matches!(
        (state, command.db_mode),
        (ServerState::Hmm(_), DbMode::Hmm) | (ServerState::Seq(_), DbMode::Seq)
    );
    if !db_ok {
        return Err("Requested database is not loaded".to_string());
    }
    let mut results = run_query(state, &command, runtime).map_err(|e| e.to_string())?;
    let (nmodels, nseqs) = searched_counts(state, &command.query);
    Ok(serialize_c_search_payload(nmodels, nseqs, &mut results))
}

fn read_header(stream: &mut TcpStream) -> std::io::Result<HmmdHeader> {
    let mut buf = [0; HMMD_HEADER_SIZE];
    stream.read_exact(&mut buf)?;
    Ok(HmmdHeader {
        length: u32::from_ne_bytes(buf[0..4].try_into().unwrap()),
        command: u32::from_ne_bytes(buf[4..8].try_into().unwrap()),
        status: u32::from_ne_bytes(buf[8..12].try_into().unwrap()),
    })
}

fn write_header(stream: &mut TcpStream, header: HmmdHeader) -> std::io::Result<()> {
    stream.write_all(&header.length.to_ne_bytes())?;
    stream.write_all(&header.command.to_ne_bytes())?;
    stream.write_all(&header.status.to_ne_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_hit_worker_payload_rejects_trailing_bytes() {
        let mut payload = Vec::new();
        write_c_stats_payload(&mut payload, 1, 2, 0, &WorkerSearchStats::default(), &[]);
        assert_eq!(payload.len(), HMMD_SEARCH_STATS_SIZE);
        assert!(parse_worker_payload(&payload).is_ok());

        payload.extend_from_slice(b"garbage");
        assert!(parse_worker_payload(&payload).is_err());
    }

    fn sample_search_hit_with_domains() -> SearchHit {
        let n = 5usize;
        let ad = AliDisplay {
            model: "MKLVE".to_string(),
            mline: "MK+VE".to_string(),
            aseq: "MKIVE".to_string(),
            ppline: "*****".to_string(),
            rfline: "xxxxx".to_string(),
            hmmfrom: 1,
            hmmto: n,
            sqfrom: 10,
            sqto: 14,
        };
        let domain = Domain {
            iali: 10,
            jali: 14,
            ienv: 9,
            jenv: 15,
            bitscore: 42.5,
            lnp: -30.0,
            dombias: 0.5,
            oasc: 4.5,
            envsc: 50.0,
            domcorrection: 1.0,
            is_reported: true,
            is_included: true,
            ad: Some(ad),
        };
        SearchHit {
            name: "target1".to_string(),
            acc: "ACC1".to_string(),
            desc: "a description".to_string(),
            window_length: 0,
            sortkey: 42.5,
            score: 42.5,
            pre_score: 43.0,
            sum_score: 42.5,
            lnp: -30.0,
            pre_lnp: -29.0,
            sum_lnp: -30.0,
            nexpected: 1.0,
            nregions: 1,
            nclustered: 0,
            noverlaps: 0,
            nenvelopes: 1,
            flags: 0,
            nreported: 1,
            nincluded: 1,
            best_domain: 0,
            seqidx: 0,
            subseq_start: 0,
            hmm_name: "modelA".to_string(),
            hmm_acc: "PF00001".to_string(),
            hmm_desc: "model description".to_string(),
            seq_name: "target1".to_string(),
            seq_acc: "ACC1".to_string(),
            seq_desc: "a description".to_string(),
            model_length: n as i32,
            sequence_length: 100,
            domains: vec![domain],
        }
    }

    #[test]
    fn full_hit_payload_passes_full_validation() {
        // Serialize a hit carrying a full P7_DOMAIN + P7_ALIDISPLAY payload and
        // confirm it round-trips through full (not shell-only) validation:
        // domain record + alidisplay record with all lines/strings present.
        let hit = sample_search_hit_with_domains();
        let mut out = Vec::new();
        serialize_p7_hit(&mut out, &hit);

        // Hit base + name/acc/desc, then exactly one domain (size 92) and one
        // alidisplay record must consume the whole buffer.
        let end = serialized_hit_payload_end(&out, 0).expect("hit payload should validate");
        assert_eq!(end, out.len(), "full hit payload must be exactly consumed");
        validate_serialized_hit_payload(&out).expect("full hit payload must validate");

        // The single domain must be exactly the C base size (no scores_per_pos).
        let hit_size = u32::from_be_bytes(out[0..4].try_into().unwrap()) as usize;
        let domain_size =
            u32::from_be_bytes(out[hit_size..hit_size + 4].try_into().unwrap()) as usize;
        assert_eq!(domain_size, P7_DOMAIN_BASE_SIZE);

        let (iali, jali) = serialized_hit_first_domain_alipos(&out).unwrap();
        assert_eq!((iali, jali), (10, 14));
        assert_eq!(serialized_hit_name(&out).unwrap(), "target1");
    }

    #[test]
    fn full_hit_payload_round_trips_through_search_payload_and_merge() {
        // A complete search payload with a real domain/alidisplay hit must parse
        // and merge (master-side gather) without falling back to shell records.
        let mut results = vec![sample_search_hit_with_domains()];
        let payload = serialize_c_search_payload(1, 2, &mut results);

        let parsed = parse_worker_payload(&payload).expect("search payload must parse");
        assert_eq!(parsed.hits.len(), 1);
        assert_eq!(parsed.hits[0].name, "target1");

        let merged = merge_worker_payloads(&[payload], 1, 2).expect("merge must succeed");
        let reparsed = parse_worker_payload(&merged).expect("merged payload must parse");
        assert_eq!(reparsed.hits.len(), 1);
        // Merged hit bytes are byte-identical to the original serialized hit.
        assert_eq!(reparsed.hits[0].bytes, parsed.hits[0].bytes);
    }

    #[test]
    fn coalesce_db_ranges_merges_overlapping_and_adjacent() {
        // Overlapping and adjacent user ranges collapse to the disjoint set of
        // distinct indices C visits once. 1..5 & 3..8 overlap -> 1..8; 10..12 &
        // 13..15 are adjacent -> 10..15; unsorted input is normalized.
        let coalesced = coalesce_db_ranges(&[
            DbRange { start: 13, end: 15 },
            DbRange { start: 1, end: 5 },
            DbRange { start: 3, end: 8 },
            DbRange { start: 10, end: 12 },
        ]);
        assert_eq!(
            coalesced,
            vec![DbRange { start: 1, end: 8 }, DbRange { start: 10, end: 15 }]
        );
        // Coalesced length is the distinct-index count (8 + 6), not the naive
        // sum of the raw ranges (5 + 6 + 3 + 3 = 17) which would double-count.
        assert_eq!(count_db_ranges(&coalesced), 14);
    }

    #[test]
    fn overlapping_seqdb_ranges_produce_disjoint_worker_shards() {
        // A SEQUENCE query (ASCII-sharded) with overlapping user ranges must be
        // split across workers so that each target index is searched by exactly
        // one worker, matching C's "visit each physical index once" master.
        let request = "@--seqdb 1 --seqdb_ranges 1..5,3..8\n>q\nACDEF\n//\n";
        let command = ClientCommand {
            db_mode: DbMode::Seq,
            query: ClientQuery::Sequence {
                name: "q".to_string(),
                desc: String::new(),
                seq: "ACDEF".to_string(),
            },
            options: SearchOptions::default(),
            seqdb_ranges: Some(vec![
                DbRange { start: 1, end: 5 },
                DbRange { start: 3, end: 8 },
            ]),
            db_slice: None,
        };

        // Two workers split the coalesced 1..8 (8 distinct targets) into 4 + 4.
        let requests = worker_sharded_requests(request, &command, 2, 1, 8);
        assert_eq!(requests.len(), 2);

        // Collect the indices each worker is told to search and assert they form
        // a disjoint cover of exactly 1..=8 (no index appears in two shards).
        let mut all_indices = Vec::new();
        for req in &requests {
            let body = std::str::from_utf8(&req.body).unwrap();
            let ranges_str = body
                .split("--seqdb_ranges")
                .nth(1)
                .expect("worker request carries --seqdb_ranges")
                .trim()
                .split_whitespace()
                .next()
                .unwrap();
            for part in ranges_str.split(',') {
                let (s, e) = part.split_once("..").unwrap();
                for idx in s.parse::<usize>().unwrap()..=e.parse::<usize>().unwrap() {
                    all_indices.push(idx);
                }
            }
        }
        all_indices.sort_unstable();
        let deduped: Vec<usize> = {
            let mut d = all_indices.clone();
            d.dedup();
            d
        };
        assert_eq!(
            all_indices, deduped,
            "no target index may be assigned to more than one worker"
        );
        assert_eq!(all_indices, (1..=8).collect::<Vec<_>>());
    }

    #[test]
    fn merge_concatenates_disjoint_shard_hit_lists_without_duplication() {
        // Two workers each return a hit for a distinct target (disjoint shards,
        // as the master guarantees). The master merge must contain both hits,
        // each exactly once, sorted — no dedup pass drops or double-counts them.
        // sortkey is lnP-style (lower = more significant = sorted first), matching
        // Pipeline::hit_sortkey and the merge's ascending compare_serialized_hits.
        let mut a = sample_search_hit_with_domains();
        a.name = "targetA".to_string();
        a.seq_name = "targetA".to_string();
        a.sortkey = -50.0;
        let mut b = sample_search_hit_with_domains();
        b.name = "targetB".to_string();
        b.seq_name = "targetB".to_string();
        b.sortkey = -40.0;

        let payload_a = serialize_c_search_payload(1, 8, &mut vec![a]);
        let payload_b = serialize_c_search_payload(1, 8, &mut vec![b]);

        let merged =
            merge_worker_payloads(&[payload_a, payload_b], 1, 8).expect("merge must succeed");
        let reparsed = parse_worker_payload(&merged).expect("merged payload must parse");
        assert_eq!(reparsed.hits.len(), 2);
        let names: Vec<&str> = reparsed.hits.iter().map(|h| h.name.as_str()).collect();
        // Ascending sortkey: targetA (-50) is more significant, comes first.
        assert_eq!(names, vec!["targetA", "targetB"]);
    }

    #[test]
    fn noncontiguous_seqdb_hmm_ranges_are_not_binary_sharded() {
        let request = "@--seqdb 1 --seqdb_ranges 1..1,3..3,5..5\nHMMER3/f\n//\n";
        let abc = Alphabet::amino();
        let command = ClientCommand {
            db_mode: DbMode::Seq,
            query: ClientQuery::Hmm(Hmm::new(1, AlphabetType::Amino, abc.k)),
            options: SearchOptions::default(),
            seqdb_ranges: Some(vec![
                DbRange { start: 1, end: 1 },
                DbRange { start: 3, end: 3 },
                DbRange { start: 5, end: 5 },
            ]),
            db_slice: None,
        };

        let requests = worker_sharded_requests(request, &command, 2, 0, 5);

        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].command, HMMD_CMD_SEARCH);
        assert!(requests[0].body.starts_with(b"@"));
        let body = std::str::from_utf8(&requests[0].body).unwrap();
        assert!(body.contains("--seqdb_ranges 1..1,3..3,5..5"));
    }
}
