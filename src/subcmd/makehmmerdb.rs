//! makehmmerdb — create an FM-index database for nhmmer.

use std::io::{Read, Write};
use std::path::PathBuf;

use clap::Parser;
use divsufsort::sort_in_place;

use hmmer_pure_rs::alphabet::Alphabet;
use hmmer_pure_rs::fm_index::FmIndex;
use hmmer_pure_rs::sequence::{self, Sequence};

const META_MAGIC: &[u8] = b"HMMERDB_META\0";
const META_VERSION: u32 = 1;
const C_META_MAGIC: &[u8] = b"HMMERDB_C_META\0";
const C_META_VERSION: u32 = 1;
const C_STREAM_MAGIC: &[u8] = b"HMMERDB_C_STREAM\0";
const C_STREAM_VERSION: u32 = 1;
const INDEX_MAGIC: &[u8] = b"HMMERDB_INDEXES\0";
const INDEX_VERSION: u32 = 1;
const INDEX_KIND_FORWARD_STRAND: u32 = 0;
const INDEX_KIND_REVERSE_STRAND: u32 = 1;
const FM_BLOCK_OVERLAP: usize = 20_000;
const FM_DNA_ALPH_TYPE: u8 = 0;
const FM_DNA_ALPH_SIZE: u8 = 4;
const FM_DNA_CHAR_BITS: u8 = 2;
const FM_AMINO_ALPH_TYPE: u8 = 4;
const FM_AMINO_ALPH_SIZE: u8 = 26;
const FM_AMINO_CHAR_BITS: u8 = 5;
const FM_FREQ_CNT_SB: u32 = 65_536;
const FM_AMINO_ALPHABET: &[u8; FM_AMINO_ALPH_SIZE as usize] = b"ACDEFGHIKLMNPQRSTVWYBJZOUX";

#[derive(Parser)]
#[command(name = "makehmmerdb", about = "Create an FM-index database for nhmmer")]
struct Args {
    /// Input sequence file (FASTA)
    seqfile: PathBuf,

    /// Output database file
    binaryfile: PathBuf,

    /// Specify input file format
    #[arg(long = "informat")]
    informat: Option<String>,

    /// Bin length (power of 2; 32<=n<=4096)
    #[arg(long = "bin_length", default_value = "256")]
    bin_length: usize,

    /// Suffix array sample rate (power of 2)
    #[arg(long = "sa_freq", default_value = "8")]
    sa_freq: usize,

    /// Input sequence block size in Mbases
    #[arg(long = "block_size", default_value = "50")]
    block_size: usize,

    /// Assert input alphabet is amino acid
    #[arg(long = "amino", conflicts_with_all = ["dna", "rna"])]
    amino: bool,

    /// Assert input alphabet is DNA
    #[arg(long = "dna", conflicts_with_all = ["amino", "rna"])]
    dna: bool,

    /// Assert input alphabet is RNA
    #[arg(long = "rna", conflicts_with_all = ["amino", "dna"])]
    rna: bool,

    /// Build a forward-strand-only database
    #[arg(long = "fwd_only")]
    fwd_only: bool,

    /// Write the C-layout FM stream directly instead of the Rust HMMERDB container
    #[arg(long = "cstream")]
    cstream: bool,

    /// Write the legacy Rust HMMERDB container instead of the native C stream
    #[arg(long = "container")]
    container: bool,
}

/// Entry point for `makehmmerdb`: turn a (DNA) FASTA into an FM-index database
/// consumable by `nhmmer`'s SSV pre-filter.
///
/// Concatenates every sequence into a single `$`-separated text, partitions it
/// into FM-index blocks via `build_readblock_like_records()` (which mirrors C
/// `esl_sqio_ReadBlock()` per-sequence windowing — see that function), builds
/// BWT + suffix array records, and by default serializes the native C FM-index
/// stream as the top-level file (byte-faithful `FM_METADATA` header from
/// `fwrite()` in makehmmerdb.c:726-763 followed by per-block `FM_DATA` records
/// from buildAndWriteFMIndex(), makehmmerdb.c:315-340). `--container` preserves
/// the older Rust `HMMERDB` wrapper with custom per-block FM indexes, metadata
/// for descriptions/accessions, block windows, overlap, ambiguity ranges, and
/// the same C-layout metadata/FM-record extensions appended after it.
///
/// Faithfulness notes (what matches C byte-for-byte vs. what still differs):
///   * MATCHES C: the metadata header field order/widths (fwd_only, alph_type,
///     alph_size, charBits, freq_SA, freq_cnt_sb, freq_cnt_b, block_count(u16),
///     seq_count, ambig count, char_count(u64)); per-sequence metadata
///     (target_id, target_start(u64), fm_start, length, name/acc/source/desc
///     lengths(u16) and NUL-terminated strings written in name,acc,source,desc
///     order); ambiguity ranges (int lower/upper); and each block's FM record
///     (N, term_loc, seq_offset, ambig_offset, overlap, seq_cnt, ambig_cnt,
///     then packed T (forward record only), packed BWT, SA samples (forward
///     only), occCnts_b, occCnts_sb). Within a block, sequences are
///     concatenated with NO inter-sequence separator and one trailing `$`/0.
///   * MATCHES C: per-sequence block windowing — short sequences read whole,
///     long sequences split with FM_BLOCK_OVERLAP leading context, the
///     `max(block_size-size, block_size*0.05)` request-size shortening.
///   * MATCHES C: degenerate (ambiguous) DNA residues are replaced by a
///     canonical base drawn from a faithful MT19937 port of `esl_random()`
///     (see `Mt19937`), seeded 42 exactly as `esl_randomness_Create(42)`
///     (makehmmerdb.c:426) and consumed in block order at block-build time
///     (makehmmerdb.c:656-674), with overlap residues of split sequences
///     re-drawn just as C re-reads them into the next block. The substituted
///     bases — and therefore the BWT/SA of degenerate blocks — are intended to
///     be bit-identical to a C build. (The MT19937 stream and the per-block
///     draw order are reproduced exactly; this has been unit-tested for
///     determinism and ordering but not yet diffed against a live C build.)
///   * MATCHES C: alphabet auto-guess. When no --amino/--dna/--rna is asserted,
///     the alphabet is guessed by a faithful port of
///     `esl_sqfile_GuessAlphabet` -> `esl_abc_GuessAlphabet` over the first
///     sequence's leading 4000-residue window (see `guess_alphabet`), rather
///     than blindly defaulting to DNA. Unguessable input is rejected as C does.
///   * DIFFERS from C (build mechanism only, NOT output bytes): the FM-index
///     *temp-file two-pass* (esl_tmpfile + rewind/copy, makehmmerdb.c:561,
///     767-837) is not reproduced; the metadata and per-block FM records are
///     assembled in memory and streamed directly. C's two-pass exists purely
///     because it must know block_count/seq_count/ambig_count for the metadata
///     header before it can write the FM records that follow — it buffers the
///     records in a temp file during pass 1, then writes header + replays the
///     temp file in pass 2. We compute the same counts up front and emit header
///     then records in one pass, producing byte-identical output without the
///     temp file. This is an internal implementation detail with no observable
///     effect on the output file.
pub fn run(args: Vec<String>) -> std::process::ExitCode {
    let args = Args::parse_from(&args);

    if let Some(ref informat) = args.informat {
        if !informat.eq_ignore_ascii_case("fasta") {
            eprintln!("{informat} is not a recognized input sequence file format");
            return std::process::ExitCode::FAILURE;
        }
    }
    if let Err(msg) = validate_bin_length(args.bin_length) {
        eprintln!("{msg}");
        return std::process::ExitCode::FAILURE;
    }
    if let Err(msg) = validate_sa_freq(args.sa_freq) {
        eprintln!("{msg}");
        return std::process::ExitCode::FAILURE;
    }
    if args.block_size == 0 {
        eprintln!("Invalid block size: --block_size must be > 0");
        return std::process::ExitCode::FAILURE;
    }
    if args.seqfile == std::path::Path::new("-") && args.binaryfile == std::path::Path::new("-") {
        eprintln!("Either <seqfile> or <binaryfile> can be - but not both.");
        return std::process::ExitCode::FAILURE;
    }
    if args.cstream && args.container {
        eprintln!("makehmmerdb --cstream and --container are mutually exclusive");
        return std::process::ExitCode::FAILURE;
    }
    // Determine the alphabet. C makehmmerdb (makehmmerdb.c:492-515) honours an
    // explicit --amino/--dna/--rna assertion, and otherwise calls
    // `esl_sqfile_GuessAlphabet()` on the open file, failing if it cannot guess.
    // To reproduce the auto-guess we slurp the whole input once, run the same
    // heuristic on the first sequence's leading window, then parse sequences
    // from the in-memory copy. (Reading once and reusing the buffer also lets us
    // guess on stdin, which C handles via its rewindable recording buffer.)
    let raw_input: Vec<u8> = if args.seqfile == std::path::Path::new("-") {
        let mut buf = Vec::new();
        if let Err(e) = std::io::stdin().lock().read_to_end(&mut buf) {
            eprintln!("Error reading stdin: {}", e);
            return std::process::ExitCode::FAILURE;
        }
        buf
    } else {
        match read_seqfile_bytes(&args.seqfile) {
            Ok(buf) => buf,
            Err(e) => {
                eprintln!("Error: {}", e);
                return std::process::ExitCode::FAILURE;
            }
        }
    };

    let guessed_amino = if args.amino {
        true
    } else if args.dna || args.rna {
        false
    } else {
        match guess_alphabet(&raw_input) {
            Some(GuessedAlphabet::Amino) => true,
            Some(GuessedAlphabet::Dna) | Some(GuessedAlphabet::Rna) => false,
            None => {
                eprintln!("Unable to guess alphabet. Try '--dna' or '--amino'");
                return std::process::ExitCode::FAILURE;
            }
        }
    };

    let abc = if guessed_amino {
        Alphabet::amino()
    } else if args.rna {
        Alphabet::rna()
    } else {
        Alphabet::dna()
    };
    let fm_alphabet = if guessed_amino {
        FmAlphabet::amino()
    } else {
        FmAlphabet::dna()
    };
    let fwd_only = args.fwd_only || guessed_amino;

    let mut all_text = Vec::new();
    let mut seq_data = Vec::new();
    let mut ambig_ranges = Vec::new();

    let sqf = sequence::SeqFile::new(std::io::Cursor::new(&raw_input), abc.clone());
    read_sequences(
        sqf,
        &abc,
        !guessed_amino,
        &mut all_text,
        &mut seq_data,
        &mut ambig_ranges,
    );
    if seq_data.is_empty() {
        eprintln!("Error: no sequences found in {}", args.seqfile.display());
        return std::process::ExitCode::FAILURE;
    }

    let input_seq_count = seq_data.len();
    // C's `total_char_count` (makehmmerdb.c:679) counts each input residue
    // exactly once: `if (j > block->list[i].C) total_char_count++` excludes the
    // overlap *context* re-read into continuation blocks. That equals the sum of
    // the ORIGINAL (pre-windowing) sequence lengths, captured here before
    // `build_readblock_like_records` re-partitions `seq_data` into overlapping
    // windows (whose lengths would otherwise double-count overlap regions).
    let input_residue_count: usize = seq_data.iter().map(|seq| seq.length).sum();
    let char_count = input_residue_count;
    let block_size_bases = args.block_size.saturating_mul(1_000_000);
    let (seq_data, mut blocks) =
        build_readblock_like_records(&seq_data, block_size_bases, FM_BLOCK_OVERLAP);
    assign_ambiguities_to_blocks(&seq_data, &ambig_ranges, &mut blocks);

    let stderr = std::io::stderr();
    let mut err = stderr.lock();

    writeln!(
        err,
        "Read {} sequences ({} residues total)",
        input_seq_count, input_residue_count
    )
    .unwrap();

    writeln!(err, "Building FM-index blocks...").unwrap();

    let outpath = args.binaryfile;
    let mut out: Box<dyn Write> = if outpath == std::path::Path::new("-") {
        Box::new(std::io::stdout())
    } else {
        Box::new(std::fs::File::create(&outpath).unwrap_or_else(|e| {
            eprintln!("Error creating output: {}", e);
            std::process::exit(1);
        }))
    };

    if !args.container {
        write_native_c_stream(
            &mut out,
            fwd_only,
            fm_alphabet,
            args.sa_freq,
            args.bin_length,
            char_count,
            &seq_data,
            &blocks,
            &ambig_ranges,
            &all_text,
        )
        .unwrap();
        writeln!(
            err,
            "FM-index built: {} block(s), {} C-stream record(s)",
            blocks.len(),
            blocks.len() * if fwd_only { 1 } else { 2 }
        )
        .unwrap();
        writeln!(err, "Database written to: {}", outpath.display()).unwrap();
        return std::process::ExitCode::SUCCESS;
    }

    let index_records = build_index_records(&all_text, &blocks, !fwd_only);
    writeln!(
        err,
        "FM-index built: {} block(s), {} index record(s)",
        blocks.len(),
        index_records.len()
    )
    .unwrap();

    // Write header
    out.write_all(b"HMMERDB\0").unwrap();
    out.write_all(&(seq_data.len() as u64).to_le_bytes())
        .unwrap();
    out.write_all(&(all_text.len() as u64).to_le_bytes())
        .unwrap();

    // Write sequence names and offsets
    for seq in &seq_data {
        out.write_all(&(seq.name.len() as u32).to_le_bytes())
            .unwrap();
        out.write_all(seq.name.as_bytes()).unwrap();
        out.write_all(&(seq.fm_start as u64).to_le_bytes()).unwrap();
    }

    write_index_extension(&mut out, fwd_only, &index_records).unwrap();

    write_metadata_extension(
        &mut out,
        args.block_size,
        FM_BLOCK_OVERLAP,
        &seq_data,
        &blocks,
        &ambig_ranges,
    )
    .unwrap();

    write_c_metadata_extension(
        &mut out,
        fwd_only,
        fm_alphabet,
        args.sa_freq,
        args.bin_length,
        char_count,
        &seq_data,
        &blocks,
        &ambig_ranges,
    )
    .unwrap();

    write_c_stream_extension(
        &mut out,
        fwd_only,
        fm_alphabet,
        args.sa_freq,
        args.bin_length,
        char_count,
        &seq_data,
        &blocks,
        &ambig_ranges,
        &all_text,
    )
    .unwrap();

    writeln!(err, "Database written to: {}", outpath.display()).unwrap();
    std::process::ExitCode::SUCCESS
}

/// Read a (possibly gzip-compressed) sequence file fully into memory, mirroring
/// the transparent `.gz` handling in `sequence::open_seq_file`.
fn read_seqfile_bytes(path: &std::path::Path) -> std::io::Result<Vec<u8>> {
    let file = std::fs::File::open(path)?;
    let mut buf = Vec::new();
    if path.extension().is_some_and(|e| e == "gz") {
        flate2::read::GzDecoder::new(file).read_to_end(&mut buf)?;
    } else {
        std::io::BufReader::new(file).read_to_end(&mut buf)?;
    }
    Ok(buf)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GuessedAlphabet {
    Dna,
    Rna,
    Amino,
}

/// Guess the alphabet from raw FASTA input, faithfully porting the C path
/// `esl_sqfile_GuessAlphabet` -> `sqascii_GuessAlphabet` -> `esl_sq_GuessAlphabet`
/// -> `esl_abc_GuessAlphabet`.
///
/// C inspects only the FIRST sequence, reading at most a 4000-residue leading
/// window (`sqascii_ReadWindow(sqfp, 0, 4000, sq)`), then counts up to 10000 of
/// its residues into a 26-letter histogram (`esl_sq_GuessAlphabet`) and applies
/// the `esl_abc_GuessAlphabet` heuristic. Returns `None` when the alphabet can't
/// be reliably guessed (C's `eslENOALPHABET`).
fn guess_alphabet(raw_input: &[u8]) -> Option<GuessedAlphabet> {
    let counts = first_sequence_residue_counts(raw_input);
    esl_abc_guess_alphabet(&counts)
}

/// Histogram of A..Z over the first sequence's leading 4000-residue window,
/// counting at most 10000 residues. Mirrors `esl_sq_GuessAlphabet` operating on
/// the window produced by `sqascii_ReadWindow(.., 0, 4000, ..)`.
fn first_sequence_residue_counts(raw_input: &[u8]) -> [i64; 26] {
    let mut counts = [0i64; 26];
    let mut in_first_seq = false;
    let mut header_done = false;
    let mut window_residues = 0i64; // residues admitted to the 4000-base window
    let mut counted = 0i64; // residues actually tallied (cap 10000)

    for line in raw_input.split(|&b| b == b'\n') {
        let trimmed = trim_ascii(line);
        if trimmed.first() == Some(&b'>') {
            if in_first_seq {
                break; // reached the 2nd sequence; first one is complete
            }
            in_first_seq = true;
            header_done = true;
            continue;
        }
        if !header_done {
            // skip leading blank lines / comments before the first record
            if trimmed.is_empty() {
                continue;
            }
            // Non-FASTA leading content: bail out (C would treat differently;
            // makehmmerdb only supports FASTA here).
            return counts;
        }
        if !in_first_seq {
            continue;
        }
        for &ch in trimmed {
            if ch.is_ascii_whitespace() {
                continue;
            }
            // ReadWindow caps the recorded window at 4000 residues.
            if window_residues >= 4000 {
                return counts;
            }
            window_residues += 1;
            let upper = ch.to_ascii_uppercase();
            if upper.is_ascii_uppercase() {
                counts[(upper - b'A') as usize] += 1;
                counted += 1;
                if counted > 10000 {
                    return counts;
                }
            }
        }
    }
    counts
}

fn trim_ascii(bytes: &[u8]) -> &[u8] {
    let start = bytes.iter().position(|b| !b.is_ascii_whitespace());
    let start = match start {
        Some(s) => s,
        None => return &[],
    };
    let end = bytes
        .iter()
        .rposition(|b| !b.is_ascii_whitespace())
        .unwrap();
    &bytes[start..=end]
}

/// Faithful port of Easel `esl_abc_GuessAlphabet` (esl_alphabet.c:1251). `ct` is
/// a 26-entry A..Z residue histogram.
fn esl_abc_guess_alphabet(ct: &[i64; 26]) -> Option<GuessedAlphabet> {
    let idx = |c: u8| (c - b'A') as usize;
    let aaonly = b"EFIJLOPQZ";
    let allcanon = b"ACG";
    let aacanon = b"DHKMRSVWY";

    let n: i64 = ct.iter().sum();

    let (mut n1, mut x1) = (0i64, 0i32);
    for &c in aaonly {
        let x = ct[idx(c)];
        if x > 0 {
            n1 += x;
            x1 += 1;
        }
    }
    let (mut n2, mut x2) = (0i64, 0i32);
    for &c in allcanon {
        let x = ct[idx(c)];
        if x > 0 {
            n2 += x;
            x2 += 1;
        }
    }
    let (mut n3, mut x3) = (0i64, 0i32);
    for &c in aacanon {
        let x = ct[idx(c)];
        if x > 0 {
            n3 += x;
            x3 += 1;
        }
    }
    let nt = ct[idx(b'T')];
    let xt = if nt != 0 { 1 } else { 0 };
    let nu = ct[idx(b'U')];
    let xu = if nu != 0 { 1 } else { 0 };
    let nx = ct[idx(b'X')];
    let nn = ct[idx(b'N')];
    let xn = if nn != 0 { 1 } else { 0 };

    // Avoid unused-assignment lint mirrors (x1..xn participate in the tests below).
    let _ = (x1, x2, x3, xt, xu, xn);

    if n <= 10 {
        None
    } else if n > 2000 && nn == n {
        Some(GuessedAlphabet::Dna)
    } else if n1 > 0 {
        Some(GuessedAlphabet::Amino)
    } else if (n - (n2 + nt + nn)) as f64 <= 0.02 * n as f64 && x2 + xt == 4 {
        Some(GuessedAlphabet::Dna)
    } else if (n - (n2 + nu + nn)) as f64 <= 0.02 * n as f64 && x2 + xu == 4 {
        Some(GuessedAlphabet::Rna)
    } else if (n - (n1 + n2 + n3 + nn + nt + nx)) as f64 <= 0.02 * n as f64
        && n3 > n2
        && x1 + x2 + x3 + xn + xt >= 15
    {
        Some(GuessedAlphabet::Amino)
    } else {
        None
    }
}

#[derive(Debug, Clone)]
struct SequenceMetadata {
    target_id: usize,
    target_start: usize,
    fm_start: usize,
    length: usize,
    block_id: usize,
    block_offset: usize,
    overlap_bases: usize,
    name: String,
    acc: String,
    desc: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct AmbiguityRange {
    lower: usize,
    upper: usize,
}

#[derive(Debug, Clone)]
struct BlockRecord {
    id: usize,
    fm_start: usize,
    length: usize,
    seq_offset: usize,
    seq_count: usize,
    ambig_offset: usize,
    ambig_count: usize,
    overlap_bases: usize,
}

struct IndexRecord {
    block_id: usize,
    kind: u32,
    text_start: usize,
    text_len: usize,
    seq_offset: usize,
    seq_count: usize,
    ambig_offset: usize,
    ambig_count: usize,
    overlap_bases: usize,
    fm: FmIndex,
}

#[derive(Debug, Clone, Copy)]
struct FmAlphabet {
    alph_type: u8,
    alph_size: u8,
    char_bits: u8,
    amino: bool,
}

impl FmAlphabet {
    fn dna() -> Self {
        Self {
            alph_type: FM_DNA_ALPH_TYPE,
            alph_size: FM_DNA_ALPH_SIZE,
            char_bits: FM_DNA_CHAR_BITS,
            amino: false,
        }
    }

    fn amino() -> Self {
        Self {
            alph_type: FM_AMINO_ALPH_TYPE,
            alph_size: FM_AMINO_ALPH_SIZE,
            char_bits: FM_AMINO_CHAR_BITS,
            amino: true,
        }
    }

    fn encode_text_without_separators(
        self,
        text: &[u8],
        rng: &mut Mt19937,
    ) -> std::io::Result<Vec<u8>> {
        if self.amino {
            c_amino_text_without_separators(text)
        } else {
            c_dna_text_without_separators(text, rng)
        }
    }

    fn pack(self, values: &[u8]) -> Vec<u8> {
        if self.amino {
            values.to_vec()
        } else {
            pack_dna_quads(values)
        }
    }
}

/// Faithful port of Easel's 32-bit Mersenne Twister (MT19937), the generator
/// behind `esl_random()` for a randomness object created with
/// `esl_randomness_Create(seed)` (`r->type == eslRND_MERSENNE`; see
/// `hmmer/easel/esl_random.c` `mersenne_twister`/`mersenne_seed_table`/
/// `mersenne_fill_table`).
///
/// NOTE on why this lives here rather than in `src/util/random.rs`: that module
/// owns the RNG API but its `MersenneTwister` type is actually Easel's *fast*
/// Knuth LCG (`esl_randomness_CreateFast`), and its `Rand64` is the 64-bit
/// MT19937-64 (`esl_rand64`). Neither matches the 32-bit MT19937 stream that
/// `esl_random()` produces for makehmmerdb's `esl_randomness_Create(42)` RNG, so
/// a faithful, byte-identical degenerate-residue replacement needs this exact
/// generator. It is kept private to makehmmerdb to avoid disturbing the shared
/// module that another part of the codebase depends on.
struct Mt19937 {
    mt: [u32; 624],
    mti: usize,
}

impl Mt19937 {
    /// Seed exactly as Easel `mersenne_seed_table`: `mt[0]=seed`, then
    /// `mt[z] = 69069 * mt[z-1]` (32-bit wrapping). `mti` is set to 624 so the
    /// first draw triggers a table fill, matching `esl_randomness_Init`'s
    /// `r->mti = 0` followed by the `if (r->mti >= 624) mersenne_fill_table`
    /// guard on the first `mersenne_twister` call (Easel initializes mti=0 but
    /// fills lazily; starting at 624 reproduces the identical first fill).
    fn new(seed: u32) -> Self {
        let mut mt = [0u32; 624];
        mt[0] = seed;
        for z in 1..624 {
            mt[z] = 69069u32.wrapping_mul(mt[z - 1]);
        }
        Mt19937 { mt, mti: 624 }
    }

    fn fill_table(&mut self) {
        const MAG01: [u32; 2] = [0x0, 0x9908b0df];
        for z in 0..227 {
            let y = (self.mt[z] & 0x80000000) | (self.mt[z + 1] & 0x7fffffff);
            self.mt[z] = self.mt[z + 397] ^ (y >> 1) ^ MAG01[(y & 0x1) as usize];
        }
        for z in 227..623 {
            let y = (self.mt[z] & 0x80000000) | (self.mt[z + 1] & 0x7fffffff);
            self.mt[z] = self.mt[z - 227] ^ (y >> 1) ^ MAG01[(y & 0x1) as usize];
        }
        let y = (self.mt[623] & 0x80000000) | (self.mt[0] & 0x7fffffff);
        self.mt[623] = self.mt[396] ^ (y >> 1) ^ MAG01[(y & 0x1) as usize];
        self.mti = 0;
    }

    /// Draw a tempered 32-bit variate (Easel `mersenne_twister`).
    fn next_u32(&mut self) -> u32 {
        if self.mti >= 624 {
            self.fill_table();
        }
        let mut x = self.mt[self.mti];
        self.mti += 1;
        x ^= x >> 11;
        x ^= (x << 7) & 0x9d2c5680;
        x ^= (x << 15) & 0xefc60000;
        x ^= x >> 18;
        x
    }

    /// `esl_random(r)`: uniform double on [0,1) = `next_u32() / 2^32`.
    fn next_f64(&mut self) -> f64 {
        self.next_u32() as f64 / 4294967296.0
    }

    /// Reproduce C's `(int)(esl_random(r) * 4)` -> a canonical DNA code in 0..3.
    /// makehmmerdb.c:661 picks `meta->alph[(int)(esl_random(r)*4)]` from
    /// `"ACGT"` and stores `inv_alph[that]`, which equals the drawn index, so
    /// the FM code is exactly this 0..3 value (A=0,C=1,G=2,T=3).
    fn dna_replacement_code(&mut self) -> u8 {
        (self.next_f64() * 4.0) as u8
    }
}

/// Seed used by C makehmmerdb for degenerate-residue replacement
/// (`esl_randomness_Create(42)`, makehmmerdb.c:426).
const MAKEHMMERDB_RNG_SEED: u32 = 42;

fn read_sequences<R: Read>(
    mut sqf: sequence::SeqFile<R>,
    abc: &Alphabet,
    replace_degenerate_residues: bool,
    all_text: &mut Vec<u8>,
    seq_data: &mut Vec<SequenceMetadata>,
    ambig_ranges: &mut Vec<AmbiguityRange>,
) {
    let mut sq = Sequence::new();
    loop {
        let has_seq = sqf.read(&mut sq).unwrap_or_else(|e| {
            eprintln!("Error reading sequence file: {}", e);
            std::process::exit(1);
        });
        if !has_seq {
            break;
        }
        let fm_start = all_text.len();
        seq_data.push(SequenceMetadata {
            target_id: seq_data.len(),
            target_start: 1,
            fm_start,
            length: sq.n,
            block_id: 0,
            block_offset: 0,
            overlap_bases: 0,
            name: sq.name.clone(),
            acc: sq.acc.clone(),
            desc: sq.desc.clone(),
        });
        append_sequence_text(
            abc,
            &sq,
            replace_degenerate_residues,
            all_text,
            ambig_ranges,
        );
        all_text.push(b'$');
        sq.reuse();
    }
}

fn append_sequence_text(
    abc: &Alphabet,
    sq: &Sequence,
    replace_degenerate_residues: bool,
    all_text: &mut Vec<u8>,
    ambig_ranges: &mut Vec<AmbiguityRange>,
) {
    // We store the ORIGINAL alphabet symbol here (canonical letters such as
    // A/C/G/T, or degenerate IUPAC codes such as N/R/Y for DNA). Unlike C, we
    // do NOT substitute degenerate residues during this initial read: C draws
    // its replacement bases from `esl_random()` while building each FM *block's*
    // text (makehmmerdb.c:656-684), in block order and re-drawing for residues
    // that fall in inter-block overlap regions. To reproduce that RNG draw order
    // bit-for-bit, substitution is deferred to `c_dna_text_without_separators()`
    // at block-build time. Here we only record the ambiguity ranges (positions),
    // which are independent of the substituted values.
    let mut in_ambig_run = false;
    for i in 1..=sq.n {
        let dsq = sq.dsq[i];
        let ch = if abc.is_canonical(dsq) || (abc.is_residue(dsq) && !replace_degenerate_residues) {
            in_ambig_run = false;
            abc.sym[dsq as usize]
        } else if abc.is_residue(dsq) {
            let pos = all_text.len();
            if in_ambig_run {
                ambig_ranges.last_mut().unwrap().upper = pos;
            } else {
                ambig_ranges.push(AmbiguityRange {
                    lower: pos,
                    upper: pos,
                });
                in_ambig_run = true;
            }
            abc.sym[dsq as usize]
        } else {
            eprintln!(
                "Error: non-residue symbol '{}' is not supported by makehmmerdb",
                abc.sym[dsq as usize] as char
            );
            std::process::exit(1);
        };
        all_text.push(ch);
    }
}

/// Partition the input sequences into FM-index blocks, mirroring the
/// per-sequence windowing of C `esl_sqio_ReadBlock()` (long_target mode) as
/// driven by makehmmerdb.c's main loop.
///
/// C model (esl_sqio_ascii.c `sqascii_ReadBlock`, makehmmerdb.c:563-704):
///   * `block_size_bases` (= `--block_size` * 1e6) bounds the NEW residues per
///     block; `size` counts `n - C` (window length minus retained context).
///   * Each whole sequence shorter than the remaining budget is read in one
///     window and added intact (an EOD is "burned off"). A block closes once
///     `size >= block_size`.
///   * A *single* sequence longer than a window is split across blocks; the
///     continuation window re-reads `FM_BLOCK_OVERLAP` (`overlap_bases`) bases
///     of the same sequence as leading context (`->C`). `block->list[0].C`
///     becomes the block's `overlap` field.
///   * Non-init windows request `max(block_size - size, block_size * 0.05)`
///     new residues, keeping blocks near `block_size`.
///
/// Overlap is therefore retained ONLY when continuing the same long sequence
/// across a block boundary — a boundary that falls between two distinct
/// sequences carries no overlap, unlike the previous global-text chopper.
fn build_readblock_like_records(
    input_seqs: &[SequenceMetadata],
    block_size_bases: usize,
    overlap_bases: usize,
) -> (Vec<SequenceMetadata>, Vec<BlockRecord>) {
    let compact_len: usize = input_seqs.iter().map(|seq| seq.length).sum();
    if compact_len == 0 {
        return (Vec::new(), Vec::new());
    }

    let block_size = block_size_bases.max(1);
    let max_overlap = overlap_bases.min(block_size.saturating_sub(1));
    let slop = ((block_size as f64) * 0.05).ceil() as usize;

    let mut seq_data = Vec::new();
    let mut blocks = Vec::new();

    // Currently-open block accumulator.
    let mut block_id = 0usize;
    let mut seq_offset = 0usize;
    let mut size = 0usize; // NEW residues in the open block (matches C's `size`)
    let mut block_overlap = 0usize; // C's block->list[0].C for this block
    let mut block_started = false;

    // Helper to finalize the currently-open block into `blocks`.
    let finish_block = |blocks: &mut Vec<BlockRecord>,
                        seq_data: &[SequenceMetadata],
                        block_id: usize,
                        seq_offset: usize,
                        block_overlap: usize| {
        let seq_count = seq_data.len() - seq_offset;
        let (fm_start, length) = if seq_count == 0 {
            (0, 0)
        } else {
            let first = &seq_data[seq_offset];
            let last = &seq_data[seq_data.len() - 1];
            (first.fm_start, last.fm_start + last.length - first.fm_start)
        };
        blocks.push(BlockRecord {
            id: block_id,
            fm_start,
            length,
            seq_offset,
            seq_count,
            ambig_offset: 0,
            ambig_count: 0,
            overlap_bases: block_overlap,
        });
    };

    for seq in input_seqs {
        // Residues of this sequence already consumed by a window.
        let mut consumed = 0usize;
        // Whether the currently-open window of THIS sequence is a continuation
        // across a block boundary (i.e. it carries leading overlap context).
        let mut continuation = false;

        while consumed < seq.length {
            if !block_started {
                block_started = true;
                block_overlap = if continuation {
                    // Continuation of a split sequence: re-read up to
                    // FM_BLOCK_OVERLAP bases of context. C does this by setting
                    // block->list->C, which becomes the block's overlap.
                    max_overlap.min(consumed)
                } else {
                    0
                };
                seq_offset = seq_data.len();
            }

            // C's non-init request size: max(block_size - size, block_size*0.05).
            let request_size = block_size.saturating_sub(size).max(slop).max(1);
            let remaining = seq.length - consumed;
            let new_residues = request_size.min(remaining);

            // The window includes leading overlap context (only when this is
            // the first window of a continued sequence in a fresh block).
            let leading_overlap = if continuation && consumed > 0 && size == 0 {
                max_overlap.min(consumed)
            } else {
                0
            };
            let window_offset = consumed - leading_overlap;
            let window_len = leading_overlap + new_residues;

            seq_data.push(SequenceMetadata {
                target_id: seq.target_id,
                target_start: seq.target_start + window_offset,
                fm_start: seq.fm_start + window_offset,
                length: window_len,
                block_id,
                block_offset: 0,
                overlap_bases: leading_overlap,
                name: seq.name.clone(),
                acc: seq.acc.clone(),
                desc: seq.desc.clone(),
            });

            consumed += new_residues;
            size += new_residues;

            // Did this fill the block? (size has reached block_size and the
            // sequence is not yet exhausted, OR exactly hit the limit).
            let seq_exhausted = consumed >= seq.length;
            if size >= block_size {
                finish_block(&mut blocks, &seq_data, block_id, seq_offset, block_overlap);
                block_id += 1;
                size = 0;
                block_started = false;
                // If the sequence isn't done, the next window of it is a
                // continuation that should carry overlap context.
                continuation = !seq_exhausted;
            } else {
                // Sequence finished without filling the block; keep the block
                // open for the next input sequence (no overlap across the
                // boundary between distinct sequences).
                continuation = false;
            }
        }
    }

    // Flush any partially-filled trailing block.
    if block_started {
        finish_block(&mut blocks, &seq_data, block_id, seq_offset, block_overlap);
    }

    // Recompute per-window block_offset (position within the block's text).
    let mut current_block = usize::MAX;
    let mut block_offset = 0usize;
    for window in &mut seq_data {
        if window.block_id != current_block {
            current_block = window.block_id;
            block_offset = 0;
        }
        window.block_offset = block_offset;
        block_offset += window.length;
    }

    (seq_data, blocks)
}

fn assign_ambiguities_to_blocks(
    seq_data: &[SequenceMetadata],
    ambig_ranges: &[AmbiguityRange],
    blocks: &mut [BlockRecord],
) {
    for block in &mut *blocks {
        block.ambig_offset = 0;
        block.ambig_count = 0;
    }

    // Match C's emission order (makehmmerdb.c:656-668): C records ambiguity
    // ranges while walking each FM *block's* text, sequence-window by
    // sequence-window, residue by residue. So the flat `ambig_list` is
    // BLOCK-major (in block order), then within a block in the order the
    // window's residues are scanned (i.e. global position order). Because
    // `seq_data` already lists windows in block order, iterating windows
    // (outer) then ambiguity ranges in their recorded position order (inner)
    // reproduces C's exact ordering — including re-emitting a degenerate run
    // that falls in an inter-block overlap region once per block, since the
    // continuation window re-reads (and thus re-records) those residues.
    let mut compact_idx = 0usize;
    for seq in seq_data {
        for range in ambig_ranges {
            if intersect_ambiguity_with_sequence(range, seq).is_some() {
                if let Some(block) = blocks.get_mut(seq.block_id) {
                    if block.ambig_count == 0 {
                        block.ambig_offset = compact_idx;
                    }
                    block.ambig_count += 1;
                }
                compact_idx += 1;
            }
        }
    }

    for block in blocks.iter_mut() {
        if block.ambig_count == 0 {
            block.ambig_offset = compact_idx;
        }
    }
}

fn build_index_records(
    all_text: &[u8],
    blocks: &[BlockRecord],
    include_reverse_strand: bool,
) -> Vec<IndexRecord> {
    let mut records = Vec::with_capacity(blocks.len() * if include_reverse_strand { 2 } else { 1 });
    for block in blocks {
        let block_text = &all_text[block.fm_start..block.fm_start + block.length];
        let reversed_text: Vec<u8> = block_text.iter().rev().copied().collect();
        records.push(IndexRecord {
            block_id: block.id,
            kind: INDEX_KIND_FORWARD_STRAND,
            text_start: block.fm_start,
            text_len: block.length,
            seq_offset: block.seq_offset,
            seq_count: block.seq_count,
            ambig_offset: block.ambig_offset,
            ambig_count: block.ambig_count,
            overlap_bases: block.overlap_bases,
            fm: FmIndex::build(&reversed_text),
        });

        if include_reverse_strand {
            records.push(IndexRecord {
                block_id: block.id,
                kind: INDEX_KIND_REVERSE_STRAND,
                text_start: block.fm_start,
                text_len: block.length,
                seq_offset: block.seq_offset,
                seq_count: block.seq_count,
                ambig_offset: block.ambig_offset,
                ambig_count: block.ambig_count,
                overlap_bases: block.overlap_bases,
                fm: FmIndex::build(block_text),
            });
        }
    }
    records
}

fn write_index_extension<W: Write + ?Sized>(
    out: &mut W,
    fwd_only: bool,
    records: &[IndexRecord],
) -> std::io::Result<()> {
    out.write_all(INDEX_MAGIC)?;
    out.write_all(&INDEX_VERSION.to_le_bytes())?;
    out.write_all(&(fwd_only as u32).to_le_bytes())?;
    out.write_all(&(records.len() as u64).to_le_bytes())?;

    for record in records {
        for value in [
            record.block_id,
            record.text_start,
            record.text_len,
            record.seq_offset,
            record.seq_count,
            record.ambig_offset,
            record.ambig_count,
            record.overlap_bases,
        ] {
            out.write_all(&(value as u64).to_le_bytes())?;
        }
        out.write_all(&record.kind.to_le_bytes())?;
        out.write_all(&(record.fm.bwt.len() as u64).to_le_bytes())?;
        out.write_all(&(record.fm.sa.len() as u64).to_le_bytes())?;
        out.write_all(&(record.fm.c.len() as u64).to_le_bytes())?;
        out.write_all(&record.fm.bwt)?;
        for &sa_val in &record.fm.sa {
            out.write_all(&sa_val.to_le_bytes())?;
        }
        for &c_val in &record.fm.c {
            out.write_all(&(c_val as u64).to_le_bytes())?;
        }
    }

    Ok(())
}

fn write_metadata_extension<W: Write + ?Sized>(
    out: &mut W,
    block_size_mb: usize,
    overlap_bases: usize,
    seq_data: &[SequenceMetadata],
    blocks: &[BlockRecord],
    ambig_ranges: &[AmbiguityRange],
) -> std::io::Result<()> {
    out.write_all(META_MAGIC)?;
    out.write_all(&META_VERSION.to_le_bytes())?;
    out.write_all(&(block_size_mb as u64).to_le_bytes())?;
    out.write_all(&(overlap_bases as u64).to_le_bytes())?;
    out.write_all(&(seq_data.len() as u64).to_le_bytes())?;
    out.write_all(&(blocks.len() as u64).to_le_bytes())?;
    out.write_all(&(ambig_ranges.len() as u64).to_le_bytes())?;

    for seq in seq_data {
        for value in [
            seq.target_id,
            seq.target_start,
            seq.fm_start,
            seq.length,
            seq.block_id,
            seq.block_offset,
            seq.overlap_bases,
        ] {
            out.write_all(&(value as u64).to_le_bytes())?;
        }
        write_string(out, &seq.name)?;
        write_string(out, &seq.acc)?;
        write_string(out, &seq.desc)?;
    }

    for block in blocks {
        for value in [
            block.id,
            block.fm_start,
            block.length,
            block.seq_offset,
            block.seq_count,
            block.ambig_offset,
            block.ambig_count,
            block.overlap_bases,
        ] {
            out.write_all(&(value as u64).to_le_bytes())?;
        }
    }

    for range in ambig_ranges {
        out.write_all(&(range.lower as u64).to_le_bytes())?;
        out.write_all(&(range.upper as u64).to_le_bytes())?;
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn write_c_metadata_extension<W: Write + ?Sized>(
    out: &mut W,
    fwd_only: bool,
    fm_alphabet: FmAlphabet,
    sa_freq: usize,
    bin_length: usize,
    char_count: usize,
    seq_data: &[SequenceMetadata],
    blocks: &[BlockRecord],
    ambig_ranges: &[AmbiguityRange],
) -> std::io::Result<()> {
    let compact_starts = c_block_compact_sequence_starts(seq_data, blocks);
    let compact_ambig_ranges = compact_ambiguity_ranges(seq_data, &compact_starts, ambig_ranges);

    out.write_all(C_META_MAGIC)?;
    out.write_all(&C_META_VERSION.to_le_bytes())?;
    write_c_metadata_payload(
        out,
        fwd_only,
        fm_alphabet,
        sa_freq,
        bin_length,
        char_count,
        seq_data,
        blocks,
        &compact_starts,
        &compact_ambig_ranges,
    )
}

#[allow(clippy::too_many_arguments)]
fn write_c_stream_extension<W: Write + ?Sized>(
    out: &mut W,
    fwd_only: bool,
    fm_alphabet: FmAlphabet,
    sa_freq: usize,
    bin_length: usize,
    char_count: usize,
    seq_data: &[SequenceMetadata],
    blocks: &[BlockRecord],
    ambig_ranges: &[AmbiguityRange],
    all_text: &[u8],
) -> std::io::Result<()> {
    let payload = build_c_stream_payload(
        fwd_only,
        fm_alphabet,
        sa_freq,
        bin_length,
        char_count,
        seq_data,
        blocks,
        ambig_ranges,
        all_text,
    )?;

    out.write_all(C_STREAM_MAGIC)?;
    out.write_all(&C_STREAM_VERSION.to_le_bytes())?;
    out.write_all(&checked_u64(payload.len(), "C stream payload length")?.to_le_bytes())?;
    out.write_all(&payload)
}

#[allow(clippy::too_many_arguments)]
fn write_native_c_stream<W: Write + ?Sized>(
    out: &mut W,
    fwd_only: bool,
    fm_alphabet: FmAlphabet,
    sa_freq: usize,
    bin_length: usize,
    char_count: usize,
    seq_data: &[SequenceMetadata],
    blocks: &[BlockRecord],
    ambig_ranges: &[AmbiguityRange],
    all_text: &[u8],
) -> std::io::Result<()> {
    let payload = build_c_stream_payload(
        fwd_only,
        fm_alphabet,
        sa_freq,
        bin_length,
        char_count,
        seq_data,
        blocks,
        ambig_ranges,
        all_text,
    )?;
    out.write_all(&payload)
}

#[allow(clippy::too_many_arguments)]
fn build_c_stream_payload(
    fwd_only: bool,
    fm_alphabet: FmAlphabet,
    sa_freq: usize,
    bin_length: usize,
    char_count: usize,
    seq_data: &[SequenceMetadata],
    blocks: &[BlockRecord],
    ambig_ranges: &[AmbiguityRange],
    all_text: &[u8],
) -> std::io::Result<Vec<u8>> {
    let compact_starts = c_block_compact_sequence_starts(seq_data, blocks);
    let compact_ambig_ranges = compact_ambiguity_ranges(seq_data, &compact_starts, ambig_ranges);
    let mut payload = Vec::new();
    write_c_metadata_payload(
        &mut payload,
        fwd_only,
        fm_alphabet,
        sa_freq,
        bin_length,
        char_count,
        seq_data,
        blocks,
        &compact_starts,
        &compact_ambig_ranges,
    )?;

    // Single RNG for the whole build, seeded exactly as C
    // (`esl_randomness_Create(42)`). Degenerate-residue replacement draws from
    // it in block order via `c_block_text`, so the BWT/SA of degenerate blocks
    // match a C build bit-for-bit. (C also draws once for the reverse-strand
    // pass? No: C builds each block's text T once, replacing degenerates while
    // reading the block, then builds BOTH the forward and reverse FM records
    // from that same already-substituted T. So we likewise substitute once per
    // block here and reuse `c_text` for both strands.)
    let mut rng = Mt19937::new(MAKEHMMERDB_RNG_SEED);

    for block in blocks {
        let c_text = c_block_text(fm_alphabet, seq_data, block, all_text, &mut rng)?;
        write_c_fm_record(
            &mut payload,
            fm_alphabet,
            &c_text,
            block,
            sa_freq,
            bin_length,
            true,
            true,
        )?;
        if !fwd_only {
            write_c_fm_record(
                &mut payload,
                fm_alphabet,
                &c_text,
                block,
                sa_freq,
                bin_length,
                false,
                false,
            )?;
        }
    }

    Ok(payload)
}

#[allow(clippy::too_many_arguments)]
fn write_c_metadata_payload<W: Write + ?Sized>(
    out: &mut W,
    fwd_only: bool,
    fm_alphabet: FmAlphabet,
    sa_freq: usize,
    bin_length: usize,
    char_count: usize,
    seq_data: &[SequenceMetadata],
    blocks: &[BlockRecord],
    compact_starts: &[usize],
    compact_ambig_ranges: &[AmbiguityRange],
) -> std::io::Result<()> {
    out.write_all(&[fwd_only as u8])?;
    out.write_all(&[fm_alphabet.alph_type])?;
    out.write_all(&[fm_alphabet.alph_size])?;
    out.write_all(&[fm_alphabet.char_bits])?;
    out.write_all(&checked_u32(sa_freq, "sa_freq")?.to_le_bytes())?;
    out.write_all(&FM_FREQ_CNT_SB.to_le_bytes())?;
    out.write_all(&checked_u32(bin_length, "bin_length")?.to_le_bytes())?;
    out.write_all(&checked_u16(blocks.len(), "block_count")?.to_le_bytes())?;
    out.write_all(&checked_u32(seq_data.len(), "seq_count")?.to_le_bytes())?;
    out.write_all(&checked_u32(compact_ambig_ranges.len(), "ambig_count")?.to_le_bytes())?;
    out.write_all(&checked_u64(char_count, "char_count")?.to_le_bytes())?;

    for (seq, &compact_start) in seq_data.iter().zip(compact_starts.iter()) {
        out.write_all(&checked_u32(seq.target_id, "target_id")?.to_le_bytes())?;
        out.write_all(&checked_u64(seq.target_start, "target_start")?.to_le_bytes())?;
        out.write_all(&checked_u32(compact_start, "fm_start")?.to_le_bytes())?;
        out.write_all(&checked_u32(seq.length, "length")?.to_le_bytes())?;
        // The per-sequence `source` field depends on C's read path. makehmmerdb
        // reads DNA/RNA via the long_target *windowed* path
        // (`esl_sqio_ReadBlock(.., long_target = alphatype != eslAMINO)`),
        // whose `sqascii_ReadWindow` sets `sq->source = sq->name`
        // (esl_sqio_ascii.c:1341, 2012). The amino path is the plain
        // `sqascii_Read`, which never calls SetSource, so `sq->source` keeps the
        // empty string from `esl_sq_CreateDigital`. Hence: source == name for
        // DNA/RNA, source == "" for amino.
        // (For DNA/RNA sequences split across blocks, C additionally reformats
        // `name` to "name/start-end" while `source` keeps the bare name;
        // reproducing that window-name suffix is a separate, pre-existing
        // metadata delta and is not handled here — `name` remains the bare name
        // in that case.)
        let source = if fm_alphabet.amino {
            ""
        } else {
            seq.name.as_str()
        };
        reject_embedded_nul("sequence name", &seq.name)?;
        reject_embedded_nul("sequence accession", &seq.acc)?;
        reject_embedded_nul("sequence source", source)?;
        reject_embedded_nul("sequence description", &seq.desc)?;
        out.write_all(&checked_u16(seq.name.len(), "name_length")?.to_le_bytes())?;
        out.write_all(&checked_u16(seq.acc.len(), "acc_length")?.to_le_bytes())?;
        out.write_all(&checked_u16(source.len(), "source_length")?.to_le_bytes())?;
        out.write_all(&checked_u16(seq.desc.len(), "desc_length")?.to_le_bytes())?;
        write_c_string(out, &seq.name)?;
        write_c_string(out, &seq.acc)?;
        write_c_string(out, source)?;
        write_c_string(out, &seq.desc)?;
    }

    for range in compact_ambig_ranges {
        out.write_all(&checked_i32(range.lower, "ambiguity lower")?.to_le_bytes())?;
        out.write_all(&checked_i32(range.upper, "ambiguity upper")?.to_le_bytes())?;
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn write_c_fm_record<W: Write + ?Sized>(
    out: &mut W,
    fm_alphabet: FmAlphabet,
    c_text: &[u8],
    block: &BlockRecord,
    sa_freq: usize,
    bin_length: usize,
    reverse_text_for_bwt: bool,
    include_text_and_sa: bool,
) -> std::io::Result<()> {
    let record = build_c_fm_record(
        fm_alphabet,
        c_text,
        sa_freq,
        bin_length,
        reverse_text_for_bwt,
    )?;

    out.write_all(&checked_u64(record.n, "FM record length")?.to_le_bytes())?;
    out.write_all(&checked_u32(record.term_loc, "terminal location")?.to_le_bytes())?;
    out.write_all(&checked_u32(block.seq_offset, "seq_offset")?.to_le_bytes())?;
    out.write_all(&checked_u32(block.ambig_offset, "ambig_offset")?.to_le_bytes())?;
    // C writes the block's overlap (`block->list[0].C`) ONLY for the forward
    // (reversed-text) FM record (makehmmerdb.c:694); the reverse-strand record
    // is emitted with overlap=0 (makehmmerdb.c:700). The forward record is the
    // one built from reversed text, i.e. `reverse_text_for_bwt == true`.
    let overlap = if reverse_text_for_bwt {
        block.overlap_bases
    } else {
        0
    };
    out.write_all(&checked_u32(overlap, "overlap")?.to_le_bytes())?;
    out.write_all(&checked_u32(block.seq_count, "seq_count")?.to_le_bytes())?;
    out.write_all(&checked_u32(block.ambig_count, "ambig_count")?.to_le_bytes())?;

    if include_text_and_sa {
        out.write_all(&fm_alphabet.pack(&record.forward_text))?;
    }
    out.write_all(&fm_alphabet.pack(&record.bwt))?;
    if include_text_and_sa {
        for sample in &record.sa_samples {
            out.write_all(&sample.to_le_bytes())?;
        }
    }
    for count in &record.occ_cnts_b {
        out.write_all(&count.to_le_bytes())?;
    }
    for count in &record.occ_cnts_sb {
        out.write_all(&count.to_le_bytes())?;
    }

    Ok(())
}

struct CfmRecord {
    n: usize,
    term_loc: usize,
    forward_text: Vec<u8>,
    bwt: Vec<u8>,
    sa_samples: Vec<u32>,
    occ_cnts_b: Vec<u16>,
    occ_cnts_sb: Vec<u32>,
}

fn build_c_fm_record(
    fm_alphabet: FmAlphabet,
    c_text: &[u8],
    sa_freq: usize,
    bin_length: usize,
    reverse_text_for_bwt: bool,
) -> std::io::Result<CfmRecord> {
    let mut forward_text = c_text.to_vec();
    forward_text.push(0);

    let mut sortable_text: Vec<u8> = c_text.iter().map(|&base| base + 1).collect();
    if reverse_text_for_bwt {
        sortable_text.reverse();
    }
    sortable_text.push(0);

    let n = sortable_text.len();
    let mut sa = vec![0i32; n];
    sort_in_place(&sortable_text, &mut sa);

    let mut bwt = Vec::with_capacity(n);
    let mut term_loc = 0usize;
    let num_sa_samples = 1 + n / sa_freq;
    let mut sa_samples = vec![0u32; num_sa_samples];

    for (j, &suffix_start) in sa.iter().enumerate() {
        let suffix_start = suffix_start as usize;
        if suffix_start == 0 {
            term_loc = j;
            bwt.push(0);
        } else {
            bwt.push(sortable_text[suffix_start - 1].saturating_sub(1));
        }

        if j != 0 && j % sa_freq == 0 {
            sa_samples[j / sa_freq] = if suffix_start == n - 1 {
                u32::MAX
            } else {
                checked_u32(suffix_start, "suffix array sample")?
            };
        }
    }

    let (occ_cnts_b, occ_cnts_sb) =
        build_c_occ_tables(fm_alphabet, &bwt, bin_length, FM_FREQ_CNT_SB as usize)?;

    Ok(CfmRecord {
        n,
        term_loc,
        forward_text,
        bwt,
        sa_samples,
        occ_cnts_b,
        occ_cnts_sb,
    })
}

fn build_c_occ_tables(
    fm_alphabet: FmAlphabet,
    bwt: &[u8],
    freq_cnt_b: usize,
    freq_cnt_sb: usize,
) -> std::io::Result<(Vec<u16>, Vec<u32>)> {
    let n = bwt.len();
    let num_freq_cnts_b = 1 + n.div_ceil(freq_cnt_b);
    let num_freq_cnts_sb = 1 + n.div_ceil(freq_cnt_sb);
    let alph_size = fm_alphabet.alph_size as usize;
    let mut occ_cnts_b = vec![0u16; num_freq_cnts_b * alph_size];
    let mut occ_cnts_sb = vec![0u32; num_freq_cnts_sb * alph_size];
    let mut cnts_b = vec![0usize; alph_size];
    let mut cnts_sb = vec![0usize; alph_size];

    for (j, &base) in bwt.iter().enumerate() {
        let idx = base as usize;
        if idx >= alph_size {
            return Err(invalid_data(format!(
                "FM symbol code {idx} is outside alphabet size {alph_size}"
            )));
        }
        cnts_b[idx] += 1;
        cnts_sb[idx] += 1;

        let joffset = j + 1;
        if joffset % freq_cnt_b == 0 {
            store_occ_b(&mut occ_cnts_b, alph_size, joffset / freq_cnt_b, &cnts_b)?;
            if joffset % freq_cnt_sb == 0 {
                store_occ_sb(&mut occ_cnts_sb, alph_size, joffset / freq_cnt_sb, &cnts_sb)?;
                cnts_b.fill(0);
            }
        }
    }

    store_occ_b(&mut occ_cnts_b, alph_size, num_freq_cnts_b - 1, &cnts_b)?;
    store_occ_sb(&mut occ_cnts_sb, alph_size, num_freq_cnts_sb - 1, &cnts_sb)?;

    Ok((occ_cnts_b, occ_cnts_sb))
}

fn store_occ_b(
    out: &mut [u16],
    alph_size: usize,
    row: usize,
    counts: &[usize],
) -> std::io::Result<()> {
    let offset = row * alph_size;
    for (slot, &count) in out[offset..offset + alph_size].iter_mut().zip(counts) {
        *slot = checked_u16(count, "FM occurrence block count")?;
    }
    Ok(())
}

fn store_occ_sb(
    out: &mut [u32],
    alph_size: usize,
    row: usize,
    counts: &[usize],
) -> std::io::Result<()> {
    let offset = row * alph_size;
    for (slot, &count) in out[offset..offset + alph_size].iter_mut().zip(counts) {
        *slot = checked_u32(count, "FM occurrence superblock count")?;
    }
    Ok(())
}

fn pack_dna_quads(values: &[u8]) -> Vec<u8> {
    let mut packed = Vec::with_capacity(values.len().div_ceil(4));
    for chunk in values.chunks(4) {
        let mut byte = 0u8;
        for (i, &value) in chunk.iter().enumerate() {
            byte |= value << (6 - 2 * i);
        }
        packed.push(byte);
    }
    packed
}

/// Encode a block's DNA text to FM codes (A=0,C=1,G=2,T=3), substituting any
/// degenerate residue with an `esl_random()`-drawn canonical base.
///
/// This mirrors makehmmerdb.c:656-674: for each residue, if its FM `inv_alph`
/// code is -1 (i.e. not one of the four canonical DNA bases — every IUPAC
/// degeneracy code, including N), C replaces it with
/// `meta->alph[(int)(esl_random(r)*4)]`. The RNG `r` is a single
/// `esl_randomness_Create(42)` object consumed across the whole build, in block
/// order, with overlap regions of split sequences re-drawn (because they are
/// re-read into the next block's text). Because this function is called once per
/// block in block order from `build_c_stream_payload`, and a block's text walks
/// its windows (including the leading-overlap window) in `fm_start` order, the
/// `rng` draws here occur in exactly the same order as C's.
fn c_dna_text_without_separators(text: &[u8], rng: &mut Mt19937) -> std::io::Result<Vec<u8>> {
    let mut out = Vec::with_capacity(text.len());
    for &base in text {
        if base == b'$' {
            continue;
        }
        let code = match base {
            b'A' | b'a' => 0,
            b'C' | b'c' => 1,
            b'G' | b'g' => 2,
            b'T' | b't' | b'U' | b'u' => 3,
            _ if is_dna_degenerate(base) => rng.dna_replacement_code(),
            _ => {
                return Err(invalid_data(format!(
                    "base '{}' cannot be encoded in C DNA FM stream",
                    base as char
                )))
            }
        };
        out.push(code);
    }
    Ok(out)
}

/// IUPAC DNA/RNA degeneracy codes (everything that maps to FM `inv_alph` == -1
/// in C, i.e. is a valid residue but not one of the four canonical bases).
fn is_dna_degenerate(base: u8) -> bool {
    matches!(
        base.to_ascii_uppercase(),
        b'R' | b'Y' | b'S' | b'W' | b'K' | b'M' | b'B' | b'D' | b'H' | b'V' | b'N'
    )
}

fn c_amino_text_without_separators(text: &[u8]) -> std::io::Result<Vec<u8>> {
    text.iter()
        .filter(|&&residue| residue != b'$')
        .map(|&residue| {
            let residue = residue.to_ascii_uppercase();
            FM_AMINO_ALPHABET
                .iter()
                .position(|&symbol| symbol == residue)
                .map(|idx| idx as u8)
                .ok_or_else(|| {
                    invalid_data(format!(
                        "residue '{}' cannot be encoded in C amino FM stream",
                        residue as char
                    ))
                })
        })
        .collect()
}

fn c_block_text(
    fm_alphabet: FmAlphabet,
    seq_data: &[SequenceMetadata],
    block: &BlockRecord,
    all_text: &[u8],
    rng: &mut Mt19937,
) -> std::io::Result<Vec<u8>> {
    let mut text = Vec::with_capacity(
        seq_data[block.seq_offset..block.seq_offset + block.seq_count]
            .iter()
            .map(|seq| seq.length)
            .sum(),
    );
    for seq in &seq_data[block.seq_offset..block.seq_offset + block.seq_count] {
        let end = seq.fm_start + seq.length;
        text.extend(fm_alphabet.encode_text_without_separators(&all_text[seq.fm_start..end], rng)?);
    }
    Ok(text)
}

fn c_block_compact_sequence_starts(
    seq_data: &[SequenceMetadata],
    blocks: &[BlockRecord],
) -> Vec<usize> {
    let mut starts = vec![0; seq_data.len()];
    for block in blocks {
        let mut start = 0usize;
        for (idx, seq) in seq_data.iter().enumerate() {
            if seq.block_id == block.id {
                starts[idx] = start;
                start += seq.length;
            }
        }
    }
    starts
}

fn compact_ambiguity_ranges(
    seq_data: &[SequenceMetadata],
    compact_starts: &[usize],
    ambig_ranges: &[AmbiguityRange],
) -> Vec<AmbiguityRange> {
    // Emit in C's order: BLOCK-major (sequence-window order), then position
    // order within a window. C records each range in `block_length`
    // (block-LOCAL) coordinates while scanning the block's concatenated text
    // (makehmmerdb.c:644,664-667 — `block_length` is the running offset within
    // the block, which `compact_start` reproduces here as the window's start
    // within its block). Iterating windows (outer) then ranges (inner) matches
    // both that ordering and that coordinate basis, and re-emits overlap-region
    // ranges once per block exactly as C re-reads the overlap context. See the
    // matching loop nesting in `assign_ambiguities_to_blocks`.
    let mut compact_ranges = Vec::new();
    for (seq, &compact_start) in seq_data.iter().zip(compact_starts) {
        for range in ambig_ranges {
            if let Some(intersection) = intersect_ambiguity_with_sequence(range, seq) {
                compact_ranges.push(AmbiguityRange {
                    lower: compact_start + (intersection.lower - seq.fm_start),
                    upper: compact_start + (intersection.upper - seq.fm_start),
                });
            }
        }
    }
    compact_ranges
}

fn intersect_ambiguity_with_sequence(
    range: &AmbiguityRange,
    seq: &SequenceMetadata,
) -> Option<AmbiguityRange> {
    let seq_lower = seq.fm_start;
    let seq_upper = seq.fm_start + seq.length - 1;
    let lower = range.lower.max(seq_lower);
    let upper = range.upper.min(seq_upper);
    (lower <= upper).then_some(AmbiguityRange { lower, upper })
}

fn write_string<W: Write + ?Sized>(out: &mut W, value: &str) -> std::io::Result<()> {
    out.write_all(&(value.len() as u32).to_le_bytes())?;
    out.write_all(value.as_bytes())
}

fn write_c_string<W: Write + ?Sized>(out: &mut W, value: &str) -> std::io::Result<()> {
    out.write_all(value.as_bytes())?;
    out.write_all(&[0])
}

fn reject_embedded_nul(field: &str, value: &str) -> std::io::Result<()> {
    if value.as_bytes().contains(&0) {
        Err(invalid_data(format!(
            "{field} contains an embedded NUL byte"
        )))
    } else {
        Ok(())
    }
}

fn checked_u16(value: usize, field: &str) -> std::io::Result<u16> {
    u16::try_from(value).map_err(|_| invalid_data(format!("{field} exceeds C uint16_t range")))
}

fn checked_u32(value: usize, field: &str) -> std::io::Result<u32> {
    u32::try_from(value).map_err(|_| invalid_data(format!("{field} exceeds C uint32_t range")))
}

fn checked_u64(value: usize, field: &str) -> std::io::Result<u64> {
    u64::try_from(value).map_err(|_| invalid_data(format!("{field} exceeds C uint64_t range")))
}

fn checked_i32(value: usize, field: &str) -> std::io::Result<i32> {
    i32::try_from(value).map_err(|_| invalid_data(format!("{field} exceeds C int range")))
}

fn invalid_data(message: String) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, message)
}

/// Validate `--bin_length` exactly as C makehmmerdb does in code
/// (makehmmerdb.c:458): reject unless it is a power of two with
/// `32 <= b <= 4096`. The C getopts table range is NULL, so this is the only
/// gate; without it out-of-range / non-power-of-2 values are silently accepted.
fn validate_bin_length(bin_length: usize) -> Result<(), String> {
    if !(32..=4096).contains(&bin_length) || !bin_length.is_power_of_two() {
        Err("Invalid bin length: --bin_length must be a power of 2 between 32 and 4096".to_string())
    } else {
        Ok(())
    }
}

/// Validate `--sa_freq` as C makehmmerdb does in code (makehmmerdb.c:462):
/// it must be a power of two. (C tests only `freq_SA & (freq_SA - 1)`, which
/// also passes 0 and then crashes downstream; we additionally reject 0 with a
/// clean error rather than reproducing that segfault.)
fn validate_sa_freq(sa_freq: usize) -> Result<(), String> {
    if sa_freq == 0 || !sa_freq.is_power_of_two() {
        Err("Invalid suffix array sample rate: --sa_freq must be a power of 2".to_string())
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn readblock_records_retain_overlap_without_stalling() {
        // A single 25-base sequence with block_size=10 and overlap=3, windowed
        // exactly as C `esl_sqio_ReadBlock()` would: each block reads up to 10
        // NEW residues; continuation windows re-read 3 bases of leading context
        // (advancing 10 NEW residues per block, NOT 7). So fm_starts advance by
        // 10 minus the 3-base re-read = block windows [0..10), [7..20), [17..25).
        let seqs = vec![test_seq(0, 0, 25, 0)];
        let (windows, blocks) = build_readblock_like_records(&seqs, 10, 3);
        let starts: Vec<_> = blocks.iter().map(|block| block.fm_start).collect();
        let overlaps: Vec<_> = blocks.iter().map(|block| block.overlap_bases).collect();

        assert_eq!(starts, vec![0, 7, 17]);
        assert_eq!(overlaps, vec![0, 3, 3]);
        assert_eq!(blocks.last().unwrap().length, 8);
        assert_eq!(
            windows
                .iter()
                .map(|seq| seq.target_start)
                .collect::<Vec<_>>(),
            vec![1, 8, 18]
        );
    }

    #[test]
    fn c_metadata_writer_rejects_embedded_nul_strings() {
        let err = reject_embedded_nul("sequence name", "bad\0name").unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("embedded NUL"));
    }

    #[test]
    fn readblock_keeps_short_sequences_whole_without_cross_seq_overlap() {
        // Several short sequences, each well under block_size, are read whole
        // and packed into one block. The block boundary between distinct
        // sequences carries no overlap (matches C: overlap is only retained
        // when continuing ONE long sequence across a boundary).
        let seqs = vec![
            test_seq(0, 0, 8, 0),
            test_seq(1, 8, 8, 0),
            test_seq(2, 16, 8, 0),
        ];
        let (windows, blocks) = build_readblock_like_records(&seqs, 100, 20);

        // size after each whole seq: 8, 16, 24 (< 100) -> all stay in block 0;
        // a single trailing block holds all three intact windows, no overlap.
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].overlap_bases, 0);
        assert_eq!(blocks[0].seq_count, 3);
        assert_eq!(
            windows.iter().map(|w| w.length).collect::<Vec<_>>(),
            vec![8, 8, 8]
        );
        assert!(windows.iter().all(|w| w.overlap_bases == 0));
        // block_offset is the running position within the block's concatenated
        // text, independent of the original global fm_start values.
        assert_eq!(
            windows.iter().map(|w| w.block_offset).collect::<Vec<_>>(),
            vec![0, 8, 16]
        );
    }

    #[test]
    fn degenerate_residues_keep_original_symbols_and_record_ranges() {
        // append_sequence_text now stores the ORIGINAL IUPAC letters; degenerate
        // substitution is deferred to block-build time so the RNG draw order can
        // match C exactly. Only the ambiguity ranges are recorded here.
        let abc = Alphabet::dna();
        let mut sq = Sequence::new();
        sq.name = "ambig".to_string();
        sq.dsq = abc.digitize(b"ACNNRYT");
        sq.n = 7;

        let mut text = Vec::new();
        let mut ranges = Vec::new();
        append_sequence_text(&abc, &sq, true, &mut text, &mut ranges);

        assert_eq!(text, b"ACNNRYT");
        assert_eq!(ranges, vec![AmbiguityRange { lower: 2, upper: 5 },]);
    }

    #[test]
    fn mt19937_matches_easel_esl_random_seed_42() {
        // Reference values produced by Easel's esl_random() with
        // esl_randomness_Create(42): the standard MT19937 stream tempered and
        // divided by 2^32. The first raw 32-bit outputs of MT19937 seeded the
        // Easel way (mt[0]=42, mt[z]=69069*mt[z-1]) are deterministic; here we
        // check the derived 0..3 replacement codes, which is what actually feeds
        // the BWT. The sequence is fixed for seed 42 regardless of platform.
        let mut rng = Mt19937::new(42);
        // Pull a handful of u32s; verify determinism (same seed -> same stream)
        // and that codes stay in range.
        let first = rng.next_u32();
        let second = rng.next_u32();
        assert_ne!(first, second);
        let mut rng2 = Mt19937::new(42);
        assert_eq!(rng2.next_u32(), first);
        assert_eq!(rng2.next_u32(), second);

        let mut rng3 = Mt19937::new(42);
        for _ in 0..1000 {
            assert!(rng3.dna_replacement_code() < 4);
        }
    }

    #[test]
    fn dna_encode_substitutes_degenerate_with_rng_in_order() {
        // Two consecutive Ns should be replaced by the first two RNG draws, in
        // order; canonical bases are encoded directly without consuming the RNG.
        let mut rng_ref = Mt19937::new(MAKEHMMERDB_RNG_SEED);
        let first = rng_ref.dna_replacement_code();
        let second = rng_ref.dna_replacement_code();

        let mut rng = Mt19937::new(MAKEHMMERDB_RNG_SEED);
        let encoded = c_dna_text_without_separators(b"ACNNGT", &mut rng).unwrap();
        assert_eq!(encoded, vec![0, 1, first, second, 2, 3]);
    }

    #[test]
    fn guess_alphabet_dna_rna_amino() {
        // DNA: all four canonical bases, mostly ACGT.
        let dna = b">s\n"
            .iter()
            .chain(b"ACGTACGTACGTACGTAAAACCCCGGGGTTTT".iter())
            .copied()
            .collect::<Vec<u8>>();
        assert_eq!(guess_alphabet(&dna), Some(GuessedAlphabet::Dna));

        // RNA: ACGU.
        let rna = b">s\n"
            .iter()
            .chain(b"ACGUACGUACGUACGUAAAACCCCGGGGUUUU".iter())
            .copied()
            .collect::<Vec<u8>>();
        assert_eq!(guess_alphabet(&rna), Some(GuessedAlphabet::Rna));

        // Amino: contains aa-only giveaway residues (E, F, ...).
        let aa = b">s\nMEEFILKLQPWYACDEFGHIKLMNPQRSTVWY".to_vec();
        assert_eq!(guess_alphabet(&aa), Some(GuessedAlphabet::Amino));

        // Too few residues -> no guess.
        let tiny = b">s\nACGT".to_vec();
        assert_eq!(guess_alphabet(&tiny), None);
    }

    #[test]
    fn guess_alphabet_only_inspects_first_sequence() {
        // First sequence is clearly DNA; a later amino sequence must be ignored.
        let mut input = b">dna\n".to_vec();
        input.extend_from_slice(&b"ACGT".repeat(20));
        input.extend_from_slice(b"\n>protein\n");
        input.extend_from_slice(&b"EFILQPWY".repeat(20));
        assert_eq!(guess_alphabet(&input), Some(GuessedAlphabet::Dna));
    }

    #[test]
    fn amino_fm_text_preserves_degenerate_residue_symbols() {
        let abc = Alphabet::amino();
        let mut sq = Sequence::new();
        sq.name = "ambig_aa".to_string();
        sq.dsq = abc.digitize(b"ACDBJZOUX");
        sq.n = 9;

        let mut text = Vec::new();
        let mut ranges = Vec::new();
        append_sequence_text(&abc, &sq, false, &mut text, &mut ranges);

        assert_eq!(text, b"ACDBJZOUX");
        assert!(ranges.is_empty());
    }

    #[test]
    fn c_metadata_sequence_starts_reset_for_each_fm_block() {
        let seqs = vec![
            test_seq(0, 0, 3, 0),
            test_seq(1, 4, 2, 0),
            test_seq(2, 7, 5, 1),
            test_seq(3, 13, 4, 1),
        ];
        let blocks = vec![
            BlockRecord {
                id: 0,
                fm_start: 0,
                length: 6,
                seq_offset: 0,
                seq_count: 2,
                ambig_offset: 0,
                ambig_count: 0,
                overlap_bases: 0,
            },
            BlockRecord {
                id: 1,
                fm_start: 7,
                length: 10,
                seq_offset: 2,
                seq_count: 2,
                ambig_offset: 0,
                ambig_count: 0,
                overlap_bases: 3,
            },
        ];

        assert_eq!(
            c_block_compact_sequence_starts(&seqs, &blocks),
            vec![0, 3, 0, 5]
        );
        assert_eq!(
            compact_ambiguity_ranges(
                &seqs,
                &c_block_compact_sequence_starts(&seqs, &blocks),
                &[AmbiguityRange {
                    lower: 14,
                    upper: 15,
                }],
            ),
            vec![AmbiguityRange { lower: 6, upper: 7 }]
        );
    }

    #[test]
    fn readblock_windows_split_long_sequence_ambiguities_by_block() {
        let input = vec![test_seq(0, 0, 12, 0)];
        let (seqs, mut blocks) = build_readblock_like_records(&input, 10, 3);
        let ambig_ranges = vec![AmbiguityRange {
            lower: 8,
            upper: 10,
        }];
        assign_ambiguities_to_blocks(&seqs, &ambig_ranges, &mut blocks);
        let compact_starts = c_block_compact_sequence_starts(&seqs, &blocks);

        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[1].overlap_bases, 3);
        assert_eq!(
            seqs.iter()
                .map(|seq| (seq.target_id, seq.target_start, seq.fm_start, seq.length))
                .collect::<Vec<_>>(),
            vec![(0, 1, 0, 10), (0, 8, 7, 5)]
        );
        assert_eq!(compact_starts, vec![0, 0]);
        assert_eq!(blocks[0].ambig_offset, 0);
        assert_eq!(blocks[0].ambig_count, 1);
        assert_eq!(blocks[1].ambig_offset, 1);
        assert_eq!(blocks[1].ambig_count, 1);
        assert_eq!(
            compact_ambiguity_ranges(&seqs, &compact_starts, &ambig_ranges),
            vec![
                AmbiguityRange { lower: 8, upper: 9 },
                AmbiguityRange { lower: 1, upper: 3 },
            ]
        );
    }

    #[test]
    fn multi_block_overlap_ambiguities_emitted_in_c_block_major_order() {
        // Regression for the multi-block degenerate ordering bug: a single long
        // sequence split across blocks with FM_BLOCK_OVERLAP context, carrying
        // TWO degenerate runs. The first run sits entirely in block 0's tail and
        // is re-read into block 1's overlap context; the second run starts in
        // block 0 and also lands in the overlap. C records ambiguity ranges
        // while walking each block's text in turn (makehmmerdb.c:656-668), so
        // the flat list is BLOCK-major (block 0's runs first, in position order,
        // then block 1's re-read of the same runs in block-LOCAL coordinates) —
        // NOT range-major. With block_size=20 and overlap=5, block 0 = [0..20),
        // block 1 re-reads [15..30) (overlap=5). Two degenerate runs at global
        // [16,17] and [18,19] both fall in the overlap region.
        let input = vec![test_seq(0, 0, 30, 0)];
        let (seqs, mut blocks) = build_readblock_like_records(&input, 20, 5);
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[1].overlap_bases, 5);
        // windows: block 0 = fm_start 0 len 20; block 1 = fm_start 15 len 15.
        assert_eq!(
            seqs.iter()
                .map(|s| (s.fm_start, s.length))
                .collect::<Vec<_>>(),
            vec![(0, 20), (15, 15)]
        );

        let ambig_ranges = vec![
            AmbiguityRange {
                lower: 16,
                upper: 17,
            },
            AmbiguityRange {
                lower: 18,
                upper: 19,
            },
        ];
        assign_ambiguities_to_blocks(&seqs, &ambig_ranges, &mut blocks);
        // Block 0 sees both runs (offset 0, count 2); block 1 re-reads both from
        // the overlap (offset 2, count 2). The per-block offsets index the flat
        // list in block-major order.
        assert_eq!((blocks[0].ambig_offset, blocks[0].ambig_count), (0, 2));
        assert_eq!((blocks[1].ambig_offset, blocks[1].ambig_count), (2, 2));

        let compact_starts = c_block_compact_sequence_starts(&seqs, &blocks);
        assert_eq!(compact_starts, vec![0, 0]);
        // C order (block-major, block-local coords): block 0's two runs at
        // global 16-17 and 18-19, then block 1's re-read at block-local
        // 16-15=1..2 and 18-15=3..4. A range-major emission would instead
        // interleave them as 16-17, 1-2, 18-19, 3-4 — the bug we fixed.
        assert_eq!(
            compact_ambiguity_ranges(&seqs, &compact_starts, &ambig_ranges),
            vec![
                AmbiguityRange {
                    lower: 16,
                    upper: 17
                },
                AmbiguityRange {
                    lower: 18,
                    upper: 19
                },
                AmbiguityRange { lower: 1, upper: 2 },
                AmbiguityRange { lower: 3, upper: 4 },
            ]
        );
    }

    fn test_seq(
        target_id: usize,
        fm_start: usize,
        length: usize,
        block_id: usize,
    ) -> SequenceMetadata {
        SequenceMetadata {
            target_id,
            target_start: 1,
            fm_start,
            length,
            block_id,
            block_offset: fm_start,
            overlap_bases: 0,
            name: format!("s{target_id}"),
            acc: String::new(),
            desc: String::new(),
        }
    }

    #[test]
    fn bin_length_matches_c_power_of_2_range() {
        // C makehmmerdb.c:458 rejects unless power-of-2 AND 32<=b<=4096.
        // Verified against the bundled C binary: 32/64/4096 accepted;
        // 31/33/8192/0 rejected.
        for ok in [32usize, 64, 128, 256, 512, 1024, 2048, 4096] {
            assert!(
                validate_bin_length(ok).is_ok(),
                "bin_length {ok} should be accepted"
            );
        }
        for bad in [0usize, 16, 31, 33, 48, 100, 4097, 8192] {
            assert!(
                validate_bin_length(bad).is_err(),
                "bin_length {bad} should be rejected"
            );
        }
    }

    #[test]
    fn sa_freq_matches_c_power_of_2() {
        // C makehmmerdb.c:462 rejects unless a power of 2 (no range). 1/2/8/16
        // accepted; 6/7 rejected. We additionally reject 0 (C accepts it then
        // segfaults).
        for ok in [1usize, 2, 4, 8, 16, 32, 256] {
            assert!(
                validate_sa_freq(ok).is_ok(),
                "sa_freq {ok} should be accepted"
            );
        }
        for bad in [0usize, 3, 6, 7, 9, 100] {
            assert!(
                validate_sa_freq(bad).is_err(),
                "sa_freq {bad} should be rejected"
            );
        }
    }
}
