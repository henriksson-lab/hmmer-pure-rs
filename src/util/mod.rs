#[doc(hidden)]
pub mod cmath;
pub mod random;
pub mod rootfinder;
pub mod scorematrix;
pub mod simd_env;
pub mod vectorops;

///
/// Apply the `HMMER_NCPU` environment default to CLI arguments.
///
/// `args` includes the executable/subcommand name at index 0. If no `--cpu`
/// or `--cpu=...` argument is present and `HMMER_NCPU` is valid Unicode,
/// inserts `--cpu` and the environment value immediately after index 0
/// (or at index 0 for an empty vector). Returns the updated arguments.
/// The value is passed through unchanged; the CLI parser validates it later.
pub fn apply_hmmer_ncpu_env_default(mut args: Vec<String>) -> Vec<String> {
    let cpu_on_cmdline = args
        .iter()
        .any(|arg| arg == "--cpu" || arg.starts_with("--cpu="));
    if cpu_on_cmdline {
        return args;
    }
    let Ok(ncpu) = std::env::var("HMMER_NCPU") else {
        return args;
    };
    let insert_at = usize::from(!args.is_empty());
    args.insert(insert_at, "--cpu".to_string());
    args.insert(insert_at + 1, ncpu);
    args
}
