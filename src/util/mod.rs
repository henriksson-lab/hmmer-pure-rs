#[doc(hidden)]
pub mod cmath;
pub mod random;
pub mod simd_env;
pub mod vectorops;

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
