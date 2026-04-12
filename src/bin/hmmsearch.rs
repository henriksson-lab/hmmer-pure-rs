use std::env;
use std::ffi::CString;
use std::os::raw::{c_char, c_int};

// Force linking the hmmer library crate which triggers build.rs
use hmmer as _;

extern "C" {
    fn hmmsearch_main(argc: c_int, argv: *mut *mut c_char) -> c_int;
}

fn main() {
    // Build argv from process args, prepending "hmmsearch" as argv[0]
    let args: Vec<String> = env::args().collect();
    let c_strings: Vec<CString> = args
        .iter()
        .map(|s| CString::new(s.as_str()).expect("CString::new failed"))
        .collect();
    let mut c_ptrs: Vec<*mut c_char> = c_strings
        .iter()
        .map(|cs| cs.as_ptr() as *mut c_char)
        .collect();
    c_ptrs.push(std::ptr::null_mut()); // NULL terminator

    let argc = args.len() as c_int;
    let status = unsafe { hmmsearch_main(argc, c_ptrs.as_mut_ptr()) };

    std::process::exit(status);
}
