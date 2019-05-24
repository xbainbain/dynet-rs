extern crate dynet_sys as dn;

use std::ffi::CString;
use std::env;

/// Initializes dynet from command line arguments
/// 
/// Please call this function as soon as you enter the main function
/// 
/// # Examples
/// ```
/// extern crate dynet as dy;
/// 
/// fn main() {
///     dy::initialize();
///     // ...
/// }
/// ```
pub fn initialize() {
    let args = env::args()
        .map(|arg| CString::new(arg).unwrap())
        .collect::<Vec<CString>>();
    let c_args = args.iter()
        .map(|arg| arg.as_ptr() as *mut _)
        .collect::<Vec<*mut u8>>();
    unsafe {
        dn::DN_InitializeFromArgs(c_args.len() as i32, c_args.as_ptr() as *mut _, false);
    }
}

/// Resets the random seed and the random number generator
pub fn reset_rand_seed(seed: u32) {
    unsafe {
        dn::DN_ResetRng(seed);
    }
}