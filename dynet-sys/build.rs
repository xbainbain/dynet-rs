extern crate bindgen;

use std::process::Command;
use std::path::PathBuf;
use std::fs;
use std::env;

const EIGEN_REPOSITORY:&'static str = "https://bitbucket.org/eigen/eigen/";
const EIGEN_VERSION:&'static str = "b2e267d";
const DYNET_REPOSITORY:&'static str = "https://github.com/clab/dynet.git";
//const DYNET_TAG:&'static str = "b2e267d";
const DYNET_LIBRARY:&'static str = "dynet";
const DYNETC_REPOSITORY:&'static str = "https://github.com/xbainbain/dynet-c.git";
const DYNETC_LIBRARY:&'static str = "dynetc";

fn main() {
    check_prerequisites();
    build_from_src();
    create_bindings();
}

fn check_prerequisites() {
    Command::new("git").status().expect("Unable to find git, please install git first");
    Command::new("cmake").status().expect("Unable to find cmake, please install cmake first");
    Command::new("hg")
            .status()
            .expect("Unable to find mercurial, please install mercurial first");
}

fn build_from_src() {
    let source_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
        .join(format!("target/source"));
    let eigen_dir = source_dir.join("eigen");
    let dynet_dir = source_dir.join("dynet");
    let dynet_build_dir = dynet_dir.join("build");
    let dynetc_dir = source_dir.join("dynet-c");
    let dynetc_build_dir = dynetc_dir.join("build");

    let dynetc_lib_dir = dynetc_dir.join("lib");

    let (dynet_lib_path, dynetc_lib_path)  = if cfg!(target_os = "macos") {
        (dynet_build_dir.join(format!("dynet/lib{}.dylib", DYNET_LIBRARY)),
        dynetc_lib_dir.join(format!("lib{}.dylib", DYNETC_LIBRARY)))
    } else if cfg!(target_os = "linux") {
        (dynet_build_dir.join(format!("dynet/lib{}.so", DYNET_LIBRARY)),
        dynetc_lib_dir.join(format!("lib{}.so", DYNETC_LIBRARY)))
    } else {
        panic!("Unsupport platform, exit...")
    };

    if source_dir.exists() {
        println!("Directory for source {:?} already exists", source_dir);
    } else {
        fs::create_dir(source_dir.clone())
           .expect(&format!("Unalble to create directory {:?}", source_dir));
    }
    
    // Download eigen
    if eigen_dir.exists() {
        println!("Eigen already downloaded");
    } else {
        println!("Downloading eigen now...");
        match Command::new("hg")
                      .args(&["clone", EIGEN_REPOSITORY, "-r", EIGEN_VERSION])
                      .arg(&eigen_dir)
                      .output() {
            Ok(_) => println!("Download Eigen successfully at {:?}", eigen_dir),
            Err(_) => {
                panic!("Unable to download Eigen at {:?}", eigen_dir);
                // May do some clean-ups!
            }
        }
    }

    // Downlaod dynet
    if dynet_dir.exists() {
        println!("Dynet already downloaded");
    } else {
        println!("Downloading dynet now...");
        match Command::new("git")
                      .args(&["clone", DYNET_REPOSITORY])
                      .arg(&dynet_dir)
                      .output() {
            Ok(_) => println!("Download DyNet successfully at {:?}", dynet_dir),
            Err(_) => {
                panic!("Unable to download DyNet at {:?}", dynet_dir);
                // May do some clean-ups!
            }
        }
    }

    // Download dynet-c
    if dynetc_dir.exists() {
        println!("Dynet-c already downloaded");
    } else {
        println!("Downloading dynet-c now...");
        match Command::new("git")
                      .args(&["clone", DYNETC_REPOSITORY])
                      .arg(&dynetc_dir)
                      .output() {
            Ok(_) => println!("Download dynet-c successfully at {:?}", dynetc_dir),
            Err(_) => {
                panic!("Unable to download dynet-c at {:?}", dynetc_dir);
                // May do some clean-ups!
            }
        }
    }

    // Build dynet
    if dynet_lib_path.exists() {
        println!("Library {:?} already exists, no need to rebuild", dynet_lib_path);
    } else {
        println!("Buiding dynet now...");
        if !dynet_build_dir.exists() {
            fs::create_dir(dynet_build_dir.clone())
               .expect(&format!("Unable to create build directory {:?}", dynet_build_dir));
        }
        let eigen_dir_str = eigen_dir.to_str().unwrap();
        let output = Command::new("cmake")
                             .current_dir(&dynet_build_dir)
                             .arg("..")
                             .arg(format!("-DEIGEN3_INCLUDE_DIR={}", eigen_dir_str))
                             .output()
                             .expect("Unable to run cmake");
        println!("stdout: {}", String::from_utf8_lossy(&output.stdout));

        match Command::new("make")
                      .current_dir(&dynet_build_dir)
                      .args(&["-j", "2"])
                      .output() {
            Ok(output) => {
                println!("Building library {:?} success", dynet_lib_path);
                println!("status: {}", output.status);
                println!("stdout: {}", String::from_utf8_lossy(&output.stdout));
                println!("stderr: {}", String::from_utf8_lossy(&output.stderr));
            }
            Err(_) => {
                // May do some clean-ups!
                panic!(format!("Unable to build library {}", DYNET_LIBRARY));  
            }
        }
    }

    // Build dynet-c
    if dynetc_lib_path.exists() {
        println!("Library {:?} already exists, no need to rebuild", dynetc_lib_path);
    } else {
        println!("Buiding dynet-c now...");
        if !dynetc_build_dir.exists() {
            fs::create_dir(dynetc_build_dir.clone())
               .expect(&format!("Unalble to create build directory {:?}", dynetc_build_dir));
        }

        let dynet_dir_str = dynet_dir.to_str().unwrap();
        Command::new("cmake")
                .current_dir(&dynetc_build_dir)
                .arg("..")
                .arg(format!("-DDYNET_INCLUDE_DIR={}", dynet_dir_str))
                .output()
                .expect("Unable to run cmake");

        match Command::new("make")
                      .current_dir(&dynetc_build_dir)
                      .args(&["-j", "2"])
                      .output() {
            Ok(_) => println!("Building library {:?} success", dynetc_lib_path),
            Err(_) => {
                // May do some clean-ups!
                panic!(format!("Unable to build library {}", DYNETC_LIBRARY));
                
            }
        }
    }

    println!("{:?}", env::current_dir().unwrap());
    println!("cargo:rustc-link-lib=dylib={}", DYNETC_LIBRARY);
    println!("cargo:rustc-link-search={}", dynetc_lib_dir.display());
}


fn create_bindings() {
    let bindings = bindgen::Builder::default()
        // The input header we would like to generate
        // bindings for.
        .header("wrapper.h")
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    bindings.write_to_file("src/bindings.rs").expect(
        "Couldn't write bindings!",
    );
}


