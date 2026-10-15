use std::{env, error::Error, path::PathBuf, process::Command};

fn main() -> Result<(), Box<dyn Error>> {
    println!("cargo::rerun-if-changed=shaders");
    println!("cargo::rustc-link-arg=-mmacosx-version-min=27.0");
    println!("cargo::rerun-if-env-changed=DEVELOPER_DIR");
    println!("cargo::rerun-if-env-changed=SDKROOT");

    let output = env::var_os("OUT_DIR").ok_or("Cargo did not set OUT_DIR")?;
    let output = PathBuf::from(output);
    let air = output.join("oco.air");
    let library = output.join("oco.metallib");

    let mut compiler = Command::new("xcrun");
    compiler.args(["-sdk", "macosx", "metal", "-std=metal4.1", "-ffast-math"]);
    compiler.args(["-mmacosx-version-min=27.0", "-c", "shaders/oco.metal", "-o"]);
    compiler.arg(&air);
    run(&mut compiler)?;

    let mut linker = Command::new("xcrun");
    linker.args(["-sdk", "macosx", "metallib"]);
    linker.arg(&air).arg("-o").arg(&library);
    run(&mut linker)?;
    Ok(())
}

fn run(command: &mut Command) -> Result<(), Box<dyn Error>> {
    let status = command
        .status()
        .map_err(|error| format!("failed to run {command:?}: {error}"))?;
    if !status.success() {
        return Err(format!("{command:?} failed with {status}").into());
    }
    Ok(())
}
