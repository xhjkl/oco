mod attention;
mod audio;
mod dense_bench;
mod matrix_precision;
mod mxfp4;
mod nemotron;
mod nemotron_bench;
mod numerics;
mod recurrent;

const GUARD: usize = 32;
const SENTINEL: u32 = 0x7fc1_2345;

fn noise(index: usize) -> u32 {
    let value = index as u32;
    let value = (value ^ (value >> 16)).wrapping_mul(0x7feb_352d);
    let value = (value ^ (value >> 15)).wrapping_mul(0x846c_a68b);
    value ^ (value >> 16)
}
