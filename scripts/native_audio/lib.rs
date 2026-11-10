//! PCM16 little-endian frame energy. No Python ABI or third-party dependencies.
use std::slice;

/// The caller supplies `count * 2` readable bytes and ceil(count/frame) writable
/// u64 slots. Bytes need not be aligned; integer accumulation preserves the
/// Python detector's exact sum, including -32768 samples.
#[no_mangle]
pub unsafe extern "C" fn pcm16_energy(
    pcm: *const u8,
    count: usize,
    frame: usize,
    output: *mut u64,
) {
    if count == 0 || frame == 0 {
        return;
    }
    let bytes = unsafe { slice::from_raw_parts(pcm, count * 2) };
    for (index, chunk) in bytes.chunks(frame * 2).enumerate() {
        let energy = chunk.chunks_exact(2).map(|pair| {
            let value = i16::from_le_bytes([pair[0], pair[1]]) as i64;
            (value * value) as u64
        }).sum();
        unsafe { output.add(index).write(energy) };
    }
}
