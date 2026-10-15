use super::{BufferView, Gpu};
use crate::gpu::{Buffer, BufferRef};
use bytemuck::Pod;
use objc2_metal::MTLBuffer;

pub(crate) fn buffer(gpu: &Gpu, values: &[impl Pod]) -> Buffer {
    let buffer = gpu.new_buffer(std::mem::size_of_val(values));
    write(&buffer, 0, values);
    buffer
}

/// Copy shared storage after all commands using it have completed.
pub(crate) fn read<T: Pod>(buffer: &BufferRef, values: &mut [T]) {
    BufferView::new(buffer, 0, buffer.length()).read(values);
}

/// Write shared storage after all commands using it have completed.
pub(crate) fn write<T: Pod>(buffer: &BufferRef, offset: usize, values: &[T]) {
    BufferView::new(buffer, 0, buffer.length()).write(offset, values);
}
