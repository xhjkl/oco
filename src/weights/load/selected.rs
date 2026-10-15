//! Transient validated operands shared by the architecture importers.
use crate::gguf::Tensor;

pub struct Common<'a> {
    pub embedding: Tensor<'a>,
    pub output: Projection<'a, Option<Tensor<'a>>>,
    pub norm: Tensor<'a>,
}

pub struct Projection<'a, B> {
    pub matrix: Tensor<'a>,
    pub bias: B,
}

pub struct Layer<'a, M, F> {
    pub norm: Tensor<'a>,
    pub post_norm: Tensor<'a>,
    pub mixer: M,
    pub ff: F,
}
