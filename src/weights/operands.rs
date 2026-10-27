use crate::gpu::{Pipeline, ReadOnly};

/// Learned linear transformation with kernels chosen for its shape and weight encoding at load time.
pub struct DenseMatrix {
    pub values: ReadOnly,
    pub input: usize,
    pub output: usize,
    pub plan: DensePlan,
}

/// Linear projection; import enforces each model's bias requirements.
pub struct Projection {
    pub matrix: DenseMatrix,
    pub bias: Option<ReadOnly>,
}

/// Vocabulary lookup that turns input token IDs into the feature vectors processed by the layers.
pub struct Embedding {
    pub values: ReadOnly,
    pub pipeline: Pipeline,
    pub width: usize,
    pub vocabulary: usize,
}

/// Projection from the final token's features to a score for every possible next token.
/// Only that row predicts the continuation, so this operand needs no batch alternatives.
pub struct Output {
    pub values: ReadOnly,
    pub bias: Option<ReadOnly>,
    pub vector: Vector,
    pub width: usize,
    pub vocabulary: usize,
}

pub struct Vector {
    pub pipeline: Pipeline,
    pub column_groups: usize,
    /// Threads per threadgroup; vector kernels derive their rows per group from it.
    pub threads: usize,
}

pub struct Tile {
    pub pipeline: Pipeline,
    pub column_groups: usize,
    pub rows: usize,
}

/// Prepared kernel choices for one matrix across single-token decoding and multi-token prefill.
pub enum DensePlan {
    Vector(Vector),
    Tiled {
        vector: Vector,
        vector_rows: usize,
        /// Tiled Q4 products consume one gain/inverse pair per input row.
        input_scales: bool,
        /// 8, 16, 24 and 32-row tiles for short batches and the tail rows of long ones.
        narrow: [Tile; 4],
        /// 64-row tiles for whole multiples of 64 rows.
        wide: Tile,
    },
}

/// One transformation across the expert bank, with MXFP4 weights and required F32 biases.
pub struct ExpertWeight {
    pub values: ReadOnly,
    pub bias: ReadOnly,
}

/// Learned scoring transformation for choosing which feed-forward experts process a token.
/// Routing uses F32 coefficients and biases independently of the experts' weight encoding.
pub struct Router {
    pub values: ReadOnly,
    pub bias: ReadOnly,
}
