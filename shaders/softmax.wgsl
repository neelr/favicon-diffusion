// algorithm: computes softmax function across rows of input matrix
// - uses parallel reduction for finding max value for numerical stability
// - employs tree reduction pattern in shared memory for efficient reduction
// - processes multiple elements per thread in strided pattern
// - unrolled tree reduction for threads < 32 to avoid warp-level sync
// - fuses exp and normalization to minimize memory accesses

@group(0) @binding(0) var<storage, read> scores: array<f32>;  // [rows, cols]
@group(0) @binding(1) var<storage, read_write> out: array<f32>;  // [rows, cols]
@group(0) @binding(2) var<uniform> dims: vec2<u32>;  // (rows, cols)

var<workgroup> shared_max: array<f32, 256>;  // For parallel reduction
var<workgroup> shared_sum: array<f32, 256>;  // For parallel reduction

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let row: u32 = workgroup_id.x;
    if (row >= dims.x) { return; }
    let numCols: u32 = dims.y;
    let rowStart: u32 = row * numCols;

    var local_max: f32 = -1e20;
    for (var col: u32 = local_id.x; col < numCols; col += 256u) {
        local_max = max(local_max, scores[rowStart + col]);
    }
    shared_max[local_id.x] = local_max;
    workgroupBarrier();

    if (local_id.x < 128u) {
        shared_max[local_id.x] = max(shared_max[local_id.x], shared_max[local_id.x + 128u]);
    }
    workgroupBarrier();
    if (local_id.x < 64u) {
        shared_max[local_id.x] = max(shared_max[local_id.x], shared_max[local_id.x + 64u]);
    }
    workgroupBarrier();
    if (local_id.x < 32u) {
        shared_max[local_id.x] = max(shared_max[local_id.x], shared_max[local_id.x + 32u]);
        shared_max[local_id.x] = max(shared_max[local_id.x], shared_max[local_id.x + 16u]);
        shared_max[local_id.x] = max(shared_max[local_id.x], shared_max[local_id.x + 8u]);
        shared_max[local_id.x] = max(shared_max[local_id.x], shared_max[local_id.x + 4u]);
        shared_max[local_id.x] = max(shared_max[local_id.x], shared_max[local_id.x + 2u]);
        shared_max[local_id.x] = max(shared_max[local_id.x], shared_max[local_id.x + 1u]);
    }
    workgroupBarrier();
    let max_val: f32 = shared_max[0];

    var local_sum: f32 = 0.0;
    for (var col: u32 = local_id.x; col < numCols; col += 256u) {
        let idx: u32 = rowStart + col;
        let exp_val: f32 = exp(scores[idx] - max_val);
        out[idx] = exp_val;
        local_sum += exp_val;
    }
    shared_sum[local_id.x] = local_sum;
    workgroupBarrier();

    if (local_id.x < 128u) {
        shared_sum[local_id.x] += shared_sum[local_id.x + 128u];
    }
    workgroupBarrier();
    if (local_id.x < 64u) {
        shared_sum[local_id.x] += shared_sum[local_id.x + 64u];
    }
    workgroupBarrier();
    if (local_id.x < 32u) {
        shared_sum[local_id.x] += shared_sum[local_id.x + 32u];
        shared_sum[local_id.x] += shared_sum[local_id.x + 16u];
        shared_sum[local_id.x] += shared_sum[local_id.x + 8u];
        shared_sum[local_id.x] += shared_sum[local_id.x + 4u];
        shared_sum[local_id.x] += shared_sum[local_id.x + 2u];
        shared_sum[local_id.x] += shared_sum[local_id.x + 1u];
    }
    workgroupBarrier();
    let sum_val: f32 = shared_sum[0];

    for (var col: u32 = local_id.x; col < numCols; col += 256u) {
        let idx: u32 = rowStart + col;
        out[idx] = out[idx] / sum_val;
    }
}