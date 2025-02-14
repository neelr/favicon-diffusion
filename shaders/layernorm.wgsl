struct AdaLNUniforms {
  nRows: f32,
  rowWidth: f32,
  eps: f32,
};

@group(0) @binding(0)
var<storage, read> xBuffer: array<f32>;          // Input tensor: flattened (nRows * dim)

@group(0) @binding(1)
var<storage, read> shiftBuffer: array<f32>;        // Shift parameters, shape: [dim]

@group(0) @binding(2)
var<storage, read> scaleBuffer: array<f32>;        // Scale projection output (to which we add 1), shape: [dim]

@group(0) @binding(3)
var<storage, read_write> outputBuffer: array<f32>; // Output tensor, same size as xBuffer

@group(0) @binding(4)
var<uniform> uniforms: AdaLNUniforms;

//
// Declare workgroup shared memory for reductions.
// We use a fixed workgroup size of 64.
var<workgroup> sharedSum: array<f32, 16>;
// Shared variables for the computed mean and standard deviation.
var<workgroup> rowMean: f32;
var<workgroup> rowStd: f32;

@compute @workgroup_size(16)
fn main(@builtin(workgroup_id) group_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {

  // Each workgroup handles one row.
  let row: u32 = group_id.x;
  let rowWidth: u32 = u32(uniforms.rowWidth);
  let rowStart: u32 = row * rowWidth;
  let localSize: u32 = 16u;

  // ---------- First Pass: Compute the Mean ----------
  var sum: f32 = 0.0;
  var i: u32 = local_id.x;
  while (i < rowWidth) {
    let index: u32 = rowStart + i;
    sum = sum + xBuffer[index];
    i = i + localSize;
  }
  // Save each thread’s partial sum.
  sharedSum[local_id.x] = sum;
  workgroupBarrier();

  var stride: u32 = localSize / 2u;
  // Parallel reduction over sharedSum.
  while (stride > 0u) {
    if (local_id.x < stride) {
      sharedSum[local_id.x] = sharedSum[local_id.x] + sharedSum[local_id.x + stride];
    }
    workgroupBarrier();
    stride = stride / 2u;
  }
  if (local_id.x == 0u) {
    rowMean = sharedSum[0] / f32(rowWidth);
  }
  workgroupBarrier();

  // ---------- Second Pass: Compute the Variance ----------
  var sqDiff: f32 = 0.0;
  i = local_id.x;
  while (i < rowWidth) {
    let index: u32 = rowStart + i;
    let x_val: f32 = xBuffer[index];
    let diff: f32 = x_val - rowMean;
    sqDiff = sqDiff + diff * diff;
    i = i + localSize;
  }
  // Overwrite sharedSum with the partial sums for variance.
  sharedSum[local_id.x] = sqDiff;
  workgroupBarrier();

  stride = localSize / 2u;
  while (stride > 0u) {
    if (local_id.x < stride) {
      sharedSum[local_id.x] = sharedSum[local_id.x] + sharedSum[local_id.x + stride];
    }
    workgroupBarrier();
    stride = stride / 2u;
  }
  if (local_id.x == 0u) {
    // Compute variance as the mean squared deviation.
    let variance: f32 = sharedSum[0] / f32(rowWidth);
    // The addition of eps inside sqrt prevents division by zero (and is how PyTorch applies eps).
    rowStd = sqrt(variance + uniforms.eps);
  }
  workgroupBarrier();

  // ---------- Normalize and Affine Transform ----------
  i = local_id.x;
  while (i < rowWidth) {
    let index: u32 = rowStart + i;
    let x_val: f32 = xBuffer[index];
    let norm: f32 = (x_val - rowMean) / rowStd;
    // Apply the affine transform: norm * (1 + scale[i]) + shift[i]
    let scaled: f32 = norm * (1.0 + scaleBuffer[i]) + shiftBuffer[i];
    outputBuffer[index] = scaled;
    i = i + localSize;
  }
}