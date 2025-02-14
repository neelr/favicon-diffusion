/* shaders.js */
const PATCHIFY_SHADER_CODE = `struct PatchifyUniforms {
  imageWidth: u32,
  imageHeight: u32,
  channels: u32,
  patchSize: u32,
}
@group(0) @binding(0) var<storage, read> imageInput: array<f32>;
@group(0) @binding(1) var<uniform> params: PatchifyUniforms;
@group(0) @binding(2) var<storage, read_write> patchesOutput: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  // The total number of elements equals (number of patches × patchSize² × channels)
  let index = gid.x;
  let width     = params.imageWidth;
  let height    = params.imageHeight;
  let channels  = params.channels;
  let pSize     = params.patchSize;

  let numPatchesX = width / pSize;
  let numPatchesY = height / pSize;
  let patchCount  = numPatchesX * numPatchesY;
  let patchDim    = channels * pSize * pSize;
  let totalElements = patchCount * patchDim;
  if(index >= totalElements) {
    return;
  }
  
  // Determine which patch and which element in that patch.
  let patchIndex = index / patchDim;
  let innerIndex = index % patchDim;
  
  // Compute the channel as well as the spatial (local) indices.
  let channel    = innerIndex / (pSize * pSize);
  let pixelIndex = innerIndex % (pSize * pSize);
  let localY     = pixelIndex / pSize;
  let localX     = pixelIndex % pSize;
  
  // Locate the patch in the grid.
  let patchRow = patchIndex / numPatchesX;
  let patchCol = patchIndex % numPatchesX;
  let x = patchCol * pSize + localX;
  let y = patchRow * pSize + localY;
  
  // Calculate the index into the flat image array.
  let imageIndex = (y * width + x) * channels + channel;
  if(x < width && y < height && imageIndex < arrayLength(&imageInput)) {
    patchesOutput[index] = imageInput[imageIndex];
  }
}`;

const UNPATCHIFY_SHADER_CODE = `
struct PatchifyUniforms {
  imageWidth: u32,
  imageHeight: u32,
  channels: u32,
  patchSize: u32,
}
@group(0) @binding(0) var<storage, read> patchesInput: array<f32>;
@group(0) @binding(1) var<uniform> params: PatchifyUniforms;
@group(0) @binding(2) var<storage, read_write> imageOutput: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  let width     = params.imageWidth;
  let height    = params.imageHeight;
  let channels  = params.channels;
  let pSize     = params.patchSize;

  let numPatchesX = width / pSize;
  let numPatchesY = height / pSize;
  let patchCount  = numPatchesX * numPatchesY;
  let patchDim    = channels * pSize * pSize;
  let totalElements = patchCount * patchDim;
  if(index >= totalElements) {
    return;
  }
  
  let patchIndex = index / patchDim;
  let innerIndex = index % patchDim;
  
  // Reverse the patchify ordering.
  let channel    = innerIndex / (pSize * pSize);
  let pixelIndex = innerIndex % (pSize * pSize);
  let localY     = pixelIndex / pSize;
  let localX     = pixelIndex % pSize;
  
  let patchRow = patchIndex / numPatchesX;
  let patchCol = patchIndex % numPatchesX;
  let x = patchCol * pSize + localX;
  let y = patchRow * pSize + localY;
  
  // Write back to the appropriate location in the image output.
  let imageIndex = (y * width + x) * channels + channel;
  if(x < width && y < height && imageIndex < arrayLength(&imageOutput)) {
    imageOutput[imageIndex] = patchesInput[index];
  }
}
`;

const MM_SHADER_CODE = `const TILE_SIZE: u32 = 16;

struct MatrixDims {
 aRows: u32,
k: u32,
bCols: u32,
pad: u32,   // Padding to ensure 16-byte alignment.
};

@group(0) @binding(0)
var<storage, read> A: array<f32>;

@group(0) @binding(1)
var<storage, read> B: array<f32>;

@group(0) @binding(2)
var<storage, read_write> C: array<f32>;

@group(0) @binding(3)
var<uniform> dims: MatrixDims;

// Use a flattened workgroup array for the tile.
var<workgroup> tileA: array<f32, TILE_SIZE * TILE_SIZE>;
var<workgroup> tileB: array<f32, TILE_SIZE * TILE_SIZE>;

@compute @workgroup_size(TILE_SIZE, TILE_SIZE)
fn main(
    @builtin(local_invocation_id) localID: vec3<u32>,
    @builtin(workgroup_id) groupID: vec3<u32>
) {
    // Compute the global coordinates for C.
    let row = groupID.x * TILE_SIZE + localID.x;
    let col = groupID.y * TILE_SIZE + localID.y;
    var sum: f32 = 0.0;
    let numTiles = (dims.k + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t: u32 = 0u; t < numTiles; t = t + 1u) {
        // Calculate the flattened index within the workgroup tile.
        let tileIndex = localID.x * TILE_SIZE + localID.y;
        
        // Load A into tileA.
        let aCol = t * TILE_SIZE + localID.y;
        if (row < dims.aRows && aCol < dims.k) {
            tileA[tileIndex] = A[row * dims.k + aCol];
        } else {
            tileA[tileIndex] = 0.0;
        }

        // Load B into tileB.
        // Here, use the appropriate condition for B’s coordinates:
        let bRow = t * TILE_SIZE + localID.x;
        if (bRow < dims.k && col < dims.bCols) {
            tileB[tileIndex] = B[bRow * dims.bCols + col];
        } else {
            tileB[tileIndex] = 0.0;
        }

        // Ensure that every thread has finished writing its tile element.
        workgroupBarrier();

        // Multiply the tile elements together.
        for (var kIndex: u32 = 0u; kIndex < TILE_SIZE; kIndex = kIndex + 1u) {
            // Compute flattened indices for the k-th element in the row/column.
            let aIndex = localID.x * TILE_SIZE + kIndex;
            let bIndex = kIndex * TILE_SIZE + localID.y;
            sum = sum + tileA[aIndex] * tileB[bIndex];
        }

        workgroupBarrier();
    }

    // Write the output only if within bounds.
    if (row < dims.aRows && col < dims.bCols) {
        C[row * dims.bCols + col] = sum;
    }
}`;

const ADALN_SHADER_CODE = `
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
`;

const SOFTMAX_WGSL = `
  @group(0) @binding(0) var<storage, read> scores: array<f32>;
  @group(0) @binding(1) var<storage, read_write> out: array<f32>;
  @group(0) @binding(2) var<uniform> dims: vec2<u32>; // (rows, cols)

  // Use separate shared arrays for max and sum.
  var<workgroup> shared_max: array<f32, 256>;
  var<workgroup> shared_sum: array<f32, 256>;

  @compute @workgroup_size(256)
  fn main(
      @builtin(workgroup_id) workgroup_id: vec3<u32>,
      @builtin(local_invocation_id) local_id: vec3<u32>
  ) {
      let row: u32 = workgroup_id.x;
      if (row >= dims.x) { return; }
      let numCols: u32 = dims.y;
      // Cache the beginning of the row to avoid repeated multiplication.
      let rowStart: u32 = row * numCols;

      // --- Step 1: Compute the maximum of this row ---
      var local_max: f32 = -1e20;
      // Each thread processes multiple columns in a strided loop.
      for (var col: u32 = local_id.x; col < numCols; col += 256u) {
          local_max = max(local_max, scores[rowStart + col]);
      }
      shared_max[local_id.x] = local_max;
      workgroupBarrier();

      // Unrolled tree reduction for the maximum.
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

      // --- Step 2: Compute exponentials (exp(val - max)) and sum them ---
      var local_sum: f32 = 0.0;
      for (var col: u32 = local_id.x; col < numCols; col += 256u) {
          let idx: u32 = rowStart + col;
          let exp_val: f32 = exp(scores[idx] - max_val);
          out[idx] = exp_val; // Store the computed exponentials
          local_sum += exp_val;
      }
      shared_sum[local_id.x] = local_sum;
      workgroupBarrier();

      // Unrolled tree reduction for the sum.
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

      // --- Step 3: Normalize the row ---
      for (var col: u32 = local_id.x; col < numCols; col += 256u) {
          let idx: u32 = rowStart + col;
          out[idx] = out[idx] / sum_val;
      }
  }
`;

const BATCHED_MAT_MUL_WGSL = `const TILE_SIZE: u32 = 16u;

struct MatrixDims {
    aRows: u32,        // rows of A and C
    k: u32,            // inner dimension
    bCols: u32,        // columns of C (and, for normal mode, columns of B)
    flagTranspose: u32 // 0 = B stored normally (dims.k x dims.bCols)
                       // 1 = B stored transposed (dims.bCols x dims.k)
};

@group(0) @binding(0)
var<storage, read> A: array<f32>;

@group(0) @binding(1)
var<storage, read> B: array<f32>;

@group(0) @binding(2)
var<storage, read_write> C: array<f32>;

@group(0) @binding(3)
var<uniform> dims: MatrixDims;

// Flattened workgroup arrays.
var<workgroup> tileA: array<f32, TILE_SIZE * TILE_SIZE>;
var<workgroup> tileB: array<f32, TILE_SIZE * TILE_SIZE>;

@compute @workgroup_size(TILE_SIZE, TILE_SIZE)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let row = global_id.x;
    let col = global_id.y;
    var sum: f32 = 0.0;
    let numTiles = (dims.k + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t: u32 = 0u; t < numTiles; t = t + 1u) {
        let tileIndex = local_id.x * TILE_SIZE + local_id.y;
        
        // Load tile element from A.
        let aCol = t * TILE_SIZE + local_id.y;
        if (row < dims.aRows && aCol < dims.k) {
            tileA[tileIndex] = A[row * dims.k + aCol];
        } else {
            tileA[tileIndex] = 0.0;
        }

        // Always load B for every workgroup thread.
        if (dims.flagTranspose == 1u) {
            // In transposed mode, B is stored as [bCols, k]. We swap the loading indices.
            let bIndex = t * TILE_SIZE + local_id.x;
            let tileBIndex = local_id.y * TILE_SIZE + local_id.x;
            if (col < dims.bCols && bIndex < dims.k) {
                tileB[tileBIndex] = B[col * dims.k + bIndex];
            } else {
                tileB[tileBIndex] = 0.0;
            }
        } else {
            // Normal mode: B is stored as [dims.k, dims.bCols] in row–major order.
            let bIndex = t * TILE_SIZE + local_id.x;
            if (bIndex < dims.k && col < dims.bCols) {
                tileB[tileIndex] = B[bIndex * dims.bCols + col];
            } else {
                tileB[tileIndex] = 0.0;
            }
        }

        // Synchronize to ensure the entire tile is loaded.
        workgroupBarrier();

        // Multiply the elements of the tile.
        if (dims.flagTranspose == 1u) {
            // With transposed B, our load swapped the indices.
            for (var kIdx: u32 = 0u; kIdx < TILE_SIZE; kIdx = kIdx + 1u) {
                // Note: For A the index remains the same.
                let aVal = tileA[local_id.x * TILE_SIZE + kIdx];
                let bVal = tileB[local_id.y * TILE_SIZE + kIdx];
                sum = sum + aVal * bVal;
            }
        } else {
            // Normal multiplication.
            for (var kIdx: u32 = 0u; kIdx < TILE_SIZE; kIdx = kIdx + 1u) {
                let aVal = tileA[local_id.x * TILE_SIZE + kIdx];
                let bVal = tileB[kIdx * TILE_SIZE + local_id.y];
                sum = sum + aVal * bVal;
            }
        }

        workgroupBarrier();
    }

    // Write back the final result only if within C’s bounds.
    if (row < dims.aRows && col < dims.bCols) {
        C[row * dims.bCols + col] = sum;
    }
}`;

const SCALE_MATRIX_WGSL = `
@group(0) @binding(0)
var<storage, read_write> matrix : array<f32>;

@group(0) @binding(1)
var<uniform> scalar : f32;

@group(0) @binding(2)
var<uniform> dims : vec2<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) globalID: vec3<u32>) {
  let totalEle = dims.x * dims.y;
  let numVec4 = totalEle / 4u;
  let remainder = totalEle % 4u;
  let tid = globalID.x;
  let s = scalar;

  if(tid < numVec4) {
    let offset = tid * 4u;
    // Load 4 elements at once.
    var v = vec4<f32>(
      matrix[offset],
      matrix[offset + 1u],
      matrix[offset + 2u],
      matrix[offset + 3u]
    );
    v = v * s;
    matrix[offset] = v.x;
    matrix[offset + 1u] = v.y;
    matrix[offset + 2u] = v.z;
    matrix[offset + 3u] = v.w;
  }
  // Let one thread handle any extra elements.
  if(tid == 0u && remainder > 0u) {
    for(var i = numVec4 * 4u; i < totalEle; i = i + 1u) {
      matrix[i] = matrix[i] * s;
    }
  }
}`;

const GELU_SHADER_CODE = `struct Uniforms {
  totalElements : u32,
};

@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read_write> output : array<f32>;
@group(0) @binding(2) var<uniform> uniforms : Uniforms;

fn sigmoid(x: f32) -> f32 {
  return 1.0 / (1.0 + exp(-x));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
  let idx : u32 = global_id.x;
  if (idx >= uniforms.totalElements) {
    return;
  }
  let a : f32 = input[idx];
  
  // Fast GELU approximation.
  output[idx] = a * sigmoid(1.702 * a);
}`;

const ELEMENT_WISE_ADD_SHADER_CODE = (n) => `
  // Process 4 elements per thread.
  const VECTOR_SIZE: u32 = 4u;
  const TOTAL_ELEMENTS: u32 = ${n}u;

  @group(0) @binding(0) var<storage, read> A : array<f32>;
  @group(0) @binding(1) var<storage, read> B : array<f32>;
  @group(0) @binding(2) var<storage, read_write> C : array<f32>;

  // Use a workgroup size that suits your hardware (here 64 threads per group)
  @compute @workgroup_size(64)
  fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Each thread starts at a multiple of VECTOR_SIZE. (Dispatch count should be ceil(n/VECTOR_SIZE))
    let index: u32 = gid.x * VECTOR_SIZE;

    // When there’s a full vector available, load 4 f32’s at once.
    if (index + VECTOR_SIZE <= TOTAL_ELEMENTS) {
      // Load 4 elements from A and B
      let a: vec4<f32> = vec4<f32>(
          A[index + 0u],
          A[index + 1u],
          A[index + 2u],
          A[index + 3u]
      );
      let b: vec4<f32> = vec4<f32>(
          B[index + 0u],
          B[index + 1u],
          B[index + 2u],
          B[index + 3u]
      );
      let result: vec4<f32> = a + b;

      // Write back the result in one shot.
      C[index + 0u] = result.x;
      C[index + 1u] = result.y;
      C[index + 2u] = result.z;
      C[index + 3u] = result.w;
    } else {
      // Fallback for any leftover tail elements when TOTAL_ELEMENTS isn’t a multiple of VECTOR_SIZE.
      for (var i: u32 = index; i < TOTAL_ELEMENTS; i = i + 1u) {
         C[i] = A[i] + B[i];
      }
    }
  }
`;
