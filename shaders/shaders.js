/** THIS FILE IS AUTO-GENERATED SHADER CODE. DO NOT MODIFY. */

const ADD_SHADER_CODE = `
// algorithm: performs element-wise addition of two arrays
// - vectorized processing using vec4 for efficient memory access
// - handles non-aligned array lengths with scalar fallback
// - coalesced memory access pattern for better cache utilization
// - processes 4 elements per thread to increase arithmetic intensity

// Process 4 elements per thread.
const VECTOR_SIZE: u32 = 4u;
const TOTAL_ELEMENTS: u32 = _NUM_ELEMENTS_u;

@group(0) @binding(0) var<storage, read> A: array<f32>;  // [TOTAL_ELEMENTS]
@group(0) @binding(1) var<storage, read> B: array<f32>;  // [TOTAL_ELEMENTS]
@group(0) @binding(2) var<storage, read_write> C: array<f32>;  // [TOTAL_ELEMENTS]

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Each thread starts at a multiple of VECTOR_SIZE. (Dispatch count should be ceil(n/VECTOR_SIZE))
    let index: u32 = gid.x * VECTOR_SIZE;

    // When there's a full vector available, load 4 f32's at once.
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
        // Fallback for any leftover tail elements when TOTAL_ELEMENTS isn't a multiple of VECTOR_SIZE.
        for (var i: u32 = index; i < TOTAL_ELEMENTS; i = i + 1u) {
            C[i] = A[i] + B[i];
        }
    }
}`;

const BATCHED_MATMUL_SHADER_CODE = `
// algorithm: performs tiled matrix multiplication with optional b-transpose
// - uses shared memory tiling for better cache efficiency
// - supports both normal and transposed b matrix layouts
// - processes 16x16 tiles to maximize cache utilization
// - coalesced memory access pattern for global memory
// - handles non-tile-aligned matrix dimensions
// - fuses transpose with multiplication when needed

const TILE_SIZE: u32 = 16u;

struct MatrixDims {
    aRows: u32,        // rows of A and C
    k: u32,            // inner dimension
    bCols: u32,        // columns of C (and, for normal mode, columns of B)
    flagTranspose: u32 // 0: B is [k, bCols], 1: B is [bCols, k]
};

@group(0) @binding(0)
var<storage, read> A: array<f32>;  // [aRows, k]

@group(0) @binding(1)
var<storage, read> B: array<f32>;  // [k, bCols] or [bCols, k] if transposed

@group(0) @binding(2)
var<storage, read_write> C: array<f32>;  // [aRows, bCols]

@group(0) @binding(3)
var<uniform> dims: MatrixDims;

var<workgroup> tileA: array<f32, TILE_SIZE * TILE_SIZE>;  // [TILE_SIZE, TILE_SIZE]
var<workgroup> tileB: array<f32, TILE_SIZE * TILE_SIZE>;  // [TILE_SIZE, TILE_SIZE]

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
        
        let aCol = t * TILE_SIZE + local_id.y;
        if (row < dims.aRows && aCol < dims.k) {
            tileA[tileIndex] = A[row * dims.k + aCol];
        } else {
            tileA[tileIndex] = 0.0;
        }

        if (dims.flagTranspose == 1u) {
            let bIndex = t * TILE_SIZE + local_id.x;
            let tileBIndex = local_id.y * TILE_SIZE + local_id.x;
            if (col < dims.bCols && bIndex < dims.k) {
                tileB[tileBIndex] = B[col * dims.k + bIndex];
            } else {
                tileB[tileBIndex] = 0.0;
            }
        } else {
            let bIndex = t * TILE_SIZE + local_id.x;
            if (bIndex < dims.k && col < dims.bCols) {
                tileB[tileIndex] = B[bIndex * dims.bCols + col];
            } else {
                tileB[tileIndex] = 0.0;
            }
        }

        workgroupBarrier();

        if (dims.flagTranspose == 1u) {
            for (var kIdx: u32 = 0u; kIdx < TILE_SIZE; kIdx = kIdx + 1u) {
                let aVal = tileA[local_id.x * TILE_SIZE + kIdx];
                let bVal = tileB[local_id.y * TILE_SIZE + kIdx];
                sum = sum + aVal * bVal;
            }
        } else {
            for (var kIdx: u32 = 0u; kIdx < TILE_SIZE; kIdx = kIdx + 1u) {
                let aVal = tileA[local_id.x * TILE_SIZE + kIdx];
                let bVal = tileB[kIdx * TILE_SIZE + local_id.y];
                sum = sum + aVal * bVal;
            }
        }

        workgroupBarrier();
    }

    if (row < dims.aRows && col < dims.bCols) {
        C[row * dims.bCols + col] = sum;
    }
}`;

const FLASH_ATTENTION_SHADER_CODE = `
// NOT USED IN THE CURRENT IMPLEMENTATION because tile sizes are too small for it to be worth it
// algorithm: implements flash attention for efficient transformer attention computation
// - uses block-wise processing to reduce memory bandwidth
// - maintains running max for numerical stability
// - processes attention scores in tiles to maximize cache usage
// - uses vec4 for coalesced memory access
// - shared memory caching of k/v pairs to avoid repeated global memory access
// - fuses softmax computation with attention to reduce memory roundtrips

const BLOCK_SIZE: u32 = 16;
const MAX_DIM_HEAD: u32 = 128;
const MAX_VEC4: u32 = MAX_DIM_HEAD / 4u;

@group(0) @binding(0) var<storage, read> Q: array<f32>;  // [seqLen, dimHead]
@group(0) @binding(1) var<storage, read> K: array<f32>;  // [seqLen, dimHead]
@group(0) @binding(2) var<storage, read> V: array<f32>;  // [seqLen, dimHead]
@group(0) @binding(3) var<storage, read_write> O: array<f32>;  // [seqLen, dimHead]
@group(0) @binding(4) var<uniform> dims: vec4<u32>;  // (seqLen, dimHead, blockSize, 0)

var<workgroup> shared_K: array<f32, BLOCK_SIZE * MAX_DIM_HEAD>;  // [blockSize, dimHead]
var<workgroup> shared_V: array<f32, BLOCK_SIZE * MAX_DIM_HEAD>;  // [blockSize, dimHead]
var<workgroup> shared_m: array<f32, BLOCK_SIZE>;  // Max values per row
var<workgroup> shared_l: array<f32, BLOCK_SIZE>;  // Sum accumulators per row

@compute @workgroup_size(BLOCK_SIZE)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let seqLen = dims.x;
    let dimHead = dims.y;
    let blockSize = dims.z;
    
    let row = wg_id.x * BLOCK_SIZE + local_id.x;
    let isActive = row < seqLen;
    let numVec4 = dimHead / 4u;

    var mi = -1e20;
    var li = 0.0;
    var oi: array<vec4<f32>, MAX_VEC4>;
    for (var i = 0u; i < numVec4; i++) {
        oi[i] = vec4<f32>(0.0);
    }

    for (var block_start = 0u; block_start < seqLen; block_start += blockSize) {
        let block_end = min(block_start + blockSize, seqLen);
        
        for (var i = local_id.x; i < blockSize; i += BLOCK_SIZE) {
            let key_row = block_start + i;
            if (key_row < seqLen) {
                for (var d = 0u; d < dimHead; d++) {
                    let shared_idx = i * dimHead + d;
                    shared_K[shared_idx] = K[key_row * dimHead + d];
                    shared_V[shared_idx] = V[key_row * dimHead + d];
                }
            } else {
                for (var d = 0u; d < dimHead; d++) {
                    let shared_idx = i * dimHead + d;
                    shared_K[shared_idx] = 0.0;
                    shared_V[shared_idx] = 0.0;
                }
            }
        }
        workgroupBarrier();

        if (isActive) {
            for (var j = 0u; j < block_end - block_start; j++) {
                var score = 0.0;
                for (var d = 0u; d < dimHead; d++) {
                    score += Q[row * dimHead + d] * shared_K[j * dimHead + d];
                }
                score = score / sqrt(f32(dimHead));

                let mi_new = max(mi, score);
                let exp_score = exp(score - mi_new);
                
                let scale = exp(mi - mi_new);
                for (var i = 0u; i < numVec4; i++) {
                    oi[i] = scale * oi[i];
                }
                li = scale * li;

                for (var i = 0u; i < numVec4; i++) {
                    let base = i * 4u;
                    let vi = exp_score * vec4<f32>(
                        shared_V[j * dimHead + base],
                        shared_V[j * dimHead + base + 1u],
                        shared_V[j * dimHead + base + 2u],
                        shared_V[j * dimHead + base + 3u]
                    );
                    oi[i] += vi;
                }
                
                li += exp_score;
                mi = mi_new;
            }
        }
        workgroupBarrier();
    }

    if (isActive) {
        let out_idx = row * dimHead;
        for (var i = 0u; i < numVec4; i++) {
            let base = i * 4u;
            let normalized = oi[i] / vec4<f32>(li);
            O[out_idx + base] = normalized.x;
            O[out_idx + base + 1u] = normalized.y;
            O[out_idx + base + 2u] = normalized.z;
            O[out_idx + base + 3u] = normalized.w;
        }
    }
}
`;

const LAYERNORM_SHADER_CODE = `
// algorithm: computes layer normalization with adaptive scaling and shifting
// - uses two-pass algorithm for numerical stability (mean then variance)
// - parallel reduction in shared memory for efficient mean/variance computation
// - processes multiple elements per thread in strided pattern
// - fuses normalization and affine transform into single pass
// - uses workgroup-level synchronization to coordinate parallel reductions

struct AdaLNUniforms {
  nRows: f32,
  rowWidth: f32,
  eps: f32,
};

@group(0) @binding(0)
var<storage, read> xBuffer: array<f32>;          // [nRows, dim]

@group(0) @binding(1)
var<storage, read> shiftBuffer: array<f32>;      // [dim]

@group(0) @binding(2)
var<storage, read> scaleBuffer: array<f32>;      // [dim]

@group(0) @binding(3)
var<storage, read_write> outputBuffer: array<f32>; // [nRows, dim]

@group(0) @binding(4)
var<uniform> uniforms: AdaLNUniforms;

var<workgroup> sharedSum: array<f32, 16>;  // For parallel reduction
var<workgroup> rowMean: f32;
var<workgroup> rowStd: f32;

@compute @workgroup_size(16)
fn main(@builtin(workgroup_id) group_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  let row: u32 = group_id.x;
  let rowWidth: u32 = u32(uniforms.rowWidth);
  let rowStart: u32 = row * rowWidth;
  let localSize: u32 = 16u;

  var sum: f32 = 0.0;
  var i: u32 = local_id.x;
  while (i < rowWidth) {
    let index: u32 = rowStart + i;
    sum = sum + xBuffer[index];
    i = i + localSize;
  }
  sharedSum[local_id.x] = sum;
  workgroupBarrier();

  var stride: u32 = localSize / 2u;
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

  var sqDiff: f32 = 0.0;
  i = local_id.x;
  while (i < rowWidth) {
    let index: u32 = rowStart + i;
    let x_val: f32 = xBuffer[index];
    let diff: f32 = x_val - rowMean;
    sqDiff = sqDiff + diff * diff;
    i = i + localSize;
  }
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
    let variance: f32 = sharedSum[0] / f32(rowWidth);
    rowStd = sqrt(variance + uniforms.eps);
  }
  workgroupBarrier();

  i = local_id.x;
  while (i < rowWidth) {
    let index: u32 = rowStart + i;
    let x_val: f32 = xBuffer[index];
    let norm: f32 = (x_val - rowMean) / rowStd;
    let scaled: f32 = norm * (1.0 + scaleBuffer[i]) + shiftBuffer[i];
    outputBuffer[index] = scaled;
    i = i + localSize;
  }
}`;

const MATMUL_SHADER_CODE = `
// algorithm: performs tiled matrix multiplication for standard matrix layout
// - uses shared memory tiling for better cache efficiency
// - processes 16x16 tiles to maximize cache utilization
// - coalesced memory access pattern for global memory
// - handles non-tile-aligned matrix dimensions
// - minimizes bank conflicts in shared memory access
// - uses barrier synchronization for tile loading

const TILE_SIZE: u32 = 16;

struct MatrixDims {
    aRows: u32,        // rows of A and C
    k: u32,           // inner dimension
    bCols: u32,       // columns of B and C
    pad: u32,         // padding for 16-byte alignment
};

@group(0) @binding(0)
var<storage, read> A: array<f32>;  // [aRows, k]

@group(0) @binding(1)
var<storage, read> B: array<f32>;  // [k, bCols]

@group(0) @binding(2)
var<storage, read_write> C: array<f32>;  // [aRows, bCols]

@group(0) @binding(3)
var<uniform> dims: MatrixDims;

var<workgroup> tileA: array<f32, TILE_SIZE * TILE_SIZE>;  // [TILE_SIZE, TILE_SIZE]
var<workgroup> tileB: array<f32, TILE_SIZE * TILE_SIZE>;  // [TILE_SIZE, TILE_SIZE]

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
        // Here, use the appropriate condition for B's coordinates:
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

const PATCHIFY_SHADER_CODE = `
// algorithm: converts standard image format into patch-based representation
// - direct index computation for efficient memory mapping
// - handles arbitrary patch sizes and channel counts
// - coalesced memory access pattern for image data
// - single-pass transformation without intermediate buffers
// - boundary checking for non-perfect divisions
// - maintains spatial locality in patch output

struct PatchifyUniforms {
    imageWidth : u32,
    imageHeight : u32,
    channels : u32,
    patchSize : u32,
}
@group(0) @binding(0) var<storage, read> imageInput : array<f32>;  // [height, width, channels]
@group(0) @binding(1) var<uniform> params : PatchifyUniforms;
@group(0) @binding(2) var<storage, read_write> patchesOutput : array<f32>;  // [numPatches, channels, patchSize, patchSize]

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3 < u32>)
{
    //The total number of elements equals (number of patches × patchSize² × channels)
    let index = gid.x;
    let width = params.imageWidth;
    let height = params.imageHeight;
    let channels = params.channels;
    let pSize = params.patchSize;

    let numPatchesX = width / pSize;
    let numPatchesY = height / pSize;
    let patchCount = numPatchesX * numPatchesY;
    let patchDim = channels * pSize * pSize;
    let totalElements = patchCount * patchDim;
    if(index >= totalElements)
    {
        return;
    }

    //Determine which patch and which element in that patch.
    let patchIndex = index / patchDim;
    let innerIndex = index % patchDim;

    //Compute the channel as well as the spatial (local) indices.
    let channel = innerIndex / (pSize * pSize);
    let pixelIndex = innerIndex % (pSize * pSize);
    let localY = pixelIndex / pSize;
    let localX = pixelIndex % pSize;

    //Locate the patch in the grid.
    let patchRow = patchIndex / numPatchesX;
    let patchCol = patchIndex % numPatchesX;
    let x = patchCol * pSize + localX;
    let y = patchRow * pSize + localY;

    //Calculate the index into the flat image array.
    let imageIndex = (y * width + x) * channels + channel;
    if(x < width && y < height && imageIndex < arrayLength(&imageInput))
    {
        patchesOutput[index] = imageInput[imageIndex];
    }
}
`;

const SCALE_SHADER_CODE = `
// algorithm: performs scalar multiplication on a matrix
// - uses vec4 vectorization for efficient memory access
// - handles non-aligned matrix sizes with scalar fallback
// - coalesced memory access pattern for better throughput
// - single thread handles remainder to avoid thread divergence
// - minimizes atomic operations by using vectorized writes

@group(0) @binding(0)
var<storage, read_write> matrix : array<f32>;  // [dims.x, dims.y]

@group(0) @binding(1)
var<uniform> scalar : f32;

@group(0) @binding(2)
var<uniform> dims : vec2<u32>;  // (rows, cols)

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

const SILU_SHADER_CODE = `
struct Uniforms {
    totalElements : u32,
};

@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read_write> output : array<f32>;
@group(0) @binding(2) var<uniform> uniforms : Uniforms;

fn sigmoid(x : f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id : vec3 < u32>)
{
    let idx : u32 = global_id.x;
    if (idx >= uniforms.totalElements)
    {
        return;
    }
    let a : f32 = input[idx];

    //SiLU activation: x * sigmoid(x)
    output[idx] = a * sigmoid(a);
}
`;

const SOFTMAX_SHADER_CODE = `
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
}`;

const UNPATCHIFY_SHADER_CODE = `
// algorithm: converts patch-based image representation back to standard image format
// - handles arbitrary patch sizes and channel counts
// - uses direct index computation instead of nested loops
// - coalesces memory accesses by computing patch indices efficiently
// - handles boundary conditions for non-perfect divisions
// - single-pass transformation without intermediate buffers

struct PatchifyUniforms {
    imageWidth : u32,
    imageHeight : u32,
    channels : u32,
    patchSize : u32,
}

@group(0) @binding(0) var<storage, read> patchesInput : array<f32>;  // [numPatches, channels, patchSize, patchSize]
@group(0) @binding(1) var<uniform> params : PatchifyUniforms;
@group(0) @binding(2) var<storage, read_write> imageOutput : array<f32>;  // [height, width, channels]

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let index = gid.x;
    let width = params.imageWidth;
    let height = params.imageHeight;
    let channels = params.channels;
    let pSize = params.patchSize;

    let numPatchesX = width / pSize;
    let numPatchesY = height / pSize;
    let patchCount = numPatchesX * numPatchesY;
    let patchDim = channels * pSize * pSize;
    let totalElements = patchCount * patchDim;
    if(index >= totalElements) {
        return;
    }

    let patchIndex = index / patchDim;
    let innerIndex = index % patchDim;

    let channel = innerIndex / (pSize * pSize);
    let pixelIndex = innerIndex % (pSize * pSize);
    let localY = pixelIndex / pSize;
    let localX = pixelIndex % pSize;

    let patchRow = patchIndex / numPatchesX;
    let patchCol = patchIndex % numPatchesX;
    let x = patchCol * pSize + localX;
    let y = patchRow * pSize + localY;

    let imageIndex = (y * width + x) * channels + channel;
    if(x < width && y < height && imageIndex < arrayLength(&imageOutput)) {
        imageOutput[imageIndex] = patchesInput[index];
    }
}
`;

// Utility functions

async function logBuffer(device, buffer, label = "Buffer stats (first 300)") {
  return;
  //  return;
  // Create a staging buffer for reading
  const stagingBuffer = device.createBuffer({
    size: buffer.size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  // Create and submit copy command
  const commandEncoder = device.createCommandEncoder();
  commandEncoder.copyBufferToBuffer(buffer, 0, stagingBuffer, 0, buffer.size);
  device.queue.submit([commandEncoder.finish()]);

  // Map the staging buffer and read its contents
  await stagingBuffer.mapAsync(GPUMapMode.READ);
  const copyArrayBuffer = stagingBuffer.getMappedRange();
  const data = new Float32Array(copyArrayBuffer);

  // Calculate stats of first 300 numbers
  const length = data.length;
  let sum = 0;
  let min = Infinity;
  let max = -Infinity;

  // First pass: sum, min, max
  for (let i = 0; i < length; i++) {
    const value = data[i];
    sum += value;
    min = Math.min(min, value);
    max = Math.max(max, value);
  }

  const mean = sum / length;

  // Second pass: standard deviation
  let sumSquaredDiff = 0;
  for (let i = 0; i < length; i++) {
    const diff = data[i] - mean;
    sumSquaredDiff += diff * diff;
  }
  const std = Math.sqrt(sumSquaredDiff / length);

  const stats = {
    length,
    mean,
    std,
    min,
    max,
    first5: data.slice(0, 5),
  };

  // Log the results
  console.log(`${label}:`, stats);

  // Clean up
  stagingBuffer.unmap();
  stagingBuffer.destroy();

  return stats;
}

function createBufferFromArray(device, array, usage) {
  const buffer = device.createBuffer({
    size: array.byteLength,
    usage,
    mappedAtCreation: true,
  });
  new Float32Array(buffer.getMappedRange()).set(array);
  buffer.unmap();
  return buffer;
}

// Helper function to create a GPU buffer with random values
function createRandomBuffer(device, size, mean = 0.0, stddev = 0.02) {
  // Create random values using normal distribution
  const values = new Float32Array(size);

  const buffer = device.createBuffer({
    size: values.byteLength,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_DST |
      GPUBufferUsage.COPY_SRC,
    mappedAtCreation: true,
  });
  new Float32Array(buffer.getMappedRange()).set(values);
  buffer.unmap();
  return buffer;
}

function flattenWeight(weight) {
  // Check if the first element is itself an array.
  if (Array.isArray(weight[0])) {
    // Manually flatten (you could also use weight.flat() if available)
    const flatArr = [];
    for (let i = 0; i < weight.length; i++) {
      for (let j = 0; j < weight[i].length; j++) {
        flatArr.push(weight[i][j]);
      }
    }
    return new Float32Array(flatArr);
  } else {
    return new Float32Array(weight);
  }
}
