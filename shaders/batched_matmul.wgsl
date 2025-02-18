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
}