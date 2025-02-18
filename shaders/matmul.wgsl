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
}