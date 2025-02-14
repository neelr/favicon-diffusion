const TILE_SIZE : u32 = 16u;

struct MatrixDims {
    aRows : u32,    //rows of A and C
    k : u32,        //inner dimension
    bCols : u32,    //columns of C (and, for normal mode, columns of B)
    flagTranspose : u32 = B stored normally (dims.k x dims.bCols)
                    //1 = B stored transposed (dims.bCols x dims.k)
};

@group(0) @binding(0)
var<storage, read> A : array<f32>;

@group(0) @binding(1)
var<storage, read> B : array<f32>;

@group(0) @binding(2)
var<storage, read_write> C : array<f32>;

@group(0) @binding(3)
var<uniform> dims : MatrixDims;

//Flattened workgroup arrays.
var<workgroup> tileA : array<f32, TILE_SIZE * TILE_SIZE>;
var<workgroup> tileB : array<f32, TILE_SIZE * TILE_SIZE>;

@compute @workgroup_size(TILE_SIZE, TILE_SIZE)
fn main(
@builtin(global_invocation_id) global_id : vec3 < u32>,
@builtin(local_invocation_id) local_id : vec3 < u32>
)
{
    let row = global_id.x;
    let col = global_id.y;
    var sum : f32 = 0.0;
    let numTiles = (dims.k + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t : u32 = 0u; t < numTiles; t = t + 1u)
    {
        let tileIndex = local_id.x * TILE_SIZE + local_id.y;

        //Load tile element from A.
        let aCol = t * TILE_SIZE + local_id.y;
        if (row < dims.aRows && aCol < dims.k)
        {
            tileA[tileIndex] = A[row * dims.k + aCol];
        } else {
            tileA[tileIndex] = 0.0;
        }

        //Always load B for every workgroup thread.
        if (dims.flagTranspose == 1u)
        {
            //In transposed mode, B is stored as [bCols, k]. We swap the loading indices.
            let bIndex = t * TILE_SIZE + local_id.x;
            let tileBIndex = local_id.y * TILE_SIZE + local_id.x;
            if (col < dims.bCols && bIndex < dims.k)
            {
                tileB[tileBIndex] = B[col * dims.k + bIndex];
            } else {
                tileB[tileBIndex] = 0.0;
            }
        } else {
            //Normal mode: B is stored as [dims.k, dims.bCols] in row–major order.
            let bIndex = t * TILE_SIZE + local_id.x;
            if (bIndex < dims.k && col < dims.bCols)
            {
                tileB[tileIndex] = B[bIndex * dims.bCols + col];
            } else {
                tileB[tileIndex] = 0.0;
            }
        }

        //Synchronize to ensure the entire tile is loaded.
        workgroupBarrier();

        //Multiply the elements of the tile.
        if (dims.flagTranspose == 1u)
        {
            //With transposed B, our load swapped the indices.
            for (var kIdx : u32 = 0u; kIdx < TILE_SIZE; kIdx = kIdx + 1u)
            {
                //Note: For A the index remains the same.
                let aVal = tileA[local_id.x * TILE_SIZE + kIdx];
                let bVal = tileB[local_id.y * TILE_SIZE + kIdx];
                sum = sum + aVal * bVal;
            }
        } else {
            //Normal multiplication.
            for (var kIdx : u32 = 0u; kIdx < TILE_SIZE; kIdx = kIdx + 1u)
            {
                let aVal = tileA[local_id.x * TILE_SIZE + kIdx];
                let bVal = tileB[kIdx * TILE_SIZE + local_id.y];
                sum = sum + aVal * bVal;
            }
        }

        workgroupBarrier();
    }

    //Write back the final result only if within C’s bounds.
    if (row < dims.aRows && col < dims.bCols)
    {
        C[row * dims.bCols + col] = sum;
    }
}
