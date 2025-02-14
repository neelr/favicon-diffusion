@group(0) @binding(0) var<storage, read> scores : array<f32>;
@group(0) @binding(1) var<storage, read_write> out : array<f32>;
@group(0) @binding(2) var<uniform> dims : vec2 < u32>; //(rows, cols)

//Use separate shared arrays for max and sum.
var<workgroup> shared_max : array<f32, 256>;
var<workgroup> shared_sum : array<f32, 256>;

@compute @workgroup_size(256)
fn main(
@builtin(workgroup_id) workgroup_id : vec3 < u32>,
@builtin(local_invocation_id) local_id : vec3 < u32>
)
{
    let row : u32 = workgroup_id.x;
    if (row >= dims.x)
    { return; }
        let numCols : u32 = dims.y;
        che the beginning of the row to avoid repeated multiplication.
        let rowStart : u32 = row * numCols;

        - Step 1: Compute the maximum of this row ---
        var local_max : f32 = -1e20;
        ch thread processes multiple columns in a strided loop.
        for (var col : u32 = local_id.x; col < numCols; col += 256u)
        {
            local_max = max(local_max, scores[rowStart + col]);
        }
        shared_max[local_id.x] = local_max;
        workgroupBarrier();

        rolled tree reduction for the maximum.
        if (local_id.x < 128u)
        {
            shared_max[local_id.x] = max(shared_max[local_id.x], shared_max[local_id.x + 128u]);
        }
        workgroupBarrier();
        if (local_id.x < 64u)
        {
            shared_max[local_id.x] = max(shared_max[local_id.x], shared_max[local_id.x + 64u]);
        }
        workgroupBarrier();
        if (local_id.x < 32u)
        {
            shared_max[local_id.x] = max(shared_max[local_id.x], shared_max[local_id.x + 32u]);
            shared_max[local_id.x] = max(shared_max[local_id.x], shared_max[local_id.x + 16u]);
            shared_max[local_id.x] = max(shared_max[local_id.x], shared_max[local_id.x + 8u]);
            shared_max[local_id.x] = max(shared_max[local_id.x], shared_max[local_id.x + 4u]);
            shared_max[local_id.x] = max(shared_max[local_id.x], shared_max[local_id.x + 2u]);
            shared_max[local_id.x] = max(shared_max[local_id.x], shared_max[local_id.x + 1u]);
        }
        workgroupBarrier();
        let max_val : f32 = shared_max[0];

        - Step 2: Compute exponentials (exp(val - max)) and sum them ---
        var local_sum : f32 = 0.0;
        for (var col : u32 = local_id.x; col < numCols; col += 256u)
        {
            let idx : u32 = rowStart + col;
            let exp_val : f32 = exp(scores[idx] - max_val);
            out[idx] = exp_val;tore the computed exponentials
            local_sum += exp_val;
        }
        shared_sum[local_id.x] = local_sum;
        workgroupBarrier();

        rolled tree reduction for the sum.
        if (local_id.x < 128u)
        {
            shared_sum[local_id.x] += shared_sum[local_id.x + 128u];
        }
        workgroupBarrier();
        if (local_id.x < 64u)
        {
            shared_sum[local_id.x] += shared_sum[local_id.x + 64u];
        }
        workgroupBarrier();
        if (local_id.x < 32u)
        {
            shared_sum[local_id.x] += shared_sum[local_id.x + 32u];
            shared_sum[local_id.x] += shared_sum[local_id.x + 16u];
            shared_sum[local_id.x] += shared_sum[local_id.x + 8u];
            shared_sum[local_id.x] += shared_sum[local_id.x + 4u];
            shared_sum[local_id.x] += shared_sum[local_id.x + 2u];
            shared_sum[local_id.x] += shared_sum[local_id.x + 1u];
        }
        workgroupBarrier();
        let sum_val : f32 = shared_sum[0];

        - Step 3: Normalize the row ---
        for (var col : u32 = local_id.x; col < numCols; col += 256u)
        {
            let idx : u32 = rowStart + col;
            out[idx] = out[idx] / sum_val;
        }
    }
