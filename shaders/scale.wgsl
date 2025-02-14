
@group(0) @binding(0)
var<storage, read_write> matrix : array<f32>;

@group(0) @binding(1)
var<uniform> scalar : f32;

@group(0) @binding(2)
var<uniform> dims : vec2 < u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) globalID : vec3 < u32>)
{
    let totalEle = dims.x * dims.y;
    let numVec4 = totalEle / 4u;
    let remainder = totalEle % 4u;
    let tid = globalID.x;
    let s = scalar;

    if(tid < numVec4)
    {
        let offset = tid * 4u;
        ad 4 elements at once.
        var v = vec4 < f32 > (
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
    //Let one thread handle any extra elements.
    if(tid == 0u && remainder > 0u)
    {
        for(var i = numVec4 * 4u; i < totalEle; i = i + 1u)
        {
            matrix[i] = matrix[i] * s;
        }
    }
}
