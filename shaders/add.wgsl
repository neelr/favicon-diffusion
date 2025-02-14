const VECTOR_SIZE : u32 = 4u;
const TOTAL_ELEMENTS : u32 = _NUM_ELEMENTS_u;

@group(0) @binding(0) var<storage, read> A : array<f32>;
@group(0) @binding(1) var<storage, read> B : array<f32>;
@group(0) @binding(2) var<storage, read_write> C : array<f32>;

    //Use a workgroup size that suits your hardware (here 64 threads per group)
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3 < u32>)
{
    ch thread starts at a multiple of VECTOR_SIZE. (Dispatch count should be ceil(n / VECTOR_SIZE))
    let index : u32 = gid.x * VECTOR_SIZE;

    en there’s a full vector available, load 4 f32’s at once.
    if (index + VECTOR_SIZE <= TOTAL_ELEMENTS)
    {
        elements from A and B
        let a : vec4 < f32> = vec4 < f32 > (
        A[index + 0u],
        A[index + 1u],
        A[index + 2u],
        A[index + 3u]
        );
        let b : vec4 < f32> = vec4 < f32 > (
        B[index + 0u],
        B[index + 1u],
        B[index + 2u],
        B[index + 3u]
        );
        let result : vec4 < f32> = a + b;

        back the result in one shot.
        C[index + 0u] = result.x;
        C[index + 1u] = result.y;
        C[index + 2u] = result.z;
        C[index + 3u] = result.w;
    } else {
        ck for any leftover tail elements when TOTAL_ELEMENTS isn’t a multiple of VECTOR_SIZE.
        for (var i : u32 = index; i < TOTAL_ELEMENTS; i = i + 1u)
        {
            C[i] = A[i] + B[i];
        }
    }
}
