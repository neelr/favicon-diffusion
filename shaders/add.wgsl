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
}