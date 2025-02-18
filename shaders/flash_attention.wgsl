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
