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
