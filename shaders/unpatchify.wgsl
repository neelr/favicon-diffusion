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
