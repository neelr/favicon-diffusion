// algorithm: converts standard image format into patch-based representation
// - direct index computation for efficient memory mapping
// - handles arbitrary patch sizes and channel counts
// - coalesced memory access pattern for image data
// - single-pass transformation without intermediate buffers
// - boundary checking for non-perfect divisions
// - maintains spatial locality in patch output

struct PatchifyUniforms {
    imageWidth : u32,
    imageHeight : u32,
    channels : u32,
    patchSize : u32,
}
@group(0) @binding(0) var<storage, read> imageInput : array<f32>;  // [height, width, channels]
@group(0) @binding(1) var<uniform> params : PatchifyUniforms;
@group(0) @binding(2) var<storage, read_write> patchesOutput : array<f32>;  // [numPatches, channels, patchSize, patchSize]

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3 < u32>)
{
    //The total number of elements equals (number of patches × patchSize² × channels)
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
    if(index >= totalElements)
    {
        return;
    }

    //Determine which patch and which element in that patch.
    let patchIndex = index / patchDim;
    let innerIndex = index % patchDim;

    //Compute the channel as well as the spatial (local) indices.
    let channel = innerIndex / (pSize * pSize);
    let pixelIndex = innerIndex % (pSize * pSize);
    let localY = pixelIndex / pSize;
    let localX = pixelIndex % pSize;

    //Locate the patch in the grid.
    let patchRow = patchIndex / numPatchesX;
    let patchCol = patchIndex % numPatchesX;
    let x = patchCol * pSize + localX;
    let y = patchRow * pSize + localY;

    //Calculate the index into the flat image array.
    let imageIndex = (y * width + x) * channels + channel;
    if(x < width && y < height && imageIndex < arrayLength(&imageInput))
    {
        patchesOutput[index] = imageInput[imageIndex];
    }
}
