/* dit_webgpu.js – A skeletal DiT implementation using WebGPU

   This script sets up a GPU device, defines simplified classes corresponding
   to the PyTorch DiT model modules (SinusoidalPosEmb, AdaLN, Attention,
   FeedForward, TransformerBlock, DiT) and then runs a dummy forward pass on a 
   random input image and random time value.
*/

// Utility: create a GPU buffer from an array.
function createBufferFromArray(device, array, usage) {
  const buffer = device.createBuffer({
    size: array.byteLength,
    usage,
    mappedAtCreation: true,
  });
  new Float32Array(buffer.getMappedRange()).set(array);
  buffer.unmap();
  return buffer;
}

// Helper function to create a GPU buffer with random values
function createRandomBuffer(device, size, mean = 0.0, stddev = 0.02) {
  // Create random values using normal distribution
  const values = new Float32Array(size);

  const buffer = device.createBuffer({
    size: values.byteLength,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_DST |
      GPUBufferUsage.COPY_SRC,
    mappedAtCreation: true,
  });
  new Float32Array(buffer.getMappedRange()).set(values);
  buffer.unmap();
  return buffer;
}

function flattenWeight(weight) {
  // Check if the first element is itself an array.
  if (Array.isArray(weight[0])) {
    // Manually flatten (you could also use weight.flat() if available)
    const flatArr = [];
    for (let i = 0; i < weight.length; i++) {
      for (let j = 0; j < weight[i].length; j++) {
        flatArr.push(weight[i][j]);
      }
    }
    return new Float32Array(flatArr);
  } else {
    return new Float32Array(weight);
  }
}

/* WGSL shader code for patch extraction (“patchify”).
   Assumes the input image is stored as a flattened Float32Array
   in CHW order (i.e. channels, then height, then width). For an image
   of dimensions [channels, H, W], and a given patchSize, we have:
     • numPatchesX = H / patchSize,
     • numPatchesY = W / patchSize, and
     • each patch is flattened into a vector of length channels * patchSize * patchSize.
*/
// Patchify shader: converts image to patches
const patchifyShaderCode = `
struct PatchifyUniforms {
    imageWidth: u32,
    imageHeight: u32,
    channels: u32,
    patchSize: u32,
}
@group(0) @binding(0) var<storage, read> imageInput : array<f32>;
@group(0) @binding(1) var<uniform> params : PatchifyUniforms;
@group(0) @binding(2) var<storage, read_write> patchesOutput : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
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
    
    if (index >= totalElements) {
        return;
    }

    let patchIndex = index / patchDim;
    let innerIndex = index % patchDim;
    
    let patchY = patchIndex / numPatchesX;
    let patchX = patchIndex % numPatchesX;
    
    let localY = (innerIndex / pSize) % pSize;
    let localX = innerIndex % pSize;
    
    let x = patchX * pSize + localX;
    let y = patchY * pSize + localY;
    
    if (x < width && y < height) {
        let imageIndex = y * width + x;
        if (imageIndex < arrayLength(&imageInput)) {
            patchesOutput[index] = imageInput[imageIndex];
        }
    }
}`;

// Unpatchify shader: converts patches back to image
const unpatchifyShaderCode = `
struct PatchifyUniforms {
    imageWidth: u32,
    imageHeight: u32,
    channels: u32,
    patchSize: u32,
}
@group(0) @binding(0) var<storage, read> patchesInput : array<f32>;
@group(0) @binding(1) var<uniform> params : PatchifyUniforms;
@group(0) @binding(2) var<storage, read_write> imageOutput : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
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
    
    if (index >= totalElements) {
        return;
    }

    let patchIndex = index / patchDim;
    let innerIndex = index % patchDim;
    
    let patchY = patchIndex / numPatchesX;
    let patchX = patchIndex % numPatchesX;
    
    let localY = (innerIndex / pSize) % pSize;
    let localX = innerIndex % pSize;
    
    let x = patchX * pSize + localX;
    let y = patchY * pSize + localY;
    
    if (x < width && y < height) {
        let imageIndex = y * width + x;
        if (imageIndex < arrayLength(&imageOutput)) {
            imageOutput[imageIndex] = patchesInput[index];
        }
    }
}`;

// Matrix multiplication shader
const mmShaderCode = `
struct MatrixDims {
    aRows: u32,
    k: u32,
    bCols: u32,
};

@group(0) @binding(0) var<storage, read> A : array<f32>;
@group(0) @binding(1) var<storage, read> B : array<f32>;
@group(0) @binding(2) var<storage, read_write> C : array<f32>;
@group(0) @binding(3) var<uniform> dims : MatrixDims;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
    let row = GlobalInvocationID.x;
    let col = GlobalInvocationID.y;

    if (row >= dims.aRows || col >= dims.bCols) {
        return;
    }

    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < dims.k; i = i + 1u) {
        let aIndex = row * dims.k + i;
        let bIndex = i * dims.bCols + col;
        if (aIndex < arrayLength(&A) && bIndex < arrayLength(&B)) {
            sum = sum + A[aIndex] * B[bIndex];
        }
    }

    let outIndex = row * dims.bCols + col;
    if (outIndex < dims.aRows * dims.bCols) {
        C[outIndex] = sum;
    }
}`;

// Main patchify function
async function patchifyImage(device, imageBuffer, params) {
  // Create shader module and pipeline if they don't exist
  if (!device.patchifyPipeline) {
    const shaderModule = device.createShaderModule({
      code: patchifyShaderCode,
    });
    device.patchifyPipeline = device.createComputePipeline({
      layout: "auto",
      compute: {
        module: shaderModule,
        entryPoint: "main",
      },
    });
  }

  // Compute dimensions
  const numPatchesX = params.imageWidth / params.patchSize;
  const numPatchesY = params.imageHeight / params.patchSize;
  const patchCount = numPatchesX * numPatchesY;
  const patchDim = params.channels * params.patchSize * params.patchSize;
  const totalElements = patchCount * patchDim;

  // Create uniform buffer
  const uniformArray = new Uint32Array([
    params.imageWidth,
    params.imageHeight,
    params.channels,
    params.patchSize,
  ]);
  const uniformBuffer = device.createBuffer({
    size: uniformArray.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(uniformBuffer, 0, uniformArray);

  // Create output buffer
  const patchBuffer = device.createBuffer({
    size: totalElements * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  // Create bind group
  const bindGroup = device.createBindGroup({
    layout: device.patchifyPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: imageBuffer } },
      { binding: 1, resource: { buffer: uniformBuffer } },
      { binding: 2, resource: { buffer: patchBuffer } },
    ],
  });

  // Execute compute pass
  const commandEncoder = device.createCommandEncoder();
  const pass = commandEncoder.beginComputePass();
  pass.setPipeline(device.patchifyPipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(totalElements / 64));
  pass.end();

  device.queue.submit([commandEncoder.finish()]);
  await device.queue.onSubmittedWorkDone();

  return patchBuffer;
}

// Main unpatchify function
async function unpatchifyPatches(device, patchesBuffer, params) {
  // Create shader module and pipeline if they don't exist
  if (!device.unpatchifyPipeline) {
    const shaderModule = device.createShaderModule({
      code: unpatchifyShaderCode,
    });
    device.unpatchifyPipeline = device.createComputePipeline({
      layout: "auto",
      compute: {
        module: shaderModule,
        entryPoint: "main",
      },
    });
  }

  // Compute dimensions
  const numPatchesX = params.imageWidth / params.patchSize;
  const numPatchesY = params.imageHeight / params.patchSize;
  const patchCount = numPatchesX * numPatchesY;
  const patchDim = params.channels * params.patchSize * params.patchSize;
  const totalElements = patchCount * patchDim;

  // Create uniform buffer
  const uniformArray = new Uint32Array([
    params.imageWidth,
    params.imageHeight,
    params.channels,
    params.patchSize,
  ]);
  const uniformBuffer = device.createBuffer({
    size: uniformArray.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(uniformBuffer, 0, uniformArray);

  // Create output buffer
  const imageBuffer = device.createBuffer({
    size: params.channels * params.imageWidth * params.imageHeight * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  // Create bind group
  const bindGroup = device.createBindGroup({
    layout: device.unpatchifyPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: patchesBuffer } },
      { binding: 1, resource: { buffer: uniformBuffer } },
      { binding: 2, resource: { buffer: imageBuffer } },
    ],
  });

  // Execute compute pass
  const commandEncoder = device.createCommandEncoder();
  const pass = commandEncoder.beginComputePass();
  pass.setPipeline(device.unpatchifyPipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(totalElements / 64));
  pass.end();

  device.queue.submit([commandEncoder.finish()]);
  await device.queue.onSubmittedWorkDone();

  return imageBuffer;
}

(async () => {
  // Request a GPU adapter and device.
  if (!navigator.gpu) {
    console.error(
      "WebGPU not available. Try using Chrome Canary or Edge with WebGPU enabled."
    );
    return;
  }
  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice();

  const adaLNShaderCode = `

struct AdaLNUniforms {
  nRows: f32,      // number of rows (will be cast to u32)
  rowWidth: f32,   // number of elements per row (i.e. hidden dimension)
  epsilon: f32,
};

@group(0) @binding(0) var<storage, read> x : array<f32>;  
// “shift” buffer (the β parameters), assumed to be of length rowWidth.
@group(0) @binding(1) var<storage, read> shift : array<f32>;  
// “scale” buffer (the γ parameters), assumed to be of length rowWidth.
@group(0) @binding(2) var<storage, read> scale : array<f32>;  
@group(0) @binding(3) var<storage, read_write> out : array<f32>;
@group(0) @binding(4) var<uniform> uniforms : AdaLNUniforms;

// Use a workgroup size of 256 (one workgroup per sample row)
var<workgroup> sharedSum: array<f32, 256>;
var<workgroup> sharedSumSq: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(workgroup_id) workgroup_id: vec3<u32>) {
  // Each workgroup is responsible for one “row” (sample)
  let row : u32 = workgroup_id.x;
  let width : u32 = u32(uniforms.rowWidth);  
  let tid : u32 = local_id.x;
  let numThreads : u32 = 256u;

  // Each thread accumulates a partial sum and partial sum-of-squares over multiple
  // elements in the row.
  var sum: f32 = 0.0;
  var sumSq: f32 = 0.0;
  for (var col: u32 = tid; col < width; col = col + numThreads) {
    let idx = row * width + col;
    let value = x[idx];
    sum = sum + value;
    sumSq = sumSq + value * value;
  }
  sharedSum[tid] = sum;
  sharedSumSq[tid] = sumSq;
  workgroupBarrier();

  // Perform parallel reduction to compute the full sum and sum-of-squares.
  var stride = numThreads / 2u;
  while (stride > 0u) {
    if (tid < stride) {
      sharedSum[tid] = sharedSum[tid] + sharedSum[tid + stride];
      sharedSumSq[tid] = sharedSumSq[tid] + sharedSumSq[tid + stride];
    }
    workgroupBarrier();
    stride = stride / 2u;
  }
  
  // Now thread 0 holds the complete sum and sumSq for this row.
  let mean : f32 = sharedSum[0] / f32(width);
  let variance : f32 = sharedSumSq[0] / f32(width) - mean * mean;
  let invStd : f32 = 1.0 / sqrt(variance + uniforms.epsilon);

  // Each thread then writes normalized outputs for its chunk.
  for (var col: u32 = tid; col < width; col = col + numThreads) {
    let idx = row * width + col;
    // Normalize: subtract the mean and multiply by 1/sqrt(variance + epsilon)
    let normalized = (x[idx] - mean) * invStd;
    // Apply the adaptive parameters.
    // Here we use “(1 + scale)” multiplied by the normalized value, and then add shift.
    out[idx] = normalized * (1.0 + scale[col]) + shift[col];
  }
}
`;

  // Create a single shader module for matrix multiplication.
  const mmShaderModule = device.createShaderModule({ code: mmShaderCode });

  // GPULinear simulates a linear (fully-connected) layer using a matrix multiplication.
  class GPULinear {
    constructor(device, inFeatures, outFeatures) {
      this.device = device;
      this.inFeatures = inFeatures;
      this.outFeatures = outFeatures;

      // Initialize weights using Xavier/Glorot initialization
      const stddev = Math.sqrt(2.0 / (inFeatures + outFeatures));
      const weightArray = new Float32Array(outFeatures * inFeatures);
      // Create weight buffer with proper usage flags
      this.weightBuffer = device.createBuffer({
        size: weightArray.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
      });
      new Float32Array(this.weightBuffer.getMappedRange()).set(weightArray);
      this.weightBuffer.unmap();

      // Create the compute pipeline
      const shaderModule = device.createShaderModule({
        code: mmShaderCode,
      });

      this.pipeline = device.createComputePipeline({
        layout: "auto",
        compute: {
          module: shaderModule,
          entryPoint: "main",
        },
      });
    }

    async forward(inputBuffer, inputShape /*[m, k]*/) {
      const m = inputShape[0]; // Number of samples
      const k = this.inFeatures;
      const n = this.outFeatures;

      // Create uniform buffer for dimensions
      const dimsArray = new Uint32Array([m, k, n]);
      const dimsBuffer = this.device.createBuffer({
        size: dimsArray.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      this.device.queue.writeBuffer(dimsBuffer, 0, dimsArray);

      // Create output buffer
      const outputBuffer = this.device.createBuffer({
        size: m * n * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });

      // Create bind group
      const bindGroup = this.device.createBindGroup({
        layout: this.pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: inputBuffer } },
          { binding: 1, resource: { buffer: this.weightBuffer } },
          { binding: 2, resource: { buffer: outputBuffer } },
          { binding: 3, resource: { buffer: dimsBuffer } },
        ],
      });

      // Execute compute pass
      const commandEncoder = this.device.createCommandEncoder();
      const pass = commandEncoder.beginComputePass();
      pass.setPipeline(this.pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(Math.ceil(m / 16), Math.ceil(n / 16));
      pass.end();

      this.device.queue.submit([commandEncoder.finish()]);
      await this.device.queue.onSubmittedWorkDone();

      return outputBuffer;
    }
  }

  // GPUAdaLN – a skeleton for adaptive layer normalization.
  class GPUAdaLN {
    constructor(device, dim, timeEmbDim) {
      this.device = device;
      this.dim = dim;
      this.timeEmbDim = timeEmbDim;

      // Initialize the linear projections for scale and shift using GPULinear
      this.scaleProj = new GPULinear(device, timeEmbDim, dim);
      this.shiftProj = new GPULinear(device, timeEmbDim, dim);

      // Create the AdaLN compute pipeline using existing shader code
      this.adaLNModule = device.createShaderModule({ code: adaLNShaderCode });
      this.pipeline = device.createComputePipeline({
        layout: "auto",
        compute: { module: this.adaLNModule, entryPoint: "main" },
      });
    }

    async forward(xBuffer, timeEmbBuffer, numElements) {
      // First, project time embeddings to get scale and shift parameters
      // timeEmbBuffer shape is [1, timeEmbDim]
      const shiftValues = await this.shiftProj.forward(timeEmbBuffer, [
        1,
        this.timeEmbDim,
      ]);
      const scaleValues = await this.scaleProj.forward(timeEmbBuffer, [
        1,
        this.timeEmbDim,
      ]);

      // Determine dimensions for normalization
      const rowWidth = this.dim;
      const nRows = numElements / this.dim;

      // Create uniform buffer for AdaLN
      const uniformData = new Float32Array([nRows, rowWidth, 1e-5]); // epsilon = 1e-5
      const uniformBuffer = createBufferFromArray(
        this.device,
        uniformData,
        GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
      );

      // Create output buffer
      const outputBuffer = this.device.createBuffer({
        size: numElements * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });

      // Create bind group for the AdaLN pipeline
      const bindGroup = this.device.createBindGroup({
        layout: this.pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: xBuffer } },
          { binding: 1, resource: { buffer: shiftValues } },
          { binding: 2, resource: { buffer: scaleValues } },
          { binding: 3, resource: { buffer: outputBuffer } },
          { binding: 4, resource: { buffer: uniformBuffer } },
        ],
      });

      // Execute compute pass
      const commandEncoder = this.device.createCommandEncoder();
      const pass = commandEncoder.beginComputePass();
      pass.setPipeline(this.pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(nRows);
      pass.end();

      this.device.queue.submit([commandEncoder.finish()]);
      await this.device.queue.onSubmittedWorkDone();

      return outputBuffer;
    }
  }
  // GPUAttention – a skeleton for multi-head attention.
  class GPUAttention {
    constructor(device, dim, dimHead) {
      this.dim = dim; // Input embedding dimension.
      this.dimHead = dimHead; // Projection dimension for Q, K, V.
      this.device = device;

      // ---------- Softmax shader ---------------------------
      const softmaxWGSL = `
  @group(0) @binding(0) var<storage, read> scores: array<f32>;
  @group(0) @binding(1) var<storage, read_write> out: array<f32>;
  @group(0) @binding(2) var<uniform> dims: vec2<u32>; // (rows, cols)
  
  var<workgroup> shared_vals: array<f32, 256>;
  
  @compute @workgroup_size(256)
  fn main(
      @builtin(workgroup_id) workgroup_id: vec3<u32>,
      @builtin(local_invocation_id) local_id: vec3<u32>
  ) {
      let row: u32 = workgroup_id.x;
      if (row >= dims.x) { return; }
      let numCols: u32 = dims.y;
  
      // Step 1: Compute max in this row.
      var max_val: f32 = -1e20;
      for (var col: u32 = local_id.x; col < numCols; col = col + 256u) {
          max_val = max(max_val, scores[row * numCols + col]);
      }
      shared_vals[local_id.x] = max_val;
      workgroupBarrier();
  
      var stride: u32 = 256u / 2u;
      loop {
          if (local_id.x < stride) {
              shared_vals[local_id.x] = max(shared_vals[local_id.x], shared_vals[local_id.x + stride]);
          }
          workgroupBarrier();
          if (stride == 1u) { break; }
          stride = stride / 2u;
      }
      max_val = shared_vals[0];
  
      // Step 2: Compute exp(value - max) and row sum.
      var sum_val: f32 = 0.0;
      for (var col: u32 = local_id.x; col < numCols; col = col + 256u) {
          let idx = row * numCols + col;
          let v = exp(scores[idx] - max_val);
          out[idx] = v;
          sum_val = sum_val + v;
      }
      shared_vals[local_id.x] = sum_val;
      workgroupBarrier();
      stride = 256u / 2u;
      loop {
          if (local_id.x < stride) {
              shared_vals[local_id.x] = shared_vals[local_id.x] + shared_vals[local_id.x + stride];
          }
          workgroupBarrier();
          if (stride == 1u) { break; }
          stride = stride / 2u;
      }
      sum_val = shared_vals[0];
  
      // Step 3: Normalize each row.
      for (var col: u32 = local_id.x; col < numCols; col = col + 256u) {
          let idx = row * numCols + col;
          out[idx] = out[idx] / sum_val;
      }
  }
  `;
      this.softmaxModule = device.createShaderModule({ code: softmaxWGSL });
      this.softmaxPipeline = device.createComputePipeline({
        layout: "auto",
        compute: { module: this.softmaxModule, entryPoint: "main" },
      });

      // ---------- Batched Matrix Multiplication shader -----------
      const batchedMatMulWGSL = `
  struct MatrixDims {
    aRows: u32,
    k: u32,
    bCols: u32,
    flagTranspose: u32,
  };
  
  @group(0) @binding(3) var<uniform> dims : MatrixDims;
  @group(0) @binding(0) var<storage, read> A : array<f32>;
  @group(0) @binding(1) var<storage, read> B : array<f32>;
  @group(0) @binding(2) var<storage, read_write> C : array<f32>;
  
  @compute @workgroup_size(16, 16)
  fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let col = global_id.y;
    if (row >= dims.aRows || col >= dims.bCols) { return; }
    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < dims.k; i = i + 1u) {
      let aIndex = row * dims.k + i;
      var bIndex: u32;
      if (dims.flagTranspose == 1u) {
        bIndex = col * dims.k + i;
      } else {
        bIndex = i * dims.bCols + col;
      }
      sum = sum + A[aIndex] * B[bIndex];
    }
    C[row * dims.bCols + col] = sum;
  }
      `;
      this.batchedMatMulModule = device.createShaderModule({
        code: batchedMatMulWGSL,
      });
      this.batchedMatMulPipeline = device.createComputePipeline({
        layout: "auto",
        compute: { module: this.batchedMatMulModule, entryPoint: "main" },
      });

      // ---------- Scale Matrix shader ------------------------
      const scaleMatrixWGSL = `
  @group(0) @binding(0) var<storage, read_write> matrix : array<f32>;
  @group(0) @binding(1) var<uniform> scalar : f32;
  @group(0) @binding(2) var<uniform> dims : vec2<u32>;
  
  @compute @workgroup_size(64)
  fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    let totalElements = dims.x * dims.y;
    if (index >= totalElements) { return; }
    matrix[index] = matrix[index] * scalar;
  }
      `;
      this.scaleMatrixModule = device.createShaderModule({
        code: scaleMatrixWGSL,
      });
      this.scaleMatrixPipeline = device.createComputePipeline({
        layout: "auto",
        compute: { module: this.scaleMatrixModule, entryPoint: "main" },
      });

      // ---------------- Q, K, V Projections ----------------
      // Instead of projecting [dim, dim] we now project from dim to dimHead.
      // Initialize Q, K, V projections with random values
      const projSize = dim * dimHead;
      const projStddev = Math.sqrt(2.0 / (dim + dimHead));

      this.qProj = createRandomBuffer(device, projSize, 0.0, projStddev);
      this.kProj = createRandomBuffer(device, projSize, 0.0, projStddev);
      this.vProj = createRandomBuffer(device, projSize, 0.0, projStddev);

      // Initialize output projection
      const outProjSize = dimHead * dim;
      const outProjStddev = Math.sqrt(2.0 / (dimHead + dim));
      this.outProj = createRandomBuffer(
        device,
        outProjSize,
        0.0,
        outProjStddev
      );
    }

    // ---------------- Helper: createBufferFromArray ----------------
    static createBufferFromArray(device, array, usage) {
      const buffer = device.createBuffer({
        size: array.byteLength,
        usage: usage,
        mappedAtCreation: true,
      });
      if (array instanceof Float32Array) {
        new Float32Array(buffer.getMappedRange()).set(array);
      } else if (array instanceof Uint32Array) {
        new Uint32Array(buffer.getMappedRange()).set(array);
      }
      buffer.unmap();
      return buffer;
    }

    // ---------------- batchedMatMul  ----------------
    // Computes C = A * B. When flagTranspose==1, B is read as transposed.
    async batchedMatMul(A, B, aRows, k, bCols, flagTranspose) {
      const outBufferSize = aRows * bCols * 4;
      const C = this.device.createBuffer({
        size: outBufferSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });
      const dimsArray = new Uint32Array([aRows, k, bCols, flagTranspose]);
      const dimsBuffer = GPUAttention.createBufferFromArray(
        this.device,
        dimsArray,
        GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
      );
      const bindGroup = this.device.createBindGroup({
        layout: this.batchedMatMulPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: A } },
          { binding: 1, resource: { buffer: B } },
          { binding: 2, resource: { buffer: C } },
          { binding: 3, resource: { buffer: dimsBuffer } },
        ],
      });
      const commandEncoder = this.device.createCommandEncoder();
      const pass = commandEncoder.beginComputePass();
      pass.setPipeline(this.batchedMatMulPipeline);
      pass.setBindGroup(0, bindGroup);
      const workgroupCountX = Math.ceil(aRows / 16);
      const workgroupCountY = Math.ceil(bCols / 16);
      pass.dispatchWorkgroups(workgroupCountX, workgroupCountY);
      pass.end();
      this.device.queue.submit([commandEncoder.finish()]);

      await device.queue.onSubmittedWorkDone(); // Synchronization added
      return C;
    }

    // ---------------- scaleMatrix ----------------
    async scaleMatrix(matrix, rows, cols, scalarValue) {
      const totalElements = rows * cols;
      const dimsArray = new Uint32Array([rows, cols]);
      const dimsBuffer = GPUAttention.createBufferFromArray(
        this.device,
        dimsArray,
        GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
      );
      const scalarBuffer = GPUAttention.createBufferFromArray(
        this.device,
        new Float32Array([scalarValue]),
        GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
      );
      const bindGroup = this.device.createBindGroup({
        layout: this.scaleMatrixPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: matrix } },
          { binding: 1, resource: { buffer: scalarBuffer } },
          { binding: 2, resource: { buffer: dimsBuffer } },
        ],
      });
      const commandEncoder = this.device.createCommandEncoder();
      const pass = commandEncoder.beginComputePass();
      pass.setPipeline(this.scaleMatrixPipeline);
      pass.setBindGroup(0, bindGroup);
      const workgroupCount = Math.ceil(totalElements / 64);
      pass.dispatchWorkgroups(workgroupCount);
      pass.end();
      this.device.queue.submit([commandEncoder.finish()]);

      await device.queue.onSubmittedWorkDone(); // Synchronization added
      return matrix;
    }

    // ---------------- softmax ----------------
    // Applies row-wise softmax to an input matrix.
    async softmax(matrix, rows, cols) {
      const outBufferSize = rows * cols * 4;
      const outBuffer = this.device.createBuffer({
        size: outBufferSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });
      const dimsArray = new Uint32Array([rows, cols]);
      const dimsBuffer = GPUAttention.createBufferFromArray(
        this.device,
        dimsArray,
        GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
      );
      const bindGroup = this.device.createBindGroup({
        layout: this.softmaxPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: matrix } },
          { binding: 1, resource: { buffer: outBuffer } },
          { binding: 2, resource: { buffer: dimsBuffer } },
        ],
      });
      const commandEncoder = this.device.createCommandEncoder();
      const pass = commandEncoder.beginComputePass();
      pass.setPipeline(this.softmaxPipeline);
      pass.setBindGroup(0, bindGroup);
      // One workgroup per row.
      pass.dispatchWorkgroups(rows);
      pass.end();
      this.device.queue.submit([commandEncoder.finish()]);

      await device.queue.onSubmittedWorkDone(); // Synchronization added
      return outBuffer;
    }

    // ---------------- attention ----------------
    // Given Q, K, and V buffers, this computes:
    //   scores = Q * Kᵀ, scaled by 1/sqrt(dimHead),
    //   probs = softmax(scores),
    //   out = probs * V.
    // Q, K, V are assumed to have shape [nTokens, dimHead].
    async attention(Q, K, V, nTokens, dimHead) {
      // 1. Compute attention scores = Q * Kᵀ.
      const scoresBuffer = await this.batchedMatMul(
        Q,
        K,
        nTokens,
        dimHead,
        nTokens,
        1 // interpret K as transposed.
      );

      // 2. Scale scores by 1/sqrt(dimHead).
      const scaleFactor = 1.0 / Math.sqrt(dimHead);
      await this.scaleMatrix(scoresBuffer, nTokens, nTokens, scaleFactor);

      // 3. Apply row-wise softmax.
      const softmaxBuffer = await this.softmax(scoresBuffer, nTokens, nTokens);

      // 4. Multiply softmax probabilities times V.
      const attnOutput = await this.batchedMatMul(
        softmaxBuffer,
        V,
        nTokens,
        nTokens,
        dimHead,
        0
      );
      return attnOutput;
    }

    // ---------------- forward ----------------
    // Given an input x (GPUBuffer with shape [nTokens, dim]),
    // first project x to Q, K, and V using the projection matrices,
    // then run the attention forward pass and finally project out.
    async forward(x, nTokens) {
      // Each projection is computed as: Projection = x * W,
      // where now W has shape [dim, dimHead].
      const Q = await this.batchedMatMul(
        x,
        this.qProj,
        nTokens,
        this.dim,
        this.dimHead,
        0
      );
      const K = await this.batchedMatMul(
        x,
        this.kProj,
        nTokens,
        this.dim,
        this.dimHead,
        0
      );
      const V = await this.batchedMatMul(
        x,
        this.vProj,
        nTokens,
        this.dim,
        this.dimHead,
        0
      );

      // Run attention: scores = Q * Kᵀ, then weighted sum with V.
      // Note: scaling is now based on dimHead.
      const attnBuffer = await this.attention(Q, K, V, nTokens, this.dimHead);

      // Finally, project the attention output (shape [nTokens, dimHead])
      // back to the original embedding dimension [nTokens, dim].
      const outputBuffer = await this.batchedMatMul(
        attnBuffer,
        this.outProj,
        nTokens,
        this.dimHead,
        this.dim,
        0
      );
      return outputBuffer;
    }
  }
  // GPUFeedForward – a simple two‐layer MLP skeleton.
  class GPUFeedForward {
    constructor(device, dim, mult = 4) {
      this.device = device;
      this.dim = dim;
      this.mult = mult;
      // Linear layer 1: expand dimension from dim to dim*mult.
      this.linear1 = new GPULinear(device, dim, dim * mult);
      // Linear layer 2: project from dim*mult back to dim.
      this.linear2 = new GPULinear(device, dim * mult, dim);

      // WGSL shader code for GeLU activation.
      // This shader reads an input buffer and writes into an output buffer
      // the GeLU activation computed elementwise.
      const geluShaderCode = `
  struct Uniforms {
    totalElements : u32,
  };
  
  @group(0) @binding(0) var<storage, read> input : array<f32>;
  @group(0) @binding(1) var<storage, read_write> output : array<f32>;
  @group(0) @binding(2) var<uniform> uniforms : Uniforms;
  
  // A simple tanh approximation using the definition: tanh(x) = (exp(2x)-1)/(exp(2x)+1)
  fn myTanh(x: f32) -> f32 {
    let exp2x = exp(2.0 * x);
    return (exp2x - 1.0) / (exp2x + 1.0);
  }
  
  @compute @workgroup_size(64)
  fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let idx : u32 = global_id.x;
    if (idx >= uniforms.totalElements) {
      return;
    }
    let a : f32 = input[idx];
    // Compute a^3
    let a3 : f32 = a * a * a;
    // constant: sqrt(2/pi)
    let sqrt2overPi : f32 = sqrt(2.0 / 3.141592653589793);
    let inner : f32 = sqrt2overPi * (a + 0.044715 * a3);
    let tanh_val : f32 = myTanh(inner);
    output[idx] = 0.5 * a * (1.0 + tanh_val);
  }
  `;

      // Create the WGSL module and pipeline for GeLU.
      this.geluModule = device.createShaderModule({ code: geluShaderCode });
      this.geluPipeline = device.createComputePipeline({
        layout: "auto",
        compute: {
          module: this.geluModule,
          entryPoint: "main",
        },
      });
    }

    // Helper: apply the GeLU activation on an input buffer.
    // numElements is the total number of float elements to process.
    async applyGelu(inputBuffer, numElements) {
      // Allocate an output buffer (same size as input).
      const outputBuffer = this.device.createBuffer({
        size: numElements * 4, // each f32 takes 4 bytes.
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });
      // Create a uniform buffer holding the number of elements.
      const uniformArray = new Uint32Array([numElements]);
      const uniformBuffer = createBufferFromArray(
        this.device,
        uniformArray,
        GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
      );
      // Create a bind group for the GeLU shader.
      const bindGroup = this.device.createBindGroup({
        layout: this.geluPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: inputBuffer } },
          { binding: 1, resource: { buffer: outputBuffer } },
          { binding: 2, resource: { buffer: uniformBuffer } },
        ],
      });
      // Launch the compute pass (using workgroup size 64).
      const commandEncoder = this.device.createCommandEncoder();
      const pass = commandEncoder.beginComputePass();
      pass.setPipeline(this.geluPipeline);
      pass.setBindGroup(0, bindGroup);
      const workgroupCount = Math.ceil(numElements / 64);
      pass.dispatchWorkgroups(workgroupCount);
      pass.end();
      this.device.queue.submit([commandEncoder.finish()]);

      await device.queue.onSubmittedWorkDone(); // Synchronization added
      return outputBuffer;
    }

    // The forward pass now includes three steps:
    // 1. Compute hidden activations: linear1(x)
    // 2. Apply GeLU activation elementwise on the result.
    // 3. Compute output: linear2(GeLU(linear1(x)))
    async forward(xBuffer, batchTimesSeq) {
      // xBuffer is assumed to have shape: [batchTimesSeq, dim].
      // linear1 transforms from [batchTimesSeq, dim] to [batchTimesSeq, dim*mult]
      const hidden = await this.linear1.forward(xBuffer, [
        batchTimesSeq,
        this.dim,
      ]);
      const numHiddenElements = batchTimesSeq * (this.dim * this.mult);
      // Apply GeLU activation elementwise.
      const activatedHidden = await this.applyGelu(hidden, numHiddenElements);
      // linear2 transforms from [batchTimesSeq, dim*mult] back to [batchTimesSeq, dim].
      const output = await this.linear2.forward(activatedHidden, [
        batchTimesSeq,
        this.dim * this.mult,
      ]);
      return output;
    }
  }

  // Helper: gpuElementWiseAdd – adds two buffers element‐wise.
  async function gpuElementWiseAdd(bufferA, bufferB, numElements) {
    const addShaderCode = `
  @group(0) @binding(0) var<storage, read> A : array<f32>;
  @group(0) @binding(1) var<storage, read> B : array<f32>;
  @group(0) @binding(2) var<storage, read_write> C : array<f32>;
  
  @compute @workgroup_size(64)
  fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i: u32 = id.x;
    if (i < ${numElements}u) {
      C[i] = A[i] + B[i];
    }
  }
  `;
    const addModule = device.createShaderModule({ code: addShaderCode });
    const addPipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module: addModule, entryPoint: "main" },
    });
    const outputBuffer = device.createBuffer({
      size: numElements * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const bindGroup = device.createBindGroup({
      layout: addPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: bufferA } },
        { binding: 1, resource: { buffer: bufferB } },
        { binding: 2, resource: { buffer: outputBuffer } },
      ],
    });
    const commandEncoder = device.createCommandEncoder();
    const pass = commandEncoder.beginComputePass();
    pass.setPipeline(addPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(numElements / 64));
    pass.end();
    device.queue.submit([commandEncoder.finish()]);
    return outputBuffer;
  }

  // GPUTransformerBlock – one transformer block using AdaLN, attention, feed-forward.
  class GPUTransformerBlock {
    constructor(device, dim, timeEmbDim, dimHead, mlpMult) {
      this.device = device;
      this.dim = dim;
      this.timeEmbDim = timeEmbDim;
      this.attn = new GPUAttention(device, dim, dimHead);
      this.norm1 = new GPUAdaLN(device, dim, timeEmbDim);
      this.norm2 = new GPUAdaLN(device, dim, timeEmbDim);
      this.ff = new GPUFeedForward(device, dim, mlpMult);
    }
    async forward(xBuffer, tBuffer, numElements) {
      // Compute the number of tokens (rows) from numElements and the hidden dimension.
      let nTokens = numElements / this.dim; // First branch: norm then attention followed by residual-add.
      const norm1Out = await this.norm1.forward(xBuffer, tBuffer, numElements);
      const attnOut = await this.attn.forward(norm1Out, nTokens);
      const res1 = await gpuElementWiseAdd(xBuffer, attnOut, numElements); // Second branch: norm then feed‑forward followed by residual-add.
      const norm2Out = await this.norm2.forward(res1, tBuffer, numElements);
      const ffOut = await this.ff.forward(norm2Out, nTokens);
      const res2 = await gpuElementWiseAdd(res1, ffOut, numElements);
      return res2;
    }
  }

  // GPUDiT – the main model class.
  /* Updated GPUDiT forward pass
   (Note that aside from patch embedding, time embedding, transformer blocks,
    and final linear “toPixels”, everything is now wrapped by GPU patchify/unpatchify operations.)
*/
  // class GPUDiT {
  //   constructor(device, config) {
  //     this.device = device;
  //     this.inputSize = config.inputSize; // e.g. 32
  //     this.patchSize = config.patchSize; // e.g. 2
  //     this.inChannels = config.inChannels; // e.g. 3
  //     this.dim = config.dim;
  //     this.depth = config.depth;
  //     this.dimHead = config.dimHead;
  //     this.mlpMult = config.mlpMult;
  //     this.timeEmbDim = config.timeEmbDim;

  //     // Patch embedding: from a patch (flattened pixels) to model dim.
  //     const patchDim = this.inChannels * this.patchSize * this.patchSize;
  //     this.patchEmbedding = new GPULinear(device, patchDim, this.dim);

  //     // Time MLP: first through a sinusoidal embedding (computed on CPU)
  //     // then two linear layers.
  //     this.timeMLP1 = new GPULinear(
  //       device,
  //       Math.floor(this.dim / 4),
  //       this.timeEmbDim
  //     );
  //     this.timeMLP2 = new GPULinear(device, this.timeEmbDim, this.timeEmbDim);

  //     // Create transformer blocks.
  //     this.blocks = [];
  //     for (let i = 0; i < this.depth; i++) {
  //       this.blocks.push(
  //         new GPUTransformerBlock(
  //           device,
  //           this.dim,
  //           this.timeEmbDim,
  //           this.dimHead,
  //           this.mlpMult
  //         )
  //       );
  //     }
  //     // Final normalization and output projection.
  //     this.finalNorm = new GPUAdaLN(device, this.dim, this.timeEmbDim);
  //     this.toPixels = new GPULinear(device, this.dim, patchDim);
  //   }

  //   // CPU-based sinusoidal positional embedding.
  //   sinusoidalPosEmb(x, embDim) {
  //     const halfDim = Math.floor(embDim / 2);
  //     const log10000 = Math.log(10000);
  //     const emb = new Float32Array(halfDim);
  //     for (let i = 0; i < halfDim; i++) {
  //       emb[i] = Math.exp(-(i * log10000) / (halfDim - 1));
  //     }
  //     const out = new Float32Array(embDim);
  //     for (let i = 0; i < halfDim; i++) {
  //       const angle = x * emb[i];
  //       out[i] = Math.sin(angle);
  //       out[i + halfDim] = Math.cos(angle);
  //     }
  //     return out;
  //   }

  //   /* New forward pass:
  //      1. Copy the input image (Float32Array of shape [channels, inputSize, inputSize])
  //         to a GPU buffer.
  //      2. Use the new patchify shader to convert the image into a “patch tensor”
  //         (shape: [numPatches, patchDim]).
  //      3. Apply the patchEmbedding layer followed by time embedding and transformer blocks.
  //      4. After the final linear layer (toPixels), run the unpatchify shader to reconstruct
  //         the image (shape: [channels, inputSize, inputSize]).
  //   */
  //   async forward(inputImage, timeVal) {
  //     // Create a GPUBuffer for the input image
  //     const imageBuffer = createBufferFromArray(
  //       this.device,
  //       inputImage,
  //       GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
  //     );

  //     // Run patchify
  //     const patchifyParams = {
  //       imageWidth: this.inputSize,
  //       imageHeight: this.inputSize,
  //       channels: this.inChannels,
  //       patchSize: this.patchSize,
  //     };
  //     const patchesBuffer = await patchifyImage(
  //       this.device,
  //       imageBuffer,
  //       patchifyParams
  //     );

  //     // Compute dimensions
  //     const numPatchesSide = this.inputSize / this.patchSize;
  //     const numPatches = numPatchesSide * numPatchesSide;
  //     const patchDim = this.inChannels * this.patchSize * this.patchSize;

  //     // Patch embedding
  //     let x = await this.patchEmbedding.forward(patchesBuffer, [
  //       numPatches,
  //       patchDim,
  //     ]);

  //     // Time embedding
  //     const sinEmb = this.sinusoidalPosEmb(timeVal, Math.floor(this.dim / 4));
  //     const sinEmbBuffer = createBufferFromArray(
  //       this.device,
  //       sinEmb,
  //       GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
  //     );
  //     const t1 = await this.timeMLP1.forward(sinEmbBuffer, [
  //       1,
  //       Math.floor(this.dim / 4),
  //     ]);
  //     const tEmbedding = await this.timeMLP2.forward(t1, [1, this.timeEmbDim]);

  //     // Process through transformer blocks without logging
  //     const totalElements = numPatches * this.dim;
  //     for (const block of this.blocks) {
  //       x = await block.forward(x, tEmbedding, totalElements);
  //     }

  //     // Final normalization and output projection
  //     const normed = await this.finalNorm.forward(x, tEmbedding, totalElements);
  //     const projected = await this.toPixels.forward(normed, [
  //       numPatches,
  //       this.dim,
  //     ]);

  //     // Unpatchify back to image
  //     const outputBuffer = await unpatchifyPatches(
  //       this.device,
  //       projected,
  //       patchifyParams
  //     );

  //     return outputBuffer;
  //   }
  // }

  // Example usage:
  class GPUDiT {
    constructor(device, config) {
      this.device = device;
      this.inputSize = config.inputSize;
      this.patchSize = config.patchSize;
      this.inChannels = config.inChannels;
      this.dim = config.dim;
      this.depth = config.depth;
      this.dimHead = config.dimHead;
      this.mlpMult = config.mlpMult;
      this.timeEmbDim = config.timeEmbDim;

      // Calculate grid dimensions
      this.gridSize = this.inputSize / this.patchSize;
      this.numPatches = this.gridSize * this.gridSize;
      this.patchDim = this.inChannels * this.patchSize * this.patchSize;

      // Patch embedding
      this.patchEmbedding = new GPULinear(device, this.patchDim, this.dim);

      // Precompute 2D positional embeddings
      this.posEmbBuffer = this.create2DPositionalEmbeddings();

      // Rest of the initialization remains the same...
      this.timeMLP1 = new GPULinear(
        device,
        Math.floor(this.dim / 2),
        this.timeEmbDim
      );
      this.timeMLP2 = new GPULinear(device, this.timeEmbDim, this.timeEmbDim);

      this.blocks = [];
      for (let i = 0; i < this.depth; i++) {
        this.blocks.push(
          new GPUTransformerBlock(
            device,
            this.dim,
            this.timeEmbDim,
            this.dimHead,
            this.mlpMult
          )
        );
      }

      this.finalNorm = new GPUAdaLN(device, this.dim, this.timeEmbDim);
      this.toPixels = new GPULinear(device, this.dim, this.patchDim);
    }

    create2DPositionalEmbeddings() {
      // This matches PyTorch's Fixed2DPosEmb implementation

      // Create flattened position indices for the grid
      const positions = new Array(this.gridSize * this.gridSize);
      for (let h = 0; h < this.gridSize; h++) {
        for (let w = 0; w < this.gridSize; w++) {
          const idx = h * this.gridSize + w;
          positions[idx] = { h, w };
        }
      }

      // Create embeddings array (matches PyTorch's pos_emb shape)
      const posEmb = new Float32Array(this.numPatches * this.dim);

      // For each position in the grid
      positions.forEach(({ h, w }, patchIdx) => {
        // Calculate height embeddings (dim/2)
        const hEmb = this.sinusoidalPosEmb(h, this.dim / 2);

        // Calculate width embeddings (dim/2)
        const wEmb = this.sinusoidalPosEmb(w, this.dim / 2);

        // Combine embeddings for this position (matches PyTorch's torch.cat([emb_h, emb_w], dim=-1))
        const offset = patchIdx * this.dim;
        for (let i = 0; i < this.dim / 2; i++) {
          // First half: height embeddings
          posEmb[offset + i] = hEmb[i];
          // Second half: width embeddings
          posEmb[offset + i + this.dim / 2] = wEmb[i];
        }
      });

      // Create and initialize GPU buffer with position embeddings
      const buffer = this.device.createBuffer({
        size: posEmb.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
      });
      new Float32Array(buffer.getMappedRange()).set(posEmb);
      buffer.unmap();

      return buffer;
    }

    // Helper function to compute sinusoidal embeddings
    // This matches PyTorch's SinusoidalPosEmb implementation
    sinusoidalPosEmb(x, embDim) {
      const halfDim = Math.floor(embDim);
      const emb = new Float32Array(halfDim);
      const log10000 = Math.log(10000);

      // Compute embeddings
      for (let i = 0; i < halfDim; i++) {
        emb[i] = Math.exp(-(i * log10000) / (halfDim - 1));
      }

      // Compute final embeddings (sin and cos)
      const out = new Float32Array(embDim);
      for (let i = 0; i < halfDim; i++) {
        const angle = x * emb[i];
        out[i] = Math.sin(angle);
        out[i + halfDim] = Math.cos(angle);
      }

      return out;
    }

    async forward(inputImage, timeVal) {
      // Create GPU buffer for input image
      const imageBuffer = createBufferFromArray(
        this.device,
        inputImage,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
      );

      // Patchify
      const patchifyParams = {
        imageWidth: this.inputSize,
        imageHeight: this.inputSize,
        channels: this.inChannels,
        patchSize: this.patchSize,
      };
      const patchesBuffer = await patchifyImage(
        this.device,
        imageBuffer,
        patchifyParams
      );

      // Patch embedding
      let x = await this.patchEmbedding.forward(patchesBuffer, [
        this.numPatches,
        this.patchDim,
      ]);

      // Add positional embeddings
      x = await gpuElementWiseAdd(
        x,
        this.posEmbBuffer,
        this.numPatches * this.dim
      );

      // Time embedding
      const sinEmb = this.sinusoidalPosEmb(timeVal, Math.floor(this.dim / 4));
      const sinEmbBuffer = createBufferFromArray(
        this.device,
        sinEmb,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
      );
      const t1 = await this.timeMLP1.forward(sinEmbBuffer, [
        1,
        Math.floor(this.dim / 4),
      ]);
      const tEmbedding = await this.timeMLP2.forward(t1, [1, this.timeEmbDim]);

      // Process through transformer blocks
      const totalElements = this.numPatches * this.dim;
      for (const block of this.blocks) {
        x = await block.forward(x, tEmbedding, totalElements);
      }

      // Final normalization and projection
      const normed = await this.finalNorm.forward(x, tEmbedding, totalElements);
      const projected = await this.toPixels.forward(normed, [
        this.numPatches,
        this.dim,
      ]);

      // Unpatchify back to image
      const outputBuffer = await unpatchifyPatches(
        this.device,
        projected,
        patchifyParams
      );

      return outputBuffer;
    }

    async loadWeights(weightsURL) {
      // Helper: recursively compute the shape of a nested array.
      function getShape(arr) {
        if (!Array.isArray(arr)) return [];
        const shape = [arr.length];
        if (arr.length > 0 && Array.isArray(arr[0])) {
          return shape.concat(getShape(arr[0]));
        }
        return shape;
      }

      // Helper: product of numbers in an array.
      const product = (arr) => arr.reduce((a, b) => a * b, 1);

      // Fetch and parse the weights JSON.
      const response = await fetch(weightsURL);
      const weightsData = await response.json();

      // Build list of expected keys.
      const expectedKeys = [];
      expectedKeys.push("patch_embed.weight");
      expectedKeys.push("time_mlp.1.weight");
      expectedKeys.push("time_mlp.3.weight");
      for (let i = 0; i < this.depth; i++) {
        expectedKeys.push(`block_${i}.attn.to_q.weight`);
        expectedKeys.push(`block_${i}.attn.to_k.weight`);
        expectedKeys.push(`block_${i}.attn.to_v.weight`);
        expectedKeys.push(`block_${i}.attn.to_out.weight`);
        expectedKeys.push(`block_${i}.ff.net.0.weight`);
        expectedKeys.push(`block_${i}.ff.net.2.weight`);
        expectedKeys.push(`block_${i}.norm1.scale.weight`);
        expectedKeys.push(`block_${i}.norm1.shift.weight`);
        expectedKeys.push(`block_${i}.norm2.scale.weight`);
        expectedKeys.push(`block_${i}.norm2.shift.weight`);
      }
      expectedKeys.push("final_norm.scale.weight");
      expectedKeys.push("final_norm.shift.weight");
      expectedKeys.push("to_pixels.weight");

      // Alert if there are extra keys in the JSON.
      for (const key in weightsData) {
        if (!expectedKeys.includes(key)) {
          console.warn(
            `Warning: Unexpected key "${key}" found in weights file.`
          );
        }
      }
      // Alert if any expected key is missing.
      for (const key of expectedKeys) {
        if (!(key in weightsData)) {
          console.warn(
            `Warning: Expected weight "${key}" is missing from the weights file.`
          );
        }
      }

      const device = this.device;

      // Helper: process one weight –
      // It prints the JSON shape (before flattening) and the expected shape,
      // then flattens the data and writes it into the given GPUBuffer.
      function processWeight(key, weightBuffer, expectedShapeArray) {
        if (!(key in weightsData)) return;
        const raw = weightsData[key];
        const rawShape = getShape(raw);
        const flat = flattenWeight(raw);
        console.log(
          `Loading weight "${key}": JSON shape = [${rawShape.join(", ")}], ` +
            `expected shape = [${expectedShapeArray.join(", ")}]. ` +
            `Flattened length = ${
              flat.length
            }, expected total elements = ${product(expectedShapeArray)}.`
        );
        device.queue.writeBuffer(weightBuffer, 0, flat);
      }

      // 1. Load patch embedding weights.
      // Expected shape for a GPULinear is [outFeatures, inFeatures].
      processWeight("patch_embed.weight", this.patchEmbedding.weightBuffer, [
        this.patchEmbedding.outFeatures,
        this.patchEmbedding.inFeatures,
      ]);

      // 2. Load time MLP weights.
      processWeight("time_mlp.1.weight", this.timeMLP1.weightBuffer, [
        this.timeMLP1.outFeatures,
        this.timeMLP1.inFeatures,
      ]);
      processWeight("time_mlp.3.weight", this.timeMLP2.weightBuffer, [
        this.timeMLP2.outFeatures,
        this.timeMLP2.inFeatures,
      ]);

      // 3. Load Transformer block weights.
      for (let i = 0; i < this.depth; i++) {
        const block = this.blocks[i];
        const prefix = `block_${i}.`;

        // Attention projections.
        processWeight(`${prefix}attn.to_q.weight`, block.attn.qProj, [
          block.attn.dim,
          block.attn.dimHead,
        ]);
        processWeight(`${prefix}attn.to_k.weight`, block.attn.kProj, [
          block.attn.dim,
          block.attn.dimHead,
        ]);
        processWeight(`${prefix}attn.to_v.weight`, block.attn.vProj, [
          block.attn.dim,
          block.attn.dimHead,
        ]);
        processWeight(`${prefix}attn.to_out.weight`, block.attn.outProj, [
          block.attn.dimHead,
          block.attn.dim,
        ]);

        // Feed-forward linear layers.
        processWeight(
          `${prefix}ff.net.0.weight`,
          block.ff.linear1.weightBuffer,
          [block.ff.linear1.outFeatures, block.ff.linear1.inFeatures]
        );
        processWeight(
          `${prefix}ff.net.2.weight`,
          block.ff.linear2.weightBuffer,
          [block.ff.linear2.outFeatures, block.ff.linear2.inFeatures]
        );

        // Adaptive layer norms.
        processWeight(
          `${prefix}norm1.scale.weight`,
          block.norm1.scaleProj.weightBuffer,
          [block.norm1.dim, block.norm1.timeEmbDim]
        );
        processWeight(
          `${prefix}norm1.shift.weight`,
          block.norm1.shiftProj.weightBuffer,
          [block.norm1.dim, block.norm1.timeEmbDim]
        );
        processWeight(
          `${prefix}norm2.scale.weight`,
          block.norm2.scaleProj.weightBuffer,
          [block.norm2.dim, block.norm2.timeEmbDim]
        );
        processWeight(
          `${prefix}norm2.shift.weight`,
          block.norm2.shiftProj.weightBuffer,
          [block.norm2.dim, block.norm2.timeEmbDim]
        );
      }

      // 4. Load final normalization and output projection.
      processWeight(
        "final_norm.scale.weight",
        this.finalNorm.scaleProj.weightBuffer,
        [this.finalNorm.dim, this.finalNorm.timeEmbDim]
      );
      processWeight(
        "final_norm.shift.weight",
        this.finalNorm.shiftProj.weightBuffer,
        [this.finalNorm.dim, this.finalNorm.timeEmbDim]
      );
      processWeight("to_pixels.weight", this.toPixels.weightBuffer, [
        this.toPixels.outFeatures,
        this.toPixels.inFeatures,
      ]);

      console.log("Weights loaded successfully.");
    }
  }
  const config = {
    // Image dimensions
    inputSize: 64, // 256x256 image (standard size for many image models)
    patchSize: 4, // 4x4 patches -> 64x64 grid of patches
    inChannels: 3, // RGB image

    // Model architecture
    dim: 64, // Hidden dimension (similar to ViT-Base)
    depth: 6, // Number of transformer blocks
    dimHead: 32, // Dimension per attention head (768/12 = 64)
    mlpMult: 4, // MLP expansion factor (standard in transformers)
    timeEmbDim: 128, // Time embedding dimension (usually same as model dim)
  };

  const model = new GPUDiT(device, config);
  console.log("DiT model created.");
  console.log("Loading weights...");
  await model.loadWeights("matrices.json");
  await device.queue.onSubmittedWorkDone();

  const imageSize = config.inputSize;
  const numImageElements = config.inChannels * imageSize * imageSize;
  // Create a dummy input image (random data)
  const inputImage = new Float32Array(numImageElements);
  for (let i = 0; i < numImageElements; i++) {
    inputImage[i] = 1;
  }
  // Create a dummy time value.
  const timeVal = 1;
  // Run the forward pass.
  const outputBuffer = await model.forward(inputImage, timeVal);
  // (A complete implementation would map outputBuffer back to CPU memory.)
  console.log("Forward passing doing...");
  await device.queue.onSubmittedWorkDone();
  console.log("DiT model forward pass executed.");

  // Create a staging buffer to copy the data back to CPU
  const stagingBuffer = device.createBuffer({
    size: numImageElements * 4, // Float32 is 4 bytes
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  // Create and submit command encoder for the copy operation
  const commandEncoder = device.createCommandEncoder();
  commandEncoder.copyBufferToBuffer(
    outputBuffer, // source buffer
    0, // source offset
    stagingBuffer, // destination buffer
    0, // destination offset
    numImageElements * 4 // size to copy (in bytes)
  );
  device.queue.submit([commandEncoder.finish()]);

  // Map the staging buffer to CPU memory
  await stagingBuffer.mapAsync(GPUMapMode.READ);

  // Get the mapped array and create a Float32Array view of it
  const mappedArray = new Float32Array(stagingBuffer.getMappedRange());

  // Log the first few values and basic statistics
  console.log("First 10 values of the output:", mappedArray.slice(0, 10));

  // Calculate some basic statistics
  const min = Math.min(...mappedArray);
  const max = Math.max(...mappedArray);
  const mean = mappedArray.reduce((a, b) => a + b) / mappedArray.length;

  console.log("Output buffer statistics:");
  console.log("- Size:", mappedArray.length);
  console.log("- Min value:", min);
  console.log("- Max value:", max);
  console.log("- Mean value:", mean);

  // Unmap the buffer when done
  stagingBuffer.unmap();

  async function computeStatsFromBuffer(buffer, numElements) {
    // Create a staging buffer so we can map the GPU results to CPU memory.
    const stagingBuffer = device.createBuffer({
      size: numElements * 4, // each f32 is 4 bytes
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const commandEncoder = device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(
      buffer,
      0,
      stagingBuffer,
      0,
      numElements * 4
    );
    device.queue.submit([commandEncoder.finish()]);
    // Wait for the copy to complete.
    await device.queue.onSubmittedWorkDone();
    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const array = new Float32Array(stagingBuffer.getMappedRange());
    const min = Math.min(...array);
    const max = Math.max(...array);
    const mean = array.reduce((sum, v) => sum + v, 0) / array.length;
    stagingBuffer.unmap();
    return { mean, min, max };
  }

  async function runDoubleForwardComparison() {
    // Run forward pass 1:
    console.log("Running forward pass 1...");
    const outputBuffer1 = await model.forward(inputImage, timeVal);
    await device.queue.onSubmittedWorkDone();
    const stats1 = await computeStatsFromBuffer(
      outputBuffer1,
      numImageElements
    );
    console.log(
      "Forward pass 1: Mean =",
      stats1.mean,
      "Min =",
      stats1.min,
      "Max =",
      stats1.max
    );

    // Run forward pass 2:
    console.log("Running forward pass 2...");
    const outputBuffer2 = await model.forward(inputImage, timeVal);
    await device.queue.onSubmittedWorkDone();
    const stats2 = await computeStatsFromBuffer(
      outputBuffer2,
      numImageElements
    );
    console.log(
      "Forward pass 2: Mean =",
      stats2.mean,
      "Min =",
      stats2.min,
      "Max =",
      stats2.max
    );

    // Optionally, log the differences
    console.log("Difference in means:", Math.abs(stats1.mean - stats2.mean));
    console.log("Difference in mins:", Math.abs(stats1.min - stats2.min));
    console.log("Difference in maxes:", Math.abs(stats1.max - stats2.max));
  }

  // Call the comparison function:
  runDoubleForwardComparison();
})();
