/* dit_webgpu.js */
const ELEMENT_WISE_ADD_SHADER_CODE = (numElements) =>
  ADD_SHADER_CODE.replace(/_NUM_ELEMENTS_/g, numElements);

// GPUPatchify class
class GPUPatchify {
  constructor(device) {
    this.device = device;
    this.pipeline = null; // Pipeline will be created on demand
  }

  async createPipeline() {
    if (!this.pipeline) {
      const shaderModule = this.device.createShaderModule({
        code: PATCHIFY_SHADER_CODE,
      });
      this.pipeline = this.device.createComputePipeline({
        layout: "auto",
        compute: {
          module: shaderModule,
          entryPoint: "main",
        },
      });
    }
  }

  async patchifyImage(imageBuffer, params) {
    await this.createPipeline();

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
    const uniformBuffer = this.device.createBuffer({
      size: uniformArray.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(uniformBuffer, 0, uniformArray);

    // Create output buffer
    const patchBuffer = this.device.createBuffer({
      size: totalElements * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    // Create bind group
    const bindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: imageBuffer } },
        { binding: 1, resource: { buffer: uniformBuffer } },
        { binding: 2, resource: { buffer: patchBuffer } },
      ],
    });

    // Execute compute pass
    const commandEncoder = this.device.createCommandEncoder();
    const pass = commandEncoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(totalElements / 64));
    pass.end();

    this.device.queue.submit([commandEncoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();

    return patchBuffer;
  }
}

// GPUUnpatchify class
class GPUUnpatchify {
  constructor(device) {
    this.device = device;
    this.pipeline = null; // Pipeline will be created on demand
  }

  async createPipeline() {
    if (!this.pipeline) {
      const shaderModule = this.device.createShaderModule({
        code: UNPATCHIFY_SHADER_CODE,
      });
      this.pipeline = this.device.createComputePipeline({
        layout: "auto",
        compute: {
          module: shaderModule,
          entryPoint: "main",
        },
      });
    }
  }

  async unpatchifyPatches(patchesBuffer, params) {
    await this.createPipeline();

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
    const uniformBuffer = this.device.createBuffer({
      size: uniformArray.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(uniformBuffer, 0, uniformArray);

    // Create output buffer
    const imageBuffer = this.device.createBuffer({
      size: params.channels * params.imageWidth * params.imageHeight * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    // Create bind group
    const bindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: patchesBuffer } },
        { binding: 1, resource: { buffer: uniformBuffer } },
        { binding: 2, resource: { buffer: imageBuffer } },
      ],
    });

    // Execute compute pass
    const commandEncoder = this.device.createCommandEncoder();
    const pass = commandEncoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(totalElements / 64));
    pass.end();

    this.device.queue.submit([commandEncoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();

    return imageBuffer;
  }
}

// GPUMatrixMultiply class
class GPUMatrixMultiply {
  constructor(device) {
    this.device = device;
    this.pipeline = null; // Pipeline will be created on demand
  }

  async createPipeline() {
    if (!this.pipeline) {
      const shaderModule = this.device.createShaderModule({
        code: MATMUL_SHADER_CODE,
      });

      this.pipeline = this.device.createComputePipeline({
        layout: "auto",
        compute: {
          module: shaderModule,
          entryPoint: "main",
        },
      });
    }
  }

  async matrixMultiply(
    inputBuffer,
    weightBuffer,
    inputShape /*[m, k]*/,
    outFeatures
  ) {
    await this.createPipeline();

    const m = inputShape[0]; // Number of samples
    const k = inputShape[1];
    const n = outFeatures;

    // Create uniform buffer for dimensions
    const dimsArray = new Uint32Array([m, k, n, 0]);
    const dimsBuffer = this.device.createBuffer({
      size: dimsArray.byteLength, // Now 16 bytes.
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
        { binding: 1, resource: { buffer: weightBuffer } },
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

// GPULinear simulates a linear (fully-connected) layer using a matrix multiplication.
class GPULinear {
  constructor(device, inFeatures, outFeatures) {
    this.device = device;
    this.inFeatures = inFeatures;
    this.outFeatures = outFeatures;
    this.mm = new GPUMatrixMultiply(device);

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
  }

  async forward(inputBuffer, inputShape /*[m, k]*/) {
    return await this.mm.matrixMultiply(
      inputBuffer,
      this.weightBuffer,
      inputShape,
      this.outFeatures
    );
  }
}

// GPUAdaLN – a skeleton for adaptive layer normalization.
class GPUAdaLN {
  constructor(device, dim, timeEmbDim) {
    this.device = device;
    this.dim = dim;
    this.timeEmbDim = timeEmbDim;
    this.pipeline = null; // Pipeline will be created on demand

    // Initialize the linear projections for scale and shift using GPULinear
    this.scaleProj = new GPULinear(device, timeEmbDim, dim);
    this.shiftProj = new GPULinear(device, timeEmbDim, dim);
  }

  async createPipeline() {
    if (!this.pipeline) {
      this.adaLNModule = this.device.createShaderModule({
        code: LAYERNORM_SHADER_CODE,
      });
      this.pipeline = this.device.createComputePipeline({
        layout: "auto",
        compute: { module: this.adaLNModule, entryPoint: "main" },
      });
    }
  }

  async forward(xBuffer, timeEmbBuffer, numElements) {
    await this.createPipeline();

    // First, project time embeddings to get scale and shift parameters
    // timeEmbBuffer shape is [1, timeEmbDim]
    const shiftValues = await this.shiftProj.forward(timeEmbBuffer, [
      1,
      this.timeEmbDim,
    ]);

    // 1. Compute the raw scale values
    const scaleValues = await this.scaleProj.forward(timeEmbBuffer, [
      1,
      this.timeEmbDim,
    ]);

    // Determine dimensions for normalization
    const rowWidth = this.dim;
    const nRows = numElements / this.dim;

    // Create uniform buffer for AdaLN
    const uniformData = new Float32Array([nRows, rowWidth, 1e-8]); // epsilon = 1e-5
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
    this.softmaxPipeline = null;
    this.batchedMatMulPipeline = null;
    this.scaleMatrixPipeline = null;

    // Use GPULinear for Q, K, V projections.
    // “inFeatures” is the input embedding dimension (dim)
    // “outFeatures” is the head dimension (dimHead)
    this.qProj = new GPULinear(device, dim, dimHead);
    this.kProj = new GPULinear(device, dim, dimHead);
    this.vProj = new GPULinear(device, dim, dimHead);

    // Initialize output projection: from dimHead back to dim.
    this.outProj = new GPULinear(device, dimHead, dim);
  }

  async createSoftmaxPipeline() {
    if (!this.softmaxPipeline) {
      this.softmaxModule = this.device.createShaderModule({
        code: SOFTMAX_SHADER_CODE,
      });
      this.softmaxPipeline = this.device.createComputePipeline({
        layout: "auto",
        compute: { module: this.softmaxModule, entryPoint: "main" },
      });
    }
  }

  async createBatchedMatMulPipeline() {
    if (!this.batchedMatMulPipeline) {
      this.batchedMatMulModule = this.device.createShaderModule({
        code: BATCHED_MATMUL_SHADER_CODE,
      });
      this.batchedMatMulPipeline = this.device.createComputePipeline({
        layout: "auto",
        compute: { module: this.batchedMatMulModule, entryPoint: "main" },
      });
    }
  }

  async createScaleMatrixPipeline() {
    if (!this.scaleMatrixPipeline) {
      this.scaleMatrixModule = this.device.createShaderModule({
        code: SCALE_SHADER_CODE,
      });
      this.scaleMatrixPipeline = this.device.createComputePipeline({
        layout: "auto",
        compute: { module: this.scaleMatrixModule, entryPoint: "main" },
      });
    }
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
    await this.createBatchedMatMulPipeline();

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

    await this.device.queue.onSubmittedWorkDone(); // Synchronization added
    return C;
  }

  // ---------------- scaleMatrix ----------------
  async scaleMatrix(matrix, rows, cols, scalarValue) {
    await this.createScaleMatrixPipeline();

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

    await this.device.queue.onSubmittedWorkDone(); // Synchronization added
    return matrix;
  }

  // ---------------- softmax ----------------
  // Applies row-wise softmax to an input matrix.
  async softmax(matrix, rows, cols) {
    await this.createSoftmaxPipeline();

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

    await this.device.queue.onSubmittedWorkDone(); // Synchronization added
    return outBuffer;
  }

  // ---------------- attention ----------------
  // Given Q, K, and V buffers, this computes:
  //   scores = Q * Kᵀ, scaled by 1/sqrt(dimHead),
  //   probs = softmax(scores),
  //   out = probs * V.
  // Q, K, V are assumed to have shape [nTokens, dimHead].
  async attention(Q, K, V, nTokens, dimHead) {
    const scoresBuffer = await this.batchedMatMul(
      Q,
      K,
      nTokens,
      dimHead,
      nTokens,
      1 // flagTranspose = 1 so that K is interpreted as transposed.
    );
    const scaleFactor = 1.0 / Math.sqrt(dimHead);
    await this.scaleMatrix(scoresBuffer, nTokens, nTokens, scaleFactor);
    const softmaxBuffer = await this.softmax(scoresBuffer, nTokens, nTokens);
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
    // Instead of manually applying batchedMatMul against raw weight buffers,
    // we now perform the projection using GPULinear forward passes.
    const Q = await this.qProj.forward(x, [nTokens, this.dim]); // results in [nTokens, dimHead]
    const K = await this.kProj.forward(x, [nTokens, this.dim]); // results in [nTokens, dimHead]
    const V = await this.vProj.forward(x, [nTokens, this.dim]); // results in [nTokens, dimHead]

    // Compute attention = softmax( Q * Kᵀ / sqrt(dimHead) ) * V.
    const attnBuffer = await this.attention(Q, K, V, nTokens, this.dimHead);

    // Finally, project back to the original embedding dimension.
    const outputBuffer = await this.outProj.forward(attnBuffer, [
      nTokens,
      this.dimHead,
    ]);
    return outputBuffer;
  }
}
// GPUFeedForward – a simple two‐layer MLP skeleton.
class GPUFeedForward {
  constructor(device, dim, mult = 4) {
    this.device = device;
    this.dim = dim;
    this.mult = mult;
    this.siluPipeline = null;
    // Linear layer 1: expand dimension from dim to dim*mult.
    this.linear1 = new GPULinear(device, dim, dim * mult);
    // Linear layer 2: project from dim*mult back to dim.
    this.linear2 = new GPULinear(device, dim * mult, dim);
  }

  async createSiluPipeline() {
    if (!this.siluPipeline) {
      // Create the WGSL module and pipeline for SiLU.
      this.siluModule = this.device.createShaderModule({
        code: SILU_SHADER_CODE,
      });
      this.siluPipeline = this.device.createComputePipeline({
        layout: "auto",
        compute: {
          module: this.siluModule,
          entryPoint: "main",
        },
      });
    }
  }

  // Helper: apply the Silu activation on an input buffer.
  // numElements is the total number of float elements to process.
  async applySilu(inputBuffer, numElements) {
    await this.createSiluPipeline();
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
    // Create a bind group for the Silu shader.
    const bindGroup = this.device.createBindGroup({
      layout: this.siluPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: outputBuffer } },
        { binding: 2, resource: { buffer: uniformBuffer } },
      ],
    });
    // Launch the compute pass (using workgroup size 64).
    const commandEncoder = this.device.createCommandEncoder();
    const pass = commandEncoder.beginComputePass();
    pass.setPipeline(this.siluPipeline);
    pass.setBindGroup(0, bindGroup);
    const workgroupCount = Math.ceil(numElements / 64);
    pass.dispatchWorkgroups(workgroupCount);
    pass.end();
    this.device.queue.submit([commandEncoder.finish()]);

    await this.device.queue.onSubmittedWorkDone(); // Synchronization added
    return outputBuffer;
  }

  // The forward pass now includes three steps:
  // 1. Compute hidden activations: linear1(x)
  // 2. Apply SiLU activation elementwise on the result.
  // 3. Compute output: linear2(SiLU(linear1(x)))
  async forward(xBuffer, batchTimesSeq) {
    // xBuffer is assumed to have shape: [batchTimesSeq, dim].
    // linear1 transforms from [batchTimesSeq, dim] to [batchTimesSeq, dim*mult]
    logBuffer(this.device, xBuffer, "Input buffer stats");
    const hidden = await this.linear1.forward(xBuffer, [
      batchTimesSeq,
      this.dim,
    ]);

    logBuffer(this.device, hidden, "Hidden buffer stats");
    const numHiddenElements = batchTimesSeq * (this.dim * this.mult);
    // Apply SiLU activation elementwise.
    const activatedHidden = await this.applySilu(hidden, numHiddenElements);
    logBuffer(this.device, activatedHidden, "Activated hidden buffer stats");
    // linear2 transforms from [batchTimesSeq, dim*mult] back to [batchTimesSeq, dim].
    const output = await this.linear2.forward(activatedHidden, [
      batchTimesSeq,
      this.dim * this.mult,
    ]);
    logBuffer(this.device, output, "Output buffer stats");
    return output;
  }
}

// Helper: gpuElementWiseAdd – adds two buffers element‐wise.
async function gpuElementWiseAdd(device, bufferA, bufferB, numElements) {
  const addShaderCode = ELEMENT_WISE_ADD_SHADER_CODE(numElements);
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
    logBuffer(this.device, xBuffer, "BLOCK 1");
    const norm1Out = await this.norm1.forward(xBuffer, tBuffer, numElements);
    logBuffer(this.device, norm1Out, "BLOCK 1 norm(x)");

    const attnOut = await this.attn.forward(norm1Out, nTokens);

    const res1 = await gpuElementWiseAdd(
      this.device,
      xBuffer,
      attnOut,
      numElements
    ); // Second branch: norm then feed‑forward followed by residual-add.

    const norm2Out = await this.norm2.forward(res1, tBuffer, numElements);

    const ffOut = await this.ff.forward(norm2Out, nTokens);

    const res2 = await gpuElementWiseAdd(this.device, res1, ffOut, numElements);
    return res2;
  }
}

// GPUDiT – the main model class.
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
    this.patchify = new GPUPatchify(device);
    this.unpatchify = new GPUUnpatchify(device);

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

  /* Corrected sinusoidalPosEmb: 
   embDim is the full (desired) output length (even number). 
   We split that in half and then fill the first half with sin(x * exp(-...))
   and the second half with cos(x * exp(-...)), just like the PyTorch version. */
  sinusoidalPosEmb(x, embDim) {
    // embDim is the final length, so halfDim is embDim/2 (make sure embDim is even)
    const halfDim = embDim / 2;
    const log10000 = Math.log(10000);
    const emb = new Float32Array(halfDim);

    // Compute scaling factors (similar to torch.exp(torch.arange(half_dim) * -factor))
    for (let i = 0; i < halfDim; i++) {
      emb[i] = Math.exp(-(i * log10000) / (halfDim - 1));
    }

    // Allocate output vector which will contain sin and cos parts:
    const out = new Float32Array(embDim);
    for (let i = 0; i < halfDim; i++) {
      const angle = x * emb[i];
      out[i] = Math.sin(angle);
      out[i + halfDim] = Math.cos(angle);
    }
    return out;
  }

  /* Corrected create2DPositionalEmbeddings:
   This mimics the PyTorch Fixed2DPosEmb.
   For each grid coordinate (h, w) it computes a height embedding and a width embedding,
   each computed by calling sinusoidalPosEmb(x, this.dim/2). Then it concatenates them.
*/
  create2DPositionalEmbeddings() {
    // For a gridSize x gridSize grid
    const numPatches = this.gridSize * this.gridSize;
    // Final positional embedding for each patch has size this.dim.
    const posEmb = new Float32Array(numPatches * this.dim);
    let patchIdx = 0;

    for (let h = 0; h < this.gridSize; h++) {
      for (let w = 0; w < this.gridSize; w++) {
        // For the h and w coordinates, compute the sinusoidal embeddings.
        // Note: we pass this.dim/2 so that each axis embedding has the correct final size.
        const hEmb = this.sinusoidalPosEmb(h, this.dim / 2);
        const wEmb = this.sinusoidalPosEmb(w, this.dim / 2);

        // Offset in the final flattened posEmb
        const offset = patchIdx * this.dim;
        // Concatenate: first half from hEmb, second half from wEmb.
        for (let i = 0; i < this.dim / 2; i++) {
          posEmb[offset + i] = hEmb[i];
          posEmb[offset + i + this.dim / 2] = wEmb[i];
        }
        patchIdx++;
      }
    }

    // Create and initialize the GPU buffer with the computed positional embeddings
    const buffer = this.device.createBuffer({
      size: posEmb.byteLength,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
      mappedAtCreation: true,
    });
    new Float32Array(buffer.getMappedRange()).set(posEmb);
    buffer.unmap();
    return buffer;
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
    const patchesBuffer = await this.patchify.patchifyImage(
      imageBuffer,
      patchifyParams
    );

    // Patch embedding
    let x = await this.patchEmbedding.forward(patchesBuffer, [
      this.numPatches,
      this.patchDim,
    ]);

    // log position embeddings + patch embeddings for debugging
    // Add positional embeddings
    x = await gpuElementWiseAdd(
      this.device,
      x,
      this.posEmbBuffer,
      this.numPatches * this.dim
    );

    // Time embedding
    const sinEmb = this.sinusoidalPosEmb(timeVal, Math.floor(this.dim / 2));
    const sinEmbBuffer = createBufferFromArray(
      this.device,
      sinEmb,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    );

    const t1 = await this.timeMLP1.forward(sinEmbBuffer, [
      1,
      Math.floor(this.dim / 2),
    ]);
    const t1Silu = await this.blocks[0].ff.applySilu(t1, this.timeEmbDim);

    const tEmbedding = await this.timeMLP2.forward(t1Silu, [
      1,
      this.timeEmbDim,
    ]);
    logBuffer(this.device, tEmbedding, "tEmbedding");
    logBuffer(this.device, x, "Patch embeddings");
    // Process through transformer blocks
    const totalElements = this.numPatches * this.dim;
    let i = 0;
    for (const block of this.blocks) {
      i++;
      x = await block.forward(x, tEmbedding, totalElements);
    }

    // Final normalization and projection
    const normed = await this.finalNorm.forward(x, tEmbedding, totalElements);
    const projected = await this.toPixels.forward(normed, [
      this.numPatches,
      this.dim,
    ]);

    // Unpatchify back to image
    const outputBuffer = await this.unpatchify.unpatchifyPatches(
      projected,
      patchifyParams
    );

    logBuffer(this.device, outputBuffer, "outputBuffer");

    return outputBuffer;
  }

  async loadWeights(weightsURL) {
    // Fetch the binary file.
    const response = await fetch(weightsURL);
    const buffer = await response.arrayBuffer();
    const dataView = new DataView(buffer);
    let offset = 0;

    // Verify the magic header (should be "WTS1").
    let magic = "";
    console.log("Reading weights from", weightsURL);
    for (let i = 0; i < 4; i++) {
      magic += String.fromCharCode(dataView.getUint8(offset));
      offset += 1;
    }
    console.log("Magic header:", magic);
    if (magic !== "WTS1") {
      throw new Error("Invalid weights file format");
    }

    // Read the number of weights.
    const numWeights = dataView.getUint32(offset, true);
    offset += 4;
    const weightsData = {};
    const textDecoder = new TextDecoder();

    // Now iteratively extract each weight.
    for (let i = 0; i < numWeights; i++) {
      // Read key length.
      const keyLength = dataView.getUint32(offset, true);
      offset += 4;
      // Read key string.
      const keyBytes = new Uint8Array(buffer, offset, keyLength);
      const key = textDecoder.decode(keyBytes);
      offset += keyLength;
      // Skip padding bytes so that the total key bytes read (keyLength) is aligned to 4 bytes.
      const padding = (4 - (keyLength % 4)) % 4;
      offset += padding;

      // Read the number of dimensions.
      const numDims = dataView.getUint32(offset, true);
      offset += 4;
      const shape = [];
      for (let j = 0; j < numDims; j++) {
        shape.push(dataView.getUint32(offset, true));
        offset += 4;
      }

      // Read total number of elements.
      const numElements = dataView.getUint32(offset, true);
      offset += 4;
      // Read the float32 data.
      const byteLength = numElements * 4;
      // Slice out the chunk so that the float data is 4-byte aligned.
      const flatBuffer = buffer.slice(offset, offset + byteLength);
      const flatArray = new Float32Array(flatBuffer);
      offset += byteLength;

      // Save the weight data.
      weightsData[key] = {
        flat: flatArray,
        shape: shape,
      };
    }

    // Optional: warn about extra/missing keys if desired.
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

    for (const key in weightsData) {
      if (!expectedKeys.includes(key)) {
        console.warn(`Warning: Unexpected key "${key}" found in weights file.`);
      }
    }
    for (const key of expectedKeys) {
      if (!(key in weightsData)) {
        console.warn(
          `Warning: Expected weight "${key}" is missing from the weights file.`
        );
      }
    }

    const device = this.device;

    // Helper: write the weight data onto the GPU buffer.
    function processWeight(key, weightBuffer, expectedShapeArray) {
      if (!(key in weightsData)) return;
      const weight = weightsData[key];
      const flat = weight.flat;
      const rawShape = weight.shape;
      if (expectedShapeArray) {
        const expectedElements = expectedShapeArray.reduce((a, b) => a * b, 1);
        if (flat.length !== expectedElements) {
          console.error(
            `Weight "${key}" shape mismatch: expected shape ${expectedShapeArray} (total ${expectedElements} elements) but got ${flat.length} (raw shape: [${rawShape}]).`
          );
        } else {
          console.log(
            `Weight "${key}" loaded successfully with expected shape ${expectedShapeArray}. RAW SHAPE: [${rawShape}]`
          );
        }
      } else {
        console.log(
          `Weight "${key}" loaded with flattened length ${flat.length}.`
        );
      }
      device.queue.writeBuffer(weightBuffer, 0, flat);
    }

    // Load patch embedding weights.
    processWeight("patch_embed.weight", this.patchEmbedding.weightBuffer, [
      this.dim,
      this.patchDim,
    ]);

    // Load time MLP weights.
    processWeight("time_mlp.1.weight", this.timeMLP1.weightBuffer, [
      this.timeEmbDim,
      Math.floor(this.dim / 2),
    ]);
    processWeight("time_mlp.3.weight", this.timeMLP2.weightBuffer, [
      this.timeEmbDim,
      this.timeEmbDim,
    ]);

    // Load Transformer block weights.
    for (let i = 0; i < this.depth; i++) {
      const block = this.blocks[i];
      const prefix = `block_${i}.`;
      // Attention projection layers.
      processWeight(
        `${prefix}attn.to_q.weight`,
        block.attn.qProj.weightBuffer,
        [this.dimHead, this.dim]
      );
      processWeight(
        `${prefix}attn.to_k.weight`,
        block.attn.kProj.weightBuffer,
        [this.dimHead, this.dim]
      );
      processWeight(
        `${prefix}attn.to_v.weight`,
        block.attn.vProj.weightBuffer,
        [this.dimHead, this.dim]
      );
      processWeight(
        `${prefix}attn.to_out.weight`,
        block.attn.outProj.weightBuffer,
        [this.dim, this.dimHead]
      );
      // Feed-forward layers.
      processWeight(`${prefix}ff.net.0.weight`, block.ff.linear1.weightBuffer, [
        this.dim * this.mlpMult,
        this.dim,
      ]);
      processWeight(`${prefix}ff.net.2.weight`, block.ff.linear2.weightBuffer, [
        this.dim,
        this.dim * this.mlpMult,
      ]);
      // LayerNorm (adaptive) projections.
      processWeight(
        `${prefix}norm1.scale.weight`,
        block.norm1.scaleProj.weightBuffer,
        [this.dim, this.timeEmbDim]
      );
      processWeight(
        `${prefix}norm1.shift.weight`,
        block.norm1.shiftProj.weightBuffer,
        [this.dim, this.timeEmbDim]
      );
      processWeight(
        `${prefix}norm2.scale.weight`,
        block.norm2.scaleProj.weightBuffer,
        [this.dim, this.timeEmbDim]
      );
      processWeight(
        `${prefix}norm2.shift.weight`,
        block.norm2.shiftProj.weightBuffer,
        [this.dim, this.timeEmbDim]
      );
    }

    // Load final normalization and output projection.
    processWeight(
      "final_norm.scale.weight",
      this.finalNorm.scaleProj.weightBuffer,
      [this.dim, this.timeEmbDim]
    );
    processWeight(
      "final_norm.shift.weight",
      this.finalNorm.shiftProj.weightBuffer,
      [this.dim, this.timeEmbDim]
    );
    processWeight("to_pixels.weight", this.toPixels.weightBuffer, [
      this.patchDim,
      this.dim,
    ]);

    console.log("Weights loaded successfully.");
  }
}
