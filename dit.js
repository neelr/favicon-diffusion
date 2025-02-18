/* Added a global cache for elementWiseAdd compute pipelines */
const elementWiseAddPipelineCache = new Map();

/* dit_webgpu.js – batched command encoding version */
const ELEMENT_WISE_ADD_SHADER_CODE = (numElements) =>
  ADD_SHADER_CODE.replace(/_NUM_ELEMENTS_/g, numElements);

// Helper: createBufferFromArray remains unchanged.
function createBufferFromArray(device, array, usage) {
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

// Modified element‐wise add: now reuses cached pipeline if available.
async function gpuElementWiseAdd(
  device,
  encoder,
  bufferA,
  bufferB,
  numElements
) {
  // Use caching to avoid recreating the pipeline if already built for this numElements
  let addPipeline = elementWiseAddPipelineCache.get(numElements);
  if (!addPipeline) {
    const shaderCode = ELEMENT_WISE_ADD_SHADER_CODE(numElements);
    const addModule = device.createShaderModule({ code: shaderCode });
    addPipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module: addModule, entryPoint: "main" },
    });
    elementWiseAddPipelineCache.set(numElements, addPipeline);
  }
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
  const pass = encoder.beginComputePass();
  pass.setPipeline(addPipeline);
  pass.setBindGroup(0, bindGroup);
  const workgroupCount = Math.ceil(numElements / 64);
  pass.dispatchWorkgroups(workgroupCount);
  pass.end();
  return outputBuffer;
}

// GPUPatchify
class GPUPatchify {
  constructor(device) {
    this.device = device;
    this.pipeline = null; // will be created on demand.
  }
  async createPipeline() {
    if (!this.pipeline) {
      const shaderModule = this.device.createShaderModule({
        code: PATCHIFY_SHADER_CODE,
      });
      this.pipeline = this.device.createComputePipeline({
        layout: "auto",
        compute: { module: shaderModule, entryPoint: "main" },
      });
    }
  }
  // Now we record into the passed encoder instead of waiting.
  async patchifyImage(encoder, imageBuffer, params) {
    await this.createPipeline();
    // Compute dimensions.
    const numPatchesX = params.imageWidth / params.patchSize;
    const numPatchesY = params.imageHeight / params.patchSize;
    const patchCount = numPatchesX * numPatchesY;
    const patchDim = params.channels * params.patchSize * params.patchSize;
    const totalElements = patchCount * patchDim;
    // Create uniform buffer.
    const uniformArray = new Uint32Array([
      params.imageWidth,
      params.imageHeight,
      params.channels,
      params.patchSize,
    ]);
    const uniformBuffer = createBufferFromArray(
      this.device,
      uniformArray,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    );
    // Create output buffer.
    const patchBuffer = this.device.createBuffer({
      size: totalElements * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    // Create bind group.
    const bindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: imageBuffer } },
        { binding: 1, resource: { buffer: uniformBuffer } },
        { binding: 2, resource: { buffer: patchBuffer } },
      ],
    });
    // Record dispatch in the provided encoder.
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(totalElements / 64));
    pass.end();
    return patchBuffer;
  }
}

// GPUUnpatchify
class GPUUnpatchify {
  constructor(device) {
    this.device = device;
    this.pipeline = null;
  }
  async createPipeline() {
    if (!this.pipeline) {
      const shaderModule = this.device.createShaderModule({
        code: UNPATCHIFY_SHADER_CODE,
      });
      this.pipeline = this.device.createComputePipeline({
        layout: "auto",
        compute: { module: shaderModule, entryPoint: "main" },
      });
    }
  }
  async unpatchifyPatches(encoder, patchesBuffer, params) {
    await this.createPipeline();
    const numPatchesX = params.imageWidth / params.patchSize;
    const numPatchesY = params.imageHeight / params.patchSize;
    const patchCount = numPatchesX * numPatchesY;
    const patchDim = params.channels * params.patchSize * params.patchSize;
    const totalElements = patchCount * patchDim;
    // Create uniform buffer.
    const uniformArray = new Uint32Array([
      params.imageWidth,
      params.imageHeight,
      params.channels,
      params.patchSize,
    ]);
    const uniformBuffer = createBufferFromArray(
      this.device,
      uniformArray,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    );
    // Create output buffer.
    const imageBuffer = this.device.createBuffer({
      size: params.channels * params.imageWidth * params.imageHeight * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    // Create bind group.
    const bindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: patchesBuffer } },
        { binding: 1, resource: { buffer: uniformBuffer } },
        { binding: 2, resource: { buffer: imageBuffer } },
      ],
    });
    // Record dispatch.
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(totalElements / 64));
    pass.end();
    return imageBuffer;
  }
}

// GPUMatrixMultiply
class GPUMatrixMultiply {
  constructor(device) {
    this.device = device;
    this.pipeline = null;
  }
  async createPipeline() {
    if (!this.pipeline) {
      const shaderModule = this.device.createShaderModule({
        code: MATMUL_SHADER_CODE,
      });
      this.pipeline = this.device.createComputePipeline({
        layout: "auto",
        compute: { module: shaderModule, entryPoint: "main" },
      });
    }
  }
  async matrixMultiply(
    encoder,
    inputBuffer,
    weightBuffer,
    inputShape,
    outFeatures
  ) {
    await this.createPipeline();
    const m = inputShape[0];
    const k = inputShape[1];
    const n = outFeatures;
    // Create uniform buffer.
    const dimsArray = new Uint32Array([m, k, n, 0]);
    const dimsBuffer = createBufferFromArray(
      this.device,
      dimsArray,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    );
    // Create output buffer.
    const outputBuffer = this.device.createBuffer({
      size: m * n * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const bindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: weightBuffer } },
        { binding: 2, resource: { buffer: outputBuffer } },
        { binding: 3, resource: { buffer: dimsBuffer } },
      ],
    });
    // Record dispatch.
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(m / 16), Math.ceil(n / 16));
    pass.end();
    return outputBuffer;
  }
}

// GPULinear
class GPULinear {
  constructor(device, inFeatures, outFeatures) {
    this.device = device;
    this.inFeatures = inFeatures;
    this.outFeatures = outFeatures;
    this.mm = new GPUMatrixMultiply(device);
    // Initialize weights using Xavier/Glorot initialization.
    const stddev = Math.sqrt(2.0 / (inFeatures + outFeatures));
    const weightArray = new Float32Array(outFeatures * inFeatures);
    // Create weight buffer.
    this.weightBuffer = device.createBuffer({
      size: weightArray.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Float32Array(this.weightBuffer.getMappedRange()).set(weightArray);
    this.weightBuffer.unmap();
  }
  // forward now accepts a command encoder.
  async forward(encoder, inputBuffer, inputShape) {
    return await this.mm.matrixMultiply(
      encoder,
      inputBuffer,
      this.weightBuffer,
      inputShape,
      this.outFeatures
    );
  }
}

// GPUAdaLN – adaptive layer normalization skeleton.
class GPUAdaLN {
  constructor(device, dim, timeEmbDim) {
    this.device = device;
    this.dim = dim;
    this.timeEmbDim = timeEmbDim;
    this.pipeline = null;
    // Initialize linear projections for scale and shift.
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
  // forward accepts command encoder (xBuffer, timeEmbBuffer, numElements)
  async forward(encoder, xBuffer, timeEmbBuffer, numElements) {
    await this.createPipeline();
    // Project time embeddings.
    const shiftValues = await this.shiftProj.forward(encoder, timeEmbBuffer, [
      1,
      this.timeEmbDim,
    ]);
    const scaleValues = await this.scaleProj.forward(encoder, timeEmbBuffer, [
      1,
      this.timeEmbDim,
    ]);
    const rowWidth = this.dim;
    const nRows = numElements / this.dim;
    // Create uniform buffer.
    const uniformData = new Float32Array([nRows, rowWidth, 1e-8]);
    const uniformBuffer = createBufferFromArray(
      this.device,
      uniformData,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    );
    // Create output buffer.
    const outputBuffer = this.device.createBuffer({
      size: numElements * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
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
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(nRows);
    pass.end();
    return outputBuffer;
  }
}

// GPUAttention – multi-head attention skeleton.
class GPUAttention {
  constructor(device, dim, dimHead) {
    this.device = device;
    this.dim = dim;
    this.dimHead = dimHead;
    this.softmaxPipeline = null;
    this.batchedMatMulPipeline = null;
    this.scaleMatrixPipeline = null;
    // Projections for Q, K, V.
    this.qProj = new GPULinear(device, dim, dimHead);
    this.kProj = new GPULinear(device, dim, dimHead);
    this.vProj = new GPULinear(device, dim, dimHead);
    // Output projection.
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
  // Helper: createBufferFromArray (same as above)
  static createBufferFromArray(device, array, usage) {
    return createBufferFromArray(device, array, usage);
  }
  // ---------------- batchedMatMul  ----------------
  async batchedMatMul(encoder, A, B, aRows, k, bCols, flagTranspose) {
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
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.batchedMatMulPipeline);
    pass.setBindGroup(0, bindGroup);
    const workgroupCountX = Math.ceil(aRows / 16);
    const workgroupCountY = Math.ceil(bCols / 16);
    pass.dispatchWorkgroups(workgroupCountX, workgroupCountY);
    pass.end();
    return C;
  }
  // ---------------- scaleMatrix ----------------
  async scaleMatrix(encoder, matrix, rows, cols, scalarValue) {
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
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.scaleMatrixPipeline);
    pass.setBindGroup(0, bindGroup);
    const workgroupCount = Math.ceil(totalElements / 64);
    pass.dispatchWorkgroups(workgroupCount);
    pass.end();
    return matrix;
  }
  // ---------------- softmax ----------------
  async softmax(encoder, matrix, rows, cols) {
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
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.softmaxPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(rows);
    pass.end();
    return outBuffer;
  }
  // ---------------- attention ----------------
  async attention(encoder, Q, K, V, nTokens, dimHead) {
    const scoresBuffer = await this.batchedMatMul(
      encoder,
      Q,
      K,
      nTokens,
      dimHead,
      nTokens,
      1
    );
    const scaleFactor = 1.0 / Math.sqrt(dimHead);
    await this.scaleMatrix(
      encoder,
      scoresBuffer,
      nTokens,
      nTokens,
      scaleFactor
    );
    const softmaxBuffer = await this.softmax(
      encoder,
      scoresBuffer,
      nTokens,
      nTokens
    );
    const attnOutput = await this.batchedMatMul(
      encoder,
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
  async forward(encoder, x, nTokens) {
    const Q = await this.qProj.forward(encoder, x, [nTokens, this.dim]);
    const K = await this.kProj.forward(encoder, x, [nTokens, this.dim]);
    const V = await this.vProj.forward(encoder, x, [nTokens, this.dim]);
    const attnBuffer = await this.attention(
      encoder,
      Q,
      K,
      V,
      nTokens,
      this.dimHead
    );
    const outputBuffer = await this.outProj.forward(encoder, attnBuffer, [
      nTokens,
      this.dimHead,
    ]);
    return outputBuffer;
  }
}

// GPUFeedForward – two‐layer MLP skeleton.
class GPUFeedForward {
  constructor(device, dim, mult = 4) {
    this.device = device;
    this.dim = dim;
    this.mult = mult;
    this.siluPipeline = null;
    this.linear1 = new GPULinear(device, dim, dim * mult);
    this.linear2 = new GPULinear(device, dim * mult, dim);
  }
  async createSiluPipeline() {
    if (!this.siluPipeline) {
      this.siluModule = this.device.createShaderModule({
        code: SILU_SHADER_CODE,
      });
      this.siluPipeline = this.device.createComputePipeline({
        layout: "auto",
        compute: { module: this.siluModule, entryPoint: "main" },
      });
    }
  }
  // Apply SiLU activation elementwise.
  async applySilu(encoder, inputBuffer, numElements) {
    await this.createSiluPipeline();
    const outputBuffer = this.device.createBuffer({
      size: numElements * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const uniformArray = new Uint32Array([numElements]);
    const uniformBuffer = createBufferFromArray(
      this.device,
      uniformArray,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    );
    const bindGroup = this.device.createBindGroup({
      layout: this.siluPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: outputBuffer } },
        { binding: 2, resource: { buffer: uniformBuffer } },
      ],
    });
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.siluPipeline);
    pass.setBindGroup(0, bindGroup);
    const workgroupCount = Math.ceil(numElements / 64);
    pass.dispatchWorkgroups(workgroupCount);
    pass.end();
    return outputBuffer;
  }
  // forward (xBuffer shape: [batchTimesSeq, dim])
  async forward(encoder, xBuffer, batchTimesSeq) {
    const hidden = await this.linear1.forward(encoder, xBuffer, [
      batchTimesSeq,
      this.dim,
    ]);
    const numHiddenElements = batchTimesSeq * (this.dim * this.mult);
    const activatedHidden = await this.applySilu(
      encoder,
      hidden,
      numHiddenElements
    );
    const output = await this.linear2.forward(encoder, activatedHidden, [
      batchTimesSeq,
      this.dim * this.mult,
    ]);
    return output;
  }
}

// GPUTransformerBlock – one transformer block.
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
  async forward(encoder, xBuffer, tBuffer, numElements) {
    const nTokens = numElements / this.dim;
    const norm1Out = await this.norm1.forward(
      encoder,
      xBuffer,
      tBuffer,
      numElements
    );
    const attnOut = await this.attn.forward(encoder, norm1Out, nTokens);
    const res1 = await gpuElementWiseAdd(
      this.device,
      encoder,
      xBuffer,
      attnOut,
      numElements
    );
    const norm2Out = await this.norm2.forward(
      encoder,
      res1,
      tBuffer,
      numElements
    );
    const ffOut = await this.ff.forward(encoder, norm2Out, nTokens);
    const res2 = await gpuElementWiseAdd(
      this.device,
      encoder,
      res1,
      ffOut,
      numElements
    );
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
    this.gridSize = this.inputSize / this.patchSize;
    this.numPatches = this.gridSize * this.gridSize;
    this.patchDim = this.inChannels * this.patchSize * this.patchSize;
    // Patch embedding.
    this.patchEmbedding = new GPULinear(device, this.patchDim, this.dim);
    // Positional embeddings.
    this.posEmbBuffer = this.create2DPositionalEmbeddings();
    // Time embedding MLPs.
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
  // Corrected sinusoidalPosEmb with caching to avoid duplicate calculations.
  sinusoidalPosEmb(x, embDim) {
    if (!this._sinusoidalCache) {
      this._sinusoidalCache = {};
    }
    const key = `${x}_${embDim}`;
    if (this._sinusoidalCache[key]) {
      return this._sinusoidalCache[key];
    }
    const halfDim = embDim / 2;
    const log10000 = Math.log(10000);
    const emb = new Float32Array(halfDim);
    for (let i = 0; i < halfDim; i++) {
      emb[i] = Math.exp(-(i * log10000) / (halfDim - 1));
    }
    const out = new Float32Array(embDim);
    for (let i = 0; i < halfDim; i++) {
      const angle = x * emb[i];
      out[i] = Math.sin(angle);
      out[i + halfDim] = Math.cos(angle);
    }
    this._sinusoidalCache[key] = out;
    return out;
  }
  // Create 2D positional embeddings – mimicking a fixed sinusoidal embedding.
  create2DPositionalEmbeddings() {
    const numPatches = this.gridSize * this.gridSize;
    const posEmb = new Float32Array(numPatches * this.dim);
    let patchIdx = 0;
    for (let h = 0; h < this.gridSize; h++) {
      // Cache the sinusoidal embedding for height once per value
      const hEmb = this.sinusoidalPosEmb(h, this.dim / 2);
      for (let w = 0; w < this.gridSize; w++) {
        const wEmb = this.sinusoidalPosEmb(w, this.dim / 2);
        const offset = patchIdx * this.dim;
        for (let i = 0; i < this.dim / 2; i++) {
          posEmb[offset + i] = hEmb[i];
          posEmb[offset + i + this.dim / 2] = wEmb[i];
        }
        patchIdx++;
      }
    }
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
  // The forward pass now uses a single command encoder to batch work.
  async forward(inputImage, timeVal) {
    // Create input image buffer.
    const imageBuffer = createBufferFromArray(
      this.device,
      inputImage,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    );
    // Create a single command encoder.
    const encoder = this.device.createCommandEncoder();
    // Patchify.
    const patchifyParams = {
      imageWidth: this.inputSize,
      imageHeight: this.inputSize,
      channels: this.inChannels,
      patchSize: this.patchSize,
    };
    const patchesBuffer = await this.patchify.patchifyImage(
      encoder,
      imageBuffer,
      patchifyParams
    );
    // Patch embedding.
    let x = await this.patchEmbedding.forward(encoder, patchesBuffer, [
      this.numPatches,
      this.patchDim,
    ]);
    // Add positional embeddings (element‐wise addition).
    x = await gpuElementWiseAdd(
      this.device,
      encoder,
      x,
      this.posEmbBuffer,
      this.numPatches * this.dim
    );
    // Time embedding.
    const sinEmb = this.sinusoidalPosEmb(timeVal, Math.floor(this.dim / 2));
    const sinEmbBuffer = createBufferFromArray(
      this.device,
      sinEmb,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    );
    const t1 = await this.timeMLP1.forward(encoder, sinEmbBuffer, [
      1,
      Math.floor(this.dim / 2),
    ]);
    // Use the feed-forward's SiLU method directly.
    const t1Silu = await this.blocks[0].ff.applySilu(
      encoder,
      t1,
      this.timeEmbDim
    );
    const tEmbedding = await this.timeMLP2.forward(encoder, t1Silu, [
      1,
      this.timeEmbDim,
    ]);
    // Process transformer blocks.
    const totalElements = this.numPatches * this.dim;
    for (const block of this.blocks) {
      x = await block.forward(encoder, x, tEmbedding, totalElements);
    }
    // Final normalization and projection.
    const normed = await this.finalNorm.forward(
      encoder,
      x,
      tEmbedding,
      totalElements
    );
    const projected = await this.toPixels.forward(encoder, normed, [
      this.numPatches,
      this.dim,
    ]);
    // Unpatchify back to an image.
    const outputBuffer = await this.unpatchify.unpatchifyPatches(
      encoder,
      projected,
      patchifyParams
    );
    // Finish recording and submit.
    const cmdBuffer = encoder.finish();
    this.device.queue.submit([cmdBuffer]);
    // Optionally: wait for the entire batched work to complete, if needed.
    // await this.device.queue.onSubmittedWorkDone();
    return outputBuffer;
  }
  async loadWeights(weightsURL) {
    // Initialize IndexedDB
    const dbName = "weights-cache";
    const storeName = "weights";
    const version = 1;

    // Helper function to open DB
    const openDB = () => {
      return new Promise((resolve, reject) => {
        const request = indexedDB.open(dbName, version);

        request.onerror = () => reject(request.error);
        request.onsuccess = () => resolve(request.result);

        request.onupgradeneeded = (event) => {
          const db = event.target.result;
          if (!db.objectStoreNames.contains(storeName)) {
            db.createObjectStore(storeName);
          }
        };
      });
    };

    // Helper function to get from IndexedDB
    const getFromCache = async (db, key) => {
      return new Promise((resolve, reject) => {
        const transaction = db.transaction(storeName, "readonly");
        const store = transaction.objectStore(storeName);
        const request = store.get(key);

        request.onerror = () => reject(request.error);
        request.onsuccess = () => resolve(request.result);
      });
    };

    // Helper function to save to IndexedDB
    const saveToCache = async (db, key, value) => {
      return new Promise((resolve, reject) => {
        const transaction = db.transaction(storeName, "readwrite");
        const store = transaction.objectStore(storeName);
        const request = store.put(value, key);

        request.onerror = () => reject(request.error);
        request.onsuccess = () => resolve();
      });
    };

    try {
      // Open IndexedDB
      const db = await openDB();

      // Try to get cached weights
      // let buffer = await getFromCache(db, weightsURL);

      console.log("Cache miss - fetching weights from network");
      // If not in cache, fetch from network
      const response = await fetch(weightsURL);
      const buffer = await response.arrayBuffer();

      // Process the weights (existing logic)
      const dataView = new DataView(buffer);
      let offset = 0;
      let magic = "";
      for (let i = 0; i < 4; i++) {
        magic += String.fromCharCode(dataView.getUint8(offset));
        offset += 1;
      }
      if (magic !== "WTS1") {
        throw new Error("Invalid weights file format");
      }
      const numWeights = dataView.getUint32(offset, true);
      offset += 4;
      const weightsData = {};
      const textDecoder = new TextDecoder();

      for (let i = 0; i < numWeights; i++) {
        const keyLength = dataView.getUint32(offset, true);
        offset += 4;
        const keyBytes = new Uint8Array(buffer, offset, keyLength);
        const key = textDecoder.decode(keyBytes);
        offset += keyLength;
        const padding = (4 - (keyLength % 4)) % 4;
        offset += padding;
        const numDims = dataView.getUint32(offset, true);
        offset += 4;
        const shape = [];
        for (let j = 0; j < numDims; j++) {
          shape.push(dataView.getUint32(offset, true));
          offset += 4;
        }
        const numElements = dataView.getUint32(offset, true);
        offset += 4;
        const byteLength = numElements * 4;
        const flatBuffer = buffer.slice(offset, offset + byteLength);
        const flatArray = new Float32Array(flatBuffer);
        offset += byteLength;
        weightsData[key] = {
          flat: flatArray,
          shape: shape,
        };
      }

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

      // Validation checks
      for (const key in weightsData) {
        if (!expectedKeys.includes(key)) {
          console.warn(
            `Warning: Unexpected key "${key}" found in weights file.`
          );
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
      function processWeight(key, weightBuffer, expectedShapeArray) {
        if (!(key in weightsData)) return;
        const weight = weightsData[key];
        const flat = weight.flat;
        const rawShape = weight.shape;
        if (expectedShapeArray) {
          const expectedElements = expectedShapeArray.reduce(
            (a, b) => a * b,
            1
          );
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

      // Process all weights
      processWeight("patch_embed.weight", this.patchEmbedding.weightBuffer, [
        this.dim,
        this.patchDim,
      ]);
      processWeight("time_mlp.1.weight", this.timeMLP1.weightBuffer, [
        this.timeEmbDim,
        Math.floor(this.dim / 2),
      ]);
      processWeight("time_mlp.3.weight", this.timeMLP2.weightBuffer, [
        this.timeEmbDim,
        this.timeEmbDim,
      ]);

      for (let i = 0; i < this.depth; i++) {
        const block = this.blocks[i];
        const prefix = `block_${i}.`;
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
        processWeight(
          `${prefix}ff.net.0.weight`,
          block.ff.linear1.weightBuffer,
          [this.dim * this.mlpMult, this.dim]
        );
        processWeight(
          `${prefix}ff.net.2.weight`,
          block.ff.linear2.weightBuffer,
          [this.dim, this.dim * this.mlpMult]
        );
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
    } catch (error) {
      console.error("Error loading weights:", error);
      throw error;
    }
  }
}
