async function logBuffer(device, buffer, label = "Buffer stats (first 300)") {
  return;
  //  return;
  // Create a staging buffer for reading
  const stagingBuffer = device.createBuffer({
    size: buffer.size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  // Create and submit copy command
  const commandEncoder = device.createCommandEncoder();
  commandEncoder.copyBufferToBuffer(buffer, 0, stagingBuffer, 0, buffer.size);
  device.queue.submit([commandEncoder.finish()]);

  // Map the staging buffer and read its contents
  await stagingBuffer.mapAsync(GPUMapMode.READ);
  const copyArrayBuffer = stagingBuffer.getMappedRange();
  const data = new Float32Array(copyArrayBuffer);

  // Calculate stats of first 300 numbers
  const length = data.length;
  let sum = 0;
  let min = Infinity;
  let max = -Infinity;

  // First pass: sum, min, max
  for (let i = 0; i < length; i++) {
    const value = data[i];
    sum += value;
    min = Math.min(min, value);
    max = Math.max(max, value);
  }

  const mean = sum / length;

  // Second pass: standard deviation
  let sumSquaredDiff = 0;
  for (let i = 0; i < length; i++) {
    const diff = data[i] - mean;
    sumSquaredDiff += diff * diff;
  }
  const std = Math.sqrt(sumSquaredDiff / length);

  const stats = {
    length,
    mean,
    std,
    min,
    max,
    first5: data.slice(0, 5),
  };

  // Log the results
  console.log(`${label}:`, stats);

  // Clean up
  stagingBuffer.unmap();
  stagingBuffer.destroy();

  return stats;
}

const ELEMENT_WISE_ADD_SHADER_CODE = (numElements) =>
  ADD_SHADER_CODE.replace(/_NUM_ELEMENTS_/g, numElements);

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
