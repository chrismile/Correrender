/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2023, Christoph Neuhauser
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
    {
        const uint numChannelsInPadded = layerIdx == 0 ? NUM_CHANNELS_IN_PADDED : NUM_CHANNELS_HIDDEN;
        const uint numChannelsOutPadded = layerIdx == NUM_LAYERS - 1 ? NUM_CHANNELS_OUT_PADDED : NUM_CHANNELS_HIDDEN;
        const uint nCols = numChannelsInPadded / M;
        const uint nRows = numChannelsOutPadded / M; // May be less than N_ROWS for the last layer.
        weightStride = numChannelsInPadded;
#if defined(LAYER_USE_SHARED_MEMORY_INPUT) && defined(BANK_SKEW)
        inputStride = (numChannelsInPadded + BANK_SKEW) / SMEM_FACTOR;
#else
        inputStride = numChannelsInPadded / SMEM_FACTOR;
#endif
#if defined(LAYER_USE_SHARED_MEMORY_OUTPUT) && defined(BANK_SKEW)
        outputStride = (numChannelsOutPadded + BANK_SKEW) / SMEM_FACTOR;
#else
        outputStride = numChannelsOutPadded / SMEM_FACTOR;
#endif

if (blockRowIdx < nRows) {
            // Clear the output matrices.
            [[unroll]] for (uint b = 0; b < N_BATCH; b++) {
                outputMat[b] = CoopMatAcc(0.0);
            }

            for (uint c = 0; c < nCols; c++) {
                const uint weightOffset = weightOffsetBase + c * M + blockRowIdx * M * weightStride;
                matLoad(weightsMat, parametersBuffer, weightOffset, weightStride, ROW_MAJOR);
                [[unroll]] for (uint b = 0; b < N_BATCH; b++) {
#ifdef LAYER_USE_SHARED_MEMORY_INPUT
                    inputOffset = c * (M / SMEM_FACTOR) + b * M * inputStride;
#else
                    inputOffset = c * (M / SMEM_FACTOR) + b * M * inputStride + batchOffset * NUM_CHANNELS_IN_PADDED;
#endif
                    matLoad(
                            inputMat,
#ifdef LAYER_USE_SHARED_MEMORY_INPUT
                            sharedMemory,
#else
                            inputBuffer,
#endif
                            inputOffset, inputStride, COL_MAJOR);
                    outputMat[b] = matMulAdd(weightsMat, inputMat, outputMat[b]);
                }
            }

            // Apply activation function.
#ifdef NO_OUTPUT_ACTIVATION
            if (layerIdx != NUM_LAYERS - 1) {
#endif
                /*for (uint i = localThreadIdx; i < M * M * N_BATCH; i += SUBGROUP_SIZE) {
                    const uint b = i % (M * M);
                    const uint j = i / (M * M);
                    outputMat[b][j] = ACTIVATION_FUNCTION(outputMat[b][j]);
                }*/
                [[unroll]] for (uint b = 0; b < N_BATCH; b++) {
                    for (uint i = 0; i < outputMat[b].length(); ++i) {
                        outputMat[b][i] = ACTIVATION_FUNCTION(outputMat[b][i]);
                    }
                }
#ifdef NO_OUTPUT_ACTIVATION
            }
#endif
        }

        barrier();

        if (blockRowIdx < nRows) {
            // Store to shared memory.
            [[unroll]] for (uint b = 0; b < N_BATCH; b++) {
#ifdef LAYER_USE_SHARED_MEMORY_OUTPUT
                outputOffset = blockRowIdx * (M / SMEM_FACTOR) + b * M * outputStride;
#else
                outputOffset = blockRowIdx * (M / SMEM_FACTOR) + b * M * outputStride + batchOffset * NUM_CHANNELS_OUT_PADDED;
#endif
                matStore(
                        outputMat[b],
#ifdef LAYER_USE_SHARED_MEMORY_OUTPUT
                        sharedMemory,
#else
                        outputBuffer,
#endif
                        outputOffset, outputStride, COL_MAJOR);
            }
        }

#ifdef LAYER_USE_SHARED_MEMORY_OUTPUT
        memoryBarrierShared();
        barrier();
#endif

        // Update offsets.
        weightOffsetBase += numChannelsOutPadded * numChannelsInPadded;
    }
