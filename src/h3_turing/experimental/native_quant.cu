#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

// Regular H4 Kronecker rotation: for output i the negative term is i xor 3.
__global__ void quantize_rows(const half* x, signed char* q, float* scales,
                             int k, float clip) {
    extern __shared__ float scratch[];
    float* maxima = scratch + k;
    const int row = blockIdx.x;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;
    float maximum = 0.f;
    for (int g = warp; g < k / 256; g += warps) {
        float v[8], temp[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j)
            v[j] = __half2float(x[row * k + g * 256 + lane + 32 * j]);
        #pragma unroll
        for (int shift = 0; shift < 4; shift += 2) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                float b = __shfl_xor_sync(0xffffffff, v[j], 1 << shift);
                float c = __shfl_xor_sync(0xffffffff, v[j], 2 << shift);
                float d = __shfl_xor_sync(0xffffffff, v[j], 3 << shift);
                temp[j] = ((v[j] + b) + c) - d;
            }
            #pragma unroll
            for (int j = 0; j < 8; ++j) v[j] = temp[j];
        }
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            float b = __shfl_xor_sync(0xffffffff, v[j], 16);
            float d = __shfl_xor_sync(0xffffffff, v[j ^ 1], 16);
            temp[j] = ((v[j] + b) + v[j ^ 1]) - d;
        }
        #pragma unroll
        for (int j = 0; j < 8; ++j) v[j] = temp[j];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            float value = (((v[j] + v[j ^ 2]) + v[j ^ 4]) - v[j ^ 6]) * .0625f;
            scratch[g * 256 + lane + 32 * j] = value;
            maximum = fmaxf(maximum, fabsf(value));
        }
    }
    #pragma unroll
    for (int offset = 16; offset; offset >>= 1)
        maximum = fmaxf(maximum, __shfl_down_sync(0xffffffff, maximum, offset));
    if (lane == 0) maxima[warp] = maximum;
    __syncthreads();
    if (warp == 0) {
        maximum = lane < warps ? maxima[lane] : 0.f;
        #pragma unroll
        for (int offset = 16; offset; offset >>= 1)
            maximum = fmaxf(maximum, __shfl_down_sync(0xffffffff, maximum, offset));
        if (lane == 0) maxima[0] = fmaxf(maximum * clip / 127.f, 1.e-12f);
    }
    __syncthreads();
    const float scale = maxima[0];
    if (threadIdx.x == 0) scales[row] = scale;
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        int value = __float2int_rn(scratch[i] / scale);
        value = value < -127 ? -127 : (value > 127 ? 127 : value);
        q[row * k + i] = static_cast<signed char>(value);
    }
}

extern "C" int h3_quant(const void* x, void* q, void* scales,
                        int m, int k, float clip, int warps, void* stream) {
    if (k % 256 || k > 16384 || (warps != 4 && warps != 8)) return -1;
    const int shared = (k + warps) * sizeof(float);
    if (shared > 48 * 1024) {
        cudaError_t status = cudaFuncSetAttribute(quantize_rows,
            cudaFuncAttributeMaxDynamicSharedMemorySize, shared);
        if (status != cudaSuccess) return static_cast<int>(status);
    }
    quantize_rows<<<m, warps * 32, shared, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const half*>(x), static_cast<signed char*>(q),
        static_cast<float*>(scales), k, clip);
    return static_cast<int>(cudaGetLastError());
}
