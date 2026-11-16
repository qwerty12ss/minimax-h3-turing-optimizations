#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

template<int K, int W>
__global__ void quant_regs(const half* x, signed char* q, float* s, float clip) {
    __shared__ float maxima[W];
    constexpr int G = (K / 256 + W - 1) / W;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int row = blockIdx.x;
    float saved[G][8];
    float maximum = 0.f;
    #pragma unroll
    for (int t = 0; t < G; ++t) {
        const int g = warp + t * W;
        if (g < K / 256) {
            float v[8], temp[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j)
                v[j] = __half2float(x[row * K + g * 256 + lane + 32*j]);
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
                saved[t][j] = value;
                maximum = fmaxf(maximum, fabsf(value));
            }
        }
    }
    #pragma unroll
    for (int offset = 16; offset; offset >>= 1)
        maximum = fmaxf(maximum, __shfl_down_sync(0xffffffff, maximum, offset));
    if (lane == 0) maxima[warp] = maximum;
    __syncthreads();
    if (warp == 0) {
        maximum = lane < W ? maxima[lane] : 0.f;
        #pragma unroll
        for (int offset = 16; offset; offset >>= 1)
            maximum = fmaxf(maximum, __shfl_down_sync(0xffffffff, maximum, offset));
        if (lane == 0) maxima[0] = fmaxf(maximum * clip / 127.f, 1.e-12f);
    }
    __syncthreads();
    const float scale = maxima[0];
    if (threadIdx.x == 0) s[row] = scale;
    #pragma unroll
    for (int t = 0; t < G; ++t) {
        const int g = warp + t * W;
        if (g < K / 256) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                int a = __float2int_rn(saved[t][j] / scale);
                a = a < -127 ? -127 : (a > 127 ? 127 : a);
                q[row*K + g*256 + lane + 32*j] = static_cast<signed char>(a);
            }
        }
    }
}

extern "C" int h3_quant(const void* x, void* q, void* s, int m, int k, float clip, int w, void* stream) {
    #define LAUNCH(K,W) quant_regs<K,W><<<m,W*32,0,static_cast<cudaStream_t>(stream)>>>(static_cast<const half*>(x),static_cast<signed char*>(q),static_cast<float*>(s),clip)
    if (k == 5376 && w == 4) { LAUNCH(5376,4); }
    else if (k == 5376 && w == 8) { LAUNCH(5376,8); }
    else if (k == 7168 && w == 4) { LAUNCH(7168,4); }
    else if (k == 7168 && w == 8) { LAUNCH(7168,8); }
    else return -1;
    return static_cast<int>(cudaGetLastError());
}
