#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

__device__ __forceinline__ float h4(float a, float b, float c, float d, int index) {
    if (index == 0) return (((a + b) + c) - d) * .5f;
    if (index == 1) return (((a + b) - c) + d) * .5f;
    if (index == 2) return (((a - b) + c) + d) * .5f;
    return (((b - a) + c) + d) * .5f;
}

// Exact installed FP32 SiLU is indexed by the original Half gate bits.
// The Half store/conversion is a required boundary before ConvRot and PDD.
__global__ void joint_rows(const half* gdata, half* activation, signed char* q,
                           float* scales, const float* lut, int k) {
    extern __shared__ float scratch[];
    float* maxima = scratch + k;
    const int row = blockIdx.x;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;
    float maximum = 0.f;
    for (int group = warp; group < k / 256; group += warps) {
        float v[8], temp[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            int col = group * 256 + lane + 32 * j;
            half gate = gdata[row * 2 * k + col];
            float up = __half2float(gdata[row * 2 * k + col + k]);
            float product = __fmul_rn(lut[__half_as_ushort(gate)], up);
            half rounded = __float2half_rn(__fmul_rn(product, 0.00390625f));
            activation[row * k + col] = rounded;
            v[j] = __half2float(rounded);
        }
        #pragma unroll
        for (int shift = 0; shift < 4; shift += 2) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                int base = lane & ~(3 << shift);
                float a = __shfl_sync(0xffffffff, v[j], base);
                float b = __shfl_sync(0xffffffff, v[j], base | (1 << shift));
                float c = __shfl_sync(0xffffffff, v[j], base | (2 << shift));
                float d = __shfl_sync(0xffffffff, v[j], base | (3 << shift));
                temp[j] = h4(a, b, c, d, (lane >> shift) & 3);
            }
            #pragma unroll
            for (int j = 0; j < 8; ++j) v[j] = temp[j];
        }
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            float a = __shfl_sync(0xffffffff, v[j & ~1], lane & 15);
            float b = __shfl_sync(0xffffffff, v[j & ~1], lane | 16);
            float c = __shfl_sync(0xffffffff, v[j | 1], lane & 15);
            float d = __shfl_sync(0xffffffff, v[j | 1], lane | 16);
            temp[j] = h4(a, b, c, d, ((j & 1) << 1) | (lane >> 4));
        }
        #pragma unroll
        for (int j = 0; j < 8; ++j) v[j] = temp[j];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            int base = j & ~6;
            float value = h4(v[base], v[base | 2], v[base | 4], v[base | 6], j >> 1);
            scratch[group * 256 + lane + 32 * j] = value;
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
        if (lane == 0) maxima[0] = fmaxf(fminf(maximum, 65504.f) * (1.f / 127.f), 1.e-30f);
    }
    __syncthreads();
    const float scale = maxima[0];
    if (threadIdx.x == 0) scales[row] = scale;
    for (int col = threadIdx.x; col < k; col += blockDim.x) {
        float value = __half2float(__float2half_rn(scratch[col]));
        float denominator = __half2float(__float2half_rn(scale));
        float divided;
        // Match the installed CK quantizer, including its three Half conversions.
        asm("div.approx.ftz.f32 %0, %1, %2;" : "=f"(divided) : "f"(value), "f"(denominator));
        float rounded = nearbyintf(__half2float(__float2half_rn(divided)));
        rounded = fminf(fmaxf(rounded, -128.f), 127.f);
        q[row * k + col] = static_cast<signed char>(static_cast<int>(rounded));
    }
}

extern "C" int h3_joint(const void* g, void* a, void* q, void* s, const void* lut,
                         int m, int k, int warps, void* stream) {
    if (m < 1 || k != 7168 || (warps != 4 && warps != 8)) return -1;
    joint_rows<<<m, warps * 32, (k + warps) * sizeof(float), static_cast<cudaStream_t>(stream)>>>(
        static_cast<const half*>(g), static_cast<half*>(a), static_cast<signed char*>(q),
        static_cast<float*>(s), static_cast<const float*>(lut), k);
    return static_cast<int>(cudaGetLastError());
}
