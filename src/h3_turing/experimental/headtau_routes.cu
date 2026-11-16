#include "sol_headtau_preprocess.cuh"

extern "C" __attribute__((visibility("default")))
int star7_sol_sm75_prepare_quantized_routes_headtau_v1(
    std::uintptr_t q, std::uintptr_t k, std::uintptr_t v,
    std::uintptr_t qc, std::uintptr_t kc, std::uintptr_t vc,
    std::uintptr_t mean, std::uintptr_t variance, std::uintptr_t threshold,
    std::uintptr_t mask, std::uintptr_t count,
    std::uintptr_t q8, std::uintptr_t k8, std::uintptr_t qs, std::uintptr_t ks,
    std::int64_t qb, std::int64_t qh, std::int64_t qt,
    std::int64_t kb, std::int64_t kh, std::int64_t kt,
    std::int64_t vb, std::int64_t vh, std::int64_t vt,
    int batch, int heads, int length, int padded_blocks,
    float tau, float scale, int sink_first, int sink_last,
    std::uintptr_t stream, std::uintptr_t head_tau) {
  const int blocks = (length + 63) / 64;
  if (!head_tau || !q || !k || !v || !qc || !kc || !vc || !mean || !variance ||
      !threshold || !mask || !count || !q8 || !k8 || !qs || !ks ||
      batch <= 0 || heads <= 0 || length <= 0 || padded_blocks < blocks ||
      padded_blocks % 16 != 0 || sink_first < 0 ||
      sink_last < sink_first || sink_last > blocks)
    return static_cast<int>(cudaErrorInvalidValue);
  const cudaError_t error = star7_sol_preprocess::prepare_routes(
      reinterpret_cast<const half *>(q), reinterpret_cast<const half *>(k),
      reinterpret_cast<const half *>(v), reinterpret_cast<half *>(qc),
      reinterpret_cast<half *>(kc), reinterpret_cast<half *>(vc),
      reinterpret_cast<float *>(mean), reinterpret_cast<float *>(variance),
      reinterpret_cast<float *>(threshold), reinterpret_cast<std::uint8_t *>(mask),
      reinterpret_cast<std::int32_t *>(count), batch, heads, length,
      padded_blocks, tau, scale, sink_first, sink_last,
      reinterpret_cast<cudaStream_t>(stream),
      reinterpret_cast<std::int8_t *>(q8), reinterpret_cast<std::int8_t *>(k8),
      reinterpret_cast<float *>(qs), reinterpret_cast<float *>(ks),
      {qb, qh, qt}, {kb, kh, kt}, {vb, vh, vt}, reinterpret_cast<const float *>(head_tau));
  return error == cudaSuccess ? 0 : 14000 + static_cast<int>(error);
}
