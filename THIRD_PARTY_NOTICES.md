# Source attribution

The published SM75 attention backend and preprocessing are derived from the Star7 research source snapshot. Its MIT license is retained in [THIRD_PARTY_LICENSES/Star7-MIT.txt](THIRD_PARTY_LICENSES/Star7-MIT.txt).

Files in `src/h3_turing/attention/csrc/third_party/comfy_kitchen_sage/` derive from ComfyUI comfy-kitchen and SageAttention. They retain their source copyright/SPDX headers, including SageAttention and NVIDIA attribution. The Apache-2.0 text is preserved in that directory. The original source states SageAttention revision `d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5` and describes removal of torch extension dependencies and flattened includes. The SM75 FP16-derived portion also has its retained Apache-2.0 text in `THIRD_PARTY_LICENSES/SageAttention-Apache-2.0.txt`.

The research VAE scheduler mentions independent temporal scheduling in Raylight revision `e82cad3`; it does not vendor Raylight or add a Ray/FSDP dependency. Source-pinned CK adapters derive functions at runtime from the user's installed dependency; the complete CK deployment is not bundled.

PyTorch, Triton, ComfyUI, comfy-kitchen, CUDA and model files are external dependencies governed by their respective licenses. This notice records dependency attribution; it does not assign a blanket license to other research snapshots in this release.
