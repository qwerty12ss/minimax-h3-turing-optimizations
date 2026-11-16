"""Generate the two portable API workflows; contains no external images or credentials."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def graph(width, height, length, vae_mode, name):
    nodes = {}
    def add(index, kind, **inputs):
        nodes[str(index)] = dict(class_type=kind, inputs=inputs)
    add(1, "MiniMaxH3FP16LoaderStar7", unet_name="minimax_h3_fl2va_pruned_int8_convrot.safetensors")
    add(2, "MiniMaxH3SigmaShift", model=["1", 0], shift_video=12.0, shift_audio=3.0)
    add(3, "MiniMaxH3PDDAccApply", model=["2", 0], pdd_file="MiniMax-H3-FL2VA-Acc-8Step.safetensors", nfe="8",
        lora_strength=1.0, head_strength=1.0, on_off_grid="error", partition_check="error")
    add(4, "CLIPLoader", clip_name="qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", type="minimax", device="default")
    add(5, "VAELoader", vae_name="minimax_h3_video_vae_fp16.safetensors")
    add(6, "VAELoader", vae_name="minimax_h3_audio_vae_fp32.safetensors")
    add(7, "MiniMaxH3ActivationChunkStar7", model=["3", 0], chunk_tokens=8192, auto_halve_on_oom=True, verbose=True,
        mlp_chunk_tokens=8192, disable_dynamic_prefetch="auto", qkv_chunk_tokens=8192, out_proj_chunk_tokens=4096,
        reuse_mlp_weights=True, attention_backend="comfy_kitchen_int8")
    add(8, "H3TuringApply", model=["7", 0], preset="optimized", sol_tau=1.0, reserve_gib=18.0)
    add(9, "MiniMaxH3ImageToVideo", clip=["4", 0], vae=["5", 0],
        prompt="A red rectangular block stands on a blue gray surface. The camera stays still. Soft daylight. Quiet room tone, no speech.",
        width=width, height=height, length=length, first_frame=["19", 0])
    add(10, "RandomNoise", noise_seed=42)
    add(11, "BasicGuider", model=["8", 0], conditioning=["9", 0])
    add(12, "KSamplerSelect", sampler_name="euler")
    add(13, "H3TuringVAE", vae=["5", 0], mode=vae_mode)
    add(14, "SamplerCustomAdvanced", noise=["10", 0], guider=["11", 0], sampler=["12", 0], sigmas=["3", 1], latent_image=["9", 1])
    add(15, "VAEDecodeTiled", samples=["14", 0], vae=["13", 0], tile_size=256, overlap=64, temporal_size=32, temporal_overlap=8)
    add(16, "VAEDecodeAudio", samples=["14", 0], vae=["6", 0])
    add(17, "CreateVideo", images=["20", 0], audio=["20", 1], fps=24.0, bit_depth=8)
    add(18, "SaveVideo", video=["17", 0], filename_prefix=f"video/{name}", format="mp4", **{"format.codec": "h264"})
    add(19, "H3TuringTestImage", width=width, height=height)
    add(20, "H3TuringValidateFrames", images=["15", 0], audio=["16", 0])
    return nodes


def main():
    folder = ROOT / "workflows"
    folder.mkdir(exist_ok=True)
    for name, size in [("h3_turing_smoke", (640, 352, 124, "serial")), ("h3_turing_8s", (960, 512, 192, "temporal_hybrid"))]:
        (folder / f"{name}_api.json").write_text(json.dumps(graph(*size, name), indent=2) + "\n", newline="\n")


if __name__ == "__main__":
    main()
