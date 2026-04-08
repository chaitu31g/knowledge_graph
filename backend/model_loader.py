import os
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

# ── Memory optimisation: prevent fragmentation on T4/L4 ───────
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

_model = None
_processor = None

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
# ↑ 3B fits on T4 (16 GB) with 4-bit quant (~2 GB weights).
# For L4/A100 (24 GB+) you can use "Qwen/Qwen2.5-VL-7B-Instruct".

# Pixel caps for the vision encoder.
# 512 * 28 * 28 = ~401k px — safe upper bound for T4 16 GB with 4-bit model
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 512 * 28 * 28


def load_model():
    global _model, _processor
    if _model is not None:
        return _model, _processor

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print(f"[model_loader] Loading {MODEL_ID} with 4-bit quantization...")
    _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,        # ← loads layer-by-layer, NOT full model to RAM first
    )
    _model.eval()

    # min_pixels / max_pixels are enforced by the processor before tokenisation
    _processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
    print("[model_loader] Model loaded successfully.")
    return _model, _processor


def run_inference(messages: list, max_new_tokens: int = 2048) -> str:
    """Run a single vision or text inference call with memory guards."""
    model, processor = load_model()

    torch.cuda.empty_cache()  # free fragmented cache before each call

    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text_input],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    input_ids_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,     # greedy — fastest and most deterministic
            temperature=None,    # explicitly unset: suppresses the 'invalid flag' warning
        )

    # Free input tensors immediately to reclaim VRAM
    del inputs
    torch.cuda.empty_cache()

    # Slice off the prompt tokens — only decode the new tokens
    new_tokens = output_ids[:, input_ids_len:]
    response = processor.batch_decode(
        new_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return response.strip()
