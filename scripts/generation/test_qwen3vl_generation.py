import numpy as np
import torch
from typing import Dict, List
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor


def build_demo_image(height: int = 384, width: int = 384) -> Image.Image:
    """Create a deterministic RGB gradient so the script works without external assets."""
    red = np.tile(np.linspace(0, 255, width, dtype=np.uint8), (height, 1))
    green = red[::-1]
    blue = np.full_like(red, 128, dtype=np.uint8)
    stacked = np.stack([red, green, blue], axis=-1)
    return Image.fromarray(stacked, mode="RGB")


def prefill(
    *,
    image: Image.Image,
    prompt: str,
    model: Qwen3VLForConditionalGeneration,
    processor: Qwen3VLProcessor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prefill pass that returns cache, cache_position, attention_mask, and next token."""
    content: List[Dict] = [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]
    messages = [{"role": "user", "content": content}]

    encoded = processor.apply_chat_template(
        [messages],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        text_kwargs={
            "max_length": 256,
            "truncation": True,
            "padding": "max_length",
            "return_tensors": "pt",
        },
    )
    model_inputs = {
        k: v.to(device) if torch.is_tensor(v) else v for k, v in encoded.items()
    }

    with torch.inference_mode():
        outputs = model(**model_inputs, use_cache=True)

    hidden_states = outputs.hidden_states
    past_key_values = outputs.past_key_values
    cache_position = torch.tensor(
        [model_inputs["input_ids"].shape[1]], device=device
    )
    attention_mask = model_inputs["attention_mask"]
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    return (
        hidden_states,
        past_key_values,
        cache_position,
        attention_mask,
        next_token,
    )


def decode(
    *,
    model: Qwen3VLForConditionalGeneration,
    processor: Qwen3VLProcessor,
    past_key_values,
    cache_position: torch.Tensor,
    attention_mask: torch.Tensor,
    next_token: torch.Tensor,
    max_new_tokens: int = 256,
) -> str:
    """Manual autoregressive decoding loop with explicit KV cache updates."""
    generated_ids: list[torch.Tensor] = []
    eos_token_id = model.generation_config.eos_token_id
    device = attention_mask.device

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            generated_ids.append(next_token)
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (attention_mask.shape[0], 1),
                        device=device,
                        dtype=attention_mask.dtype,
                    ),
                ],
                dim=1,
            )

            decode_inputs = model.prepare_inputs_for_generation(
                input_ids=next_token,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                cache_position=cache_position,
            )
            decode_outputs = model(**decode_inputs)
            past_key_values = decode_outputs.past_key_values
            cache_position = cache_position + 1

            next_token = decode_outputs.logits[:, -1, :].argmax(
                dim=-1, keepdim=True
            )
            if next_token.item() == eos_token_id:
                break

    return processor.decode(torch.cat(generated_ids, dim=-1)[0])


def main() -> None:
    model_id = "Qwen/Qwen3-VL-2B-Instruct"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = Qwen3VLProcessor.from_pretrained(model_id)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"

    model = (
        Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            # dtype=dtype,
            attn_implementation="sdpa",
            trust_remote_code=True,
        )
        .eval()
        .to(device)
    )

    image = build_demo_image()
    prompt = "Describe the image in English."
    hidden_states, kv_cache, cache_position, attention_mask, next_token = (
        prefill(
            image=image,
            prompt=prompt,
            model=model,
            processor=processor,
            device=device,
        )
    )
    decoded = decode(
        model=model,
        processor=processor,
        past_key_values=kv_cache,
        cache_position=cache_position,
        attention_mask=attention_mask,
        next_token=next_token,
    )
    print(f"Generated description: {decoded}")


if __name__ == "__main__":
    main()
