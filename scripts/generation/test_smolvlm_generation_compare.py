from dataclasses import replace
from pathlib import Path

import torch
from transformers.cache_utils import DynamicCache

from vla_scratch.datasets.config import DataConfig
from vla_scratch.helpers.data import create_dataset
from vla_scratch.policies.pi.config import pi_smolvlm_config
from vla_scratch.policies.pi.policy import PiPolicy
from vla_scratch.transforms.data_types import DataSample


def make_configs():
    policy_cfg = replace(
        pi_smolvlm_config,
        model_id="HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        vlm_type="SmolVLMForConditionalGeneration",
        transforms=[
            {
                "_target_": "vla_scratch.policies.modules.vlm_bridge.smolvlm.processor.SmolVLMProcessor",
                "processor_class": "SmolVLMProcessor",
                "model_id": "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
                "max_length": 180,
                "padding": "max_length",
                "image_size_longest_edge": 512,
                "max_image_size_longest_edge": 512,
            }
        ],
    )
    policy_cfg.action_dim = 8
    policy_cfg.state_dim = 8

    data_cfg = DataConfig(
        _target_="dummy_dataset.DummyDataset",
        input_transforms=[],
        output_transforms=[],
        norm_stats_path=None,
    )
    return data_cfg, policy_cfg


def decode_tokens(processor, token_ids: torch.Tensor) -> str:
    """Helper to decode a single token tensor."""
    if hasattr(processor, "decode"):
        return processor.decode(token_ids[0])
    return processor.tokenizer.decode(token_ids[0], skip_special_tokens=False)


def autoregressive_decode(
    *,
    model,
    processor,
    past_key_values,
    cache_position: torch.Tensor,
    attention_mask: torch.Tensor,
    next_token: torch.Tensor,
    max_new_tokens: int = 256,
) -> str:
    """Manual decoding loop mirroring test_qwen3vl_generation.decode."""
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

            # Manually build decode inputs to keep shapes 2D for SmolVLM.
            if next_token.ndim == 1:
                next_token = next_token.unsqueeze(0)
            decode_outputs = model(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                use_cache=True,
            )
            past_key_values = decode_outputs.past_key_values
            cache_position = cache_position + 1

            next_token = decode_outputs.logits[:, -1, :].argmax(
                dim=-1, keepdim=True
            )
            if next_token.item() == eos_token_id:
                break

    return decode_tokens(processor, torch.cat(generated_ids, dim=-1))


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cfg, policy_cfg = make_configs()

    dataset = create_dataset(
        data_cfg,
        policy_cfg,
        skip_norm_stats=True,
        add_noise=False,
    )
    sample: DataSample = dataset[0][0].to(device)
    observation = sample.observation

    with torch.device(device):
        policy: "PiPolicy" = policy_cfg.instantiate()

    # Bridge path (patched forwards)
    with torch.inference_mode():
        _, vlm_outputs, _ = policy.vlm_bridge.encode(observation.unsqueeze(0))
        hidden_states_bridge = vlm_outputs.last_hidden_state
        prefix_pad_masks = vlm_outputs.prefix_pad_masks
        key_states = vlm_outputs.key_states
        value_states = vlm_outputs.value_states
        logits = policy.vlm_bridge.causal_model.lm_head(
            hidden_states_bridge[:, -1:, :]
        )
        next_tok_bridge = logits[:, -1, :].argmax(dim=-1, keepdim=True)

    cache_pos_bridge = torch.tensor([prefix_pad_masks.shape[1]], device=device)
    attn_mask_bridge = prefix_pad_masks
    kv_cache_list = [
        (k, v)
        for k, v in zip(key_states.unbind(dim=1), value_states.unbind(dim=1))
    ]
    kv_cache_bridge = DynamicCache.from_legacy_cache(tuple(kv_cache_list))

    # Baseline path using HF forward (original implementations)
    policy_td = observation.policy_input
    input_ids_b = policy_td.input_ids.unsqueeze(0)
    attention_mask = policy_td.attention_mask.unsqueeze(0)
    hf_inputs = {
        "input_ids": input_ids_b,
        "attention_mask": attention_mask,
        "pixel_values": policy_td.pixel_values,
        "use_cache": True,
    }
    with torch.inference_mode():
        hf_outputs = policy.vlm_bridge.causal_model(**hf_inputs)

    kv_prefill = hf_outputs.past_key_values
    cache_pos_prefill = torch.tensor(
        [hf_inputs["input_ids"].shape[1]], device=device
    )
    attn_mask_prefill = hf_inputs["attention_mask"]
    next_tok_prefill = hf_outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    # Compare caches
    bridge_legacy = kv_cache_bridge.to_legacy_cache()
    hf_legacy = kv_prefill.to_legacy_cache()
    for idx, ((bk, bv), (hk, hv)) in enumerate(zip(bridge_legacy, hf_legacy)):
        k_close = torch.allclose(bk, hk, atol=1e-3, rtol=1e-3)
        v_close = torch.allclose(bv, hv, atol=1e-3, rtol=1e-3)
        print(
            f"Layer {idx}: keys match={k_close}, values match={v_close}, shape={bk.shape}"
        )

    print(
        f"Next token bridge vs prefill: {next_tok_bridge.item()} vs {next_tok_prefill.item()}"
    )
    print(
        f"Attention mask equal: {torch.equal(attn_mask_bridge, attn_mask_prefill)}"
    )
    print(
        f"Cache position bridge vs prefill: {cache_pos_bridge.item()} vs {cache_pos_prefill.item()}"
    )

    decoded_bridge = decode_tokens(
        policy.vlm_bridge.processor, torch.cat([next_tok_bridge], dim=-1)
    )
    decoded_prefill = decode_tokens(
        policy.vlm_bridge.processor, torch.cat([next_tok_prefill], dim=-1)
    )
    print(f"Decoded bridge token: {decoded_bridge}")
    print(f"Decoded prefill token: {decoded_prefill}")

    decoded_msg_bridge = autoregressive_decode(
        model=policy.vlm_bridge.causal_model,
        processor=policy.vlm_bridge.processor,
        past_key_values=kv_cache_bridge,
        cache_position=cache_pos_bridge,
        attention_mask=attn_mask_bridge.clone(),
        next_token=next_tok_bridge.clone(),
        max_new_tokens=128,
    )
    decoded_msg_prefill = autoregressive_decode(
        model=policy.vlm_bridge.causal_model,
        processor=policy.vlm_bridge.processor,
        past_key_values=kv_prefill,
        cache_position=cache_pos_prefill.clone(),
        attention_mask=attn_mask_prefill.clone(),
        next_token=next_tok_prefill.clone(),
        max_new_tokens=128,
    )
    print(f"Decoded bridge message:\n{decoded_msg_bridge}")
    print(f"Decoded prefill message:\n{decoded_msg_prefill}")

    # Restore patched forwards as default for the rest of the session.
    # replace_smolvlm_forward()


if __name__ == "__main__":
    main()
