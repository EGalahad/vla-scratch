import torch
from dataclasses import replace
from transformers.cache_utils import DynamicCache
import sys
from pathlib import Path

from vla_scratch.datasets.config import DataConfig
from vla_scratch.helpers.data import create_dataset
from vla_scratch.policies.pi.config import pi_paligemma_config
from vla_scratch.policies.modules.action_expert.cross_attention_dit import (
    DiTConfig,
)
from vla_scratch.policies.modules.vlm_bridge.qwen.utils import (
    restore_qwen3vl_forward,
)

from vla_scratch.policies.pi.policy import PiPolicy
from vla_scratch.transforms.data_types import DataSample

# Allow importing helper functions when running as a script (`python tests/...`)
_THIS_DIR = Path(__file__).resolve().parent
sys.path.append(str(_THIS_DIR))
from test_qwen3vl_generation import prefill, decode


def make_configs():
    policy_cfg = replace(
        pi_paligemma_config,
        model_id="Qwen/Qwen3-VL-2B-Instruct",
        vlm_type="Qwen3VLForConditionalGeneration",
        transforms=[
            {
                "_target_": "vla_scratch.policies.modules.vlm_bridge.qwen.processor.QwenProcessor",
                "processor_class": "Qwen3VLProcessor",
                "model_id": "Qwen/Qwen3-VL-2B-Instruct",
                "max_length": 256,
                "padding": "max_length",
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


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cfg, policy_cfg = make_configs()

    # Build a transformed dataset that runs QwenProcessor through create_dataset()
    dataset = create_dataset(
        data_cfg,
        policy_cfg,
        skip_norm_stats=True,
        add_noise=False,
    )
    sample: DataSample = dataset[0][0].to(device)

    with torch.device(device):
        policy: "PiPolicy" = policy_cfg.instantiate()

    observation = sample.observation

    # restore_qwen3vl_forward()
    # Policy path: encode_prefix through our bridge
    with torch.inference_mode():
        _, vlm_outputs, _ = policy.vlm_bridge.encode(observation.unsqueeze(0))
        # hidden_states_bridge, prefix_pad_masks, kv_cache_list, encoder_hidden_states = policy.encode_prefix(observation.unsqueeze(0))
        hidden_states_bridge = vlm_outputs.last_hidden_state
        prefix_pad_masks = vlm_outputs.prefix_pad_masks
        key_states = vlm_outputs.key_states
        value_states = vlm_outputs.value_states
        logits = policy.vlm_bridge.causal_model.lm_head(
            hidden_states_bridge[:, -1:, :]
        )
        next_tok_bridge = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    restore_qwen3vl_forward()

    cache_pos_bridge = torch.tensor([prefix_pad_masks.shape[1]], device=device)
    attn_mask_bridge = prefix_pad_masks
    kv_cache_list = [
        (k, v)
        for k, v in zip(key_states.unbind(dim=1), value_states.unbind(dim=1))
    ]
    kv_cache_bridge = DynamicCache.from_legacy_cache(tuple(kv_cache_list))

    # Baseline via HF prefill using the same model + processor
    (
        hidden_states_prefill,
        kv_prefill,
        cache_pos_prefill,
        attn_mask_prefill,
        next_tok_prefill,
    ) = prefill(
        image=observation.images[0],
        prompt=observation.task,
        model=policy.vlm_bridge.causal_model,
        processor=policy.vlm_bridge.processor,
        device=device,
    )

    # Compare
    bridge_legacy = kv_cache_bridge.to_legacy_cache()
    hf_legacy = kv_prefill.to_legacy_cache()
    matches = []
    for idx, ((bk, bv), (hk, hv)) in enumerate(zip(bridge_legacy, hf_legacy)):
        k_close = torch.allclose(bk, hk, atol=1e-3, rtol=1e-3)
        v_close = torch.allclose(bv, hv, atol=1e-3, rtol=1e-3)
        matches.append((k_close, v_close))
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

    decoded_bridge = policy.vlm_bridge.processor.decode(
        torch.cat([next_tok_bridge], dim=-1)[0]
    )
    decoded_prefill = policy.vlm_bridge.processor.decode(
        torch.cat([next_tok_prefill], dim=-1)[0]
    )
    print(f"Decoded bridge token: {decoded_bridge}")
    print(f"Decoded prefill token: {decoded_prefill}")

    decoded_msg_bridge = decode(
        model=policy.vlm_bridge.causal_model,
        processor=policy.vlm_bridge.processor,
        past_key_values=kv_cache_bridge,
        cache_position=cache_pos_bridge,
        attention_mask=attn_mask_bridge.clone(),
        next_token=next_tok_bridge.clone(),
        max_new_tokens=128,
    )
    decoded_msg_prefill = decode(
        model=policy.vlm_bridge.causal_model,
        processor=policy.vlm_bridge.processor,
        past_key_values=kv_prefill,
        cache_position=cache_pos_prefill.clone(),
        attention_mask=attn_mask_prefill.clone(),
        next_token=next_tok_prefill.clone(),
        max_new_tokens=128,
    )
    print(f"Decoded bridge message:\n{decoded_msg_bridge}")
    print(f"Decoded HF prefill message:\n{decoded_msg_prefill}")


if __name__ == "__main__":
    main()
