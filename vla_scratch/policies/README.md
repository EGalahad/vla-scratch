# Policies Folder Layout

This directory wires datasets into trainable / deployable policies. Key components:

```
policies/
├── base.py                       # BasePolicy interface
├── config.py                     # Hydra-friendly PolicyConfig + factory helpers
├── utils/
│   ├── transformers.py           # Math helpers shared across policies (noise, rotary)
│   └── training.py               # FSDP + checkpoint helpers used during policy training
├── modules/
│   ├── action_expert/
│   │   └── dit.py                # Diffusion action head (DiT)
│   └── vlm_bridge/
│       ├── base.py               # Abstract VLM bridge contract
│       ├── paligemma/
│       │   ├── processor.py      # PaligemmaPolicyInput + transform
│       │   ├── bridge.py         # Text/image encoding + KV cache extraction
│       │   └── utils.py          # Low-level custom forwards
│       └── qwen/
│           ├── processor.py      # QwenPolicyInput + tokenizer transform
│           ├── bridge.py         # Qwen3-VL encoder bridge
│           └── utils.py          # Optimized rotary / visual helpers
└── pi/
    ├── config.py                 # Pi policy Hydra config (references transforms)
    ├── policy.py                 # PiPolicy implementation
    └── transforms/...            # (none—per-VLM transforms live with bridges)
```

### Base abstractions
- `BasePolicy` defines the contract (encode prefix, predict suffix, sample actions, optional `apply_fsdp`). Any new policy should implement this interface so training/serving scripts remain generic.
- `PolicyConfig` and the Hydra registrations live in `config.py`, allowing `policy=<name>` overrides.

### Modules
- `modules/action_expert` contains reusable action heads (currently a DiT-based diffusion expert). These modules are policy-agnostic and can be shared.
- `modules/vlm_bridge` hosts VLM-specific logic: each bridge has a processor (turns `DataSample` into a policy-specific TensorClass), the bridge itself (runs the VLM forward, capturing KV caches), and any custom utilities.

### Policy implementation
- `pi/` houses the Pi policy (config + model). The policy assembles a VLM bridge and action expert, and consumes the policy-specific tensor classes produced by the bridge processors.

### Utilities
- `utils/transformers.py` provides math helpers (noise sampling, rotary embeddings, sinusoidal time embeddings).
- `utils/training.py` provides FSDP sharding + gradient checkpoint utilities used by both policies and VLM bridges.

When adding a new policy:
1. Define its config under `policies/<name>/config.py` and register it with Hydra.
2. Implement the policy model (subclass `BasePolicy`). Reuse modules/bridges/utilities where possible.
3. Add any policy-specific processors under `modules/...` if they will be shared, or locally under the policy if truly bespoke.
