# Policies Folder Layout

## Files at a Glance

| Path                     | Description                                 |
|--------------------------|---------------------------------------------|
| `base.py`                | `BasePolicy` interface.                     |
| `config.py`              | Hydra `PolicyConfig` definitions.           |
| `modules/action_expert/` | Flow matching action head.                  |
| `modules/vlm_bridge/`    | VLM bridge for Paligemma, Qwen and SmolVLM. |
| `pi/`                    | Policy implementation (config + model).     |
| `utils/`                 | Training utilities.                         |


## Core Execution Flow

1. Policies expose `encode_prefix` and `predict_suffix`; `compute_loss` and `sample_actions` compose those primitives during training and serving.
2. `encode_prefix` calls the chosen VLM bridge to produce [`VLMOutputs`](../../vla_scratch/policies/modules/vlm_bridge/data_types.py).
3. Each VLM bridge expects `Observation.policy_input` to be prepared by its matching processor (e.g., `QwenBridge` with `QwenProcessor`), ensuring model-specific tensors are present before the forward pass.
4. `predict_suffix` consumes `VLMOutputs`, denoises Gaussian noises to predict flow matching velocity via the action expert head.
5. Layer-wise FSDP sharding and gradient checkpointing are applied with `apply_fsdp` through helpers in `utils/training.py`.

## Optimizations


<table>
<tr>
<td width="70%" valign="top">
<h3>Reduction #1: Keep dynamic shape metadata on CPU</h3>

During encoding the image features, position embeddings are interpolated to match the spatial dimensions of each image and added to the input image tokens. This requires creating tensors whose shapes depend on the input image size.

In the original implementation, everything is converted to tensors after the dataloader output, then during the forward pass the shape metadata is read from CUDA tensors, causing frequent synchronizations. We instead store the shape metadata (height and width) as CPU integers in the dataloader output. That keeps it on CPU when calling `batch.to(device)` and avoids these syncs.
<pre><code class="language-python"># Reading dynamic shape from CUDA tensors
def pos_embed_interpolate(self, grid_thw):
    grid_ts, grid_hs, grid_ws = grid_thw.unbind(1)
    for t, h, w in zip(grid_ts, grid_hs, grid_ws):
        h_idxs = torch.arange(0, h, self.step)
        w_idxs = torch.arange(0, w, self.step)
</code></pre>

Related code in HF Transformers:
1. [rot_pos_emb](https://github.com/huggingface/transformers/blob/314f10929a2215b74c2ad6ecf7b2f380c9b7468a/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L643)
2. [fast_pos_embed_interpolate](https://github.com/huggingface/transformers/blob/314f10929a2215b74c2ad6ecf7b2f380c9b7468a/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L682)

</td>
<td width="30%" valign="center">
<img src="../../assets/policies/performance-reduction-1.png" width="100%" alt="Reduction #1">
<br>
<em>Reduction #1: stop the img_thw from being moved to GPU to read desired tensor shapes without sync.</em>
</td>
</tr>
</table>

<table>
<tr>
<td width="70%" valign="top">
<h3>Reduction #2: Move "heavy lifting" to dataloader</h3>

The 3D-RoPE index computation in Qwen3-VL scans inputs token-by-token with conditional logic. When performed on CUDA tensors, this introduces synchronization at fine granularity.

Since this computation doesn't depend on any model activations, we precompute these indices in the dataloader. Multiprocessing workers handle the preprocessing in parallel, shifting the heavy lifting off the forward pass and overlapping it with model computation.

<pre><code class="language-python"># Conditional flows depend on CUDA tensors
for token in input_ids:
    if token == vision_start_token_id:
        ...
    if token == image_token_id:
        ...
</code></pre>

Related code in HF Transformers:
1. [get_rope_index](https://github.com/huggingface/transformers/blob/314f10929a2215b74c2ad6ecf7b2f380c9b7468a/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L993)
</td>
<td width="30%" valign="center">
<img src="../../assets/policies/performance-reduction-2.png" width="100%" alt="Reduction #2">
<br>
<em>Reduction #2: move conditional logic to dataloader multiprocessing workers.</em>
</td>
</tr>
</table>

<table>
<tr>
<td colspan="2" valign="top">
<h3>Reduction #3: Make the forward pass shape-stable</h3>

The DeepStack operation that fuses image embeddings into the hidden states of LLM layers also introduces tensors whose shapes depend on values in CUDA tensors, which triggers synchronization.
<pre><code class="language-python"># HuggingFace implementation (dynamic shaped tensor)
local_delta = hidden_states[visual_pos_masks, :].clone() + visual_embeds
hidden_states[visual_pos_masks, :] = local_delta
</code></pre>

We rewrite the same operation with a clever <code>masked_scatter</code> operator to preserve static tensor shapes.
<pre><code class="language-python"># Optimized version (shape-stable)
global_delta = torch.zeros_like(hidden_states)
global_delta.masked_scatter_(visual_pos_masks.unsqueeze(-1), visual_embeds)
hidden_states = hidden_states + global_delta
</code></pre>

Related code in HF Transformers:
1. [_deepstack_process](https://github.com/huggingface/transformers/blob/314f10929a2215b74c2ad6ecf7b2f380c9b7468a/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L915)
</td>
</tr>
</table>

