import torch
from contextlib import contextmanager

from transformers.models.smolvlm.modeling_smolvlm import (
    SmolVLMModel,
    SmolVLMVisionEmbeddings,
)


def _smolvlm_inputs_merger_fast(
    self: SmolVLMModel,
    input_ids: torch.LongTensor,
    inputs_embeds: torch.Tensor,
    image_hidden_states: torch.Tensor,
) -> torch.Tensor:
    """GPU-friendly merger that avoids boolean indexing syncs."""
    torch.cuda.nvtx.range_push("inputs_merger_fast")

    # Identify where image tokens live in the sequence.
    image_mask = input_ids == self.config.image_token_id

    # Early return if there is nothing to merge; keep on device to avoid host sync.
    batch_size, seq_len = image_mask.shape
    hidden_size = inputs_embeds.shape[-1]

    patch_size = image_hidden_states.shape[1]
    num_images = max(image_hidden_states.shape[0] // max(batch_size, 1), 1)
    image_states = image_hidden_states.view(
        batch_size, num_images * patch_size, hidden_size
    )

    # Compute the running index of image tokens, then map to (image_idx, local_idx) per position.
    token_offsets = torch.cumsum(image_mask.to(dtype=torch.int64), dim=1) - 1
    token_offsets = torch.where(
        image_mask,
        token_offsets,
        torch.zeros(1, device=image_mask.device, dtype=token_offsets.dtype),
    )
    image_idx = torch.div(token_offsets, patch_size, rounding_mode="floor")
    local_idx = token_offsets - image_idx * patch_size

    flat_idx = image_idx * patch_size + local_idx
    max_flat_idx = image_states.shape[1] - 1
    flat_idx = flat_idx.clamp(min=0, max=max_flat_idx)

    gather_idx = flat_idx.unsqueeze(-1).expand(batch_size, seq_len, hidden_size)
    scatter_values = torch.gather(image_states, 1, gather_idx)
    merged = torch.where(
        image_mask.unsqueeze(-1), scatter_values, inputs_embeds
    )

    torch.cuda.nvtx.range_pop()
    return merged


def _smolvlm_vision_embeddings_fast(
    self: SmolVLMVisionEmbeddings,
    pixel_values: torch.FloatTensor,
    patch_attention_mask: torch.BoolTensor,
) -> torch.Tensor:
    """
    Fast path assuming `patch_attention_mask` is all True.
    Matches the reference bucketization without Python loops.
    """
    patch_embeds = self.patch_embedding(pixel_values)
    embeddings = patch_embeds.flatten(2).transpose(1, 2)

    batch_size = pixel_values.shape[0]
    nb_patches_h, nb_patches_w = patch_attention_mask.shape[-2:]
    boundaries = torch.arange(
        1 / self.num_patches_per_side,
        1.0,
        1 / self.num_patches_per_side,
        device=pixel_values.device,
    )

    h_indices = torch.arange(
        nb_patches_h, device=pixel_values.device, dtype=pixel_values.dtype
    )
    w_indices = torch.arange(
        nb_patches_w, device=pixel_values.device, dtype=pixel_values.dtype
    )
    fractional_coords_h = h_indices / nb_patches_h * (1 - 1e-6)
    fractional_coords_w = w_indices / nb_patches_w * (1 - 1e-6)
    bucket_coords_h = torch.bucketize(
        fractional_coords_h, boundaries, right=True
    )
    bucket_coords_w = torch.bucketize(
        fractional_coords_w, boundaries, right=True
    )

    pos_ids = (
        bucket_coords_h[:, None] * self.num_patches_per_side + bucket_coords_w
    ).reshape(-1)
    position_ids = pos_ids.unsqueeze(0).expand(batch_size, -1)
    return embeddings + self.position_embedding(position_ids)


def _smolvlm_get_image_features_fast(
    self: SmolVLMModel,
    pixel_values: torch.FloatTensor,
    pixel_attention_mask: torch.FloatTensor | None = None,
) -> torch.Tensor:
    """
    Simplified image encoder forward that assumes inputs are already trimmed (no padding images).
    Avoids host-side scans over zero-filled padding.
    """
    batch_size, num_images, num_channels, height, width = pixel_values.shape
    pixel_values = pixel_values.to(dtype=self.dtype)
    pixel_values = pixel_values.view(
        batch_size * num_images, *pixel_values.shape[2:]
    )

    if pixel_attention_mask is None:
        pixel_attention_mask = torch.ones(
            size=[pixel_values.shape[i] for i in (0, 2, 3)],
            dtype=torch.bool,
            device=pixel_values.device,
        )
    else:
        pixel_attention_mask = pixel_attention_mask.view(
            batch_size * num_images, *pixel_attention_mask.shape[2:]
        )

    patch_size = self.config.vision_config.patch_size
    patches_subgrid = pixel_attention_mask.unfold(
        dimension=1, size=patch_size, step=patch_size
    )
    patches_subgrid = patches_subgrid.unfold(
        dimension=2, size=patch_size, step=patch_size
    )
    patch_attention_mask = (patches_subgrid.sum(dim=(-1, -2)) > 0).bool()

    image_hidden_states = self.vision_model(
        pixel_values=pixel_values, patch_attention_mask=patch_attention_mask
    )
    image_hidden_states = self.connector(image_hidden_states.last_hidden_state)
    return image_hidden_states


orig_inputs_merger = SmolVLMModel.inputs_merger
orig_vision_forward = SmolVLMVisionEmbeddings.forward
orig_get_image_features = SmolVLMModel.get_image_features
REPLACED = False


def replace_smolvlm_forward():
    global REPLACED
    if not REPLACED:
        SmolVLMModel.inputs_merger = _smolvlm_inputs_merger_fast
        SmolVLMVisionEmbeddings.forward = _smolvlm_vision_embeddings_fast
        SmolVLMModel.get_image_features = _smolvlm_get_image_features_fast
        REPLACED = True


def restore_smolvlm_forward():
    global REPLACED
    if REPLACED:
        SmolVLMModel.inputs_merger = orig_inputs_merger
        SmolVLMVisionEmbeddings.forward = orig_vision_forward
        SmolVLMModel.get_image_features = orig_get_image_features
        REPLACED = False


def is_smolvlm_forward_replaced() -> bool:
    return REPLACED


def replace_context():
    """Context manager to temporarily swap in the optimized forward methods."""

    @contextmanager
    def _ctx():
        if REPLACED:
            yield
        else:
            try:
                replace_smolvlm_forward()
                yield
            finally:
                restore_smolvlm_forward()

    return _ctx()
