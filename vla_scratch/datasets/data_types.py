import torch
import jaxtyping as at
from tensordict import TensorClass


# In the future, tokenized prompt will also include multi-task instructions and their answers.
# In that case, a causal mask should be applied to the question answering part and a loss mask will indicate the region to compute loss on.


class Observation(TensorClass):
    images: at.Float[torch.Tensor, "*batch num_cam 3 height width"]
    image_masks: at.Bool[torch.Tensor, "*batch num_cam 1"]
    state: at.Float[torch.Tensor, "*batch state_history state_dim"]
    tokenized_prompt: at.Int64[torch.Tensor, "*batch max_tokens"]
    tokenized_prompt_mask: at.Bool[torch.Tensor, "*batch max_tokens"]


class ActionChunk(TensorClass):
    actions: at.Float[torch.Tensor, "*batch action_horizon action_dim"]


class DataSample(TensorClass):
    observation: Observation
    action_chunk: ActionChunk
