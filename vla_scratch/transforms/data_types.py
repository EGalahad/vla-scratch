import torch
import jaxtyping as at
from tensordict import TensorClass, TensorDict


# In the future, tokenized prompt will also include multi-task instructions and their answers.
# In that case, a causal mask should be applied to the question answering part and a loss mask will indicate the region to compute loss on.


# the dataset is responsible for preparing the generation prompt and answer.
# currently only support for Qwen3VL bbox detection.


class Observation(TensorClass):
    images: at.UInt8[torch.Tensor, "*batch num_cam 3 height width"]
    image_masks: at.Bool[torch.Tensor, "*batch num_cam 1"]
    state: at.Float[torch.Tensor, "*batch state_history state_dim"]
    task: str
    generation_prompt: str
    generation_answer: str
    policy_input: TensorDict = None  # Dynamic field for policy-specific inputs


class ActionChunk(TensorClass):
    actions: at.Float[torch.Tensor, "*batch action_horizon action_dim"]


class DataSample(TensorClass):
    observation: Observation
    action_chunk: ActionChunk = None
