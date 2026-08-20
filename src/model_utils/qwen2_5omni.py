import torch
from transformers import LogitsProcessor

class ForceTokensProcessor(LogitsProcessor):
    """Force the model to generate an exact token sequence, then EOS.

    Used to make the Thinker output a specific text so the Talker
    produces speech for exactly that text and nothing else.
    """

    def __init__(self, force_token_ids: list[int], eos_token_id: int, prompt_len: int):
        self.force_token_ids = force_token_ids
        self.eos_token_id = eos_token_id
        self.prompt_len = prompt_len

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        step = input_ids.shape[1] - self.prompt_len
        scores = torch.full_like(scores, float("-inf"))
        if step < len(self.force_token_ids):
            scores[:, self.force_token_ids[step]] = 0.0
        else:
            scores[:, self.eos_token_id] = 0.0
        return scores