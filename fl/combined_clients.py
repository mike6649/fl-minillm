import torch
from torch import nn
from transformers.modeling_outputs import CausalLMOutput
from transformers import GenerationMixin

class CombinedClients(nn.Module, GenerationMixin):
    def __init__(self, models: list):
        super().__init__()
        self.models = models
    
    def forward(self, **kwargs):
        logits_list = []
        for model in self.models:
            output = model(**kwargs)
            loss, logits = output.loss, output.logits
            logits_list.append(logits)
        logits_shape = logits.size()
        logits = torch.stack(logits_list, dim=0).sum(dim=0)

        assert logits.size() == logits_shape, f"{logits.size()=}, {logits_shape=}"

        return CausalLMOutput(loss, logits)
