import torch
from torch import nn
from transformers.modeling_outputs import CausalLMOutput
from transformers import GenerationMixin

class CombinedClients(nn.Model, GenerationMixin):
    def __init__(self, models: list):
        super().__init__()
        self.models = models
    
    def forward(self, **kwargs):
        losses = []
        logits_list = []
        for model in self.models:
            output = model(**kwargs)
            loss, logits = output.loss, output.logits
            losses.append(loss)
            logits_list.append(logits)
        loss_shape = loss.size()
        logits_shape = logits.size()
        loss = torch.stack(losses, dim=0).sum(dim=0)
        logits = torch.stack(logits_list, dim=0).sum(dim=0)

        assert loss.size() == loss_shape, f"{loss.size()=}, {loss_shape=}"
        assert logits.size() == logits_shape, f"{logits.size()=}, {logits_shape=}"

        return CausalLMOutput(loss, logits)
