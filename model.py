from transformers import RobertaForSequenceClassification, RobertaConfig
from transformers import AutoTokenizer
import torch
import torch.nn as nn
"""
  Model
"""


class roberta_mnli_typing(nn.Module):
    def __init__(self):
        super(roberta_mnli_typing, self).__init__()
        self.roberta_module = RobertaForSequenceClassification.from_pretrained("roberta-large-mnli")
        self.config = RobertaConfig.from_pretrained("roberta-large-mnli")

    def forward(self, input_ids, attention_mask):
        roberta_output = self.roberta_module(input_ids, attention_mask)
        res = nn.functional.softmax(roberta_output.logits, dim=-1)
        return res