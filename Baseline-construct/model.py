# from transformers import BertPreTrainedModel,BertModel,AlbertModel,AlbertPreTrainedModel,RobertaModel,XLNetPreTrainedModel,XLNetModel

from transformers import BertPreTrainedModel,BertModel,RobertaModel,XLNetPreTrainedModel,XLNetModel
from transformers.modeling_utils import SequenceSummary
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import torch
import torch.nn as nn
import pdb


# IR-CSQA
class BertForMultipleChoice(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=False,
    ):
        
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        pooled_output = outputs[1]
        # pdb.set_trace()
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        outputs = (reshaped_logits,)  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)

class SequenceSummaryLayer(nn.Module):
    def __init__(self,hidden_size,summary_layers):
        super().__init__()
        self.summary_layers = summary_layers
        self.linear = nn.Linear(hidden_size * summary_layers, hidden_size)
        # do pooler just as transformers did
        self.pooler = nn.Linear(hidden_size, hidden_size)
        self.pooler_activation = nn.Tanh()
    def forward(self, x):
        stacked_hidden_states = torch.stack(list(x[-self.summary_layers:]),dim = -2)
        # print(stacked_hidden_states.shape)
        stacked_hidden_states = stacked_hidden_states[:,0]
        # pdb.set_trace()
        concat_hidden_states = stacked_hidden_states.view(stacked_hidden_states.shape[0],stacked_hidden_states.shape[-2]*stacked_hidden_states.shape[-1])
        resized_hidden_states = self.linear(concat_hidden_states)
        pooled_hidden_states = self.pooler_activation(self.pooler(resized_hidden_states))
        return pooled_hidden_states



