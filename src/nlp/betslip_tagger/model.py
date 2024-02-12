from typing import Optional, Union, Tuple
from dataclasses import dataclass
import torch
from torch import nn
import pandas as pd
import numpy as np
import transformers
from transformers import BertForTokenClassification, BertModel
from transformers.modeling_outputs import ModelOutput


@dataclass
class BetBERTOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    coarse_logits: torch.FloatTensor = None
    fine_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class BetBERT(BertForTokenClassification):
    """
    Regular BertForTokenClassification but with two separate prediction layers for the coarse and fine label types.
    """

    def __init__(self, config, num_labels_coarse, num_labels_fine):
        super().__init__(config)
        self.num_labels_coarse = num_labels_coarse
        self.num_labels_fine = num_labels_fine

        self.bert = BertModel(config, add_pooling_layer=False)
        classifier_dropout = config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        self.dropout = nn.Dropout(classifier_dropout)
        self.coarse_classifier = nn.Linear(config.hidden_size, self.num_labels_coarse)
        self.fine_classifier = nn.Linear(config.hidden_size, self.num_labels_fine)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BetBERTOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        coarse_logits = self.coarse_classifier(sequence_output)
        fine_logits = self.fine_classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(coarse_logits.view(-1, self.num_labels_coarse), labels.view(-1))
            loss += loss_fct(fine_logits.view(-1, self.num_labels_fine), labels.view(-1))

        if not return_dict:
            output = (coarse_logits,) + (fine_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return BetBERTOutput(
            loss=loss,
            coarse_logits=coarse_logits,
            fine_logits=fine_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
