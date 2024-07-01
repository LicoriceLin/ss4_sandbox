from transformers import EsmModel, EsmConfig, EsmTokenizer,EsmForTokenClassification,EsmPreTrainedModel
import torch
from torch import Tensor
from torch import nn
from typing import Optional,Union,Tuple,List
nn.TransformerEncoder
model_name='facebook/esm2_t12_35M_UR50D'
# %%
class EsmTokenMhClassification(EsmPreTrainedModel):
    'multi-head token classification'
    def __init__(self, config:EsmConfig):
        super().__init__(config)
        self.num_mhlabels:List[int] = config.num_mhlabels
        self.num_head_layers:int = getattr(config,'num_head_layers',1)
        self.esm = EsmModel(config, add_pooling_layer=False)
        self._init_classification_head()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.init_weights()

    def _init_classification_head(self):
        cfg=self.config
        encoder_layer=nn.TransformerEncoderLayer(
            d_model=cfg.hidden_size,
            nhead=cfg.num_attention_heads,
            dim_feedforward=cfg.intermediate_size,
            activation=cfg.hidden_act,
            batch_first=True
        )
        for i,num_label in enumerate(self.num_mhlabels):
            setattr(self,f'neck_{i+1}',nn.TransformerEncoder(
                encoder_layer=encoder_layer,num_layers=self.num_head_layers))
            setattr(self,f'head_{i+1}',nn.Linear(cfg.hidden_size, num_label))
            

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> List[Tensor]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        # raise NotImplementedError
        last_hidden_state = self.esm(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        ).last_hidden_state
        src_key_padding_mask=~attention_mask.bool()
        output=[]
        for i,num_label in enumerate(self.num_mhlabels):
            neck=getattr(self,f'neck_{i+1}')
            head=getattr(self,f'head_{i+1}')
            output.append(head(self.dropout(neck(
            last_hidden_state,src_key_padding_mask=src_key_padding_mask,is_causal=False)
                )))
        # return torch.concat(output,dim=-1)
        return output
            