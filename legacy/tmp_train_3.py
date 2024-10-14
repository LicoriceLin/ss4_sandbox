# %%
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
            
model_name='facebook/esm2_t12_35M_UR50D'
config=EsmConfig.from_pretrained(model_name)
config.num_mhlabels=[3,3,4]
config.num_head_layers=1
model=EsmTokenMhClassification(config)

# %%
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict
import random
from datasets import Dataset
from transformers.models.esm.modeling_esm import create_position_ids_from_input_ids as create_position_ids
from torch.utils.data import DataLoader
from functools import partial
seed=42
random_generator=random.Random(seed)
numpy_generator=np.random.default_rng(seed)
torch_generator=torch.Generator('cpu')
torch_generator.manual_seed(seed)

tokenizer:EsmTokenizer=EsmTokenizer.from_pretrained(model_name)
NUM_CLASS=36
ITEM=Dict[str,Union[str,List[float]]]

def output_map():
    mapping = {}
    for i, char in enumerate(range(ord('A'), ord('Z') + 1)):
        mapping[chr(char)] = i

    start_ord = ord('a')
    end_ord = start_ord + NUM_CLASS -26 -1  
    for i, char in enumerate(range(start_ord, end_ord + 1), start=26):
        mapping[chr(char)] = i
    mapping['*']=-100
    return mapping
OUTPUT_MAP=output_map()

def map_fn(examples):
    tokenized_inputs = tokenizer(examples["seq"], truncation=False,padding=True)
    examples.update(tokenized_inputs)
    examples['mu_ids']=[-100]+[OUTPUT_MAP.get(i,-1) for i in examples['mu']]+[-100]
    return examples

def collate_fn(features:List[ITEM],max_length=800,train:bool=True,mask_rate=0.1):
    ret=defaultdict(list)
    for feature in features:
        del_len=len(feature['input_ids'])-max_length
        if del_len<0:
            def padding(feature,del_len):
                feature['input_ids']=feature['input_ids']+[1]*(-del_len)
                feature['attention_mask']=feature['attention_mask']+[0]*(-del_len)
                feature['mu_ids']=feature['mu_ids']+[-100]*(-del_len)
                feature['idx_b']=0
            padding(feature,del_len)
        elif del_len>0:
            idx_b=random_generator.randint(0,del_len)
            truncation=lambda x:[x[0]]+x[idx_b+1:idx_b+max_length-1]+[x[-1]]
            feature['input_ids']=truncation(feature['input_ids'])
            feature['attention_mask']=truncation(feature['attention_mask'])
            feature['mu_ids']=truncation(feature['mu_ids'])
            feature['idx_b']=idx_b+1
        else:
            feature['idx_b']=0
        
        for k in feature.keys():
            ret[k].append(feature[k])
            
    for k in ['input_ids','attention_mask','mu_ids','idx_b']:
        ret[k]=torch.tensor(ret[k],dtype=torch.long)
        
    ret['position_ids']=create_position_ids(
        ret['input_ids'],ret['attention_mask'],ret['idx_b'].reshape(-1,1))
    if train:
        _=ret['input_ids'][:,1:-1]
        _[torch.rand_like(_,dtype=torch.float16)>1-mask_rate]=tokenizer.mask_token_id
    return ret
        
mu=pd.read_pickle('data/mu.pkl')
mu=mu[mu['seq'].apply(lambda x:len(x))>0].iloc[:200]
dataset=Dataset.from_pandas(mu).map(map_fn)
loader=DataLoader(dataset,batch_size=3,collate_fn=partial(collate_fn,max_length=600,train=True,mask_rate=0.1))

# %%
for batch in loader:
    break

#%%
device=3
model.to(device)
pred=model.esm(batch['input_ids'].to(device),
                   batch['attention_mask'].to(device),
                   position_ids=batch['position_ids'].to(device))

# %%
for i,num_label in enumerate(model.num_mhlabels):
    neck=getattr(model,f'neck_{i+1}')
    head=getattr(model,f'head_{i+1}')
    break