import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict,Union,List,Optional
import random
from datasets import Dataset as ds
from transformers.models.esm.modeling_esm import create_position_ids_from_input_ids as create_position_ids
from torch.utils.data import DataLoader,Dataset
from functools import partial
import torch
from transformers import PreTrainedTokenizer,EsmTokenizer
from collections.abc import Sequence
NUM_CLASS=36
ITEM=Dict[str,Union[str,List[float]]]

seed=42
random_generator=random.Random(seed)
numpy_generator=np.random.default_rng(seed)
torch_generator=torch.Generator('cpu')
torch_generator.manual_seed(seed)
model_name='facebook/esm2_t12_35M_UR50D'
tokenizer:PreTrainedTokenizer=EsmTokenizer.from_pretrained(model_name)

torch.nn.CrossEntropyLoss
def create_mu_map():
    mapping = {}
    for i, char in enumerate(range(ord('A'), ord('Z') + 1)):
        mapping[chr(char)] = i

    start_ord = ord('a')
    end_ord = start_ord + NUM_CLASS -26 -1  
    for i, char in enumerate(range(start_ord, end_ord + 1), start=26):
        mapping[chr(char)] = i
    mapping['*']=-100
    return mapping
MU_MAP=create_mu_map()

def create_map(tokens:Sequence):
    ret={ t:i for i,t in enumerate(tokens)}

class MHClassDataset(Dataset):
    def __init__(self,dataframe:pd.DataFrame,
        feature_col:str="seq",label_cols:List[str]=['mu'],
        ignore_labels=[],ignore_index=-100
        ) -> None: #,label_maps:Optional[Dict[str,Dict[str]]]=None
        super().__init__()
        self.feature_col=feature_col
        self.label_cols=label_cols
        self.ignore_labels=ignore_labels
        self.ignore_index=ignore_index
        self.dataframe=dataframe
        

    def _init_labels(self):
        self.label_count=defaultdict(lambda:defaultdict(int))
        self.label_map=defaultdict(lambda:defaultdict(int))
        for label_col in self.label_cols:
            for label_seq in self.dataframe[label_col]:
                for label in label_seq:
                    self.label_count[label_col][label]+=1
            for i,t in enumerate(self.label_count[label_col].keys()):
                if t not in self.ignore_labels:
                    self.label_map[label_col][t]=i

        
        innner_dataset=ds.from_pandas(dataframe)
        
        
    @property
    def collate_fn(self):
        return partial(collate_fn,feature=self.feature,labels=self.labels)
    
    
def map_fn(item,feature_col:str="seq",label_cols:List[str]=['mu'],label_maps:Dict[str,Dict[str]]={'mu':MU_MAP}):
    tokenized_inputs = tokenizer(item[feature_col], truncation=False,padding=True)
    item.update(tokenized_inputs)
    for label in label_cols:
        item[f'{label}_ids']=[-100]+[label_maps['label'].get(i,-1) for i in item[label]]+[-100]
    return item


def collate_fn(items:List[ITEM],max_length=800,train:bool=True,mask_rate=0.1,feature_col:str="seq",label_cols:List[str]=['mu']):
    ret=defaultdict(list)
    for item in items:
        del_len=len(item['input_ids'])-max_length
        if del_len<0:
            def padding(item,del_len):
                item['input_ids']=item['input_ids']+[1]*(-del_len)
                item['attention_mask']=item['attention_mask']+[0]*(-del_len)
                for label in label_cols:
                    item[f'{label}_ids']=item[f'{label}_ids']+[-100]*(-del_len)
                item['idx_b']=0
            padding(item,del_len)
        elif del_len>0:
            idx_b=random_generator.randint(0,del_len)
            truncation=lambda x:[x[0]]+x[idx_b+1:idx_b+max_length-1]+[x[-1]]
            item['input_ids']=truncation(item['input_ids'])
            item['attention_mask']=truncation(item['attention_mask'])
            for label in label_cols:
                item[f'{label}_ids']=truncation(item[f'{label}_ids'])
            item['idx_b']=idx_b+1
        else:
            item['idx_b']=0
        
        for k in item.keys():
            ret[k].append(item[k])
            
    for k in ['input_ids','attention_mask','idx_b']+[f'{label}_ids' for label in feature_col]:
        ret[k]=torch.tensor(ret[k],dtype=torch.long)
        
    ret['position_ids']=create_position_ids(
        ret['input_ids'],ret['attention_mask'],ret['idx_b'].reshape(-1,1))
    if train:
        _=ret['input_ids'][:,1:-1]
        _[torch.rand_like(_,dtype=torch.float16)>1-mask_rate]=tokenizer.mask_token_id
    return ret
        