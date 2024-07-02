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
from collections.abc import Sequence,Mapping,Callable
from argparse import ArgumentParser,Namespace
NUM_CLASS=36
ITEM=Dict[str,Union[str,List[float]]]

seed=42
random_generator=random.Random(seed)
numpy_generator=np.random.default_rng(seed)
torch_generator=torch.Generator('cpu')
torch_generator.manual_seed(seed)
model_name='facebook/esm2_t12_35M_UR50D'
tokenizer:PreTrainedTokenizer=EsmTokenizer.from_pretrained(model_name)

def create_mu_map():
    NUM_CLASS=36
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
    def __init__(self,
        dataset_path:Optional[str]=None,
        dataframe:Optional[pd.DataFrame]=None,
        feature_col:str="seq",label_cols:List[str]=['mu'],
        ignore_labels=[],
        ignore_index=-100,
        max_length=800,
        label_maps:Optional[Dict[str,Dict[str,Mapping]]]=None, #{'mu':MU_MAP}
        ) -> None:
        #TODO: config system to manage parameters
        super().__init__()
        # --
        self.feature_col=feature_col
        self.label_cols=label_cols
        self.ignore_labels=ignore_labels
        self.ignore_index=ignore_index
        self.max_length=max_length
        # --
        if dataset_path is not None:
            self.innner_dataset=ds.load_from_disk(dataset_path).take(10000)
        elif dataframe:
        # self.dataframe=dataframe
            self.innner_dataset=ds.from_pandas(dataframe).take(10000)
        else:
            raise ValueError('either `dataset_path` or `dataframe` should be assigned.')
        def ds_filter(x):
            if len(x[feature_col])<3:
                return False
            else:
                for i in label_cols:
                    if i not in x:
                        return False
            return True
        self.innner_dataset=self.innner_dataset.filter(ds_filter)
        # --
        if label_maps is None:
            self._init_labels()
        else:
            self.label_counts=defaultdict(lambda:defaultdict(int))
            self.label_maps=label_maps
            
        self.innner_dataset=self.innner_dataset.map(self.map_fn)
            
    def _init_labels(self):
        self.label_counts=defaultdict(lambda:defaultdict(int))
        self.label_maps=defaultdict(lambda:defaultdict(int))
        for item in self.innner_dataset.iter():
            for label_col in self.label_cols:
            # for label_seq in self.dataframe[label_col]:
                for label in item[label_col]:
                    self.label_counts[label_col][label]+=1
            for i,t in enumerate(self.label_counts[label_col].keys()):
                if t not in self.ignore_labels:
                    self.label_maps[label_col][t]=i

        
        
        
        
    @property
    def collate_fn(self)->Callable:
        '''
        WARNING: `train` should be specified during downstream pipelines.
        '''
        def collate_fn(items:List[ITEM],train:bool,max_length=800,mask_rate=0.1,feature_col:str="seq",label_cols:List[str]=['mu']):
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
        return partial(collate_fn,max_length=self.max_length,feature_col=self.feature_col,label_cols=self.label_cols)
    
    @property
    def map_fn(self)->Callable:
        def map_fn(item:Dict[str,str],feature_col:str="seq",
                label_cols:List[str]=['mu'],
                label_maps:Dict[str,Dict[str,Mapping]]={'mu':MU_MAP},
                ignore_labels=[],
                ignore_index=-100):
            tokenized_inputs = tokenizer(item[feature_col], truncation=False,padding=True)
            item.update(tokenized_inputs)
            for label in label_cols:
                item[f'{label}_ids']=[ignore_index]+[label_maps[label].get(i,-1) for i in item[label]]+[ignore_index]
            for label in ignore_labels:
                item.pop(label)
            return item
        return partial(map_fn,feature_col=self.feature_col,label_cols=self.label_cols,
                       label_maps=self.label_maps,ignore_labels=self.ignore_labels,
                       ignore_index=self.ignore_index)



        