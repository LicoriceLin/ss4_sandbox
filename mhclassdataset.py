# %%
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
import pandas as pd
import numpy as np
from collections import defaultdict
import pickle as pkl
from typing import Dict,Union,List,Optional,Any,Literal
import random
from datasets import Dataset as ds
from transformers.models.esm.modeling_esm import create_position_ids_from_input_ids as create_position_ids
from torch.utils.data import DataLoader,Dataset,random_split
from functools import partial
import torch
from transformers import PreTrainedTokenizer,EsmTokenizer,AutoTokenizer
from collections.abc import Sequence,Mapping,Callable
from argparse import ArgumentParser,Namespace
import lightning as L
from pathlib import Path
ITEM=Dict[str,Union[str,List[float]]]
#%%
class MHClassDatasetModule(L.LightningDataModule):
    def __init__(self,
        dataset_path:Optional[str]=None,
        dataframe:Optional[pd.DataFrame]=None,
        need_preprocess:bool=True,
        preprocess_to:Optional[str]=None,
        feature_col:str="seq",
        label_cols:List[str]=['mu'],
        ignore_labels:List[str]=[],
        ignore_index:int=-100,
        min_length:int=3,
        max_length:int=800,
        label_init_subsetsize:Optional[int]=5000,
        label_maps:Optional[Dict[str,Dict[str,int]]]=None, 
        data_seed:int=42,
        split_ratio:List[float]=[0.9,0.05,0.05],
        batch_size:int=16,
        loader_thread:int=32,
        infer_batch_size:int=64,
        tokenizer_name:str='facebook/esm2_t12_35M_UR50D',
        mask_rate:float=0.1,
        plddt_strategy:Literal['mask','weight','none']='none',#None, 'mask','weight'
        # plddt_param:Optional[float]=None,#for 'mask': the threshold to ignore, for 'weight': the exponent
        prep_num_proc:Optional[int]=None #TODO now with bug and not working
        ):
        ##Conserved keys: pLDDT, stem
        super().__init__()
        self.dataset_path=dataset_path
        self.dataframe=dataframe
        self.need_preprocess=need_preprocess
        self.preprocess_to=preprocess_to
        self.feature_col=feature_col
        self.label_cols=label_cols
        self.ignore_labels=ignore_labels
        self.ignore_index=ignore_index
        self.max_length=max_length
        self.min_length=min_length
        self.label_init_subsetsize=label_init_subsetsize
        self.label_maps=label_maps
        self.seed=data_seed #TODO no need for this
        self.split_ratio=split_ratio
        self.batch_size=batch_size
        self.mask_rate=mask_rate
        self.loader_thread=loader_thread
        self.infer_batch_size=infer_batch_size
        self.tokenizer_name=tokenizer_name
        self.plddt_strategy=plddt_strategy
        if self.plddt_strategy=='none': 
            self.plddt_strategy=None
        self.prep_num_proc=prep_num_proc
        self.tokenizer=AutoTokenizer.from_pretrained('facebook/esm2_t12_35M_UR50D')
        self._init_seed()
        self.save_hyperparameters(ignore=[])
        
    def _init_seed(self):
        #TODO remove, simply use global seed
        self.random_generator=random.Random(self.seed)
        self.numpy_generator=np.random.default_rng(self.seed)
        self.torch_generator =torch.Generator().manual_seed(self.seed)
        
    def prepare_data(self):
        if self.dataset_path is not None:
            innner_dataset=ds.load_from_disk(self.dataset_path)
        elif self.dataframe is not None:
        # self.dataframe=dataframe
            innner_dataset=ds.from_pandas(self.dataframe)
        else:
            raise ValueError('either `dataset_path` or `dataframe` should be assigned.')
        
        if self.need_preprocess:
            assert self.preprocess_to,'need a dir to save processed dataset.'
            Path(self.preprocess_to).mkdir(exist_ok=True)
            #filter invalid sample
            def ds_filter(x):
                if len(x[self.feature_col])<self.min_length:
                    return False
                else:
                    for i in self.label_cols:
                        if i not in x:
                            return False
                return True
            innner_dataset=innner_dataset.filter(ds_filter)
            # init label_maps with priority: 
            #     `label_maps` parameter > label_maps.pkl in `dataset_path` > self._init_labels
            if self.label_maps is not None:
                label_maps=self.label_maps
            else:
                label_maps_file=Path(str(self.dataset_path))/'label_maps.pkl'
                if label_maps_file.exists():
                    label_maps=pkl.load(open(label_maps_file,'rb'))
                else:
                    label_maps,label_counts,subsetsize=self._init_labels(innner_dataset)
                    pkl.dump(label_counts,open(self.preprocess_to+f'/label_counts_{subsetsize}.pkl','wb'))
            pkl.dump(label_maps,open(self.preprocess_to+'/label_maps.pkl','wb'))          
            # map & save innner_dataset
            map_fn=partial(self.map_fn,label_maps=label_maps)
            innner_dataset=innner_dataset.map(map_fn,num_proc=self.prep_num_proc)
            innner_dataset.save_to_disk(self.preprocess_to)
        
    def _init_labels(self,innner_dataset:ds):
        label_counts=defaultdict(lambda:defaultdict(int))
        label_maps=defaultdict(lambda:defaultdict(int))
        if self.label_init_subsetsize and self.label_init_subsetsize<len(innner_dataset):
            subsetsize=self.label_init_subsetsize
        else:
            subsetsize=len(innner_dataset)
        for item in innner_dataset.take(subsetsize).iter(batch_size=1):
            for label_col in self.label_cols:
            # for label_seq in self.dataframe[label_col]:
                for label in item[label_col][0]:
                    label_counts[label_col][label]+=1
        for label_col in self.label_cols:
            for i,t in enumerate(label_counts[label_col].keys()):
                if t not in self.ignore_labels:
                    label_maps[label_col][t]=i
        return dict(label_maps),dict(label_counts),subsetsize
    
    def setup(self,stage:str):
        '''
        stage: fit,test,pred
        '''
        # if stage == "fit":
        #load innner_dataset
        if self.preprocess_to is not None:
            self.innner_dataset=ds.load_from_disk(self.preprocess_to)
        elif self.dataset_path is not None:
            self.innner_dataset=ds.load_from_disk(self.dataset_path)
        elif self.dataframe is not None:
             self.innner_dataset=ds.from_pandas(self.dataframe)
        else:
            raise ValueError('No valid dataset given.')
        #load label_maps (for inner control)
        if self.label_maps is None:
            label_maps_file=Path(str(self.dataset_path))/'label_maps.pkl'
            if label_maps_file.exists():
                self.label_maps=pkl.load(open(label_maps_file,'rb'))
            else:
                label_maps_file=Path(str(self.preprocess_to)+'/label_maps.pkl')
                assert label_maps_file.exists(),'No valid label_maps given.'
                self.label_maps=pkl.load(open(label_maps_file,'rb'))
        #split 
        self.trainset,self.valset,self.testset=random_split(
            self.innner_dataset,lengths=self.split_ratio,generator=self.torch_generator
        )
        # else:
        self.dataset=self.innner_dataset
        
    @property
    def map_fn(self)->Callable:
        def map_fn(item:Dict[str,str],label_maps:Dict[str,Dict[str,int]]):
            tokenized_inputs = self.tokenizer(item[self.feature_col], truncation=False,padding=True)
            item.update(tokenized_inputs)
            for label in self.label_cols:
                item[f'{label}_ids']=[self.ignore_index]+[label_maps[label].get(i,self.ignore_index) for i in item[label]]+[self.ignore_index]
            if 'pLDDT' in item:
                item['pLDDT']=[0.]+item['pLDDT']+[0.]
                if max(item['pLDDT'])>1.:
                    item['pLDDT']=[i/100 for i in item['pLDDT']]
            for label in self.ignore_labels:
                item.pop(label)
            return item
        return map_fn
            
    @property
    def collate_fn(self)->Callable:
        def collate_fn(items:List[ITEM],train:bool,
                       pad_to_longest:bool=False):
            if pad_to_longest:
                max_length=max([len(item['input_ids']) for item in items])
            else:
                max_length=self.max_length
            ret=defaultdict(list)
            for item in items:
                # if 'pLDDT' in item:
                #     item['pLDDT']=[0.]+item['pLDDT']+[0.]
                del_len=len(item['input_ids'])-max_length
                if del_len<0:
                    def padding(item,del_len):
                        item['input_ids']=item['input_ids']+[1]*(-del_len)
                        item['attention_mask']=item['attention_mask']+[0]*(-del_len)
                        for label in self.label_cols:
                            item[f'{label}_ids']=item[f'{label}_ids']+[-100]*(-del_len)
                        if 'pLDDT' in item: # if self.plddt_strategy is not None:
                            item['pLDDT']=item['pLDDT']+[0.]*(-del_len)
                        item['idx_b']=0
                    padding(item,del_len)
                elif del_len>0:
                    idx_b=self.random_generator.randint(0,del_len)
                    truncation=lambda x:[x[0]]+x[idx_b+1:idx_b+max_length-1]+[x[-1]]
                    item['input_ids']=truncation(item['input_ids'])
                    item['attention_mask']=truncation(item['attention_mask'])
                    for label in self.label_cols:
                        item[f'{label}_ids']=truncation(item[f'{label}_ids'])
                    if 'pLDDT' in item: # if self.plddt_strategy is not None:
                        item['pLDDT']=truncation(item['pLDDT'])
                    item['idx_b']=idx_b+1
                else:
                    item['idx_b']=0
                
                for k in item.keys():
                    ret[k].append(item[k])
                    
            for k in ['input_ids','attention_mask','idx_b']+[f'{label}_ids' for label in self.label_cols]:
                ret[k]=torch.tensor(ret[k],dtype=torch.long)
            # import pickle as pkl;pkl.dump(ret['pLDDT'],open('tmp.pkl','wb'))
            if 'pLDDT' in ret: #self.plddt_strategy is not None:
                ret['pLDDT']=torch.tensor(ret['pLDDT'],dtype=torch.float)/100
            ret['position_ids']=create_position_ids(
                ret['input_ids'],ret['attention_mask'],ret['idx_b'].reshape(-1,1))
            if train:
                r_=torch.rand(size=ret['input_ids'].shape,generator=self.torch_generator,dtype=torch.float16)
                ret['input_ids'][(r_>1-self.mask_rate) & (ret['input_ids']>2)]=self.tokenizer.mask_token_id
                # _[torch.rand_like(_,dtype=torch.float16)>1-mask_rate]=tokenizer.mask_token_id
            return dict(ret)
        return collate_fn
            
    def train_dataloader(self):
        collate_fn=partial(self.collate_fn,train=True)
        return DataLoader(self.trainset, batch_size=self.batch_size,
            collate_fn=collate_fn,num_workers=min(self.loader_thread,self.batch_size))
    
    def val_dataloader(self):
        collate_fn=partial(self.collate_fn,train=False)
        #tmp hack for overfit
        # return DataLoader(self.trainset, batch_size=self.batch_size,
        #     collate_fn=collate_fn,num_workers=min(self.loader_thread,self.batch_size),shuffle=False)
        return DataLoader(self.valset, batch_size=self.infer_batch_size,collate_fn=collate_fn,
                    num_workers=min(self.loader_thread,self.infer_batch_size))
    
    def test_dataloader(self):
        collate_fn=partial(self.collate_fn,train=False,pad_to_longest=True)
        return DataLoader(self.testset, batch_size=self.infer_batch_size,
            collate_fn=collate_fn,num_workers=min(self.loader_thread,self.infer_batch_size))
    
    def predict_dataloader(self):
        collate_fn=partial(self.collate_fn,train=False)
        return DataLoader(self.dataset, batch_size=self.infer_batch_size,
            collate_fn=collate_fn,num_workers=min(self.loader_thread,self.infer_batch_size))
            
    # def state_dict(self):
    #     return dict(self.hparams)
    
    # def load_state_dict(self, state_dict: Dict[str, Any]):
    #     '''
    #     untested
    #     '''
        # import pdb;pdb.set_trace()
        # self.__init__(**state_dict)
        # raise NotImplementedError
    
    
#%%
if 0:
    # mhds=MHClassDatasetModule(dataset_path='../AF_SWISS/',
    #                         label_cols=['Mu'],
    #                         ignore_labels=['Conf3', 'Conf4', 'Conf16', 'NENConf16', 'RENConf16', 'NENDist16'],
    #                         preprocess_to='data/AF_SWISS_Mu'
    #                         )
    innner_dataset=ds.load_from_disk('../AF_SWISS/').take(100)
    mhds=MHClassDatasetModule(dataset_path='data/samll_test_data',
                            label_cols=['Conf3','NENConf16', 'RENConf16', 'NENDist16'],
                            ignore_labels=['Mu', 'Conf4', 'Conf16'],
                            preprocess_to='data/small_AF_SWISS_factor',
                            need_preprocess=True,
                            label_init_subsetsize=10000
                            )
    mhds1=MHClassDatasetModule(dataset_path='data/small_AF_SWISS_factor',
                            label_cols=['Conf3','NENConf16', 'RENConf16', 'NENDist16'],
                            ignore_labels=['Mu', 'Conf4', 'Conf16'],
                            need_preprocess=False,
                            )
    mhds1.prepare_data()
    mhds1.setup('fit')
    train_loader=mhds1.train_dataloader()
    for i in train_loader:
        break
# #%%


        