# %%
from Bio.SeqIO import parse
from Bio.SeqRecord import SeqRecord
from transformers import AutoTokenizer,EsmTokenizer
from datasets import Dataset as ds
import lightning as L
from typing import Optional,List,Dict,Union
from pathlib import Path
import torch
from torch import Tensor
from collections import defaultdict
from transformers.models.esm.modeling_esm import create_position_ids_from_input_ids as create_position_ids
from torch.utils.data import DataLoader
from lightning.pytorch.cli import LightningCLI
from mhclassmodel import EsmTokenMhClassifier
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
#%%
class SimpleFastaDataModule(L.LightningDataModule):
    def __init__(self,
        dataset_path:Optional[str]=None,
        # need_preprocess:bool=True,
        preprocess_to:str='./FastaDB',
        tokenizer_name:str='facebook/esm2_t12_35M_UR50D',
        batch_size:int=20,
        loader_thread:int=32
        ):
        super().__init__()
        self.dataset_path=dataset_path
        # self.need_preprocess=need_preprocess
        self.preprocess_to=preprocess_to
        self.tokenizer_name=tokenizer_name
        self.tokenizer:EsmTokenizer=AutoTokenizer.from_pretrained(tokenizer_name)
        self.batch_size=batch_size
        self.loader_thread=loader_thread
        self.save_hyperparameters(ignore=[])
        self.set_collate_fn()

    def prepare_data(self):
        if not Path(self.preprocess_to).exists():
            assert self.dataset_path is not None
            o=[]
            for i in tqdm(parse(self.dataset_path,'fasta')):
                i:SeqRecord
                seq=str(i.seq)
                #could do parallel
                o.append({'stem':i.description,'seq':seq,
                    **self.tokenizer(seq, 
                    truncation=False,padding=True)
                    })
            o.sort(key=lambda x:len(x['seq']))
            inner_dataset=ds.from_list(o)
            inner_dataset.save_to_disk(self.preprocess_to)

    def set_collate_fn(self):
        pad_token=self.tokenizer._added_tokens_encoder['<pad>']
        def collate_fn(items:List[Dict[str,Union[str,Tensor]]])->Dict[str,Union[list,Tensor]]:
            max_length=max([len(item['input_ids']) for item in items])
            ret=defaultdict(list)
            for item in items:
                del_len=len(item['input_ids'])-max_length
                ret['stem'].append(item['stem'])
                ret['seq'].append(item['seq'])
                ret['input_ids'].append(item['input_ids']+[pad_token]*(-del_len))
                ret['attention_mask'].append(item['attention_mask']+[0]*(-del_len))
                ret['idx_b'].append(0)
                
            for k in ['input_ids','attention_mask','idx_b']:
                ret[k]=torch.tensor(ret[k],dtype=torch.long)
            ret['position_ids']=create_position_ids(
                ret['input_ids'],ret['attention_mask'],ret['idx_b'].reshape(-1,1))
            return dict(ret)
        self.collate_fn=collate_fn

    def setup(self,stage:str):
        self.inner_dataset=ds.load_from_disk(self.preprocess_to)

    def train_dataloader(self):
        raise NotImplementedError
    
    def test_dataloader(self):
        raise NotImplementedError
    
    def val_dataloader(self):
        raise NotImplementedError
    
    def predict_dataloader(self):
        return DataLoader(self.inner_dataset,
            shuffle=False,collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            num_workers=min(self.loader_thread,self.batch_size))
    
def plot_output(o:dict,k:str):
    #Temporory for Mu
    entry=o[k]
    rev_m={v:k for k,v in o['#label_maps']['Mu'].items()}
    fig,ax=plt.subplots(1,1)
    sns.heatmap(entry['Mu'].T,
        xticklabels=list(entry['seq']),
        yticklabels=[rev_m[i] for i in range(len(rev_m))],
        ax=ax,rasterized=True)
    return fig,ax

def cli_main():
    cli = LightningCLI(datamodule_class=SimpleFastaDataModule, 
                       model_class=EsmTokenMhClassifier,
                       parser_kwargs={"parser_mode": "omegaconf"}
                       )
    

if __name__ == "__main__":
    cli_main()