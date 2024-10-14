# %%
from typing import Union,List,Any,Dict,Optional
import pickle as pkl
import pandas as pd
from torch import nn
import torch
from torch.optim import AdamW,Adam
import numpy as np
import logging
import time
import os
from torch.utils.data import DataLoader
from functools import partial
import pandas as pd
from copy import deepcopy
# from math import ceil
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
# from matplotlib.axes import Axes
# from matplotlib.backends.backend_pdf import PdfPages
# from tqdm import tqdm
# import sys
from torch.utils.tensorboard import SummaryWriter
from transformers import EsmModel, EsmConfig, EsmTokenizer,EsmForTokenClassification
from datasets import Dataset
from transformers import EsmTokenizer
# from torch.utils.data import DataLoader
from metrics import MulticlassMetric
from torch.optim.lr_scheduler import StepLR,OneCycleLR,ConstantLR,ExponentialLR,ChainedScheduler,SequentialLR
from transformers.models.esm.modeling_esm import create_position_ids_from_input_ids as create_position_ids
import random
from collections import defaultdict
logger=logging.getLogger()
time_str=time.strftime("%y%m%d-%H%M%S", time.localtime())   
ITEM=Dict[str,Union[str,List[float]]]
# %%
NUM_CLASS=36
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

seed=42
random_generator=random.Random(seed)
numpy_generator=np.random.default_rng(seed)
torch_generator=torch.Generator('cpu')
torch_generator.manual_seed(seed)

model_name='facebook/esm2_t12_35M_UR50D' #'facebook/esm2_t6_8M_UR50D'
tokenizer:EsmTokenizer=EsmTokenizer.from_pretrained(model_name)

mu='Mu'
# %%
def map_fn(examples,mu='Mu'):
    tokenized_inputs = tokenizer(examples["seq"], truncation=False,padding=True)
    examples.update(tokenized_inputs)
    examples[f'{mu}_ids']=[-100]+[OUTPUT_MAP.get(i,-1) for i in examples[mu]]+[-100]
    return examples

def collate_fn(features:List[ITEM],max_length=800,train:bool=True,mask_rate=0.1,mu='Mu'):
    ret=defaultdict(list)
    for feature in features:
        del_len=len(feature['input_ids'])-max_length
        if del_len<0:
            def padding(feature,del_len):
                feature['input_ids']=feature['input_ids']+[1]*(-del_len)
                feature['attention_mask']=feature['attention_mask']+[0]*(-del_len)
                feature[f'{mu}_ids']=feature[f'{mu}_ids']+[-100]*(-del_len)
                feature['idx_b']=0
            padding(feature,del_len)
        elif del_len>0:
            idx_b=random_generator.randint(0,del_len)
            truncation=lambda x:[x[0]]+x[idx_b+1:idx_b+max_length-1]+[x[-1]]
            feature['input_ids']=truncation(feature['input_ids'])
            feature['attention_mask']=truncation(feature['attention_mask'])
            feature[f'{mu}_ids']=truncation(feature[f'{mu}_ids'])
            feature['idx_b']=idx_b+1
        else:
            feature['idx_b']=0
        
        for k in feature.keys():
            ret[k].append(feature[k])
            
    for k in ['input_ids','attention_mask',f'{mu}_ids','idx_b']:
        ret[k]=torch.tensor(ret[k],dtype=torch.long)
        
    ret['position_ids']=create_position_ids(
        ret['input_ids'],ret['attention_mask'],ret['idx_b'].reshape(-1,1))
    if train:
        _=ret['input_ids'][:,1:-1]
        _[torch.rand_like(_,dtype=torch.float16)>1-mask_rate]=tokenizer.mask_token_id
    return ret
        
# mu=pd.read_pickle('data/mu.pkl')
# mu=mu[mu['seq'].apply(lambda x:len(x))>0]
# dataset=Dataset.from_pandas(mu).map(map_fn)

if __name__=='__main__':
    dataset=Dataset.load_from_disk('data/af_swiss_muonly')
    loader=DataLoader(dataset,batch_size=8,collate_fn=partial(collate_fn,max_length=600,train=True,mask_rate=0.1))
    #%%
    device=0
    epoch=0
    config=EsmConfig.from_pretrained(model_name)
    config.num_labels=NUM_CLASS
    model=EsmForTokenClassification(config=config)
    model.load_state_dict(torch.load('data/ep-99.pt'))
    model.to(device)
    model.train()
    # model.load_state_dict(torch.load('train/esm2_t12_240625-152559_seed42/ep-19.pt'))
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    #%%
    optimizer = Adam(model.parameters(), lr=1e-6,weight_decay=1e-7)
    # scheduler = OneCycleLR(optimizer, max_lr=5e-5, steps_per_epoch=len(loader), epochs=epoch)
    # schedulers = [ConstantLR(optimizer, factor=1e-2, total_iters=1),
    #               ConstantLR(optimizer, factor=1e2, total_iters=1),
    #               ExponentialLR(optimizer, gamma=0.95)]
    # scheduler = ChainedScheduler(schedulers)\
    # scheduler = ExponentialLR(optimizer, gamma=0.95)
    schedulers = [ConstantLR(optimizer, factor=1e-1, total_iters=1),
                ExponentialLR(optimizer, gamma=0.95)]
    scheduler = SequentialLR(optimizer=optimizer,schedulers=schedulers,milestones=[1])
    criterion=nn.CrossEntropyLoss()
    metrics=MulticlassMetric(num_classes=NUM_CLASS)
    tag='esm2_t12'
    odir=f'train/{tag}_{time_str}_seed{seed}'
    os.mkdir(odir)
    writer = SummaryWriter(log_dir=f'{odir}/log')

    step=0
    def train_epoch():
        global step,losses
        for iteration, (batch) in enumerate(loader):
            optimizer.zero_grad()
            pred=model(batch['input_ids'].to(device),
                    batch['attention_mask'].to(device),
                    position_ids=batch['position_ids'].to(device)).logits
            # pred=nn.functional.softmax(pred,-1)
            target=batch[f'{mu}_ids'].to(device)
            loss:torch.Tensor = criterion(pred.permute(0, 2, 1),target)
            if loss.item()<5:
                loss.backward()
                optimizer.step()
                metrics.update(pred.reshape(-1,NUM_CLASS), target.reshape(-1))
                losses.append(loss.item())
            else:
                # print(f'{step} !!! {loss.item()} '+'\t'.join(batch['stem']))
                print(f'{step} !!! {loss.item()}')
                print('\n'.join(batch['stem'])+'\n',file=open(f'{odir}/fatal.list','a'))
            step+=1
            # writer.add_scalar(f"loss", loss.item(), step)
            # import pdb;pdb.set_trace()
            
            if step%100==0:
                result = metrics.compute()
                step_log = f"training: [{iteration}/{len(loader)}]\t"
                l_=sum(losses)/len(losses)
                writer.add_scalar("loss", l_, step)
                
                step_log += f"loss {l_:.6f}\t"
                losses=[]
                for name, score in result.items():
                    step_log += f"{name} {score:.6f}\t"
                    writer.add_scalar(f"{name}", score, step)
                # import pdb;pdb.set_trace()
                lr=scheduler._last_lr[-1]
                writer.add_scalar("lr", lr, step)
                step_log += f"lr {lr:.3e}\t"
                metrics.reset()
                print(step_log)
                
            if step%5000==0:
                torch.save(model.state_dict(), f'{odir}/step-{step}.pt')
                scheduler.step()
    for epoch in range(epoch,epoch+20):
        losses=[]
        writer.add_scalar("epoch", epoch, step)
        train_epoch()
    