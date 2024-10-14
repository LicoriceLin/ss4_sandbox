from transformers import EsmModel, EsmConfig, EsmTokenizer,EsmForTokenClassification,EsmPreTrainedModel
from torch.optim.lr_scheduler import ConstantLR,ExponentialLR,ChainedScheduler,SequentialLR,ReduceLROnPlateau
from torch.optim import lr_scheduler
from bisect import bisect_right
import torch
from torch import Tensor
from torch import nn
from typing import Optional,Union,Tuple,List,Dict,Literal
from pathlib import Path
import pickle as pkl
import lightning as L
# from metrics import MulticlassMetric
from mhcmetrics import MulticlassMetricCollection
import logging
from lightning.pytorch.callbacks import Callback
from lightning import Trainer
from lightning.pytorch.trainer.states import TrainerFn
from pytorch_lightning.callbacks import ModelCheckpoint
from collections import defaultdict
from itertools import chain
from torch.cuda import OutOfMemoryError,empty_cache
# %%
# class RelaxChainedScheduler(ChainedScheduler):
#     def step(self,metrics=None):
#         for scheduler in self._schedulers:
#             if not isinstance(scheduler,ReduceLROnPlateau):
#                 scheduler.step()
#             else:
#                 scheduler.step(metrics)
#         self._last_lr = [group['lr'] for group in 
#             self._schedulers[-1].optimizer.param_groups]
        
#%%
class EsmTokenMhClassification(EsmPreTrainedModel):
    'multi-head token classification'
    def __init__(self, 
            model_name:str='facebook/esm2_t12_35M_UR50D',
            use_pretrained_params:bool=True,
            dataset_path:Optional[str]=None,
            label_maps:Optional[Dict[str,Dict[str,int]]]=None, 
            num_head_layers:int=1,
            hidden_dropout_prob:float=0.1,
            ): 
        '''
        num_mhlabels: specify output dims of classification heads
        dataset_path: alternative. init num_mhlabels by 'label_maps.pkl' file
        '''
        config=EsmConfig.from_pretrained(model_name)
        super().__init__(config)
        # TODO module initialization is prior to label_maps generation.
        # allow direct passing of `num_mhlabels``
        if label_maps:
            self.label_maps = label_maps
        else:
            label_maps_file=(Path(str(dataset_path))/'label_maps.pkl')
            if label_maps_file.exists():
                self.label_maps=pkl.load(open(label_maps_file,'rb'))
            else:
                raise ValueError('no valid `num_mhlabels` or `dataset_path`')
        self.num_mhlabels:Dict[str,int]={k:len(v) for k,v in self.label_maps.items()}
        
        if use_pretrained_params:
            self.esm = EsmModel.from_pretrained(model_name, add_pooling_layer=False)
        else:
            self.esm = EsmModel(config, add_pooling_layer=False)
            
        self.num_head_layers:int = num_head_layers
        self._init_classification_head()
        self.dropout = nn.Dropout(hidden_dropout_prob)
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
        for k,num_label in self.num_mhlabels.items():
            setattr(self,f'neck_{k}',nn.TransformerEncoder(
                encoder_layer=encoder_layer,num_layers=self.num_head_layers))
            setattr(self,f'head_{k}',nn.Linear(cfg.hidden_size, num_label))
            
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Dict[str,Tensor]:
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
        output={}
        for k,num_label in self.num_mhlabels.items():
            neck=getattr(self,f'neck_{k}')
            head=getattr(self,f'head_{k}')
            output[k]=head(self.dropout(neck(
            last_hidden_state,src_key_padding_mask=src_key_padding_mask,is_causal=False)
                ))
        # return torch.concat(output,dim=-1)
        return output
           
class Criterion(nn.Module):
    def __init__(self,ignore_index=-100,
        plddt_strategy:Optional[Literal['mask','weight','none']]=None,
        plddt_param:Optional[float]=None):
        super().__init__()
        self.ignore_index=ignore_index
        self.plddt_strategy=plddt_strategy if plddt_strategy!='none' else None
        self.plddt_param=plddt_param

        if plddt_strategy is None:
            self._criterion=nn.CrossEntropyLoss(ignore_index=ignore_index)
        else:
            self._criterion=nn.CrossEntropyLoss(ignore_index=ignore_index,reduction='none')
        self.configure_c()

    def configure_c(self):
        if self.plddt_strategy is None:
            def _c(pred:Tensor,label:Tensor,plddt:Tensor=None)->Tensor:
                return self._criterion(pred.permute(0, 2, 1),label)
        elif self.plddt_strategy=='mask':
            assert 0<self.plddt_param<1
            def _c(pred:Tensor,label:Tensor,plddt:Tensor=None)->Tensor:
                losses:Tensor=self._criterion(pred.permute(0, 2, 1),label)
                plddt=plddt.masked_fill(plddt>=self.plddt_param,1.)
                plddt=plddt.masked_fill(plddt<self.plddt_param,0.)
                mean_weighted_loss = (losses*plddt).mean()
                return mean_weighted_loss
        elif self.plddt_strategy=='weight':
            assert self.plddt_param>=1
            def _c(pred:Tensor,label:Tensor,plddt:Tensor=None)->Tensor:
                losses:Tensor=self._criterion(pred.permute(0, 2, 1),label)
                plddt=plddt**self.plddt_param
                mean_weighted_loss = (losses*plddt).mean()
                return mean_weighted_loss
        else:
            raise ValueError(f'invalid `plddt_strategy`:{self.plddt_strategy}')
        self._c=_c

    def forward(self,pred:Tensor,label:Tensor,plddt:Tensor=None)->Tensor:
        # return self._criterion(pred.permute(0, 2, 1),label)
        return self._c(pred,label,plddt)
    
class EsmTokenMhClassifier(L.LightningModule):  
    def __init__(self,inner_model:EsmTokenMhClassification,
            finetuned_from:Optional[str]=None,
            # metrics_interval:int=100,
            ignore_index:int=-100,
            optimizer:str="Adam",
            optimizer_kwargs:Dict[str,float]={
                'lr':1e-5,'weight_decay':1e-7},
            scheduler_kwargs:Dict[str,Union[float,int,str]]={ #TODO smooth warmup
                'update_step':5000,'warm_up_iter':2,
                'warm_up_rate':1e-2,'exp_gamma':0.95},
            test_output_dir:str='test_output',
            plddt_strategy:Literal['mask','weight','none']='none',
            # None, 'mask', 'weight'
            plddt_param:Optional[float]=None,
            # for 'mask': the threshold to ignore, for 'weight': the exponent
            profile_test:bool=False
            ):
        #Conserved keys: pLDDT,stem
        super().__init__()
        self.inner_model=inner_model
        self.finetuned_from=finetuned_from
        if finetuned_from:
            # only load the model parameters and ignore training states
            # TODO handle `strict`
            self.load_state_dict(
            torch.load(finetuned_from)['state_dict'],strict=True)
        self.optimizer=optimizer
        self.optimizer_kwargs=optimizer_kwargs
        self.scheduler_kwargs=scheduler_kwargs
        # self.metrics_interval=metrics_interval
        self.ignore_index=ignore_index
        self._test_output_dir=test_output_dir
        self.profile_test=profile_test
        self.plddt_strategy=plddt_strategy
        if self.plddt_strategy=='none': 
            self.plddt_strategy=None
        self.plddt_param=plddt_param
        self.save_hyperparameters()
        # self.set_criterion()
        self.criterion=Criterion(self.ignore_index,self.plddt_strategy,self.plddt_param)
        self.set_metrics()
        self._l=5.0
        self.automatic_optimization=False
    def set_metrics(self):
        #TODO remove dpt's metrics
        # self.metrics:Dict[str,MulticlassMetric]={}
        # for head_name,num_classes in self.inner_model.num_mhlabels.items():
        #     self.metrics[head_name]=MulticlassMetric(num_classes=num_classes)

        # self.metrics:Dict[str,Dict[str,MulticlassMetricCollection]]={}
        if self.plddt_param is None:
            plddt_threshold=0.
        elif self.plddt_param>=1:
            plddt_threshold=0.5  
        else:
            plddt_threshold=self.plddt_param
        _prefix_map={'fit':'train-','validate':'val-','test':None}
        for stage in ['fit','validate','test']: # 'test' goes `None`:
            # self.metrics[stage]={}
            prefix = _prefix_map[stage]
            for head_name,num_classes in self.inner_model.num_mhlabels.items():
                # self.metrics[stage][head_name]=
                setattr(self,f'metrics_{stage}_{head_name}',
                    MulticlassMetricCollection(
                    num_classes=num_classes,
                    head_name=head_name,
                    ignore_index=self.ignore_index,
                    plddt_threshold=plddt_threshold,
                    prefix=prefix))
                
                
    @property
    def test_output_dir(self):
        try:
            return Path(self.trainer.default_root_dir)/self._test_output_dir
        except:
            self.dumblogger.info('abnormal calling for `test_output_dir`.')
            return Path(self._test_output_dir)
    
    @property
    def dumblogger(self):
        if not hasattr(self,'_dumblogger'):
            logger=logging.getLogger('dumblogger')
            logger.setLevel(logging.DEBUG)
            file_handler = logging.FileHandler(f'{self.trainer.default_root_dir}/dumblogger.log')
            file_handler.setLevel(logging.DEBUG) 
            formatter = logging.Formatter('%(asctime)s - %(message)s')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            self._dumblogger=logger
        return self._dumblogger
    
    def training_step(self, batch:dict, batch_idx):
        output:Dict[str,Tensor]=self.inner_model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            position_ids=batch['position_ids']
            )
        losses={}
        loss=0.
        for k,pred in output.items():
            label=self._mask_nan(batch,pred,batch[f'{k}_ids'])
            plddt=batch['pLDDT'] if self.plddt_strategy is not None else None
            # plddt=batch.get('pLDDT',None)
            l_:torch.Tensor=self.criterion(pred,label,plddt)
            if torch.isnan(l_):
                self.dumblogger.error(f'still nan loss: {"\t".join(batch['stem'])}')
                l_=self._l
            loss+=l_
            losses[f'train/{k}_loss']=l_.item()
            # metrics:MulticlassMetricCollection=getattr(self,f'metrics_fit_{k}')
            # metrics(pred, label,plddt)
            # self.log_dict(metrics)
            # metrics(pred, label,plddt)
            # metrics.update(pred,label,plddt)
            # self.log_dict(metrics)
        losses['train/loss']=loss.item()
        self.log_dict(losses,sync_dist=True)
        
        if torch.cuda.is_available():
            allocated_memory = torch.cuda.memory_allocated(self.device) / (1024 ** 3)  
            self.log(f'util/mem_{self.device.index}_train', allocated_memory,reduce_fx=torch.max,sync_dist=False)
        # return loss

        # heads_optimizer.zero_grad()
        self.manual_backward(loss)
        for optimizer in self.optimizers():
            optimizer:torch.optim.Optimizer
            optimizer.step()
            optimizer.zero_grad()

        # if (batch_idx + 1) % 50 == 0:
        true_step=self.trainer.global_step/len(self.optimizers())
        # if (true_step+1) % (self.trainer.check_val_every_n_epoch*2) ==0:
        if (true_step+1) % (self.update_step) ==0 or true_step==100:
            try:
                val_loss=self.trainer.callback_metrics["val_loss"]
            except:
                self.dumblogger.warn(f'step {true_step} without val_loss')
                val_loss=3.
            # TODO use `get`
            # self.dumblogger.info(f'\nstepping sch!\n val_loss={val_loss}\n')
            for sch in self.lr_schedulers():
                sch:torch.optim.lr_scheduler.LRScheduler
                if isinstance(sch,ReduceLROnPlateau):
                    sch.step(val_loss)
                else:
                    sch.step()
                # self.dumblogger.info(f'lr: {sch.optimizer.param_groups[0]['lr']}')
        # else:
        #     self.dumblogger.info(f'at global:{self.trainer.global_step},batch_idx: {batch_idx}')
            
    def _mask_nan(self,batch:dict,pred:torch.Tensor,label:torch.Tensor):
        if torch.isnan(pred).any():
            mask=torch.isnan(pred).any(dim=-1)
            fail_idx=torch.where(mask.any(dim=-1))[0].tolist()
            self.dumblogger.error('nan loss'+'\t'.join([f'{batch['stem'][i]}:{batch['idx_b'][i].item()}' 
                for i in fail_idx]))
            label=label.masked_fill(mask,self.ignore_index)
        return label

    def validation_step(self, batch, batch_idx):
        prefix='val'
        output:Dict[str,Tensor]=self.inner_model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            position_ids=batch['position_ids']
            )
        losses={}
        loss=0.
        for k,pred in output.items():
            label=self._mask_nan(batch,pred,batch[f'{k}_ids'])
            plddt=batch['pLDDT'] if self.plddt_strategy is not None else None
            l_:torch.Tensor=self.criterion(pred,label,plddt)
            if torch.isnan(l_):
                l_=self._l
            loss+=l_
            losses[f'{prefix}/{k}_loss']=l_.item()
            # metrics:MulticlassMetricCollection=getattr(self,f'metrics_validate_{k}')
            # metrics.update(pred, label,plddt)
            # self.log_dict(metrics) #(pred, label,plddt)
        losses[f'{prefix}_loss']=loss.item()
        losses[f'{prefix}/loss']=loss.item()
        self.log_dict(losses,sync_dist=True)
        # if torch.cuda.is_available():
        #     allocated_memory = torch.cuda.memory_allocated(self.device) / (1024 ** 3)  
        #     self.log(f'util/mem_{self.device.index}_val', allocated_memory,reduce_fx=torch.max,sync_dist=False)
        # self._shared_eval(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        output:Dict[str,Tensor]=self.inner_model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            position_ids=batch['position_ids']
            )
        losses={}
        loss=0.
        # raw_opts={'stem':batch['stem'],'mask':batch['attention_mask'].to('cpu')}
        raw_opts=defaultdict(dict)
        for k,pred in output.items():
            # raw_opts[f'{k}_pred']=pred.to('cpu')
            label=batch[f'{k}_ids']
            # raw_opts[f'{k}_label']=label.to('cpu')
            label=self._mask_nan(batch,pred,label)
            plddt=batch['pLDDT'] if self.plddt_strategy is not None else None
            l_:torch.Tensor=self.criterion(pred,label,plddt)
            if torch.isnan(l_):
                l_=self._l
            loss+=l_
            losses[f'{k}_loss']=l_.item()
            metrics:MulticlassMetricCollection=getattr(self,f'metrics_test_{k}')
            # self.log_dict() #TODO fixed log
            metrics.update(pred, label,plddt)
            for i,stem in enumerate(batch['stem']):
                pred_=pred[i]
                label_=label[i]
                mask=batch['attention_mask'][i].bool()&(label_!=self.ignore_index)
                raw_opts[stem][f'{k}_pred']=torch.softmax(pred_[mask],dim=-1).to('cpu')
                raw_opts[stem][f'{k}_label']=label_[mask].to('cpu')

        losses[f'loss']=loss.item()
        if self.profile_test:
            self.log_dict(losses,sync_dist=True)
            self.playground.update(raw_opts)
    
    def process_ruined_batch(self):
        if 'ruined-batch' in self.playground:
            ruined_batch:Tensor=self.playground.pop('ruined-batch')
            batch_size,max_length=ruined_batch['input_ids'].shape
            logging.warn(f'OutOfMemory at {batch_size}*{max_length}, try to run them one-by-one.')
            torch.cuda.memory_allocated()
            for i in range(batch_size):
                try:
                    attention_mask=ruined_batch['attention_mask'][i].bool()
                    output:Dict[str,Tensor]=self.inner_model(
                        input_ids=ruined_batch['input_ids'][i][attention_mask].unsqueeze(0),#.to(self.device),
                        attention_mask=ruined_batch['attention_mask'][i][attention_mask].unsqueeze(0),#.to(self.device),
                        position_ids=ruined_batch['position_ids'][i][attention_mask].unsqueeze(0)#.to(self.device)
                        )
                    o_={
                        'stem':ruined_batch['stem'][i],
                        'seq':ruined_batch['seq'][i]
                        }
                    for k,pred in output.items():
                        o_[k]= torch.softmax(pred[0][1:-1].to('cpu'),-1)
                    self.playground[ruined_batch['stem'][i]]=o_
                    del attention_mask,output,o_,k,pred
                    # del output,attention_mask
                except OutOfMemoryError:
                    logging.warn(
                    (f'OutOfMemory at {attention_mask.sum().item()} even during one-by-one run.'
                     f"skip {ruined_batch['stem'][i]} for now"))
            del ruined_batch

    def predict_step(self, batch:Dict[str,Union[list,Tensor]], batch_idx):
        self.process_ruined_batch()
            # import pdb;pdb.set_trace()
        try:
            batch_size,max_length=batch['input_ids'].shape
            output:Dict[str,Tensor]=self.inner_model(
                input_ids=batch['input_ids'].detach(),
                attention_mask=batch['attention_mask'].detach(),
                position_ids=batch['position_ids'].detach()
                )
            output={k:torch.softmax(v,dim=-1).to('cpu') for k,v in output.items()}
            for i in range(batch_size):
                o_={
                    'stem':batch['stem'][i],
                    'seq':batch['seq'][i]
                    }
                attention_mask=batch['attention_mask'][i].bool().to('cpu')
                for k,pred in output.items():
                    o_[k]=pred[i][attention_mask][1:-1]
                self.playground[batch['stem'][i]]=o_
        except OutOfMemoryError:
            # torch.cuda.memory_allocated
            # move = lambda x:x.to('cpu').detach().clone() if isinstance(x,torch.Tensor) else x
            # self.playground['ruined-batch']={k:move(v) for k,v in batch.items()}
            self.playground['ruined-batch']=batch
            return
            
    def configure_optimizers(self):
        heads_parameters=[]
        for k,num_label in self.inner_model.num_mhlabels.items():
            heads_parameters.append(getattr(self.inner_model,f'neck_{k}').parameters())
            heads_parameters.append(getattr(self.inner_model,f'head_{k}').parameters())
        heads_optimizer=torch.optim.AdamW(params=chain(*heads_parameters), lr=1e-3)
        body_optimizer=torch.optim.AdamW(params=self.inner_model.esm.parameters(), lr=1e-3)
        
        sk=self.scheduler_kwargs
        wu_iter=int(sk.pop('warm_up_iter'))
        wu_rate=sk.pop('warm_up_rate')
        update_step=int(sk.pop('update_step'))
        self.update_step=update_step
        if 'exp_gamma' in sk: sk['gamma']=sk.pop('exp_gamma')
        main_scheduler_cls=sk.pop('main_scheduler','ExponentialLR')
        
        return ([heads_optimizer,body_optimizer],[
                    {
                        "scheduler": ConstantLR(body_optimizer, 1e-7,20),
                        "interval": "step",
                        "frequency": update_step,
                    },
                    {
                        "scheduler": ExponentialLR(body_optimizer,gamma=0.98),
                        "interval": "step",
                        "frequency": update_step,
                    },
                    {
                        "scheduler": ExponentialLR(heads_optimizer,gamma=0.98),
                        "interval": "step",
                        "frequency": update_step,
                    },
                    {
                        "scheduler": ReduceLROnPlateau(body_optimizer, factor=0.5,patience=10,threshold=0.01),
                        "interval": "step",
                        "frequency": update_step,
                        "monitor": "val_loss",
                        "reduce_on_plateau": True,
                        "strict": False,
                    },
                    {
                        "scheduler": ReduceLROnPlateau(heads_optimizer, factor=0.5,patience=10,threshold=0.01),
                        "interval": "step",
                        "frequency": update_step,
                        "monitor": "val_loss",
                        "reduce_on_plateau": True,
                        "strict": False,
                    }

            ])
        # optimizer:torch.optim.Optimizer=getattr(torch.optim,
        #         self.optimizer)(params=self.inner_model.parameters(), **self.optimizer_kwargs)

        # #tmp

        # # return [optimizer],[main_scheduler]
        # main_scheduler_config={
        #                 "scheduler": getattr(lr_scheduler,main_scheduler_cls)(optimizer, **sk),
        #                 "interval": "step",
        #                 "frequency": update_step,
        #                 "monitor": "val_loss",
        #                 "reduce_on_plateau": main_scheduler_cls=='ReduceLROnPlateau',
        #                 "strict": True,
        #             }
        
        # if wu_iter==0:
        #     warmup_scheduler = None
        #     return {"optimizer": optimizer,"lr_scheduler":main_scheduler_config}
        
        # elif wu_iter>1:
        #     warmup_schedulers=[]
        #     r=(1/wu_rate)**(1/wu_iter)
        #     for i in range(wu_iter):
        #         warmup_schedulers.append(ConstantLR(optimizer, 
        #         factor=wu_rate, total_iters=1))
        #         wu_rate=wu_rate*r
        #     warmup_scheduler=SequentialLR(optimizer,warmup_schedulers,milestones=list(range(1,wu_iter)))
        # elif wu_iter==1:
        #     warmup_scheduler = ConstantLR(optimizer, factor=wu_rate, total_iters=1)
        # else:
        #     raise ValueError(f'invalid wu_iter: {wu_iter}')    
        
        # warmup_scheduler_config={
        #         "scheduler": warmup_scheduler,
        #         "interval": "step",
        #         "frequency": update_step,
        #         "monitor": "val_loss",
        #         "reduce_on_plateau": False,
        #         "strict": True,
        #     }
        # return ([optimizer],[main_scheduler_config,warmup_scheduler_config])


        # if not is_plateau:
        #     if wu_iter>1:
        #         schedulers=[]
        #         r=(1/wu_rate)**(1/wu_iter)
        #         for i in range(wu_iter):
        #             schedulers.append(ConstantLR(optimizer, 
        #             factor=wu_rate, total_iters=1))
        #             wu_rate=wu_rate*r
        #         scheduler = ChainedScheduler(schedulers=[
        #             SequentialLR(optimizer,schedulers,milestones=list(range(1,wu_iter))),
        #             main_scheduler])
        #     elif wu_iter==1:
        #         scheduler = ChainedScheduler(schedulers=[
        #             ConstantLR(optimizer, factor=wu_rate, total_iters=1),
        #             main_scheduler])
        #     else:
        #         scheduler=main_scheduler
        #     return {"optimizer": optimizer,
        #             "lr_scheduler": {
        #                 "scheduler": scheduler,
        #                 "interval": "step",
        #                 "frequency": update_step,
        #                 "monitor": "val_loss",
        #                 "reduce_on_plateau": False,
        #                 "strict": False,
        #             },}
        # else:
        #     warmup_scheduler=ConstantLR(optimizer, factor=wu_rate, total_iters=wu_iter)
        #     return ([optimizer],
        #             [
        #                 {"scheduler": warmup_scheduler,
        #                 "interval": "step",
        #                 "frequency": update_step,
        #                 "monitor": "val_loss",
        #                 "reduce_on_plateau": False,
        #                 "strict": False,
        #             },
        #             {
        #                 "scheduler": main_scheduler,
        #                 "interval": "epoch",
        #                 "frequency": 1,
        #                 "monitor": "val_loss",
        #                 "reduce_on_plateau": True,
        #                 "strict": True,
        #             }
        #             ])

    def forward(self,*arg,**kwargs):
        return self.inner_model( *arg,**kwargs)
    
    def init_playground(self):
        self.playground={}
        self.test_output_dir.mkdir(exist_ok=True)
        
    def clear_playground(self):
        #TODO make a test metrics out of it
        self.to('cpu')
        torch.save(self.playground,
           self.test_output_dir/'raw_output.pt')
        if self.profile_test:
            # It can be pretty slow!
            from mhcprofile import (plot_stack_heatmap,norm_confusion_matrix,plot_color_scheme,
                plot_confusion_matrix,plot_accuracy,cal_confusion_matrix,plt)
            from matplotlib.backends.backend_pdf import PdfPages
            
            confusion_matrices={}
            soft_confusion_matrices={}
            for k in self.inner_model.num_mhlabels.keys():
                with PdfPages(self.test_output_dir/f'{k}_heatmap.pdf') as heatpdf:
                    fig,ax=plot_color_scheme()
                    fig.suptitle('ColorMap')
                    heatpdf.savefig(fig);plt.close(fig)
                    for stem,rawopt in self.playground.items():
                        pred,label=rawopt[f'{k}_pred'],rawopt[f'{k}_label']
                        fig,ax=plot_stack_heatmap(pred,label,self.inner_model.label_maps[k])
                        ax.set_title(stem);fig.tight_layout()
                        heatpdf.savefig(fig);plt.close(fig)
                        if k not in confusion_matrices:
                            confusion_matrices[k]=cal_confusion_matrix(pred,label,soft=False)
                            soft_confusion_matrices[k]=cal_confusion_matrix(pred,label,soft=True)
                        else:
                            confusion_matrices[k]+=cal_confusion_matrix(pred,label,soft=False)
                            soft_confusion_matrices[k]+=cal_confusion_matrix(pred,label,soft=True)
                with PdfPages(self.test_output_dir/f'{k}_sum.pdf') as sumpdf:
                    for if_soft,matrix in zip(('hard','soft'),
                            (confusion_matrices[k],soft_confusion_matrices[k])):
                        torch.save(matrix,self.test_output_dir/f'{k}_{if_soft}_confusion_matrix.pt')
                        m=norm_confusion_matrix(matrix)
                        fig,ax=plot_confusion_matrix(m,label_maps=self.inner_model.label_maps[k])
                        ax.set_title(f'{if_soft}_confusion_matrix')
                        sumpdf.savefig(fig); plt.close(fig)
                        fig,ax=plot_accuracy(matrix,label_maps=self.inner_model.label_maps[k])
                        ax.set_title(f'{if_soft}_accuracy')
                        sumpdf.savefig(fig); plt.close(fig)
        del self.playground
        
# %%
# classifier=EsmTokenMhClassifier(EsmTokenMhClassification(dataset_path='data/AF_SWISS_Mu'))
if 0:
    from lightning.pytorch.demos.boring_classes import DemoModel, BoringDataModule
    from lightning.pytorch.cli import ArgsType, LightningCLI
    import yaml
    def cli_main(args: ArgsType = None):
        cli=LightningCLI(model_class=EsmTokenMhClassifier,datamodule_class=BoringDataModule,run=False,args=args)
        return cli
    cli=cli_main(["--model.inner_model.num_mhlabels=[36]"])
    print(yaml.safe_dump(cli.config))
    

# class MetricsAggCallback(Callback):
#     def setup(self, trainer:Trainer, pl_module:EsmTokenMhClassifier, stage:str):
#         for k,metrics in pl_module.metrics.items():
#             metrics.to('cpu')
    
#     def _share_agg(self, pl_module:EsmTokenMhClassifier,split:str,on_step=True):
#         for k,metrics in pl_module.metrics.items():
#             if len(metrics.inputs)>0:
#                 pl_module.metrics_agg(split,on_step)
                
#     def on_train_end(self, trainer:Trainer, pl_module:EsmTokenMhClassifier):
#         self._share_agg(pl_module,'train')
        
#     # def on_validation_epoch_start(self, trainer:Trainer, pl_module:EsmTokenMhClassifier):
#     #     self._share_agg(pl_module,'train')
#     def on_test_epoch_start(self, trainer:Trainer, pl_module:EsmTokenMhClassifier):
#         self._share_agg(pl_module,'train')
        
#     def on_validation_epoch_end(self, trainer:Trainer, pl_module:EsmTokenMhClassifier):
#         self._share_agg(pl_module,'val',on_step=False)

#     def on_test_epoch_end(self, trainer:Trainer, pl_module:EsmTokenMhClassifier):
#         self._share_agg(pl_module,'test',on_step=False)
#         pl_module.clear_playground()
        
#     def on_test_epoch_start(self, trainer:Trainer, pl_module:EsmTokenMhClassifier):
#         pl_module.init_playground()

class DebugCallback(Callback):
    # def on_fit_start(self, trainer: Trainer, pl_module: L.LightningModule) -> None:
    #     import pdb;pdb.set_trace()
    # def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
    #     if torch.cuda.is_available():
    #         allocated_memory = torch.cuda.memory_allocated() / (1024 ** 3)
    #         pl_module.log_dict({
    #             'util/gpu_mem_usage': allocated_memory,
    #         })

    # def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
    #     if torch.cuda.is_available():
    #         allocated_memory = torch.cuda.memory_allocated() / (1024 ** 3)  
    #         pl_module.log_dict({
    #             'util/gpu_mem_usage': allocated_memory,
    #         })

    def on_train_start(self, trainer: Trainer, pl_module: EsmTokenMhClassifier) -> None:
        for i,optimizer in zip(['head','body'],pl_module.optimizers()):
            optimizer:torch.optim.Optimizer
            lr=optimizer.param_groups[0]['lr']
            pl_module.log(f'util/lr-{i}',lr,rank_zero_only=True,sync_dist=False)
        # scheduler=trainer.lr_scheduler_configs[0].scheduler
        # lr=scheduler.optimizer.param_groups[0]['lr']
        # pl_module.log('util/lr',lr,rank_zero_only=True)
        pl_module.dumblogger.info('train-start!')
        # print(f'lr: {lr}')
        # return super().on_train_start(trainer, pl_module)
    
    # def on_train_batch_end(self, trainer: Trainer, pl_module: EsmTokenMhClassifier, outputs, batch, batch_idx) -> None:
        # true_step=trainer.global_step/len(pl_module.optimizers())
        # if true_step % trainer.log_every_n_steps == 0:
        # for head_name,num_classes in pl_module.inner_model.num_mhlabels.items():
        #     metrics:MulticlassMetricCollection=getattr(pl_module,f'metrics_fit_{head_name}')
        #     # pl_module.log_dict(metrics.compute(),sync_dist=False)
        #     metrics.reset()
            # getattr(pl_module,f'metrics_fit_{head_name}').reset()
        
    def on_validation_epoch_start(self, trainer:Trainer, pl_module:EsmTokenMhClassifier):
        # for head_name,num_classes in pl_module.inner_model.num_mhlabels.items():
        #     metrics:MulticlassMetricCollection=getattr(pl_module,f'metrics_fit_{head_name}')
        #     metrics.reset()

        for i,optimizer in zip(['head','body'],pl_module.optimizers()):
            optimizer:torch.optim.Optimizer
            lr=optimizer.param_groups[0]['lr']
            pl_module.log(f'util/lr-{i}',lr,rank_zero_only=True,sync_dist=False)
        # scheduler=trainer.lr_scheduler_configs[0].scheduler
        # lr=scheduler.optimizer.param_groups[0]['lr']
        # pl_module.log('util/lr',lr,rank_zero_only=True)
        # print(f'lr: {lr}')

    # def on_validation_epoch_end(self, trainer: Trainer, pl_module: EsmTokenMhClassifier) -> None:
    #     for head_name,num_classes in pl_module.inner_model.num_mhlabels.items():
    #         metrics:MulticlassMetricCollection=getattr(pl_module,f'metrics_validate_{head_name}')
    #         pl_module.log_dict(metrics.compute(),sync_dist=True)
    #         metrics.reset()
    # def on_validation_start(self, trainer: Trainer, pl_module: L.LightningModule) -> None:
    #     return super().on_validation_start(trainer, pl_module)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: EsmTokenMhClassifier) -> None:
        for head_name,num_classes in pl_module.inner_model.num_mhlabels.items():
            metrics:MulticlassMetricCollection=getattr(pl_module,f'metrics_test_{head_name}')
            metrics.reset()

    def on_test_epoch_start(self, trainer:Trainer, pl_module:EsmTokenMhClassifier):
        pl_module.init_playground()

    def on_predict_epoch_start(self, trainer:Trainer, pl_module:EsmTokenMhClassifier):
        pl_module.init_playground()

    def on_predict_epoch_end(self, trainer:Trainer, pl_module:EsmTokenMhClassifier):
        pl_module.clear_playground()
# from lightning.pytorch.callbacks import DeviceStatsMonitor
