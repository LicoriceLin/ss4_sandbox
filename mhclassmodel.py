from transformers import EsmModel, EsmConfig, EsmTokenizer,EsmForTokenClassification,EsmPreTrainedModel
from torch.optim.lr_scheduler import StepLR,OneCycleLR,ConstantLR,ExponentialLR,ChainedScheduler,SequentialLR
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
# %%
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
           
class EsmTokenMhClassifier(L.LightningModule):  
    def __init__(self,inner_model:EsmTokenMhClassification,
            finetuned_from:Optional[str]=None,
            metrics_interval:int=100,
            ignore_index:int=-100,
            optimizer:str="Adam",
            optimizer_kwargs:Dict[str,float]={
                'lr':1e-5,'weight_decay':1e-7},
            scheduler_kwargs:Dict[str,Union[float,int]]={ #TODO smooth warmup
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
        self.optimizer=optimizer
        self.optimizer_kwargs=optimizer_kwargs
        self.scheduler_kwargs=scheduler_kwargs
        
        self.metrics_interval=metrics_interval
        self.ignore_index=ignore_index
        self.inner_model=inner_model
        self.finetuned_from=finetuned_from
        self._test_output_dir=test_output_dir
        self.profile_test=profile_test
        self.plddt_strategy=plddt_strategy
        if self.plddt_strategy=='none': 
            self.plddt_strategy=None
        self.plddt_param=plddt_param
        self.save_hyperparameters()
        self.set_criterion()

        if finetuned_from:
            self.inner_model.load_state_dict(
                torch.load(finetuned_from),strict=False)
        self.set_metrics()
        # self.dumblogger.info('init dumblogger.')
    
    def set_criterion(self):
        if self.plddt_strategy is None:
            _criterion=nn.CrossEntropyLoss(ignore_index=self.ignore_index)
            def _c(pred:Tensor,label:Tensor,plddt:Tensor=None)->Tensor:
                return _criterion(pred.permute(0, 2, 1),label)
        else:
            _criterion=nn.CrossEntropyLoss(ignore_index=self.ignore_index,reduction='none')
            if self.plddt_strategy=='mask':
                assert 0<self.plddt_param<1
                def _c(pred:Tensor,label:Tensor,plddt:Tensor=None)->Tensor:
                    plddt.requires_grad=False
                    losses:Tensor=_criterion(pred.permute(0, 2, 1),label)
                    plddt=plddt.masked_fill(plddt>=self.plddt_param,1.)
                    plddt=plddt.masked_fill(plddt<self.plddt_param,0.)
                    mean_weighted_loss = (losses).mean()
                    return mean_weighted_loss
            elif self.plddt_strategy=='weight':
                assert self.plddt_param>=1
                def _c(pred:Tensor,label:Tensor,plddt:Tensor=None)->Tensor:
                    plddt.requires_grad=False
                    losses:Tensor=_criterion(pred.permute(0, 2, 1),label)
                    plddt=plddt**self.plddt_param
                    mean_weighted_loss = (losses).mean()
                    return mean_weighted_loss
            else:
                raise ValueError(f'invalid `plddt_param`:{self.plddt_param}')
            
            # def _c(pred:Tensor,label:Tensor,plddt:Tensor=None)->Tensor:
            #     plddt.requires_grad=False
            #     losses:Tensor=_criterion(pred.permute(0, 2, 1),label)
            #     if self.plddt_strategy=='mask':
            #         plddt=plddt.masked_fill(plddt>=self.plddt_param,1.)
            #         plddt=plddt.masked_fill(plddt<self.plddt_param,0.)
            #         # plddt[plddt>=self.plddt_param]=1.
            #         # plddt[plddt<=self.plddt_param]=0.
            #     elif self.plddt_strategy=='weight':
            #         plddt=plddt**self.plddt_param
            #     mean_weighted_loss = (losses).mean()
            #     return mean_weighted_loss
        self.criterion=_c
    
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
        _prefix_map={'fit':'train/','validate':'val/','test':None}
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
    
    # def metrics_update(self,k:str,pred:torch.Tensor,label:torch.Tensor):
    #     mask=(label!=self.ignore_index).reshape(-1)
    #     num_classes=pred.shape[-1]
    #     valid_pred=pred.reshape(-1,num_classes)[mask].detach()
    #     valid_label=label.reshape(-1)[mask].detach()
    #     self.metrics[k].update(valid_pred, valid_label)
        
    # def metrics_agg(self,split='train',on_step=True):
    #     metrics_dict={}
    #     for k,metrics in self.metrics.items():
    #         result = metrics.compute()
    #         for name, score in result.items():
    #             metrics_dict[f'{split}/{k}_{name}']=score
    #         metrics.reset()
    #     self.log_dict(metrics_dict,on_step=on_step, sync_dist=True)
    
    def training_step(self, batch, batch_idx):
        output:Dict[str,Tensor]=self.inner_model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            position_ids=batch['position_ids']
            )
        losses={}
        loss=0
        for k,pred in output.items():
            label=self._mask_nan(batch,pred,batch[f'{k}_ids'])
            # label:torch.Tensor=batch[f'{k}_ids']
            # if torch.isnan(pred).any():
            #     label[torch.isnan(pred[k]).any(dim=-1)]=self.ignore_index
            #     fail_idx=torch.where(torch.isnan(pred).any(-1).any(-1))[0].tolist()
            #     #tmp
            #     self.dumblogger.error('nan loss'+'\t'.join(
            #         [f'{batch['stem'][i]}:{batch['idx_b'][i].item()}' 
            #         for i in fail_idx]))
            plddt=batch['pLDDT'] if self.plddt_strategy is not None else None
            l_:torch.Tensor=self.criterion(pred,label,plddt)
            if torch.isnan(l_):
                self.dumblogger.error(f'still nan loss: {"\t".join(batch['stem'])}')
                l_=torch.tensor(5.0, requires_grad=True,device=self.device)
            loss+=l_
            losses[f'train/{k}_loss']=l_.item()
            # self.metrics_update(k,pred,label)
            metrics:MulticlassMetricCollection=getattr(self,f'metrics_fit_{k}')
            metrics(pred, label,plddt)
            self.log_dict(metrics)
        losses['train/loss']=loss.item()
        self.log_dict(losses, sync_dist=True)
        
        # if self.trainer.global_step%self.metrics_interval==0:
        #     self.metrics_agg()
        return loss

    def _mask_nan(self,batch:dict,pred:torch.Tensor,label:torch.Tensor):
        if torch.isnan(pred).any():
            mask=torch.isnan(pred).any(dim=-1)
            label=label.masked_fill(mask,self.ignore_index)
            # label[torch.isnan(pred).any(dim=-1)]=self.ignore_index
            fail_idx=torch.where(mask.any(dim=-1))[0].tolist()
            self.dumblogger.error('nan loss'+'\t'.join([f'{batch['stem'][i]}:{batch['idx_b'][i].item()}' 
                for i in fail_idx]))
        return label
    
    # def _shared_eval(self, batch, batch_idx, prefix):
    #     '''
    #     legacy, to be removed
    #     '''
    #     output:Dict[str,Tensor]=self.inner_model(
    #         input_ids=batch['input_ids'],
    #         attention_mask=batch['attention_mask'],
    #         position_ids=batch['position_ids']
    #         )
    #     losses={}
    #     loss=0
    #     for k,pred in output.items():
    #         label=batch[f'{k}_ids']
    #         label=self._mask_nan(batch,pred,label)
    #         # if torch.isnan(pred).any():
    #         #     label[torch.isnan(pred).any(dim=-1)]=self.ignore_index
    #         #     fail_idx=torch.where(torch.isnan(pred).any(-1).any(-1))[0].tolist()
    #         #     self._dumblogger.log('nan loss'+'\t'.join([f'{batch['stem'][i]}:{batch['idx_b'][i].item()}' 
    #         #         for i in fail_idx]))
    #         plddt=batch['pLDDT'] if self.plddt_strategy is not None else None
    #         l_:torch.Tensor=self.criterion(pred.permute(0, 2, 1),label,plddt)
    #         if torch.isnan(l_):
    #             l_=torch.tensor(5.0, requires_grad=True,device=self.device)
    #         loss+=l_
    #         losses[f'{prefix}/{k}_loss']=l_.item()
    #         losses[f'{prefix}_loss']=loss.item()
    #         # self.metrics_update(k,pred,label)
    #         self.metrics['validate'][k].update(pred, label,plddt)
            
    #     losses[f'{prefix}/loss']=loss.item()
    #     self.log_dict(losses,sync_dist=True)

    def validation_step(self, batch, batch_idx):
        prefix='val'
        output:Dict[str,Tensor]=self.inner_model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            position_ids=batch['position_ids']
            )
        losses={}
        loss=0
        for k,pred in output.items():
            label=self._mask_nan(batch,pred,batch[f'{k}_ids'])
            plddt=batch['pLDDT'] if self.plddt_strategy is not None else None
            l_:torch.Tensor=self.criterion(pred,label,plddt)
            if torch.isnan(l_):
                l_=torch.tensor(5.0,device=self.device)
            loss+=l_
            losses[f'{prefix}/{k}_loss']=l_.item()
            # self.metrics_update(k,pred,label)
            metrics:MulticlassMetricCollection=getattr(self,f'metrics_validate_{k}')
            metrics(pred, label,plddt)
            self.log_dict(metrics)
        losses[f'{prefix}_loss']=loss.item()
        losses[f'{prefix}/loss']=loss.item()
        self.log_dict(losses,sync_dist=True)
        # self._shared_eval(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        output:Dict[str,Tensor]=self.inner_model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            position_ids=batch['position_ids']
            )
        losses={}
        loss=0
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
                l_=torch.tensor(5.0, requires_grad=False)
            loss+=l_
            losses[f'{k}_loss']=l_.item()
            metrics:MulticlassMetricCollection=getattr(self,f'metrics_test_{k}')
            metrics(pred, label,plddt)
            self.log_dict(metrics)
            for i,stem in enumerate(batch['stem']):
                pred_=pred[i]
                label_=label[i]
                mask=batch['attention_mask'][i].bool()&(label_!=self.ignore_index)
                raw_opts[stem][f'{k}_pred']=torch.softmax(pred_[mask],dim=-1).to('cpu')
                raw_opts[stem][f'{k}_label']=label_[mask].to('cpu')

        losses[f'loss']=loss.item()
        self.log_dict(losses,sync_dist=True)
        self.playground.update(raw_opts)
        
    def configure_optimizers(self):
        optimizer:torch.optim.Optimizer=getattr(torch.optim,
                self.optimizer)(params=self.inner_model.parameters(), **self.optimizer_kwargs)
        sk=self.scheduler_kwargs
        
        wu_iter=int(sk['warm_up_iter'])
        wu_rate=sk['warm_up_rate']
        assert wu_iter>0
        # if wu_iter==1:
        #     scheduler = SequentialLR(optimizer=optimizer,
        #         schedulers=[ConstantLR(optimizer, 
        #         factor=wu_rate, 
        #         total_iters=1),
        #         ExponentialLR(optimizer, gamma=sk['exp_gamma'])],milestones=[wu_iter])
        # else:
        r=(1/sk['warm_up_rate'])**(1/wu_iter)
        schedulers=[]
        for i in range(wu_iter):
            schedulers.append(ConstantLR(optimizer, 
            factor=wu_rate, total_iters=1))
            wu_rate=wu_rate*r
        schedulers.append(ExponentialLR(optimizer, gamma=sk['exp_gamma']))
        scheduler = SequentialLR(optimizer=optimizer,
            schedulers=schedulers,milestones=list(range(1,wu_iter+1)))
        return {"optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": sk['update_step'],
                    "monitor": "train/loss",
                    "strict": False,
                },}
        
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
    

class MetricsAggCallback(Callback):
    def setup(self, trainer:Trainer, pl_module:EsmTokenMhClassifier, stage:str):
        for k,metrics in pl_module.metrics.items():
            metrics.to('cpu')
    
    def _share_agg(self, pl_module:EsmTokenMhClassifier,split:str,on_step=True):
        for k,metrics in pl_module.metrics.items():
            if len(metrics.inputs)>0:
                pl_module.metrics_agg(split,on_step)
                
    def on_train_end(self, trainer:Trainer, pl_module:EsmTokenMhClassifier):
        self._share_agg(pl_module,'train')
        
    # def on_validation_epoch_start(self, trainer:Trainer, pl_module:EsmTokenMhClassifier):
    #     self._share_agg(pl_module,'train')
    def on_test_epoch_start(self, trainer:Trainer, pl_module:EsmTokenMhClassifier):
        self._share_agg(pl_module,'train')
        
    def on_validation_epoch_end(self, trainer:Trainer, pl_module:EsmTokenMhClassifier):
        self._share_agg(pl_module,'val',on_step=False)

    def on_test_epoch_end(self, trainer:Trainer, pl_module:EsmTokenMhClassifier):
        self._share_agg(pl_module,'test',on_step=False)
        pl_module.clear_playground()
        
    def on_test_epoch_start(self, trainer:Trainer, pl_module:EsmTokenMhClassifier):
        pl_module.init_playground()