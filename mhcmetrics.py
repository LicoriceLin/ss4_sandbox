from torchmetrics import Metric,MetricCollection
from torchmetrics.functional.classification import (
    multiclass_accuracy,
    multiclass_precision_recall_curve,
    multiclass_auroc
    )
from torchmetrics.utilities import dim_zero_cat
from torchmetrics.functional.classification.auroc import _reduce_auroc
from typing import Optional,List,Union,Literal,Callable,Dict
import torch
from torch import Tensor
from functools import partial

from torchmetrics.classification import Accuracy
def multiclass_auprc(    
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    thresholds: Optional[Union[int, List[float], Tensor]] = None,
    average: Optional[Literal["micro", "macro"]] = None,
    ignore_index: Optional[int] = None,
    validate_args: bool = True):
    (precision, recall, thresh
        )=multiclass_precision_recall_curve(
        preds,target,num_classes,thresholds,
        average,ignore_index,validate_args)
    
    if average in ['micro','macro']: average_='none'
    else: average_='macro'
    # import pdb;pdb.set_trace()
    return _reduce_auroc(precision.reshape(1,-1),
            recall.reshape(1,-1),average_)

class MulticlassMetricBase(Metric):
    preds:List[Tensor]
    labels:List[Tensor]
    # pLDDTs:List[Tensor]

    def __init__(self, 
            num_classes:int,
            func_dict:Dict[str,Callable[[Tensor,Tensor],Tensor]],
            ignore_index:int=-100,
            plddt_threshold:float=0.,
            prefix:str=''
            ):
        super().__init__()
        self.num_classes=num_classes
        self.func_dict=func_dict
        self.ignore_index=ignore_index
        self.plddt_threshold=plddt_threshold
        self.prefix=prefix
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("labels",default=[], dist_reduce_fx="cat")
        # self.add_state("pLDDTs",default=[], dist_reduce_fx="cat")

    @torch.inference_mode()
    def update(self, 
               pred: Tensor, label: Tensor,
               pLDDT: Optional[Tensor]=None
               ) -> None:
        mask=(label!=self.ignore_index).reshape(-1)
        if pLDDT is not None:
            mask=mask & (pLDDT>=self.plddt_threshold).reshape(-1)
        label=label.reshape(-1).masked_select(mask)
        valid_pred=pred.reshape(-1,pred.shape[-1])[mask].half()
        self.preds.append(valid_pred)
        self.labels.append(label)
        del pLDDT
    
    @torch.inference_mode()
    def compute(self):
        preds=dim_zero_cat(self.preds)
        labels=dim_zero_cat(self.labels)
        return {self.prefix+k:func(preds,labels) for k,func in self.func_dict.items()}
    

class MulticlassMetricCollection(MetricCollection):
    def __init__(self,
        num_classes:int,
        head_name:str='',
        ignore_index:int=-100,
        plddt_threshold:float=0.,
        **kwargs):
        metrics={}
        func_dict={
            'accuracy':multiclass_accuracy,
            'auroc':multiclass_auroc,
            'auprc':multiclass_auprc
        }
        func_dict={k:partial(func,
                num_classes=num_classes,
                average='macro'
                ) for k,func in func_dict.items()}
        
        metrics[f'0-{head_name}']=MulticlassMetricBase(
            num_classes,func_dict,ignore_index,0.,f'0/{head_name}-')
        metrics[f'{head_name}']=MulticlassMetricBase(
            num_classes,func_dict,ignore_index,plddt_threshold,
            f'{int(plddt_threshold*100)}/{head_name}-')
        # for name,_func in {
        #     'accuracy':multiclass_accuracy,
        #     'auroc':multiclass_auroc,
        #     'auprc':multiclass_auprc
        # }.items():
        #     func=partial(_func,
        #         num_classes=num_classes,
        #         average='macro'
        #         #TODO more flexibility?
                # )

            # if plddt_threshold>0:
            #     metrics[f'{head_name}-{name}-thresholded']=MulticlassMetricBase(
            #         num_classes,func,ignore_index,plddt_threshold)
            # metrics[f'{head_name}-{name}']=MulticlassMetricBase(
            #     num_classes,func,ignore_index,0.)
        super().__init__(metrics,**kwargs)
