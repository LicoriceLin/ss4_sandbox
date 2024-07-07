from matplotlib.colors import LinearSegmentedColormap,Colormap
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages
from torch import nn 
from typing import List
from torch.nn import functional as F
import torch
import numpy as np
from typing import Optional,Dict,Union

to_cmap_pair=lambda c1,c2: dict(cmap_pred=LinearSegmentedColormap.from_list('pred',[(1,1,1,1),c1]),
        cmap_label=LinearSegmentedColormap.from_list('label',[(1,1,1,1),c2]))
optional_cmap_pairs={
    'default/blue_green_yellow':to_cmap_pair((0,1,1,1),(1,1,0,1)),
    'spring/red_grey_green':to_cmap_pair((1,0,0.5,1),(0,1,0.5,1)),
    'laser/rosein_cloudyblue_cyan':to_cmap_pair((1,0,1,1),(0,1,1,1)),
    'sunset/pink_coral_yellow':to_cmap_pair((1,0,1,1),(1,1,0,1)),
    'sharp/blue_grey_red':to_cmap_pair((0,1,1,1),(1,0,0,1))
}
cmap_pred,cmap_label=optional_cmap_pairs['default/blue_green_yellow'].values()

def plot_stack_heatmap(pred:torch.Tensor,label:torch.Tensor,
        label_maps:Optional[Dict[str,int]]=None,
        cmap_pred:Colormap=cmap_pred,
        cmap_label:Colormap=cmap_label,
        ax:Optional[Axes]=None,
        height:int=10):
    #warning: pred should be softmaxed first
    h,w=pred.shape
    if ax is None:
        fig,ax=plt.subplots(1,1,figsize=(height/w*h,height))
    # ax:Axes
    num_classes=pred.shape[-1]
    label_mat=F.one_hot(label,num_classes=num_classes)
    pred_mat=pred
    ax.pcolormesh((cmap_pred(pred_mat.T)+cmap_label(label_mat.float().T))/2,rasterized=True)
    ax.set_xlabel('Residue Position')
    ax.set_ylabel('Struct Token')
    if label_maps is not None:
        ax.set_yticks(list(label_maps.values()),
            list(label_maps.keys()))
    fig.tight_layout()
    return ax.figure,ax
    
def plot_color_scheme(
        cmap_pred:Colormap=cmap_pred,
        cmap_label:Colormap=cmap_label,
        height:int=10):
    fig,ax=plt.subplots(1,1,figsize=(height,height))
    ax:Axes
    pred_mat=torch.linspace(0,1,steps=10).unsqueeze(1).repeat(1,10)
    label_mat=torch.linspace(0,1,steps=2).unsqueeze(1).repeat_interleave(5,0).repeat_interleave(10,1).T
    ax.pcolormesh((cmap_pred(pred_mat)+cmap_label(label_mat))/2,label='pred')
    ax.set_xticks([2.5,7.5],['False','True'])
    ax.set_xlabel('true_label')
    ax.set_ylabel('pred_prob')
    ax.set_yticks(torch.linspace(0,10,steps=11).numpy(),torch.linspace(0,1,steps=11).numpy())
    return fig,ax

def cal_confusion_matrix(pred:torch.Tensor,label:torch.Tensor,soft:bool=True):
    num_classes=pred.shape[-1]
    confusion = torch.zeros(num_classes, num_classes)
    if not soft:
        pred=F.one_hot(torch.argmax(pred,dim=-1),num_classes=num_classes).float()
    confusion.index_add_(0, label.squeeze(), pred)
    return confusion

def norm_confusion_matrix(confusion_matrix:torch.Tensor,eps=1e-8):
    confusion_matrix=confusion_matrix+eps
    return confusion_matrix/confusion_matrix.sum(dim=1).unsqueeze(1)
    
def plot_confusion_matrix(confusion_matrix:torch.Tensor,
                          token_cluster:bool=False,
                          label_maps:Optional[Dict[str,int]]=None,
                          ax:Optional[Axes]=None,
                          cmap:Union[str,Colormap]='Greens',
                          cbar:bool=False,
                          **kwargs):
    if token_cluster:
        raise NotImplementedError('do it later ~')
    
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(15,15))
        
    r_=np.arange(confusion_matrix.shape[0])
    if label_maps is None:
        labels=r_
    else:
        r_label_maps={v:k for k,v in label_maps.items()}
        labels=[r_label_maps[i] for i in r_]
        
    sns.heatmap(confusion_matrix,ax=ax,rasterized=True,
                xticklabels=labels,yticklabels=labels,
                cmap=cmap,cbar=cbar,**kwargs)
    return ax.figure,ax

def plot_accuracy(confusion_matrix:torch.Tensor,
                  label_maps:Optional[Dict[str,int]]=None,
                  color_pair:tuple=('grey','red'),
                  ax:Optional[Axes]=None,):
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(15,15))
    
    r_=np.arange(confusion_matrix.shape[0])
    if label_maps is None:
        labels=r_
    else:
        r_label_maps={v:k for k,v in label_maps.items()}
        labels=[r_label_maps[i] for i in r_]
        
    sum_labels=confusion_matrix.sum(axis=0).tolist()
    idx=torch.arange(len(sum_labels))
    correct_labels=confusion_matrix[idx,idx].tolist()
    correct_labels=confusion_matrix[idx,idx].tolist()
    ax.bar(idx, sum_labels, label='Total Samples', color=color_pair[0], width=0.5)
    ax.bar(idx, correct_labels, label='Correct Predictions', color=color_pair[1], width=0.5)
    ax.set_xlabel('Labels')
    ax.set_ylabel('Number of Samples')
    ax.set_xticks(idx,labels)
    ax.legend()
    eps=1e-8
    for i,(c,s) in enumerate(zip(correct_labels,sum_labels)):
        prob=(c+eps)/(s+eps)
        height=s
        ax.text(i,height,f'{prob:.2f}',ha='center',va='bottom',fontsize=8)
    # ax.set_title('Total vs Correctly Predicted Samples by Label')
    return ax.figure,ax
