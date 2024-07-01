from torcheval.metrics import Metric
from collections.abc import Callable, Iterable
from torch import distributed as dist
from torcheval.metrics import functional as tef
import torch
from functools import partial

class Metrics(Metric):
    def __init__(self, **metrics):
        super().__init__()
        self._add_state("inputs", [])
        self._add_state("targets", [])
        self.metrics = metrics
        
    @torch.inference_mode()
    def update(self, input, target):  # pylint: disable=W0221, W0622
        # pylint: disable=W0201
        if isinstance(input, torch.Tensor):
            input, target = input.to(self.device), target.to(self.device)
        self.inputs.append(input)
        self.targets.append(target)
        return self
    
    @torch.inference_mode()
    def compute(self):
        return {name: metric(self.input, self.target).item() for name, metric in self.metrics.items()}

    @torch.inference_mode()
    def merge_state(self, metrics: Iterable):
        for metric in metrics:
            if metric.inputs:
                metric_inputs = torch.cat(metric.inputs, -1).to(self.device)
                metric_targets = torch.cat(metric.targets, -1).to(self.device)
                self.inputs.append(metric_inputs)
                self.targets.append(metric_targets)
        return self

    @torch.inference_mode()
    def sync(self):
        # pylint: disable=E0203, W0201
        synced_inputs = [None for _ in dist.get_world_size()]
        dist.all_gather_object(synced_inputs, self.inputs)
        self.inputs = [i for j in synced_inputs for i in j]
        synced_targets = [None for _ in dist.get_world_size()]
        dist.all_gather_object(synced_targets, self.targets)
        self.targets = [i for j in synced_targets for i in j]
        return self

    @property
    def input(self):
        return torch.cat(self.inputs, 0)

    @property
    def target(self):
        return torch.cat(self.targets, 0)
    
class BaseMetric:
    metrics: Metrics
    index: str = "loss"
    best_fn: Callable = min

    def __repr__(self):
        keys = tuple(i for i in self.metrics.metrics.keys())
        return f"{self.__class__.__name__}{keys}"

    def __getattr__(self, name):
        return getattr(self.metrics, name)
    
class MulticlassMetric(BaseMetric):
    def __init__(self, num_classes: int = 1):
        super().__init__()
        auroc = partial(tef.multiclass_auroc, num_classes=num_classes)
        auprc = partial(tef.multiclass_auprc, num_classes=num_classes)
        acc = partial(tef.multiclass_accuracy, num_classes=num_classes)
        self.metrics = Metrics(**{"auroc": auroc, "auprc": auprc, "acc": acc})