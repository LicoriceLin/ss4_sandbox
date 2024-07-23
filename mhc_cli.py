# %%
from lightning.pytorch.cli import LightningCLI
from mhclassdataset import MHClassDatasetModule
from mhclassmodel import EsmTokenMhClassifier
import torch

# %%
def cli_main():
    cli = LightningCLI(datamodule_class=MHClassDatasetModule, 
                       model_class=EsmTokenMhClassifier,
                       parser_kwargs={"parser_mode": "omegaconf"}
                       )


if __name__ == "__main__":
    with torch.autograd.set_detect_anomaly(True):
        cli_main()