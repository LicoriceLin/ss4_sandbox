from lightning.pytorch.cli import LightningCLI
from mhclassdataset import MHClassDatasetModule
from mhclassmodel import EsmTokenMhClassifier

def cli_main():
    cli = LightningCLI(datamodule_class=MHClassDatasetModule, 
                       model_class=EsmTokenMhClassifier,
                       )


if __name__ == "__main__":
    cli_main()