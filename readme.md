To run a test:

python pred.py predict \
-c pred.yaml \
--data.dataset_path=data/small-ictv.fasta \ #input fasta
--data.preprocess_to=data/small-ictv \ # the intermediate db file. note: the process could be accelerated by multiprocess, but I haven't implemented yet
--trainer.default_root_dir=train/debug_pred-2 #where to find your output,note the dir must be created before running

to inspect the output:

```
from pred import plot_output
import torch
o=torch.load('train/debug_pred-2/raw_output.pt')
fig,ax=plot_output(o,'EyV|Seg3|AF282469:VP3-like:52') #key for the entry you want to plot
fig.show()
```
