#%%
from Bio.SeqIO import read
from subprocess import run
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import gzip
import shutil
ALPHAS=['Mu','Conf3','Conf4','Conf16','NENConf16','RENConf16','NENDist16','RENConf16']

def cal_mu(infile:str,
           mubin='/home/rnalab/zfdeng/ss4/ss4_sandbox/reseek',
           alphas=ALPHAS):
    infile:Path=Path(infile).absolute()
    stem=infile.stem
    mubin =Path(mubin).absolute()
    output={'stem':stem}
    try:
        with TemporaryDirectory() as tdir:
            infile_=stem+'.pdb'
            with gzip.open(infile, 'rb') as gz_file:
                with open(tdir+'/'+infile_, 'wb') as out_file:
                    shutil.copyfileobj(gz_file, out_file)

            _=run([mubin,
                '-pdb2fasta',
                infile_,
                '-output',
                'structs.fa'],cwd=tdir,capture_output=True)
            output['seq']=str(read(f'{tdir}/structs.fa',format='fasta').seq)
            
            for i in alphas:
                _=run([mubin,
                    '-pdb2alpha',
                    infile_,
                    '-alpha',
                    i,
                    '-output',
                    'structs.fa'],cwd=tdir,capture_output=True)
                output[i]=str(read(f'{tdir}/structs.fa',format='fasta').seq)
    except:
        for i in ALPHAS+['seq']:
            output[i]=''
    return output
    
#%%
if __name__=='__main__':
    from multiprocessing import Pool
    from glob import glob
    import tqdm
    import pickle as pkl
    import pandas as pd
    pdbs=glob('data/swiss/*pdb.gz')
    bar=tqdm.tqdm(total=len(pdbs))
    opts=[]
    def update_bar(x):
        bar.update()
    pool=Pool(processes=24,maxtasksperchild=300)
    # def tmp_fn(x):
    #     cal_mu(x)
    #     bar.update()
    results = [pool.apply_async(cal_mu, args=(i,),callback=update_bar) for i in pdbs]
    # for i in pdbs:
    #     pool.apply_async(cal_mu,(i,),callback=update_bar)
    pool.close()
    pool.join()
    # result =pool.map_async(tmp_fn,pdbs)
    opts = [result.get() for result in results]


    try:
        pd.DataFrame(opts).to_pickle('reseek_AF_SIWSS.pkl')
    except:
        pkl.dump(opts,open('af-swiss.pkl','wb'))