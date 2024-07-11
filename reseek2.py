#%%
from Bio.SeqIO import read
from subprocess import run
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import gzip
import shutil
from Bio.PDB import PDBParser
from Bio.PDB.Residue import Residue
from Bio.Data.PDBData import protein_letters_3to1_extended as p3to1
from typing import List,Tuple
from datasets import Dataset 
# %%
def fetch_seq_plddt(infile:str)->Tuple[str,List[float]]:
    struct=PDBParser(QUIET=True).get_structure(id='B1YQL3',file=infile)
    assert len(list(struct.get_chains()))==1,'only accept single chain by now'
    seq=[]
    pLDDT=[]
    for res in struct.get_residues():
        res:Residue
        seq.append(p3to1.get(res.resname,'X'))
        try:
            plddt=res.child_list[0].bfactor
        except:
            plddt=0.0
        pLDDT.append(plddt)
    return ''.join(seq), pLDDT 

ALPHAS2=['Conf3','NENConf3','RENDist4','RevNbrDist4','RENConf16']

def cal_mu_v2(infile:str,
              mubin='/home/rnalab/zfdeng/ss4/ss4_sandbox/reseek2',
              alphas=ALPHAS2):
    infile:Path=Path(infile).absolute()
    stem,suffix=infile.stem,infile.suffix
    mubin =Path(mubin).absolute()
    output={'stem':stem}
    try:
        with TemporaryDirectory() as tdir:
            if suffix=='.gz':
                r_pdbfile_path=stem
                a_pdbfile_path=tdir+'/'+r_pdbfile_path
                with gzip.open(infile, 'rb') as gz_file:
                    with open(a_pdbfile_path, 'wb') as out_file:
                        shutil.copyfileobj(gz_file, out_file)
            else:
                a_pdbfile_path=r_pdbfile_path=infile
            output['seq'],output['pLDDT']=fetch_seq_plddt(a_pdbfile_path)
            for i in alphas:
                _=run([mubin,
                    '-convert',
                    r_pdbfile_path,
                    '-alpha',
                    i,
                    '-fasta',
                    'structs.fa'],cwd=tdir,capture_output=True)
                # return _
                output[i]=str(read(f'{tdir}/structs.fa',format='fasta').seq)
    except:
        output['seq'],output['pLDDT']='',''
        for i in alphas+['seq']:
            output[i]=''
    return output


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
    pool=Pool(processes=32,maxtasksperchild=300)
    # def tmp_fn(x):
    #     cal_mu(x)
    #     bar.update()
    results = [pool.apply_async(cal_mu_v2, args=(i,),callback=update_bar) for i in pdbs]
    # for i in pdbs:
    #     pool.apply_async(cal_mu,(i,),callback=update_bar)
    pool.close()
    pool.join()
    # result =pool.map_async(tmp_fn,pdbs)
    opts = [result.get() for result in results]


    try:
        # pd.DataFrame(opts).to_pickle('data/reseek_AF_SIWSS-v2.pkl')
        dataset=Dataset.from_list(opts)
        dataset.save_to_disk('data/reseek_AF_SIWSS-v2')
    except:
        pkl.dump(opts,open('data/raw-reseek_AF_SIWSS-v2.pkl','wb'))

#%%
# from datasets import Dataset 
# ds=Dataset.load_from_disk('data/reseek_AF_SIWSS-v2')
# %%
