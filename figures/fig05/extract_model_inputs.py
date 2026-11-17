"""Recover the exact W2V regression rows, count matrices and original MS4 source trains."""
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat


def recover(root, destination):
    destination.mkdir(parents=True,exist_ok=True)
    for session in (1,2):
        durations=loadmat(root/f'nets{session}_duration.mat')[f'nets{session}_duration'].ravel()
        path=next((root/'w2vemb').glob(f'*nets{session}*mat'))
        embeddings=next(v for key,v in loadmat(path).items() if not key.startswith('_'))
        original=[embeddings[:,:10],durations[:,None].astype(np.float32)]
        fitted=[]
        arrays={'duration_ms':durations,'w2v10':embeddings[:,:10]}
        for probe in (1,2):
            name=f'ses{session}probe{probe}'
            counts=name+'_counts'
            reg=loadmat(root/'regression'/f'ses{session}p{probe}_W2Vreg_10pca.mat')
            if probe==1:
                fitted.append(reg['X'])
            else:
                np.testing.assert_array_equal(fitted[0],reg['X'])
            full=loadmat(root/'counts'/f'{counts}.mat')[counts]
            original.append(full)
            fitted.append(reg[counts])
            arrays[f'probe{probe}_full_counts']=full
            arrays[f'probe{probe}_model_counts']=reg[counts]
            curated=loadmat(root/f'{name}_dredgeMS4.mat',squeeze_me=True,struct_as_record=False)['newResult']
            trains=[np.atleast_1d(t) for t in curated.time]
            prefix=f'probe{probe}_'
            arrays[prefix+'train_offsets']=np.r_[0,np.cumsum([len(x) for x in trains])]
            arrays[prefix+'spike_samples']=np.concatenate(trains)
            arrays[prefix+'noise_flags']=curated.noise
            arrays[prefix+'violation']=curated.violation
            # scipy squeeze makes one MATLAB cell a numeric array and >1 cells an object array.
            groups=np.atleast_1d(curated.merge)
            if groups.size:
                for k,group in enumerate(groups if groups.dtype==object else [groups]):
                    arrays[prefix+f'merge_group_{k+1}']=np.atleast_1d(group)
        lookup=defaultdict(list)
        for i,row in enumerate(np.column_stack(original).astype(np.float32)):
            lookup[row.tobytes()].append(i)
        rows=[]
        for source in np.column_stack(fitted).astype(np.float32):
            hits=lookup[source.tobytes()]
            if len(hits)!=1:
                raise ValueError(f'Model row matched {len(hits)} complete word rows')
            rows.append(hits[0])
        rows=np.array(rows)
        assert np.all(np.diff(rows)>0)
        np.testing.assert_array_equal(embeddings[rows,:10],fitted[0][:,:10])
        np.testing.assert_array_equal(np.asarray(durations[rows],np.float32),fitted[0][:,-1])
        for probe in (1,2):
            np.testing.assert_array_equal(arrays[f'probe{probe}_full_counts'][rows],arrays[f'probe{probe}_model_counts'])
        arrays['model_row_to_word_row_1based']=rows+1
        arrays['model_X']=fitted[0]
        omitted=np.setdiff1d(np.arange(len(durations)),rows)
        np.savez_compressed(destination/f'phase{session}_model.npz',**arrays)
        print(f'Phase {session} matched {len(rows)} words. One-based omissions: {omitted+1}')
        if session==1:
            summary=root/'regression/results/w2v/train_test/ses1probe2/ses1p2_10PCA_w2v_summary_per_neuron_NORMALIZED.csv'
            missing=np.setdiff1d(np.arange(arrays['probe2_full_counts'].shape[1]),pd.read_csv(summary)['neuron'])
            np.testing.assert_array_equal(missing,[27])
            assert not arrays['probe2_full_counts'][:,missing].any()


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('analysis_delivery',type=Path)
    cli.add_argument('out',type=Path)
    args=cli.parse_args()
    recover(args.analysis_delivery,args.out)
