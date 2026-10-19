"""C0b with a fresh evaluation set after the original C0 was observed."""
import json
import numpy as np
import torch
from experiments.session0005_common import ROOT,SEEDS,tensor,write,sha,now,preserve,snapshot,save_data,load_data,measure,fit
from experiments.session0005_controls import build,metrics
from experiments.session0004 import gdata
from experiments.session0004_transport import data as transport_data
from src.operators.session0005_normalized import NormalizedControl


def main():
    if (ROOT/'normalized_confirmation.json').exists():raise FileExistsError('normalized follow-up already evaluated')
    snapshot('normalized_control',['experiments/session0005_normalized.py','src/operators/session0005_normalized.py','research/session_0005/normalized_control_protocol.md'])
    dg,fail=gdata(64,51203);assert not fail
    fresh={'G':dg,'H':transport_data(64,51303)};hashes={f:save_data(f'C0b_{f}_confirm',d) for f,d in fresh.items()}
    write('normalized_dataset_manifest.json',{'sha256':hashes,'prepared_utc':now(),'seeds':{'G':51203,'H':51303}})
    models=[]
    for family in ['G','H']:
        ds=[{k:tensor(v) for k,v in load_data(f'C0_{family}_{s}').items()} for s in ['train','val']]
        us=ds[0]['target'].square().mean().sqrt();js=ds[0]['flux'].square().mean().sqrt()
        for kind in ['norm_physical','norm_generic','norm_static']:
            for seed in SEEDS:
                torch.manual_seed(seed);m=NormalizedControl(kind,family).double()
                m.center.copy_(ds[0]['inputs'].mean(0));m.scale.copy_(ds[0]['inputs'].std(0).clamp_min(.1))
                def loss(m,val):
                    d=ds[int(val)];u,j=m(d['inputs']);return ((u-d['target'])/us).square().mean()+((j-d['flux'])/js).square().mean()
                info=fit(m,loss,1000,f'C0b_{family}_{kind}_{seed}',seed);info.update(family=family,kind=kind)
                models.append(info);print(json.dumps({k:info[k] for k in ['family','kind','seed','seconds','best_validation_loss']}),flush=True)
    write('normalized_frozen.json',{'models':models,'frozen_utc':now()})
    original=json.loads((ROOT/'controls_frozen.json').read_text())['models'];rows=[]
    with torch.no_grad():
        for info in original+models:
            family=info['family'];kind=info['kind'];d=fresh[family]
            m=NormalizedControl(kind,family).double() if kind.startswith('norm_') else build(kind,family)
            assert sha(ROOT/info['checkpoint'])==info['sha256'];m.load_state_dict(torch.load(ROOT/info['checkpoint'],weights_only=True)['state'])
            (u,j),cost=measure(lambda:tuple(z.numpy() for z in m(tensor(d['inputs']))))
            row={'family':family,'kind':kind,'seed':info['seed'],'case':'fresh_iid',**metrics(d['inputs'],u,j,d['target'],family),**cost,
                 'input_sha256':hashes[family],'checkpoint':info['checkpoint'],'checkpoint_sha256':info['sha256']}
            path=ROOT/'arrays'/f'C0b_{family}_{kind}_{info["seed"]}.npz';np.savez_compressed(path,**d,prediction=u,predicted_flux=j)
            row['arrays']=str(path.relative_to(ROOT));rows.append(row)
    write('normalized_confirmation.json',{'records':rows,'evaluated_utc':now(),'scope':'fresh follow-up after C0 observed, no original checkpoint modifications','prior_files_preserved':preserve()})
    print(json.dumps({'normalized_new_models':len(models),'fresh_prediction_batches':len(rows)}))


if __name__=='__main__':
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1);main()
