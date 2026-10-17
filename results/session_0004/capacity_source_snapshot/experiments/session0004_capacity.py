"""Fresh confirmation for the discovered active-parameter/compute confound."""

import datetime
import json
from pathlib import Path
import numpy as np
import torch

from experiments.session0004 import ROOT,tensor,fit,measured,digest,write,gdata,restore,preserve
from experiments.session0004_transport import TransportCoGeneration,data as transport_data
from src.operators.session0004 import CoGenerationModel
from src.physics.session0004 import diagnostics


def load_original(info):
    if info['family']=='G':return restore(info)
    m=TransportCoGeneration(mode=info['mode']).double()
    m.load_state_dict(torch.load(ROOT/info['checkpoint'],weights_only=True)['state']);return m


def main():
    if (ROOT/'capacity_confirmation.json').exists():raise FileExistsError('follow-up already evaluated')
    snapshots={}
    for name in ['experiments/session0004_capacity.py','research/session_0004/capacity_control_protocol.md']:
        target=ROOT/'capacity_source_snapshot'/name;target.parent.mkdir(exist_ok=True,parents=True)
        target.write_bytes(Path(name).read_bytes());snapshots[name]=digest(name)
    write('capacity_source_snapshot/sha256.json',snapshots)
    freshg,failures=gdata(32,45003);assert not failures
    fresh={'G':freshg,'H':transport_data(32,45004)};hashes={}
    for family,d in fresh.items():
        path=ROOT/'data'/f'{family}_capacity_confirm.npz'
        if path.exists():raise FileExistsError(path)
        np.savez_compressed(path,**d);hashes[family]=digest(path)
    write('capacity_dataset_manifest.json',{'sha256':hashes,'seeds':{'G':45003,'H':45004}})
    controls=[]
    for family in ['G','H']:
        bundles=[]
        for split in ['train','val']:
            with np.load(ROOT/'data'/f'{family}_{split}.npz') as d:bundles.append({k:tensor(d[k]) for k in d.files})
        ts=bundles[0]['target'].square().mean().sqrt();js=bundles[0]['flux'].square().mean().sqrt()
        for seed in [1004,1005,1006]:
            torch.manual_seed(seed)
            m=(CoGenerationModel if family=='G' else TransportCoGeneration)(mode='direct',width=64).double()
            m.center.copy_(bundles[0]['inputs'].mean(0));m.scale.copy_(bundles[0]['inputs'].std(0).clamp_min(.1))
            def loss(model,val):
                d=bundles[int(val)];u,j=model(d['inputs'])
                return ((u-d['target'])/ts).square().mean()+((j-d['flux'])/js).square().mean()
            info=fit(m,loss,3000,seed,f'{family}_direct_width64_{seed}.pt')
            info.update(family=family,mode='direct_width64',active_parameter_count=8415)
            controls.append(info);print(json.dumps({k:info[k] for k in ['family','seed','seconds','best_validation_loss']}),flush=True)
    write('capacity_frozen.json',{'models':controls,'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    originals=[m for m in json.loads((ROOT/'frozen_models.json').read_text())['models'] if m['family']=='G']+json.loads((ROOT/'transport_frozen.json').read_text())['models']
    rows=[]
    with torch.no_grad():
        for info in originals+controls:
            family=info['family'];mode=info['mode'];d=fresh[family]
            if mode=='direct_width64':
                m=(CoGenerationModel if family=='G' else TransportCoGeneration)(mode='direct',width=64).double()
                m.load_state_dict(torch.load(ROOT/info['checkpoint'],weights_only=True)['state'])
            else:m=load_original(info)
            (u,j),timing=measured(lambda:tuple(v.numpy() for v in m(tensor(d['inputs']))))
            relative=np.linalg.norm(u-d['target'],axis=1)/np.maximum(np.linalg.norm(d['target'],axis=1),1e-14)
            rr=[];bb=[]
            for z,t,jj in zip(d['inputs'],u,j):
                if family=='G':
                    diag=diagnostics(np.exp(z[:16]),z[16:31],z[31:33],z[-1],t,jj)
                    rr.append(diag['pde_residual_inf']);bb.append(diag['generated_energy_balance_abs'])
                else:
                    full=np.r_[z[31],t,z[32]];v=z[-1]
                    physical=max(v,0)*full[:-1]+min(v,0)*full[1:]-np.exp(z[:16])*np.diff(full)*16
                    rr.append(max(abs(np.diff(physical)*16-z[16:31])));bb.append(abs(jj[-1]-jj[0]-z[16:31].sum()/16))
            row={'family':family,'mode':mode,'seed':info['seed'],'mean_relative_l2':float(relative.mean()),
                 'mean_pde_residual_inf':float(np.mean(rr)),'max_generated_energy_balance_abs':float(max(bb)),
                 'max_absolute_error':float(abs(u-d['target']).max()),'input_sha256':hashes[family],
                 'checkpoint':info['checkpoint'],'checkpoint_sha256':info['sha256'],**timing}
            path=ROOT/'arrays'/f'capacity_{family}_{mode}_{info["seed"]}.npz'
            np.savez_compressed(path,**d,prediction=u,predicted_flux=j);row['arrays']=str(path.relative_to(ROOT));rows.append(row)
    write('capacity_confirmation.json',{'records':rows,'evaluated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'prior_files_unchanged':preserve(),'scope':'new confirmation after the original test revealed a capacity confound; original checkpoints unchanged'})
    print(json.dumps({'followup_models':len(controls),'fresh_confirmation_rows':len(rows)}))


if __name__=='__main__':
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1);main()
