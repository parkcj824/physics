"""Cycle 23: same learned co-generation block with directed constitutive physics."""

import datetime
import json
from pathlib import Path
import time
import numpy as np
from scipy.linalg import solve_banded
import torch

from experiments.session0004 import ROOT, tensor, fit, measured, digest, write, preserve
from src.operators.session0004 import CoGenerationModel


def transport_reference(nu,q,g,v):
    n=len(q);h=1/(n+1);vp=max(v,0.);vm=min(v,0.)
    band=np.zeros((3,n));band[1]=(nu[:-1]+nu[1:])/h**2+abs(v)/h
    band[0,1:]=-nu[1:-1]/h**2+vm/h;band[2,:-1]=-nu[1:-1]/h**2-vp/h
    rhs=q.copy();rhs[0]+=(nu[0]/h**2+vp/h)*g[0];rhs[-1]+=(nu[-1]/h**2-vm/h)*g[1]
    u=solve_banded((1,1),band,rhs);full=np.r_[g[0],u,g[1]]
    j=vp*full[:-1]+vm*full[1:]-nu*np.diff(full)/h
    return u,j


class TransportCoGeneration(CoGenerationModel):
    def generate(self,h,raw):
        diffusion_only=raw.clone();diffusion_only[:,-1]=0.
        return super().generate(h,diffusion_only)

    def forward(self,raw):
        h=torch.tanh(self.encode((raw-self.center)/self.scale))
        for _ in range(self.depth):
            u,j=self.generate(h,raw)
            if self.mode=='feedback':
                full=torch.cat([raw[:,31:32],u,raw[:,32:33]],1);v=raw[:,-1:]
                constitutive=v.clamp_min(0)*full[:,:-1]+v.clamp_max(0)*full[:,1:]-raw[:,:16].exp()*torch.diff(full,dim=1)*16
                r=j-constitutive
                h=torch.tanh(self.evolve(h)+self.mismatch(r)+torch.tanh(self.gate(h))*torch.tanh(self.reaction(r)))
            else:h=torch.tanh(self.evolve(h))
        return self.generate(h,raw)


def data(count,seed,case='iid'):
    rng=np.random.default_rng(seed);x=np.arange(1,16)/16;ex=(np.arange(16)+.5)/16
    inputs=[];targets=[];currents=[]
    for i in range(count):
        d=rng.uniform(.05,.15) if case!='lowdiff' else rng.uniform(.015,.025)
        nu=d*np.exp(rng.uniform(-.2,.2)*np.sin(2*np.pi*ex))
        v=float(rng.choice([-1,1])*rng.uniform(.5,1.5));g=rng.uniform(-.1,.2,2)
        q=rng.uniform(.2,2)*np.exp(-.5*((x-rng.uniform(.2,.7))/.12)**2)+rng.uniform(-.2,.3)
        if case=='direction':
            # Paired identical physical inputs except for velocity sign.
            if i%2==0:paired_d=rng.uniform(.05,.15)
            nu=np.full(16,paired_d);v=1. if i%2==0 else -1.;g=np.zeros(2);q=np.exp(-.5*((x-.3)/.1)**2)
        u,j=transport_reference(nu,q,g,v)
        inputs.append(np.r_[np.log(nu),q,g,v]);targets.append(u);currents.append(j)
    return dict(inputs=np.array(inputs),target=np.array(targets),flux=np.array(currents),x=x)


def main():
    if (ROOT/'transport_confirmation.json').exists():raise FileExistsError('transport already executed')
    start=time.perf_counter();hashes={}
    specs=[('train',128,44001,'iid'),('val',32,44002,'iid'),('confirm_iid',32,44003,'iid'),
           ('confirm_direction',32,44005,'direction'),('confirm_lowdiff',16,44004,'lowdiff')]
    datasets={}
    for name,count,seed,case in specs:
        d=data(count,seed,case);datasets[name]=d;path=ROOT/'data'/('H_'+name+'.npz')
        if path.exists():raise FileExistsError(path)
        np.savez_compressed(path,**d);hashes[name]=digest(path)
    write('transport_dataset_manifest.json',{'sha256':hashes,'preparation_seconds':time.perf_counter()-start})
    snapshot={}
    for name in ['experiments/session0004_transport.py','src/operators/session0004.py','research/session_0004/transport_protocol.md']:
        dest=ROOT/'transport_source_snapshot'/name;dest.parent.mkdir(exist_ok=True,parents=True);dest.write_bytes(Path(name).read_bytes());snapshot[name]=digest(name)
    write('transport_source_snapshot/sha256.json',snapshot)
    bundles=[{k:tensor(v) for k,v in datasets[name].items()} for name in ['train','val']]
    ts=bundles[0]['target'].square().mean().sqrt();js=bundles[0]['flux'].square().mean().sqrt();models=[]
    for mode in ['feedback','no_feedback','direct']:
        for seed in [1004,1005,1006]:
            torch.manual_seed(seed);m=TransportCoGeneration(mode=mode).double()
            m.center.copy_(bundles[0]['inputs'].mean(0));m.scale.copy_(bundles[0]['inputs'].std(0).clamp_min(.1))
            def loss(m,val):
                d=bundles[int(val)];u,j=m(d['inputs'])
                return ((u-d['target'])/ts).square().mean()+((j-d['flux'])/js).square().mean()
            info=fit(m,loss,1500,seed,f'H_{mode}_{seed}.pt');info.update(family='H',mode=mode)
            models.append(info);print(json.dumps({k:info[k] for k in ['mode','seed','seconds','best_validation_loss']}),flush=True)
    write('transport_frozen.json',{'models':models,'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    records=[]
    with torch.no_grad():
        for info in models:
            m=TransportCoGeneration(mode=info['mode']).double();m.load_state_dict(torch.load(ROOT/info['checkpoint'],weights_only=True)['state'])
            for name in ['confirm_iid','confirm_direction','confirm_lowdiff']:
                d=datasets[name]
                (u,j),timing=measured(lambda:tuple(t.numpy() for t in m(tensor(d['inputs']))))
                errors=np.linalg.norm(u-d['target'],axis=1)/np.maximum(np.linalg.norm(d['target'],axis=1),1e-14)
                residuals=[];balances=[];defects=[]
                for raw,uu,jj in zip(d['inputs'],u,j):
                    nu=np.exp(raw[:16]);q=raw[16:31];g=raw[31:33];v=raw[-1];full=np.r_[g[0],uu,g[1]]
                    physical=max(v,0)*full[:-1]+min(v,0)*full[1:]-nu*np.diff(full)*16
                    residuals.append(max(abs(np.diff(physical)*16-q)))
                    balances.append(abs(jj[-1]-jj[0]-q.sum()/16));defects.append(max(abs(jj-physical)))
                row={'mode':info['mode'],'seed':info['seed'],'case':name.removeprefix('confirm_'),**timing,
                     'mean_relative_l2':float(errors.mean()),'max_absolute_error':float(abs(u-d['target']).max()),
                     'mean_pde_residual_inf':float(np.mean(residuals)),'max_generated_energy_balance_abs':float(max(balances)),
                     'mean_constitutive_mismatch_inf':float(np.mean(defects)),'boundary_error_abs':0.,
                     'input_sha256':hashes[name],'checkpoint_sha256':info['sha256']}
                if name=='confirm_direction':
                    centroid=(u*d['x']).sum(1)/u.sum(1);exact=(d['target']*d['x']).sum(1)/d['target'].sum(1)
                    row['correct_centroid_direction_pairs']=int(sum(centroid[::2]>centroid[1::2]))
                    row['reference_correct_pairs']=int(sum(exact[::2]>exact[1::2]));row['direction_pairs']=16
                path=ROOT/'arrays'/f"H_{info['mode']}_{info['seed']}_{row['case']}.npz"
                np.savez_compressed(path,**d,prediction=u,predicted_flux=j);row['arrays']=str(path.relative_to(ROOT));records.append(row)
    classical=[]
    for name in ['confirm_iid','confirm_direction','confirm_lowdiff']:
        d=datasets[name]
        def fn():return [transport_reference(np.exp(z[:16]),z[16:31],z[31:33],z[-1]) for z in d['inputs']]
        _,timing=measured(fn);classical.append({'case':name.removeprefix('confirm_'),**timing})
    write('transport_confirmation.json',{'records':records,'classical_costs':classical,'evaluated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'prior_files_unchanged':preserve(),'interpretation':'same architecture, separately trained weights; steady directed scalar transport only, not incompressible flow'})
    print(json.dumps({'transport_records':len(records),'models':len(models),'prior_files_unchanged':preserve()}))


if __name__=='__main__':
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1);main()
