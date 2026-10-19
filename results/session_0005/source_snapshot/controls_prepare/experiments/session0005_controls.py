"""G causal controls: new data, fixed protocol, all checkpoints frozen before evaluation."""
import argparse
import json
import time
import numpy as np
import torch
from experiments.session0005_common import ROOT,SEEDS,tensor,write,sha,now,preserve,snapshot,save_data,load_data,measure,fit
from experiments.session0004 import gdata
from experiments.session0004_transport import data as transport_data,transport_reference
from src.physics.session0004 import steady_heat,flux,diagnostics
from src.operators.session0005_controls import GControl,LearnedJacobi,physical_flux

KINDS=['original','physics_plus_feature','learned_feature','static_physics','direct_matched','learned_jacobi']


def prepare():
    snapshot('controls_prepare',['experiments/session0005_controls.py','src/operators/session0005_controls.py'])
    names={};failures={};start=time.perf_counter()
    specs=[('G','train',128,51001,'iid'),('G','val',32,51002,'iid'),('G','confirm_iid',64,51003,'iid'),
        ('G','confirm_lowk',16,51004,'low_k'),('G','confirm_source',16,51005,'shift_source'),
        ('G','confirm_boundary',16,51006,'boundary'),('G','confirm_positive',16,51007,'positive'),
        ('H','train',128,51101,'iid'),('H','val',32,51102,'iid'),('H','confirm_iid',64,51103,'iid'),
        ('H','confirm_lowdiff',16,51104,'lowdiff'),('H','confirm_direction',32,51105,'direction')]
    for family,split,count,seed,case in specs:
        if family=='G':
            d,fail=gdata(count,seed,'iid' if case=='positive' else case)
            if case=='positive':
                rng=np.random.default_rng(seed);d['inputs'][:,31:]=0.
                for i,z in enumerate(d['inputs']):
                    z[16:31]=rng.uniform(.5,2)*np.exp(-.5*((d['x']-rng.uniform(.2,.8))/.1)**2)
                    u,_=steady_heat(np.exp(z[:16]),z[16:31],(0,0),0.)
                    d['target'][i]=u;d['flux'][i]=flux(np.exp(z[:16]),u)
        else:d=transport_data(count,seed,case);fail=[]
        name=f'C0_{family}_{split}';names[name]=save_data(name,d);failures[name]=fail
    write('controls_dataset_manifest.json',{'sha256':names,'reference_failures':failures,'preparation_seconds':time.perf_counter()-start,'prepared_utc':now()})
    assert not any(failures.values()),failures
    print(json.dumps({'control_data_files':len(names),'prior_files_preserved':preserve()}))


def build(kind,family):return LearnedJacobi(family).double() if kind=='learned_jacobi' else GControl(kind,family).double()


def train():
    if (ROOT/'controls_frozen.json').exists():raise FileExistsError('controls frozen')
    snapshot('controls_train',['experiments/session0005_controls.py','src/operators/session0005_controls.py'])
    models=[]
    for family in ['G','H']:
        ds=[{k:tensor(v) for k,v in load_data(f'C0_{family}_{s}').items()} for s in ['train','val']]
        us=ds[0]['target'].square().mean().sqrt();js=ds[0]['flux'].square().mean().sqrt()
        for kind in KINDS:
            for seed in SEEDS:
                torch.manual_seed(seed);m=build(kind,family)
                if kind!='learned_jacobi':
                    m.center.copy_(ds[0]['inputs'].mean(0));m.scale.copy_(ds[0]['inputs'].std(0).clamp_min(.1))
                def loss(model,val):
                    d=ds[int(val)];u,j=model(d['inputs'])
                    return ((u-d['target'])/us).square().mean()+((j-d['flux'])/js).square().mean()
                info=fit(m,loss,2000 if kind=='direct_matched' else 1000,f'C0_{family}_{kind}_{seed}',seed)
                info.update(family=family,kind=kind);models.append(info)
                write('controls_training_progress.json',models)
                print(json.dumps({k:info[k] for k in ['family','kind','seed','seconds','best_validation_loss','active_parameter_count']}),flush=True)
    write('controls_frozen.json',{'models':models,'frozen_utc':now(),'prior_files_preserved':preserve()})


def metrics(raw,u,j,truth,family):
    rel=np.linalg.norm(u-truth,axis=1)/np.maximum(np.linalg.norm(truth,axis=1),1e-14)
    q=raw[:,16:31].copy()
    if family=='G':q+=raw[:,-1:]*np.exp(u)
    with torch.no_grad():physical=physical_flux(tensor(raw),tensor(u),family).numpy()
    pderes=np.diff(physical,axis=1)*16-q;balance=np.diff(j,axis=1)*16-q
    positive_required=(raw[:,16:31].min(1)>=0)&(abs(raw[:,31:33]).max(1)==0)
    return {'mean_relative_l2':float(rel.mean()),'max_absolute_error':float(abs(u-truth).max()),
            'mean_pde_residual_inf':float(abs(pderes).max(1).mean()),'max_pde_residual_inf':float(abs(pderes).max()),
            'mean_constitutive_mismatch_inf':float(abs(j-physical).max(1).mean()),
            'max_balance_residual_inf':float(abs(balance).max()),
            'max_generated_energy_balance_abs':float(abs(j[:,-1]-j[:,0]-q.sum(1)/16).max()),
            'minimum_state':float(u.min()),'positive_required_samples':int(positive_required.sum()),
            'positivity_violating_samples':int(((u.min(1)< -1e-12)&positive_required).sum()),'boundary_error_abs':0.}


def classical(raw,family,kind):
    if kind=='exact':
        out=[];curr=[]
        for z in raw:
            if family=='G':
                u,_=steady_heat(np.exp(z[:16]),z[16:31],z[31:33],z[-1]);j=flux(np.exp(z[:16]),u,z[31:33])
            else:u,j=transport_reference(np.exp(z[:16]),z[16:31],z[31:33],z[-1])
            out.append(u);curr.append(j)
        return np.array(out),np.array(curr)
    steps=int(kind.removeprefix('jacobi'))
    m=LearnedJacobi(family,steps).double()
    with torch.no_grad():m.alpha.fill_(np.log(.8/.2));return tuple(z.numpy() for z in m(tensor(raw)))


def confirm():
    if (ROOT/'controls_confirmation.json').exists():raise FileExistsError('controls already evaluated')
    snapshot('controls_confirm',['experiments/session0005_controls.py','src/operators/session0005_controls.py'])
    frozen=json.loads((ROOT/'controls_frozen.json').read_text());manifest=json.loads((ROOT/'controls_dataset_manifest.json').read_text())['sha256'];rows=[]
    for info in frozen['models']:
        family=info['family'];m=build(info['kind'],family)
        assert sha(ROOT/info['checkpoint'])==info['sha256'];m.load_state_dict(torch.load(ROOT/info['checkpoint'],weights_only=True)['state'])
        with torch.no_grad():
            for name in sorted(n for n in manifest if n.startswith(f'C0_{family}_confirm_')):
                d=load_data(name);(u,j),cost=measure(lambda:tuple(z.numpy() for z in m(tensor(d['inputs']))))
                row={'family':family,'kind':info['kind'],'seed':info['seed'],'case':name.split('_confirm_')[1],
                    **metrics(d['inputs'],u,j,d['target'],family),**cost,'input_sha256':manifest[name],
                    'checkpoint':info['checkpoint'],'checkpoint_sha256':info['sha256']}
                if row['case']=='direction':
                    cent=(u*d['x']).sum(1)/u.sum(1);row['correct_direction_pairs']=int(sum(cent[::2]>cent[1::2]));row['direction_pairs']=16
                path=ROOT/'arrays'/f"C0_{family}_{info['kind']}_{info['seed']}_{row['case']}.npz";path.parent.mkdir(exist_ok=True,parents=True)
                np.savez_compressed(path,**d,prediction=u,predicted_flux=j);row['arrays']=str(path.relative_to(ROOT));rows.append(row)
    baselines=[]
    for name in sorted(n for n in manifest if '_confirm_' in n):
        family=name.split('_')[1];d=load_data(name)
        for kind in ['exact','jacobi3','jacobi32']:
            with torch.no_grad():(u,j),cost=measure(lambda:classical(d['inputs'],family,kind))
            row={'family':family,'kind':kind,'case':name.split('_confirm_')[1],**metrics(d['inputs'],u,j,d['target'],family),**cost}
            path=ROOT/'arrays'/f'C0_classical_{family}_{kind}_{row["case"]}.npz';np.savez_compressed(path,**d,prediction=u,predicted_flux=j)
            row['arrays']=str(path.relative_to(ROOT));baselines.append(row)
    write('controls_confirmation.json',{'records':rows,'classical_records':baselines,'evaluated_utc':now(),
        'scope':'causal controls, not independent invention cycles; same-data within family; profiler coverage/time differences disclosed','prior_files_preserved':preserve()})
    print(json.dumps({'control_prediction_batches':len(rows),'classical_batches':len(baselines)}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','train','confirm']);a=p.parse_args()
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1);{'prepare':prepare,'train':train,'confirm':confirm}[a.phase]()
