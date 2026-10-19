"""Cycle25: learned interaction selection tested against bilinear/Volterra/attention."""
import argparse
import json
import time
import numpy as np
import torch
from experiments.session0005_common import ROOT,SEEDS,tensor,write,sha,now,preserve,snapshot,save_data,load_data,measure,fit
from src.physics.session0004 import steady_heat,diagnostics
from src.operators.session0005_interaction import SingletonEncoder,Interaction

KINDS=['reaction_gate','reaction_no_gate','bilinear','volterra3','attention']


def make_data(count,seed,case='train'):
    rng=np.random.default_rng(seed);n=15;x=np.arange(1,n+1)/(n+1)
    sources=np.zeros((count,4,3));ks=rng.uniform(.9,1.2,count);ls=rng.choice([0.,1.5],count)
    ms=rng.integers(1,4,count) if case in ['train','val'] else np.full(count,1 if case=='single' else (3 if case=='triple' else (4 if case=='four' else 2)))
    labels=np.zeros((count,16,n));valid=np.zeros((count,16),dtype=bool);steps=[];failures=[]
    for index,m in enumerate(ms):
        a=rng.uniform(.1,1.5,m);c=rng.uniform(.2,.8,m);w=rng.uniform(.08,.12,m)
        if case=='far':c=np.array([.12,.88])
        if case=='shift':c=np.array([.08,.92])
        if case=='tiny':a=rng.uniform(1e-3,5e-3,m)
        if case=='linear':ls[index]=0.
        sources[index,:m]=np.c_[a,c,w]
        for mask in range(1<<m):
            source=sum((a[i]*np.exp(-.5*((x-c[i])/w[i])**2) for i in range(m) if mask&(1<<i)),np.zeros(n))
            try:t,it=steady_heat(np.full(n+1,ks[index]),source,lam=ls[index]);steps.append(it)
            except RuntimeError as e:failures.append({'index':index,'mask':mask,'error':str(e)});t=np.full(n,np.nan)
            labels[index,mask]=t;valid[index,mask]=True
    return dict(sources=sources,conductivity=ks,lam=ls,source_count=ms,target_subsets=labels,valid_subsets=valid,x=x,reference_steps=np.array(steps)),failures


def prepare():
    snapshot('interaction_prepare',['experiments/session0005_interaction.py','src/operators/session0005_interaction.py'])
    hashes={};fails={};begin=time.perf_counter()
    cases=[('train',128,52001),('val',32,52002),('single',16,52003),('pair',32,52004),('triple',32,52005),('four',16,52006),
           ('far',16,52007),('shift',16,52008),('tiny',16,52009),('linear',16,52010)]
    for case,count,seed in cases:
        d,f=make_data(count,seed,case);name='F5_'+case;hashes[name]=save_data(name,d);fails[name]=f
    write('interaction_dataset_manifest.json',{'sha256':hashes,'failures':fails,'preparation_seconds':time.perf_counter()-begin,
          'scope':'all subsets are reference records; interaction fitting uses full-set mixed targets, base and singletons; repeated zero/base labels are not independent fields'})
    assert not any(fails.values()),fails
    print(json.dumps({'interaction_data_files':len(hashes),'prior_files_preserved':preserve()}))


def labels(d):
    b=len(d['source_count']);base=d['target_subsets'][:,0]
    single=np.stack([d['target_subsets'][:,1<<i]-base for i in range(4)],1)
    # Inactive source singleton slots were not separately solved: their response is exactly zero.
    single*= (np.arange(4)[None,:]<d['source_count'][:,None])[:,:,None]
    full=np.array([d['target_subsets'][i,(1<<int(m))-1] for i,m in enumerate(d['source_count'])])
    mixed=full-base-single.sum(1)
    return base,single,full,mixed


def encode_train():
    if (ROOT/'interaction_encoder_frozen.json').exists():raise FileExistsError('encoder frozen')
    snapshot('interaction_encoder',['experiments/session0005_interaction.py','src/operators/session0005_interaction.py'])
    ds=[load_data('F5_'+split) for split in ['train','val']];targets=[labels(d) for d in ds]
    tb=[{k:tensor(v) for k,v in d.items()} for d in ds]
    bs=max(float(np.sqrt(np.mean(targets[0][0]**2))),1e-8)
    rs=max(float(np.sqrt(np.mean(targets[0][1]**2))),1e-8)
    torch.manual_seed(2005);encoder=SingletonEncoder().double()
    def loss(m,val):
        index=int(val);d=tb[index];base,r=m(d['conductivity'],d['lam'],d['sources'],d['x'])
        target_base,target_r,_,_=targets[index]
        return ((base-tensor(target_base))/bs).square().mean()+((r-tensor(target_r))/rs).square().mean()
    info=fit(encoder,loss,2000,'F5_singleton_encoder',2005)
    write('interaction_encoder_frozen.json',{'model':info,'frozen_utc':now(),'shared_by_all_interaction_models':True})
    print(json.dumps({k:info[k] for k in ['seconds','best_validation_loss','active_parameter_count']}))


def encoder_restore():
    info=json.loads((ROOT/'interaction_encoder_frozen.json').read_text())['model'];m=SingletonEncoder().double()
    assert sha(ROOT/info['checkpoint'])==info['sha256'];m.load_state_dict(torch.load(ROOT/info['checkpoint'],weights_only=True)['state']);return m


def train():
    if (ROOT/'interaction_frozen.json').exists():raise FileExistsError('interaction frozen')
    snapshot('interaction_train',['experiments/session0005_interaction.py','src/operators/session0005_interaction.py'])
    encoder=encoder_restore();ds=[load_data('F5_'+split) for split in ['train','val']];targets=[labels(d) for d in ds];bundles=[]
    with torch.no_grad():
        for d in ds:
            td={k:tensor(v) for k,v in d.items()};base,r=encoder(td['conductivity'],td['lam'],td['sources'],td['x']);bundles.append((td,base.detach(),r.detach()))
    cs=max(float(np.sqrt(np.mean(targets[0][3]**2))),1e-8);us=max(float(np.sqrt(np.mean(targets[0][2]**2))),1e-8)
    models=[]
    for kind in KINDS:
        for seed in SEEDS:
            torch.manual_seed(seed);model=Interaction(kind).double()
            def loss(m,val):
                index=int(val);td,base,r=bundles[index]
                correction,_=m(base,r,td['sources'],td['conductivity'],td['lam'],td['x'])
                target_mixed=tensor(targets[index][3]);target_full=tensor(targets[index][2])
                prediction=base+r.sum(1)+correction
                return ((correction-target_mixed)/cs).square().mean()+((prediction-target_full)/us).square().mean()
            info=fit(model,loss,1000,f'F5_{kind}_{seed}',seed);info.update(kind=kind)
            models.append(info);write('interaction_training_progress.json',models)
            print(json.dumps({k:info[k] for k in ['kind','seed','seconds','best_validation_loss']}),flush=True)
    write('interaction_frozen.json',{'models':models,'frozen_utc':now(),'shared_encoder_sha256':sha(ROOT/json.loads((ROOT/'interaction_encoder_frozen.json').read_text())['model']['checkpoint']),
        'training_cross_rms':cs,'training_full_rms':us,'prior_files_preserved':preserve()})


def confirm():
    if (ROOT/'interaction_confirmation.json').exists():raise FileExistsError('interaction already evaluated')
    snapshot('interaction_confirm',['experiments/session0005_interaction.py','src/operators/session0005_interaction.py'])
    encoder=encoder_restore();manifest=json.loads((ROOT/'interaction_dataset_manifest.json').read_text())['sha256'];models=json.loads((ROOT/'interaction_frozen.json').read_text())['models'];rows=[];controls=[]
    for info in models:
        m=Interaction(info['kind']).double();assert sha(ROOT/info['checkpoint'])==info['sha256'];m.load_state_dict(torch.load(ROOT/info['checkpoint'],weights_only=True)['state'])
        with torch.no_grad():
            for name in sorted(n for n in manifest if n not in ['F5_train','F5_val']):
                d=load_data(name);targetbase,targetr,truth,cross=labels(d)
                def inference():
                    k=tensor(d['conductivity']);lam=tensor(d['lam']);s=tensor(d['sources']);x=tensor(d['x'])
                    base,r=encoder(k,lam,s,x);correction,activation=m(base,r,s,k,lam,x)
                    return (base+r.sum(1)+correction).numpy(),correction.numpy(),activation.numpy(),base.numpy(),r.numpy()
                (pred,cp,activation,base,r),cost=measure(inference)
                fieldrel=np.linalg.norm(pred-truth,axis=1)/np.maximum(np.linalg.norm(truth,axis=1),1e-14)
                crossnorm=np.linalg.norm(cross,axis=1);crossabs=np.linalg.norm(cp-cross,axis=1)
                ds=[]
                for i,u in enumerate(pred):
                    source=sum((a*np.exp(-.5*((d['x']-c)/w)**2) for a,c,w in d['sources'][i] if a>0),np.zeros(15))
                    ds.append(diagnostics(np.full(16,d['conductivity'][i]),source,(0,0),d['lam'][i],u))
                row={'kind':info['kind'],'seed':info['seed'],'case':name.removeprefix('F5_'),
                    'mean_relative_l2':float(fieldrel.mean()),'max_absolute_error':float(abs(pred-truth).max()),
                    'mean_cross_reference_l2':float(crossnorm.mean()),'max_cross_reference_l2':float(crossnorm.max()),
                    'mean_cross_absolute_l2_error':float(crossabs.mean()),
                    'mean_cross_relative_l2':float(np.mean(crossabs/np.maximum(crossnorm,1e-12))),
                    'relative_denominator_floor':1e-12,'mean_cross_to_full_norm_ratio':float(np.mean(crossnorm/np.maximum(np.linalg.norm(truth,axis=1),1e-14))),
                    'activation_fraction':float(activation.mean()),'minimum_state':float(pred.min()),
                    'negative_samples':int(sum(pred.min(1)< -1e-12)),
                    'mean_pde_residual_inf':float(np.mean([z['pde_residual_inf'] for z in ds])),
                    'mean_energy_balance_abs':float(np.mean([z['constitutive_energy_balance_abs'] for z in ds])),
                    'boundary_error_abs':0.,'input_sha256':manifest[name],'checkpoint':info['checkpoint'],'checkpoint_sha256':info['sha256'],**cost}
                path=ROOT/'arrays'/f"F5_{info['kind']}_{info['seed']}_{row['case']}.npz"
                np.savez_compressed(path,**d,prediction=pred,predicted_mixed=cp,reference=truth,reference_mixed=cross,activation=activation,encoded_base=base,encoded_responses=r)
                row['arrays']=str(path.relative_to(ROOT));rows.append(row)
    # Classical costs include the actual total-source assembly and root solve.
    for name in sorted(n for n in manifest if n not in ['F5_train','F5_val']):
        d=load_data(name);_,_,truth,cross=labels(d)
        def solve_all():
            out=[]
            for k,lam,s in zip(d['conductivity'],d['lam'],d['sources']):
                q=sum((a*np.exp(-.5*((d['x']-c)/w)**2) for a,c,w in s if a>0),np.zeros(15))
                out.append(steady_heat(np.full(16,k),q,lam=lam)[0])
            return np.array(out)
        _,cost=measure(solve_all);controls.append({'case':name.removeprefix('F5_'),**cost})
    write('interaction_confirmation.json',{'records':rows,'classical_costs':controls,'evaluated_utc':now(),
          'scope':'all models frozen before first confirmation; absolute mixed effect and field metrics accompany relative ratios; no post-confirmation fitting','prior_files_preserved':preserve()})
    print(json.dumps({'interaction_prediction_batches':len(rows)}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','encode','train','confirm']);a=p.parse_args()
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1);{'prepare':prepare,'encode':encode_train,'train':train,'confirm':confirm}[a.phase]()
