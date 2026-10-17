"""CPU-only Session 0004. Separate preparation, development and frozen confirmation."""

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import time
import tracemalloc

import numpy as np
from scipy.integrate import solve_ivp, solve_bvp
from scipy.linalg import eigh_tridiagonal
import torch

from src.physics.session0004 import (bratu_branches, bratu_lambda, bratu_profile, fold,
    steady_heat, flux, diagnostics, stability, diffusion_bands, apply_bands, source_rhs)
from src.operators.session0004 import BranchModel, InterventionModel, CoGenerationModel

ROOT = Path("results/session_0004")
DTYPE = torch.float64
SEEDS = [1004,1005,1006]


def write(name, value):
    path=ROOT/name;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+"\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot(phase):
    names=["experiments/session0004.py","src/physics/session0004.py","src/operators/session0004.py",
           "research/session_0004/preregistered_protocol.md"]
    for name in names:
        dest=ROOT/"source_snapshot"/phase/name
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(Path(name).read_bytes())
    write(f"source_snapshot/{phase}/sha256.json",{p:digest(p) for p in names})


def preserve():
    previous=json.loads((ROOT/"prior_file_sha256.json").read_text())
    changed=[p for p,h in previous.items() if digest(p)!=h]
    if changed:raise AssertionError(changed)
    return len(previous)


def tensor(a):return torch.as_tensor(a,dtype=DTYPE)


def measured(fn, repeats=7):
    fn(); times=[]
    for _ in range(repeats):
        start=time.perf_counter();fn();times.append(time.perf_counter()-start)
    tracemalloc.start();result=fn();_,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
    return result,{"median_seconds":float(np.median(times)),"repeats":repeats,
                   "python_tracemalloc_peak_bytes":peak,
                   "memory_scope":"Python traced allocations; native/BLAS/PyTorch memory may be omitted"}


def branch_data(lams,n=15):
    x=np.arange(1,n+1)/(n+1)
    y=np.array([bratu_branches(l,x)[0] for l in lams])
    return dict(lam=np.asarray(lams),x=x,target=y)


def fdata(count,seed,low=-.5,high=2.,centers=(.3,.7),n=15):
    rng=np.random.default_rng(seed);x=np.arange(1,n+1)/(n+1)
    ab=rng.uniform(low,high,(count,2));inputs=np.c_[ab,np.tile(centers,(count,1))]
    labels=[];iterations=[];failures=[]
    for index,(a,b,c1,c2) in enumerate(inputs):
        s1=np.exp(-.5*((x-c1)/.1)**2);s2=np.exp(-.5*((x-c2)/.1)**2)
        quartet=[]
        for aa,bb in [(0,0),(a,0),(0,b),(a,b)]:
            try:
                t,steps=steady_heat(np.ones(n+1),aa*s1+bb*s2,lam=1.5)
                quartet.append(t);iterations.append(steps)
            except RuntimeError as error:
                failures.append({"index":index,"amplitudes":[aa,bb],"error":str(error)})
                quartet.append(np.full(n,np.nan))
        labels.append(quartet)
    return dict(inputs=inputs,x=x,target=np.array(labels),iterations=np.array(iterations)),failures


def gdata(count,seed,case="iid",n=15):
    rng=np.random.default_rng(seed);x=np.arange(1,n+1)/(n+1);ex=(np.arange(n+1)+.5)/(n+1)
    inputs=[];labels=[];js=[];steps=[];failures=[]
    for index in range(count):
        scale=rng.uniform(.9,1.4) if case!="low_k" else rng.uniform(.15,.3)
        k=scale*np.exp(rng.uniform(-.2,.2)*np.sin(2*np.pi*ex+rng.uniform(0,2*np.pi)))
        center=rng.uniform(.25,.65) if case!="shift_source" else rng.uniform(.8,.9)
        q=rng.uniform(-.5,2)*np.exp(-.5*((x-center)/.12)**2)+rng.uniform(-.2,.5)
        g=rng.uniform(-.1,.2,2) if case!="boundary" else rng.uniform(.4,.7,2)
        lam=float(rng.choice([0.,1.5])) if case not in ["low_k","boundary"] else 0.
        try:
            t,it=steady_heat(k,q,tuple(g),lam);j=flux(k,t,g)
        except RuntimeError as error:
            failures.append({"index":index,"error":str(error)});t=np.full(n,np.nan);j=np.full(n+1,np.nan);it=-1
        inputs.append(np.r_[np.log(k),q,g,lam]);labels.append(t);js.append(j);steps.append(it)
    return dict(inputs=np.array(inputs),target=np.array(labels),flux=np.array(js),iterations=np.array(steps),x=x),failures


def save_data(name,data):
    dest=ROOT/"data"/(name+".npz");dest.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest,**data)
    return digest(dest)


def load_data(name):
    with np.load(ROOT/"data"/(name+".npz")) as f:return {k:f[k] for k in f.files}


def reference_audit():
    astar,critical=fold();rows=[]
    for lam in [1.,2.,3.,3.4]:
        for n in [15,31,63,127]:
            x=np.arange(1,n+1)/(n+1);exact,aa=bratu_branches(lam,x)
            for branch in range(2):
                pred,steps=steady_heat(np.ones(n+1),np.zeros(n),lam=lam,initial=exact[branch])
                eig=stability(np.ones(n+1),lam,pred)
                row={"lambda":lam,"n":n,"branch":branch,"a":float(aa[branch]),
                     "discrete_vs_continuum_relative_l2":float(np.linalg.norm(pred-exact[branch])/np.linalg.norm(exact[branch])),
                     "newton_steps":steps,"largest_growth_eigenvalue":eig,**diagnostics(np.ones(n+1),np.zeros(n),(0,0),lam,pred)}
                rows.append(row)
    # Independent solve_bvp uses a different discretization, analytic profiles only as initial guesses.
    bvp=[]
    for branch in range(2):
        x=np.linspace(0,1,33);y,aa=bratu_branches(2.,x);a=aa[branch]
        guess=np.array([y[branch],-2*a*np.tanh(a*(x-.5))])
        sol=solve_bvp(lambda x,z:np.vstack([z[1],-2*np.exp(z[0])]),lambda za,zb:np.array([za[0],zb[0]]),x,guess,tol=1e-9,max_nodes=4000)
        assert sol.success
        xx=np.linspace(0,1,101);truth=bratu_profile(a,xx)
        bvp.append({"branch":branch,"max_absolute_error":float(max(abs(sol.sol(xx)[0]-truth))),"nodes":len(sol.x)})
    n=31;lam=2.;x=np.arange(1,n+1)/(n+1);initials,_=bratu_branches(lam,x)
    roots=np.array([steady_heat(np.ones(n+1),np.zeros(n),lam=lam,initial=t)[0] for t in initials])
    band=diffusion_bands(np.ones(n+1),n)
    _,vec=eigh_tridiagonal(-band[1]+lam*np.exp(roots[1]),-band[0,1:])
    unstable=vec[:,-1];unstable*=np.sign(unstable.sum());unstable/=max(abs(unstable))
    trans=[];arrays={"x":x,"discrete_roots":roots}
    for name,t0 in [("cold",np.zeros(n)),("below_upper",roots[1]-.01*unstable),("above_upper",roots[1]+.01*unstable)]:
        def event(_,z):return 20-max(z)
        event.terminal=True;event.direction=-1
        sol=solve_ivp(lambda _,z:-apply_bands(band,z)+lam*np.exp(z),(0,5),t0,method="BDF",rtol=1e-8,atol=1e-10,events=event)
        assert sol.success
        arrays[name+"_initial"]=t0;arrays[name+"_final"]=sol.y[:,-1]
        trans.append({"case":name,"end_time":float(sol.t[-1]),"hit_temperature20":bool(len(sol.t_events[0])),
                      "lower_root_relative_error":float(np.linalg.norm(sol.y[:,-1]-roots[0])/np.linalg.norm(roots[0])),
                      "max_temperature":float(max(sol.y[:,-1]))})
    (ROOT/"arrays").mkdir(exist_ok=True);np.savez_compressed(ROOT/"arrays/bratu_reference.npz",**arrays)
    report={"a_fold":float(astar),"lambda_fold":float(critical),"fold_condition_error":float(abs(astar*np.tanh(astar/2)-2)),
            "discrete_cases":rows,"independent_bvp":bvp,"initial_condition_experiments":trans,
            "interpretation":"temperature20 event is a numerical runaway indicator, not a proof of finite-time blow-up"}
    write("bratu_reference.json",report)


def prepare():
    if (ROOT/"data").exists():raise FileExistsError("prepared dataset exists")
    snapshot("preparation");begin=time.perf_counter();reference_audit();hashes={};failures={}
    e={"E_train":branch_data(np.linspace(.35,3.2,48)),"E_val":branch_data(np.linspace(.4,3.4,32)),
       "E_confirm_iid":branch_data(np.random.default_rng(41001).uniform(.35,3.2,32)),
       "E_confirm_fold":branch_data(np.linspace(3.45,3.513,24)),
       "E_confirm_resolution":branch_data(np.random.default_rng(41001).uniform(.35,3.2,32),63),
       "E_confirm_absent":{"lam":np.linspace(3.53,3.9,16),"branch_count":np.zeros(16,dtype=int)}}
    for name,data in e.items():hashes[name]=save_data(name,data)
    for name,count,seed,kwargs in [("F_train",128,42001,{}),("F_val",32,42002,{}),
          ("F_confirm_iid",32,42003,{}),("F_confirm_large",16,42004,{"low":2.,"high":3.}),
          ("F_confirm_source",16,42005,{"centers":(.2,.6)})]:
        data,fail=fdata(count,seed,**kwargs);failures[name]=fail;hashes[name]=save_data(name,data)
    for name,count,seed,case in [("G_train",128,43001,"iid"),("G_val",32,43002,"iid"),
          ("G_confirm_iid",32,43003,"iid"),("G_confirm_lowk",16,43004,"low_k"),
          ("G_confirm_source",16,43005,"shift_source"),("G_confirm_boundary",16,43006,"boundary")]:
        data,fail=gdata(count,seed,case);failures[name]=fail;hashes[name]=save_data(name,data)
    write("dataset_manifest.json",{"sha256":hashes,"reference_failures":failures,"preparation_seconds":time.perf_counter()-begin,
        "confirmation_policy":"labels generated now, not used until all checkpoints frozen; near fold and source shifts predetermined",
        "prior_files_unchanged":preserve()})
    assert not any(failures.values()),failures
    print(json.dumps({"phase":"prepared","data_files":len(hashes),"failures":sum(map(len,failures.values()))}))


def quartet_input(z):
    zero=torch.zeros_like(z[:,0]);a,b,c,d=z.unbind(-1)
    return torch.stack([torch.stack([zero,zero,c,d],-1),torch.stack([a,zero,c,d],-1),
                        torch.stack([zero,b,c,d],-1),z],1)


def components(t):
    return torch.stack([t[:,0],t[:,1]-t[:,0],t[:,2]-t[:,0],t[:,3]-t[:,1]-t[:,2]+t[:,0]],1)


def fit(model,loss,steps,seed,filename):
    optimizer=torch.optim.Adam(model.parameters(),lr=.003);best=float("inf");state=None;trace=[]
    tracemalloc.start();begin=time.perf_counter()
    for step in range(steps+1):
        if step%100==0:
            with torch.no_grad():v=float(loss(model,True));train=float(loss(model,False))
            trace.append({"step":step,"train_loss":train,"validation_loss":v})
            if v<best:
                best=v;state={k:v.detach().clone() for k,v in model.state_dict().items()}
        if step==steps:break
        optimizer.zero_grad();value=loss(model,False);value.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
        optimizer.step()
    elapsed=time.perf_counter()-begin;_,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
    model.load_state_dict(state)
    dest=ROOT/"models"/filename;dest.parent.mkdir(exist_ok=True,parents=True)
    torch.save({"state":state,"seed":seed},dest)
    return {"checkpoint":str(dest.relative_to(ROOT)),"sha256":digest(dest),"seed":seed,
            "parameter_count":sum(p.numel() for p in model.parameters()),"steps":steps,
            "seconds":elapsed,"python_tracemalloc_peak_bytes":peak,"best_validation_loss":best,"trace":trace}


def train():
    if (ROOT/"frozen_models.json").exists():raise FileExistsError("models already frozen")
    snapshot("training");training=[]
    for family in ["E","F","G"]:
        train_data=load_data(family+"_train");val_data=load_data(family+"_val")
        modes={"E":["paired","two_head","conditioned","mean"],"F":["pair","additive","direct"],
               "G":["feedback","no_feedback","direct"]}[family]
        for mode in modes:
            for seed in SEEDS:
                torch.manual_seed(seed)
                if family=="E":
                    model=BranchModel(mode).double()
                    bundles=[{k:tensor(v) for k,v in d.items()} for d in [train_data,val_data]]
                    def loss(m,val):
                        d=bundles[int(val)];p=m(d["lam"],d["x"])
                        if mode=="two_head":
                            straight=((p-d["target"])**2).mean((1,2));reverse=((p.flip(1)-d["target"])**2).mean((1,2))
                            return torch.minimum(straight,reverse).mean()
                        return ((p-d["target"])**2).mean()
                    steps=2000
                elif family=="F":
                    model=InterventionModel(mode).double()
                    bundles=[{k:tensor(v) for k,v in d.items()} for d in [train_data,val_data]]
                    scale=components(bundles[0]["target"]).square().mean((0,2)).sqrt().clamp_min(1e-5)
                    def loss(m,val):
                        d=bundles[int(val)];inp=quartet_input(d["inputs"])
                        p=m(inp.flatten(0,1),d["x"]).reshape(len(inp),4,-1)
                        return (((components(p)-components(d["target"]))/scale[None,:,None])**2).mean()
                    steps=1500
                else:
                    model=CoGenerationModel(mode=mode).double()
                    bundles=[{k:tensor(v) for k,v in d.items()} for d in [train_data,val_data]]
                    model.center.copy_(bundles[0]["inputs"].mean(0));model.scale.copy_(bundles[0]["inputs"].std(0).clamp_min(.1))
                    tscale=bundles[0]["target"].square().mean().sqrt();jscale=bundles[0]["flux"].square().mean().sqrt()
                    def loss(m,val):
                        d=bundles[int(val)];p,j=m(d["inputs"])
                        return ((p-d["target"])/tscale).square().mean()+((j-d["flux"])/jscale).square().mean()
                    steps=1500
                info=fit(model,loss,steps,seed,f"{family}_{mode}_{seed}.pt")
                info.update(family=family,mode=mode)
                if family=="E" and mode=="paired":info["learned_fold"]=float(model.critical.detach())
                training.append(info);write("training_progress.json",training)
                print(json.dumps({k:info[k] for k in ["family","mode","seed","seconds","best_validation_loss"]}),flush=True)
    write("frozen_models.json",{"frozen_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                "models":training,"confirmation_evaluated":False,"prior_files_unchanged":preserve()})


def restore(info):
    family=info["family"];mode=info["mode"]
    model={"E":BranchModel,"F":InterventionModel,"G":CoGenerationModel}[family](mode=mode).double()
    checkpoint=ROOT/info["checkpoint"];assert digest(checkpoint)==info["sha256"]
    model.load_state_dict(torch.load(checkpoint,weights_only=True)["state"]);model.eval();return model


def train_oracle():
    """Predeclared diagnostic revision: supply exact fold, with unequal physical prior."""
    if (ROOT/"confirmation.json").exists():raise FileExistsError("confirmation already read")
    frozen=json.loads((ROOT/"frozen_models.json").read_text())
    if any(i["mode"]=="paired_oracle" for i in frozen["models"]):raise FileExistsError("oracle already trained")
    snapshot("oracle_revision")
    bundles=[{k:tensor(v) for k,v in load_data(name).items()} for name in ["E_train","E_val"]]
    for seed in SEEDS:
        torch.manual_seed(seed);model=BranchModel("paired_oracle").double()
        def loss(m,val):
            d=bundles[int(val)];return (m(d["lam"],d["x"])-d["target"]).square().mean()
        info=fit(model,loss,2000,seed,f"E_paired_oracle_{seed}.pt")
        info.update(family="E",mode="paired_oracle",learned_fold=float(model.critical.detach()),
                    supplied_exact_fold=True,interpretation="unequal physics prior; not matched-data superiority")
        frozen["models"].append(info);print(json.dumps({k:info[k] for k in ["mode","seed","seconds","best_validation_loss"]}),flush=True)
    frozen["frozen_utc"]=datetime.datetime.now(datetime.timezone.utc).isoformat()
    frozen["oracle_revision_before_confirmation"]=True
    write("frozen_models.json",frozen)


def train_matched():
    """Additional MLP control chosen by parameter budget before confirmation."""
    if (ROOT/"confirmation.json").exists():raise FileExistsError("confirmation already read")
    frozen=json.loads((ROOT/"frozen_models.json").read_text())
    if any(i["mode"]=="direct_matched" for i in frozen["models"]):raise FileExistsError("matched MLP already trained")
    snapshot("matched_control")
    bundles=[{k:tensor(v) for k,v in load_data(name).items()} for name in ["F_train","F_val"]]
    scale=components(bundles[0]["target"]).square().mean((0,2)).sqrt().clamp_min(1e-5)
    for seed in SEEDS:
        torch.manual_seed(seed);model=InterventionModel("direct_matched").double()
        def loss(m,val):
            d=bundles[int(val)];inp=quartet_input(d["inputs"])
            p=m(inp.flatten(0,1),d["x"]).reshape(len(inp),4,-1)
            return (((components(p)-components(d["target"]))/scale[None,:,None])**2).mean()
        info=fit(model,loss,1500,seed,f"F_direct_matched_{seed}.pt")
        info.update(family="F",mode="direct_matched",selection="width64 chosen from parameter budget before confirmation")
        frozen["models"].append(info);print(json.dumps({k:info[k] for k in ["mode","seed","seconds","best_validation_loss"]}),flush=True)
    frozen["frozen_utc"]=datetime.datetime.now(datetime.timezone.utc).isoformat();frozen["matched_control_before_confirmation"]=True
    write("frozen_models.json",frozen)


def confirm():
    if (ROOT/"confirmation.json").exists():raise FileExistsError("confirmation already evaluated; new development is exploratory")
    snapshot("confirmation");frozen=json.loads((ROOT/"frozen_models.json").read_text());records=[]
    with torch.no_grad():
        for info in frozen["models"]:
            model=restore(info);family=info["family"]
            for path in sorted((ROOT/"data").glob(family+"_confirm_*.npz")):
                name=path.stem;d=load_data(name);case=name.split("_confirm_")[1]
                row={"family":family,"mode":info["mode"],"seed":info["seed"],"case":case,
                     "input_sha256":digest(path),"checkpoint_sha256":info["sha256"]}
                arrays=dict(d)
                if family=="E":
                    if case=="absent":
                        count=(2*(d["lam"]<float(model.critical))).astype(int) if info["mode"].startswith("paired") else np.full(len(d["lam"]),2)
                        row.update(spurious_branches=int(count.sum()),cases=len(count));arrays["predicted_branch_count"]=count
                    else:
                        pred,timing=measured(lambda:model(tensor(d["lam"]),tensor(d["x"])).numpy())
                        # Match whole fields, never sort pointwise and fabricate a composite branch.
                        raw_prediction=pred.copy()
                        straight=((pred-d["target"])**2).sum((1,2))
                        reverse=((pred[:,::-1]-d["target"])**2).sum((1,2))
                        assignment=reverse<straight
                        pred[assignment]=pred[assignment,::-1]
                        truth=d["target"];relative=np.linalg.norm(pred-truth,axis=2)/np.linalg.norm(truth,axis=2)
                        refined_x=np.arange(1,256)/256
                        refined=model(tensor(d["lam"]),tensor(refined_x)).numpy()
                        refined[assignment]=refined[assignment,::-1]
                        perbranch=[]
                        for branch in range(2):
                            res=[];growth=[]
                            for lam,t,tr in zip(d["lam"],pred[:,branch],refined[:,branch]):
                                res.append(diagnostics(np.ones(len(t)+1),np.zeros(len(t)),(0,0),lam,t))
                                growth.append(stability(np.ones(len(tr)+1),lam,tr))
                            perbranch.append({"branch":branch,"mean_relative_l2":float(relative[:,branch].mean()),
                                              "max_pde_residual_inf":max(z["pde_residual_inf"] for z in res),
                                              "mean_pde_residual_inf":float(np.mean([z["pde_residual_inf"] for z in res])),
                                              "mean_energy_balance_abs":float(np.mean([z["constitutive_energy_balance_abs"] for z in res])),
                                              "wrong_growth_sign_count":int(sum((np.array(growth)>0)!=(branch==1)))})
                        counts=2*(d["lam"]<float(model.critical)) if info["mode"].startswith("paired") else np.full(len(d["lam"]),2)
                        pairwise=np.linalg.norm(pred[:,:,None,:]-truth[:,None,:,:],axis=-1)/np.linalg.norm(truth,axis=-1)[:,None,:]
                        distances=[];false=[]
                        for i,count in enumerate(counts):
                            distances.append(2 if count==0 else int(sum(np.min(pairwise[i],axis=0)>.05)))
                            false.append(0 if count==0 else int(sum(np.min(pairwise[i],axis=1)>.05)))
                        row.update(timing,branches=perbranch,mean_relative_l2=float(relative.mean()),
                                   max_absolute_error=float(abs(pred-truth).max()),missing_branches=int(sum(2-counts)),
                                   missing_branches_at_5percent=int(sum(distances)),spurious_branches_at_5percent=int(sum(false)),
                                   stability_grid_n=255,boundary_error_abs=0.)
                        arrays.update(prediction=pred,raw_prediction=raw_prediction,swapped_assignment=assignment,predicted_branch_count=counts)
                elif family=="F":
                    inp=quartet_input(tensor(d["inputs"]));x=tensor(d["x"])
                    pred=model(inp.flatten(0,1),x).reshape(len(inp),4,-1).numpy()
                    final,timing=measured(lambda:model(tensor(d["inputs"]),tensor(d["x"])).numpy())
                    np.testing.assert_allclose(final,pred[:,3],rtol=1e-12,atol=1e-12)
                    truth=d["target"];rel=np.linalg.norm(pred[:,3]-truth[:,3],axis=1)/np.linalg.norm(truth[:,3],axis=1)
                    pcomp=components(tensor(pred)).numpy();tcomp=components(tensor(truth)).numpy()
                    crossrel=np.linalg.norm(pcomp[:,3]-tcomp[:,3],axis=1)/np.maximum(np.linalg.norm(tcomp[:,3],axis=1),1e-14)
                    diags=[]
                    for z,t in zip(d["inputs"],pred[:,3]):
                        a,b,c1,c2=z;q=a*np.exp(-.5*((d["x"]-c1)/.1)**2)+b*np.exp(-.5*((d["x"]-c2)/.1)**2)
                        diags.append(diagnostics(np.ones(len(t)+1),q,(0,0),1.5,t))
                    row.update(timing,mean_relative_l2=float(rel.mean()),mean_cross_relative_l2=float(crossrel.mean()),
                               max_absolute_error=float(abs(pred[:,3]-truth[:,3]).max()),
                               mean_pde_residual_inf=float(np.mean([z["pde_residual_inf"] for z in diags])),
                               mean_energy_balance_abs=float(np.mean([z["constitutive_energy_balance_abs"] for z in diags])),boundary_error_abs=0.)
                    arrays["prediction"]=pred
                else:
                    (pred,j),timing=measured(lambda:tuple(v.numpy() for v in model(tensor(d["inputs"]))))
                    rel=np.linalg.norm(pred-d["target"],axis=1)/np.maximum(np.linalg.norm(d["target"],axis=1),1e-14)
                    diags=[];n=pred.shape[1]
                    for z,t,jj in zip(d["inputs"],pred,j):
                        diags.append(diagnostics(np.exp(z[:n+1]),z[n+1:2*n+1],z[2*n+1:2*n+3],z[-1],t,jj))
                    row.update(timing,mean_relative_l2=float(rel.mean()),max_absolute_error=float(abs(pred-d["target"]).max()),
                        mean_flux_relative_l2=float(np.mean(np.linalg.norm(j-d["flux"],axis=1)/np.maximum(np.linalg.norm(d["flux"],axis=1),1e-14))),
                        mean_pde_residual_inf=float(np.mean([z["pde_residual_inf"] for z in diags])),
                        mean_energy_balance_abs=float(np.mean([z["constitutive_energy_balance_abs"] for z in diags])),
                        max_generated_balance_inf=max(z["generated_balance_inf"] for z in diags),
                        max_generated_energy_balance_abs=max(z["generated_energy_balance_abs"] for z in diags),
                        mean_constitutive_mismatch_inf=float(np.mean([z["constitutive_mismatch_inf"] for z in diags])),boundary_error_abs=0.)
                    arrays.update(prediction=pred,predicted_flux=j)
                dest=ROOT/"arrays"/f"{family}_{info['mode']}_{info['seed']}_{case}.npz"
                np.savez_compressed(dest,**arrays);row["arrays"]=str(dest.relative_to(ROOT));records.append(row)
    write("confirmation.json",{"evaluated_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"records":records,
                               "prior_files_unchanged":preserve(),"policy":"all checkpoints frozen before first confirmation; no checkpoint selection from these results"})
    print(json.dumps({"confirmation_records":len(records),"prior_files_unchanged":preserve()}))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("phase",choices=["prepare","train","train_oracle","train_matched","confirm"])
    args=parser.parse_args();torch.set_default_dtype(DTYPE);torch.set_num_threads(1)
    {"prepare":prepare,"train":train,"train_oracle":train_oracle,"train_matched":train_matched,"confirm":confirm}[args.phase]()
