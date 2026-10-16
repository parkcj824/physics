"""Reproducible bounded CPU study. Failed hypotheses are data, not passing claims.

Run: python -m experiments.run_research --output results/session_0002
Every invocation requires a new output directory to preserve previous results.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import resource
import time
import tracemalloc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.linalg import expm, solve
import torch
from torch import nn

from src.physics.heat import heat_1d, heat_2d, neumann_1d
from src.operators.primitives import (anchored_response, resistance_response, exchange_response,
    action_response, multiplicative_response, material_event_paths, LifetimeResponse,
    spectral_response, resolvent_response)


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def errors(prediction, reference):
    difference = np.asarray(prediction)-np.asarray(reference)
    return {"relative_l2": float(np.linalg.norm(difference)/max(np.linalg.norm(reference),1e-14)),
            "max_absolute_error": float(np.max(np.abs(difference)))}


def measured(function, repeats=5):
    result = function()  # warm-up outside measurements
    elapsed=[]
    for _ in range(repeats):
        start=time.perf_counter()
        result=function()
        elapsed.append(time.perf_counter()-start)
    tracemalloc.start()
    function()
    _, peak=tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, {"latency_median_seconds": float(np.median(elapsed)),
                    "latency_min_seconds": float(min(elapsed)), "latency_repeats": repeats,
                    "traced_peak_bytes": int(peak),
                    "process_rss_high_water_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    "memory_scope": "traced allocations exclude some native/BLAS allocations; RSS is cumulative process high-water, not isolated model memory"}


class Study:
    def __init__(self, output):
        self.output=Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        for name in ['metrics','plots','comparisons','predictions','models']:
            (self.output/name).mkdir()
        self.rows=[]
        self.started=time.perf_counter()
        dump(self.output/'run.json', {'status':'running', 'started_utc':datetime.now(timezone.utc).isoformat(),
            'versions':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
                        'torch':torch.__version__,'matplotlib':matplotlib.__version__},
            'hardware':{'logical_cpu_count':os.cpu_count(),'torch_threads':torch.get_num_threads(),
                        'cuda_available':torch.cuda.is_available()},
            'training_seeds':[824,825,826], 'training_samples':128, 'validation_samples':32,
            'test_samples_per_condition':32, 'training_steps':1000,
            'publication':'local only; not uploaded', 'novelty':'unverified; classical equivalences explicitly tracked'})

    @classmethod
    def resume(cls, output):
        instance=cls.__new__(cls)
        instance.output=Path(output)
        run=json.loads((instance.output/'run.json').read_text())
        if run['status']!='initial_completed':
            raise ValueError('Only a completed initial phase can be resumed once')
        instance.rows=json.loads((instance.output/'metrics/all_cases.json').read_text())
        instance.started=time.perf_counter()-run['initial_phase_seconds']
        run['status']='continuing'
        dump(instance.output/'run.json',run)
        return instance

    def record(self, cycle, candidate, case, prediction, reference, metrics=None, timing=None, inputs=None):
        index=len(self.rows)
        path=f'predictions/{index:03d}_{candidate}_{case}.npz'
        arrays={'prediction':np.asarray(prediction),'reference':np.asarray(reference)}
        if inputs:
            arrays.update(inputs)
        np.savez_compressed(self.output/path, **arrays)
        row={'cycle':cycle,'candidate':candidate,'case':case,**errors(prediction,reference),
             'predictions_file':path, **(metrics or {}), **(timing or {})}
        self.rows.append(row)
        dump(self.output/'metrics/all_cases.json',self.rows)
        return row

    def heat(self, cycle, candidate, name, problem, function, extra=None):
        reference=problem.reference()
        prediction,timing=measured(function)
        metrics={**problem.diagnostics(prediction),
                 'flux_relative_l2':float(np.linalg.norm(problem.flux(prediction)-problem.flux(reference))/max(np.linalg.norm(problem.flux(reference)),1e-14)),
                 'governing_equation':'-d/dx(k dT/dx)=Q', 'n':len(problem.x),
                 'boundary_values':list(problem.boundary),
                 'k_min':float(problem.edges_k.min()),'k_max':float(problem.edges_k.max()), **(extra or {})}
        return self.record(cycle,candidate,name,prediction,reference,metrics,timing,
            {'x':problem.x,'source':problem.source,'edge_conductivity':problem.edges_k,
             'predicted_flux':problem.flux(prediction),'reference_flux':problem.flux(reference)})


def initial_cycles(study):
    uniform=heat_1d()
    local=heat_1d(source=lambda x:20*np.exp(-((x-.25)/.055)**2))
    material=heat_1d(conductivity=lambda x:np.where(x<.5,1.,8.))
    shifted=heat_1d(conductivity=lambda x:np.where(x<.5,1.,8.),source=0.,boundary=(0.,1.))
    for name,problem in [('uniform',uniform),('localized',local),('material',material),('boundary',shifted)]:
        _,green=anchored_response(problem)
        study.heat(1,'C001',name,problem,lambda p=problem:anchored_response(p)[0],
            {'reciprocity_relative_defect':float(np.linalg.norm(green-green.T)/np.linalg.norm(green)),
             'length':0.2,'failure_hypothesis':'normalization discards reciprocal resistance structure'})
    for n in [7,15,31]:
        problem=heat_1d(n)
        largest=np.linalg.eigvalsh(problem.matrix)[-1]
        study.heat(2,'C002',f'unstable_n{n}',problem,
            lambda p=problem:exchange_response(p.matrix,p.rhs,steps=8,step=.1),
            {'iterations':8,'step':.1,'largest_eigenvalue':float(largest),'max_update_amplification':float(np.max(abs(1-.1*np.linalg.eigvalsh(problem.matrix))))})
        study.heat(2,'C002R',f'stable_n{n}',problem,
            lambda p=problem,l=largest:exchange_response(p.matrix,p.rhs,steps=32,step=1/l),
            {'iterations':32,'step':float(1/largest),'equivalence':'Richardson/Jacobi-like explicit relaxation'})
    for name,problem in [('uniform',uniform),('localized',local),('material',material),
                         ('high_frequency',heat_1d(source=lambda x:np.sin(8*np.pi*x)))]:
        basis=np.sin(np.pi*problem.x[:,None]*np.arange(1,5)[None,:])
        study.heat(3,'C003',name,problem,lambda p=problem,b=basis:action_response(p.matrix,p.rhs,b),
            {'basis_dimension':4,'equivalence':'Ritz/Galerkin reduced-order model'})
    for name,problem in [('positive_source',uniform),('cooling_deviation',heat_1d(source=-1.))]:
        study.heat(4,'C004',name,problem,lambda p=problem:multiplicative_response(p.matrix,p.rhs),
            {'iterations':128,'step':.001,'state_representation':'strictly positive scalar; cannot represent signed deviations'})
    forward,reverse,loop,start=material_event_paths(uniform)
    endpoint=heat_1d(conductivity=2.).reference()
    study.record(5,'C005','material_path12',forward,endpoint,
        {'path_order_relative_defect':float(np.linalg.norm(forward-reverse)/np.linalg.norm(endpoint)),
         'governing_equation':'steady -d/dx(k dT/dx)=1; endpoint k=2; path-independent truth',
         'event_amount':float(np.log(2)), 'n':15}, inputs={'reverse_path':reverse,'initial_response':start})
    study.record(5,'C005','closed_material_loop',loop,start,
        {'governing_equation':'steady -d/dx(k dT/dx)=1; endpoint returns to k=1','n':15})
    for name,problem in [('uniform',uniform),('localized',local),('material',material),('boundary',shifted),
                         ('resolution31',heat_1d(31,conductivity=lambda x:1+3*x,source=lambda x:np.sin(np.pi*x)))]:
        study.heat(6,'C001R',name,problem,lambda p=problem:resistance_response(p)[0],
            {'equivalence':'classical one-dimensional resistance-coordinate Green function'})
    for n in [16,32,64]:
        x,a,b,reference=neumann_1d(n)
        eig,v=np.linalg.eigh(a)
        mask=eig>1e-9
        prediction=v[:,mask] @ ((v[:,mask].T @ b)/eig[mask])
        study.record(6,'gauge_control',f'neumann_n{n}',prediction,reference,
            {'governing_equation':'-Txx=(2pi)^2 cos(2pi x), insulated boundaries, mean(T)=0',
             'mean_abs':float(abs(prediction.mean())), 'compatibility_abs':float(abs(b.sum()/n)),
             'pde_residual_inf':float(np.max(abs(a@prediction-b))),
             'analytic_relative_l2':float(np.linalg.norm(reference-np.cos(2*np.pi*x))/np.linalg.norm(np.cos(2*np.pi*x)))},
            inputs={'x':x,'rhs':b,'analytic':np.cos(2*np.pi*x)})
    try:
        neumann_1d(16,source=np.ones(16))
    except ValueError as error:
        dump(study.output/'metrics/neumann_incompatible.json',{'status':'correctly rejected','reason':str(error)})


def dataset(samples,seed,n=15,k_range=(.5,2.),kind='smooth'):
    rng=np.random.default_rng(seed)
    p=heat_1d(n)
    x=p.x
    basis=np.sin(np.pi*x[:,None]*np.arange(1,5)[None,:])
    q=rng.normal(size=(samples,4)) @ basis.T
    q+=rng.normal(size=(samples,1))
    for i in range(samples):
        center=rng.uniform(.1,.9)
        width=rng.uniform(.04,.15) if kind=='smooth' else .012
        q[i]+=rng.uniform(-5,5)*np.exp(-((x-center)/width)**2)
    if kind=='oscillatory':
        q=np.sin(12*np.pi*x)[None,:]*rng.uniform(.5,2,size=(samples,1))
    k=np.exp(rng.uniform(np.log(k_range[0]),np.log(k_range[1]),samples))
    eig,v=np.linalg.eigh(p.matrix)
    y=((q@v)/eig) @ v.T / k[:,None]
    return k,q,y,eig,v


def train_models(study):
    train=dataset(128,1701)
    validation=dataset(32,1702)
    conditions={
        'iid':dataset(32,1703), 'low_conductivity_ood':dataset(32,1704,k_range=(.01,.05)),
        'high_conductivity_ood':dataset(32,1705,k_range=(10.,30.)),
        'localized_source_ood':dataset(32,1706,kind='localized'),
        'high_frequency_source_ood':dataset(32,1707,kind='oscillatory')}
    k,q,y,eig,v=train
    np.savez_compressed(study.output/'comparisons/training_data.npz',k=k,source=q,target=y,
                        validation_k=validation[0],validation_source=validation[1],validation_target=validation[2])
    eigen=torch.tensor(eig,dtype=torch.float64)
    vectors=torch.tensor(v,dtype=torch.float64)
    labels=torch.tensor(y,dtype=torch.float64)
    sources=torch.tensor(q,dtype=torch.float64)
    values=torch.tensor(k[:,None],dtype=torch.float64)*eigen
    coeff=sources@vectors
    denominator=labels.square().sum(-1).clamp_min(1e-12)
    features=torch.tensor(np.c_[q,np.log(k)],dtype=torch.float64)
    records=[]
    representative=None
    for seed in [824,825,826]:
        torch.manual_seed(seed)
        models={'C006':LifetimeResponse(seed=seed),
                'MLP':nn.Sequential(nn.Linear(16,32),nn.Tanh(),nn.Linear(32,15)).double()}
        for name,model in models.items():
            start=time.perf_counter()
            optimizer=torch.optim.Adam(model.parameters(),lr=.025 if name=='C006' else .003)
            first=None
            for step in range(1000):
                optimizer.zero_grad(set_to_none=True)
                pred=(coeff*model(values))@vectors.T if name=='C006' else model(features)
                loss=((pred-labels).square().sum(-1)/denominator).mean()
                if first is None: first=loss.item()
                loss.backward()
                optimizer.step()
            elapsed=time.perf_counter()-start
            row={'model':name,'seed':seed,'training_samples':128,'steps':1000,
                 'parameters':sum(p.numel() for p in model.parameters()),
                 'training_seconds':elapsed,'initial_relative_mse':first,
                 'training_loss_before_final_update':loss.item(),
                 'input_prior':'full discretized SPD eigensystem and source' if name=='C006' else 'source vector and log scalar conductivity',
                 'fairness':'same supervised samples, normalized loss, steps, CPU; different priors, parameter counts and wall times; not a matched-budget superiority test'}
            if name=='C006':
                row['learned_ages']=model.log_age.detach().exp().tolist()
                row['learned_masses']=model.log_mass.detach().exp().tolist()
            torch.save(model.state_dict(),study.output/f'models/{name}_seed{seed}.pt')
            for condition,data in {'validation':validation,**conditions}.items():
                ck,cq,cy,ce,cv=data
                def predict():
                    with torch.no_grad():
                        if name=='C006':
                            multiplier=model(torch.tensor(ck[:,None]*ce,dtype=torch.float64)).numpy()
                            return ((cq@cv)*multiplier)@cv.T
                        return model(torch.tensor(np.c_[cq,np.log(ck)],dtype=torch.float64)).numpy()
                prediction,timing=measured(predict)
                relative=np.linalg.norm(prediction-cy,axis=1)/np.maximum(np.linalg.norm(cy,axis=1),1e-14)
                row[condition]={'mean_relative_l2':float(relative.mean()),'std_relative_l2':float(relative.std(ddof=1)),
                                'max_relative_l2':float(relative.max()),**timing}
                study.record(7,name,f'{condition}_seed{seed}',prediction,cy,
                    {'seed':seed,'sample_count':len(ck),'mean_sample_relative_l2':float(relative.mean()),
                     'sample_relative_l2_std':float(relative.std(ddof=1)),
                     'governing_equation':'-k Txx=Q on (0,1), homogeneous Dirichlet','n':15},timing,
                    {'source':cq,'conductivity':ck,'per_sample_relative_error':relative})
            records.append(row)
            print(f"trained {name} seed {seed}: iid mean relative error {row['iid']['mean_relative_l2']:.6g}",flush=True)
        if seed==824:
            representative=models['C006']
    dump(study.output/'comparisons/learning.json',records)
    # Expose extrapolation even if training data primarily excite low modes.
    spectral=np.geomspace(.01,1e6,1000)
    multiplier=representative.numpy_filter(spectral)
    np.savez(study.output/'comparisons/spectral_filter.npz',eigenvalue=spectral,response=multiplier)
    fig,ax=plt.subplots(figsize=(7,4))
    ax.semilogx(spectral,spectral*multiplier,label='C006: lambda*g(lambda)')
    ax.axhline(1,color='black',linestyle='--',label='exact inverse')
    ax.set(xlabel='eigenvalue',ylabel='inverse response ratio',ylim=(-.05,1.5))
    ax.legend(); fig.tight_layout(); fig.savefig(study.output/'plots/spectral_extrapolation.png',dpi=160); plt.close(fig)
    return representative


def extended_heat(study,model):
    for n in [7,15,31,63]:
        for k in [.02,1.,20.]:
            p=heat_1d(n,conductivity=k,source=lambda x:np.exp(-((x-.3)/.05)**2))
            study.heat(8,'C006',f'n{n}_k{k}',p,lambda p=p:spectral_response(p.matrix,p.rhs,model))
            study.heat(8,'C006R',f'n{n}_k{k}',p,lambda p=p:resolvent_response(p.matrix,p.rhs),
                       {'lifetime':1e6,'equivalence':'shifted exact solve, not an inference speedup'})
            study.heat(8,'direct_solver',f'n{n}_k{k}',p,p.reference)
    cases=[('rectangle8',heat_2d(8)),('rectangle16',heat_2d(16)),
           ('aspect_changed',heat_2d(12,aspect=1.5)),('square_hole',heat_2d(16,hole=True)),
           ('heterogeneous',heat_2d(16,tensor=lambda x,y:np.eye(2)*(1 if x<.5 else 8))),
           ('rotated_anisotropy',heat_2d(16,tensor=np.array([[3.,1.],[1.,1.]]))),
           ('changed_boundary',heat_2d(12,boundary=lambda x,y:.2*x+.1*y)),
           ('local_source',heat_2d(16,source=lambda x,y:100*np.exp(-((x-.3)**2+(y-.6)**2)/.01)))]
    for case,p in cases:
        reference=p.reference()
        for name,function in [('C006',lambda p=p:p.decode(spectral_response(p.matrix,p.rhs,model))),
                              ('C006R',lambda p=p:p.decode(resolvent_response(p.matrix,p.rhs))),
                              ('direct_solver',p.reference)]:
            prediction,timing=measured(function)
            flux=p.flux(prediction); reference_flux=p.flux(reference)
            study.record(8,name,case,prediction,reference,
                {**p.diagnostics(prediction),'governing_equation':'-div(K grad T)=Q; linear triangular FEM',
                 'n_free':len(p.free),'flux_relative_l2':float(np.linalg.norm(flux-reference_flux)/max(np.linalg.norm(reference_flux),1e-14)),
                 'boundary_handling':'prescribed lifting, not learned', 'geometry':case},timing,
                {'points':p.points,'triangles':p.triangles,'predicted_flux':flux,'reference_flux':reference_flux})
        if case=='square_hole':
            fig,ax=plt.subplots(figsize=(5,4))
            image=ax.tripcolor(p.points[:,0],p.points[:,1],p.triangles,reference,shading='gouraud')
            fig.colorbar(image,ax=ax,label='reference temperature')
            ax.set_aspect('equal'); fig.tight_layout(); fig.savefig(study.output/'plots/heat_hole.png',dpi=160);plt.close(fig)
    # Independent manufactured-solution convergence, not a comparison to the same solver.
    convergence=[]
    tensor=np.array([[3.,1.],[1.,1.]])
    for m in [8,16,24]:
        source=lambda x,y:6*y*(1-y)+2*x*(1-x)-2*(1-2*x)*(1-2*y)
        p=heat_2d(m,tensor=tensor,source=source)
        reference=p.reference()
        x,y=p.points.T
        analytic=x*(1-x)*y*(1-y)
        centers=p.points[p.triangles].mean(axis=1)
        cx,cy=centers.T
        gradient=np.c_[(1-2*cx)*cy*(1-cy),(1-2*cy)*cx*(1-cx)]
        analytic_flux=-gradient@tensor.T
        convergence.append({'intervals':m,**errors(reference,analytic),
            'flux_relative_l2':float(np.linalg.norm(p.flux(reference)-analytic_flux)/np.linalg.norm(analytic_flux)),
            **p.diagnostics(reference)})
    dump(study.output/'metrics/fem_manufactured_convergence.json',convergence)


def advection_cycles(study):
    """C006R transient extension is backward Euler, not a new time integrator.

    Dropping the skew part tests why self-adjoint heat primitives cannot be
    claimed to handle directed transport without modification.
    """
    n=32; h=1/n; x=np.arange(n)*h; dt=.01; steps=10
    identity=np.eye(n); back=np.roll(identity,1,axis=0); forward=back.T
    initial=1+.3*np.cos(2*np.pi*x)
    for peclet in [.1,1.,10.]:
        for velocity in [-1.,1.]:
            k=abs(velocity)*h/peclet
            # np.roll(I,1,axis=0) maps u_i to u_{i-1}.
            transport=(identity-back)*velocity/h if velocity>0 else (identity-forward)*abs(velocity)/h
            diffusion=k/h**2*(2*identity-back-forward)
            operator=transport+diffusion
            reference=expm(-dt*steps*operator)@initial
            symmetric=(operator+operator.T)/2
            bad=expm(-dt*steps*symmetric)@initial
            state=initial.copy()
            start=time.perf_counter()
            for _ in range(steps): state=solve(identity+dt*operator,state)
            runtime=time.perf_counter()-start
            analytic=1+.3*np.exp(-k*(2*np.pi)**2*dt*steps)*np.cos(2*np.pi*(x-velocity*dt*steps))
            for name,pred in [('selfadjoint_transport_failure',bad),('backward_euler_control',state)]:
                study.record(9,name,f'periodic_Pe{peclet}_v{velocity}',pred,reference,
                    {'governing_equation':'u_t+v u_x=k u_xx; periodic; upwind spatial reference',
                     'peclet_cell':peclet,'velocity':velocity,'diffusivity':k,'dt':dt,'steps':steps,
                     'min_state':float(pred.min()),'mass_drift_abs':float(abs(h*(pred.sum()-initial.sum()))),
                     'boundary_error_abs':0.,'reference_vs_continuous_relative_l2':errors(reference,analytic)['relative_l2'],
                     'total_step_seconds':runtime if name=='backward_euler_control' else None},
                    inputs={'x':x,'initial':initial,'continuous_analytic':analytic})
    # Boundary change: inflow Dirichlet 1, right Dirichlet 0, initially zero.
    p=heat_1d(31,conductivity=.01,source=0.,boundary=(1.,0.))
    n=len(p.x); h=p.h
    adv=(np.eye(n)-np.diag(np.ones(n-1),-1))/h
    operator=p.matrix+adv
    forcing=p.rhs.copy();forcing[0]+=1/h
    augmented=np.zeros((n+1,n+1));augmented[:n,:n]=-operator;augmented[:n,-1]=forcing
    reference=(expm(dt*steps*augmented)@np.r_[np.zeros(n),1.])[:n]
    state=np.zeros(n)
    for _ in range(steps): state=solve(np.eye(n)+dt*operator,state+dt*forcing)
    study.record(9,'backward_euler_control','inflow_dirichlet',state,reference,
        {'governing_equation':'u_t+u_x=0.01 u_xx, u(0)=1, u(1)=0, initially 0',
         'min_state':float(state.min()),'max_state':float(state.max()),'boundary_error_abs':0.,'dt':dt,'steps':steps},
         inputs={'x':p.x,'initial':np.zeros(n)})


def flow_cycles(study,model):
    for n in [15,31]:
        for viscosity in [.1,1.,10.]:
            p=heat_1d(n,conductivity=viscosity)
            analytic=p.x*(1-p.x)/(2*viscosity)
            for name,function in [('C006',lambda:spectral_response(p.matrix,p.rhs,model)),
                                  ('C006R',lambda:resolvent_response(p.matrix,p.rhs))]:
                prediction,timing=measured(function)
                study.record(10,name,f'poiseuille_n{n}_mu{viscosity}',prediction,analytic,
                    {**p.diagnostics(prediction),'governing_equation':'-mu u_yy=1; v=0; no-slip; steady channel Stokes',
                     'divergence_inf':0.,'viscosity':viscosity,'boundary_error_abs':0.,'weights':'same heat-trained seed824 filter; no flow fine tuning'},timing,
                    {'y':p.x,'forcing':p.source})
    couette=heat_1d(15,source=0.,boundary=(0.,1.))
    # The linear lift handles this case exactly; do not attribute success to learning.
    study.record(10,'lifting_control','couette',couette.x,couette.x,
        {'governing_equation':'-mu u_yy=0; u(0)=0,u(1)=1; v=0',
         'divergence_inf':0.,'boundary_error_abs':0.,'note':'exact prescribed linear lift; does not validate learned primitive'})
    periodic_manufactured_flow(study,model)


def periodic_manufactured_flow(study,model):
    n=16; x=np.arange(n)/n; xx,yy=np.meshgrid(x,x,indexing='ij')
    freq=2*np.pi*np.fft.fftfreq(n,d=1/n)
    kx,ky=np.meshgrid(freq,freq,indexing='ij'); square=kx*kx+ky*ky
    inverse=np.zeros_like(square); inverse[square>0]=1/square[square>0]
    def derivative(field,k): return np.fft.ifft2(1j*k*np.fft.fft2(field)).real
    psi=np.sin(2*np.pi*xx)*np.sin(2*np.pi*yy)+.2*np.sin(4*np.pi*xx)*np.sin(2*np.pi*yy)
    u=derivative(psi,ky);v=-derivative(psi,kx)
    pressure=np.cos(2*np.pi*xx)*np.cos(2*np.pi*yy)
    exact=np.stack((u,v),axis=-1)
    for viscosity,inertia in [(1.,0.),(.1,0.),(.1,1.)]:
        lapu=np.fft.ifft2(-square*np.fft.fft2(u)).real
        lapv=np.fft.ifft2(-square*np.fft.fft2(v)).real
        fx=-viscosity*lapu+derivative(pressure,kx)+inertia*(u*derivative(u,kx)+v*derivative(u,ky))
        fy=-viscosity*lapv+derivative(pressure,ky)+inertia*(u*derivative(v,kx)+v*derivative(v,ky))
        fhat=np.stack((np.fft.fft2(fx),np.fft.fft2(fy)),axis=-1)
        dot=kx*fhat[...,0]+ky*fhat[...,1]
        projected=fhat.copy();projected[...,0]-=kx*dot*inverse;projected[...,1]-=ky*dot*inverse
        for name in ['C006','C006R','spectral_stokes_control']:
            multiplier=np.zeros_like(square); mask=square>0
            if name=='C006': multiplier[mask]=model.numpy_filter(viscosity*square[mask])
            elif name=='C006R': multiplier[mask]=1/(viscosity*square[mask]+1e-6)
            else: multiplier[mask]=1/(viscosity*square[mask])
            pred=np.fft.ifft2(projected*multiplier[...,None],axes=(0,1)).real
            div=derivative(pred[...,0],kx)+derivative(pred[...,1],ky)
            pred_hat=np.fft.fft2(pred,axes=(0,1))
            linear_residual=viscosity*square[...,None]*pred_hat-projected
            physical_residual=np.fft.ifft2(linear_residual,axes=(0,1)).real
            if inertia:
                pu,pv=pred[...,0],pred[...,1]
                conv=np.stack((pu*derivative(pu,kx)+pv*derivative(pu,ky),pu*derivative(pv,kx)+pv*derivative(pv,ky)),axis=-1)
                ch=np.fft.fft2(conv,axes=(0,1));cd=kx*ch[...,0]+ky*ch[...,1]
                ch[...,0]-=kx*cd*inverse;ch[...,1]-=ky*cd*inverse
                physical_residual+=np.fft.ifft2(ch,axes=(0,1)).real
            study.record(10,name,f'periodic_mu{viscosity}_inertia{inertia}',pred,exact,
                {'governing_equation':'rho(u.grad)u=-grad p+mu Laplacian(u)+f; divergence free; periodic manufactured solution',
                 'rho':inertia,'viscosity':viscosity,'divergence_inf':float(np.max(abs(div))),
                 'projected_momentum_residual_l2':float(np.linalg.norm(physical_residual)),
                 'boundary_error_abs':0.,'expected_limitation':'all tested response filters omit nonlinear convection'},
                inputs={'x':x,'force':np.stack((fx,fy),axis=-1),'pressure_reference':pressure})


def summarize(study):
    learning=json.loads((study.output/'comparisons/learning.json').read_text())
    aggregate={}
    for name in ['C006','MLP']:
        runs=[r for r in learning if r['model']==name]
        aggregate[name]={}
        for condition in ['iid','low_conductivity_ood','high_conductivity_ood','localized_source_ood','high_frequency_source_ood']:
            values=[r[condition]['mean_relative_l2'] for r in runs]
            aggregate[name][condition]={'mean_over_seeds':float(np.mean(values)),
                'std_over_seeds':float(np.std(values,ddof=1)), 'seed_values':values}
    dump(study.output/'comparisons/aggregate.json',aggregate)
    fig,ax=plt.subplots(figsize=(9,4))
    labels=['iid','low_conductivity_ood','high_conductivity_ood','high_frequency_source_ood']
    positions=np.arange(len(labels))
    for j,name in enumerate(['C006','MLP']):
        means=[aggregate[name][c]['mean_over_seeds'] for c in labels]
        stds=[aggregate[name][c]['std_over_seeds'] for c in labels]
        ax.bar(positions+(j-.5)*.35,means,.35,yerr=stds,label=name)
    ax.set_xticks(positions, ['IID','low k OOD','high k OOD','high frequency OOD'])
    ax.set_yscale('log');ax.set_ylabel('mean relative L2; 3 training seeds');ax.legend()
    fig.tight_layout();fig.savefig(study.output/'plots/learning_comparison.png',dpi=160);plt.close(fig)
    run=json.loads((study.output/'run.json').read_text())
    run.update({'status':'completed','completed_utc':datetime.now(timezone.utc).isoformat(),
                'total_seconds':time.perf_counter()-study.started,'recorded_case_count':len(study.rows),
                'cycle_count':10,'invention_selected':False})
    dump(study.output/'run.json',run)
    print(json.dumps({'output':str(study.output),'cases':len(study.rows),'wall_seconds':run['total_seconds'],'aggregate':aggregate},indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--phase',choices=['full','initial','continue'],default='full')
    args=parser.parse_args()
    torch.set_num_threads(1)
    study=Study.resume(args.output) if args.phase=='continue' else Study(args.output)
    if args.phase!='continue':
        initial_cycles(study)
        print('Cycles 1-6 complete',flush=True)
    if args.phase=='initial':
        run=json.loads((study.output/'run.json').read_text())
        run.update({'status':'initial_completed','initial_phase_seconds':time.perf_counter()-study.started})
        dump(study.output/'run.json',run)
        return
    model=train_models(study)
    extended_heat(study,model)
    advection_cycles(study)
    flow_cycles(study,model)
    summarize(study)


if __name__=='__main__': main()
