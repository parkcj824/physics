"""Cycles24/26: execute fast falsification before inventing unnecessary neural models."""
import argparse
import json
import time
import numpy as np
from scipy.linalg import expm
from src.physics.session0005 import joint_operators,defects,defect_gradient
from src.physics.session0004 import steady_heat,flux
from experiments.session0004_transport import transport_reference
from experiments.session0005_common import ROOT,write,snapshot,preserve


def compatibility():
    if (ROOT/'compatibility.json').exists():raise FileExistsError('compatibility already executed')
    snapshot('compatibility',['experiments/session0005_hypotheses.py','src/physics/session0005.py'])
    rows=[];arrays={};n=15;x=np.arange(1,n+1)/(n+1);q=np.ones(n)
    for name,k,v in [('heat',np.ones(n+1),0.),('lowk',np.full(n+1,.01),0.),
                     ('adv_plus',np.full(n+1,.1),1.),('adv_minus',np.full(n+1,.1),-1.)]:
        truth,jref=transport_reference(k,q,(0,0),v)
        d,c,cg=joint_operators(k,(0,0),v)
        pert=1e-4*np.sin(n*np.pi*x);u=truth+pert;j=jref.copy()
        b,r=defects(u,j,k,q,velocity=v)
        pde=d@(c@u+cg)-q
        identity=max(abs(pde-(b-d@r)))
        # Normal equations are the exact proposed simultaneous-defect gradient.
        lmat=np.block([[np.zeros((n,n)),d],[-c,np.eye(n+1)]])
        rhs=np.r_[q,cg];state=np.r_[u,j];g=np.r_[*defect_gradient(u,j,k,q,velocity=v)]
        equality=max(abs(g-lmat.T@(lmat@state-rhs)))
        begin=time.perf_counter();eigenbound=np.linalg.norm(lmat,2)**2;setup=time.perf_counter()-begin
        eta=.5/eigenbound;z=np.zeros(2*n+1);history=[]
        for step in range(129):
            residual=lmat@z-rhs
            if step in [0,1,8,32,128]:history.append({'step':step,'defect_squared_norm':float(residual@residual),
                'temperature_relative_l2':float(np.linalg.norm(z[:n]-truth)/np.linalg.norm(truth))})
            if step<128:z-=eta*lmat.T@residual
        # Feasibility of positivity is not guaranteed by this least-squares direction.
        small=np.full(n,1e-8);arbitrary_flux=10*np.sin(np.arange(n+1)*2.1)
        gu,gj=defect_gradient(small,arbitrary_flux,k,q,velocity=v)
        trial=small-eta*gu
        row={'case':name,'velocity':v,'conductivity':float(k[0]),
             'field_relative_l2':float(np.linalg.norm(pert)/np.linalg.norm(truth)),
             'balance_inf':float(max(abs(b))),'constitutive_inf':float(max(abs(r))),
             'pde_residual_inf':float(max(abs(pde))),'compatibility_identity_error':float(identity),
             'gradient_equals_normal_equations_error':float(equality),'stable_gradient_step':float(eta),
             'spectral_setup_seconds':setup,'joint_matrix_condition':float(np.linalg.cond(lmat)),
             'unconstrained_update_min_temperature':float(trial.min()),'gradient_history':history}
        rows.append(row)
        arrays[name+'_truth']=truth;arrays[name+'_prediction']=u;arrays[name+'_flux']=j;arrays[name+'_gradient_end']=z
    # Exact finite increments in a nonlinear thermal reaction, not just a Jacobian approximation.
    rng=np.random.default_rng(54001);u=rng.uniform(0,.3,n);j=rng.normal(size=n+1);du=rng.normal(scale=.02,size=n);dj=rng.normal(scale=.02,size=n+1)
    k=np.exp(rng.normal(scale=.2,size=n+1));d,c,_=joint_operators(k)
    b0,c0=defects(u,j,k,q,lam=1.5);b1,c1=defects(u+du,j+dj,k,q,lam=1.5)
    db=d@dj-1.5*np.exp(u)*np.expm1(du);dc=dj-c@du
    finite={'balance_increment_error':float(max(abs(b1-b0-db))),'constitutive_increment_error':float(max(abs(c1-c0-dc)))}
    (ROOT/'arrays').mkdir(exist_ok=True);np.savez_compressed(ROOT/'arrays/compatibility_counterexamples.npz',x=x,**arrays)
    write('compatibility.json',{'records':rows,'finite_increment_identity':finite,'verdict':'REJECT current gradient mechanism as independent invention: exactly joint least-squares defect correction; keep separated-defect diagnostic',
        'trained_neural_model':False,'prior_files_preserved':preserve()})
    print(json.dumps({'compatibility_cases':len(rows),'finite_increment_errors':finite,'minimum_update':min(r['unconstrained_update_min_temperature'] for r in rows)}))


def causal():
    if (ROOT/'causal.json').exists():raise FileExistsError('causal already executed')
    snapshot('causal',['experiments/session0005_hypotheses.py'])
    n=16;h=1/n;omega=2*n*np.sin(np.pi/(2*n));tau=.2;k=1.;dt=.05
    generator=np.array([[0.,-omega],[k*omega/tau,-1/tau]])
    p=expm(dt*generator);true_coef=np.array([np.trace(p),-np.linalg.det(p)])
    rng=np.random.default_rng(53001);z0=np.c_[rng.uniform(-.3,.3,8),rng.uniform(-.5,.5,8)]
    z1=z0@p.T;z2=z1@p.T;x=np.c_[z1[:,0],z0[:,0]]
    learned=np.linalg.lstsq(x,z2[:,0],rcond=None)[0]
    ar1=np.linalg.lstsq(z1[:,0,None],z2[:,0],rcond=None)[0]
    rng=np.random.default_rng(53003);initial=np.c_[rng.uniform(-.3,.3,32),rng.uniform(-.5,.5,32)]
    next1=initial@p.T;next2=next1@p.T;pred=np.c_[next1[:,0],initial[:,0]]@learned;bad=next1[:,0]*ar1[0]
    pair=np.array([[.25,.5],[.25,-.5]]);future=pair@p.T
    omega_d=np.sqrt(k*omega**2/tau-1/(4*tau**2));alias_dt=np.pi/omega_d;pa=expm(alias_dt*generator)
    at_alias=pair@pa.T;after_alias=at_alias@p.T
    energy=.5*(initial[:,0]**2+tau/k*initial[:,1]**2);energy_next=.5*(next1[:,0]**2+tau/k*next1[:,1]**2)
    alt=expm(.1*generator);a1=initial@alt.T;a2=a1@alt.T;wrong_dt=np.c_[a1[:,0],initial[:,0]]@learned
    result={'physical_equations':'T_t=-j_x; tau j_t+j=-k T_x; closed Neumann flux; one cosine/sine staggered mode',
        'n':n,'omega_discrete':float(omega),'tau':tau,'k':k,'dt':dt,'training_initial_states':8,'confirm_initial_states':32,
        'learned_ar2_coefficients':learned.tolist(),'cayley_hamilton_coefficients':true_coef.tolist(),
        'coefficient_identity_max_error':float(max(abs(learned-true_coef))),
        'ar2_max_absolute_error':float(max(abs(pred-next2[:,0]))),'ar1_max_absolute_error':float(max(abs(bad-next2[:,0]))),
        'same_temperature_distinct_flux_future_gap':float(abs(future[0,0]-future[1,0])),
        'observability_determinant':float(np.linalg.det(np.array([[1,0],p[0]]))),
        'alias_dt':float(alias_dt),'alias_observation_gap':float(abs(at_alias[0,0]-at_alias[1,0])),
        'after_alias_future_gap':float(abs(after_alias[0,0]-after_alias[1,0])),
        'alias_observability_condition':float(np.linalg.cond(np.array([[1,0],pa[0]]))),
        'energy_max_increase':float(max(energy_next-energy)),
        'fixed_coefficients_new_dt_max_error':float(max(abs(wrong_dt-a2[:,0]))),
        'verdict':'REJECT AR2/observability implementation as a new primitive; hidden flux explains temperature-history benefit, full physical state is Markov',
        'fitting':'2 coefficients by least squares, not neural network training','prior_files_preserved':preserve()}
    np.savez_compressed(ROOT/'arrays/causal_counterexamples.npz',generator=generator,transition=p,train_initial=z0,train_first=z1,train_second=z2,
         confirm_initial=initial,confirm_first=next1,reference=next2[:,0],prediction=pred,ar1_prediction=bad,
         ambiguous_pair=pair,ambiguous_future=future,alias_observations=at_alias,after_alias=after_alias)
    write('causal.json',result);print(json.dumps({k:result[k] for k in ['ar2_max_absolute_error','ar1_max_absolute_error','alias_observation_gap','after_alias_future_gap']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['compatibility','causal']);a=p.parse_args();{'compatibility':compatibility,'causal':causal}[a.phase]()
