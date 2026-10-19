"""Cycle27: exact physical input collision for H-only nonlinear response closure."""
import json
import numpy as np
from src.physics.session0004 import steady_heat,diffusion_bands
from experiments.session0005_common import ROOT,write,snapshot,preserve


def main():
    if (ROOT/'closure_counterexample.json').exists():raise FileExistsError('closure counter already executed')
    snapshot('closure_counterexample',['experiments/session0005_closure.py','research/session_0005/cycle27_protocol.md'])
    rows=[];arrays={}
    for n in [15,31]:
        x=np.arange(1,n+1)/(n+1);f=4*x*(1-x);lam=1.5;k=np.ones(n+1)
        bands=diffusion_bands(k,n);a=np.diag(bands[1])+np.diag(bands[0,1:],1)+np.diag(bands[2,:-1],-1)
        base,_=steady_heat(k,np.zeros(n),lam=lam);responses=[];hs=[];deltas=[];newton=[];qs=[]
        for label,coeff in [('A',[.3,.3]),('B',[.1,.9])]:
            r=np.array([np.log1p(c*f) for c in coeff]);q=np.array([a@ri-lam*np.exp(base)*np.expm1(ri) for ri in r])
            s=r.sum(0);h=lam*np.exp(base)*(np.expm1(s)-np.expm1(r).sum(0))
            true,_=steady_heat(k,q.sum(0),lam=lam,initial=base+s)
            delta=true-base-s;one=np.linalg.solve(a-np.diag(lam*np.exp(base+s)),h)
            responses.append(r);hs.append(h);deltas.append(delta);newton.append(one);qs.append(q)
            arrays[f'n{n}_{label}_sources']=q;arrays[f'n{n}_{label}_responses']=r
            arrays[f'n{n}_{label}_reference']=true;arrays[f'n{n}_{label}_mixed_defect']=h
            arrays[f'n{n}_{label}_required_correction']=delta;arrays[f'n{n}_{label}_newton_correction']=one
        difference=np.linalg.norm(deltas[0]-deltas[1])
        rows.append({'n':n,'minimum_source':float(np.min(qs)),'H_input_max_difference':float(max(abs(hs[0]-hs[1]))),
            'S_state_max_difference':float(max(abs(responses[0].sum(0)-responses[1].sum(0)))),
            'required_correction_l2_difference':float(difference),'H_only_worst_case_l2_lower_bound':float(difference/2),
            'reference_correction_l2':[float(np.linalg.norm(t)) for t in deltas],
            'state_informed_one_newton_relative_errors':[float(np.linalg.norm(p-t)/np.linalg.norm(t)) for p,t in zip(newton,deltas)]})
    np.savez_compressed(ROOT/'arrays/closure_counterexample.npz',**arrays)
    write('closure_counterexample.json',{'records':rows,'verdict':'REVISE H-only closure for arbitrary physical sources: collective state S is also necessary; repair shown is classical Newton, not an invention',
        'scope':'analytic positive source family outside Gaussian training inputs; ideal physically faithful singleton footprints; not a model IID accuracy result','prior_files_preserved':preserve()})
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
