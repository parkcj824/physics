"""Supplementary reproducibility measurements, not original-run timing or new discoveries."""
import json
import numpy as np
from scipy.linalg import expm
from experiments.session0005_common import ROOT,write,snapshot,preserve,measure,now
from src.physics.session0005 import joint_operators
from src.physics.session0004 import steady_heat


def main():
    if (ROOT/'supplementary_costs.json').exists():raise FileExistsError('cost reproduction already measured')
    snapshot('cost_replay',['experiments/session0005_costs.py'])
    jrows=json.loads((ROOT/'compatibility.json').read_text())['records'];a=np.load(ROOT/'arrays/compatibility_counterexamples.npz');jc=[]
    for row in jrows:
        def calculation():
            n=15;d,c,cg=joint_operators(np.full(16,row['conductivity']),velocity=row['velocity'])
            l=np.block([[np.zeros((n,n)),d],[-c,np.eye(n+1)]]);rhs=np.r_[np.ones(n),cg]
            eta=.5/np.linalg.norm(l,2)**2;z=np.zeros(2*n+1)
            for _ in range(128):z-=eta*l.T@(l@z-rhs)
            return z
        z,cost=measure(calculation);np.testing.assert_allclose(z,a[row['case']+'_gradient_end'],atol=1e-13)
        jc.append({'case':row['case'],**cost,'scope':'dense operators, spectral norm setup and128 gradient steps; reference labels already available'})
    ca=np.load(ROOT/'arrays/causal_counterexamples.npz')
    def causal():
        p=expm(.05*ca['generator']);z0=ca['train_initial'];z1=z0@p.T;z2=z1@p.T
        coeff=np.linalg.lstsq(np.c_[z1[:,0],z0[:,0]],z2[:,0],rcond=None)[0]
        initial=ca['confirm_initial'];first=initial@p.T
        return np.c_[first[:,0],initial[:,0]]@coeff
    pred,kcost=measure(causal);np.testing.assert_allclose(pred,ca['prediction'],atol=1e-14)
    cl=[];a=np.load(ROOT/'arrays/closure_counterexample.npz')
    for n in [15,31]:
        def closure():
            x=np.arange(1,n+1)/(n+1);k=np.ones(n+1);d,c,_=joint_operators(k);amat=d@c;lam=1.5
            base=steady_heat(k,np.zeros(n),lam=lam)[0];outputs=[]
            for co in [[.3,.3],[.1,.9]]:
                r=np.log1p(np.array(co)[:,None]*4*x*(1-x));s=r.sum(0)
                q=r@amat.T-lam*np.exp(base)*np.expm1(r)
                h=lam*np.exp(base)*(np.expm1(s)-np.expm1(r).sum(0))
                t=steady_heat(k,q.sum(0),lam=lam,initial=base+s)[0]
                one=np.linalg.solve(amat-np.diag(lam*np.exp(base+s)),h);outputs.append((t-base-s,one))
            return outputs
        outputs,cost=measure(closure)
        for label,(delta,one) in zip(['A','B'],outputs):
            np.testing.assert_allclose(delta,a[f'n{n}_{label}_required_correction'],atol=1e-12)
            np.testing.assert_allclose(one,a[f'n{n}_{label}_newton_correction'],atol=1e-12)
        cl.append({'n':n,**cost,'scope':'base root, analytical source construction, two total-source roots and dense one-Newton corrections'})
    dataset=[]
    for path in sorted((ROOT/'data').glob('*.npz')):
        with np.load(path) as d:
            dataset.append({'file':str(path.relative_to(ROOT)),'input_sets':int(len(d['inputs']) if 'inputs' in d else len(d['sources'])),
                'stored_subset_reference_records':int(d['valid_subsets'].sum()) if 'valid_subsets' in d else None,
                'compressed_bytes':path.stat().st_size,'uncompressed_array_bytes':sum(d[k].nbytes for k in d.files)})
    write('supplementary_costs.json',{'measured_utc':now(),'compatibility':jc,'causal':kcost,'closure':cl,'datasets':dataset,
        'scope':'same saved conditions, reproducibility measurements; no refit/model change/independent discovery. Original whole J/K/closure runtime was not recorded and is not reconstructed as if measured then.',
        'preparation_limits':'controls and initial F preparation seconds are in manifests; original extra_lowk generation runtime was not recorded. Read costs and subset-label counts separately.',
        'causal_scope':'transition exponential, training eight histories, two-coefficient LS and32 predictions including first observation generation',
        'prior_files_preserved':preserve()})
    print(json.dumps({'compatibility_cost_rows':len(jc),'causal_cost_rows':1,'closure_cost_rows':len(cl),'dataset_rows':len(dataset)}))


if __name__=='__main__':main()
