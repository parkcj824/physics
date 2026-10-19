"""Post-counterexample derivation check of the exact state-conditioned correction equation."""
import json
import numpy as np
from src.physics.session0005 import joint_operators
from src.physics.session0004 import steady_heat
from experiments.session0005_common import ROOT,write,snapshot,preserve,now


def main():
    if (ROOT/'closure_equation_audit.json').exists():raise FileExistsError('closure equation audit exists')
    snapshot('closure_equation_audit',['experiments/session0005_closure_audit.py'])
    data=np.load(ROOT/'arrays/closure_counterexample.npz');rows=[]
    for n in [15,31]:
        d,c,_=joint_operators(np.ones(n+1));a=d@c;lam=1.5
        base=steady_heat(np.ones(n+1),np.zeros(n),lam=lam)[0]
        for label in ['A','B']:
            s=data[f'n{n}_{label}_responses'].sum(0);h=data[f'n{n}_{label}_mixed_defect']
            delta=data[f'n{n}_{label}_required_correction'];v=lam*np.exp(base+s)
            res=a@delta-v*np.expm1(delta)-h
            rows.append({'n':n,'case':label,'exact_correction_equation_residual_inf':float(max(abs(res))),
                'state_dependent_reaction_coefficient_max':float(v.max())})
            assert max(abs(res))<1e-9
    write('closure_equation_audit.json',{'evaluated_utc':now(),'records':rows,
        'equation':'A_h delta - lambda exp(T0+S) expm1(delta) - H = 0',
        'scope':'post-counterexample algebra audit on frozen fields; not a new experiment/model or independent confirmation. This is the original nonlinear PDE in correction coordinates, not a new solver/primitive. S,H plus operator/base parameters and branch selection may close the problem; existence/uniqueness are not automatically guaranteed.',
        'prior_files_preserved':preserve()})
    print(json.dumps(rows))


if __name__=='__main__':main()
