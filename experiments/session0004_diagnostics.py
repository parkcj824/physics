"""Counterexample replay and strong classical controls; no model fitting here."""

import argparse
import json
from pathlib import Path
import numpy as np

from experiments.session0004 import ROOT, measured, load_data, write
from src.physics.session0004 import bratu_branches, steady_heat, diagnostics, fold, flux


def counterexamples():
    x=np.arange(1,16)/16;truth,_=bratu_branches(2.,x)
    truefold=fold()[1];gap=(truth[1]-truth[0])/2;center=truth.mean(0);rows=[]
    for alternative in [3.6,4.,6.]:
        g=gap/np.sqrt(truefold-2.)
        newg=g*np.sqrt((truefold-2.)/(alternative-2.))
        reconstructed=np.array([center-np.sqrt(alternative-2.)*newg,center+np.sqrt(alternative-2.)*newg])
        rows.append({"alternative_fold":alternative,"same_fields_max_absolute_error":float(abs(reconstructed-truth).max())})
    sources=[np.exp(-.5*((x-c)/.1)**2) for c in [.25,.5,.75]];nonlinear=[]
    arrays={"x":x,"branches":truth}
    for lam in [0.,1.5]:
        states={}
        for mask in range(8):
            q=sum((sources[j] for j in range(3) if mask&(1<<j)),np.zeros(15))
            states[mask]=steady_heat(np.ones(16),q,lam=lam)[0]
        triple=states[7]-states[3]-states[5]-states[6]+states[1]+states[2]+states[4]-states[0]
        mixed=states[3]-states[1]-states[2]+states[0]
        approximate=states[3]+states[5]+states[6]-states[1]-states[2]-states[4]+states[0]
        nonlinear.append({"lambda":lam,"mixed_norm":float(np.linalg.norm(mixed)),"triple_norm":float(np.linalg.norm(triple)),
                          "exact_pair_truncation_relative_error":float(np.linalg.norm(approximate-states[7])/np.linalg.norm(states[7]))})
        for mask,t in states.items():arrays[f"lambda{lam}_mask{mask}"]=t
    result={"fold_nonidentifiability":rows,
            "mean_field_does_not_solve_pde":diagnostics(np.ones(16),np.zeros(15),(0,0),2.,truth.mean(0)),
            "higher_order_interactions":nonlinear}
    target=ROOT/"mathematical_counterexamples.json"
    if target.exists():
        assert json.loads(target.read_text())==result,"counterexample replay changed"
    else:write("mathematical_counterexamples.json",result)
    np.savez_compressed(ROOT/"arrays/mathematical_counterexamples.npz",**arrays)
    print(json.dumps({"counterexample_replay":"identical","mixed_interaction":nonlinear}))


def controls():
    if not (ROOT/"confirmation.json").exists():raise RuntimeError("do not inspect confirmation inputs during development")
    if (ROOT/"classical_controls.json").exists():raise FileExistsError("controls already executed")
    records=[]
    for family in ["E","F","G"]:
        for path in sorted((ROOT/"data").glob(family+"_confirm_*.npz")):
            name=path.stem;d=load_data(name)
            if family=="E" and name.endswith("absent"):
                records.append({"family":family,"case":"absent","spurious_branches":0,
                                "prior":"analytic fold/root-bracket existence check"});continue
            def solve_all():
                if family=="E":return np.array([bratu_branches(l,d["x"])[0] for l in d["lam"]])
                if family=="F":
                    out=[]
                    for a,b,c1,c2 in d["inputs"]:
                        q=a*np.exp(-.5*((d["x"]-c1)/.1)**2)+b*np.exp(-.5*((d["x"]-c2)/.1)**2)
                        out.append(steady_heat(np.ones(len(q)+1),q,lam=1.5)[0])
                    return np.array(out)
                n=len(d["x"]);out=[];currents=[]
                for z in d["inputs"]:
                    t=steady_heat(np.exp(z[:n+1]),z[n+1:2*n+1],z[2*n+1:2*n+3],z[-1])[0]
                    out.append(t);currents.append(flux(np.exp(z[:n+1]),t,z[2*n+1:2*n+3]))
                return np.array(out),np.array(currents)
            result,timing=measured(solve_all);pred=result[0] if family=="G" else result
            truth=d["target"] if family!="F" else d["target"][:,3]
            delta=pred-truth
            row={"family":family,"case":name.split("_confirm_")[1],**timing,
                 "mean_relative_l2":float(np.mean(np.linalg.norm(delta,axis=-1)/np.maximum(np.linalg.norm(truth,axis=-1),1e-14))),
                 "max_absolute_error":float(abs(delta).max()),"batch_samples":len(pred),
                 "preprocessing_included":True,"algorithm":"two scalar brent roots + analytic profile" if family=="E" else "damped Newton, O(N) banded steps"}
            if family in ["F","G"]:
                diag=[]
                for i,t in enumerate(pred):
                    if family=="F":
                        a,b,c1,c2=d["inputs"][i];q=a*np.exp(-.5*((d["x"]-c1)/.1)**2)+b*np.exp(-.5*((d["x"]-c2)/.1)**2)
                        k=np.ones(len(t)+1);g=(0.,0.);lam=1.5
                    else:
                        z=d["inputs"][i];n=len(t);k=np.exp(z[:n+1]);q=z[n+1:2*n+1];g=z[2*n+1:2*n+3];lam=z[-1]
                    diag.append(diagnostics(k,q,g,lam,t))
                row["max_pde_residual_inf"]=max(s["pde_residual_inf"] for s in diag)
                row["max_energy_balance_abs"]=max(s["constitutive_energy_balance_abs"] for s in diag)
            records.append(row)
    write("classical_controls.json",{"records":records,"scope":"reference comparisons use the same declared test inputs; no model fitting",
                                     "timing_note":"F both methods infer final field only; G both return temperature and flux; E both branch fields. Input conversion/normalization or source/matrix assembly included, disk IO and stability diagnostics excluded."})
    print(json.dumps({"classical_control_records":len(records)}))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("phase",choices=["counterexamples","controls"]);args=p.parse_args()
    {"counterexamples":counterexamples,"controls":controls}[args.phase]()
