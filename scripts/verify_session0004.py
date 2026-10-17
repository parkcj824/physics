"""Independent reference/metric/checkpoint audit; writes only Session 0004 validation."""

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import root
import torch

from experiments.session0004 import ROOT, tensor, quartet_input, restore, preserve
from src.physics.session0004 import bratu_branches
from experiments.session0004_transport import TransportCoGeneration
from src.operators.session0004 import CoGenerationModel


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def independent_physics_metrics(data,row):
    family=row['family'];p=data['prediction'];residuals=[];energies=[];generated=[]
    if family=='F':p=p[:,3]
    if family not in ['F','G']:return
    for index,t in enumerate(p):
        n=len(t)
        if family=='F':
            a,b,c1,c2=data['inputs'][index];x=data['x']
            k=np.ones(n+1);q=a*np.exp(-.5*((x-c1)/.1)**2)+b*np.exp(-.5*((x-c2)/.1)**2);g=(0.,0.);lam=1.5
        else:
            z=data['inputs'][index];k=np.exp(z[:n+1]);q=z[n+1:2*n+1];g=z[2*n+1:2*n+3];lam=z[-1]
        j=-k*np.diff(np.r_[g[0],t,g[1]])*(n+1);s=q+lam*np.exp(t)
        residuals.append(max(abs(np.diff(j)*(n+1)-s)))
        energies.append(abs(j[-1]-j[0]-s.sum()/(n+1)))
        if family=='G':
            jj=data['predicted_flux'][index];generated.append(abs(jj[-1]-jj[0]-s.sum()/(n+1)))
    np.testing.assert_allclose(row['mean_pde_residual_inf'],np.mean(residuals),rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(row['mean_energy_balance_abs'],np.mean(energies),rtol=1e-12,atol=1e-12)
    if generated:np.testing.assert_allclose(row['max_generated_energy_balance_abs'],max(generated),rtol=1e-12,atol=1e-13)


def independent_reference(k,q,g,lam):
    n=len(q);h=1/(n+1)
    matrix=np.diag((k[:-1]+k[1:])/h**2)+np.diag(-k[1:-1]/h**2,1)+np.diag(-k[1:-1]/h**2,-1)
    rhs=q.copy();rhs[0]+=k[0]*g[0]/h**2;rhs[-1]+=k[-1]*g[1]/h**2
    sol=root(lambda t:matrix@t-rhs-lam*np.exp(t),np.zeros(n),
             jac=lambda t:matrix-np.diag(lam*np.exp(t)),tol=1e-11)
    # scipy can report no further progress at roundoff; judge the actual equation as well.
    assert max(abs(matrix@sol.x-rhs-lam*np.exp(sol.x)))<5e-10
    return sol.x


def verify():
    count=preserve();datasets=json.loads((ROOT/"dataset_manifest.json").read_text());references=0
    for name,expected in datasets["sha256"].items():
        path=ROOT/"data"/(name+".npz");assert digest(path)==expected
        with np.load(path) as data:
            if name.startswith("E_") and not name.endswith("absent"):
                for lam,target in zip(data["lam"],data["target"]):
                    np.testing.assert_allclose(target,bratu_branches(lam,data["x"])[0],rtol=1e-13,atol=1e-13)
                    references+=2
            elif name.startswith("F_"):
                n=len(data["x"]);x=data["x"]
                for z,targets in zip(data["inputs"],data["target"]):
                    a,b,c1,c2=z
                    for (aa,bb),truth in zip([(0,0),(a,0),(0,b),(a,b)],targets):
                        q=aa*np.exp(-.5*((x-c1)/.1)**2)+bb*np.exp(-.5*((x-c2)/.1)**2)
                        np.testing.assert_allclose(truth,independent_reference(np.ones(n+1),q,np.zeros(2),1.5),atol=5e-11,rtol=5e-10)
                        references+=1
            elif name.startswith("G_"):
                n=len(data["x"])
                for z,truth,j in zip(data["inputs"],data["target"],data["flux"]):
                    k=np.exp(z[:n+1]);q=z[n+1:2*n+1];g=z[2*n+1:2*n+3];lam=z[-1]
                    expected=independent_reference(k,q,g,lam)
                    np.testing.assert_allclose(truth,expected,rtol=5e-10,atol=5e-11)
                    np.testing.assert_allclose(j,-k*np.diff(np.r_[g[0],expected,g[1]])*(n+1),rtol=5e-9,atol=5e-10)
                    references+=1
    frozen=json.loads((ROOT/"frozen_models.json").read_text());confirmed=json.loads((ROOT/"confirmation.json").read_text())
    assert frozen["frozen_utc"]<confirmed["evaluated_utc"]
    assert len(frozen["models"])==36
    bykey={(i["family"],i["mode"],i["seed"]):i for i in frozen["models"]}
    models={key:restore(info) for key,info in bykey.items()};seen=set();replays=0
    with torch.no_grad():
        for row in confirmed["records"]:
            assert row["arrays"] not in seen;seen.add(row["arrays"])
            key=(row["family"],row["mode"],row["seed"]);model=models[key]
            assert row["checkpoint_sha256"]==bykey[key]["sha256"]
            dataset_key=row["family"]+"_confirm_"+row["case"]
            assert row["input_sha256"]==datasets["sha256"][dataset_key]
            with np.load(ROOT/row["arrays"]) as data:
                if row["family"]=="E":
                    if row["case"]=="absent":
                        counts=(2*(data["lam"]<float(model.critical))).astype(int) if row["mode"].startswith("paired") else np.full(len(data["lam"]),2)
                        np.testing.assert_array_equal(counts,data["predicted_branch_count"])
                        assert row["spurious_branches"]==counts.sum();replays+=1;continue
                    pred=model(tensor(data["lam"]),tensor(data["x"])).numpy()
                    np.testing.assert_allclose(pred,data["raw_prediction"],rtol=1e-12,atol=1e-12)
                    mask=data["swapped_assignment"];pred[mask]=pred[mask,::-1]
                    truth=data["target"]
                elif row["family"]=="F":
                    inp=quartet_input(tensor(data["inputs"]));pred=model(inp.flatten(0,1),tensor(data["x"])).reshape(len(inp),4,-1).numpy()
                    truth=data["target"]
                else:
                    pred,j=model(tensor(data["inputs"]));pred=pred.numpy();j=j.numpy()
                    np.testing.assert_allclose(j,data["predicted_flux"],rtol=1e-12,atol=1e-12);truth=data["target"]
                np.testing.assert_allclose(pred,data["prediction"],rtol=1e-12,atol=1e-12)
                if row["family"]=="F":pred,truth=pred[:,3],truth[:,3]
                error=np.linalg.norm(pred-truth,axis=-1)/np.maximum(np.linalg.norm(truth,axis=-1),1e-14)
                np.testing.assert_allclose(row["mean_relative_l2"],error.mean(),rtol=1e-12,atol=1e-14)
                np.testing.assert_allclose(row["max_absolute_error"],abs(pred-truth).max(),rtol=1e-12,atol=1e-14)
                assert row["median_seconds"]>0 and row["python_tracemalloc_peak_bytes"]>=0
                independent_physics_metrics(data,row)
                replays+=1
    assert len(confirmed["records"])==132
    # Directed-domain reuse: independently assemble the upwind matrix and audit all labels.
    transport_manifest=json.loads((ROOT/"transport_dataset_manifest.json").read_text())
    for name,expected in transport_manifest['sha256'].items():
        path=ROOT/'data'/('H_'+name+'.npz');assert digest(path)==expected
        with np.load(path) as d:
            for z,truth,j in zip(d['inputs'],d['target'],d['flux']):
                nu=np.exp(z[:16]);q=z[16:31];g=z[31:33];v=z[-1];vp=max(v,0);vm=min(v,0)
                mat=np.diag((nu[:-1]+nu[1:])*256+abs(v)*16)+np.diag(-nu[1:-1]*256+vm*16,1)+np.diag(-nu[1:-1]*256-vp*16,-1)
                rhs=q.copy();rhs[0]+=(nu[0]*256+vp*16)*g[0];rhs[-1]+=(nu[-1]*256-vm*16)*g[1]
                state=np.linalg.solve(mat,rhs);np.testing.assert_allclose(truth,state,rtol=1e-11,atol=1e-12)
                full=np.r_[g[0],state,g[1]];current=vp*full[:-1]+vm*full[1:]-nu*np.diff(full)*16
                np.testing.assert_allclose(j,current,rtol=1e-11,atol=1e-12);references+=1
    tfrozen=json.loads((ROOT/'transport_frozen.json').read_text());tconfirmed=json.loads((ROOT/'transport_confirmation.json').read_text())
    assert tfrozen['frozen_utc']<tconfirmed['evaluated_utc'];transport_replays=0
    tmodels={}
    for info in tfrozen['models']:
        assert digest(ROOT/info['checkpoint'])==info['sha256']
        m=TransportCoGeneration(mode=info['mode']).double()
        m.load_state_dict(torch.load(ROOT/info['checkpoint'],weights_only=True)['state']);tmodels[(info['mode'],info['seed'])]=m
    with torch.no_grad():
        for row in tconfirmed['records']:
            assert row['input_sha256']==transport_manifest['sha256']['confirm_'+row['case']]
            with np.load(ROOT/row['arrays']) as d:
                u,j=tmodels[(row['mode'],row['seed'])](tensor(d['inputs']));u=u.numpy();j=j.numpy()
                np.testing.assert_allclose(u,d['prediction'],rtol=1e-12,atol=1e-12)
                np.testing.assert_allclose(j,d['predicted_flux'],rtol=1e-12,atol=1e-12)
                rel=np.linalg.norm(u-d['target'],axis=1)/np.maximum(np.linalg.norm(d['target'],axis=1),1e-14)
                np.testing.assert_allclose(row['mean_relative_l2'],rel.mean(),rtol=1e-12,atol=1e-14)
                if row['case']=='direction':
                    centroid=(u*d['x']).sum(1)/u.sum(1)
                    assert row['correct_centroid_direction_pairs']==sum(centroid[::2]>centroid[1::2])
                transport_replays+=1
    assert len(tfrozen['models'])==9 and transport_replays==27
    # Follow-up uses fresh test data after discovering the active-capacity confound.
    capmanifest=json.loads((ROOT/'capacity_dataset_manifest.json').read_text())
    capfrozen=json.loads((ROOT/'capacity_frozen.json').read_text())
    capconfirmed=json.loads((ROOT/'capacity_confirmation.json').read_text())
    assert capfrozen['frozen_utc']<capconfirmed['evaluated_utc']
    assert capmanifest['sha256']['G']!=datasets['sha256']['G_confirm_iid']
    assert capmanifest['sha256']['H']!=transport_manifest['sha256']['confirm_iid']
    for family,expected in capmanifest['sha256'].items():
        path=ROOT/'data'/f'{family}_capacity_confirm.npz';assert digest(path)==expected
        with np.load(path) as data:
            for z,truth in zip(data['inputs'],data['target']):
                if family=='G':
                    state=independent_reference(np.exp(z[:16]),z[16:31],z[31:33],z[-1])
                else:
                    nu=np.exp(z[:16]);v=z[-1];vp=max(v,0);vm=min(v,0)
                    mat=np.diag((nu[:-1]+nu[1:])*256+abs(v)*16)+np.diag(-nu[1:-1]*256+vm*16,1)+np.diag(-nu[1:-1]*256-vp*16,-1)
                    rhs=z[16:31].copy();rhs[0]+=(nu[0]*256+vp*16)*z[31];rhs[-1]+=(nu[-1]*256-vm*16)*z[32]
                    state=np.linalg.solve(mat,rhs)
                np.testing.assert_allclose(truth,state,atol=5e-11,rtol=5e-10);references+=1
    capreplays=0
    with torch.no_grad():
        for row in capconfirmed['records']:
            assert digest(ROOT/row['checkpoint'])==row['checkpoint_sha256']
            if row['mode']=='direct_width64':
                model=(CoGenerationModel if row['family']=='G' else TransportCoGeneration)(mode='direct',width=64).double()
                model.load_state_dict(torch.load(ROOT/row['checkpoint'],weights_only=True)['state'])
            elif row['family']=='G':model=models[('G',row['mode'],row['seed'])]
            else:model=tmodels[(row['mode'],row['seed'])]
            with np.load(ROOT/row['arrays']) as data:
                u,j=model(tensor(data['inputs']));u=u.numpy();j=j.numpy()
                np.testing.assert_allclose(u,data['prediction'],atol=1e-12,rtol=1e-12)
                np.testing.assert_allclose(j,data['predicted_flux'],atol=1e-12,rtol=1e-12)
                relative=np.linalg.norm(u-data['target'],axis=1)/np.maximum(np.linalg.norm(data['target'],axis=1),1e-14)
                np.testing.assert_allclose(row['mean_relative_l2'],relative.mean(),rtol=1e-12,atol=1e-14)
                assert row['input_sha256']==capmanifest['sha256'][row['family']];capreplays+=1
    assert len(capfrozen['models'])==6 and capreplays==24
    positivity=json.loads((ROOT/'transport_positivity_audit.json').read_text())['records']
    for p in positivity:
        path=ROOT/'arrays'/f"H_{p['mode']}_{p['seed']}_direction.npz"
        with np.load(path) as data:
            assert np.all(data['inputs'][:,16:31]>=0) and np.all(data['inputs'][:,31:33]==0)
            assert p['negative_samples']==sum(data['prediction'].min(1)< -1e-12)
            np.testing.assert_allclose(p['minimum_prediction'],data['prediction'].min(),atol=1e-14)
            assert data['target'].min()>0
    snapshots=0
    for phase in (ROOT/"source_snapshot").iterdir():
        for name,expected in json.loads((phase/"sha256.json").read_text()).items():
            assert digest(phase/name)==expected;snapshots+=1
            if phase.name=="confirmation":assert digest(name)==expected
    for name,expected in json.loads((ROOT/'transport_source_snapshot/sha256.json').read_text()).items():
        assert digest(ROOT/'transport_source_snapshot'/name)==expected
        assert digest(name)==expected;snapshots+=1
    for name,expected in json.loads((ROOT/'capacity_source_snapshot/sha256.json').read_text()).items():
        assert digest(ROOT/'capacity_source_snapshot'/name)==expected
        assert digest(name)==expected;snapshots+=1
    tests=ET.parse(ROOT/"validation/pytest.xml").getroot()
    suites=list(tests.iter("testsuite"));testcount=sum(int(s.attrib["tests"]) for s in suites)
    assert all(int(s.attrib.get(k,0))==0 for s in suites for k in ["failures","errors","skipped"])
    result={"status":"verified","prior_tracked_files_unchanged":count,"dataset_files":len(datasets["sha256"])+len(transport_manifest['sha256'])+len(capmanifest['sha256']),
            "reference_branch_fields_or_discrete_solutions_checked":references,"trained_models":len(frozen["models"])+len(tfrozen['models'])+len(capfrozen['models']),
            "checkpoint_prediction_batches_reproduced":replays+transport_replays+capreplays,"confirmation_records":len(confirmed["records"])+transport_replays+capreplays,
            "posthoc_positivity_records_verified":len(positivity),
            "source_snapshot_files_checked":snapshots,"pytest_passed":testcount,"pytest_failed":0,"pytest_skipped":0,
            "confirmation_after_freeze":True,"new_architecture_proven":False,
            "scope":"repeated method predictions are not independent samples; stored timestamps are provenance, not blind access enforcement"}
    output=ROOT/"validation/integrity.json";output.parent.mkdir(exist_ok=True,parents=True)
    output.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result,indent=2))


if __name__=="__main__":
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64);verify()
