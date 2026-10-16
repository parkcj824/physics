"""Read-only prior validation, Session 0003 checkpoint/reference/metric verification."""

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.linalg import solve, solve_banded
import torch
from torch import nn

from experiments.session0003 import verify_preservation
from src.operators.session0003 import segment_response


def direct_heat(k,q,boundary):
    n=len(q);h=1/(n+1)
    k=np.broadcast_to(k,(n+1,))
    band=np.zeros((3,n));band[1]=(k[:-1]+k[1:])/h**2
    band[0,1:]=-k[1:-1]/h**2;band[2,:-1]=-k[1:-1]/h**2
    b=q.copy();b[0]+=k[0]/h**2*boundary[0];b[-1]+=k[-1]/h**2*boundary[1]
    return solve_banded((1,1),band,b)


def verify(output):
    root=Path(output)
    run=json.loads((root/"run.json").read_text())
    assert run["status"]=="completed"
    count=verify_preservation()
    rows=json.loads((root/"metrics.json").read_text())
    assert len(rows)==run["record_count"]==136
    cycles=sorted({row["cycle"] for row in rows if row["cycle"]>0})
    assert cycles==list(range(13,19))
    sources_checked=0
    for folder in (root/"source_snapshot").iterdir():
        for name,digest in json.loads((folder/"sha256.json").read_text()).items():
            assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==digest
            if folder.name=="baseline_refinement":
                assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest
            sources_checked+=1
    checkpoint_predictions=0;independent_heat_references=0;direction_cases=0
    fingerprint={}
    names=set()
    for row in rows:
        assert row["arrays"] not in names;names.add(row["arrays"])
        assert row["median_seconds"]>=0 and row["tracemalloc_peak_bytes"]>=0
        with np.load(root/row["arrays"]) as archive:
            data={name:archive[name] for name in archive.files}
        prediction,reference=data["prediction"],data["reference"]
        assert prediction.shape==reference.shape
        assert np.isfinite(prediction).all() and np.isfinite(reference).all()
        difference=prediction-reference
        rel=np.linalg.norm(difference)/max(np.linalg.norm(reference),1e-14)
        np.testing.assert_allclose(rel,row["relative_l2"],rtol=1e-12,atol=1e-15)
        np.testing.assert_allclose(np.max(abs(difference)),row["max_absolute_error"],rtol=1e-12,atol=1e-15)
        if "source" in data and "conductivity" in data:
            source=data["source"];k=data["conductivity"]
            if source.ndim==1:
                g=np.asarray(row.get("boundary",[0.,0.]))
                exact=direct_heat(k,source,g)
                independent_heat_references+=1
            else:
                g=data.get("boundary",np.zeros((len(source),2)))
                exact=np.array([direct_heat(ki,qi,gi) for ki,qi,gi in zip(k,source,g)])
                independent_heat_references+=len(source)
            np.testing.assert_allclose(reference,exact,atol=1e-10,rtol=1e-10)
        if "mean_sample_relative_l2" in row:
            per=np.linalg.norm(difference,axis=1)/np.maximum(np.linalg.norm(reference,axis=1),1e-14)
            np.testing.assert_allclose(row["mean_sample_relative_l2"],per.mean(),rtol=1e-12,atol=1e-15)
        if row["cycle"]==18:
            label=row["case"].split("_seed")[0]
            dig=hashlib.sha256(b"".join(data[name].tobytes() for name in ["source","conductivity","boundary"])).hexdigest()
            if label in fingerprint:assert fingerprint[label]==dig
            fingerprint[label]=dig
            if row["candidate"]=="MLP":
                saved=torch.load(root/f"models/MLP_seed{row['seed']}.pt",weights_only=True)
                model=nn.Sequential(nn.Linear(33,64),nn.Tanh(),nn.Linear(64,15)).double()
                model.load_state_dict(saved["state"])
                raw=np.c_[np.log(data["conductivity"]),data["source"],data["boundary"]]
                with torch.no_grad():
                    inputs=(torch.tensor(raw,dtype=torch.float64)-saved["center"])/saved["scale"]
                    reconstructed=(saved["output_scale"]*model(inputs)).numpy()
                np.testing.assert_allclose(prediction,reconstructed,atol=1e-12,rtol=1e-12)
                checkpoint_predictions+=1
            elif row["candidate"]=="D_fitted_transfer":
                theta=json.loads((root/"models/segment_theta.json").read_text())["theta"]
                reconstructed=np.array([segment_response(k,q,tuple(g),theta) for k,q,g in
                                        zip(data["conductivity"],data["source"],data["boundary"])])
                np.testing.assert_allclose(prediction,reconstructed,atol=1e-12,rtol=1e-12)
                checkpoint_predictions+=1
        if row["cycle"]==17 and row["candidate"]=="A_oriented":
            state=data["initial"].copy();n=len(state)
            for _ in range(row["steps"]):
                state=solve(np.eye(n)+row["dt"]*data["operator"],state)
            np.testing.assert_allclose(reference,state,atol=1e-12,rtol=1e-12)
            phase=-np.angle(np.fft.fft(prediction)[1])
            assert phase*row["velocity"]>0  # direction independently checked, not just v-flip difference
            direction_cases+=1
    suites=list(ET.parse(root/"validation/pytest.xml").getroot().iter("testsuite"))
    test_count=sum(int(s.attrib["tests"]) for s in suites)
    assert test_count==34
    assert all(int(s.attrib.get(k,0))==0 for s in suites for k in ["failures","errors","skipped"])
    checks={"status":"verified","records":len(rows),"cycles":cycles,"new_cycle_count":len(cycles),
            "prior_result_files_unchanged":count,"source_snapshot_files_checked":sources_checked,
            "independent_banded_reference_evaluations":independent_heat_references,
            "saved_checkpoint_predictions_reproduced":checkpoint_predictions,"signed_advection_direction_cases":direction_cases,
            "matched_test_condition_fingerprints":fingerprint,"pytest_passed":test_count,"pytest_failed":0,"pytest_skipped":0,
            "invention_selected":False,
            "interpretation":"reference evaluations include repeated comparisons on the same inputs, not independent data samples"}
    (root/"validation/integrity.json").write_text(json.dumps(checks,indent=2)+"\n")
    # Validation source is explicit; earlier scientific phase snapshots remain immutable.
    validation_files=["scripts/verify_session0003.py","tests/test_session0003.py"]
    (root/"validation/source_hashes.json").write_text(json.dumps({name:hashlib.sha256(Path(name).read_bytes()).hexdigest()
                                                                  for name in validation_files},indent=2)+"\n")
    print(json.dumps(checks,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("output",nargs="?",default="results/session_0003")
    args=parser.parse_args();torch.set_num_threads(1);verify(args.output)
