"""Session 0003: staged CPU falsification, immutable prior results, no publication."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
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
from scipy.linalg import expm, solve, solve_banded
import torch
from torch import nn

from src.physics.heat import heat_1d
from src.operators.primitives import LifetimeResponse
from src.operators.session0003 import (
    incidence, material_response, oriented_inverse_update, balance_probes,
    witness_decode, witness_collision, compose_responses,
)

PRIOR = Path("results/session_0002_verified")
PRESERVATION = Path("research/session_0003/prior_result_sha256.json")


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def verify_preservation():
    manifest = json.loads(PRESERVATION.read_text())
    assert all(Path(name).is_file() and hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest
               for name, digest in manifest.items()), "Prior result changed"
    return len(manifest)


def measure(function, repeats=7):
    result = function()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        times.append(time.perf_counter()-start)
    tracemalloc.start()
    function()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, {"median_seconds": float(np.median(times)), "repeats": repeats,
                    "tracemalloc_peak_bytes": peak,
                    "process_rss_high_water_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    "memory_scope": "traced allocations omit some native/BLAS memory; RSS cumulative across process"}


def error_metrics(prediction, reference):
    difference = prediction-reference
    return {"relative_l2": float(np.linalg.norm(difference)/max(np.linalg.norm(reference), 1e-14)),
            "max_absolute_error": float(np.max(abs(difference)))}


class Study:
    def __init__(self, output, resume=False):
        self.root = Path(output)
        self.start = time.perf_counter()
        if resume:
            run = json.loads((self.root/"run.json").read_text())
            assert run["status"] == "initial_completed"
            self.rows = json.loads((self.root/"metrics.json").read_text())
            self.phase = "continuation"
        else:
            self.root.mkdir(parents=True, exist_ok=False)
            for name in ["arrays", "plots", "models", "source_snapshot"]:
                (self.root/name).mkdir()
            self.rows = []
            self.phase = "initial"
            dump(self.root/"run.json", {"status": "running", "session": "0003",
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "versions": {"python": platform.python_version(), "numpy": np.__version__,
                             "scipy": scipy.__version__, "torch": torch.__version__, "matplotlib": matplotlib.__version__},
                "torch_threads": torch.get_num_threads(), "cuda_available": torch.cuda.is_available(),
                "external_actions": "none; no push, publication, paid resource or GPU training"})
        self.snapshot()

    def snapshot(self):
        paths = ["src/operators/session0003.py", "experiments/session0003.py",
                 "src/physics/heat.py", "src/operators/primitives.py", "requirements-cpu.lock"]
        hashes = {}
        for path in paths:
            data = Path(path).read_bytes()
            target = self.root/"source_snapshot"/self.phase/path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            hashes[path] = hashlib.sha256(data).hexdigest()
        dump(self.root/"source_snapshot"/self.phase/"sha256.json", hashes)

    def record(self, cycle, candidate, case, prediction, reference, timing, extra=None, arrays=None):
        index = len(self.rows)
        array_path = f"arrays/{index:03d}_{candidate}_{case}.npz"
        np.savez_compressed(self.root/array_path, prediction=prediction, reference=reference, **(arrays or {}))
        row = {"cycle": cycle, "candidate": candidate, "case": case, "phase": self.phase,
               **error_metrics(np.asarray(prediction), np.asarray(reference)), **timing,
               "arrays": array_path, **(extra or {})}
        self.rows.append(row)
        dump(self.root/"metrics.json", self.rows)
        return row

    def heat(self, cycle, candidate, case, problem, function, extra=None):
        prediction, timing = measure(function)
        reference = problem.reference()
        flux = problem.flux(prediction)
        ref_flux = problem.flux(reference)
        return self.record(cycle, candidate, case, prediction, reference, timing,
            {"equation": "-(k T_x)_x=Q", "n": len(problem.x), **problem.diagnostics(prediction),
             "flux_relative_l2": float(np.linalg.norm(flux-ref_flux)/max(np.linalg.norm(ref_flux),1e-14)),
             "k_min": float(problem.edges_k.min()), "k_max": float(problem.edges_k.max()),
             "boundary": list(problem.boundary), **(extra or {})},
            {"x": problem.x, "conductivity": problem.edges_k, "source": problem.source,
             "predicted_flux": flux, "reference_flux": ref_flux})

    def checkpoint(self, status):
        run = json.loads((self.root/"run.json").read_text())
        run.update({"status": status, "record_count": len(self.rows),
                    f"{self.phase}_compute_seconds": time.perf_counter()-self.start,
                    "prior_result_files_unchanged": verify_preservation()})
        dump(self.root/"run.json", run)


def c006_audit(study):
    old_rows = json.loads((PRIOR/"metrics/all_cases.json").read_text())
    output = []
    for seed in [824, 825, 826]:
        model = LifetimeResponse(seed=seed)
        model.load_state_dict(torch.load(PRIOR/f"models/C006_seed{seed}.pt", weights_only=True))
        masses = model.log_mass.detach().exp().numpy()
        for label in ["iid", "low_conductivity_ood", "high_frequency_source_ood"]:
            row = next(r for r in old_rows if r["candidate"] == "C006" and r["case"] == f"{label}_seed{seed}")
            with np.load(PRIOR/row["predictions_file"]) as d:
                q, k, reference, old_prediction = (d[name] for name in ["source", "conductivity", "reference", "prediction"])
            eig, v = np.linalg.eigh(heat_1d(15).matrix)
            lam = k[:, None]*eig
            multiplier = model.numpy_filter(lam)
            exact_coefficients = (q@v)/lam
            modal_error_sq = ((1-lam*multiplier)*exact_coefficients)**2
            predicted_relative = np.sqrt(modal_error_sq.sum(1)/(exact_coefficients**2).sum(1))
            def infer():
                return ((q@v)*model.numpy_filter(lam))@v.T
            prediction, timing = measure(infer)
            actual_relative = np.linalg.norm(prediction-reference,axis=1)/np.linalg.norm(reference,axis=1)
            np.testing.assert_allclose(prediction,old_prediction,atol=1e-12,rtol=1e-11)
            np.testing.assert_allclose(actual_relative,predicted_relative,atol=1e-12,rtol=1e-10)
            study.record(0,"C006_frozen",f"{label}_seed{seed}",prediction,reference,timing,
                {"seed":seed,"audit_only":True,"trained_this_session":False,
                 "mean_sample_relative_l2":float(actual_relative.mean()),
                 "modal_prediction_discrepancy_max":float(np.max(abs(actual_relative-predicted_relative))),
                 "finite_g_at_zero":float(masses.sum()),
                 "smallest_eigenvalue":float(lam.min())},
                {"source":q,"conductivity":k,"eigenvalues":lam,"inverse_response_ratio":lam*multiplier,
                 "modal_error_sq":modal_error_sq,"modal_exact_solution_sq":exact_coefficients**2})
            output.append({"seed":seed,"condition":label,"mean_relative_l2":float(actual_relative.mean()),
                           "modal_prediction_max_difference":float(np.max(abs(actual_relative-predicted_relative)))})
    dump(study.root/"c006_modal_audit.json",output)


def scale_audit(study):
    entries = []
    for kind, shape in [("uniform",lambda x:np.ones_like(x)),
                        ("fixed_heterogeneous_shape",lambda x:1+4*(x>.5))]:
        for boundary in [(0.,0.),(.5,1.5)]:
            for scale in [.02,3.]:
                p = heat_1d(15,conductivity=shape,boundary=boundary)
                changed = heat_1d(15,conductivity=lambda x:scale*shape(x),boundary=boundary)
                lift = heat_1d(15,conductivity=shape,source=0.,boundary=boundary).reference()
                correct,timing = measure(lambda:lift+(p.reference()-lift)/scale)
                study.record(0,"scaling_control",f"{kind}_g{boundary[0]}_s{scale}",correct,changed.reference(),timing,
                    {"equation":"-(k T_x)_x=1", "boundary":list(boundary),"scale":scale,
                     "naive_full_temperature_scaling_relative_error":error_metrics(p.reference()/scale,changed.reference())["relative_l2"]},
                    {"x":p.x,"original_solution":p.reference(),"harmonic_lift":lift,"new_conductivity":changed.edges_k})
                entries.append(study.rows[-1])
    source = lambda x:np.exp(-((x-.27)/.06)**2)
    left = heat_1d(conductivity=lambda x:np.where(x<.5,1.,8.),source=source)
    right = heat_1d(conductivity=lambda x:np.where(x<.5,8.,1.),source=source)
    prediction,timing = measure(left.reference)
    study.record(0,"effective_material_counterexample","same_harmonic_mean",prediction,right.reference(),timing,
        {"equation":"-(k T_x)_x=Q", "claim_tested":"one scalar harmonic material value is sufficient", "boundary":[0.,0.]},
        {"x":left.x,"source":left.source,"left_conductivity":left.edges_k,"right_conductivity":right.edges_k})


def initial_candidates(study):
    # Cycle 13: finite packets vs tangent/frozen approximations.
    p0 = heat_1d(15)
    base = solve(p0.matrix,np.eye(15),assume_a="pos")
    conditions = [
        ("uniform_low",heat_1d(conductivity=.02)),
        ("interface",heat_1d(conductivity=lambda x:np.where(x<.5,.1,3.))),
        ("source_left",heat_1d(source=lambda x:10*np.exp(-((x-.2)/.05)**2))),
        ("source_right",heat_1d(conductivity=lambda x:1+4*x,source=lambda x:10*np.exp(-((x-.8)/.05)**2))),
        ("nonzero_boundary",heat_1d(conductivity=lambda x:1+4*x,source=0.,boundary=(.2,1.))) ]
    for label,p in conditions:
        for mode in ["finite","frozen"]:
            study.heat(13,"A_"+mode,label,p,lambda p=p,m=mode:material_response(base,np.ones(16),p.edges_k,p.rhs,p.h,m),
                {"base_inverse_prior":True,"trained_this_session":False,"equivalence":"Sherman-Morrison if packets are updated"})
    # Small positive perturbation makes the tangent counterexample finite and easy to inspect.
    k=np.ones(16);k[6]=3.
    p=heat_1d(conductivity=k)
    for mode in ["finite","tangent"]:
        study.heat(13,"A_"+mode,"single_finite_perturbation",p,
            lambda m=mode:material_response(base,np.ones(16),p.edges_k,p.rhs,p.h,m))

    # Cycle 14: witness sufficiency before decoder training.
    p=heat_1d()
    probes=balance_probes(p.x)
    invisible=witness_collision(probes)
    np.testing.assert_allclose(probes@invisible,0,atol=1e-14)
    for label,q in [("signed_plus",invisible),("signed_minus",-invisible),
                    ("positive_plus",1+.8*invisible),("positive_minus",1-.8*invisible)]:
        problem=heat_1d(source=q)
        study.heat(14,"B_witness",label,problem,
            lambda p=problem:witness_decode(p.matrix,p.rhs,probes),
            {"witness_rank":4,"source_dimension":15,"witness":(probes@q).tolist(),
             "squared_source_norm":float(q@q),"decoder":"classical minimum-norm weak constraint closure; not trained"})
    same_source=p.source
    for rank in [4,15]:
        r=probes if rank==4 else np.eye(15)
        study.heat(14,"B_full_control" if rank==15 else "B_witness",f"rank{rank}",p,
                   lambda r=r:witness_decode(p.matrix,same_source,r),
                   {"rank":rank,"equivalence":"full rank closure is a linear solve"})

    # Cycle 15: symmetry/finite-change composition independent of A/B.
    for n in [7,15,31]:
        for label,problem in [("uniform",heat_1d(n)),
                              ("local",heat_1d(n,source=lambda x:10*np.exp(-((x-.3)/.07)**2))),
                              ("low_material",heat_1d(n,conductivity=.05)),
                              ("signed_source",heat_1d(n,source=lambda x:4*np.sin(2*np.pi*x)))]:
            g=solve(problem.matrix,np.eye(n),assume_a="pos")
            contributions=g*problem.rhs[None,:]
            for conjugate in [False,True]:
                study.heat(15,"C_conjugate" if conjugate else "C_raw",f"{label}_n{n}",problem,
                    lambda c=contributions,j=conjugate:compose_responses(c,gamma=4.,conjugate=j),
                    {"gamma":4.,"inverse_prior":True,"trained_this_session":False,
                     "equivalence":"asinh-coordinate conjugacy to addition"})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",default="results/session_0003")
    parser.add_argument("--phase",choices=["initial"],default="initial")
    args=parser.parse_args()
    torch.set_num_threads(1)
    study=Study(args.output)
    c006_audit(study)
    scale_audit(study)
    initial_candidates(study)
    study.checkpoint("initial_completed")
    print(json.dumps({"status":"initial_completed","records":len(study.rows),
                      "output":str(study.root)},indent=2))


if __name__=="__main__":
    main()
