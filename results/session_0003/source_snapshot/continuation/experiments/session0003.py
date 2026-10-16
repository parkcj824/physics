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
    segment_compose, segment_features, segment_response, periodic_transport,
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


def segment_cycle(study):
    cases = [
        ("uniform",heat_1d()),
        ("heterogeneous",heat_1d(conductivity=lambda x:np.where(x<.5,.3,4.))),
        ("low_material",heat_1d(conductivity=.001)),
        ("source_left",heat_1d(source=lambda x:12*np.exp(-((x-.2)/.05)**2))),
        ("source_right",heat_1d(conductivity=lambda x:1+2*x,source=lambda x:12*np.exp(-((x-.8)/.05)**2))),
        ("boundary_change",heat_1d(conductivity=lambda x:1+3*x,source=0.,boundary=(-.5,1.))),
        ("resolution7",heat_1d(7,source=lambda x:np.sin(np.pi*x))),
        ("resolution31",heat_1d(31,conductivity=lambda x:1+3*x,source=lambda x:np.sin(np.pi*x))),
    ]
    for label,p in cases:
        for theta in [1.,0.]:
            study.heat(16,"D_transfer" if theta==1 else "D_missing_area",label,p,
                lambda p=p,t=theta:segment_response(p.edges_k,p.source,p.boundary,t),
                {"theta":theta,"trained_this_session":False,
                 "equivalence":"classical ordered affine transfer/prefix scan; not a new neural operation"})
    segments=[np.array([.7,.3,.2]),np.array([1.2,-.4,.1]),np.array([.4,.6,-.1])]
    a,b,c=segments
    lhs=segment_compose(segment_compose(a,b),c)
    rhs=segment_compose(a,segment_compose(b,c))
    dump(study.root/"segment_algebra.json",{
        "associativity_max_difference":float(np.max(abs(lhs-rhs))),
        "order_commutator":(segment_compose(a,b)-segment_compose(b,a)).tolist(),
        "interpretation":"physical spatial subdomain ordering, not fictitious history of material edits",
        "matrix_equivalence":"[[1,-R,-C],[0,1,S],[0,0,1]] on [T,j,1]; compose second matrix times first"})


def transport_cycle(study):
    n=24;dt=.02;steps=10
    x=np.arange(n)/n
    initial=1+.3*np.cos(2*np.pi*x)
    for peclet in [.2,2.,20.]:
        diffusivity=(1/n)/peclet
        base_operator=periodic_transport(n,0.,diffusivity)
        base_step=np.eye(n)+dt*base_operator
        inverse_base=solve(base_step,np.eye(n))
        for velocity in [-1.,1.]:
            operator=periodic_transport(n,velocity,diffusivity)
            step_matrix=np.eye(n)+dt*operator
            reference=initial.copy()
            for _ in range(steps):
                reference=solve(step_matrix,reference)
            continuous_discrete=expm(-steps*dt*operator)@initial
            for reciprocal in [False,True]:
                def infer():
                    response=oriented_inverse_update(inverse_base,step_matrix-base_step,reciprocal)
                    state=initial.copy()
                    for _ in range(steps):
                        state=response@state
                    return state
                pred,timing=measure(infer)
                study.record(17,"A_reciprocal_shortcut" if reciprocal else "A_oriented",f"Pe{peclet}_v{velocity}",
                    pred,reference,timing,
                    {"equation":"u_t+v*u_x=k*u_xx; periodic; upwind space and backward Euler time",
                     "dt":dt,"steps":steps,"velocity":velocity,"cell_peclet":peclet,
                     "minimum_state":float(pred.min()),"mass_drift_abs":float(abs(pred.sum()-initial.sum())/n),
                     "boundary_error_abs":0.,
                     "BE_reference_vs_exact_semidiscrete_relative_l2":error_metrics(reference,continuous_discrete)["relative_l2"],
                     "equivalence":"general left/right Sherman-Morrison; base diffusion inverse is supplied", "trained_this_session":False},
                    {"x":x,"initial":initial,"operator":operator,"semidiscrete_exact":continuous_discrete})
    # Same v,k and the same global mass/energy witness, but a different initial phase.
    operator=periodic_transport(n,1.,.01)
    propagator=expm(-.2*operator)
    initial_a=1+.3*np.cos(2*np.pi*x)
    initial_b=np.roll(initial_a,n//4)
    state_a,state_b=propagator@initial_a,propagator@initial_b
    witness=lambda u:np.array([u.mean(),np.mean(u*u)])
    witness_gap=np.max(abs(witness(initial_a)-witness(initial_b)))
    assert witness_gap<1e-14
    pred,timing=measure(lambda:state_a.copy())
    study.record(17,"B_phase_collision","mass_energy_equal",pred,state_b,timing,
        {"equation":"u_t+u_x=.01*u_xx; periodic; same velocity/material, different initial phase",
         "initial_witness_gap_max":float(witness_gap),
         "final_mass_energy_witness_gap_max":float(np.max(abs(witness(state_a)-witness(state_b)))),
         "minimum_state":float(pred.min()),"boundary_error_abs":0.,
         "claim_tested":"global mass and energy witnesses determine the state; false",
         "decoder":"one canonical phase, not trained"},
        {"x":x,"initial_a":initial_a,"initial_b":initial_b,"initial_witness":witness(initial_a)})
    # Heat superposition failure persists for conservative transient response.
    component_a=.5+.3*np.cos(2*np.pi*x)
    component_b=np.full(n,.5)
    contributions=np.column_stack((propagator@component_a,propagator@component_b))
    for conjugate in [False,True]:
        pred,timing=measure(lambda:compose_responses(contributions,gamma=4.,conjugate=conjugate))
        reference=propagator@(component_a+component_b)
        study.record(17,"C_conjugate" if conjugate else "C_raw","transient_superposition",pred,reference,timing,
            {"equation":"u_t+u_x=.01*u_xx; periodic", "gamma":4.,
             "mass_drift_abs":float(abs(pred.mean()-(component_a+component_b).mean())),
             "minimum_state":float(pred.min()),"boundary_error_abs":0.,"propagator_prior":True},
            {"x":x,"initial":component_a+component_b})
    dump(study.root/"unrun_extensions.json",{
        "D_transient":"not defined by this steady spatial relation; not silently used as a time stepper",
        "stokes":"not run: no new primitive survived novelty/independent-operation screening",
        "large_GPU":"not authorized and not run"})


def heat_dataset(samples,seed,condition="train",n=15):
    rng=np.random.default_rng(seed)
    x=np.arange(1,n+1)/(n+1);edges=(np.arange(n+1)+.5)/(n+1)
    q=rng.normal(size=(samples,4))@np.sin(np.pi*x[:,None]*np.arange(1,5)[None,:]).T
    k=np.empty((samples,n+1));boundary=rng.uniform(-.5,.5,(samples,2))
    for i in range(samples):
        center=rng.uniform(.2,.55) if condition!="new_source" else rng.uniform(.75,.9)
        q[i]+=rng.uniform(-3,3)*np.exp(-((x-center)/.06)**2)
        scale=np.exp(rng.uniform(np.log(.5),np.log(2.)))
        if condition=="low_k":scale=np.exp(rng.uniform(np.log(.01),np.log(.05)))
        a,b,c=rng.uniform(-.4,.4,3)
        shape=np.exp(a*np.cos(2*np.pi*edges)+b*np.sin(np.pi*edges)+c*(edges>.5))
        if condition=="new_material":shape=np.where(edges<.3,1.,10.)
        k[i]=scale*shape
    if condition=="new_boundary":boundary=rng.uniform(-2,2,(samples,2))
    y=np.array([heat_1d(n,conductivity=ki,source=qi,boundary=tuple(gi)).reference()
                for ki,qi,gi in zip(k,q,boundary)])
    return {"k":k,"q":q,"boundary":boundary,"target":y}


def batch_heat_diagnostics(data,prediction):
    metrics=[heat_1d(len(q),conductivity=k,source=q,boundary=tuple(g)).diagnostics(u)
             for k,q,g,u in zip(data["k"],data["q"],data["boundary"],prediction)]
    relative=np.linalg.norm(prediction-data["target"],axis=1)/np.maximum(np.linalg.norm(data["target"],axis=1),1e-14)
    return {"mean_sample_relative_l2":float(relative.mean()),"std_sample_relative_l2":float(relative.std(ddof=1)),
            "pde_residual_inf":max(m["pde_residual_inf"] for m in metrics),
            "net_energy_balance_abs":max(m["net_energy_balance_abs"] for m in metrics),
            "boundary_error_abs":0.,"boundary_handling":"explicit prescribed Dirichlet lift/constraints"}


def fair_comparisons(study):
    train=heat_dataset(128,3003)
    validation=heat_dataset(32,3004)
    tests={label:heat_dataset(32,seed,label) for label,seed in
           [("iid",3010),("low_k",3011),("new_source",3012),("new_material",3013),("new_boundary",3014)]}
    np.savez_compressed(study.root/"training_data.npz",**train)
    np.savez_compressed(study.root/"validation_data.npz",**validation)
    def features(data):
        return np.c_[np.log(data["k"]),data["q"],data["boundary"]]
    raw=features(train);center=raw.mean(0);scale=np.maximum(raw.std(0),1e-8)
    design=np.c_[(raw-center)/scale,np.ones(len(raw))]
    fit_start=time.perf_counter()
    feature_linear=np.linalg.lstsq(design,train["target"],rcond=1e-12)[0]
    linear_train_seconds=time.perf_counter()-fit_start
    # Fit D's cross moment on exactly the same 128 labels, with relative squared loss.
    start=time.perf_counter()
    constants,sensitivities=zip(*(segment_features(k,q,tuple(g)) for k,q,g in zip(train["k"],train["q"],train["boundary"])))
    constant=np.array(constants);sensitivity=np.array(sensitivities)
    normalizer=np.maximum(np.sum(train["target"]**2,axis=1),1e-14)
    theta=float(np.sum(sensitivity*(train["target"]-constant)/normalizer[:,None])/
                np.sum(sensitivity*sensitivity/normalizer[:,None]))
    theta_train_seconds=time.perf_counter()-start
    dump(study.root/"models/segment_theta.json",{"theta":theta,"training_samples":128,
        "training_method":"one-parameter weighted least squares, not SGD","training_seconds":theta_train_seconds,
        "interpretation":"recovering the known unit transfer coefficient is not a neural invention"})
    np.savez(study.root/"models/feature_linear.npz",coefficient=feature_linear,center=center,scale=scale)
    # Strong fixed-material response control: same source vectors/BCs, unit-k labels only.
    # These are an EXTRA explicitly counted 128 labels, so no equal-data claim is made for it.
    unit_labels=np.array([heat_1d(source=q).reference() for q in train["q"]])
    start=time.perf_counter()
    compliance=np.linalg.lstsq(train["q"],unit_labels,rcond=1e-12)[0]
    compliance_seconds=time.perf_counter()-start
    np.savez(study.root/"models/unit_compliance.npz",coefficient=compliance,source=train["q"],target=unit_labels)
    frozen=LifetimeResponse(seed=824)
    frozen.load_state_dict(torch.load(PRIOR/"models/C006_seed824.pt",weights_only=True))
    training_records=[]

    def banded_case(k,q,g):
        n=len(q);h=1/(n+1)
        bands=np.zeros((3,n));bands[1]=(k[:-1]+k[1:])/h**2
        bands[0,1:]=-k[1:-1]/h**2;bands[2,:-1]=-k[1:-1]/h**2
        rhs=q.copy();rhs[0]+=k[0]/h**2*g[0];rhs[-1]+=k[-1]/h**2*g[1]
        return solve_banded((1,1),bands,rhs)

    def record_test(name,label,data,function,metadata):
        prediction,timing=measure(function)
        study.record(18,name,label,prediction,data["target"],timing,
            {"equation":"-(k T_x)_x=Q; variable k(x), local sources and nonzero Dirichlet",
             "sample_count":32,**batch_heat_diagnostics(data,prediction),**metadata},
            {"conductivity":data["k"],"source":data["q"],"boundary":data["boundary"]})

    for label,data in tests.items():
        def segment_predict():
            return np.array([segment_response(k,q,tuple(g),theta) for k,q,g in zip(data["k"],data["q"],data["boundary"])])
        record_test("D_fitted_transfer",label,data,segment_predict,
            {"theta":theta,"training_samples":128,"parameters":1,"trained_this_session":True,
             "training_seconds":theta_train_seconds,"prior":"exact transfer algebra and material/source features; coefficient-only fit"})
        record_test("feature_linear",label,data,
            lambda d=data:np.c_[(features(d)-center)/scale,np.ones(len(d["q"]))]@feature_linear,
            {"training_samples":128,"parameters":int(feature_linear.size),"training_seconds":linear_train_seconds,"trained_this_session":True})
        def uniform_compliance_predict():
            harmonic=1/np.mean(1/data["k"],axis=1)
            x=np.arange(1,16)/16
            return (data["q"]/harmonic[:,None])@compliance+data["boundary"][:,0,None]*(1-x)+data["boundary"][:,1,None]*x
        record_test("unit_compliance",label,data,uniform_compliance_predict,
            {"training_samples":128,"training_label_distribution":"EXTRA unit-material zero-BC labels; not matched data",
             "parameters":225,"training_seconds":compliance_seconds,"trained_this_session":True,
             "prior":"known 1/k rule, valid for scalar uniform material only; deliberately tested outside that condition"})
        def frozen_predict():
            output=[]
            for k,q,g in zip(data["k"],data["q"],data["boundary"]):
                p=heat_1d(conductivity=k,source=q,boundary=tuple(g))
                eig,v=np.linalg.eigh(p.matrix)
                output.append(v@(frozen.numpy_filter(eig)*(v.T@p.rhs)))
            return np.array(output)
        record_test("C006_frozen",label,data,frozen_predict,
            {"trained_this_session":False,"parameters":12,"prior_training":"Session0002 uniform-material sources; not matched new training data",
             "latency_scope":"includes matrix assembly and eigendecomposition; no large scale claim"})
        record_test("banded_classical",label,data,
            lambda d=data:np.array([banded_case(k,q,g) for k,q,g in zip(d["k"],d["q"],d["boundary"])]),
            {"trained_this_session":False,"training_samples":0,"prior":"governing equation; includes band assembly; O(N) per source"})

    labels=torch.tensor(train["target"],dtype=torch.float64)
    norm=labels.square().sum(1).clamp_min(1e-14)
    inputs=torch.tensor((raw-center)/scale,dtype=torch.float64)
    output_scale=float(np.sqrt(np.mean(train["target"]**2)))
    for seed in [903,904,905]:
        torch.manual_seed(seed)
        model=nn.Sequential(nn.Linear(33,64),nn.Tanh(),nn.Linear(64,15)).double()
        optimizer=torch.optim.Adam(model.parameters(),lr=.003)
        start=time.perf_counter()
        for _ in range(1000):
            optimizer.zero_grad(set_to_none=True)
            prediction=output_scale*model(inputs)
            loss=((prediction-labels).square().sum(1)/norm).mean()
            loss.backward();optimizer.step()
        train_seconds=time.perf_counter()-start
        torch.save({"state":model.state_dict(),"center":torch.tensor(center),"scale":torch.tensor(scale),"output_scale":output_scale},
                   study.root/f"models/MLP_seed{seed}.pt")
        with torch.no_grad():
            vp=(output_scale*model(torch.tensor((features(validation)-center)/scale,dtype=torch.float64))).numpy()
        validation_error=float(np.mean(np.linalg.norm(vp-validation["target"],axis=1)/np.linalg.norm(validation["target"],axis=1)))
        for label,data in tests.items():
            def predict(d=data):
                with torch.no_grad():
                    return (output_scale*model(torch.tensor((features(d)-center)/scale,dtype=torch.float64))).numpy()
            record_test("MLP",label+f"_seed{seed}",data,predict,
                {"seed":seed,"training_samples":128,"parameters":sum(p.numel() for p in model.parameters()),
                 "trained_this_session":True,"training_steps":1000,"training_seconds":train_seconds,
                 "validation_mean_relative_l2":validation_error,"checkpoint_selection":"fixed 1000 steps; no test selection",
                 "fairness":"same 128 labels as D/feature-linear; different physical priors, capacity and optimizer cost"})
        training_records.append({"seed":seed,"seconds":train_seconds,"last_training_relative_mse":loss.item(),
                                 "validation_mean_relative_l2":validation_error})
    dump(study.root/"baseline_training.json",training_records)
    # Material-coordinate normalization alone is not claimed as an invented primitive.
    # It is only a recorded baseline limitation and a future verification issue.


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",default="results/session_0003")
    parser.add_argument("--phase",choices=["initial","continue"],default="initial")
    args=parser.parse_args()
    torch.set_num_threads(1)
    study=Study(args.output,resume=args.phase=="continue")
    if args.phase=="initial":
        c006_audit(study)
        scale_audit(study)
        initial_candidates(study)
        status="initial_completed"
    else:
        segment_cycle(study)
        transport_cycle(study)
        fair_comparisons(study)
        status="completed"
    study.checkpoint(status)
    print(json.dumps({"status":status,"records":len(study.rows),
                      "output":str(study.root)},indent=2))


if __name__=="__main__":
    main()
