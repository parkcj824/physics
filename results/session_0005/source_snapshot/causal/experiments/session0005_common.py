"""Local provenance and CPU measurements shared by Session 0005 only."""
import datetime
import hashlib
import json
from pathlib import Path
import time
import tracemalloc
import numpy as np
import torch

ROOT=Path('results/session_0005')
SEEDS=[2005,2006,2007]


def tensor(x):return torch.as_tensor(x,dtype=torch.float64)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(name,value):
    path=ROOT/name;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
def preserve():
    old=json.loads((ROOT/'prior_file_sha256.json').read_text())
    changed=[p for p,h in old.items() if sha(p)!=h]
    assert not changed,changed
    return len(old)
def snapshot(name,files):
    files=list(dict.fromkeys(['experiments/session0005_common.py','research/session_0005/preregistered_protocol.md']+files))
    for p in files:
        target=ROOT/'source_snapshot'/name/p;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(p).read_bytes())
    write(f'source_snapshot/{name}/sha256.json',{p:sha(p) for p in files})
def save_data(name,d):
    target=ROOT/'data'/(name+'.npz');target.parent.mkdir(exist_ok=True,parents=True)
    if target.exists():raise FileExistsError(target)
    np.savez_compressed(target,**d);return sha(target)
def load_data(name):
    with np.load(ROOT/'data'/(name+'.npz')) as d:return {k:d[k] for k in d.files}
def measure(fn,repeats=7):
    fn();times=[]
    for _ in range(repeats):
        begin=time.perf_counter();fn();times.append(time.perf_counter()-begin)
    tracemalloc.start();value=fn();_,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
    return value,{'median_seconds':float(np.median(times)),'repeats':repeats,'python_tracemalloc_peak_bytes':peak,
                  'memory_scope':'native/BLAS/PyTorch allocations partly omitted; not full peak memory'}
def fit(model,loss,steps,label,seed):
    model.zero_grad(set_to_none=True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],with_flops=True) as prof:
        loss(model,False).backward()
    flops=sum(event.flops for event in prof.key_averages())
    active=sum(p.numel() for p in model.parameters() if p.grad is not None)
    optimizer=torch.optim.Adam(model.parameters(),lr=.003);best=float('inf');best_state=None;trace=[]
    begin=time.perf_counter()
    for step in range(steps+1):
        if step%100==0:
            with torch.no_grad():v=float(loss(model,True));tr=float(loss(model,False))
            trace.append({'step':step,'train_loss':tr,'validation_loss':v})
            if v<best:
                best=v;best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
        if step==steps:break
        optimizer.zero_grad(set_to_none=True);value=loss(model,False)
        if not torch.isfinite(value):raise FloatingPointError(label)
        value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.);optimizer.step()
    elapsed=time.perf_counter()-begin;model.load_state_dict(best_state)
    target=ROOT/'models'/(label+'.pt');target.parent.mkdir(exist_ok=True,parents=True)
    if target.exists():raise FileExistsError(target)
    torch.save({'state':best_state,'seed':seed},target)
    return {'checkpoint':str(target.relative_to(ROOT)),'sha256':sha(target),'seed':seed,'steps':steps,
        'parameter_count':sum(p.numel() for p in model.parameters()),'active_parameter_count':active,
        'profiled_flops_one_train_batch':flops,'profiled_training_flops_estimate':flops*steps,
        'flops_scope':'PyTorch profiler supported dense/operator subset; not complete arithmetic count',
        'seconds':elapsed,'best_validation_loss':best,'trace':trace}
