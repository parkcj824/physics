"""Replay Session 0005 into a new local directory, preserving all original assets."""
import argparse
import importlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
os.environ.setdefault('XDG_CACHE_HOME','/workspace/.cache')
os.environ.setdefault('MPLCONFIGDIR','/workspace/.cache/matplotlib')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',help='New directory under /workspace or /tmp; must not already exist')
    args=parser.parse_args();project=Path(__file__).resolve().parents[1];os.chdir(project)
    output=Path(args.output).resolve()
    if output.exists():raise FileExistsError(output)
    if not any(output.is_relative_to(Path(p)) for p in ['/workspace','/tmp']):raise ValueError('local writable roots only')
    if output.is_relative_to(project/'results') and not output.name.startswith('session_0005_replay'):
        raise ValueError('Use results/session_0005_replay_* or a separate /tmp directory')
    import numpy as np
    import scipy
    import torch
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    from experiments import session0005_common as common
    canonical=project/'results/session_0005';output.mkdir(parents=True)
    common.ROOT=output
    common.write('prior_file_sha256.json',json.loads((canonical/'prior_file_sha256.json').read_text()))
    common.write('replay_provenance.json',{'start_utc':common.now(),'python':platform.python_version(),
        'numpy':np.__version__,'scipy':scipy.__version__,'torch':torch.__version__,'output':str(output),
        'original_environment_sha256':common.sha(canonical/'environment.json'),
        'scope':'same fixed seeds/protocol; reproducibility replay, not a new independent confirmation',
        'runtime_override':'ROOT set in all new Session0005 modules; old Session modules only provide pure functions'})
    names=['controls','interaction','hypotheses','closure','normalized','audit','costs','closure_audit']
    modules={n:importlib.import_module('experiments.session0005_'+n) for n in names}
    for m in modules.values():m.ROOT=output
    phases=[('C0 prepare',modules['controls'].prepare),('C0 train',modules['controls'].train),
        ('C0 confirm',modules['controls'].confirm),('J',modules['hypotheses'].compatibility),
        ('F prepare',modules['interaction'].prepare),('F low-k extension',modules['interaction'].extra_lowk),
        ('F encoder',modules['interaction'].encode_train),('F train',modules['interaction'].train),
        ('F confirm',modules['interaction'].confirm),('K',modules['hypotheses'].causal),
        ('closure collision',modules['closure'].main),('exact correction equation audit',modules['closure_audit'].main),('C0b',modules['normalized'].main),('posthoc diagnostics',modules['audit'].main),('supplementary cost reproduction',modules['costs'].main)]
    for label,fn in phases:
        print(common.now(),label,flush=True);fn()
    subprocess.run([sys.executable,'-m','pytest','-q','--junitxml='+str(output/'validation/pytest.xml')],check=True)
    from scripts import verify_session0005 as verifier
    verifier.ROOT=output;verifier.main()
    common.write('replay_complete.json',{'completed_utc':common.now(),'prior_files_preserved':common.preserve()})


if __name__=='__main__':main()
