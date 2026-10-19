"""Close local Session0005 provenance after tests; no Git writes or external transfer."""
import datetime
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.session0005_common import ROOT,sha,preserve,write,now


def main():
    if (ROOT/'validation/assets_sha256.json').exists():raise FileExistsError('Session0005 asset manifest already finalized')
    assert not subprocess.check_output(['git','diff','--name-only']).strip(),'tracked files changed'
    prior=preserve();env=json.loads((ROOT/'environment.json').read_text())
    xml=ET.parse(ROOT/'validation/pytest.xml').getroot();suites=list(xml) if xml.tag=='testsuites' else [xml]
    count=sum(int(s.attrib['tests']) for s in suites)
    assert count==54 and not any(int(s.attrib.get('failures',0))+int(s.attrib.get('errors',0)) for s in suites)
    checkpoint_infos=[]
    for name in ['controls_frozen','normalized_frozen','interaction_frozen']:
        checkpoint_infos+=json.loads((ROOT/(name+'.json')).read_text())['models']
    checkpoint_infos+=[json.loads((ROOT/'interaction_encoder_frozen.json').read_text())['model']]
    end=now();elapsed=(datetime.datetime.fromisoformat(end)-datetime.datetime.fromisoformat(env['start_utc'])).total_seconds()
    run={'session':'0005','research_cycles':[24,25,26,27],'independent_starting_hypotheses':['J compatibility','L finite reaction selection','K causal observability'],
        'controls_not_invention_cycles':['C0','C0b'],'start_utc':env['start_utc'],'closed_utc':end,'wall_seconds_to_close':elapsed,
        'session_budget_seconds':5400,'within_budget':elapsed<=5400,'head_at_start':env['head'],
        'head_at_close':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'prior_tracked_files_preserved':prior,'pytest_passed':count,'checkpoint_files':len(checkpoint_infos),
        'optimizer_updates_executed':sum(i['steps'] for i in checkpoint_infos),
        'sum_measured_fit_loop_seconds':sum(i['seconds'] for i in checkpoint_infos),
        'prediction_batches':333,'classical_prediction_batches':24,'dataset_files':25,
        'exact_correction_equation_checks':4,'new_architecture_proven':False,'level3_primitive_proven':False,
        'best_architecture':None,'retained_level2_probe':'L/F5, limited positive-source finite response representation',
        'next_research_object':'collective state S plus finite nonlinear defect H; no new learned rule yet',
        'stop_reason':'Four meaningful cycles established causal controls and information-loss counterexamples. Next credible step requires a distinct mathematical learning mechanism and closest-prior-art review, not wider models or larger CFD. No filler cycles.',
        'unrun':['learned state-conditioned new primitive','learned2D/resolution/geometry tests','incompressible momentum/Stokes','RANS/HVAC','GPU training','new full-paper/patent search','full70-model retraining wrapper replay'],
        'external_research_data_transfer':False,'github_push':False,'paid_resources':False,'gpu_training':False,
        'full_memory_peak_measured':False,'memory_scope':'Python tracemalloc only; native tensor/BLAS allocations partly omitted',
        'next_session':'0006','next_cycle':28,
        'verification_mutable_exclusions':['results/session_0005/validation/integrity.json','results/session_0005/validation/assets_sha256.json']}
    assert run['within_budget'];assert run['head_at_close']==env['head']
    write('run.json',run)
    files=subprocess.check_output(['git','ls-files','--others','--exclude-standard','-z']).decode().split('\0')
    files=sorted(p for p in files if p and Path(p).is_file() and p not in run['verification_mutable_exclusions'])
    code={p:sha(p) for p in files if Path(p).suffix in ['.py','.md'] and not p.startswith('results/')}
    write('validation/source_hashes.json',code)
    path=str(ROOT/'validation/source_hashes.json')
    if path not in files:files.append(path)
    run['session0005_asset_files']=len(files);write('run.json',run)
    write('validation/assets_sha256.json',{p:sha(p) for p in sorted(files)})
    print(json.dumps({'closed_utc':end,'elapsed_minutes':elapsed/60,'source_and_document_files':len(code),
        'new_assets':len(files),'prior_files_preserved':prior,'pytest_passed':count,'checkpoints':len(checkpoint_infos)},indent=2))


if __name__=='__main__':main()
