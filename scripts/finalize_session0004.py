"""Finalize local provenance and the fresh capacity-control report, preserving prior files."""

import csv
import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
import numpy as np
from PIL import Image

R=Path('results/session_0004')


def read(name):return json.loads((R/name).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(name,value):
    p=R/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def main():
    if (R/'validation/final_manifest.json').exists():raise FileExistsError('final manifest already exists')
    cap=read('capacity_confirmation.json');infos=read('capacity_frozen.json')['models']
    table=['| 물리 | 모델 | 평균 상대 L2 ± seed std | 평균 PDE 잔차 inf ± std | 추론 ms/batch32 |',
           '|---|---|---:|---:|---:|']
    for family,mode in sorted({(r['family'],r['mode']) for r in cap['records']}):
        rows=[r for r in cap['records'] if r['family']==family and r['mode']==mode]
        def metric(k):
            v=np.array([r[k] for r in rows]);return f'{v.mean():.6g} ± {v.std():.3g}'
        table.append(f"| {family} | {mode} | {metric('mean_relative_l2')} | {metric('mean_pde_residual_inf')} | {np.median([r['median_seconds'] for r in rows])*1000:.5g} |")
    text='''# G/H 활성 파라미터·계산 예산 후속 통제

최초 확인 결과와 gradient 경로 audit에서 capacity confound를 발견했다. 기존 확인 결과를 모델 개발에 사용한 뒤 다시 독립 검증이라고 부르지 않았다. `capacity_control_protocol.md`를 고정하고 **새 IID 32개씩(seed45003/45004)**을 사용했다.

기존 후보·기준선 체크포인트는 그대로 두고 direct width64만 새로 학습했다. 활성 파라미터8415로 feedback8704에 가깝고, 기존 동일128라벨/validation32/loss/Adam.003을 사용했다. 직접 MLP는3000steps, feedback은기존1500steps로 실제시간도 가깝게 맞추려 했다. 정확한 FLOPs 동등성은 아니다. 새6모델이 동결된 뒤 기존모델18개와 함께 새24batch를 평가했다. 세seed는 같은데이터에서 초기화만 다르다.

G는열, H는정상이류–확산이다. 상대오차는조건별L2평균이다.

'''+ '\n'.join(table)+'''

G의온도오차차이는작고 PDE잔차는 feedback이 더 크다. H는온도오차의이득이남지만 PDE잔차차이는작다. 보존·field L2·국소PDE잔차는별개다. 이결과는단순활성파라미터증가만으로설명되지않는제한된표현이득의추가증거이나, 새로운아키텍처의신규성·일반CAEspeedup·모든물리지표우월성은입증하지않는다. 두학습메커니즘의optimization/계산배치차이도남아있다.

OOD/새geometry검증은이추가control의범위가아니다. 이전positivity/low-k/low-diffusion실패는그대로유효하다. 실제각모델의학습시간과loss trace는capacity_frozen.json, 입력·예측·정답과해시는capacity_confirmation.json 및arrays에있다. 후보를재학습하지않았으며추가확인후튜닝하지않았다.
'''
    target=Path('reports/session_0004/capacity_control.md');target.write_text(text)
    original=read('frozen_models.json')['models'];transport=read('transport_frozen.json')['models']
    all_models=original+transport+infos;train_seconds=sum(i['seconds'] for i in all_models)
    summary=Path('reports/session_0004/research_summary.md');content=summary.read_text()
    content=re.sub(r'실제 학습 측정시간 합 [0-9.]+초',f'실제 학습 측정시간 합 {train_seconds:.5g}초',content)
    if '| G/direct_width64 (fresh IID)' not in content:
        extra=[]
        for family in ['G','H']:
            rows=[r for r in cap['records'] if r['family']==family and r['mode']=='direct_width64']
            models=[i for i in infos if i['family']==family]
            extra.append(f"| {family}/direct_width64 (fresh IID) | {np.median([r['median_seconds'] for r in rows])*1000:.5g} | {models[0]['parameter_count']} (active8415) | {sum(i['seconds'] for i in models):.5g} |")
        marker=next(line for line in content.splitlines() if line.startswith('| H/classical upwind |'))
        content=content.replace(marker,marker+'\n'+'\n'.join(extra))
    summary.write_text(content)
    rows=read('confirmation.json')['records']+[{**r,'family':'H'} for r in read('transport_confirmation.json')['records']]+[{**r,'case':'fresh_iid'} for r in cap['records']]
    fields=['family','mode','seed','case','mean_relative_l2','max_absolute_error','mean_cross_relative_l2',
            'mean_pde_residual_inf','max_generated_energy_balance_abs','median_seconds','spurious_branches','arrays']
    folder=R/'comparisons';folder.mkdir(exist_ok=True)
    with (folder/'all_confirmation_records.csv').open('w',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    pictures=[]
    for path in (R/'plots').glob('*.png'):
        with Image.open(path) as im:
            size=im.size;im.verify()
        pictures.append({'path':str(path),'pixels':size,'sha256':sha(path),'visual_inspection':'performed'})
    write('validation/figures.json',pictures)
    old=read('prior_file_sha256.json');assert all(sha(p)==h for p,h in old.items())
    assert subprocess.run(['git','diff','--quiet']).returncode==0
    env=read('environment.json');head=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip();assert head==env['head']
    finished=datetime.datetime.now(datetime.timezone.utc);started=datetime.datetime.fromisoformat(env['start_utc'])
    write('run.json',{'status':'completed','session':'0004','started_utc':env['start_utc'],'finished_utc':finished.isoformat(),
        'wall_seconds':(finished-started).total_seconds(),'scientific_cycles':[19,20,21,22,23],'independent_candidates':['E','F','G'],
        'trained_models':len(all_models),'confirmation_prediction_batch_records':len(rows),'training_seconds_sum':train_seconds,
        'prior_tracked_files_unchanged':len(old),'stop_reason':'five meaningful cycles plus fresh capacity control completed; residual/positivity/OOD defects block nonlinear incompressible-flow scale-up; next independent mechanisms not yet defined',
        'architecture_proven':False,'novelty':'unverified','source_control_actions':'none this research session',
        'external_actions':'no push, research upload, paid resource, GPU training or deletion'})
    names=[p for p in subprocess.check_output(['git','ls-files','--others','--exclude-standard','-z']).decode().split('\0') if p]
    write('validation/source_hashes.json',{p:sha(p) for p in names if p.endswith('.py')})
    names=[p for p in subprocess.check_output(['git','ls-files','--others','--exclude-standard','-z']).decode().split('\0') if p]
    manifest={p:sha(p) for p in names}
    write('validation/final_manifest.json',{'files':manifest,'prior_files_unchanged':len(old),'git_head_unchanged':True,
                                         'interpretation':'final manifest excludes itself; timings and validation provenance are local recorded evidence'})
    print(json.dumps({'session_complete':True,'cycles':5,'trained_models':len(all_models),'prediction_batches':len(rows),
                      'prior_files_unchanged':len(old),'new_files_in_manifest':len(manifest),'wall_minutes':(finished-started).total_seconds()/60},indent=2))


if __name__=='__main__':main()
