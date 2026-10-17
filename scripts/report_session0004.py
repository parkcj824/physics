"""Build Korean reports and standalone plots from executed Session 0004 evidence."""

import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R=Path('results/session_0004')
OUT=Path('reports/session_0004')
RESEARCH=Path('research/session_0004')


def get(name):return json.loads((R/name).read_text())


def create(path,content):
    if path.exists():raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(content.strip()+'\n')


def summarize(rows,key):
    v=np.array([r[key] for r in rows]);return f'{v.mean():.5g} ± {v.std():.3g}'


def table(rows,family,keys):
    selected=[r for r in rows if r['family']==family and 'mean_relative_l2' in r]
    body=['| 모델 | 조건 | '+' | '.join(keys)+' |','|---|---|'+'---|'*len(keys)]
    for mode,case in sorted({(r['mode'],r['case']) for r in selected}):
        group=[r for r in selected if r['mode']==mode and r['case']==case]
        body.append('| '+mode+' | '+case+' | '+' | '.join(summarize(group,k) for k in keys)+' |')
    return '\n'.join(body)


def main():
    confirm=get('confirmation.json');transport=get('transport_confirmation.json')
    capacity=get('capacity_confirmation.json');capmodels=get('capacity_frozen.json')['models']
    rows=confirm['records']+[{**r,'family':'H'} for r in transport['records']]
    frozen=get('frozen_models.json');models=frozen['models']+get('transport_frozen.json')['models']
    br=get('bratu_reference.json');counter=get('mathematical_counterexamples.json');controls=get('classical_controls.json')['records']
    integrity=get('validation/integrity.json');env=get('environment.json')
    e_table=table(rows,'E',['mean_relative_l2','mean_relative_l2'])
    # Avoid duplicating a metric in the final table.
    e_table=table(rows,'E',['mean_relative_l2','missing_branches_at_5percent','spurious_branches_at_5percent'])
    f_table=table(rows,'F',['mean_relative_l2','mean_cross_relative_l2','mean_pde_residual_inf'])
    g_table=table(rows,'G',['mean_relative_l2','mean_pde_residual_inf','max_generated_energy_balance_abs'])
    h_table=table(rows,'H',['mean_relative_l2','mean_pde_residual_inf','max_generated_energy_balance_abs'])
    learned=[m['learned_fold'] for m in models if m['mode']=='paired']
    feedback=max(r['max_generated_energy_balance_abs'] for r in rows if r['family']=='G' and r['mode']=='feedback')
    losses={}
    for family,mode in sorted({(m['family'],m['mode']) for m in models}):
        losses[family+'/'+mode]=float(np.mean([m['best_validation_loss'] for m in models if m['family']==family and m['mode']==mode]))
    training_total=sum(m['seconds'] for m in models+capmodels)
    costs=['| family/model | IID batch32 ms | parameter수 | 실제 학습 seconds,3seed 합 |','|---|---:|---:|---:|']
    for family,mode in sorted({(m['family'],m['mode']) for m in models}):
        rr=[r for r in rows if r['family']==family and r['mode']==mode and r['case']=='iid']
        mm=[m for m in models if m['family']==family and m['mode']==mode]
        costs.append(f"| {family}/{mode} | {np.median([r['median_seconds'] for r in rr])*1000:.5g} | {mm[0]['parameter_count']} | {sum(m['seconds'] for m in mm):.5g} |")
    for c in controls:
        if c['case']=='iid':costs.append(f"| {c['family']}/classical | {c['median_seconds']*1000:.5g} | 0 learned | label/control 비용 별도 |")
    hc=next(c for c in transport['classical_costs'] if c['case']=='iid')
    costs.append(f"| H/classical upwind | {hc['median_seconds']*1000:.5g} | 0 learned | label/control 비용 별도 |")
    for family in ['G','H']:
        cr=[r for r in capacity['records'] if r['family']==family and r['mode']=='direct_width64']
        cm=[m for m in capmodels if m['family']==family]
        costs.append(f"| {family}/direct_width64 (fresh IID) | {np.median([r['median_seconds'] for r in cr])*1000:.5g} | {cm[0]['parameter_count']} (active8415) | {sum(m['seconds'] for m in cm):.5g} |")
    absent=['| 모델 | 없는16조건에서 선언한 허위 branch수,3seed 평균 |','|---|---:|']
    for mode in sorted({r['mode'] for r in rows if r['family']=='E'}):
        rr=[r for r in rows if r['family']=='E' and r['mode']==mode and r['case']=='absent']
        absent.append(f"| {mode} | {np.mean([r['spurious_branches'] for r in rr]):.5g} |")
    text=f'''# Physics Foundation Architecture Research — Session 0004

**독립 후보3개(E/F/G), 명시적 수정 E-R, G의 방향성 확장을 Cycle19–23의5개 Cycle에서 검증했다. 새로운 범용 Architecture/Level3 신규성을 입증하지 못했다.** 기존 수학을 사용했다는 이유로 배제하지 않고, 표현·학습 메커니즘·효과·선행연구를 구분했다. 후속 capacity 통제까지 실제 학습 모델51개(3seed씩), 확인 평가183행, pytest {integrity['pytest_passed']}개 통과, 이전 추적 파일 {integrity['prior_tracked_files_unchanged']}개 보존을 검증했다. 183행은 method/seed/case별 예측 batch이며 독립 dataset 수가 아니다.

## A. 새로 정의한 계산 문제

단일 고정 선형 map 대신 (1) 가능한 정상해 집합과 동적 선택의 분리, (2) finite source edit의 비선형 교차응답, (3) 상태와 수지의 공동 생성을 연구했다. Bratu는 중간 검증 문제이며 모든 물리에 branch 객체가 필요하다는 주장이 아니다. Architecture 재사용과 도메인 간 가중치 공유도 구별한다.

Bratu 해는 `T_a=-2log(cosh[a(x-1/2)]/cosh(a/2))`, `lambda(a)=2a²/cosh²(a/2)`이다. 직접 PDE 대입, scalar root, 서로 다른 solve_bvp 기준선으로 검증했다. Fold a={br['a_fold']:.12g}, lambda={br['lambda_fold']:.12g}, fold condition 오차 {br['fold_condition_error']:.3g}. 독립 BVP 최대 온도 오차는 {max(r['max_absolute_error'] for r in br['independent_bvp']):.4g}다.

lambda2에서 낮은 branch의 성장 고유값(N127)은 약-7.20755, 높은 branch는+15.86456이다. Cold 및 upper보다0.01 eigenmode 낮은 초기상태는 낮은 branch로 수렴(상대오차 약1e-10)했고, upper보다 높은 초기상태는 t≈0.27035에 온도20 기준에 도달했다. 이는 수치 runaway 지표이며 blow-up의 증명이 아니다. **두 정상해가 존재해도 둘 다 안정하지 않다.**

## B. 독립 후보와 완전한 연산

정확한 변수 타입·단위·고정/학습 구분·보장·복잡도는 `architectures/session_0004/candidate_E.md`, F/G에 있다. 입력은 무차원화했고 이전 C006/응답 inverse-update/불충분 witness/좌표군/transfer를 재명명하지 않았다.

- **E: paired fold-coordinate field.** `(u,v)=MLP(lambda/4,(x-.5)²)`, `c=8x(1-x)u`, `g=8x(1-x)softplus(v)`, `d=3.2+softplus(alpha)-lambda`, `T±=c±sqrt(d)g`(d>0). 중심·간격·존재를 공유하는 Level2 표현이다. 비용 O(Nw²), 학습은 paired-field MSE다.
- **F: finite-intervention mixed response.** `T=x(1-x)[f0(x)+a f(a,c1,x)+b f(b,c2,x)+ab/sqrt(r) sum_j q_j(x)e_j(a,c1)e_j(b,c2)]`. 네 counterfactual 해의 차이를 정규화해 함께 학습한다. Rank16, 단독항 O(MNw²), pair O(M²r+Nr). 유한 비선형 상호작용 채널이며 Jacobian oracle을 공급하지 않는다.
- **G: state/balance co-generation.** Latent에서 `T=lift+x(1-x)a`, `s=Q+lambda expT`, `j_e=j0+h sum_(i≤e)s_i`를 공동 생성하고 `r=j+k gradT`를 만든다. 다음 latent는 `tanh(Wh+w+Ur+tanh(Gh+g)⊙tanh(Vr))`다. 구성식 mismatch가 내부 정보 흐름에 들어가며 projection/추가 PINN loss는 실행하지 않는다. 비용 O(L(Nw+w²)), 고정N15다.

## C. 기존 계산 방식과의 차이

E는 구조화된 conditional/two-head MLP, F는 learned symmetric bilinear pooling, G는 hard-balanced decoder와 물리 mismatch를 입력으로 받는 gated recurrent block으로 펼쳐진다. 이 사실은 기존 수학을 썼다는 자동 기각이 아니다. 실제 분해·학습·공유 제약이 기존 모델과 얼마나 다르고 어떤 효과를 제공하는지가 판단 대상이다.

E의 출력 집합 구조와 F의 교차응답 학습은 Level2 연구 가치가 있다. G는 Level3을 목표로 한 정보 흐름 가설이지만 ablation 이득과 prior-art 검토가 필요하다. 어느 것도 이번에 검증된 Level3/4 발명으로 승격하지 않는다. E-R은 정확한 fold를 공급한 **명시적 수정/불균등 physics-prior control**이며 새 독립 후보가 아니다.

## D. 실제 물리 검증

사전 정의한 train/validation/confirmation seed·범위·step은 `research/session_0004/preregistered_protocol.md`에 있다. 확인 전 metric·parameter-matched MLP 보강은 별도 outcome-blind amendment에 기록했다. 각 실험군의 checkpoint를 모두 동결한 뒤 해당 확인 집합을 평가했고 그 결과로 재학습/선택하지 않았다. 방향성 확장은 main 확인 결과 전에 개발 validation을 근거로 별도 사전 정의했다. 데이터 파일은 미리 생성했지만 OS 수준 blind access 통제는 아니며 실행 순서와 기록에 근거한 분리다.

E는48lambda×2branches×15points 학습, validation32, 확인 IID32/near-fold24/absent16/N63 query32다. N63은 새 물리 데이터 분포가 아니라 같은 IIDlambda의 resolution query 검사다. 표는 seed별 sample 평균의 평균±표준편차다. Branch수의5% 진단은 모델이 선언하는 존재 gate와 별개이며 field 전체 permutation matching을 사용했다.

{e_table}

{chr(10).join(absent)}

E의 학습 fold값은 {', '.join(f'{v:.8g}' for v in learned)}다. Paired-oracle은 정확한 fold를 공급했으므로 no-branch 개선을 데이터만의 학습 효과로 해석하지 않는다. Single mean control의 collapse는 강한 mixture 모델에 대한 우월성 증거가 아니다. 안정성은 N255에 query한 field의 linearized eigenvalue로 별도 평가했다.

F는 `-T''=1.5 expT + a s1+b s2`, N15, Gaussians width.1, 영BC다. 128개 edit pair마다4개 solution을 사용해 **512개 field label record**, validation128field, 확인256field다. 같은 T00가 반복되어 학습 target의 서로 다른 hash는385개다. Base를 cache하면 준비 solve수를 줄일 수 있으며512를 독립 field 수라고 주장하지 않는다. 모든 F모델이 같은 quartet/loss를 받았다. Larger edits/변경 source 위치는 사전 정의한 OOD다.

{f_table}

F pair의 IID 온도 오차0.630%와 additive0.693%는 근접한다. 더 분명한 이점은 mixed-response 오차2.29% 대 parameter-budget MLP6.62%다. 물리학적 상호작용을 분리하는 표현의 잠정 효과이며 일반적인 온도 정확도 우월성을 과장하지 않는다. 세 seed는 같은 dataset에서 초기화만 달리한 것이고 통계적 유의성/데이터 효율 curve를 입증하지 않았다.

G는 균일/부드러운 비균질 k, 국소+균일source, BC 및 reaction0/1.5를 혼합한128label+validation32다. 확인 IID32/lowk16/변경 source16/BC16이다. Lowk와 변경BC 확인은 lambda0이다. 표에는 다른 물리 regime을 섞은 평균이 있으므로 세부 입력은 저장 배열이 기준이다.

{g_table}

BC는 세 후보에서 지정 envelope로 정확히 주입한다. 이는 학습 성취가 아니라 architectural prior다. G feedback의 생성flux 전역수지 최대 위반은 {feedback:.4g}였지만 구성식 flux와 온도 PDE는 별도 지표이며 작은 generated balance가 정답을 보장하지 않는다. F/E는 구성식 flux 수지를 검사했고 E의 coarse residual에는 해석해의 이산오차도 포함되므로 continuum/이산 오차를 혼동하지 않는다.

특히 G feedback의 IID 온도 오차7.45%는 direct7.92%/no-feedback12.75%보다 작지만 PDE잔차 평균5.00은 direct1.57/no-feedback2.02보다 크다. Low-k 오차69.34%도 남았다. 작은 field L2와 정확한 생성flux 수지만으로 물리 통과를 선언하지 않는다. 과도한 curvature/구성식 오차가 다음 수정 대상이며 추가 GPU학습으로 구제하지 않았다.

### Cycle23 — 같은 G block의 정상 이류–확산 방향성

개발 validation을 근거로 사전에 확장을 결정했다. `(v u-nu u_x)_x=S`, N15, upwind constitutive flux를 사용한다. Source누적/상태 생성/3-layer gate와학습규칙은 유지하고 구성식만 바꿨으며 weights는 별도로 학습했다. 128train/32validation, 확인 IID32/방향쌍16×2/낮은확산16이다.

{h_table}

방향 centroid는 feedback뿐 아니라 모든 비교 모델이16/16쌍에서 맞혔다. 따라서 방향을 구별한다는 사실 자체가 새 gate의 독점 장점은 아니다. Feedback IID 상대오차14.60%는 no-feedback39.98%/direct43.68%보다 낮았지만, low-diffusion33.94% 및 PDE잔차가 남았다. 생성 flux수지는 약1e-16 수준이다. 실제 upwind/banded reference는 같은 데이터에서 roundoff 수준의 온도·수지 오차를 낸다. 이 결과는 **정상 scalar transport**이며 시간 안정성/positivity 또는 비압축성Navier–Stokes 성공이 아니다.

저장한 방향성 예측의 사후 진단에서 **positivity 위반도 확인**했다. 양의 source/영BC에서 reference 최솟값은9.436e-5인데 feedback 예측의 최솟값은 seed별 -0.02998/-0.0001012/-0.004914였다. 32개 조건 중 음수 예측을 낸 sample은 각각10/1/10개다. 모델을 재학습하지 않고 `transport_positivity_audit.json`에 기록했다. 일반적인 signed 온도편차와 이 nonnegative-source benchmark의 최대원리 요구를 혼동하지 않는다.

### 실제 비용과 데이터

{chr(10).join(costs)}

CPU Python {env['python']}, torch {env['versions']['torch']}, thread1. 실제 학습 측정시간 합 {training_total:.5g}초(트레이싱 overhead 포함), label/reference 준비는 dataset_manifest에 별도 기록했다. F 추론은 최종 field 하나, G는 T와flux, E는 두branch를 비교한다. Tensor변환/정규화 또는 source/matrix조립 포함, disk I/O와 N255 안정성 진단 제외. Classical solver는 reference 허용오차까지 수렴하고 neural은 근사해이므로 같은 오차 허용의 speedup을 입증한 비교가 아니다. 알려진 해석적 parameterization이 있는 Bratu 시간은 일반 CAE 속도로 확장할 수 없다.

Tracemalloc은 native/BLAS/PyTorch allocations를 일부 누락한다. 모델별 실제 peak 전체메모리나 GPU속도는 미측정이다. G/H no-feedback에는 사용하지 않는 feedback parameter도 있어 총 parameter수만으로 공정성/유효계산량이 같다고 주장하지 않는다. Feedback는 활성 parameter/compute가 더 많으므로 이 실험만으로 효과의 원인을 고유한 물리 mechanism으로 확정할 수 없다. F의 width64 MLP도 parameter/time을 함께 제시하며 완전한 architecture HPO sweep은 아니다.

대표 train입력의 gradient 경로를 검사한 결과 G/H에서 실제 gradient를 받은 parameter수는 feedback8704, no-feedback4816, direct5551이었다. 이 진단은 weight를 갱신하지 않았고 `parameter_accounting.json`에 저장했다. 이를 발견한 뒤 별도 사전 계획과 **새 IID 확인 집합**으로 active8415/width64/3000step MLP 통제를 추가했다. 원래 후보 checkpoint와 확인 결과는 바꾸지 않았다. 추가 학습6개·새 평가24행을 원래45개·159행과 구분한다.

새 확인에서 G의 온도 오차는 feedback7.37% 대 matched-control8.09%였지만 PDE잔차는4.89 대1.37로 여전히 불리했다. 방향성 H의 온도 오차는8.31% 대23.95%, PDE잔차는0.637 대0.644였다. 후속 통제에서도 제한된 표현 이득이 남았으나 모든물리지표의 우월성이나 신규성을 입증하지 않았다. 전체seed표·실제시간은 `capacity_control.md`, 새사전계획은 `research/session_0004/capacity_control_protocol.md`에 있다. 최초확인집합을다시독립검증이라부르지않는다.

## E. 반증 및 수정

1. E의 `g`를 다시 scale하면 fold3.6/4/6으로도 같은 training field를 최대약7e-16오차로 재현할 수 있었다. Branch label만으로 existence boundary가 식별되지 않는다. 정확한 fold를 제공한 E-R이 이 결함을 제거해도 이는 추가 물리정보를 사용한 결과다.
2. 두 정상해를 평균한 field는 lambda2에서 상대 PDE잔차 {counter['mean_field_does_not_solve_pde']['relative_pde_residual']:.5g}, 에너지수지 오류 {counter['mean_field_does_not_solve_pde']['constitutive_energy_balance_abs']:.5g}였다. 단일 평균은 물리적 branch 선택이 아니다.
3. F의 exact pair truncate도 triple response를 누락한다. lambda0 삼중차이 norm {counter['higher_order_interactions'][0]['triple_norm']:.4g}, lambda1.5에서는 {counter['higher_order_interactions'][1]['triple_norm']:.4g}였다. 이번 작은 counterexample의 상대오차가 작다고 arbitrary edit수에서 오차가 작다는 보장은 없다.
4. G는 수지와 구성식 정확도를 분리한다. Feedback가 존재한다는 사실이나 보존 guarantee만으로 accurate PDE를 해결했다고 주장하지 않는다. Ablation 비교에서 확인한 이득·실패를 아래 판정과 저장 결과에 남겼다.

## F. 발견한 구조와 현재 판정

새 물리법칙/항등식은 주장하지 않는다. 유용한 계산상의 구분은 **해 집합과 selector**, **finite mixed effect와 일차응답**, **생성flux의 보존과 constitutive consistency**, **표현·메커니즘·문헌 신규성**이다. 이들을 같은 성과로 혼동하지 않도록 실제 반례와 모델을 만들었다.

현재 **제한된 Level2 후속 probe는 F**, **Level3을 목표로 유지할 경쟁 mechanism은 G**다. 확인 전에도 두 후보를 provisional next probe로 기록했다. 이는 범용 architecture 선정이나 독립 발명 승인과 다르다. E는 branch existence 정보 식별 문제를 먼저 해결해야 하고 G는 PDE잔차/OOD와 feedback의 추가 효과를 해결해야 한다. 기존 prefix 수지연산은 G의 일부 고정decoder일 뿐 Session3 D처럼 전체 transfer solver를 재현한 것은 아니지만, 이 차이만으로 신규성을 주장하지 않는다. 모든 후보의 신규성은 UNVERIFIED이며 어떤 모델도 확립한 새로운 Level3 primitive로 선정하지 않았다.

## G. 논문·특허

새 원문 접근을 반복하지 않았고 이전403기록을 재사용했다. 저장한 DEQ/OptNet/CVXPYlayers/NeuralOperator/MeshGraphNets 공식 README·일부 code의 확인 범위만 인용한다. Full paper/특허 검색은 미실행이다. Saddle-node, Volterra, bilinear/set pooling, gated residual models와의 가까운 관계는 직접 식 비교/배경지식으로 명시했으며 특정 읽지 않은 논문의 수식으로 꾸미지 않았다. **신규성·진보성 미확인**, 새 architecture 논문/특허의 기여를 입증하지 못했다.

## H. 현재 최고 후보와 미해결 과제

확립한 architecture 없음. 정확한 classical banded/Newton/Bratu root control은 유지하고 새 발명으로 이름을 바꾸지 않는다. F는 matched stronger baseline/추가 source geometry/삼중 이상 interaction/label efficiency와 이미 알려진 quadratic-set learning과의 차이를 우선 검증해야 한다. E-R은 추가 fold정보가 주는 효과와 배포시 해당 정보 획득 비용을 분리해야 한다. G는 동일 joint decoder의 효과와 latent feedback의 효과를 분리해야 한다.

## I. 다음 독립 방향과 종료

Cycle19–23의5개 의미 있는 가설/수정/확장검증을 완료했다. 확인 결과를 본 뒤 모델을 반복 튜닝하거나 숫자를 맞추기 위한 Cycle을 채우지 않는다. 정상 방향성은 실행했지만 열 PDE/OOD와transport 오차가 남아 **비압축성 nonlinear 유동/RANS/대규모GPU는 확장하지 않았다**. 다음에는 **물리적 basin을 선택하는 causal relaxation/threshold 정보**, **조합되지 않는 고차 interaction의 적응적 활성화**, **구성식과 balance 사이의 오차를 운반하는 재사용 가능한 객체**를 기존RNN/Volterra/learned solver와 비교해 새 mechanism부터 정의한다. 이는 아직 미실행 질문이다.

명령과 정확한 재개 순서는 `research/session_0004/next_steps.md`에 있다. GitHub push, 외부 연구데이터 업로드, 유료 자원, GPU학습, 삭제는 수행하지 않았다. 이전1,154파일 해시를 보존했다.
'''
    create(OUT/'research_summary.md',text)
    create(OUT/'current_best.md','''# Session 0004 현재 판정

확립한 범용 architecture/Level3 발명: **없음**. 후속 연구용 Level2 probe: F finite mixed-response 표현. 확인 결과로 모델을 재학습하거나 최종 발명으로 선택하지 않았다.

E: branch representation은 가능하지만 fold 식별이 불충분(REVISE). E-R은 정확한 fold를 별도 공급한 control. F: KEEP as limited representation probe; 이미 알려진 bilinear/set pooling 대비 신규성 미확인. G: Level3 목표의 경쟁 mechanism으로 유지하되 열PDE잔차/OOD 및 active-parameter 통제가 미해결(REVISE/UNRESOLVED). G는 별도weights로 정상 이류–확산도 실행했고 방향성은맞지만정밀한물리해를확보하지못했다.

정확한 수치 기준선은 classical banded/Newton/analytic Bratu다. 세부 결과와 한계는 research_summary.md, 재개는 research/session_0004/next_steps.md.
''')
    create(RESEARCH/'candidate_registry.md','''# Session 0004 후보 등록

| 후보 | 독립 원리/객체 | 실제 학습 | 보장 | 판정·수준 |
|---|---|---|---|---|
| E | 다중 정상해 집합의 중심·gap | MLP+fold scalar,3seed | BC/순서/반사대칭 | REVISE; Level2 표현,fold미식별 |
| F | finite 변화의 mixed response | singleton/pair MLP채널,3seed | BC/swap/zero cross anchor | KEEP limited Level2 probe,Level3 미입증 |
| G | 상태와source balance의 공동 형성 | shared latent gate/decoder,3seed | 생성flux수지/BC | REVISE/UNRESOLVED; feedback 효과/신규성 미해결 |
| E-R | E에 정확한fold 추가 | 같은field net,fold고정 | 올바른존재gate prior | classical-info control; 독립 발명 아님 |

기존 수학이라는 이유로 자동 REJECT하지 않았다. 실제 학습과 정보흐름의 차이·효과·한계를 평가했다. E/F/G는 과거 후보의 재명명이 아니다. 원문/특허 신규성은 모두 미확인이다.
''')
    cycle_info=[(19,'E — branch 집합과 안정성','다중해/상태','(c,g,d) paired field','T±=c±sqrt(d)g','fold와g의 재매개화로 같은field에 다른fold 가능','REVISE; Level2 표현','E-R에서 추가fold정보의 역할 분리'),
                (20,'F — finite nonlinear mixed response','비선형 교란 상호작용','counterfactual quartet','T=BCenvelope*(base+single_a+single_b+ab*bilinear channels)','삼중mixed response가0이 아니므로 pair만으로 완전 합성 불가','KEEP limited Level2 probe','고차interaction의 적응적활성화를 별도원리로 검토'),
                (21,'G — state/balance co-generation','보존과구성식의공동계산','latent→(T,j)→constitutive mismatch','h+=tanh(Wh+Ur+tanh(Gh)*tanh(Vr))','생성flux 보존은 constitutive/PDE 정확도를 보장하지 않음','REVISE/UNRESOLVED','jointdecoder와feedback의추가효과를분리'),
                (22,'E-R — calibrated fold와 독립 확인','E의정보식별결함검토','추가물리정보를받는paired state','d=lambda_star_exact-lambda; T±=c±sqrt(d)g','정확한fold를공급한효과를학습성취로오인하면안됨','control; 독립새후보아님','basin선택에필요한causal정보를새계산문제로정의'),
                (23,'G 방향성 확장 — constitutive law의 역할','상태와방향성있는flux의공동생성','기존Gblock에upwind이류구성식','j=j0+h cumsum(S); r=j-(v+uL+v-uR-nu grad u)','모든모델이centroid방향을맞혔지만feedback도14.6%IID오차와33.9%낮은확산오차','REVISE; 부분적인architecture재사용','구성식잔차를줄이는새오차표현을solver재현과구분')]
    for cycle,title,principle,obj,equation,objection,verdict,nxt in cycle_info:
        family='E' if cycle in [19,22] else ('F' if cycle==20 else ('H' if cycle==23 else 'G'))
        relevant=[r for r in rows if r['family']==family]
        if cycle==19:relevant=[r for r in relevant if r['mode']!='paired_oracle']
        if cycle==22:relevant=[r for r in relevant if r['mode']=='paired_oracle']
        cycle_table=table(relevant,family,['mean_relative_l2'])
        compact=[{k:v for k,v in r.items() if k in ['mode','seed','case','mean_relative_l2','mean_cross_relative_l2','spurious_branches','missing_branches_at_5percent','mean_pde_residual_inf','max_generated_energy_balance_abs']} for r in relevant]
        evidence=R/f'cycle_{cycle}_evidence.json';evidence.write_text(json.dumps(compact,indent=2)+'\n')
        body=f'''# Cycle {cycle} — {title}

1. **출발 원리:** {principle}.
2. **최소 계산 객체:** {obj}. 단위/입력은 architectures/session_0004의 해당 후보 수식 참조.
3. **구현 연산:** `{equation}`. 내부MLP와고정/학습변수는후보파일에완전히전개했다.
4. **기존 계산과 차이:** {family}의학습정보흐름을일반MLP/mixture/bilinear/gated residual구조와대조했다. 기존수학사용만으로기각하지않았다.
5. **검증 방정식:** Bratu/1D정상열반응/variable-k 열전도중해당후보범위. 실제lambda,k,source,BC와정답은data/arrays에있다.
6. **실제 결과:** seed3개의 sample평균 상대오차는 아래와 같다. 전체수치는 `results/session_0004/cycle_{cycle}_evidence.json`이며 실제입력/예측/정답/절대오차/보존/비용은confirmation 또는transport_confirmation의arrays경로와1:1대응한다. 개발결과→수학반례→동결모델의사전정의확인평가를분리했다.

{cycle_table}

7. **자기반박:** {objection}.
8. **신규성 검토:** 기존수학/신경연산과의구체관계를기록했다. 저장된공식source만확인했고full paper/특허미검토이므로신규성미확인이다.
9. **판정:** {verdict}. 새로운범용architecture의입증은없다.
10. **다음 가설:** {nxt}. 미실행가설을실행결과로세지않는다.
'''
        create(OUT/f'cycles/cycle_{cycle}.md',body)
    # Standalone scientific plots: use only recorded confirmation arrays/metrics.
    fig,axes=plt.subplots(1,3,figsize=(15,4.4),layout='constrained')
    for mode in ['paired','paired_oracle','two_head']:
        row=next(r for r in rows if r['family']=='E' and r['mode']==mode and r['case']=='fold' and r['seed']==1004)
        with np.load(R/row['arrays']) as d:
            order=np.argsort(d['lam']);mid=len(d['x'])//2
            axes[0].plot(d['lam'][order],d['prediction'][order,0,mid],label=mode)
            axes[0].plot(d['lam'][order],d['prediction'][order,1,mid],color=axes[0].lines[-1].get_color())
            truth=d['target'];ll=d['lam']
    axes[0].plot(ll,truth[:,0,mid],'k--',lw=1,label='analytic branches');axes[0].plot(ll,truth[:,1,mid],'k--',lw=1)
    axes[0].axvline(br['lambda_fold'],color='k',alpha=.3);axes[0].set(xlabel='lambda',ylabel='T(0.5)',title='Held-out fold neighborhood');axes[0].legend(fontsize=7)
    positions=np.arange(3)
    for shift,mode in zip([-.25,0,.25],['pair','direct_matched','additive']):
        means=[];std=[]
        for case in ['iid','large','source']:
            vals=[r['mean_cross_relative_l2'] for r in rows if r['family']=='F' and r['mode']==mode and r['case']==case]
            means.append(np.mean(vals));std.append(np.std(vals))
        axes[1].errorbar(positions+shift,means,yerr=std,marker='o',ls='none',capsize=3,label=mode)
    axes[1].set_yscale('log');axes[1].set_xticks(positions,['IID','larger edits','source shift']);axes[1].set(ylabel='Mixed-response relative error',title='Finite interaction learning');axes[1].legend(fontsize=8)
    for mode in ['feedback','no_feedback','direct']:
        rr=[r for r in rows if r['family']=='G' and r['mode']==mode]
        axes[2].scatter([r['max_generated_energy_balance_abs'] for r in rr],[r['mean_pde_residual_inf'] for r in rr],label=mode,alpha=.7)
    axes[2].set(xscale='log',yscale='log',xlabel='Generated-flux energy imbalance',ylabel='Temperature PDE residual (inf)',title='Balance is not constitutive accuracy');axes[2].legend(fontsize=8)
    plots=R/'plots';plots.mkdir(exist_ok=True);fig.savefig(plots/'session0004_evidence.png',dpi=180);plt.close(fig)
    # Independent directed-domain figure, showing both target and prediction.
    fig,ax=plt.subplots(figsize=(7,4.2),layout='constrained')
    row=next(r for r in rows if r['family']=='H' and r['mode']=='feedback' and r['seed']==1004 and r['case']=='direction')
    with np.load(R/row['arrays']) as d:
        for i,color in [(0,'tab:orange'),(1,'tab:blue')]:
            ax.plot(d['x'],d['target'][i],color=color,label=f"upwind reference v={d['inputs'][i,-1]:+.0f}")
            ax.plot(d['x'],d['prediction'][i],'--',color=color,label=f"G prediction v={d['inputs'][i,-1]:+.0f}")
    ax.set(xlabel='x',ylabel='u',title='Steady advection–diffusion: velocity reversal');ax.legend(fontsize=8)
    fig.savefig(plots/'transport_direction.png',dpi=180);plt.close(fig)
    create(RESEARCH/'development_losses.json',json.dumps(losses,indent=2))
    print(json.dumps({'reports_written':True,'cycle_count':5,'figure':str(plots/'session0004_evidence.png')}))


if __name__=='__main__':main()
