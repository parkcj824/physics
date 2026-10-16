"""Generate Korean Session 0003 reports and figures from verified recorded arrays."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT=Path(__file__).resolve().parents[1]
RESULT=ROOT/"results/session_0003"
REPORT=ROOT/"reports/session_0003"
FIGURES_ONLY=False


def write_new(path,text):
    if FIGURES_ONLY:
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Preserve previous report: {path}")
    path.write_text(text.strip()+"\n")


def main():
    rows=json.loads((RESULT/"metrics.json").read_text())
    integrity=json.loads((RESULT/"validation/integrity.json").read_text())
    assert integrity["status"]=="verified"
    def find(candidate,case):
        return next(r for r in rows if r["candidate"]==candidate and r["case"]==case)
    def models(condition):
        return [r for r in rows if r["cycle"]==18 and (r["case"]==condition or r["case"].startswith(condition+"_seed"))]
    def mean_mlp(condition):
        values=[r["mean_sample_relative_l2"] for r in models(condition) if r["candidate"]=="MLP"]
        return float(np.mean(values)),float(np.std(values,ddof=1))
    candidates=[
      (13,"A — 유한 response packet",r"G^+=G-\alpha(Gd)(d^TG)/(1+\alpha d^TGd)",
       "유한 교란의 반응과 cross-susceptibility를 기본 단위로 본다.",
       "분모 없는 일차 감도와 frozen packet을 대조했다. 정확한 수정은 Sherman–Morrison와 연산 자체가 같다.",
       "학습하지 않았다. Exact base inverse를 공급한 oracle/control이며 그 준비 O(N³), 각 갱신 O(N²)를 별도 비용으로 기록했다.",
       "REJECT","고전 inverse update로서 유효하지만 신규 primitive가 아니다. B의 witness 충분성을 검토한다."),
      (14,"B — witness 충분성",r"w=Rb,\quad u=(RA)^+w",
       "해 대신 coarse balance 관계와 전역 invariant를 표현한다.",
       "동일 witness·같은 source norm을 갖는 다른 양수 source와 다른 해를 구성했다. Signed pair는 어떠한 decoder도 동시에 정확히 예측할 수 없다.",
       "원리적 식별 불가능성을 먼저 검사해 큰 decoder를 학습하지 않았다. 실험 decoder는 minimum-norm closure이고 그 약99% 오차가 최적 하한이라는 뜻은 아니다.",
       "REJECT","현재 witness-only 표현을 제외한다. 보조 인증 용도 전체를 반증한 것은 아니다. C에서 변환군 기반 합성을 시험한다."),
      (15,"C — 비선형 변화량 합성",r"a\star b=a\sqrt{1+\gamma^2b^2}+b\sqrt{1+\gamma^2a^2}",
       "대칭/변환 법칙의 원소로 변화량을 합성한다. A/B의 조합이 아니다.",
       "Raw 연산은 선형 source superposition을 위반한다. asinh 좌표를 되돌린 control은 정확하지만 phi^-1(phi(a)+phi(b))라는 덧셈의 재표현이다.",
       "gamma=4 고정, 학습 미실행. 주어진 exact response contributions를 합쳤으므로 정확한 correction을 새 학습 성능으로 세지 않는다.",
       "REJECT","새 coupling이 아닌 좌표 변화다. 세 후보가 배제되어 실제 공간 순서의 다른 물리 원리 D로 전환한다."),
      (16,"D — 순서 있는 구간 관계",r"(R,S,C)\circ(R',S',C')=(R+R',S+S',C+C'+R'S)",
       "실제 공간 구간의 순서가 source와 저항의 cross moment를 운반한다.",
       "정상 열전도를 정확히 계산하나 affine transfer matrix와 동일하다. Root tuple은 boundary port만 설명하며 전체 장에는 모든 prefix 관계가 필요하다.",
       "이 단계 theta=1은 고정한 물리 값이다. Cycle18에서 같은128개 label로 scalar를 실제 적합한다. Prefix O(N)과 banded solver O(N)을 비교한다.",
       "REJECT","물리적으로 유효한 고전 control로 보존한다. 단순 matrix 표현이라는 이유가 아니라 전체 transfer 알고리즘과 동일하여 신규 주장하지 않는다."),
      (17,"방향성·보존·위상 반증",r"G^+=G-(Ga)(c^TG)/(1+c^TGa),\quad u_{n+1}=G^+u_n",
       "비자기수반 response의 좌·우 정보를 구별하고 global witness가 위상을 결정하는지 확인한다.",
       "Full oriented update는 backward Euler reference와 일치한다. Reciprocal shortcut은 질량 보존 오차가 작아도 위상을 틀린다. 같은 질량/에너지의 다른 초기 위상도 구별해야 한다.",
       "신규 학습 없음. Reference는 upwind 공간 이산+backward Euler; 정확한 semidiscrete exponential과의 시간 오차를 별도 저장했다. D는 transient 규칙을 정의하지 않으므로 억지 적용하지 않았다.",
       "REJECT","새 범용 primitive가 검증된 것은 아니다. Stokes는 새 후보가 남지 않아 확장하지 않는다. Cycle18에서 강한 기준선과 비용을 검토한다."),
      (18,"공정 비교와 baseline 자기반박",r"T_\theta=a+\theta s,\quad \hat\theta=\frac{\sum s^T(y-a)/\|y\|^2}{\sum\|s\|^2/\|y\|^2}",
       "동일한 heterogeneous 물리 label에서 알려진 cross coefficient와 일반 surrogate를 비교한다.",
       "Scalar는1로 복원된다. Unregularized 선형 control의 ill-conditioning을 진단하고 validation-selected ridge를 추가했다. 동일 label 수는 동일 prior/compute budget이 아니다.",
       "D/feature-linear/MLP는 동일128개 label. MLP는3seed,1000Adam step. Ridge는 validation32개만으로5개 alpha 선택. Unit-compliance는 별도 unit-k label128개, C006은 이전 pretraining이므로 matched-data 주장하지 않는다.",
       "REJECT","Known physical algebra가 우수해도 발명이 아니다. 의미 없는 추가 Cycle을 만들지 않고6Cycle에서 중단한다."),
    ]
    for cycle,title,formula,principle,counter,training,verdict,next_state in candidates:
        subset=[r for r in rows if r["cycle"]==cycle]
        lines=[f"# Cycle {cycle} — {title}","", "## 가설과 계산 객체",principle,"",
               "## 구현 수식", "\\[",formula,"\\]","",
               "입력 타입·단위·학습 변수·성질·복잡도와 기존 연산 비교는 `../../../architectures/session_0003/candidate_*.md`에 전개했다.","",
               "## 수학적 반례와 기존 계산 비교",counter,"", "## 최소 구현·실행·학습 여부",training,"",
               "실행 코드 `experiments/session0003.py`, phase별 source snapshot 및 각 행의 NPZ가 대응한다. 모든 열 식은 -(kT_x)_x=Q, 이류 식은 u_t+v*u_x=k*u_xx다.","",
               "## 실제 기록",f"해당 Cycle의 결과 {len(subset)}행. 표의 상대L2는 전체 배열 norm이며 batch의 mean-per-sample 오차는 JSON 별도 항목이다.","",
               "| 후보 | 조건 | 상대L2 | 최대 절대오차 | 중앙시간(s) | 추적 peak(byte) |",
               "|---|---|---:|---:|---:|---:|"]
        for r in subset:
            lines.append(f"| {r['candidate']} | {r['case']} | {r['relative_l2']:.7g} | {r['max_absolute_error']:.7g} | {r['median_seconds']:.7g} | {r['tracemalloc_peak_bytes']} |")
        lines += ["", "물리 flux/balance/BC, batch sample mean/std와 input/prediction/reference는 `results/session_0003/metrics.json` 및 `arrays/`에 보존했다. 시간은7회 CPU 반복 중앙값이다. A/C base inverse 준비는 별도 profile에 있으며 native/BLAS 일부 메모리는 tracemalloc에 잡히지 않는다. RSS는 누적 process high-water다.","",
                  "## 판정",f"**{verdict} as an independently established invention.**",next_state,"",
                  "## 미확인 사항", "원문 사이트403, 새 공식 README3개와 이전 저장 코드만 확인했다. 정확한 수학적 연산 동일성은 별도로 증명했지만 논문/특허 전체 검색은 미완료다. 실행하지 않은 GPU/Stokes/대규모 데이터 결과를 주장하지 않는다.",""]
        write_new(REPORT/f"cycles/cycle_{cycle}.md","\n".join(lines))

    comparison=["| 평가 분포 | D transfer | O(N) banded | Ridge | MLP(3seed 평균±표준편차) | Frozen C006 |",
                "|---|---:|---:|---:|---:|---:|"]
    for condition in ["iid","low_k","new_source","new_material","new_boundary"]:
        values={r["candidate"]:r["mean_sample_relative_l2"] for r in models(condition) if r["candidate"]!="MLP"}
        mm,ss=mean_mlp(condition)
        comparison.append(f"| {condition} | {values['D_fitted_transfer']:.5g} | {values['banded_classical']:.5g} | {values['feature_ridge']:.5g} | {mm:.5g} ± {ss:.4g} | {values['C006_frozen']:.5g} |")
    modal=json.loads((RESULT/"c006_modal_audit.json").read_text())
    discrepancy=max(r["modal_prediction_max_difference"] for r in modal)
    low=find("C006_frozen","low_conductivity_ood_seed824")
    with np.load(RESULT/low["arrays"]) as data:
        first_mode_fraction=float(np.mean(data["modal_error_sq"][:,0]/data["modal_error_sq"].sum(1)))
    ridge=find("feature_ridge","new_source")["mean_sample_relative_l2"]
    bad=find("feature_linear","new_source")["mean_sample_relative_l2"]
    theta=json.loads((RESULT/"models/segment_theta.json").read_text())["theta"]
    timings=[]
    for name in ["D_fitted_transfer","banded_classical","feature_ridge","C006_frozen"]:
        row=find(name,"iid")
        timings.append(f"| {name} | {row['median_seconds']*1e3:.5g} | {row['tracemalloc_peak_bytes']} |")
    mlp_times=[r["median_seconds"] for r in models("iid") if r["candidate"]=="MLP"]
    timings.append(f"| MLP (3seed median) | {np.median(mlp_times)*1e3:.5g} | checkpoint별 JSON 참조 |")
    summary=f"""# Session 0003 — Physics Foundation Architecture Research

**새 계산 객체4종을6개 의미 있는 Cycle(13–18)에서 검토했으나, 신규 범용 architecture는 입증하지 못했다.** 기존 결과를 유지한 채 표현의 식별 가능성, 실제 연산 동일성, 방향성과 물리적 순서를 분리했다. 결과136행과 test34개 통과를 저장했다. 136은 서로 다른 조건/seed/control의 기록 수이며, 중복 평가와 과거 결과 재현을 독립 데이터로 세지 않는다.

## A. 이전 연구의 근본 문제

일곱 후보를 물리 불가능성/표현력/분포/재료·해상도/기존 수학과의 동일성/비용/객체 선택으로 재분류했다(`research/session_0003/initial_failure_audit.md`). 고정 선형 PDE의 출력이 Green response와 같다는 사실과 **계산 규칙 자체가 기존 알고리즘인 것**을 구별했다. 출력이 같다는 이유만으로 새 알고리즘을 배제해서도 안 된다.

C006의 실제 source-weighted spectral 오차를 재학습 없이 재계산했다. `sum[(1-lambda*g)^2*(Vᵀb/lambda)^2]`로 예측한 상대오차와 측정값의 차이는 최대 **{discrepancy:.5g}**다. Seed824 낮은k 오류의 첫 고유모드 기여율은 샘플별 평균 **{first_mode_fraction:.5%}**였다. 현재 학습된 finite-age 분포의 낮은 고유값 응답 부족이 관측 오류를 설명한다. **어떤 유한 지수합도 이 특정 유한 OOD 구간을 잘 맞출 수 없다는 증명은 아니다**. 전 양의 축의 균일 inverse 근사는 불가능하지만 compact 구간은 개선 가능하며, 지시대로 튜닝하지 않았다.

정확한1/k 관계도 검증했다. 고정 k(x) 모양 전체를 s로 곱하면 영 Dirichlet에서는 T/s다. 비영 BC에서는 `H_k g + T_source/s`이고 전체 T/s가 아니다. 다른 material 배열을 같은 harmonic scalar로 바꾸는 것은 유효하지 않았다.

## B. 이번 가설의 완전한 연산과 분류

| 후보 | 물리 원리와 객체 | 실제 규칙 | 학습/비용 | 판정 |
|---|---|---|---|---|
| A | finite perturbation, left/right response packet | `G⁺=G-(Ga)(cᵀG)/(1+cᵀGa)` | exact base inverse 공급, 학습 안 함; 준비N³, 갱신 rN² | Sherman–Morrison/Woodbury와 동일 |
| B | coarse balance witness | `w=Rb`, `u=(RA)^+w` | decoder 학습 전 null-source 반증; rankr closure | 불충분한 입력 표현; full closure는 기존 solve |
| C | 변환군의 변화량 | `a⋆b=a√(1+γ²b²)+b√(1+γ²a²)` | gamma4 고정, pairO(d); response 준비 별도 | asinh 좌표의 덧셈 재표현 |
| D | 실제 공간의 ordered relation | `(R,S,C)∘(R',S',C')=(R+R',S+S',C+C'+θR'S)` | θ를128label로 적합; prefixO(N) | affine transfer/prefix 알고리즘 |

상세 A–G 정의와 보장/미보장, 차원, 신경 연산과의 비교는 `architectures/session_0003/`에 있다. A/B/C를 배제한 뒤 독립 원리 D를 실제로 탐색했으며 이름만 변경하지 않았다. D의 theta는 **{theta:.17g}**로, 기존 단위 coefficient를 복원했다. Matrix 곱으로 표현 가능하다는 사실만으로 신규성이 사라지는 것은 아니다. 여기서는 전체 구간 입출력·합성·prefix 복원까지 알려진 transfer 절차와 같다는 구체적 이유로 배제했다.

## C. 실제 수치 검증

열전도 -(kT_x)_x=Q에서 N7/15/31, uniform/heterogeneous/low k(최소0.001), 국소 source와 새 위치, 비영 BC를 실행했다. 기준은 독립 banded/direct solve와 기존 analytic-reference 검증이다. 이류–확산은 N24, 속도±1, cell Pe0.2/2/20, periodic, dt0.02로10step; BE reference와 semidiscrete exact exponential의 시간 오차는 구분했다.

Variable-material 학습은128label, validation32, held-out5분포×32개다. 각 모델의 평가 입력 fingerprint가 같음을 확인했다. 아래는 샘플별 상대L2 평균이다. MLP만3seed 평균±표준편차다.

{chr(10).join(comparison)}

D/feature-linear/MLP는 동일128개 variable-material label을 사용한다. Ridge는 별도validation32개로alpha를 선택했다. 다만 정규화 기준선을 추가한 계기는 이미 관측한 test 실패이므로 이번 비교는 탐색적이다. 새로운 untouched confirmatory test set은 실행하지 않았다. Unit-compliance의 unit-k label128개는 **추가 label**이고 frozen C006은 **과거 다른 분포 pretraining**이므로 동일 데이터 예산의 순위로 묶지 않는다. 정확한 물리 prior를 가진 D를 MLP와 parameter수만으로 공정 비교했다고 주장하지 않는다.

CPU batch32개의 IID 추론 시간(7회 중앙값)과 추적 peak:

| 방법 | ms/batch32 | tracemalloc byte |
|---|---:|---:|
{chr(10).join(timings)}

D는 prefix feature, banded는 band 조립, C006은 matrix 조립/eigensetup, MLP는 normalization/forward를 포함한다. A/C base inverse 준비와 reference label 생성은 `preparation_profiles.json`에 별도 측정했다. Tracemalloc은 native/BLAS 일부를 누락하고 RSS는 누적 process high-water다. GPU speedup이나 production CAE 가속을 검증하지 않았다. 비용은 고전 O(N) solver가 이미 경쟁력 있음을 보여주며 신규 효과를 입증하지 않는다.

## D. 반증과 수정

- A: full update는 정확하지만 frozen packet은 물성 변화의 cross reaction을 누락한다. 분모 없는 tangent도 유한 변화에서 틀린다. 수정하면 기존 inverse-update 공식이다.
- B: 서로 다른 양수 source가 같은 coarse witness/norm을 갖고 해는 다르다. 어떤 decoder도 양쪽을 동시에 정확히 예측할 수 없다. Minimum-norm decoder의 나쁜 점수 전부를 최적 정보 하한으로 주장하지 않는다.
- C: raw 합성은 열 source superposition을 위반하고 낮은 k에서 커진다. 이류의 raw 합성 상대오차 **{find('C_raw','transient_superposition')['relative_l2']:.6g}**, 질량 위반 **{find('C_raw','transient_superposition')['mass_drift_abs']:.6g}**. 좌표 보정 후에는 보통 덧셈과 동등하다.
- D: 실제 공간의 ordered cross moment는 필요하지만 transfer matrix다. Total tuple3개로 전체 field를 압축한 것이 아니며 O(N) prefix 정보가 필요하다.
- 방향성: reciprocal shortcut은 질량 오차가 약1e-15여도 최대 **{max(r['relative_l2'] for r in rows if r['candidate']=='A_reciprocal_shortcut'):.5g}** 상대오차였다. Full left/right update는 올바른 방향과 reference를 재현했다.
- 나쁜 baseline 자기반박: unregularized feature-linear의 새 source 오류 **{bad:.6g}**는 tiny singular direction과 coefficient norm~1.67e6에 의존했다. Test label을 보지 않고 validation으로 고른 ridge(alpha1)의 오류는 **{ridge:.6g}**다. 기존 실패와 수정 모두 남겼으며 나쁜 기준선으로 우월성을 주장하지 않는다.

## E. 기존 모델과의 엄격한 차별성

A는 알려진 finite response update, B는 latent witness 제한과 projection, C는 좌표 재매개변수화, D는 기존 physical transfer에 scalar를 적합한 것이다. 어느 것도 이번에 독립 신규 primitive로 확립되지 않았다. 실제로 학습한 것은 D scalar의 least squares, feature-linear/ridge/compliance 회귀와 PyTorch MLP3개다. A/B/C나 C006을 새로 학습했다고 보고하지 않는다.

## F. 현재 가장 유망한 architecture

**선정 없음.** D가 정확하다고 기존 알고리즘을 최종 발명으로 채택하지 않는다. 원리적으로 틀린 후보를 대규모 학습으로 구제하지 않았고, 새 primitive로 남은 것이 없어 Stokes 확장도 실행하지 않았다. 큰 GPU/유료 자원/새 데이터 도메인은 미실행이다. CPU에서 가능한6개 Cycle의 핵심 반증·비교가 끝나 불필요한 나머지 Cycle을 채우지 않고 중단했다.

## G. 논문·특허

arXiv/JMLR 원문은 여전히403이었다. CVXPYlayers/OptNet/DEQ 공식 README를 새로 실제 읽었고 metadata와 displayed optimization/fixed-point 수식을 확인했다. 저장된 이전 NeuralOperator/HNN/LNN/graph 공식 자료도 참고했다. Full paper 및 선행특허는 읽거나 검색한 것으로 주장하지 않는다. 신규성은 **미확인**, 발명/진보성 판정은 없다. 구체적인 계산 동일성의 증명은 원문 접근과 별개로 수행했다.

## H. 다음 세션

단일-valued 선형 map에 임의 nonlinear 합성을 넣는 대신, **여러 해 branch와 안정성 정보가 필요한 열/유동의 relation**을 계산 객체로 검토한다. 예를 들어 Bratu 열반응 문제는 새 benchmark 질문이며 이번에 실행하지 않았다. 여러 branch가 모두 안정하다고 가정하지 않으며 안정성은 별도 검증한다. 새 primitive를 정하지 않은 채 polynomial elimination, mixture, implicit solver에 이름만 붙이지 않는다. 원문·정보 충분성·알고리즘 동일성부터 점검하고 작은1D 열 문제와 방향성 gate를 다시 통과시킨다. 정확한 명령과 보존 규칙은 `research/session_0003/next_steps.md`에 저장했다.

기존 Session0002 결과 **232개 파일과 기존 source hash를 보존**했다. Test34개 통과, 저장 오차 재계산, checkpoint20회 예측 재현, banded reference1,786회 대조 및6개 signed 방향 검증을 완료했다. 이1,786회는 같은 입력의 중복 method 평가를 포함하며 독립 data 수가 아니다. 코드·그림·무결성 검증은 `results/session_0003/validation/`에 있다. 이번 세션의 GitHub push, environment publication, 대규모 삭제는 하지 않았다.
"""
    write_new(REPORT/"research_summary.md",summary)
    write_new(REPORT/"current_best.md","""# Session 0003 현재 판정

신규 architecture 후보: **없음**. A는 Sherman–Morrison, B는 불충분 witness 또는 기존 closure, C는 덧셈의 좌표 변환, D는 transfer/prefix 알고리즘으로 판정했다. 물리 성공과 독립 계산 신규성을 구별한다.

정확한 practical control은 O(N) banded/transfer다. 이는 발명이 아니다. C006은 재학습하지 않았고, 낮은k 실패를 modal 식으로 재현했다. MLP/선형/ridge는 비교용이며 foundation architecture로 승격하지 않는다.

다음에는 실제 물리의 multi-branch relation/안정성 정보를 객체로 삼을지 먼저 판단한다. 아직 구현한 primitive나 성공 결과는 없다. 새 세션은 `research/session_0003/next_steps.md`와 후보/실패 기록에서 재개한다.
""")
    # Two scientific figures from saved arrays/metrics, no synthetic performance values.
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for condition,label in [("iid","old IID"),("low_conductivity_ood","old low-k OOD")]:
        row=find("C006_frozen",condition+"_seed824")
        with np.load(RESULT/row["arrays"]) as data:
            lam=data["eigenvalues"].ravel();ratio=data["inverse_response_ratio"].ravel()
            order=np.argsort(lam)
            axes[0].semilogx(lam[order],ratio[order],".",ms=2,label=label)
            if condition=="low_conductivity_ood":
                fraction=data["modal_error_sq"]/data["modal_error_sq"].sum(1,keepdims=True)
                axes[1].bar(np.arange(1,16),fraction.mean(0))
    axes[0].axhline(1,color="black",ls="--");axes[0].legend()
    axes[0].set(xlabel="physical eigenvalue",ylabel="lambda * frozen g(lambda)")
    axes[1].set(xlabel="eigenmode",ylabel="mean fraction of low-k error energy")
    fig.tight_layout();fig.savefig(RESULT/"plots/c006_modal_audit.png",dpi=170);plt.close(fig)
    conditions=["iid","low_k","new_source","new_material","new_boundary"]
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for name in ["feature_ridge","C006_frozen","unit_compliance"]:
        axes[0].plot(np.arange(5),[find(name,c)["mean_sample_relative_l2"] for c in conditions],"o-",label=name)
    means,stdev=zip(*(mean_mlp(c) for c in conditions))
    axes[0].errorbar(np.arange(5),means,yerr=stdev,fmt="o-",label="MLP 3 seeds")
    axes[0].set_xticks(np.arange(5),["IID","low k","new source","new k(x)","new BC"],rotation=15)
    axes[0].set_yscale("log");axes[0].set_ylabel("mean sample relative L2")
    axes[0].set_title("Priors/data budgets differ; no novelty ranking")
    handles,legend_labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,legend_labels,loc="lower center",ncol=4,fontsize=8)
    names=["D_fitted_transfer","banded_classical","feature_ridge","C006_frozen"]
    axes[1].bar(np.arange(4),[find(name,"iid")["median_seconds"]*1e3 for name in names])
    axes[1].set_xticks(np.arange(4),["transfer","banded","ridge","C006+eig"],rotation=15)
    axes[1].set_ylabel("ms / batch32");axes[1].set_title("CPU, full recorded inference scopes")
    fig.tight_layout(rect=(0,.08,1,1));fig.savefig(RESULT/"plots/controls_and_cost.png",dpi=170);plt.close(fig)
    image_checks={}
    for path in (RESULT/"plots").glob("*.png"):
        with Image.open(path) as im:im.verify()
        image_checks[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    (RESULT/"validation/figures.json").write_text(json.dumps(image_checks,indent=2)+"\n")
    if FIGURES_ONLY:
        print(f"Regenerated {len(image_checks)} verified figures; existing reports preserved.")
    else:
        print(f"Wrote 6 cycle reports, Korean A-H report, current state and {len(image_checks)} verified figures.")


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--figures-only",action="store_true")
    FIGURES_ONLY=parser.parse_args().figures_only
    main()
