# Session 0004 사전 계획 — Cycle 19부터

이 문서는 새 모델 구현·학습·확인용 평가 전에 작성했다. 이전 결과는 수정하지 않는다. 현재 입증한 신규 범용 architecture는 없다. 기존 수학 사용은 배제 근거가 아니며 알고리즘·학습·정보 흐름과 실제 효과를 평가한다.

## 새 신규성 기준

Level 1: 기존 알고리즘 그대로의 실행(채택 안 함). Level 2: 새로운 학습 표현과 입증할 장점. Level 3: 새로운 학습 가능한 객체 상호작용/정보 흐름과 입증할 효과(중심 목표). Level 4: 새 수학/이론(필수 아님). 구현 차이, 유용성, 문헌상 신규성은 별개의 판정이다. 모두 미리 신규성 미확인이다.

## 독립 가설

- E: saddle-node에서 두 field를 독립 mixture head 대신 중심 field와 분리 방향·signed gap으로 공동 생성한다. 정상해 집합 표현이며 초기조건 선택기는 아니다. 학습 fold와 field 사이의 식별 가능성이 반증 대상이다.
- F: 서로 다른 finite source edit의 단독 효과와 교차효과를 구분해 학습한다. 일차 Jacobian/Green update가 아니라 두 유한 변화의 inclusion–exclusion 항을 전달한다. Pairwise polynomial/Volterra/DeepSets와 비교하고, 이것만으로 신규라고 부르지 않는다.
- G: full physical input에서 온도와 flux를 함께 생성한다. 각 층의 온도에 따른 reaction source를 flux balance에 넣고, 구성식 mismatch가 다음 latent state의 multiplicative gate에 직접 들어간다. 출력 후 projection이나 penalty만의 모델이 아니다. Learned iterative/GNN/gated network와 실질적으로 얼마나 다른지 ablation한다.

## 분할과 종료 기준

모든 실험 CPU, float64, torch thread1. 기존 source/결과 보존. Training seeds 1004/1005/1006. Validation만 checkpoint를 선택한다. 확인 집합은 아래 seed/범위로 미리 정하고 **모든 모델 checkpoint 동결 후 한 번만** 평가한다. 확인 결과 뒤 모델 수정 시 새 독립 검증으로 주장하지 않는다.

E: Bratu 양 branch의 해석해를 scalar brentq reference로 생성한다. Train lambda=0.35–3.2 균일48개, validation 0.4–3.4의 별도32개; x는15 interior점. 2000 full-batch Adam steps, lr0.003. 확인 lambda: IID uniform(0.35,3.2)32개(seed41001), near fold linspace(3.45,3.513,24), no-solution linspace(3.53,3.9,16). Resolution63도 고정한다. Paired model, branch-conditioned MLP, unordered two-head MLP, single-output mean MLP를 같은 label로 비교한다. 알려진 fold를 공급하는 oracle-normal-form control은 별도 prior를 표시한다. 각 branch 오차와 집합 거리, missing/spurious branch, 안정성은 분리한다.

F: 작은 nonlinear heat -T''=lambda exp(T)+a s1+b s2, 영 Dirichlet, N15. lambda=1.5 고정, s1/s2는 center0.3/0.7, width0.1 Gaussian, amplitude a,b∈[-0.5,2.0]. Train128(seed42001), validation32(seed42002). 1500 Adam steps, lr0.003. 모든 방법은 같은 4-state quartet label(T00,T10,T01,T11)을 받는다. Pair-interaction model, additive-only ablation, direct MLP를 비교한다. 확인 seed42003 IID32개, larger-edit a,b∈[2,3]16개, changed-source centers0.2/0.6 16개. Root failure는 숨기거나 성공 sample만 조용히 선택하지 않는다. 정확한 reference 비용 및 quartet 4배 label 비용을 포함한다.

G: N15, variable k/source/Dirichlet에서 full-state/flux co-generation. Train128(seed43001), validation32(seed43002), thermal reaction lambda=0 및1.5를 따로 label한다. Train1500 Adam steps, lr0.003; learned gated feedback, feedback 없는 동일 joint decoder, direct MLP를 비교한다. 확인 seed43003: iid32, low k(0.15–0.3)16 linear-only, shifted source16, changed BC16. 임의 비선형 문제에서 steady state 존재를 보장하지 않는다. 보존은 생성 flux에 대한 것이며 constitutive flux 또는 정확한 온도를 보장하지 않는다. 고정 N 모델의 해상도 일반화는 주장하지 않는다.

각 후보의 물리적 실패는 scale을 키워 구제하지 않는다. E/F/G의 개발 결과를 검토한 후 survivor만 방향성/비선형 유동으로 확장한다. 예비 자원 상한6–8 의미 있는 Cycle, 최대12 또는90분. 의미 없는 수 채우기는 하지 않는다.

## 지표와 비용

Actual arrays: physical inputs, predictions, references, residuals and source. Relative L2, max absolute error, boundary error, source/flux energy balance, branch existence, eigen stability. 시간은 label 준비/학습/추론(전처리 포함)을 나눠 저장한다. Python tracemalloc peak는 native/BLAS 누락 한계를 명시한다. GPU 속도 미측정. 선형 banded solver 및 nonlinear root solver를 강한 기준선으로 둔다.

원문 접근은 기존403 기록을 재사용하고 같은 요청을 반복하지 않는다. 저장된 공식 source와 직접 유도한 식을 구별한다. 새 외부 전송/Push/유료 자원/GPU는 수행하지 않는다.
