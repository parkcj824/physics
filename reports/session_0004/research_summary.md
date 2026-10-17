# Physics Foundation Architecture Research — Session 0004

**독립 후보3개(E/F/G), 명시적 수정 E-R, G의 방향성 확장을 Cycle19–23의5개 Cycle에서 검증했다. 새로운 범용 Architecture/Level3 신규성을 입증하지 못했다.** 기존 수학을 사용했다는 이유로 배제하지 않고, 표현·학습 메커니즘·효과·선행연구를 구분했다. 후속 capacity 통제까지 실제 학습 모델51개(3seed씩), 확인 평가183행, pytest 42개 통과, 이전 추적 파일 1154개 보존을 검증했다. 183행은 method/seed/case별 예측 batch이며 독립 dataset 수가 아니다.

## A. 새로 정의한 계산 문제

단일 고정 선형 map 대신 (1) 가능한 정상해 집합과 동적 선택의 분리, (2) finite source edit의 비선형 교차응답, (3) 상태와 수지의 공동 생성을 연구했다. Bratu는 중간 검증 문제이며 모든 물리에 branch 객체가 필요하다는 주장이 아니다. Architecture 재사용과 도메인 간 가중치 공유도 구별한다.

Bratu 해는 `T_a=-2log(cosh[a(x-1/2)]/cosh(a/2))`, `lambda(a)=2a²/cosh²(a/2)`이다. 직접 PDE 대입, scalar root, 서로 다른 solve_bvp 기준선으로 검증했다. Fold a=2.39935728052, lambda=3.51383071913, fold condition 오차 2.22e-16. 독립 BVP 최대 온도 오차는 1.587e-12다.

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

| 모델 | 조건 | mean_relative_l2 | missing_branches_at_5percent | spurious_branches_at_5percent |
|---|---|---|---|---|
| conditioned | fold | 0.23341 ± 0.0162 | 48 ± 0 | 48 ± 0 |
| conditioned | iid | 0.037683 ± 0.00751 | 16 ± 4.32 | 16 ± 4.32 |
| conditioned | resolution | 0.037727 ± 0.0075 | 16.333 ± 3.86 | 16.333 ± 3.86 |
| mean | fold | 0.12627 ± 0.00076 | 44.333 ± 0.471 | 43.333 ± 0.943 |
| mean | iid | 4.1566 ± 0.000639 | 64 ± 0 | 64 ± 0 |
| mean | resolution | 4.1566 ± 0.000639 | 64 ± 0 | 64 ± 0 |
| paired | fold | 0.13747 ± 0.00631 | 48 ± 0 | 48 ± 0 |
| paired | iid | 0.018934 ± 0.000873 | 4 ± 1.63 | 4 ± 1.63 |
| paired | resolution | 0.018946 ± 0.000869 | 4 ± 1.63 | 4 ± 1.63 |
| paired_oracle | fold | 0.010664 ± 0.00271 | 0 ± 0 | 0 ± 0 |
| paired_oracle | iid | 0.016249 ± 0.000227 | 2 ± 0 | 2 ± 0 |
| paired_oracle | resolution | 0.016262 ± 0.00023 | 2 ± 0 | 2 ± 0 |
| two_head | fold | 0.20388 ± 0.0145 | 48 ± 0 | 48 ± 0 |
| two_head | iid | 0.025375 ± 0.00866 | 6.6667 ± 4.11 | 6.6667 ± 4.11 |
| two_head | resolution | 0.025402 ± 0.00866 | 6.6667 ± 4.11 | 6.6667 ± 4.11 |

| 모델 | 없는16조건에서 선언한 허위 branch수,3seed 평균 |
|---|---:|
| conditioned | 32 |
| mean | 32 |
| paired | 9.3333 |
| paired_oracle | 0 |
| two_head | 32 |

E의 학습 fold값은 3.6490318, 3.6445121, 3.6286413다. Paired-oracle은 정확한 fold를 공급했으므로 no-branch 개선을 데이터만의 학습 효과로 해석하지 않는다. Single mean control의 collapse는 강한 mixture 모델에 대한 우월성 증거가 아니다. 안정성은 N255에 query한 field의 linearized eigenvalue로 별도 평가했다.

F는 `-T''=1.5 expT + a s1+b s2`, N15, Gaussians width.1, 영BC다. 128개 edit pair마다4개 solution을 사용해 **512개 field label record**, validation128field, 확인256field다. 같은 T00가 반복되어 학습 target의 서로 다른 hash는385개다. Base를 cache하면 준비 solve수를 줄일 수 있으며512를 독립 field 수라고 주장하지 않는다. 모든 F모델이 같은 quartet/loss를 받았다. Larger edits/변경 source 위치는 사전 정의한 OOD다.

| 모델 | 조건 | mean_relative_l2 | mean_cross_relative_l2 | mean_pde_residual_inf |
|---|---|---|---|---|
| additive | iid | 0.006935 ± 0.000302 | 1 ± 1.87e-14 | 0.54678 ± 0.00707 |
| additive | large | 0.015361 ± 0.00139 | 1 ± 9.61e-16 | 1.2835 ± 0.166 |
| additive | source | 0.035926 ± 0.00488 | 1 ± 4.76e-13 | 0.69427 ± 0.0748 |
| direct | iid | 0.016959 ± 0.00604 | 0.10987 ± 0.0439 | 0.76381 ± 0.0252 |
| direct | large | 0.04653 ± 0.0121 | 0.097352 ± 0.0193 | 1.3672 ± 0.0488 |
| direct | source | 0.041398 ± 0.00493 | 0.2059 ± 0.0403 | 0.76894 ± 0.0874 |
| direct_matched | iid | 0.018569 ± 0.0111 | 0.066185 ± 0.0111 | 0.73569 ± 0.0195 |
| direct_matched | large | 0.034191 ± 0.00477 | 0.11578 ± 0.037 | 1.2702 ± 0.035 |
| direct_matched | source | 0.048032 ± 0.0101 | 0.37143 ± 0.0881 | 0.69908 ± 0.0384 |
| pair | iid | 0.0063047 ± 0.000947 | 0.022877 ± 0.00397 | 0.50653 ± 0.0501 |
| pair | large | 0.015174 ± 0.0019 | 0.046411 ± 0.0342 | 1.3922 ± 0.242 |
| pair | source | 0.034924 ± 0.0026 | 0.13394 ± 0.0386 | 0.69043 ± 0.0406 |

F pair의 IID 온도 오차0.630%와 additive0.693%는 근접한다. 더 분명한 이점은 mixed-response 오차2.29% 대 parameter-budget MLP6.62%다. 물리학적 상호작용을 분리하는 표현의 잠정 효과이며 일반적인 온도 정확도 우월성을 과장하지 않는다. 세 seed는 같은 dataset에서 초기화만 달리한 것이고 통계적 유의성/데이터 효율 curve를 입증하지 않았다.

G는 균일/부드러운 비균질 k, 국소+균일source, BC 및 reaction0/1.5를 혼합한128label+validation32다. 확인 IID32/lowk16/변경 source16/BC16이다. Lowk와 변경BC 확인은 lambda0이다. 표에는 다른 물리 regime을 섞은 평균이 있으므로 세부 입력은 저장 배열이 기준이다.

| 모델 | 조건 | mean_relative_l2 | mean_pde_residual_inf | max_generated_energy_balance_abs |
|---|---|---|---|---|
| direct | boundary | 0.082783 ± 0.00996 | 4.0666 ± 0.419 | 0.63819 ± 0.075 |
| direct | iid | 0.079218 ± 0.0065 | 1.5745 ± 0.122 | 0.25756 ± 0.0528 |
| direct | lowk | 0.93961 ± 0.0839 | 1.8547 ± 0.686 | 0.81791 ± 0.302 |
| direct | source | 0.18199 ± 0.0409 | 3.8859 ± 0.496 | 0.75686 ± 0.177 |
| feedback | boundary | 0.03241 ± 0.00836 | 15.419 ± 2.86 | 4.1633e-17 ± 4.94e-17 |
| feedback | iid | 0.074527 ± 0.0146 | 5.0012 ± 0.734 | 7.8641e-17 ± 1.01e-16 |
| feedback | lowk | 0.69344 ± 0.0112 | 5.7934 ± 1.3 | 2.3997e-17 ± 2.28e-17 |
| feedback | source | 0.14461 ± 0.0197 | 11.292 ± 3.5 | 3.7007e-17 ± 5.23e-17 |
| no_feedback | boundary | 0.096428 ± 0.00894 | 5.0257 ± 0.355 | 3.2382e-17 ± 6.54e-18 |
| no_feedback | iid | 0.12747 ± 0.0228 | 2.0178 ± 0.101 | 8.3267e-17 ± 9.88e-17 |
| no_feedback | lowk | 0.82139 ± 0.0308 | 1.5264 ± 0.277 | 3.0069e-17 ± 1.82e-17 |
| no_feedback | source | 0.2927 ± 0.0523 | 4.2218 ± 1.2 | 9.2519e-17 ± 9.44e-17 |

BC는 세 후보에서 지정 envelope로 정확히 주입한다. 이는 학습 성취가 아니라 architectural prior다. G feedback의 생성flux 전역수지 최대 위반은 2.22e-16였지만 구성식 flux와 온도 PDE는 별도 지표이며 작은 generated balance가 정답을 보장하지 않는다. F/E는 구성식 flux 수지를 검사했고 E의 coarse residual에는 해석해의 이산오차도 포함되므로 continuum/이산 오차를 혼동하지 않는다.

특히 G feedback의 IID 온도 오차7.45%는 direct7.92%/no-feedback12.75%보다 작지만 PDE잔차 평균5.00은 direct1.57/no-feedback2.02보다 크다. Low-k 오차69.34%도 남았다. 작은 field L2와 정확한 생성flux 수지만으로 물리 통과를 선언하지 않는다. 과도한 curvature/구성식 오차가 다음 수정 대상이며 추가 GPU학습으로 구제하지 않았다.

### Cycle23 — 같은 G block의 정상 이류–확산 방향성

개발 validation을 근거로 사전에 확장을 결정했다. `(v u-nu u_x)_x=S`, N15, upwind constitutive flux를 사용한다. Source누적/상태 생성/3-layer gate와학습규칙은 유지하고 구성식만 바꿨으며 weights는 별도로 학습했다. 128train/32validation, 확인 IID32/방향쌍16×2/낮은확산16이다.

| 모델 | 조건 | mean_relative_l2 | mean_pde_residual_inf | max_generated_energy_balance_abs |
|---|---|---|---|---|
| direct | direction | 0.27644 ± 0.0404 | 0.69157 ± 0.272 | 0.065746 ± 0.0309 |
| direct | iid | 0.43682 ± 0.0719 | 1.0632 ± 0.378 | 0.095804 ± 0.0447 |
| direct | lowdiff | 0.7637 ± 0.0605 | 1.5716 ± 0.159 | 0.17463 ± 0.0499 |
| feedback | direction | 0.093155 ± 0.0383 | 0.51646 ± 0.0617 | 0 ± 0 |
| feedback | iid | 0.14599 ± 0.0251 | 0.7769 ± 0.0592 | 1.1102e-16 ± 0 |
| feedback | lowdiff | 0.33939 ± 0.039 | 1.2322 ± 0.069 | 5.5511e-17 ± 0 |
| no_feedback | direction | 0.23366 ± 0.0374 | 0.66284 ± 0.0896 | 2.7756e-17 ± 0 |
| no_feedback | iid | 0.39982 ± 0.084 | 0.9195 ± 0.0197 | 1.1102e-16 ± 0 |
| no_feedback | lowdiff | 0.94399 ± 0.155 | 1.8833 ± 0.312 | 4.1633e-17 ± 1.13e-17 |

방향 centroid는 feedback뿐 아니라 모든 비교 모델이16/16쌍에서 맞혔다. 따라서 방향을 구별한다는 사실 자체가 새 gate의 독점 장점은 아니다. Feedback IID 상대오차14.60%는 no-feedback39.98%/direct43.68%보다 낮았지만, low-diffusion33.94% 및 PDE잔차가 남았다. 생성 flux수지는 약1e-16 수준이다. 실제 upwind/banded reference는 같은 데이터에서 roundoff 수준의 온도·수지 오차를 낸다. 이 결과는 **정상 scalar transport**이며 시간 안정성/positivity 또는 비압축성Navier–Stokes 성공이 아니다.

저장한 방향성 예측의 사후 진단에서 **positivity 위반도 확인**했다. 양의 source/영BC에서 reference 최솟값은9.436e-5인데 feedback 예측의 최솟값은 seed별 -0.02998/-0.0001012/-0.004914였다. 32개 조건 중 음수 예측을 낸 sample은 각각10/1/10개다. 모델을 재학습하지 않고 `transport_positivity_audit.json`에 기록했다. 일반적인 signed 온도편차와 이 nonnegative-source benchmark의 최대원리 요구를 혼동하지 않는다.

### 실제 비용과 데이터

| family/model | IID batch32 ms | parameter수 | 실제 학습 seconds,3seed 합 |
|---|---:|---:|---:|
| E/conditioned | 0.60285 | 1217 | 22.504 |
| E/mean | 0.37031 | 1185 | 20.745 |
| E/paired | 0.26695 | 1219 | 20.429 |
| E/paired_oracle | 0.45365 | 1219 | 21.15 |
| E/two_head | 0.33287 | 1218 | 18.627 |
| F/additive | 0.52949 | 1890 | 91.94 |
| F/direct | 0.62224 | 2689 | 129.02 |
| F/direct_matched | 1.0015 | 4609 | 204.26 |
| F/pair | 1.0504 | 4010 | 143.51 |
| G/direct | 0.42929 | 9439 | 18.51 |
| G/feedback | 0.8709 | 8704 | 43.351 |
| G/no_feedback | 0.61544 | 8704 | 22.27 |
| H/direct | 0.4249 | 9439 | 24.1 |
| H/feedback | 1.1601 | 8704 | 49.398 |
| H/no_feedback | 0.67227 | 8704 | 25.342 |
| E/classical | 3.3421 | 0 learned | label/control 비용 별도 |
| F/classical | 7.8688 | 0 learned | label/control 비용 별도 |
| G/classical | 5.4392 | 0 learned | label/control 비용 별도 |
| H/classical upwind | 1.7385 | 0 learned | label/control 비용 별도 |
| G/direct_width64 (fresh IID) | 0.37898 | 14623 (active8415) | 46.356 |
| H/direct_width64 (fresh IID) | 0.68719 | 14623 (active8415) | 49.429 |

CPU Python 3.12.14, torch 2.8.0+cpu, thread1. 실제 학습 측정시간 합 950.94초(트레이싱 overhead 포함), label/reference 준비는 dataset_manifest에 별도 기록했다. F 추론은 최종 field 하나, G는 T와flux, E는 두branch를 비교한다. Tensor변환/정규화 또는 source/matrix조립 포함, disk I/O와 N255 안정성 진단 제외. Classical solver는 reference 허용오차까지 수렴하고 neural은 근사해이므로 같은 오차 허용의 speedup을 입증한 비교가 아니다. 알려진 해석적 parameterization이 있는 Bratu 시간은 일반 CAE 속도로 확장할 수 없다.

Tracemalloc은 native/BLAS/PyTorch allocations를 일부 누락한다. 모델별 실제 peak 전체메모리나 GPU속도는 미측정이다. G/H no-feedback에는 사용하지 않는 feedback parameter도 있어 총 parameter수만으로 공정성/유효계산량이 같다고 주장하지 않는다. Feedback는 활성 parameter/compute가 더 많으므로 이 실험만으로 효과의 원인을 고유한 물리 mechanism으로 확정할 수 없다. F의 width64 MLP도 parameter/time을 함께 제시하며 완전한 architecture HPO sweep은 아니다.

대표 train입력의 gradient 경로를 검사한 결과 G/H에서 실제 gradient를 받은 parameter수는 feedback8704, no-feedback4816, direct5551이었다. 이 진단은 weight를 갱신하지 않았고 `parameter_accounting.json`에 저장했다. 이를 발견한 뒤 별도 사전 계획과 **새 IID 확인 집합**으로 active8415/width64/3000step MLP 통제를 추가했다. 원래 후보 checkpoint와 확인 결과는 바꾸지 않았다. 추가 학습6개·새 평가24행을 원래45개·159행과 구분한다.

새 확인에서 G의 온도 오차는 feedback7.37% 대 matched-control8.09%였지만 PDE잔차는4.89 대1.37로 여전히 불리했다. 방향성 H의 온도 오차는8.31% 대23.95%, PDE잔차는0.637 대0.644였다. 후속 통제에서도 제한된 표현 이득이 남았으나 모든물리지표의 우월성이나 신규성을 입증하지 않았다. 전체seed표·실제시간은 `capacity_control.md`, 새사전계획은 `research/session_0004/capacity_control_protocol.md`에 있다. 최초확인집합을다시독립검증이라부르지않는다.

## E. 반증 및 수정

1. E의 `g`를 다시 scale하면 fold3.6/4/6으로도 같은 training field를 최대약7e-16오차로 재현할 수 있었다. Branch label만으로 existence boundary가 식별되지 않는다. 정확한 fold를 제공한 E-R이 이 결함을 제거해도 이는 추가 물리정보를 사용한 결과다.
2. 두 정상해를 평균한 field는 lambda2에서 상대 PDE잔차 0.7175, 에너지수지 오류 3.4438였다. 단일 평균은 물리적 branch 선택이 아니다.
3. F의 exact pair truncate도 triple response를 누락한다. lambda0 삼중차이 norm 4.053e-17, lambda1.5에서는 9.52e-05였다. 이번 작은 counterexample의 상대오차가 작다고 arbitrary edit수에서 오차가 작다는 보장은 없다.
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
