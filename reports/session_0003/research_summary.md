# Session 0003 — Physics Foundation Architecture Research

**새 계산 객체4종을6개 의미 있는 Cycle(13–18)에서 검토했으나, 신규 범용 architecture는 입증하지 못했다.** 기존 결과를 유지한 채 표현의 식별 가능성, 실제 연산 동일성, 방향성과 물리적 순서를 분리했다. 결과136행과 test34개 통과를 저장했다. 136은 서로 다른 조건/seed/control의 기록 수이며, 중복 평가와 과거 결과 재현을 독립 데이터로 세지 않는다.

## A. 이전 연구의 근본 문제

일곱 후보를 물리 불가능성/표현력/분포/재료·해상도/기존 수학과의 동일성/비용/객체 선택으로 재분류했다(`research/session_0003/initial_failure_audit.md`). 고정 선형 PDE의 출력이 Green response와 같다는 사실과 **계산 규칙 자체가 기존 알고리즘인 것**을 구별했다. 출력이 같다는 이유만으로 새 알고리즘을 배제해서도 안 된다.

C006의 실제 source-weighted spectral 오차를 재학습 없이 재계산했다. `sum[(1-lambda*g)^2*(Vᵀb/lambda)^2]`로 예측한 상대오차와 측정값의 차이는 최대 **4.4409e-16**다. Seed824 낮은k 오류의 첫 고유모드 기여율은 샘플별 평균 **91.99855%**였다. 현재 학습된 finite-age 분포의 낮은 고유값 응답 부족이 관측 오류를 설명한다. **어떤 유한 지수합도 이 특정 유한 OOD 구간을 잘 맞출 수 없다는 증명은 아니다**. 전 양의 축의 균일 inverse 근사는 불가능하지만 compact 구간은 개선 가능하며, 지시대로 튜닝하지 않았다.

정확한1/k 관계도 검증했다. 고정 k(x) 모양 전체를 s로 곱하면 영 Dirichlet에서는 T/s다. 비영 BC에서는 `H_k g + T_source/s`이고 전체 T/s가 아니다. 다른 material 배열을 같은 harmonic scalar로 바꾸는 것은 유효하지 않았다.

## B. 이번 가설의 완전한 연산과 분류

| 후보 | 물리 원리와 객체 | 실제 규칙 | 학습/비용 | 판정 |
|---|---|---|---|---|
| A | finite perturbation, left/right response packet | `G⁺=G-(Ga)(cᵀG)/(1+cᵀGa)` | exact base inverse 공급, 학습 안 함; 준비N³, 갱신 rN² | Sherman–Morrison/Woodbury와 동일 |
| B | coarse balance witness | `w=Rb`, `u=(RA)^+w` | decoder 학습 전 null-source 반증; rankr closure | 불충분한 입력 표현; full closure는 기존 solve |
| C | 변환군의 변화량 | `a⋆b=a√(1+γ²b²)+b√(1+γ²a²)` | gamma4 고정, pairO(d); response 준비 별도 | asinh 좌표의 덧셈 재표현 |
| D | 실제 공간의 ordered relation | `(R,S,C)∘(R',S',C')=(R+R',S+S',C+C'+θR'S)` | θ를128label로 적합; prefixO(N) | affine transfer/prefix 알고리즘 |

상세 A–G 정의와 보장/미보장, 차원, 신경 연산과의 비교는 `architectures/session_0003/`에 있다. A/B/C를 배제한 뒤 독립 원리 D를 실제로 탐색했으며 이름만 변경하지 않았다. D의 theta는 **1.0000000000000002**로, 기존 단위 coefficient를 복원했다. Matrix 곱으로 표현 가능하다는 사실만으로 신규성이 사라지는 것은 아니다. 여기서는 전체 구간 입출력·합성·prefix 복원까지 알려진 transfer 절차와 같다는 구체적 이유로 배제했다.

## C. 실제 수치 검증

열전도 -(kT_x)_x=Q에서 N7/15/31, uniform/heterogeneous/low k(최소0.001), 국소 source와 새 위치, 비영 BC를 실행했다. 기준은 독립 banded/direct solve와 기존 analytic-reference 검증이다. 이류–확산은 N24, 속도±1, cell Pe0.2/2/20, periodic, dt0.02로10step; BE reference와 semidiscrete exact exponential의 시간 오차는 구분했다.

Variable-material 학습은128label, validation32, held-out5분포×32개다. 각 모델의 평가 입력 fingerprint가 같음을 확인했다. 아래는 샘플별 상대L2 평균이다. MLP만3seed 평균±표준편차다.

| 평가 분포 | D transfer | O(N) banded | Ridge | MLP(3seed 평균±표준편차) | Frozen C006 |
|---|---:|---:|---:|---:|---:|
| iid | 9.0914e-16 | 1.0074e-15 | 0.21795 | 0.16758 ± 0.02183 | 0.011828 |
| low_k | 1.6281e-15 | 1.336e-15 | 1.003 | 1.0246 ± 0.001128 | 0.77521 |
| new_source | 1.2243e-15 | 1.0926e-15 | 0.2143 | 0.15105 ± 0.01686 | 0.011951 |
| new_material | 1.3102e-15 | 1.3418e-15 | 0.77525 | 0.74017 ± 0.01652 | 0.060217 |
| new_boundary | 1.074e-15 | 1.152e-15 | 0.099586 | 0.27474 ± 0.004835 | 0.011218 |

D/feature-linear/MLP는 동일128개 variable-material label을 사용한다. Ridge는 별도validation32개로alpha를 선택했다. 다만 정규화 기준선을 추가한 계기는 이미 관측한 test 실패이므로 이번 비교는 탐색적이다. 새로운 untouched confirmatory test set은 실행하지 않았다. Unit-compliance의 unit-k label128개는 **추가 label**이고 frozen C006은 **과거 다른 분포 pretraining**이므로 동일 데이터 예산의 순위로 묶지 않는다. 정확한 물리 prior를 가진 D를 MLP와 parameter수만으로 공정 비교했다고 주장하지 않는다.

CPU batch32개의 IID 추론 시간(7회 중앙값)과 추적 peak:

| 방법 | ms/batch32 | tracemalloc byte |
|---|---:|---:|
| D_fitted_transfer | 1.3619 | 12867 |
| banded_classical | 1.3842 | 12640 |
| feature_ridge | 0.020174 | 26496 |
| C006_frozen | 5.1641 | 25304 |
| MLP (3seed median) | 0.055205 | checkpoint별 JSON 참조 |

D는 prefix feature, banded는 band 조립, C006은 matrix 조립/eigensetup, MLP는 normalization/forward를 포함한다. A/C base inverse 준비와 reference label 생성은 `preparation_profiles.json`에 별도 측정했다. Tracemalloc은 native/BLAS 일부를 누락하고 RSS는 누적 process high-water다. GPU speedup이나 production CAE 가속을 검증하지 않았다. 비용은 고전 O(N) solver가 이미 경쟁력 있음을 보여주며 신규 효과를 입증하지 않는다.

## D. 반증과 수정

- A: full update는 정확하지만 frozen packet은 물성 변화의 cross reaction을 누락한다. 분모 없는 tangent도 유한 변화에서 틀린다. 수정하면 기존 inverse-update 공식이다.
- B: 서로 다른 양수 source가 같은 coarse witness/norm을 갖고 해는 다르다. 어떤 decoder도 양쪽을 동시에 정확히 예측할 수 없다. Minimum-norm decoder의 나쁜 점수 전부를 최적 정보 하한으로 주장하지 않는다.
- C: raw 합성은 열 source superposition을 위반하고 낮은 k에서 커진다. 이류의 raw 합성 상대오차 **1.3251**, 질량 위반 **1.24706**. 좌표 보정 후에는 보통 덧셈과 동등하다.
- D: 실제 공간의 ordered cross moment는 필요하지만 transfer matrix다. Total tuple3개로 전체 field를 압축한 것이 아니며 O(N) prefix 정보가 필요하다.
- 방향성: reciprocal shortcut은 질량 오차가 약1e-15여도 최대 **0.10542** 상대오차였다. Full left/right update는 올바른 방향과 reference를 재현했다.
- 나쁜 baseline 자기반박: unregularized feature-linear의 새 source 오류 **388501**는 tiny singular direction과 coefficient norm~1.67e6에 의존했다. Test label을 보지 않고 validation으로 고른 ridge(alpha1)의 오류는 **0.214296**다. 기존 실패와 수정 모두 남겼으며 나쁜 기준선으로 우월성을 주장하지 않는다.

## E. 기존 모델과의 엄격한 차별성

A는 알려진 finite response update, B는 latent witness 제한과 projection, C는 좌표 재매개변수화, D는 기존 physical transfer에 scalar를 적합한 것이다. 어느 것도 이번에 독립 신규 primitive로 확립되지 않았다. 실제로 학습한 것은 D scalar의 least squares, feature-linear/ridge/compliance 회귀와 PyTorch MLP3개다. A/B/C나 C006을 새로 학습했다고 보고하지 않는다.

## F. 현재 가장 유망한 architecture

**선정 없음.** D가 정확하다고 기존 알고리즘을 최종 발명으로 채택하지 않는다. 원리적으로 틀린 후보를 대규모 학습으로 구제하지 않았고, 새 primitive로 남은 것이 없어 Stokes 확장도 실행하지 않았다. 큰 GPU/유료 자원/새 데이터 도메인은 미실행이다. CPU에서 가능한6개 Cycle의 핵심 반증·비교가 끝나 불필요한 나머지 Cycle을 채우지 않고 중단했다.

## G. 논문·특허

arXiv/JMLR 원문은 여전히403이었다. CVXPYlayers/OptNet/DEQ 공식 README를 새로 실제 읽었고 metadata와 displayed optimization/fixed-point 수식을 확인했다. 저장된 이전 NeuralOperator/HNN/LNN/graph 공식 자료도 참고했다. Full paper 및 선행특허는 읽거나 검색한 것으로 주장하지 않는다. 신규성은 **미확인**, 발명/진보성 판정은 없다. 구체적인 계산 동일성의 증명은 원문 접근과 별개로 수행했다.

## H. 다음 세션

단일-valued 선형 map에 임의 nonlinear 합성을 넣는 대신, **여러 해 branch와 안정성 정보가 필요한 열/유동의 relation**을 계산 객체로 검토한다. 예를 들어 Bratu 열반응 문제는 새 benchmark 질문이며 이번에 실행하지 않았다. 여러 branch가 모두 안정하다고 가정하지 않으며 안정성은 별도 검증한다. 새 primitive를 정하지 않은 채 polynomial elimination, mixture, implicit solver에 이름만 붙이지 않는다. 원문·정보 충분성·알고리즘 동일성부터 점검하고 작은1D 열 문제와 방향성 gate를 다시 통과시킨다. 정확한 명령과 보존 규칙은 `research/session_0003/next_steps.md`에 저장했다.

기존 Session0002 결과 **232개 파일과 기존 source hash를 보존**했다. Test34개 통과, 저장 오차 재계산, checkpoint20회 예측 재현, banded reference1,786회 대조 및6개 signed 방향 검증을 완료했다. 이1,786회는 같은 입력의 중복 method 평가를 포함하며 독립 data 수가 아니다. 코드·그림·무결성 검증은 `results/session_0003/validation/`에 있다. 이번 세션의 GitHub push, environment publication, 대규모 삭제는 하지 않았다.
