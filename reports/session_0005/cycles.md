# Cycle 24–27 — 가설에서 반증까지

C0/C0b는 기존 G 주장 통제이므로 아래 발명 탐색 Cycle과 별개다. 수치는 실제 저장된 JSON/NPZ에서 가져왔다. 아래 J/K는 새 NN를 학습한 Cycle이 아니다.

## Cycle 24 — J: 상태·오차의 유한 compatibility increment

1. **출발 원리:** 보존 수지와 구성식은 다른 관계다. 하나의 joint state가 두 관계를 함께 운반하게 할 수 있는가?
2. **계산 객체:** 무차원 `(T,j;b,c;ΔT,Δj,Δb,Δc)`; `b=Dj−Q−λexp(T)`, `c=j−C_vT−c_g`.
3. **실행 식:** `Δb=DΔj−λexp(T)expm1(ΔT)`, `Δc=Δj−C_vΔT`. 시험 갱신은 `ΔT=−η(−λexp(T)b−C_vᵀc)`, `Δj=−η(Dᵀb+c)`다. 선형 case에서 η=0.5/||L||²로 고정했다.
4. **기존 연산과 차이:** b/c를 따로 정의한 진단과 새로운 메커니즘을 구분한다. 실제 갱신은 joint least-squares gradient와 정확히 같았다. Learning mobility는 미실행이다.
5. **검증 방정식:** `−(kT′)′=1`, k1/0.01, BC0, N15; `(vT−νT′)′=1`, ν0.1,v±1. 기준해는 banded solve, 별도 verifier는 dense solve를 사용한다.
6. **결과:** 열 field L2 0.07746%, 수지∞1.07e−14, 구성식∞0.003169, PDE∞0.1014. Nonlinear finite increment 오류7.44e−15/6.66e−16. 코드 `experiments/session0005_hypotheses.py compatibility`, 결과 `compatibility.json`와 대응 배열.
7. **자기 반박:** stable defect descent도 양수 온도를 −0.1326으로 바꿀 수 있다. Low-k는128step 뒤 field 오차99.998%다. 작은 defect와 정확한 해는 다르다. 단위는 무차원화했으며 임의 단위의 b/c norm을 그대로 결합하는 정리로 일반화하지 않는다.
8. **신규성:** 현 계산은 `−ηJ_FᵀF`; exact equivalence다. 기본 수학을 썼다는 이유로 기각한 것이 아니다. 논문 원문/특허 신규성 별도 미검증.
9. **판정:** 현 갱신 REJECT / Level1. 분리된 defect와 유한 증분은 진단용으로 보존한다.
10. **다음 가설:** correction 크기를 더 키우는 대신 어떤 state 문맥을 defect와 함께 운반해야 하는가를 묻는다. 독립 challenger로 nonlinear finite interaction 및 관측 상태를 검토한다.

## Cycle 25 — L/F5: 반응 중첩에 따른 상호작용 활성화

1. **출발 원리:** 여러 유한 변화가 만든 반응의 교차항을 실제 비선형 법칙에서 정의하고, 필요한 공간 반응만 활성화한다.
2. **계산 객체:** source별 물리 response footprint m_i(x), base T0, 집단 상태 S 및 mixed reaction H. Source 위치·폭·세기는 연속값이다.
3. **실행 식:** `S=Σm_i`, `H=λexp(T0)[expm1(S)−Σexpm1(m_i)]`; `τ=softplus(t0+t1λ/k)`, `K=softplus(W0+(λ/k)W1+log(k)W2)`; `T_hat=T0+S+x(1−x)K[H−τ]+`. Encoder 및 supervised loss의 전체 식은 candidate_L.md에 있다.
4. **기존 연산과 차이:** finite nonlinear overlap과 threshold라는 물리 표현이다. 하지만 `(m_i,expm1(m_i))`를 sum-pooling하고 Ψ를 적용하는 graph로 정확히 구현되며 source pair graph를 직접 선택하지 않는다. Bilinear/cubic/anchored attention과 같은 encoder/자료로 비교했다.
5. **방정식:** `−kT″=λexp(T)+Σa_i Gaussian(c_i,w_i)`, BC0,N15. Train/val128/32sets, M1/2/3; 확인은 single/pair/triple/four/far/shift/tiny/linear/lowk-linear. λ0/1.5, scalar k다. 비균질 k와 비영 BC를 처리한 F 실험이 아니다.
6. **결과:** Pair gate mixed 오차3.583e−5, bilinear5.151e−5, cubic3.909e−5. Gate 전체 field 오차2.103%. Four 고차 오차는 gate9.478e−5, cubic7.645e−5. Source shift field29.32%. 데이터/체크포인트 동결 후135 batches 실행, 전체 배열 보존.
7. **자기 반박:** Tiny의 실제 효과6.315e−9를 gate가 지운다. Activation 감소가 dense kernel 계산 절감을 뜻하지 않는다. Source 위치 OOD의 주된 오류는 공유 encoder다. 양수 source/BC0 positivity와 정확한 PDE는 다르다.
8. **신규성:** 일부 제한된 표현 효용은 관찰했지만 concrete pooled-set graph와 최근접 Volterra/attention 대비 독자적인 학습 메커니즘이 미입증이다. 원문 최근접 연구·특허 미확인.
9. **판정:** REVISE / Level2. Level3 primitive로 승격하지 않는다.
10. **다음 가설:** H만으로 correction이 정해지는가? 같은 H지만 서로 다른 state 문맥의 물리 반례를 만들어 정보 충분성을 시험한다(Cycle27).

## Cycle 26 — K: 순간 관측과 완전한 causal state

1. **출발 원리:** 이력의 이득이 새로운 memory 계산인지 누락된 물리 자유도의 복원인지 분리한다. F/G와 다른 관측가능성 가설이다.
2. **계산 객체:** Cattaneo 열 상태의 온도 진폭 a, flux 진폭 b, 관측 `(a_n,a_{n+1})`. 완전한 state는 z=(a,b)다.
3. **실행 식:** `z′=Az`, `A=[[0,−ωh],[kωh/τ,−1/τ]]`; `P=expm(dt A)`; `a_(n+2)=c1 a_(n+1)+c2 a_n`. c1,c2는8 histories로 최소제곱 학습했다. τ0.2,k1,dt0.05,N16이다.
4. **기존 연산과 차이:** 현 구현은 AR2와 같다. Cayley–Hamilton의 trace(P)/−det(P)와 직접 대조했다. 새로운 RNN/branch mixture라고 부르지 않는다.
5. **방정식:** `T_t=−j_x`, `τj_t+j=−kT_x`, closed Neumann flux, cosine/sine 한 모드의 정확한 반이산 전파. 미래가 다른 두 flux와 같은 T를 구성했다.
6. **결과:** 새32 initial states에서 AR2 최대오차8.33e−17, AR1 0.06005. 같은 현재 T의 미래 차이0.13594. 학습은 LS scalar2개이며 NN가 아니다. `causal.json`, `causal_counterexamples.npz`.
7. **자기 반박:** dt0.479425 aliasing에서 두 관측 차이2.25e−15여도 이후 차이는0.04100. 새 dt0.1에 고정 계수를 적용하면 최대오차0.08784. 완전한(T,j) state는 Markov다.
8. **신규성:** 구체적 algorithm과 coefficient가 고전 관측가능성/AR2 전파와 같다. 전체 물리 state가 과거 없이 결정되지 않는다는 발견이 아니다.
9. **판정:** 현 구현 REJECT / Level1. 관측에서 누락되는 상태에 관한 질문은 보존한다.
10. **다음 가설:** 새 도메인에서 정보의 선택·망각을 계산할 때 기존 memory/observability와 다른 학습 구조가 가능한지 먼저 정의한다. 같은 AR2를 다시 제안하지 않는다.

## Cycle 27 — L의 H-only closure 정보 충분성 반증

1. **출발 원리:** 같게 압축된 물리 defect가 정말 같은 필요한 보정을 뜻하는가?
2. **계산 객체:** 물리적으로 충실한 singleton footprints, S, H와 true mixed correction δ. 새 NN 객체를 강제하지 않는다.
3. **식:** `m_i=log(1+c_i f)`, `f=4x(1−x)`, 두 c쌍 `(0.3,0.3)/(0.1,0.9)`; `Qi=A_hmi−λexp(T0)expm1(mi)`. 두 H는 `λexp(T0)0.09f²`로 같고 S는 다르다. 수리 기준선은 `(A_h−λdiag exp(T0+S))^−1H` 한 번이다.
4. **기존 연산과 차이:** 반증 자체가 새로운 architecture는 아니다. S를 추가한 1-Newton은 기존 수치해석이다. 같은 이름의 새 후보로 등록하지 않았다.
5. **방정식:** nonlinear heat k1,λ1.5,BC0,N15/31, Q_i>0. Exact lower reference를 직접 계산했다. Gaussian 훈련 family 밖의 analytic source다.
6. **결과:** N15의 같은 H 차이2.08e−16, S 차이0.2124, 필요한 δ 차이0.00630886. 어떤 deterministic H-only 함수도 최악 L2 오차가 최소0.00315443. N31에서도 같은 정보 충돌을 확인했다.
7. **자기 반박의 범위:** 이는 물리적 footprint 의미를 유지한 closure의 반증이다. End-to-end feature에 다른 정보를 숨기는 모든 NN의 불가능성 정리가 아니다. Norm의 N 의존성을 mesh convergence 실패율로 해석하지 않는다. 학습한 Gaussian IID model을 새로 선택하거나 재학습하지 않았다.
8. **신규성:** 필요한 정보의 분석이며 새 알고리즘의 우위/원문·특허 신규성은 미입증이다. Newton1step 상대오차0.3966%/0.5916%는 강한 고전 기준선의 결과다.
9. **판정:** H-only 범용 closure는 REJECT, L의 전체 범위는 REVISE / Level2 유지. `(S,H)`는 다음 미검증 객체다.
10. **다음 가설:** State 문맥과 finite defect를 함께 보존하면서 기존 learned correction/MLP/set operator와 실제 정보 흐름이 다른 연산이 있는지 먼저 수식으로 정의한다. 새 메커니즘이 없으면 대규모 학습 대신 기록과 문헌 검토로 재개한다.
