# Session 0005 신규성 검토와 확인 범위

**Level 3 primitive와 범용 architecture는 아직 입증하지 못했다. 논문·특허 신규성은 미검증이다.** 기존 수학을 사용한다는 이유로 기각하지 않는다. 아래에서는 실제 계산 순서의 동일성, 학습 표현의 유용성, 물리적 보장, 문헌 신규성을 별도로 판단한다.

## 실제로 읽은 출처

이번 세션에는 외부 요청을 보내지 않았다. Session 0003의 arXiv/JMLR proxy CONNECT 403 기록을 읽고 같은 실패 요청을 반복하지 않았다. 현재 네트워크에서 새로 403을 관측했다거나 접근이 복구되었다고 주장하지 않는다. 저장된 공식 README/코드를 다시 읽었고 원문 논문을 새로 읽은 것은 아니다.

| 출처 | 메타데이터와 실제 확인 범위 | 이번 비교에서의 의미 |
|---|---|---|
| [Deep Equilibrium Models 공식 저장소](https://github.com/locuslab/deq) | Shaojie Bai, J. Zico Kolter, Vladlen Koltun; NeurIPS 2019. `research/session_0003/sources/deq.txt`의 fixed-point solver 및 implicit backward 설명. 원문 미열람. | G는 내부 root convergence를 요구하는 DEQ 구현은 아니다. 그렇다고 residual을 latent에 넣는 것 자체의 신규성을 입증한 것도 아니다. |
| [MeshGraphNets 공식 코드](https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets) | 이전 ledger의 **Learning Mesh-Based Simulation with Graph Networks**, Tobias Pfaff, Meire Fortunato, Alvaro Sanchez-Gonzalez, Peter W. Battaglia; ICLR 2021. `research/literature_sources/meshgraphnet_core_code.txt`의 edge MLP, incoming sum, node MLP 및 residual 갱신을 확인. 원문 미열람. | G의 dense gated latent와 mesh edge aggregation은 동일 구현이 아니다. 그러나 physical residual을 입력 feature로 추가했다는 설명만으로 새로운 message mechanism을 주장할 수 없다. |
| [CVXPYlayers 공식 저장소](https://github.com/cvxgrp/cvxpylayers) | **Differentiable Convex Optimization Layers**; A. Agrawal, B. Amos, S. Barratt, S. Boyd, S. Diamond, Z. Kolter; NeurIPS 2019. 저장된 `research/session_0003/sources/cvxpylayers.txt`와 이전 검토의 parameterized optimization 예제. 원문 미열람. | constraint를 layer 안에 둔다는 일반적 발상은 이미 존재한다. J를 이 논문과 동일한 알고리즘이라고 단정하지 않는다. |

과거 원출처 URL·접근 기록·SHA는 변경하지 않은 `research/literature_sources/access_log.json`, `code_access_log.json`, `research/session_0003/sources/access.json`에 있다. 아래의 AR, least-squares, Volterra, pooled-set 비교는 직접 전개한 식과 기본 지식에 근거한다. 새 논문에서 실제로 확인한 식인 것처럼 인용하지 않는다.

## 후보별 수식·알고리즘 대조

### G 통제 — 신규 후보 아님

`h'=tanh(Wh+Us+tanh(Gh)⊙tanh(Vs))`에 s를 물리 구성식 잔차, 일반 학습 feature, 초기 잔차로 바꾼다. 이 gate는 일반적인 recurrent/residual feature update로 구현된다. 원래 G가 Newton/Jacobi와 정확히 같다고 주장하지 않는다. 대신 같은 활성 파라미터 9,472개의 `s=r+Ph`와 `s=Ph`를 비교했다. 물리 feature의 보편적인 고유 이점은 이번 데이터에서 입증되지 않았다. 입력 정보 자체의 유용성과 이 particular update의 효과를 구분한다.

### J — joint finite compatibility increment

`Δb=DΔj−λexp(T)expm1(ΔT)`, `Δc=Δj−C_vΔT`는 유한 변화 관계다. 실제 시도한 `Δz=−η J_FᵀF(z)`는 joint nonlinear least-squares gradient와 **계산 절차가 정확히 같다**. 선형 case에서 normal equation과 수치적으로 대조했다. 매개변수를 추가하지 않고도 알고리즘 동일성을 판별했으며, 새 NN를 학습했다고 보고하지 않는다. 기존 수학을 썼기 때문이 아니라 구체적 갱신 메커니즘이 같은 이유로 현 구현을 Level 1 / REJECT한다. 분리된 오차 객체는 진단 자산이다.

### L / F5 — reaction overlap과 학습 threshold

`H=λexp(T0)[expm1(Σm_i)−Σexpm1(m_i)]`는 유한 반응의 교차항을 모든 차수에서 묶는다. `Φ(m_i)=(m_i,expm1(m_i))`를 sum pooling한 뒤 Ψ로 H를 계산하고 threshold와 양의 kernel을 적용하는 graph로 **정확히 재구성**했다. 저장된 checkpoint의 NumPy graph 재구성 오차가 1e-12 미만이다. 이것은 임의 함수가 보편근사 가능하다는 약한 주장과 다르다.

그렇더라도 반응에서 유도한 표현과 singleton/선형 limit/positivity 보장은 실증할 가치가 있다. Bilinear·cubic·anchored attention과 같은 labels/encoder로 비교했으며 일부 mixed 지표의 차이는 있다. 하지만 threshold는 작은 진짜 효과도 지우고, cubic은 일부 고차 지표에서 더 좋았다. 공간 activation은 source pair/subset graph 선택과 다르며 dense 계산은 생략되지 않는다. **Level 2 / REVISE**, 새로운 Level 3으로 승격하지 않는다.

Cycle 27은 H가 같고 필요한 correction이 다른 양수 source 반례다. 물리적으로 충실한 singleton footprint라는 의미를 유지하면 S도 필요하다. `(S,H)`를 주고 1-step Newton으로 고친 결과는 알려진 수치해석 기준선이다. 새 architecture의 발명으로 세지 않는다. 정확한 correction 방정식 `A_h delta−λexp(T0+S)expm1(delta)−H=0`도 유도하고 저장 reference에서 검사했다. 이는 원래 PDE의 좌표 변경이며 새 학습 규칙이 아니다.

### K — causal heat state

두 온도 관측의 AR(2) 계수는 `trace(exp(dt A)),−det(exp(dt A))`와 같다. 현 학습 규칙은 최소제곱 AR(2), 정보 흐름은 선형 상태 관측·전파다. 완전한 `(T,j)`를 누락한 관측 모델의 이점이며 새로운 물리 법칙이나 memory primitive가 아니다. **Level 1 / REJECT**. 관측 간격 aliasing을 통해 두 snapshot도 항상 충분하지 않음을 확인했다.

## 미확인 최근접 기술과 다음 문헌 조사

- Bilinear/Volterra interaction learning, DeepSets/set interaction, anchored attention: 논문 원문과 학습 목적·가변 source generalization 비교가 필요하다.
- Hard-constrained state–flux networks, learned defect correction, differentiable solvers, gated residual networks: G의 같은 예산 물리 feature 통제를 포함하는 선행 실험 여부는 확인하지 못했다.
- Multi-solution neural operators, causal memory, Mori–Zwanzig/state-space learning: K의 고전 동일성을 넘어서는 학습 메커니즘은 제안하지 못했다.
- 실제 CAE의 nonlinear constitutive response, local defect와 state-dependent susceptibility를 공동 운반하는 architecture: 다음 가설의 최근접 문헌을 먼저 찾아야 한다.

새 선행특허 검색은 수행하지 않았다. 특허 신규성·진보성·권리범위 판정을 하지 않는다. 이번 성과는 재현 가능한 통제와 반례이며, 발명 또는 특허 가능성을 확정할 근거는 없다.
