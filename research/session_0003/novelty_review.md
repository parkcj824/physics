# Session 0003 신규성 검토 — 확인과 미확인

원문 접근을 현재 세션에 다시 검사했다. arXiv `2010.08895`, JMLR `v24/21-1524.html` 요청 모두 CONNECT proxy403이었다(`literature_access.json`). 같은 실패를 반복하지 않았고 수학 실험을 중단하지 않았다. 추가로 저자/공식 README3개를 실제 내려받아 읽었으며 exact URL/byte/SHA256은 `sources/access.json`에 있다. Fetched 자료는 데이터로만 취급하고 설치하거나 실행하지 않았다.

## 새로 확인한 공식 자료

| 제목·저자·연도·발표 | 확인한 출처 | 실제 확인한 식/내용 | 본 연구와의 관계 및 한계 |
|---|---|---|---|
| **Differentiable Convex Optimization Layers**; A. Agrawal, B. Amos, S. Barratt, S. Boyd, S. Diamond, Z. Kolter (README에 표기된 initials); NeurIPS2019 | [공식 CVXPYlayers](https://github.com/cvxgrp/cvxpylayers), [저자 제공 원문 링크](https://web.stanford.edu/~boyd/papers/pdf/diff_cvxpy.pdf) | parameterized convex problem을 풀어 solution을 반환/미분한다는 설명과 `Problem(Minimize(...), constraints)` 예제, citation | B의 full constraint closure를 layer로 넣는 것만으로 새 primitive가 되지 않는다는 관련 배경. **원문은 읽지 않았고** B와 논문 모든 수식이 동일하다고 주장하지 않는다. |
| **OptNet: Differentiable Optimization as a Layer in Neural Networks**; Brandon Amos, J. Zico Kolter; ICML2017/PMLR70,136–145 | [저자 공식 repository](https://github.com/locuslab/optnet), [저자 제공 paper](https://arxiv.org/abs/1703.00443) | 제목·저자·venue/year·QP layer library `qpth` 소개 | optimization을 neural layer로 사용했다는 일반성이 이미 있다. Original QP/KKT equation 전체는 README만으로 검토하지 않았으며 원문 미열람. |
| **Deep Equilibrium Models**; Shaojie Bai, J. Zico Kolter, Vladlen Koltun; NeurIPS2019 | [저자 repository](https://github.com/locuslab/deq) | README의 `z_star=solver(lambda z:f(z,x),...)`와 implicit backward fixed-point 설명. Anderson/Broyden solver 사용. Citation metadata. | 기존 solve를 implicit layer로 숨기는 형태는 새로운 연구 object라고 단정할 수 없다. B/D를 DEQ와 동일 architecture라고 주장하지 않는다. **원문 미열람**, README code/설명만 확인. |

논문 출처 URL은 실제 공식 자료에 나온 링크이며, 링크의 원문을 읽은 것으로 세지 않는다. CVXPY 저자 이름은 source에 있는 initials대로 적어 확인되지 않은 확장을 피했다.

## 수식 수준의 정확한 비교

- **A:** `G⁺=G-(Ga)(cᵀG)/(1+cᵀGa)`를 원래 행렬과 곱해 inverse 성질을 검사했다. 이는 표준 Sherman–Morrison identity와 같은 식/계산 순서다. 유한 분모와 cross-susceptibility 보완이 기존 identity를 넘어선다고 주장할 수 없다. Original identity publication을 새로 읽은 것은 아니다.
- **B:** coarse witness의 `ker R` 충돌을 구성했다. Full rank에서는 최소 norm constraint solve/pseudoinverse다. 추가적인 certificate를 만든다는 표현만으로 residual/projection/dual 계열에서 독립하지 않는다. 정보 충분성이 없다는 수학 반례와 관련 최적화 layer의 존재를 구분한다.
- **C:** `phi(a⋆b)=phi(a)+phi(b)`를 항등식·테스트로 확인했다. Input pointwise coordinate change, addition, inverse change로 전부 전개되므로 새로운 coupling을 보이지 않는다. 특정 formal-group neural paper를 읽었다고 주장하지 않으며 exact conjugacy만으로 현재 발명 claim을 보류한다.
- **D:** 구간 `[T,j,1]`에 작용하는 upper triangular transfer matrix를 정의하고 `M_AB=M_B M_A`를 검증했다. 전체 source/저항 prefix 복원도 알려진 공간 integration이다. **일반적으로 어떤 연산이 matrix multiplication으로 표현된다는 사실만으로 신규성을 배제하는 것은 아니다.** 여기서는 이 특정 알고리즘이 고전 transfer 전체와 같다는 의미다.

## 요청 분야별 범위

| 분야 | 이번 세션 확인 | 남은 일 |
|---|---|---|
| Learned Green / sensitivity-response / iterative solvers | A식의 정확한 inverse-update 동일성, 이전 official operator code·저장 기록 | 원문 learned response/Green/sensitivity 논문 |
| Neural Operators / multigrid / domain decomposition / neural finite volume | 이전 공식 코드/문헌 ledger 참고, D의 transfer equivalence | 관련 원문/가장 가까운 architecture 검토 미완료 |
| Equivariant / Hamiltonian / Lagrangian / GENERIC / port-Hamiltonian | 이전 세션 자료를 읽었고 C의 coordinate conjugacy를 별도 증명 | GENERIC/port-Hamiltonian 원문 gaps 미해결; 새 검토를 했다고 세지 않음 |
| Physics-constrained / projection / dual | 이번 OptNet/CVXPYlayers/DEQ official README/code snippet 확인 | full papers와 B의 구체적 차이, certificate literature |

**신규성 미확인, 확립한 신규 architecture 없음.** Patent search는 수행하지 않았으며 특허의 신규성/진보성을 판정하지 않았다. 현재 가설의 명확한 기존 계산 동일성과 전체 문헌·특허 검색의 미완료는 별개의 사실이다. 원문 접근이 풀리더라도 기존 exact equivalence를 이름으로 덮어쓰지 않는다.
