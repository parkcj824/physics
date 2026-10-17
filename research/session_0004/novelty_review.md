# Session 0004 — 신규성 기준과 확인 범위

Level1 기존 수치절차 재구현, Level2 새로운 학습 표현, Level3 새로운 학습 상호작용/정보 흐름, Level4 새 수학으로 구분한다. Layer가 matmul/tanh/pooling을 쓴다는 이유로 Level3 가능성을 배제하지 않는다. 반대로 아키텍처의 연산 순서까지 기존 모델로 전개되고 새로운 유용성이 확인되지 않으면 이름만으로 승격하지 않는다. **수학적 관련성, architecture 동일성, 실증 이점, 문헌 신규성은 네 개의 별도 판단이다.**

## 실제 확인 자료

이번 세션에 원문 사이트 요청을 반복하지 않았다. 기존 arXiv/JMLR CONNECT403 기록을 유지하며 네트워크가 풀렸다고 주장하지 않는다. 이전에 저장해 실제 확인한 다음 자료를 읽었다.

- `research/session_0003/sources/deq.txt`: **Deep Equilibrium Models**, Shaojie Bai, J. Zico Kolter, Vladlen Koltun; NeurIPS2019. 공식 https://github.com/locuslab/deq . README의 fixed-point solver/implicit backward. Full paper 미열람.
- `research/session_0003/sources/cvxpylayers.txt`: **Differentiable Convex Optimization Layers**, A. Agrawal, B. Amos, S. Barratt, S. Boyd, S. Diamond, Z. Kolter; NeurIPS2019. https://github.com/cvxgrp/cvxpylayers . parameterized optimization layer 설명/예제. Full paper 미열람.
- `research/session_0003/sources/optnet.txt`: **OptNet: Differentiable Optimization as a Layer in Neural Networks**, Brandon Amos, J. Zico Kolter; ICML2017/PMLR70. https://github.com/locuslab/optnet . QP layer README. Full paper 미열람.
- 과거 저장된 MeshGraphNets 구현은 `e'=MLP([v_sender,v_receiver,e])`, `v'=MLP([v,sum e'])`를 사용하고 Neural Operator spectral 구현은 FFT/channel contraction/IFFT를 사용한다. 정확한 출처/메타데이터와 저장 위치는 변경하지 않은 `research/novelty_review.md`, `research/literature_sources/`에 있다. 이 분야 전체 문헌을 새로 검토했다고 세지 않는다.

## 후보별 정확한 비교

E의 square-root branch gap은 saddle-node normal form이라는 **기초 지식 및 직접 유도**에 근거한다. 특정 논문에서 새로 확인한 수식으로 인용하지 않는다. MLP plus tied two-head representation과 차이가 있는 것은 공유 중심/순서/존재 gate이고, 정보 교환이 새로운지와 고유 장점은 실험으로 따져야 한다. Multi-solution neural operators/branch learning/normal-form networks 원문을 읽지 못해 nearest-prior-art 검토는 미완료다.

F의 finite mixed response는 Taylor derivative로 대체하지 않은 inclusion–exclusion이며 구현은 learned symmetric multiplicative pooling이다. Volterra expansion, factorization machines, bilinear networks, quadratic DeepSets와 수학적으로 연관되어 있다. 이 관계만으로 모든 응용 architecture를 자동 기각하지 않지만 **현재의 계산을 그들로 전개할 수 있다는 점**과 quartet supervision의 실질적 장점을 분리한다. Counterfactual-response/interaction operator 원문을 새로 읽지 않아 독립 신규성은 미확인이다.

G는 output projection을 호출하지 않고 full source/state에서 balanced flux를 생성해 constitutive mismatch를 latent gate에 피드백한다. 알고리즘은 단순한 fixed Richardson/Newton/primal-dual step과 같지는 않다. 하지만 일반 gated recurrent network에 residual feature를 주는 구체적 형태이고 hard-constraint parametrization/learned solver 계열과 가깝다. Source-dependent constraint와 feedback의 추가 이점은 ablation으로 평가하며, 구조가 새롭다고 단정하지 않는다. PINN loss/DEQ의 내부 root solver를 실행한 것으로 잘못 표현하지 않는다.

## 아직 조사해야 할 분야

Multi-solution operator 및 bifurcation-learning, finite response interaction/Volterra neural operators, hard-constrained jointly generated state/flux, learned iterative and graph residual networks를 우선 원문 검토해야 한다. GENERIC/port-Hamiltonian, equivariance, neural finite volume/domain decomposition/multigrid 및 특허 범위의 이전 공백도 그대로 미해결이다. 새 선행특허 검색은 수행하지 않았다. **논문·특허 신규성 미확인; 확립한 신규 범용 architecture 없음.**
