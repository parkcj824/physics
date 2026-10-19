# K — 관측 가능한 열 상태와 이력 객체

## 물리적 출발점과 계산 객체

E의 branch ID를 더 학습하는 대신 순간 관측과 완전한 물리 상태를 구별한다. Cattaneo 열식 `T_t=-j_x`, `tau j_t+j=-kT_x`, 닫힌 Neumann flux 경계에서 cell-centered cosine 온도 모드의 진폭 a와 staggered sine flux 모드의 진폭 b를 쌍으로 둔다. 공간·온도·flux·시간을 각각 L,Tref,Jref,tref로 무차원화했다. 양의 평균 온도 1에 작은 cosine 편차를 더한 경우이며, 편차 자체에는 부호가 있다.

## 실제 계산식과 학습

\[
z=(a,b)^T,\quad \dot z=A z,\quad
A=\begin{bmatrix}0&-\omega_h\\k\omega_h/\tau&-1/\tau\end{bmatrix},\quad
P=e^{\Delta t A},\\
a_{n+2}=c_1 a_{n+1}+c_2 a_n,\quad
(c_1,c_2)=\arg\min_c\sum_{8\ histories}(a_2-c_1a_1-c_2a_0)^2.
\]

`omega_h=2N sin(pi/(2N))`, N=16, tau=0.2, k=1, dt=0.05다. 실제로 두 계수를 최소제곱 적합하고 새로운 초기 상태 32개에서 검사했다. 신경망을 학습한 실험은 아니다. 완전한 물리 상태 z를 제공하면 Pz로 미래가 유일하게 결정되므로, 이 물리가 본질적으로 비Markov라고 주장하지 않는다.

## 기존 알고리즘과의 관계 및 보장

Cayley–Hamilton에서 `c1=trace(P), c2=-det(P)`이므로 실제 구현은 기존 AR(2)/관측 가능한 상태공간 전파와 같다. 서로 다른 b와 같은 a라는 정보 반례에서는 이력이 유용하지만, 이것이 새로운 학습 메커니즘은 아니다. Branch mixture의 이름을 바꾼 후보도 아니다.

연속 에너지 `E=(a²+tau b²/k)/2`는 `dE/dt=-b²/k≤0`이다. 이는 고전 기준계의 성질이며 임의로 적합한 AR 계수의 보장은 아니다. `dt=pi/sqrt(k omega_h²/tau-1/(4tau²))`에서는 두 온도 snapshot도 flux를 식별하지 못한다. 이후 다른 간격의 미래 온도는 달라진다. 고정 계수는 새로운 dt, 모드, k, tau에 자동으로 일반화하지 않는다.

## 실제 반증 결과

학습한 AR2의 확인 오차는 최대 8.33e-17, AR1은 0.0601이다. 그러나 두 관측이 거의 같은 aliasing 사례에서 이후 온도 차이는 0.0410이다. 기존 계수를 dt=0.1에 재사용하면 최대 오차가 0.0878이다. 실제 수치는 `results/session_0005/causal.json`, 입력과 출력은 `arrays/causal_counterexamples.npz`에 보존했다.

## 비용·재사용·판정

알려진 단일 모드에서는 학습 계수 두 개, 시간 단계당 O(1), field 복원 O(N)이다. 전이행렬 준비와 학습용 이력 8개의 기준해 생성을 별도로 포함해야 한다. 일반적인 d차원 숨은 상태의 dense 전파 준비는 O(d³), 단계당 O(d²)다. 미지 형상, 여러 모드의 혼합, 비선형 branch basin은 미검증이다.

**현 구현을 신규 primitive로 REJECT한다(Level 1).** 보존할 질문은 관측에서 어떤 물리 정보를 잃었는가다. 완전한 상태를 누락한 이점과 새로운 architecture의 이점을 구분한다. 다른 도메인에서도 관측가능성 분석을 재사용할 수 있지만, 그 원리 자체를 신규 발명으로 주장하지 않는다. Cycle 26, `experiments/session0005_hypotheses.py causal`.
