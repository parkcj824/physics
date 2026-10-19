# G 정보 흐름 통제의 실행 가능한 정의

## 입력·고정 물리와 출력

N=15, h=1/16, 입력 `raw∈R^34=[log(k_edges)∈R^16,Q∈R^15,gL,gR,p]`다. G의 p는 λ, H의 p는 유속 v다. H의 k는 ν다. 온도·flux·source·길이 등은 기준 단위로 무차원화했다. G와 H는 **별도 가중치**를 학습한다. 동일 architecture의 재사용과 가중치 공유는 구분한다.

훈련 데이터만으로 `mu=mean(raw)`, `sigma=max(std(raw),0.1)`를 고정한다. `h0=tanh(E((raw−mu)/sigma)+e)`다. 여기서 h0는 latent 벡터이며 격자 간격 h와 다른 기호다. Decoder `d=Ah_l+a`에서

\[
T_i=g_L(1-x_i)+g_Rx_i+x_i(1-x_i)d_i,\qquad
j_e=d_{N+1}+h\sum_{i=1}^{e}q_i^{eff},
\]

`q_eff=Q+λexp(T)`(G), `q_eff=Q`(H)다. e=0의 누적합은 0이다. 따라서 `Dj=q_eff`는 구현 산술오차 내에서 정확하다. 물리 flux는 별도로

\[
J_G(T)_e=-k_e(T_{e+1}-T_e)/h,\qquad
J_H(T)_e= v_+T_e+v_-T_{e+1}-\nu_e(T_{e+1}-T_e)/h
\]

로 계산한다. 양 끝 온도는 지정 경계값으로 삽입한다. `r_l=j_l−J(T_l)`는 구성식 불일치다. 수지 불일치 `Dj−q_eff`와 다른 정보다.

## 세 latent 갱신의 식

모든 joint 비교에서 다음의 같은 행렬 블록을 사용한다.

\[
h_{l+1}=\tanh(W h_l+w+U s_l+
\tanh(Bh_l+b)\odot\tanh(Vs_l)),\quad l=0,1,2.
\]

- original: `s_l=r_l`, latent width48.
- physics_plus_feature: `s_l=r_l+P h_l`.
- learned_feature: `s_l=P h_l`. P는 bias 없는 48→16 행렬이다. 직전 모델과 같은 seed/초기 파라미터, 같은 활성 θ=9472개다.
- static_physics: `s_l=r_0`. 최초 decoder와 물리 flux로 r0를 한 번 만든다. 이후 세 latent 블록은 계속 수행한다. 이 모델은 물리 정보를 처음에 한 번 제공하는 통제이며 모든 물리 입력은 이미 raw에도 있다.
- direct_matched: latent width64, `h_{l+1}=tanh(Wh_l+w)`, 온도는 위 envelope, flux 16개는 별도 decoder 출력. 수지 제약을 강제하지 않는다. 총 θ14623 중 사용되지 않는 inherited gate 파라미터를 제외하면 활성 θ8415개다.

C0b는 같은 P가 있는 width48 블록을 사용한다. `n(z)=z/sqrt(mean(z²)+1e−12)`라 하면 normalized generic은 `s=n(Ph_l)`, normalized physical은 `s=n(n(Ph_l)+n(r_l))`, normalized static은 `s=n(n(Ph_l)+n(r_0))`다. 잔차 크기를 제거한 별도 개입이며 raw 비교와 같은 가설이 아니다.

## 학습 규칙과 비교 한계

`L=mean(((T−Tref)/u_scale)²)+mean(((j−jref)/j_scale)²)`. 두 scale은 training reference의 RMS다. Adam lr0.003, gradient norm clip10, 매100step validation loss가 가장 작은 checkpoint를 선택했다. Direct는2000step, 나머지는1000step이다. PDE penalty를 loss에 추가하지 않았다. 확인 결과로 checkpoint를 선택하지 않았다.

같은 파라미터 수는 같은 연산량이 아니다. Physics-plus는 물리 flux와 중간 decoder를 다시 계산하므로 learned-feature보다 profile 연산이 약10% 많다. Original/static는 θ8704개로, P가 있는 모델과 다르다. Direct는 연산 예산을 근사적으로 맞추기 위해 2배 step을 썼지만 정확한 시간/FLOPs 일치는 아니다. 총 θ만 보고 인과 결론을 내리지 않는다. C0b는 scale 차이를 추가 점검하지만 3seed/작은 분포/정해진 gate의 한계가 있다.

## 고전/학습 solver 기준선

초기 온도는 affine 경계 lifting이다. 구성식 flux로 `R=q_eff−DJ(T)`, diagonal `a_ii=(k_i+k_{i+1})/h²`(H는 `+|v|/h`)를 만들고

\[
T^{l+1}=T^l+\sigma(\alpha_l)R/a_{ii},\quad l=0,1,2
\]

를 수행한다. α 세 개만 같은 supervised loss로 학습했다. 이는 learned Jacobi이며 발명 후보가 아니다. α=0 초기화가 모든 seed에서 같아 반복 seed 결과는 결정론적 재현이다. 일반 classical Jacobi는 damping0.8로3/32step을 실행했고, exact 기준해는 banded solve/damped Newton이다.

N 자유도, latent width d, L개 갱신에서 dense E/decoder O(Nd), 각 latent O(d²+Nd), 전체 O(L(d²+Nd)), 저장 O(Nd+d²)다. 역전파 활성 메모리는 batch 및 L에 비례한다. 1D classical banded solve는 선형 문제에서 O(N), Newton은 수렴한 반복 수에 추가 비례한다. 작은 고정 N에서 관측한 batch latency를 모든 geometry/GPU에서의 우위로 확대하지 않는다.
