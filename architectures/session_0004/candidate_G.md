# G — State/balance co-generation with latent constitutive feedback

## 객체·입력

객체는 온도만이 아니라 **생성 중인 (T,j) pair와 구성식 mismatch**다. Full input `p=[log k_edges,Q_nodes,gL,gR,lambda]∈R^(2N+4)`를 보존한다. Witness-only bottleneck을 반복하지 않는다. 무차원 격자 h=1/(N+1),N15. 모든 잠재 변수는 dimensionless이고 실제 flux는 kref*Tref/L로 복원한다.

## 완전한 연산과 학습

Train mean/std로 정규화한 p에서 `h0=tanh(E(p-center)/scale+e)`를 만든다. width48,3개 shared block. 각 block에 아래 식을 적용한다.

\[
(a,j_0)=D h+d,\quad T_i=(1-x_i)g_L+x_i g_R+x_i(1-x_i)a_i,\\
s_i=Q_i+\lambda e^{T_i},\quad j_e=j_0+h\sum_{i=1}^{e}s_i,\ e=0,\ldots,N,\\
r_e=j_e+k_e(T_{e+1}-T_e)/h,\\
h^+=\tanh\left(W h+w+U r+
\tanh(Gh+g)\odot\tanh(Vr)\right).
\]

마지막에도 같은 decoder로 (T,j)를 생성한다. 온도가 source를 결정해 balanced flux를 만들고 구성식 관계가 내부 latent gate를 바꾼다. E,D,W,U,G,V와 bias를 학습하며 BC/source누적/구성식 진단은 고정이다. **Loss는 교사학습 온도·flux MSE이며 PINN penalty를 추가하지 않는다.** 128labels/validation32,3seeds,Adam1500steps,lr.003.

비교는 feedback 없는 같은 joint decoder, T/j를 독립 생성하는 MLP, 고전 banded/Newton이다. Flux 라벨 추가를 숨기지 않는다.

## 메커니즘과 동등성 감사

출력 이후 projection 대신 생성 중 balance를 만족하는 매개화이며 잔차가 학습 정보 흐름에 참여한다. 고정 solver step `T+=M residual` 대신 latent에서 상태를 재생성한다. 하지만 `h+=tanh(Wh+Ur+gate(h)gate(r))`는 일반 gated recurrent/residual network로 전개되며 r을 물리 edge message로 보면 GNN/learned iterative 계열과도 관련된다. **기존 행렬 연산이라는 이유로 배제하는 것이 아니라** feedback ablation·정보 흐름·효과·선행연구 검증이 필요하다. Level3 목표의 계산 제안이며 확립한 새 primitive는 아니다.

## 보장과 한계

유한 exp/산술에서 생성 flux의 `j_e-j_(e-1)=h(Q+lambda exp T)`와 BC는 정확하다. 생성 flux와 `-k gradT`의 일치는 보장하지 않으므로 구성식 flux의 에너지 수지와 PDE 잔차를 별도로 검사한다. C006의 divergence-only 성공과 같은 혼동을 피한다. 열원이 T에 의존해도 생성 balance는 만족한다. Positivity/안정성/branch선택/수렴/임의N·geometry는 보장하지 않는다.

한 layer O(Nw+w²), L layer O(L(Nw+w²)), parameters O(Nw+w²), activation O(L(N+w)). 준비 normalization O(BN), reference label은 Newton 반복마다 banded O(N). Dense gate의 GPU 실행은 가능하지만 cumsum 동기화/작은 batch 효율은 미측정이다. 선형·비선형 열을 같은 architecture로 검사하며 유동 divergence/momentum·HVAC/RANS에는 별도 구조 정의와 검증이 필요하고 이번에는 미실행이다.
