# J — 유한 compatibility increment를 운반하는 상태·오류 객체

## 1. 계산 객체와 단위

상태 `(T∈R^N,j∈R^(N+1))`, 보존오차 `b=Dj-Q-lambda exp(T)`, 구성식오차 `c=j-C_vT-c_g`를 구분한다. D는 edge flux의 divergence, C_vT+c_g는 열/이류–확산의 물리 flux다. 원래단위는 Tref,Jref,Qref이고 식은 x/L,T/Tref,j/Jref,Q/Qref로무차원화했다. 서로 다른단위의오차를그대로더한물리보장을주장하지않는다.

변화객체는 `(deltaT,deltaj;delta b,delta c)`이며 다음유한식으로함께운반한다.

\[
\Delta b=D\Delta j-\lambda e^T\odot\operatorname{expm1}(\Delta T),\qquad
\Delta c=\Delta j-C_v\Delta T.
\]

이는tangent근사가아니라반응항의유한변화다. 선형구성식에대해

\[
R_{PDE}=D(C_vT+c_g)-Q-\lambda e^T=b-Dc
\]

가정확하다. 따라서 b=0과 c=0은다른조건이고,작은c도divergence를취하면클수있다.

## 2. 시도한 실제 갱신 규칙

두오차를함께줄이는첫제안은 E=(||b||²+||c||²)/2의상태변화다.

\[
g_T=-\lambda e^T\odot b-C_v^Tc,\quad g_j=D^Tb+c,\qquad
(T',j')=(T,j)-\eta(g_T,g_j).
\]

이를전개하면 `delta z=-eta J_F^T F(z)`라는 **고전 nonlinear least-squares gradient/defect correction**이다. lambda0에서 `L=[[0,D],[-C_v,I]]`, `g=L^T(Lz-rhs)`와수치적으로일치한다. M_theta=diag(exp theta)같은학습mobility를붙이는확장도learned optimizer에해당한다. 실제코드는M=I,eta=.5/||L||²를시험했다. theta학습은실행하지않았다. 신규학습연산이확보되지않아억지NN를만들지않았다.

## 3. 보장·반증

선형L에대해0<eta<2/||L||²이면defect energy가감소한다. 이것이온도L2정확도/positivity/빠른수렴을보장하지않는다. 양수source/BC0에서임의joint state의한step이음수온도를만드는반례를실행했다. Low-k에서는작은defect와큰state error/느린수렴을구별했다. 유속±1에서는서로다른C_v를사용하고방향을지우지않았다. 새로운geometry/해상도범용성은증명하지않았다.

## 4. 복잡도·기존 모델과의 차이

현재 scalar d=1이다. Sparse D/C에서한step O(Nd),저장O(Nd). 학습Lstep은L배다. 실험의dense norm/condition준비는O(N³),저장O(N²)이며, 실제 dense L을 곱하는 실험 step은 O(N²)다. Sparse 구현의 이론 비용과 구분한다. 정확한1D banded solver는O(N)이므로이규칙의재구현에계산상우월성이없다. Attention/GNN이름을쓰지않아도실제알고리즘은gradient defect correction이다.

**판정: 현재갱신을Level3발명으로REJECT.** 분리된 `(state,b,c,finite increments)`는다음계산규칙을찾기위한진단표현으로보존한다. 새primitive를확립한것은아니다. 열및방향성이류–확산에같은compatibility정의는재사용되지만그것과새architecture는별개다. 코드/증거: `experiments/session0005_hypotheses.py compatibility`, `results/session_0005/compatibility.json`.
