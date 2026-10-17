# F — Finite-intervention mixed-response channels

## 객체·물리적 의미

기본 객체는 무차원 finite source edit `(a,c,width=.1)`와 그 **counterfactual quartet**이다. PDE는 `-T''=1.5exp(T)+a s_c1+b s_c2`, 영 Dirichlet, `s_c(x)=exp(-(x-c)^2/(2*.1²))`. 두 edit는 동시에 주어지고 가상 edit 순서나 physical hysteresis를 만들지 않는다. 소스·온도·거리 모두 무차원; 유차원화 시 a는 Qref, c는 L을 곱한다.

## 실제 학습 연산

모든 MLP는 Linear→tanh→Linear→tanh→Linear다. q(x)∈R^r, e(a,c)∈R^r, f(a,c,x)∈R, f0(x)∈R, r16. b(x)=x(1-x):

\[
\hat T_{ab}(x)=b(x)\left[f_0(x)+a f(a,c_1,x)+b f(b,c_2,x)
+{ab\over\sqrt r}\sum_{j=1}^{r}q_j(x)e_j(a,c_1)e_j(b,c_2)\right].
\]

단독 response와 mixed interaction을 별도 channels로 전달한다. 식의 b amplitude와 b(x) envelope는 구현에서 각각 amplitude/exact BC factor로 구분한다. Cross term은 어느 edit가0이면 정확히0이고 edit object 교환에 불변이다. **Green/Jacobian oracle을 입력하지 않는다.**

정확한 targets `(T00,T10,T01,T11)`를 각128input에4번의 고전 Newton solve로 생성한다. Components `(T00,T10-T00,T01-T00,T11-T10-T01+T00)`의 train RMS로 정규화해 네 항의 MSE를 동일 가중합한다. Additive ablation/direct symmetric MLP도 **같은 quartet 및 loss**를 사용한다. 숨겨진4배 label 이득을 주장하지 않는다. Adam1500steps, lr.003, validation32,3seeds. 가중치 모두 학습; source shape/envelope/PDE reference는 고정이다.

## 무엇이 새로울 수 있는가 / 무엇이 아닌가

Finite inclusion–exclusion은 일차 미분이 아니라 실제 유한 두 변화의 nonlinear response다. 그러나 유한차분/Volterra라는 기존 수학 자체를 새로 주장하지 않는다. 이 구현의 전달 rule은 learned bilinear pooling이며 DeepSets의 제곱 pooled feature로도 표현 가능하다. 기존 학습 모델과의 관계는 인정하되 counterfactual factorization이 보이는 실제 데이터 효율/일반화 효과를 검증한다. 현재 Level2/Level3-target proposal, 신규 mechanism 입증 전이다. 단순한 analytic nonlinear 좌표합(C, Session3)과는 달리 e,q,f를 물리 quartet에서 학습하고 출력은 physical 온도다.

## 정확한 보장·한계·비용

BC, edit swap invariance, zero mixed anchor 보장. PDE 잔차·보존·안정성·positivity·임의 resolution/geometry는 미보장. 3개 이상의 실제 edit는 삼중 이상 상호작용이 필요할 수 있어 pair truncate를 exact compositional law로 주장하지 않는다.

Medit,Nquery,rankr,widthw라 하면 단독O(MNw²), pair featureO(M²r+Nr), parametersO(w²+wr). 모든 pair를 동일 factor로 pool하면 `(sum e)^2-sum e²` 항등식으로 pair합O(Mr+Nr) 가능하지만 이번 실제 코드는M2다. Training S번 forward/backward와 quartet4배 준비비용을 포함한다. Known solver의 root-stepO(N)보다 본질적으로 빠르다는 보장은 없으며 GPU 적합성은 dense 작은 MLP 정도이고 GPU 측정하지 않는다. Nonlinear thermal reaction 외 도메인 재사용은 미검증이다.
