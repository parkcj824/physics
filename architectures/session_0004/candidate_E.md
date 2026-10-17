# E — Paired fold-coordinate field generation

## 객체·입출력·완전한 식

정상해 집합을 `(c(x),v(x),d)`로 표현한다. Bratu benchmark에서 L,Tref,kref로 무차원화했고 lambda와 x도 무차원이다. 유차원 온도는 Tref를 곱하고 기준온도를 더한다. 이 객체는 초기조건/이력을 담지 않는다.

`MLP(z)=W3 tanh(W2 tanh(W1 z+b1)+b2)+b3`, hidden width32, z=(lambda/4,(x-.5)^2). `(u,v)=MLP(z)`, `b(x)=x(1-x)`로

\[
\hat\lambda_* =3.2+\log(1+e^\alpha),\quad d=\hat\lambda_*-\lambda,
\quad c=8b(x)u,\quad g=8b(x)\log(1+e^v),\quad
T_\pm=c\pm\sqrt{d}\,g\quad(d>0).
\]

d≤0이면 empty-set 출력으로 해석하며 구현에서 수치진단용 field는 clamp(d,1e-10)로 계속 계산하되 **branch count0**을 함께 기록한다. 3.2는 training maximum에서 얻은 prior이며 정확한 fold를 공급하지 않았다. `alpha=-.4` 초기화. 학습은 paired field MSE,48lambda×2branches×15points, Adam0.003/2000steps, validation32로 checkpoint 선택, seed3개다. 정답해의 좌표는 scalar brent root로 생성한 고전 해석식이다.

## 원리·차이·반증

낮은/높은 branch를 독립 experts로 만들지 않고 중심·분리방향·존재 gap을 공유한다. Query-key attention이나 node aggregation은 없고 pointwise conditional field generator다. Saddle-node square-root 정상형을 기존 수학으로 사용한다는 이유로 배제하지 않는다. 하지만 이 실제 구현은 **구조화된 두-head conditional MLP**로 펼칠 수 있다. 새로운 정보 교환 primitive로 확립되려면 표준 branch-conditioned/mixture MLP 대비 실증 이점이 필요하다. Level2 표현 가설이며 Level3은 미입증이다.

임의의 더 큰 fold c'>max(train lambda)에 대해 `g'=g*sqrt((old_fold-lambda)/(c'-lambda))`로 모든 training paired field를 그대로 재현할 수 있다. 따라서 학습 가능한 g를 제한하지 않는 한 branch label만으로 fold 위치가 식별되지 않는다. 수학 반례이며 단순히 더 오래 학습하면 해결된다는 주장은 없다.

## 보장·한계

BC, reflection symmetry, d>0의 branch ordering은 구현상 정확하다. 양의 온도, 잔차, 정확한 branch수, 선형 안정성·동적 선택·인과성은 보장하지 않는다. fold에서 Tminus=Tplus를 강제하지만 올바른 fold값은 별개다. 다중해를 표기하는 것이 초기조건에 따른 basin 선택을 푸는 것은 아니다. 두 branch보다 복잡한 bifurcation을 처리하지 않는다. no-branch 라벨 없이 existence prediction을 평가한다는 한계도 공개한다.

Nquery, widthw, sampleB일 때 추론O(BNw²), memoryO(BNw+w²), 학습S배와 역전파 저장이 필요하다. Label은 branch당 scalar root와 profileO(N), reference stability는 eigenvalue computation 별도 비용이다. 좌표 query로 N63까지 계산 가능하나 임의 geometry 보장은 없다. Thermal saddle-node 이외 재사용은 미검증이다.
