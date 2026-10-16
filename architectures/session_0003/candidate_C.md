# C — 변환군으로 합성하는 유한 변화량

## A. 물리 출발점/객체

대칭과 좌표 변환의 합성에서 출발한다. 기본 객체는 연속값 변화량 `a,b∈R^d`이고 scalar coordinate scale `gamma≥0`를 둔다. 유차원 값이라면 gamma는 변화량 단위의 역수다. 실험에서는 무차원 gamma=4다. A의 sensitivity packet이나 B의 제약 인증을 결합한 후보가 아니다.

## B. 완전히 전개한 연산

\[
a\star_\gamma b
=a\odot\sqrt{1+\gamma^2 b\odot b}
+b\odot\sqrt{1+\gamma^2 a\odot a}.
\]

연산은 채널별이다. `phi_gamma(x)=asinh(gamma*x)/gamma`라 두면 정확히

\[
a\star b=\phi^{-1}(\phi(a)+\phi(b)).
\]

따라서 associative/commutative, identity0, inverse -a를 갖지만 이는 **덧셈군의 좌표 재표현**이다. 새로운 이름이나 nonlinear appearance로 신규성을 주장하지 않는다.

열전도 실험에서는 oracle response contribution `r_j=G_:j b_j`를 받아 `star_j r_j`로 합쳤다. 이는 raw physical response의 선형 superposition을 위반했다. 수정 control은 `star_j phi^-1(r_j)`를 만든 뒤 phi로 decode하며 결과가 정확히 `sum_j r_j`가 된다.

## C. 학습 변수

gamma 또는 채널별 gamma를 trainable parameter로 둘 수 있지만 **실제 gamma 학습은 하지 않았다**. 선형 물리 상태를 raw로 합성할 때 gamma→0 외에 보편적 superposition을 만족할 수 없음을 먼저 반증했다. 정확한 response G는 공급한 prior이고 이 후보가 학습한 것이 아니다. 원리적으로 틀린 합성에 학습량을 늘리지 않았다.

## D/E. 성질

Associativity와 inverse는 정확한 대수 성질이다. 보존·온도 원점 gauge·물리 단위의 단순 scale equivariance·source linearity는 자동으로 따라오지 않는다. 작은 양수 packet에서도 합산 결과가 과대해지고 낮은 k의 큰 response에서 문제가 확대된다. 이류에서 방향을 갖는 정확한 propagator를 공급해도 raw 합성은 질량을 위반했다.

특정 nonlinear constitutive coordinate에서 유사한 변환이 유효할 가능성은 있지만 이는 현재의 선형 k(x) 열전도 검증을 통과했다는 뜻이 아니다. 그러한 별도 PDE는 이번에 실행하지 않았다.

## F. 비용

한 pair 연산 `O(d)` 시간/메모리, m개의 vector response 길이N을 합치면 `O(mNd)` work. Associativity로 tree depth `O(log m)` 가능하지만 실측은 CPU left fold다. 여기서 response contribution 자체를 만드는 dense G는 준비 `O(N³)`, 저장 `O(N²)`, source 적용 `O(N²)`이므로 합성 cost만으로 전체가 빠르다고 주장하지 않는다. 큰 gamma/많은 packet에서 overflow 가능하며 global 안정성 보장은 없다.

## G. 비교와 판정

완전한 coordinate conjugacy 때문에 encoding→sum→decoding인 기존 pooled latent computation과 동등하다. Raw 차이는 물리적 오류였고 corrected 차이는 재매개변수화였다. 새 primitive에 해당하지 않는다.

**판정: REJECT.** Cycle15/17의 raw 실패와 conjugate control의 exact linear equivalence를 함께 보존했다.
