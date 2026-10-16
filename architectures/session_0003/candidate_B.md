# B — Constraint witness의 정보 충분성

## A. 객체와 입력

고정 무차원 `A∈R^(N×N)`에 대해 source `b∈R^N`를 coarse balance witness `w=R b∈R^r`, `r<N`로 표현한다. R은 상수/위치/삼각함수 test function의 QR-정규화된 행이며 시험에서는 `N=15,r=4`다. Witness에는 추가로 `||b||²` 같은 비선형 전역 관측도 고려했다. 물리 정보인 재료와 BC는 별도 조건이고, **full source를 decoder에 다시 주지 않는 표현**을 시험한다.

## B. 구현 가능한 연산

Witness가 해를 결정한다는 가설을 최소 구현으로 시험한다:

\[
C=RA,\quad w=Rb,\quad
u=C^+w=\arg\min_z\{\|z\|_2: Cz=w\}.
\]

이는 정확히 최소 norm 선형 제약 closure다. 학습형 대안 `u=W_2 tanh(W_1 w+a_1)+a_2`를 크게 학습하기 전에, **어떤 decoder에도 적용되는 충돌 반례**를 검사했다.

## C. 학습 가능량과 실제 실행

R_theta와 decoder weights를 학습 가능량으로 둘 수 있다. 실제 반증은 고정 R과 pseudoinverse로 했고, witness model을 학습한 것으로 보고하지 않는다. 모든 `rank(R)<N`에 비영 `delta b∈ker R`가 있다. `b±=b0±epsilon delta b`는 같은 witness를 갖지만 `A^-1 b±`는 다르다. `b0=1`과 작은 epsilon으로 **두 source 모두 양수**인 반례도 구성했다. R에 상수 관측이 있으면 두 source의 norm도 같게 만들 수 있어 `||b||²`를 추가해도 이 충돌은 사라지지 않는다.

## D/E. 물리 의미와 보장 한계

약한 balance를 만족하는 인증 관계는 전역 해를 유일하게 결정하지 않는다. 같은 witness에 다른 target `u±`가 있으면 어떤 단일 출력 z에도 `max(||z-u+||,||z-u-||)≥||u+-u-||/2`다. Signed source pair에서 zero decoder는 양쪽 상대오차1이 되며, 이는 학습량 문제가 아니다.

다만 실험의 minimum-norm decoder에서 관측한 약99% 오차 전부를 정보 손실의 최적 하한이라고 주장하지 않는다. 더 좋은 decoder가 restricted distribution에서 근사 성능을 개선할 수 있다. 배제하는 것은 **현재 witness만으로 모든 source의 해를 결정한다는 주장**이다.

순환 이류에서도 같은 속도/재료하에 서로 다른 초기 위상의 질량과 에너지가 같다. 전역 invariant witness가 같아도 response 위치는 다르다. 보존 검증 객체는 예측 상태를 대신하는 충분한 표현과 다르다.

## F. 비용과 크기

Witness encoding `O(rN)`, dense C 준비 `O(rN²)` (sparse A면 `O(rN k_nb)`), 최소 norm closure `O(Nr²+r³)`다. Cached decoder는 `O(Nr)`다. `r=N`까지 보강하면 full source 정보가 복원되지만 closure는 일반 선형 solve다. 부족한 witness를 작은 것처럼 세고 solve 비용을 숨기지 않는다. GPU 행렬 연산 적합성은 가능하나 정보 충돌을 해결하지 않는다.

## G. 기존 연산과 판정

현재 구현은 projection/pseudoinverse와 latent bottleneck의 조합이지 그들을 넘어선 primitive가 아니다. C003은 output trial span을 제한했고, 여기서는 input witness의 식별 가능성을 시험했다. 표현 위치는 다르지만 rank/information obstruction은 연관되어 있다. 이 차이를 새 발명이라고 주장하지 않는다. CVXPYlayers/OptNet의 공식 설명에서도 제약 solve를 layer로 쓰는 계열이 존재함을 확인했다(원문 미열람).

**판정: REJECT — 현재 witness-only 예측 표현.** Witness를 보조 인증/오차 추정에 쓰는 더 넓은 아이디어는 반증하지 않았으며 아직 미검증이다. Cycle14/17.
