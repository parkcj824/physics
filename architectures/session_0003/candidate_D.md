# D — 순서가 있는 공간 balance relation

## A. 새 출발점과 계산 객체

A/B/C가 독립 primitive로 부적합한 뒤 **실제 공간 구간을 이어 붙이는 순서**를 검토했다. 물성 edit의 가상 순서(C005)와 달리 서로 다른 재료/source 구간을 공간에서 바꾸면 물리 문제가 달라진다. 계산 객체는 field token이 아니라 입·출력 관계 `(R,S,C)`다. 구간은 `j_out=j_in+S`, `T_out=T_in-R*j_in-C`로 작용한다. R은 열저항, S는 source 총량, C는 source와 저항의 순서 있는 적분이다.

## B. 연산식과 출력 복원

두 구간 A 다음 B의 합성:

\[
(R,S,C)\circ_\theta(R',S',C')
=(R+R',\ S+S',\ C+C'+\theta R'S).
\]

물리적으로 theta=1이다. 이 경우 정확히

\[
M(R,S,C)=\begin{pmatrix}1&-R&-C\\0&1&S\\0&0&1\end{pmatrix},\qquad
M_{AB}=M_BM_A
\]

이므로 **고전 affine transfer matrix 합성**이다. `R'S` 항이 없어지면 같은 total R/S에서도 물리 response를 틀리게 계산한다. 합성은 associative이며 일반적으로 noncommutative다. 이 대수만으로 신규성이 생기지는 않는다.

실행한 이산식에서 edge e의 `r_e=h/k_e`, node source injection `s=[0,hQ_1,...,hQ_N]`, `c_e=r_e s_e`다. Prefix 관계를 `(R_i,S_i,C_i)`라 할 때

\[
j_0=(T_L-T_R-C_*)/R_*,\qquad
T_i=T_L-R_i j_0-C_i.
\]

Root tuple 하나는 boundary 입출력만 설명한다. **전체 interior field를 복원하려면 모든 prefix 관계가 필요해 O(N) 정보를 보관한다.** 세 숫자만으로 모든 장을 압축한 것처럼 보고하지 않는다.

## C. 실제 학습

실험의 확장 가설은 cross coefficient theta 하나를 학습한다. 고정 feature를 전개하면 `T(theta)=a+theta s`다. 같은128개 물리 label에서 normalized least squares로

\[
\hat\theta=\frac{\sum_m s_m^T(y_m-a_m)/\|y_m\|^2}
{\sum_m\|s_m\|^2/\|y_m\|^2}
\]

를 계산했다. 관측값은 `1.0000000000000002`였다. 이는 알려진 물리 coefficient를 데이터로 복원한 것일 뿐 새 neural primitive를 학습한 근거가 아니다. SGD/neural training으로 잘못 표현하지 않는다. 비교 MLP만 PyTorch Adam으로 실제 학습했다.

## D/E. 물리 성질

theta=1의 positive-k Dirichlet 문제는 이 이산 balance/constitutive law를 정확히 풀며 low k, heterogeneous material, local/new source, BC와 resolution 변화를 통과했다. theta≠1도 formal associativity는 있지만 물리 constitutive consistency는 잃는다. 보존 관계와 정확한 온도 변화는 별개다. 이 정상 1D 공간 관계가 transient 시간 propagator나 multidimensional foundation architecture를 자동 정의하지는 않는다.

## F. 비용과 GPU 적합성

Prefix feature와 source 출력 work `O(N)`, 저장 `O(N)`; boundary port relation만 필요하면 O(1) state로 reduce할 수 있다. Associative parallel scan의 이론 depth `O(log N)`는 GPU에 적합할 수 있지만 GPU 측정은 하지 않았다. 실제 CPU 비용은 source/BC 포함한 O(N) banded solver와 비교했다. Reference label 생성, coefficient fitting, inverse prior 준비 비용도 따로 기록했다.

## G. 기존 연산과 판정

Global attention 평균과 다르게 ordered cross moment를 운반하지만, matrix multiplication/transfer formalism으로 **계산 규칙 자체가 정확히 같고** 학습은 scalar를 넣은 solver 수준이다. 사용자가 요구한 독립 발명 기준을 충족하지 않는다. Neural operator와의 성능 우월성을 주장하지 않는다.

**판정: REJECT as invention; 정확한 classical control로 보존.** Cycle16/18. 물리 검증 성공이 신규성·아키텍처 성공으로 승격되지 않는다.
