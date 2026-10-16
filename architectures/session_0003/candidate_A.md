# A — 유한 교란의 양방향 response packet

## A. 계산 객체와 입력

무차원 강성 연산자 `A∈R^(N×N)`와 response `G=A^-1`, 현재 상태 `u=Gb`, 교란 `delta A=alpha d dᵀ`를 고려한다. 기본 객체는 node가 아니라 `(g_R,g_L,chi,alpha)`다. `g_R=Gd∈R^N`, `g_L=Gᵀd∈R^N`, `chi=dᵀGd∈R`는 양방향 response와 compliance다. 물리 단위가 있을 때 `alpha*chi`가 무차원이어야 한다. 열전도는 grounded incidence `d`와 `alpha=delta k/h²`를 사용한다.

## B. 실제 연산식

\[
u^+=u-\frac{\alpha}{1+\alpha\chi}\,g_R(d^Tu),\qquad
G^+=G-\frac{\alpha}{1+\alpha\chi}g_Rg_L^T.
\]

일반 rank-one 교란 `delta A=a cᵀ`에서는

\[
G^+=G-\frac{(Ga)(c^TG)}{1+c^TGa}.
\]

유한 변화의 분모는 일차 감도 `delta u=-alpha Gd dᵀu`에는 없다. 하지만 **전개된 식은 정확히 Sherman–Morrison이며 여러 packet은 Woodbury다.** 도함수보다 유한 변화에 정확하다는 것과 신규 계산법이라는 것은 다르다.

Frozen 가설은 다음 교란에서도 `g,chi`를 초기 G에서 가져온다. 이를 고치는 데 필요한 cross-reaction은 바로 `G^+` 갱신이다. 새 primitive로 이름만 바꿔 재등록하지 않는다.

## C. 학습/고정 구분

실행된 packet 실험은 **학습하지 않았다**. 정확한 base inverse를 강한 prior로 공급하고 유한 변화 규칙의 표현력/동일성을 시험했다. 잠재 학습 변수로 `G_theta=U_theta U_thetaᵀ` 또는 source-response 행렬을 둘 수 있지만 이는 learned Green/sensitivity 계열이며 이번 세션에서 학습하지 않았다. 새로 학습한 MLP·선형·D-coefficient는 별도 비교 기록이다.

## D/E. 물리 의미와 성질

- 양의 열전도율 갱신으로 SPD가 유지되고 `1+alpha chi>0`이면 완전 갱신은 정확한 새 이산 평형을 준다. 변경 BC가 만드는 rhs도 함께 반영했다.
- Frozen packet은 후속 반응 변화를 누락한다. 열 source의 위치/물성 변화에서 실패했다.
- 비자기수반 이류 문제에서는 좌·우 packet이 다르다. reciprocal shortcut은 질량 오차가 작아도 방향/위상을 틀리게 만들었다.
- 이 연산 자체가 positivity/causality를 새로 만드는 것은 아니다. 결과는 공급한 reference operator의 성질을 물려받는다.
- 같은 최종 A의 정확한 inverse 갱신은 endpoint 일관성을 갖는다. 임의의 edit-order 물리성을 가정한 C005와 구별된다.

## F. 비용

Dense base inverse 준비 `O(N³)` 시간/`O(N²)` 메모리. 교란 r개에서 전체 G를 유지하면 `O(rN²)`이며 spatial material 전체를 바꾸는 r≈N이면 다시 `O(N³)`다. Base 준비 비용은 `results/session_0003/preparation_profiles.json`에 별도로 측정했다. 재료가 고정된 cached source 응답은 `O(N²)`다. 정확한 sparse tridiagonal solver의 `O(N)`보다 유리하다고 주장하지 않는다. Dense rank-one 갱신은 GPU 친화적일 수 있지만 순차 dependency와 memory traffic이 있고 GPU 실측은 미실행이다.

## G. 기존 연산과의 구별/판정

Attention의 normalized query-key 평균이나 일반 GNN 합산과 수식은 다르지만 **이미 알려진 inverse-update 수식과 연산 자체가 같다**. 단순히 출력이 Green response라는 이유가 아니라 내부 계산식의 정확한 동일성으로 제외한다. 논문/특허 전체 검색은 미완료다.

**판정: REJECT — 독립 발명으로 제외, finite/sensitivity 비교용 고전 control로 보존.** 열전도와 방향 반전 이류 실험은 Cycle13/17에 있다. 이 성공을 새로운 neural architecture 학습 성공으로 세지 않는다.
