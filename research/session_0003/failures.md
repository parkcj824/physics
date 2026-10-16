# Session 0003 실패와 배제 이유

## A: finite response packet

Frozen packets의 낮은k 상대오차0.92661, 비영 BC·비균질 재료0.51484. 현재 G를 갱신하면 각각 약5.6e-14,1e-15로 정확해지지만 그 수정이 Sherman–Morrison다. 단일 유한 교란에서 tangent는 상대오차0.025509였고 finite denominator는 direct solve와 일치했다. 물리적 효과는 확인했지만 계산 규칙 신규성은 없다. 이 결과가 모든 response-learning architecture의 불가능성을 뜻하지는 않는다.

## B: witness-only 표현

R의 null source를 구성했다. `b±=1±0.8delta`는 양수이고 coarse witness와 source norm이 같지만 해가 다르다. `b±=±delta`에서는 동일 zero witness로 반대 해를 구해야 해 어떤 deterministic decoder도 모두 맞힐 수 없다. 순환 이류에서 같은 질량/에너지라도 초기 위상이 다르면 후속 해가 다르다(선택한 canonical phase의 상대오차0.23230).

실험 minimum-norm decoder의 약99% 오차는 그 decoder의 점수이며 모든 가능한 decoder의 최적 하한으로 해석하지 않는다. 원리적 배제 근거는 같은 입력 witness에 다른 target이 존재한다는 것이다. 제한된 분포의 근사·보조 certificate·충분한 full-information witness의 가능성은 별도다. Full-rank closure는 기존 linear solve가 된다.

## C: 물리 상태에 잘못 적용한 비선형 군

Raw 합성은 source superposition을 위반한다. N15 낮은 재료 조건에서 상대오차278.51로 커졌다. 이류에서도 상대오차1.3251, 평균 질량 증가1.24706이었다. Encoding/decoding을 맞춰 고치면 associative 합성이 정확히 보통 덧셈으로 돌아간다. Gamma나 packet 수를 늘리는 것은 독립 primitive가 아니다. Supplied Green/propagator가 정확한 것이지 이 규칙을 학습한 결과가 아니다.

## D: 물리 성공이지만 기존 알고리즘

공간 ordered cross moment는 실제로 필요했다. 제거하면 uniform source 상대오차0.98063, source-left1.2117. Theta1이면 variable k/BC/source/resolution을 direct solve와 약1e-15 수준으로 일치시킨다. 학습 theta도1.0000000000000002였다. 그러나 전체 수식이 affine transfer/prefix 계산이고 학습은 기존 항의 coefficient 적합이다. 새 primitive나 CAE foundation을 주장하지 않는다. Root tuple3개만으로 전체 field를 알 수 없으며 모든 prefix 관계를 보관했다.

## 방향성과 global invariant의 구분

Reciprocal shortcut은 Pe20 역방향에서 상대오차0.10542였지만 질량 오차는~1e-15다. 올바른 좌·우 response를 유지하면 reference와 약1e-15로 일치했다. 질량 보존 검사만으로 방향/위상 예측 성공을 결론 내리지 않는다. BE reference와 exact semidiscrete solution의 시간오차도 별도다.

## 나쁜 기준선의 자기 반박

Unregularized feature-linear는 새 source 위치에서 mean relative error388500.7766을 냈다. Retained 작은 singular value~1.1e-7, coefficient norm~1.67e6이 진단됐다. Label noise만의 문제가 아니라 variable material의 nonlinear dependence를 거의 관측되지 않는 feature 방향으로 맞추면서 OOD에서 증폭한 현상이다. Validation32개만으로 alpha를 선택한 weighted ridge(alpha1)는 같은 조건에서0.214296이었다.

수정에는 ridge regularization뿐 아니라 상대오차 기준의 sample weighting도 포함된다. 둘의 효과를 분리한 ablation은 실행하지 않았다. Test 실패를 관찰한 뒤 baseline class를 보완했으므로 이 비교는 **탐색적**이다. Alpha fitting/selection에는 OOD target을 사용하지 않았지만, 완전히 blind confirmatory evaluation이라고 주장하지 않는다. 초기 실패와 보정 모델/점수를 모두 보존했다. Ridge 선택으로 현재 primitive의 신규성/우월성이 입증된 것은 아니다.

## 제외한 추가 계산

위 가설은 구조적 반례 또는 정확한 계산 동등성이 확인되어 더 큰 network/더 많은 데이터/GPU로 확대하지 않았다. 새 독립 primitive가 남지 않아 Stokes 확장은 미실행이다. D를 transient time stepper로 억지 해석하지 않았다. RANS/2D 신경 architecture/대규모 GPU/회사 데이터는 사용하지 않았다. 다음 branch-set 가설은 아직 수식·구현·실험이 없다.
