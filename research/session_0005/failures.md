# Session 0005 실패·반증 원장

## C0/C0b — G의 물리 feature 고유 이득 미입증

같은 활성 θ9472개의 physics-plus와 generic을 비교해도 물리 feature의 보편적 장점을 얻지 못했다. 열 IID에서는 generic/static이 더 좋았고, H에서는 field와 PDE 순위가 달랐다. Conditioning 통제를 새 확인 집합으로 추가해도 결과는 조건에 의존했다. 추가 파라미터/반복/크기 효과를 완전히 제거한 보편적 인과 증명은 아니다. 기존 G의 장점을 부정하는 모든 분포의 정리로 해석하지 않는다.

수지 정확성은 구성식/PDE/positivity를 보장하지 않았다. G positive-source/BC0에서 원래 feedback의 음수 표본 수는 16개 중 seed별5/1/6개다. Low-k field 오차는 원래 G도69.35%다. H 유속 반전의 centroid를 맞혀도 음수 상태가 나왔다. `controls_confirmation.json`, `normalized_confirmation.json` 및 원시 arrays를 보존한다.

## J /Cycle24 — joint gradient는 새로운 계산 규칙이 아님

분리된 b/c 유한 증분은 진단에 유용하지만 시도한 동시 갱신은 `−ηJ_FᵀF`와 같다. 작은 field L2 0.07746%, 정확한 생성 수지에도 PDE∞0.1014다. 안정적인 defect-gradient step에서도 양의 온도를 −0.1326으로 바꾸는 joint-state 반례가 있다. Low-k에서는128step 후 field 오차99.998%로, defect 감소를 해 정확성으로 읽으면 안 된다. 학습을 더 크게 해서 구조적 결함을 덮지 않았다.

## L/F5 /Cycle25 — 공간 활성화의 효용은 제한됨

Pair mixed의 절대오차 일부는 bilinear보다 작지만 cubic이 triple/four의 고차 효과를 더 잘 맞혔다. Tiny source에서는 진짜 mixed 크기6.31e−9를 gate가 제거한다. 높은 상대오차만으로 의미를 과장하지 않으며 절대 크기도 기록한다. Source 위치 OOD field 오차29.32%는 공유 encoder 성분이 지배한다. 이 오류를 head만의 책임으로 돌리거나 head 성능을 전체 model의 개선으로 부풀리지 않는다.

선택 대상은 source pair topology가 아닌 공간 H다. Dense kernel이 실행되므로 activation이 줄어도 실제 계산은 건너뛰지 않는다. 계산 graph는 sum pooling으로 정확히 재구성된다. Level3를 입증하지 못해 Level2로 유지했다. λ0 low-k 결과를 nonlinear OOD 성공으로 재사용하지 않는다.

## K /Cycle26 — 불완전 관측의 이득은 신규 memory가 아님

온도가 같고 flux가 다른 상태의 미래는 다르다. 두 snapshot의 이득은 AR2/관측가능성으로 설명된다. 특정 간격에서 두 snapshot도 구별 불가능하지만 이후 차이가0.0410이다. 고정 계수를 새 dt에 적용하면 최대 오차0.08784다. 완전한 Cattaneo(T,j) state는 Markov이며, Bratu의 완전한 온도 상태가 비Markov라는 주장을 하지 않는다.

## L의 범위 /Cycle27 — H만으로는 보정이 결정되지 않음

두 양수 source set에서 정확한 H는 같지만 S와 실제 필요한 correction이 다르다. N15에서 correction 차이0.00630886, deterministic H-only closure의 최악 절대 L2 오차 하한0.00315443이다. 입력 정보 충분성의 한계이며 데이터 부족/optimizer 문제와 다르다. Footprint가 물리적으로 충실하다는 가정이 중요하다. 실제 학습분포 밖 Gaussian이 아닌 source를 사용했으므로 IID 모델 오차 결과라고 주장하지 않는다.

S를 포함한 Newton1step은 오차를 줄이지만 고전 알고리즘이다. 이것을 새로운 후보라고 부르지 않는다. 다음 세션에서 필요한 것은 rank/order/폭 증가가 아니라 state 문맥을 보존하는 새로운 학습 가능한 규칙이다.
