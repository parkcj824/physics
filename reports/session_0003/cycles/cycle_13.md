# Cycle 13 — A — 유한 response packet

## 가설과 계산 객체
유한 교란의 반응과 cross-susceptibility를 기본 단위로 본다.

## 구현 수식
\[
G^+=G-\alpha(Gd)(d^TG)/(1+\alpha d^TGd)
\]

입력 타입·단위·학습 변수·성질·복잡도와 기존 연산 비교는 `../../../architectures/session_0003/candidate_*.md`에 전개했다.

## 수학적 반례와 기존 계산 비교
분모 없는 일차 감도와 frozen packet을 대조했다. 정확한 수정은 Sherman–Morrison와 연산 자체가 같다.

## 최소 구현·실행·학습 여부
학습하지 않았다. Exact base inverse를 공급한 oracle/control이며 그 준비 O(N³), 각 갱신 O(N²)를 별도 비용으로 기록했다.

실행 코드 `experiments/session0003.py`, phase별 source snapshot 및 각 행의 NPZ가 대응한다. 모든 열 식은 -(kT_x)_x=Q, 이류 식은 u_t+v*u_x=k*u_xx다.

## 실제 기록
해당 Cycle의 결과 12행. 표의 상대L2는 전체 배열 norm이며 batch의 mean-per-sample 오차는 JSON 별도 항목이다.

| 후보 | 조건 | 상대L2 | 최대 절대오차 | 중앙시간(s) | 추적 peak(byte) |
|---|---|---:|---:|---:|---:|
| A_finite | uniform_low | 5.602499e-14 | 4.920508e-13 | 0.000163409 | 13688 |
| A_frozen | uniform_low | 0.926606 | 5.849178 | 0.000197678 | 7736 |
| A_finite | interface | 3.643919e-15 | 2.331468e-15 | 0.00032401 | 13688 |
| A_frozen | interface | 0.1513431 | 0.05957501 | 0.000110494 | 7736 |
| A_finite | source_left | 1.510641e-16 | 2.775558e-17 | 1.3685e-05 | 6945 |
| A_frozen | source_left | 1.510641e-16 | 2.775558e-17 | 1.3669e-05 | 6969 |
| A_finite | source_right | 1.209838e-15 | 4.857226e-17 | 0.000435402 | 13712 |
| A_frozen | source_right | 0.4289972 | 0.01429589 | 0.000124643 | 7760 |
| A_finite | nonzero_boundary | 1.006299e-15 | 1.332268e-15 | 0.00015824 | 13712 |
| A_frozen | nonzero_boundary | 0.5148406 | 0.5146283 | 8.1901e-05 | 7760 |
| A_finite | single_finite_perturbation | 6.626115e-16 | 1.110223e-16 | 0.000197959 | 13712 |
| A_tangent | single_finite_perturbation | 0.0255088 | 0.004298998 | 2.3133e-05 | 7736 |

물리 flux/balance/BC, batch sample mean/std와 input/prediction/reference는 `results/session_0003/metrics.json` 및 `arrays/`에 보존했다. 시간은7회 CPU 반복 중앙값이다. A/C base inverse 준비는 별도 profile에 있으며 native/BLAS 일부 메모리는 tracemalloc에 잡히지 않는다. RSS는 누적 process high-water다.

## 판정
**REJECT as an independently established invention.**
고전 inverse update로서 유효하지만 신규 primitive가 아니다. B의 witness 충분성을 검토한다.

## 미확인 사항
원문 사이트403, 새 공식 README3개와 이전 저장 코드만 확인했다. 정확한 수학적 연산 동일성은 별도로 증명했지만 논문/특허 전체 검색은 미완료다. 실행하지 않은 GPU/Stokes/대규모 데이터 결과를 주장하지 않는다.
