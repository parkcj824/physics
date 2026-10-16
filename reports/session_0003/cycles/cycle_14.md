# Cycle 14 — B — witness 충분성

## 가설과 계산 객체
해 대신 coarse balance 관계와 전역 invariant를 표현한다.

## 구현 수식
\[
w=Rb,\quad u=(RA)^+w
\]

입력 타입·단위·학습 변수·성질·복잡도와 기존 연산 비교는 `../../../architectures/session_0003/candidate_*.md`에 전개했다.

## 수학적 반례와 기존 계산 비교
동일 witness·같은 source norm을 갖는 다른 양수 source와 다른 해를 구성했다. Signed pair는 어떠한 decoder도 동시에 정확히 예측할 수 없다.

## 최소 구현·실행·학습 여부
원리적 식별 불가능성을 먼저 검사해 큰 decoder를 학습하지 않았다. 실험 decoder는 minimum-norm closure이고 그 약99% 오차가 최적 하한이라는 뜻은 아니다.

실행 코드 `experiments/session0003.py`, phase별 source snapshot 및 각 행의 NPZ가 대응한다. 모든 열 식은 -(kT_x)_x=Q, 이류 식은 u_t+v*u_x=k*u_xx다.

## 실제 기록
해당 Cycle의 결과 6행. 표의 상대L2는 전체 배열 norm이며 batch의 mean-per-sample 오차는 JSON 별도 항목이다.

| 후보 | 조건 | 상대L2 | 최대 절대오차 | 중앙시간(s) | 추적 peak(byte) |
|---|---|---:|---:|---:|---:|
| B_witness | signed_plus | 1 | 0.007944269 | 1.7886e-05 | 3084 |
| B_witness | signed_minus | 1 | 0.007944269 | 1.5097e-05 | 3084 |
| B_witness | positive_plus | 0.9929032 | 0.118668 | 1.9212e-05 | 3084 |
| B_witness | positive_minus | 0.9940985 | 0.131332 | 1.9263e-05 | 3084 |
| B_witness | rank4 | 0.9935417 | 0.125 | 1.5631e-05 | 3084 |
| B_full_control | rank15 | 3.246429e-15 | 5.551115e-16 | 3.7038e-05 | 4580 |

물리 flux/balance/BC, batch sample mean/std와 input/prediction/reference는 `results/session_0003/metrics.json` 및 `arrays/`에 보존했다. 시간은7회 CPU 반복 중앙값이다. A/C base inverse 준비는 별도 profile에 있으며 native/BLAS 일부 메모리는 tracemalloc에 잡히지 않는다. RSS는 누적 process high-water다.

## 판정
**REJECT as an independently established invention.**
현재 witness-only 표현을 제외한다. 보조 인증 용도 전체를 반증한 것은 아니다. C에서 변환군 기반 합성을 시험한다.

## 미확인 사항
원문 사이트403, 새 공식 README3개와 이전 저장 코드만 확인했다. 정확한 수학적 연산 동일성은 별도로 증명했지만 논문/특허 전체 검색은 미완료다. 실행하지 않은 GPU/Stokes/대규모 데이터 결과를 주장하지 않는다.
