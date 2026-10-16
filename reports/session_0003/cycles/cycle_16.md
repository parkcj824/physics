# Cycle 16 — D — 순서 있는 구간 관계

## 가설과 계산 객체
실제 공간 구간의 순서가 source와 저항의 cross moment를 운반한다.

## 구현 수식
\[
(R,S,C)\circ(R',S',C')=(R+R',S+S',C+C'+R'S)
\]

입력 타입·단위·학습 변수·성질·복잡도와 기존 연산 비교는 `../../../architectures/session_0003/candidate_*.md`에 전개했다.

## 수학적 반례와 기존 계산 비교
정상 열전도를 정확히 계산하나 affine transfer matrix와 동일하다. Root tuple은 boundary port만 설명하며 전체 장에는 모든 prefix 관계가 필요하다.

## 최소 구현·실행·학습 여부
이 단계 theta=1은 고정한 물리 값이다. Cycle18에서 같은128개 label로 scalar를 실제 적합한다. Prefix O(N)과 banded solver O(N)을 비교한다.

실행 코드 `experiments/session0003.py`, phase별 source snapshot 및 각 행의 NPZ가 대응한다. 모든 열 식은 -(kT_x)_x=Q, 이류 식은 u_t+v*u_x=k*u_xx다.

## 실제 기록
해당 Cycle의 결과 16행. 표의 상대L2는 전체 배열 norm이며 batch의 mean-per-sample 오차는 JSON 별도 항목이다.

| 후보 | 조건 | 상대L2 | 최대 절대오차 | 중앙시간(s) | 추적 peak(byte) |
|---|---|---:|---:|---:|---:|
| D_transfer | uniform | 1.206397e-15 | 1.94289e-16 | 2.873e-05 | 2740 |
| D_missing_area | uniform | 0.9806344 | 0.1230469 | 2.8602e-05 | 2587 |
| D_transfer | heterogeneous | 1.692194e-15 | 2.63678e-16 | 2.9561e-05 | 2740 |
| D_missing_area | heterogeneous | 0.9491412 | 0.1285429 | 3.5927e-05 | 2587 |
| D_transfer | low_material | 2.384919e-15 | 3.268497e-13 | 2.8538e-05 | 2587 |
| D_missing_area | low_material | 0.9806344 | 123.0469 | 2.671e-05 | 2587 |
| D_transfer | source_left | 1.32482e-15 | 2.359224e-16 | 2.6977e-05 | 2587 |
| D_missing_area | source_left | 1.211687 | 0.191525 | 2.9749e-05 | 2587 |
| D_transfer | source_right | 1.452419e-15 | 9.714451e-17 | 2.7848e-05 | 2587 |
| D_missing_area | source_right | 0.7125804 | 0.04484178 | 3.1466e-05 | 2587 |
| D_transfer | boundary_change | 2.718711e-16 | 2.775558e-16 | 2.7212e-05 | 2740 |
| D_missing_area | boundary_change | 2.718711e-16 | 2.775558e-16 | 2.6847e-05 | 2587 |
| D_transfer | resolution7 | 2.888239e-16 | 4.163336e-17 | 2.8719e-05 | 1947 |
| D_missing_area | resolution7 | 0.9274681 | 0.09482087 | 3.1083e-05 | 1947 |
| D_transfer | resolution31 | 1.948694e-15 | 9.714451e-17 | 2.7767e-05 | 3867 |
| D_missing_area | resolution31 | 0.9791925 | 0.04355198 | 2.6353e-05 | 3867 |

물리 flux/balance/BC, batch sample mean/std와 input/prediction/reference는 `results/session_0003/metrics.json` 및 `arrays/`에 보존했다. 시간은7회 CPU 반복 중앙값이다. A/C base inverse 준비는 별도 profile에 있으며 native/BLAS 일부 메모리는 tracemalloc에 잡히지 않는다. RSS는 누적 process high-water다.

## 판정
**REJECT as an independently established invention.**
물리적으로 유효한 고전 control로 보존한다. 단순 matrix 표현이라는 이유가 아니라 전체 transfer 알고리즘과 동일하여 신규 주장하지 않는다.

## 미확인 사항
원문 사이트403, 새 공식 README3개와 이전 저장 코드만 확인했다. 정확한 수학적 연산 동일성은 별도로 증명했지만 논문/특허 전체 검색은 미완료다. 실행하지 않은 GPU/Stokes/대규모 데이터 결과를 주장하지 않는다.
