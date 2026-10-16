# Cycle 15 — C — 비선형 변화량 합성

## 가설과 계산 객체
대칭/변환 법칙의 원소로 변화량을 합성한다. A/B의 조합이 아니다.

## 구현 수식
\[
a\star b=a\sqrt{1+\gamma^2b^2}+b\sqrt{1+\gamma^2a^2}
\]

입력 타입·단위·학습 변수·성질·복잡도와 기존 연산 비교는 `../../../architectures/session_0003/candidate_*.md`에 전개했다.

## 수학적 반례와 기존 계산 비교
Raw 연산은 선형 source superposition을 위반한다. asinh 좌표를 되돌린 control은 정확하지만 phi^-1(phi(a)+phi(b))라는 덧셈의 재표현이다.

## 최소 구현·실행·학습 여부
gamma=4 고정, 학습 미실행. 주어진 exact response contributions를 합쳤으므로 정확한 correction을 새 학습 성능으로 세지 않는다.

실행 코드 `experiments/session0003.py`, phase별 source snapshot 및 각 행의 NPZ가 대응한다. 모든 열 식은 -(kT_x)_x=Q, 이류 식은 u_t+v*u_x=k*u_xx다.

## 실제 기록
해당 Cycle의 결과 24행. 표의 상대L2는 전체 배열 norm이며 batch의 mean-per-sample 오차는 JSON 별도 항목이다.

| 후보 | 조건 | 상대L2 | 최대 절대오차 | 중앙시간(s) | 추적 peak(byte) |
|---|---|---:|---:|---:|---:|
| C_raw | uniform_n7 | 0.03245778 | 0.005079839 | 3.8864e-05 | 976 |
| C_conjugate | uniform_n7 | 2.31209e-16 | 4.163336e-17 | 5.6594e-05 | 1128 |
| C_raw | local_n7 | 0.06333007 | 0.01765044 | 3.8448e-05 | 976 |
| C_conjugate | local_n7 | 1.449309e-16 | 2.775558e-17 | 5.2132e-05 | 1128 |
| C_raw | low_material_n7 | 76.49193 | 272.2472 | 3.9714e-05 | 976 |
| C_conjugate | low_material_n7 | 1.665538e-16 | 4.440892e-16 | 5.3474e-05 | 1128 |
| C_raw | signed_source_n7 | 0.0093016 | 0.001060987 | 3.8751e-05 | 976 |
| C_conjugate | signed_source_n7 | 4.212598e-16 | 5.551115e-17 | 5.0984e-05 | 1128 |
| C_raw | uniform_n15 | 0.0333908 | 0.005227284 | 8.1852e-05 | 1232 |
| C_conjugate | uniform_n15 | 2.09896e-16 | 4.163336e-17 | 0.000108164 | 1448 |
| C_raw | local_n15 | 0.09290484 | 0.03041665 | 0.000153929 | 1232 |
| C_conjugate | local_n15 | 3.132228e-16 | 8.326673e-17 | 0.000107647 | 1448 |
| C_raw | low_material_n15 | 278.5096 | 1074.853 | 8.1755e-05 | 1232 |
| C_conjugate | low_material_n15 | 1.928977e-16 | 4.440892e-16 | 0.000107841 | 1448 |
| C_raw | signed_source_n15 | 0.01738922 | 0.002304081 | 8.1752e-05 | 1232 |
| C_conjugate | signed_source_n15 | 5.687261e-16 | 9.714451e-17 | 0.000108987 | 1448 |
| C_raw | uniform_n31 | 0.03361191 | 0.005262314 | 0.000167738 | 1744 |
| C_conjugate | uniform_n31 | 2.369899e-16 | 5.551115e-17 | 0.000260296 | 2064 |
| C_raw | local_n31 | 0.1091875 | 0.03595241 | 0.000198679 | 1720 |
| C_conjugate | local_n31 | 4.515625e-16 | 1.387779e-16 | 0.000268319 | 2088 |
| C_raw | low_material_n31 | 509.6647 | 2049.741 | 0.000222235 | 1744 |
| C_conjugate | low_material_n31 | 3.397651e-16 | 1.332268e-15 | 0.000296179 | 2088 |
| C_raw | signed_source_n31 | 0.02067698 | 0.002676567 | 0.000422408 | 1744 |
| C_conjugate | signed_source_n31 | 4.935699e-16 | 9.714451e-17 | 0.000383748 | 2088 |

물리 flux/balance/BC, batch sample mean/std와 input/prediction/reference는 `results/session_0003/metrics.json` 및 `arrays/`에 보존했다. 시간은7회 CPU 반복 중앙값이다. A/C base inverse 준비는 별도 profile에 있으며 native/BLAS 일부 메모리는 tracemalloc에 잡히지 않는다. RSS는 누적 process high-water다.

## 판정
**REJECT as an independently established invention.**
새 coupling이 아닌 좌표 변화다. 세 후보가 배제되어 실제 공간 순서의 다른 물리 원리 D로 전환한다.

## 미확인 사항
원문 사이트403, 새 공식 README3개와 이전 저장 코드만 확인했다. 정확한 수학적 연산 동일성은 별도로 증명했지만 논문/특허 전체 검색은 미완료다. 실행하지 않은 GPU/Stokes/대규모 데이터 결과를 주장하지 않는다.
