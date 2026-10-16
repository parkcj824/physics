# Cycle 17 — 방향성·보존·위상 반증

## 가설과 계산 객체
비자기수반 response의 좌·우 정보를 구별하고 global witness가 위상을 결정하는지 확인한다.

## 구현 수식
\[
G^+=G-(Ga)(c^TG)/(1+c^TGa),\quad u_{n+1}=G^+u_n
\]

입력 타입·단위·학습 변수·성질·복잡도와 기존 연산 비교는 `../../../architectures/session_0003/candidate_*.md`에 전개했다.

## 수학적 반례와 기존 계산 비교
Full oriented update는 backward Euler reference와 일치한다. Reciprocal shortcut은 질량 보존 오차가 작아도 위상을 틀린다. 같은 질량/에너지의 다른 초기 위상도 구별해야 한다.

## 최소 구현·실행·학습 여부
신규 학습 없음. Reference는 upwind 공간 이산+backward Euler; 정확한 semidiscrete exponential과의 시간 오차를 별도 저장했다. D는 transient 규칙을 정의하지 않으므로 억지 적용하지 않았다.

실행 코드 `experiments/session0003.py`, phase별 source snapshot 및 각 행의 NPZ가 대응한다. 모든 열 식은 -(kT_x)_x=Q, 이류 식은 u_t+v*u_x=k*u_xx다.

## 실제 기록
해당 Cycle의 결과 15행. 표의 상대L2는 전체 배열 norm이며 batch의 mean-per-sample 오차는 JSON 별도 항목이다.

| 후보 | 조건 | 상대L2 | 최대 절대오차 | 중앙시간(s) | 추적 peak(byte) |
|---|---|---:|---:|---:|---:|
| A_oriented | Pe0.2_v-1.0 | 1.589557e-15 | 2.88658e-15 | 0.000376541 | 25216 |
| A_reciprocal_shortcut | Pe0.2_v-1.0 | 0.01454475 | 0.02856146 | 0.000333165 | 25024 |
| A_oriented | Pe0.2_v1.0 | 1.901156e-15 | 3.108624e-15 | 0.000246458 | 25216 |
| A_reciprocal_shortcut | Pe0.2_v1.0 | 0.01395729 | 0.02299821 | 0.000232036 | 25024 |
| A_oriented | Pe2.0_v-1.0 | 5.280295e-16 | 1.332268e-15 | 0.00022194 | 25216 |
| A_reciprocal_shortcut | Pe2.0_v-1.0 | 0.06361362 | 0.2607053 | 0.000224713 | 25024 |
| A_oriented | Pe2.0_v1.0 | 9.8428e-16 | 1.998401e-15 | 0.000270511 | 25216 |
| A_reciprocal_shortcut | Pe2.0_v1.0 | 0.05993745 | 0.1130417 | 0.000211262 | 25024 |
| A_oriented | Pe20.0_v-1.0 | 8.95261e-16 | 2.220446e-15 | 0.000296452 | 25216 |
| A_reciprocal_shortcut | Pe20.0_v-1.0 | 0.1054163 | 0.4916016 | 0.000206498 | 25024 |
| A_oriented | Pe20.0_v1.0 | 1.062774e-15 | 2.220446e-15 | 0.000217201 | 25216 |
| A_reciprocal_shortcut | Pe20.0_v1.0 | 0.09502981 | 0.2445424 | 0.000286526 | 25024 |
| B_phase_collision | mass_energy_equal | 0.2323031 | 0.3323104 | 3.279999e-07 | 288 |
| C_raw | transient_superposition | 1.325101 | 1.961151 | 1.2653e-05 | 1496 |
| C_conjugate | transient_superposition | 1.13986e-16 | 2.220446e-16 | 1.772e-05 | 1808 |

물리 flux/balance/BC, batch sample mean/std와 input/prediction/reference는 `results/session_0003/metrics.json` 및 `arrays/`에 보존했다. 시간은7회 CPU 반복 중앙값이다. A/C base inverse 준비는 별도 profile에 있으며 native/BLAS 일부 메모리는 tracemalloc에 잡히지 않는다. RSS는 누적 process high-water다.

## 판정
**REJECT as an independently established invention.**
새 범용 primitive가 검증된 것은 아니다. Stokes는 새 후보가 남지 않아 확장하지 않는다. Cycle18에서 강한 기준선과 비용을 검토한다.

## 미확인 사항
원문 사이트403, 새 공식 README3개와 이전 저장 코드만 확인했다. 정확한 수학적 연산 동일성은 별도로 증명했지만 논문/특허 전체 검색은 미완료다. 실행하지 않은 GPU/Stokes/대규모 데이터 결과를 주장하지 않는다.
