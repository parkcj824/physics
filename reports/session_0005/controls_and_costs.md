# C0/C0b 통제 및 비용 부록

세 seed 평균 ± seed 간 표준편차. 양수 조건의 분모는 G positive 16개, H direction 32개이며 그 밖에는 양수 조건 표본이 없을 수 있다. 값 0을 무조건적 positivity 성공으로 읽지 않는다.

| 물리 | 조건 | 모델 | Field L2 (%) | PDE ∞ | 음수 표본 수 / seed | 방향 정답 쌍 /16 |
|---|---|---|---|---|---|---|
| G | lowk | original | 69.35 ± 9.1 | 5.76 ± 1.8 | 0/0/0 | —/—/— |
| G | lowk | physics_plus_feature | 86.3 ± 15 | 5.673 ± 1.2 | 0/0/0 | —/—/— |
| G | lowk | learned_feature | 98.59 ± 15 | 2.139 ± 0.32 | 0/0/0 | —/—/— |
| G | lowk | static_physics | 89.66 ± 25 | 4.996 ± 2.1 | 0/0/0 | —/—/— |
| G | lowk | direct_matched | 82.55 ± 21 | 1.522 ± 0.26 | 0/0/0 | —/—/— |
| G | lowk | learned_jacobi | 79.8 ± 0 | 0.6143 ± 0 | 0/0/0 | —/—/— |
| G | source | original | 24.57 ± 7.4 | 11.93 ± 2.1 | 0/0/0 | —/—/— |
| G | source | physics_plus_feature | 22.99 ± 5.4 | 14 ± 1.8 | 0/0/0 | —/—/— |
| G | source | learned_feature | 14.07 ± 1.8 | 4.793 ± 1.5 | 0/0/0 | —/—/— |
| G | source | static_physics | 17.77 ± 4.4 | 10.99 ± 2.5 | 0/0/0 | —/—/— |
| G | source | direct_matched | 22.95 ± 4.4 | 2.615 ± 0.38 | 0/0/0 | —/—/— |
| G | source | learned_jacobi | 53.44 ± 0 | 1.277 ± 0 | 0/0/0 | —/—/— |
| G | boundary | original | 4.672 ± 1.3 | 23.77 ± 5.8 | 0/0/0 | —/—/— |
| G | boundary | physics_plus_feature | 4.351 ± 1 | 23.43 ± 3.8 | 0/0/0 | —/—/— |
| G | boundary | learned_feature | 8.669 ± 4.5 | 5.5 ± 0.26 | 0/0/0 | —/—/— |
| G | boundary | static_physics | 4.519 ± 0.52 | 18.99 ± 2.5 | 0/0/0 | —/—/— |
| G | boundary | direct_matched | 8.531 ± 2.7 | 4.181 ± 0.77 | 0/0/0 | —/—/— |
| G | boundary | learned_jacobi | 6.844 ± 0 | 0.8096 ± 0 | 0/0/0 | —/—/— |
| G | positive | original | 73.41 ± 37 | 12.46 ± 2.1 | 5/1/6 | —/—/— |
| G | positive | physics_plus_feature | 63.77 ± 26 | 13.58 ± 3.5 | 7/5/3 | —/—/— |
| G | positive | learned_feature | 26.58 ± 3.9 | 3.438 ± 0.87 | 1/3/4 | —/—/— |
| G | positive | static_physics | 41.55 ± 6 | 9.67 ± 0.61 | 3/3/3 | —/—/— |
| G | positive | direct_matched | 57.69 ± 14 | 1.875 ± 0.2 | 4/6/7 | —/—/— |
| G | positive | learned_jacobi | 93.97 ± 0 | 0.7728 ± 0 | 0/0/0 | —/—/— |
| H | lowdiff | original | 39.02 ± 2.6 | 1.549 ± 0.09 | 0/0/0 | —/—/— |
| H | lowdiff | physics_plus_feature | 47.1 ± 5.1 | 1.643 ± 0.12 | 0/0/0 | —/—/— |
| H | lowdiff | learned_feature | 59.91 ± 25 | 1.601 ± 0.14 | 0/0/0 | —/—/— |
| H | lowdiff | static_physics | 50.18 ± 5.7 | 1.718 ± 0.093 | 0/0/0 | —/—/— |
| H | lowdiff | direct_matched | 114.4 ± 10 | 2.439 ± 0.06 | 0/0/0 | —/—/— |
| H | lowdiff | learned_jacobi | 74.95 ± 0 | 1.021 ± 0 | 0/0/0 | —/—/— |
| H | direction | original | 11.32 ± 3.1 | 0.7454 ± 0.011 | 2/7/5 | 16/16/16 |
| H | direction | physics_plus_feature | 15.66 ± 3.1 | 0.7794 ± 0.1 | 0/0/0 | 16/16/16 |
| H | direction | learned_feature | 21.73 ± 4.9 | 0.6443 ± 0.25 | 11/0/8 | 16/16/16 |
| H | direction | static_physics | 22.04 ± 3.2 | 0.7508 ± 0.013 | 0/5/0 | 16/16/16 |
| H | direction | direct_matched | 39.13 ± 13 | 0.7314 ± 0.17 | 5/16/4 | 16/14/16 |
| H | direction | learned_jacobi | 89.44 ± 1.7e-14 | 0.6874 ± 1.4e-16 | 0/0/0 | 16/16/16 |

## 학습·추론 비용

| 물리 | 모델 | 전체 θ | 활성 θ | Steps | Profile MOp/batch | 총 profile GOp | 학습 초 | 추론 ms/64 |
|---|---|---|---|---|---|---|---|---|
| G | original | 8704 | 8704 | 1000 | 17.536 | 17.536 | 5.122 ± 0.14 | 1.055 ± 0.13 |
| G | physics_plus_feature | 9472 | 9472 | 1000 | 19.313 | 19.313 | 5.14 ± 0.02 | 1.173 ± 0.099 |
| G | learned_feature | 9472 | 9472 | 1000 | 17.470 | 17.470 | 3.501 ± 0.062 | 0.4767 ± 0.048 |
| G | static_physics | 8704 | 8704 | 1000 | 16.312 | 16.312 | 10.85 ± 12 | 1.335 ± 0.97 |
| G | direct_matched | 14623 | 8415 | 2000 | 12.093 | 24.186 | 5.063 ± 0.11 | 0.2891 ± 0.13 |
| G | learned_jacobi | 3 | 3 | 1000 | 0.081 | 0.081 | 2.101 ± 0.018 | 0.3019 ± 0.041 |
| H | original | 8704 | 8704 | 1000 | 17.562 | 17.562 | 9.7 ± 6.1 | 1.335 ± 0.19 |
| H | physics_plus_feature | 9472 | 9472 | 1000 | 19.338 | 19.338 | 8.924 ± 1.8 | 1.281 ± 0.085 |
| H | learned_feature | 9472 | 9472 | 1000 | 17.470 | 17.470 | 3.959 ± 0.11 | 0.5793 ± 0.11 |
| H | static_physics | 8704 | 8704 | 1000 | 16.321 | 16.321 | 4.975 ± 0.11 | 0.7089 ± 0.0063 |
| H | direct_matched | 14623 | 8415 | 2000 | 12.093 | 24.186 | 5 ± 0.19 | 0.2394 ± 0.032 |
| H | learned_jacobi | 3 | 3 | 1000 | 0.099 | 0.099 | 3.12 ± 0.29 | 0.422 ± 0.11 |
| G | norm_physical | 9472 | 9472 | 1000 | 19.370 | 19.370 | 10.43 ± 1.4 | 1.402 ± 0.21 |
| G | norm_generic | 9472 | 9472 | 1000 | 17.489 | 17.489 | 4.222 ± 0.2 | 0.6022 ± 0.057 |
| G | norm_static | 9472 | 9472 | 1000 | 18.134 | 18.134 | 6.018 ± 0.074 | 1.181 ± 0.56 |
| H | norm_physical | 9472 | 9472 | 1000 | 19.396 | 19.396 | 8.568 ± 0.57 | 2.346 ± 0.75 |
| H | norm_generic | 9472 | 9472 | 1000 | 17.489 | 17.489 | 4.224 ± 0.14 | 0.8667 ± 0.36 |
| H | norm_static | 9472 | 9472 | 1000 | 18.142 | 18.142 | 6.881 ± 0.33 | 1.117 ± 0.071 |

활성 θ는 훈련 loss backward에서 grad가 존재하는 parameter의 수다. Forward를 실행하지 않는 Direct의 gate module은 전체 θ에만 포함된다. 이 수가 모든 parameter의 유의미한 기능/identifiability를 보장하지 않는다. Profile M/GOp는 지원되는 연산 일부만 포함한다. 학습 시간은 실제 end-to-end optimizer loop, 추론은 fresh batch64 기준이며 현 CPU 환경의 관측값이다. 전체 native 메모리와 정확한 FLOPs/동일 wall-time budget을 주장하지 않는다.

## 고전 기준선

| 물리 | 고전 기준선 | Field L2 (%) | PDE ∞ | ms/64 |
|---|---|---|---|---|
| G | exact | 0 | 3.674e-12 | 9.527 |
| G | jacobi3 | 58.44 | 1.485 | 0.1211 |
| G | jacobi32 | 38.9 | 0.9617 | 1.177 |
| H | exact | 7.91e-14 | 4.772e-15 | 2.901 |
| H | jacobi3 | 76.68 | 1.023 | 0.179 |
| H | jacobi32 | 28.91 | 0.3745 | 1.397 |

Exact G는 같은 원래 banded solver를 다시 호출하므로 오차 0이 자연스럽다. 별도 dense/root 검증은 integrity.json에 있다. Jacobi3/32는 vectorized NumPy 구현이며 각 호출에 torch module을 새로 만드는 비용을 넣지 않았다.
