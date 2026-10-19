# 보조 재현 계측과 데이터 비용

원래 J/K/closure 실행의 전체 runtime을 기록하지 않았으므로 아래 값을 원래 측정값으로 소급하지 않는다. 같은 저장 조건의 재현 계산을 별도로 실행한 7회 중앙값이다. refit/새 발견이 아니다. 일부 setup만 측정했던 J의 원래 값도 그대로 보존한다.

| 계산 | 재현 median ms | Python peak bytes | 범위 |
|---|---|---|---|
| J heat | 0.87753 | 26352 | dense operators, spectral norm setup and128 gradient steps; reference labels already available |
| J lowk | 0.72958 | 26352 | dense operators, spectral norm setup and128 gradient steps; reference labels already available |
| J adv_plus | 0.76015 | 26352 | dense operators, spectral norm setup and128 gradient steps; reference labels already available |
| J adv_minus | 0.811 | 26352 | dense operators, spectral norm setup and128 gradient steps; reference labels already available |
| K | 0.56677 | 62013 | transition exponential, training eight histories, two-coefficient LS and32 predictions including first observation generation |
| Closure N15 | 1.2112 | 16408 | base root, analytical source construction, two total-source roots and dense one-Newton corrections |
| Closure N31 | 0.86922 | 44088 | base root, analytical source construction, two total-source roots and dense one-Newton corrections |

Tracemalloc은 Python 중심이며 native tensor/BLAS allocation을 완전히 측정하지 못한다. 전체 peak 메모리의 비교나 우월성으로 읽지 않는다. C0/F 추론의 측정값도 같은 한계가 있다.

| 물리/확인 batch | 모델 | 평균 Python peak bytes |
|---|---|---|
| G | original | 2002 |
| G | physics_plus_feature | 2002 |
| G | learned_feature | 1689 |
| G | static_physics | 1900 |
| G | direct_matched | 1577 |
| G | learned_jacobi | 1919 |
| H | original | 2054 |
| H | physics_plus_feature | 2071 |
| H | learned_feature | 1769 |
| H | static_physics | 1980 |
| H | direct_matched | 1657 |
| H | learned_jacobi | 2176 |
| F pair | reaction_gate | 2605 |
| F pair | reaction_no_gate | 2649 |
| F pair | bilinear | 2627 |
| F pair | volterra3 | 2581 |
| F pair | attention | 3831 |

## Label 및 실제 저장 비용

| Dataset | Input sets | 저장된 subset reference records | NPZ bytes | 배열 bytes |
|---|---|---|---|---|
| data/C0_G_confirm_boundary.npz | 16 | 단일 reference/set | 8926 | 8568 |
| data/C0_G_confirm_iid.npz | 64 | 단일 reference/set | 32549 | 33912 |
| data/C0_G_confirm_lowk.npz | 16 | 단일 reference/set | 8984 | 8568 |
| data/C0_G_confirm_positive.npz | 16 | 단일 reference/set | 8673 | 8568 |
| data/C0_G_confirm_source.npz | 16 | 단일 reference/set | 8952 | 8568 |
| data/C0_G_train.npz | 128 | 단일 reference/set | 63976 | 67704 |
| data/C0_G_val.npz | 32 | 단일 reference/set | 16834 | 17016 |
| data/C0_H_confirm_direction.npz | 32 | 단일 reference/set | 8669 | 16760 |
| data/C0_H_confirm_iid.npz | 64 | 단일 reference/set | 29476 | 33400 |
| data/C0_H_confirm_lowdiff.npz | 16 | 단일 reference/set | 8074 | 8440 |
| data/C0_H_train.npz | 128 | 단일 reference/set | 58108 | 66680 |
| data/C0_H_val.npz | 32 | 단일 reference/set | 15197 | 16760 |
| data/C0b_G_confirm.npz | 64 | 단일 reference/set | 32552 | 33912 |
| data/C0b_H_confirm.npz | 64 | 단일 reference/set | 29445 | 33400 |
| data/F5_far.npz | 16 | 64 | 8394 | 33528 |
| data/F5_four.npz | 16 | 256 | 31301 | 35064 |
| data/F5_linear.npz | 16 | 64 | 8190 | 33528 |
| data/F5_lowk_linear.npz | 16 | 64 | 8214 | 33528 |
| data/F5_pair.npz | 32 | 128 | 16010 | 66936 |
| data/F5_shift.npz | 16 | 64 | 8404 | 33528 |
| data/F5_single.npz | 16 | 32 | 4924 | 33272 |
| data/F5_tiny.npz | 16 | 64 | 8653 | 33528 |
| data/F5_train.npz | 128 | 602 | 68373 | 268104 |
| data/F5_triple.npz | 32 | 256 | 31262 | 67960 |
| data/F5_val.npz | 32 | 178 | 21632 | 67336 |

초기 준비 wall-time은 dataset_manifest에 저장했다. 추가 low-k 생성 시간은 최초 실행에 별도 기록하지 않았다. 모든 subset 생성 수는 훈련에 사용한 독립 label 수와 다르다. Encoder용 base/singleton과 head용 full mixed를 사용했고 중복 base는 여러 set에 반복된다.
