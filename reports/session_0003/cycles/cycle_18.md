# Cycle 18 — 공정 비교와 baseline 자기반박

## 가설과 계산 객체
동일한 heterogeneous 물리 label에서 알려진 cross coefficient와 일반 surrogate를 비교한다.

## 구현 수식
\[
T_\theta=a+\theta s,\quad \hat\theta=\frac{\sum s^T(y-a)/\|y\|^2}{\sum\|s\|^2/\|y\|^2}
\]

입력 타입·단위·학습 변수·성질·복잡도와 기존 연산 비교는 `../../../architectures/session_0003/candidate_*.md`에 전개했다.

## 수학적 반례와 기존 계산 비교
Scalar는1로 복원된다. Unregularized 선형 control의 ill-conditioning을 진단하고 validation-selected ridge를 추가했다. 동일 label 수는 동일 prior/compute budget이 아니다.

## 최소 구현·실행·학습 여부
D/feature-linear/MLP는 동일128개 label. MLP는3seed,1000Adam step. Ridge는 validation32개만으로5개 alpha 선택. Unit-compliance는 별도 unit-k label128개, C006은 이전 pretraining이므로 matched-data 주장하지 않는다.

실행 코드 `experiments/session0003.py`, phase별 source snapshot 및 각 행의 NPZ가 대응한다. 모든 열 식은 -(kT_x)_x=Q, 이류 식은 u_t+v*u_x=k*u_xx다.

## 실제 기록
해당 Cycle의 결과 45행. 표의 상대L2는 전체 배열 norm이며 batch의 mean-per-sample 오차는 JSON 별도 항목이다.

| 후보 | 조건 | 상대L2 | 최대 절대오차 | 중앙시간(s) | 추적 peak(byte) |
|---|---|---:|---:|---:|---:|
| D_fitted_transfer | iid | 1.300693e-15 | 1.998401e-15 | 0.00136189 | 12867 |
| feature_linear | iid | 0.2487212 | 0.2985853 | 1.945e-05 | 26496 |
| unit_compliance | iid | 0.08648852 | 0.09634359 | 2.1156e-05 | 17392 |
| C006_frozen | iid | 0.01181709 | 0.02081186 | 0.005164065 | 25304 |
| banded_classical | iid | 1.75006e-15 | 2.997602e-15 | 0.001384184 | 12640 |
| D_fitted_transfer | low_k | 1.861007e-15 | 6.394885e-14 | 0.001189521 | 12894 |
| feature_linear | low_k | 0.9880935 | 21.8885 | 2.1967e-05 | 26496 |
| unit_compliance | low_k | 0.08548602 | 2.051662 | 2.5193e-05 | 17392 |
| C006_frozen | low_k | 0.8857249 | 20.0639 | 0.006606048 | 25629 |
| banded_classical | low_k | 1.542319e-15 | 4.618528e-14 | 0.000726714 | 12640 |
| D_fitted_transfer | new_source | 1.307779e-15 | 9.436896e-16 | 0.001470183 | 12894 |
| feature_linear | new_source | 457827 | 539111.5 | 2.4416e-05 | 26496 |
| unit_compliance | new_source | 0.1362441 | 0.2453382 | 2.6929e-05 | 17392 |
| C006_frozen | new_source | 0.01202932 | 0.01868866 | 0.005700555 | 25304 |
| banded_classical | new_source | 1.243769e-15 | 1.44329e-15 | 0.000999442 | 12640 |
| D_fitted_transfer | new_material | 1.545957e-15 | 1.609823e-15 | 0.001160179 | 12894 |
| feature_linear | new_material | 0.8284274 | 0.650188 | 3.3753e-05 | 26496 |
| unit_compliance | new_material | 0.5107913 | 0.4717892 | 1.8688e-05 | 17392 |
| C006_frozen | new_material | 0.06516926 | 0.1277939 | 0.006907942 | 25629 |
| banded_classical | new_material | 1.600302e-15 | 1.720846e-15 | 0.000565694 | 12640 |
| D_fitted_transfer | new_boundary | 1.144695e-15 | 3.108624e-15 | 0.00120341 | 12894 |
| feature_linear | new_boundary | 0.1070898 | 0.3310346 | 1.9224e-05 | 26496 |
| unit_compliance | new_boundary | 0.09679331 | 0.2977946 | 1.9132e-05 | 17392 |
| C006_frozen | new_boundary | 0.01244579 | 0.08237206 | 0.005059601 | 25629 |
| banded_classical | new_boundary | 1.213821e-15 | 3.330669e-15 | 0.000659452 | 12640 |
| MLP | iid_seed903 | 0.1595115 | 0.1908601 | 7.0072e-05 | 26632 |
| MLP | low_k_seed903 | 0.9898313 | 21.67611 | 0.000496651 | 26632 |
| MLP | new_source_seed903 | 0.1817687 | 0.1891252 | 8.1992e-05 | 26632 |
| MLP | new_material_seed903 | 0.6981527 | 0.4616013 | 7.3978e-05 | 26632 |
| MLP | new_boundary_seed903 | 0.3370557 | 0.8590765 | 6.0948e-05 | 26632 |
| MLP | iid_seed904 | 0.1747557 | 0.2661187 | 5.5205e-05 | 26632 |
| MLP | low_k_seed904 | 0.9839431 | 21.5743 | 5.0649e-05 | 26632 |
| MLP | new_source_seed904 | 0.1877541 | 0.1900354 | 0.000136822 | 26632 |
| MLP | new_material_seed904 | 0.690113 | 0.4785607 | 6.4819e-05 | 26632 |
| MLP | new_boundary_seed904 | 0.3289318 | 0.9024092 | 5.8987e-05 | 26632 |
| MLP | iid_seed905 | 0.1832809 | 0.201785 | 5.4355e-05 | 26632 |
| MLP | low_k_seed905 | 0.9872272 | 21.66718 | 8.0317e-05 | 26632 |
| MLP | new_source_seed905 | 0.1653598 | 0.1836223 | 0.000110621 | 26632 |
| MLP | new_material_seed905 | 0.6548994 | 0.4177253 | 8.6533e-05 | 26632 |
| MLP | new_boundary_seed905 | 0.3439864 | 0.8657727 | 6.1774e-05 | 26632 |
| feature_ridge | iid | 0.209212 | 0.2968785 | 2.0174e-05 | 26496 |
| feature_ridge | low_k | 0.9904252 | 21.95302 | 2.1058e-05 | 26496 |
| feature_ridge | new_source | 0.2382931 | 0.2136103 | 2.8423e-05 | 26496 |
| feature_ridge | new_material | 0.6701318 | 0.5679027 | 2.0556e-05 | 26496 |
| feature_ridge | new_boundary | 0.1103378 | 0.3663725 | 1.9782e-05 | 26496 |

물리 flux/balance/BC, batch sample mean/std와 input/prediction/reference는 `results/session_0003/metrics.json` 및 `arrays/`에 보존했다. 시간은7회 CPU 반복 중앙값이다. A/C base inverse 준비는 별도 profile에 있으며 native/BLAS 일부 메모리는 tracemalloc에 잡히지 않는다. RSS는 누적 process high-water다.

## 판정
**REJECT as an independently established invention.**
Known physical algebra가 우수해도 발명이 아니다. 의미 없는 추가 Cycle을 만들지 않고6Cycle에서 중단한다.

## 미확인 사항
원문 사이트403, 새 공식 README3개와 이전 저장 코드만 확인했다. 정확한 수학적 연산 동일성은 별도로 증명했지만 논문/특허 전체 검색은 미완료다. 실행하지 않은 GPU/Stokes/대규모 데이터 결과를 주장하지 않는다.
