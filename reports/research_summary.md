# Physics Architecture Research — 세션 0002

**가설7개와 수정안을12개 Cycle에서 검증했지만, 새로운 범용 아키텍처를 발명했다는 근거는 확보하지 못했다.** 제한된 성공과 실패를 모두 보존했고, 강한 기준선으로 초기의 유리해 보이는 비교도 반박했다. 이번 결과는 발명 완료가 아니라 다음 탐색을 위한 중간 연구 성과다.

최종 실행은 `results/session_0002_verified`다. 결과198행에 입력·예측·기준값 배열과 시간·메모리가 대응한다. 일부 행은32개 샘플을 묶으므로198을 독립 실험 수로 해석하면 안 된다. pytest **21개 통과, 실패0, 건너뜀0**. 배열에서 오차를 재계산하고 source 해시와 PNG도 확인했다. 기존 조화진동자 기록과 이번 세션의 이전 실행/보고서를 보존했다.

## A. 물리적 결론

새로운 물리 법칙을 발견했다고 주장하지 않는다. 아키텍처 설계에서 혼동하기 쉬운 조건을 실제 반례로 분리했다: 내부 보존과 정상해 수렴, SPD와 점별 양수성, potential gauge와 source 변화, 물성 편집 이력과 정상해, 비압축성과 운동량 만족은 각각 다르다. 물리식을 사용해 정확해진 모델도 고전 Green/Galerkin/solver와 같으면 발명이 아니다.

## B. 구체적으로 학습한 primitive

C006은 공간 token 대신 **이완 시간 packet**을 가정한다. 양의 자기수반 `A=V diag(lambda)Vᵀ`와 경계 lift/mass 변환 후 source `b`에 대해

\[
g_\theta(\lambda)=\sum_{j=1}^{6}e^{\alpha_j}\exp[-e^{\beta_j}\lambda],\qquad
T=V\operatorname{diag}(g_\theta(\lambda))V^Tb.
\]

12개 scalar를 학습한다. 근거는 고전 `A^-1=integral_0^infinity exp(-sA) ds`다. 이 항등식이나 finite exponential sum을 발명한 것으로 간주하지 않는다. 모든 후보의 완전한 타입/단위/수식/조건/복잡도는 `architectures/candidate_001.md`부터`007.md`에 있다.

## C. 실제 architecture 흐름

물리 문제 → 고전 연산자·경계 restriction·mass 좌표 → eigensystem → source 변환 → 양의 age packet 응답 합 → 역변환 → 지정 경계 lift다. 2D는 `z=M^(1/2)T` 좌표를 사용한다. 유동은 고전 Helmholtz projection을 먼저 적용한다.

**Full eigensystem과 projection이라는 강한 물리 prior**를 받으므로 새로운 geometry 처리 자체를 학습했다고 볼 수 없다. Solver를 모델 안에 재구현하는 것을 최종 목표로 인정하지 않으며, 이 prototype의 해당 비용을 숨기지 않았다.

## D. 보장과 미보장

양의 고유값에 대한 양의 응답과 reciprocal positive-work 구조는 수식으로 보장한다. 점별 양수성은 positivity-preserving generator 조건이 있어야 한다. SPD만으로는 불충분하며 비단조 이방성 FEM semigroup의 음수 응답을 실제 확인했다.

유한 age 합은 `lambda→0`에서 유한하고 큰lambda에서 지수 감소하므로 전 구간의 `1/lambda` 근사는 불가능하다. 수정 `1/(lambda+epsilon)`은 shifted solve와 같다. 비선형 유동, arbitrary geometry 학습, 시간 semigroup 정합성, 대규모 빠른 추론은 보장하지 않는다.

## E. 구현과 재현

참조 물리는 `src/physics/heat.py`, 후보는 `src/operators/primitives.py`, 실행은 `experiments/run_research.py`, 검증은 `tests/test_research_physics.py`다. 저장 결과 검사는 `scripts/verify_results.py`이며, 각 Cycle의 요청된10항목 보고서는 `reports/cycles/cycle_01.md`부터`12.md`다.

무결성 검토에서 다중 우변의 축 오류를 찾아 수정했다. 이전 실제 물리 실험의 단일 source 경로는 영향을 받지 않았지만, 개별/일괄 결과 일치 assertion을 추가하고 전체 연구를 수정 코드로 재실행했다. 구현 버그와 후보의 구조적 실패를 구분했다.

## F. 실제 결과

학습 표는 샘플별 상대 L2 평균의3시드 평균±표준편차다. 나머지는 seed824 또는 지정된 control이며 이산해 대비 오차와 continuum 오차를 구분한다.

| 조건 | 관측 상대 오차 | 의미 |
|---|---:|---|
| 1D IID, k=0.5–2 | `0.0026435 ± 0.0001439` | 약0.264%의 제한된 성공 |
| 낮은 k OOD, 0.01–0.05 | `0.80424 ± 0.00306` | 약80.4%, 범용성 실패 |
| 높은 k OOD, 10–30 | `0.04259 ± 0.00822` | 약4.26% |
| mode12 source OOD | `0.09862 ± 0.01993` | 고주파 한계 |
| 2D 16×16분할 사각형 | 온도 `0.0009550`, flux `0.0023169` | 이산 FEM 기준 |
| 2D 정사각 구멍 | 온도 `0.0098739`, flux `0.0104005` | energy balance 절대 위반 `0.0056682` |
| Poiseuille, mu=1,N15 | `0.0025721` | 열 가중치 재사용, 선형 조건 |
| Poiseuille, mu=0.1,N15 | `0.47005` | 낮은 점성 실패 |
| 비선형 제조 유동 | `0.27242` | divergence~`6.75e-14`여도 momentum 상대 잔차~0.682 |

추가로1D Neumann의 source 호환·상수 gauge를 별도 control에서 검증했다. 2D 회전 이방성 제조해 온도 오차는8/16/24분할에서 `0.006709→0.001676→0.0007451`로 감소했고 flux도 수렴했다. 이류–확산은 속도±1, cell Péclet0.1/1/10, periodic/inflow BC, 시간 간격 감소와t=10을 검증했다. 시간 적분의 성공은 **고전 backward Euler** control이며 신경망 성능으로 집계하지 않는다. Couette의 정확성도 지정 선형 boundary lift의 결과다.

RANS, arbitrary CAD, 학습 Neumann, GPU 확장 및 다른 물리 분야는 **미실행**이다.

## G. 기준선·시간·메모리·데이터

초기 MLP의 IID 오차0.45762는 입력·출력 정규화, 별도 validation 선택 및3,000step으로 `0.05628±0.00729`까지 개선됐다. 서로 다른 budget/prior/parameter 수로 공정한 전체 순위를 주장할 수 없다.

더 강한 기준선은 알려진 `1/k` 스케일링과225개 데이터 적합 compliance 계수다. 같은 고정 격자에서는32/128개 label로 floating-point 수준의 오차를 낸다. Source 설계 행렬이 full rank인 간단한 선형 family다. 이 결과는 새로운 신경 연산의 필요성·우월성을 반박한다. C006 자체의32-label 비교는 미실행이다.

C006은12parameter지만 고유분해 `O(N³)`, 저장 `O(N²)`, cached 응답 `O(N²)`가 필요하다. 이번2D 사각형의 response 중앙값은 약5.81ms 대 직접 dense solve0.98ms, 구멍 형상은4.70ms 대0.39ms였다. 양쪽 모두 조립 제외, C006은 eigensetup 포함, CPU5회 반복 값이다. Cached 비용과 전체 setup을 혼합하거나 GPU/production CAE 속도로 일반화하지 않는다.

모든 최종 행에 시간과 추적 peak를 기록했다. tracemalloc은 일부 native/BLAS 메모리를 빠뜨리고 RSS는 누적 process high-water라 모델별 격리 사용량이 아니다. 학습 시간·샘플 수·분산은 JSON, 강한 control 포함 그림은 `results/session_0002_verified/plots/strong_control_comparison.png`에 있다.

## H. 폐기한 후보

C001 정규화 응답, C002 교환, C003 action span, C004 positive-only state, C005 material events, C007 source contrast를 반례 또는 기존 연산과의 명확한 동일성으로 제외했다. C006은 numerical probe로만 남긴다. 정확해진 C001R/C006R도 각각 Green/shifted solver라 발명 claim에서 제외했다. 원인과 판정은 `research/failures.md`와`candidate_registry.md`에 있다.

## I. 논문 기여 가능성

새 범용 아키텍처 논문을 뒷받침할 근거는 아직 없다. 반례·검증 suite·strong baseline 검토는 연구 자산이지만 독창적인 연산/이론/기술 효과가 추가로 필요하다. 공식 구현/라이브러리 설명10곳과 일부 코드를 실제 확인했으나 원문 사이트는 proxy403으로 차단됐다. 신규성 검토는 미완료이며 읽지 않은 논문을 읽었다고 쓰지 않았다. 검색 분야별 상태는 `research/novelty_review.md`에 있다.

## J. 특허 발명 후보

현재 근거 있는 발명 후보는 없다. 선행특허 검색도 수행하지 않았으며 신규성·진보성을 판정하지 않았다. 계산 절차·효과·배제 이유와 조사 조건은 `reports/invention_candidates.md`에 있다.

## K. 다음 방향과 운영

12Cycle 예산에 도달해 이번 탐색을 종료했다. 다음은 oriented perturbation-response와 constraint witness라는 다른 객체를 검토하되 아직 수식/실험이 없는 미검증 질문으로 다룬다. 원문/nearest-prior-art 확인 → 이질 물성/BC held-out → positivity 조건 → 비선형 momentum 반증 순서다. 정확한 재개 명령은 `research/next_steps.md`에 있다.

외부 연구 데이터 전송, 삭제, 유료 자원, GPU 학습, GitHub commit/push는 하지 않았다. 환경 초안에는 `start_skill`과 원문 접근용4개 domain을 저장했고 설치 script는 보존했다. 필요하면 환경 설정에서 검토·저장 후 Publish해야 한다. 초안 저장, 현재 수치 검증, publication과 새 task 복원은 서로 다른 상태다.
