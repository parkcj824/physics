# 다음 재개 — Session0006, Cycle28부터

## 현재 상태

검증된 Level3 primitive/범용 architecture 없음. 가장 구체적인 다음 연구 객체는 **집단 상태 S와 유한 nonlinear defect H의 쌍**이다. 이는 아직 새 학습 규칙이 아니다. L/F5는 제한된 Level2 표현 probe로 유지하며, J gradient와 K AR2는 기존 알고리즘 재구현으로 제외한다. G의 구성식 feature가 일반 feature/conditioning보다 고유하게 유리한지는 미해결이다. 실패와 표준편차를 그대로 읽는다.

Session0002–0005 자료를 수정하지 않는다. 이 세션 확인 자료를 모델 선택에 사용한 뒤 새 untouched test라고 주장하지 않는다. 기존 `research/SESSION_INDEX.md`는0003 당시 상태이므로 `research/session_0005/README.md`를 최신 진입점으로 사용한다.

## 재학습 없는 정확한 검증 명령

저장소 `/workspace/physics`에서 다음을 실행한다.

```bash
XDG_CACHE_HOME=/workspace/.cache MPLCONFIGDIR=/workspace/.cache/matplotlib OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /workspace/.physics-venv/bin/python -m pytest -q
XDG_CACHE_HOME=/workspace/.cache MPLCONFIGDIR=/workspace/.cache/matplotlib OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /workspace/.physics-venv/bin/python scripts/verify_session0005.py
```

기존 Session4 verifier는 canonical 경로에서 다시 실행하지 않는다(과거 validation 파일을 쓴다). 이미 수행한 복사본 검증의 증거는 `results/session_0005/prior_validation.json`이다. 현재 verifier는 Session5 validation만 갱신하고 과거1515개 파일을 검사한다.

## 선택적인 전체 재현

새로운 빈 로컬 출력 경로만 사용한다. 같은 seed의 재현이며 새로운 과학적 독립 검증이 아니다. 약70 checkpoints의 작은 CPU 학습을 다시 실행하므로 단순 integrity 확인에는 위 명령으로 충분하다.

```bash
XDG_CACHE_HOME=/workspace/.cache MPLCONFIGDIR=/workspace/.cache/matplotlib OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /workspace/.physics-venv/bin/python scripts/replay_session0005.py /workspace/physics/results/session_0005_replay_01
```

Wrapper는 ROOT를 모든 새 Session5 모듈에 명시적으로 지정하고 canonical 자료를 덮어쓰지 않는다. 모델 생성 시각/계측 시간 등은 새로 기록되므로 output byte 전체가 같을 것을 요구하지 않는다. 원래 scientific source snapshot과 checkpoint는 canonical Session5 경로에 보존된다.

## 다음 연구의 정확한 순서

1. AGENTS, Session5 A–J 보고서, 후보 수식 J/L/K, 실패·novelty ledger, C0/C0b 프로토콜, Cycle27 반례를 읽는다. 54 tests와 integrity를 확인한다. 새 Session0006 전용 경로와 immutable prior manifest를 만든다.
2. **정보 충분성부터:** 정확한 보정식 `A_h delta−λexp(T0+S)expm1(delta)−H=0`의 저장 field 검증을 먼저 읽는다(`closure_equation_audit.json`). 이는 원래 PDE의 좌표 변경이며 새 solver가 아니다. 일반 nonlinear reaction `R(T)`에서 H-only 입력 충돌을 재유도하고 `(S,H)`가 무엇을 결정할 수 있는지, 다른 material/BC/source 표현에서 추가로 필요한 정보가 무엇인지 분석한다. Phase/branch 정보가 필요하면 별도 독립 조건으로 명시한다. S를 주면 정확한 inverse/Newton을 호출하는 수리는 기준선일 뿐이다.
3. **새 계산 흐름의 조건:** 상태와 별개로 필요한 물리 정보를 보존하고, 학습 과정에서 그 정보가 어떤 다른 역할을 하는지 구체적인 행렬식·학습 규칙으로 정의한다. Conditional MLP/gated residual/learned correction/DeepSets와 같은 graph로 전개되는지 먼저 검사한다. 같은 것이라도 Level2 표현의 구체적 장점은 실험할 수 있지만 Level3이라고 선언하지 않는다. 완전히 같은 알고리즘이면 이름을 바꿔 살리지 않는다.
4. **사전 등록:** 새로운 seed, train/validation/untouched confirmation 분포, loss, 계산 예산, 평가 지표를 모델 학습 전 고정한다. Source 위치와 scalar/비균질 low-k를 구분한다. 기존 반례는 개발용 regression case이며 새 최종 독립 집합은 따로 만든다.
5. **가장 작은 반증:** conservation0/constitutive≠0, small field/large PDE, positive-source/BC0, low-k, direction reversal 및 동일 H/다른 S를 사용한다. Positivity/BC/enumeration 없는 고차 반응의 보장 범위를 정확히 쓴다. 새 연산이 없으면 큰 모델을 만들지 않는다.
6. **공정 비교:** shared encoder 오류와 interaction head 오류를 따로 보고, 총 θ/활성 θ/실제 profile 연산/추론 전처리/학습 데이터·시간을 포함한다. Classical banded/Newton, conditioned direct MLP, bilinear/cubic/anchored attention을 필요한 범위에서 비교한다. 물리 feature 크기와 정보 효과를 혼동하지 않는다.
7. **독립 challenger:** 관측 가능한 상태와 완전한 state를 구분하는 물리 정보의 선택/망각 원리를 탐색하되 AR2, standard memory kernel, branch mixture를 그대로 재제안하지 않는다. Cattaneo와 다른 계에서 필요한 상태 정보를 먼저 명시한다. 완전한 Markov 상태에 불필요한 memory를 넣지 않는다.
8. 새 1D 규칙이 잔차/OOD/positivity를 실제로 통과한 경우에만 2D/새 geometry/resolution, 이후 Stokes로 확장한다. RANS/HVAC 및 대규모 GPU는 별도 단계다. 통과하지 않으면 실패 기록과 새 원리를 남긴다.
9. 문헌은 접근 가능한 공식 저장 자료부터 확인한다. 원문/특허를 확인하지 못했으면 신규성 미검증. 403 실패를 반복하지 않는다. Source의 실제 알고리즘·학습 규칙과 비교한다.

## 피해야 할 반복

- C006 재튜닝, Sherman–Morrison/transfer/prefix 재명명.
- J에 diagonal mobility만 붙여 새 primitive라고 주장하기.
- L의 rank/order/폭을 올리거나 S를 입력한 Newton을 새 발명으로 세기.
- Activation0을 실제 sparse execution/FLOPs 절감으로 주장하기.
- λ0 low-k scaling을 nonlinear OOD 성공으로 해석하기.
- K의 두 snapshot AR2를 독립 memory architecture라고 다시 제안하기.
- 보존 생성 flux만 보고 온도 PDE를 통과했다고 판단하기.

GitHub push/외부 연구 데이터 전송, 유료 자원, 큰 GPU 학습, 주요 구조 변경, 대규모 삭제는 별도 승인 전 수행하지 않는다. 과거 PR/Merge는 다음 세션의 push 승인으로 간주하지 않는다.
