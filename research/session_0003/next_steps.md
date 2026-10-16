# Session 0003 종료/정확한 재개 지침

## 현재 상태

- 최신 entry point: `research/SESSION_INDEX.md`.
- 이번 세션: Cycle13–18의 **6개 Cycle**, 가설 A/B/C/D4개, 결과136행. 과거 modal audit와 profiling replay를 독립 발견으로 세지 않았다.
- Test34개 통과(기존21+신규13), 실패/skip0.
- **최고 발명 후보 없음.** 정확한 control은 banded/transfer이고 둘 다 기존 알고리즘이다.
- C006은 frozen old checkpoint를 읽었고 재학습/증대/튜닝하지 않았다. 낮은k 오차의 mode 식 재현과 진단은 끝났으므로 다음 세션의 주력으로 되돌리지 않는다.
- Session0002 결과232개 파일과 기존 scientific source를 보존했다. 새 코드/문서는 별도 경로다. 이번 GitHub push/commit/환경 Publish/유료 자원/GPU 학습은 하지 않았다.

## 1. 먼저 읽고 검증

다음 순서: `AGENTS.md` → `reports/session_0003/research_summary.md` → `candidate_registry.md` → `failures.md` → `novelty_review.md` → `architectures/session_0003/candidate_*.md` → 필요한 `reports/session_0003/cycles/`.

```bash
cd /workspace/physics
source /workspace/.physics-venv/bin/activate
export XDG_CACHE_HOME=/workspace/.cache
export MPLCONFIGDIR=/workspace/.cache/matplotlib
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
python -m pytest -q
python -m scripts.verify_session0003 results/session_0003
```

Verifier는 이전 결과를 읽기만 하며 출력은 Session0003 validation에만 쓴다. 기존 `scripts/verify_results.py`는 과거 결과 안에 integrity를 다시 쓸 수 있으므로 strict preservation 감사에서는 실행하지 않는다. 이후 source를 의도적으로 바꾸면 phase snapshot과 현재 파일의 hash 차이를 검토하고 과거 snapshot을 수정하지 않는다.

## 2. 반복 금지 사항

- A finite packet을 Woodbury/Green layer의 새 이름으로 제안하지 않는다. Frozen packet 반응 갱신을 추가하는 것은 이미 inverse-update identity다.
- B의 coarse witness만으로 모든 field를 결정한다고 다시 주장하지 않는다. Full input이나 보조 인증은 다른 문제이며 충분성부터 검토한다.
- C처럼 `phi^-1(phi(a)+phi(b))`를 새로운 coupling으로 주장하지 않는다. 물리 scalar 변화량이 임의 군좌표를 따르지 않는다.
- D에 latent channel을 추가하거나 theta를 더 학습하는 것을 신규 primitive로 세지 않는다. Full port relation→matrix product→prefix 복원까지 기존 알고리즘과 같다.
- 같은 최종 function이라고 모든 새로운 계산법을 배제하지도 않는다. **실제 알고리즘·학습·정보 흐름의 차이**가 검토 대상이다.
- Uniform scalar1/k를 arbitrary heterogeneous material이나 전체 비영BC 온도에 적용하지 않는다. 고정 material 모양의 global scale과 harmonic lift 성분을 구별한다.

## 3. 다음 독립 연구 질문: 물리적 다중 해 relation

현재 네 가설의 정확한 연산 동일성/정보 반례가 끝나 추가로 의미 있는 수정 연산이 정의되지 않았다. 이번에는 최대12개를 억지로 채우지 않았다. 새 세션은 Cycle19부터 **새 최대12Cycle/약90분** 한도로 시작한다.

다음 계산 객체 질문은 **단일 값을 평균하는 것 대신 허용 가능한 해 branch와 선택 조건을 함께 나타내는 relation**이다. 물리 입력에 branch 선택 조건이 없는데 하나의 정답을 정해 주는 문제부터 피한다. 이는 아직 **학습 primitive 수식이 완성되지 않은 미실행 질문**이며 새로운 architecture로 등록하지 않는다.

열 분야의 작은 출발 benchmark 후보는 Bratu형 열반응

\[
-T''=\lambda e^T,\quad T(0)=T(1)=0.
\]

기초 지식/직접 chain-rule로 확인할 수 있는 해 계열은

\[
T_a(x)=-2\log\frac{\cosh[a(x-1/2)]}{\cosh(a/2)},\qquad
\lambda(a)=2a^2/\cosh^2(a/2).
\]

이 계열의 fold는 `a*tanh(a/2)=2`로 찾을 수 있다. **이번 세션에서는 이 benchmark를 계산하지 않았고** 수치 threshold/branch 안정성/관련 논문을 새로 확인하지 않았다. 여러 해가 모두 안정하다고 가정하지 않는다. Parabolic 선형화의 eigenvalue와 branch 선택에 필요한 초기조건/이력을 별도 검증해야 한다.

다음 단계의 정확한 순서:

1. 위 식을 symbolic/수치 미분으로 확인하고5–20DOF reference와 비교하되, 이 고전 해 자체를 발명으로 세지 않는다.
2. 기본 객체가 branch-set인지, branch를 선택하는 물리 이력인지, relation 자체인지 구분한다. 전체 source/BC/물성 정보를 잃지 않았는지 확인한다.
3. 입력 타입·단위·고정/학습 변수·실제로 전개된 primitive를 먼저 쓴다. **아직 실행할 prototype 명령은 없다.** 수식 없는 임의 network/data 수집부터 하지 않는다.
4. Polynomial root elimination, differentiable implicit solve, mixture/ensemble, projection의 재표현이면 즉시 제외한다. DEQ/OptNet 및 multi-solution operator literature와 대조한다.
5. 실제 새로운 계산 규칙이 남으면 선형 열전도 limit에서 source superposition/BC/저물성/해상도와 orientation gate를 다시 검사한다.
6. 그 다음 물리적으로 nonlinear branch 문제를 작은 CPU로 시험한다. 유망할 때만 Stokes/비선형 유동으로 확장한다. 새 dataset은 새 Session ID에 저장한다.

독립 challenger로는 **부분 관측의 causal memory/observable closure**를 검토할 수 있다. 동일한 순간 관측이 다른 미래를 만드는지부터 증명해야 한다. 이를 RNN·memory kernel·Mori–Zwanzig의 새 이름으로 구현하면 신규성으로 인정하지 않는다. 이 역시 미실행 질문이며 B의 source witness 충돌을 이름만 바꾼 새 발견으로 세지 않는다.

## 4. 문헌과 비교의 다음 조건

원문 arXiv/JMLR는403이었고 CVXPYlayers/OptNet/DEQ README3개만 새로 읽었다. 실제 runtime access가 바뀌기 전에는 같은403을 반복하지 않는다. 논문 설정 때문에 독립 수학 실험을 멈추지 않는다. 특허/논문 전체검색을 했다고 주장하지 않는다.

새 baseline 비교는128개 matched variable-material label과32validation을 사용했고, source/material/BC OOD5×32개를 동일 fingerprint로 맞췄다. 다만 ridge family는 test 실패 뒤 추가해 **탐색적**이다. 다음에 성능 가설을 주장하려면 새 untouched test set을 사전 정의한다. Extra unit-material128label과 C006의 과거 pretraining을 데이터 효율 비교에서 분리한다. Parameter 수만으로 공정성을 주장하지 않는다.

## 5. 재현 명령 (새 연구 Cycle 아님)

재현 문제가 있을 때만 새 경로를 사용한다:

```bash
python -m experiments.session0003 --output results/session_0003_replay --phase initial
python -m experiments.session0003 --output results/session_0003_replay --phase continue
python -m experiments.session0003 --output results/session_0003_replay --phase baseline_refinement
```

기존 output 경로를 초기화하면 실패하도록 되어 있다. Report/validation script는 이번 Session의136행/34test를 검증하는 전용 코드이므로 새 실험의 count를 억지로 맞추지 않는다. Report는 기존 문서를 덮어쓰지 않으며 `--figures-only`는 기록된 수치에서 그림만 재생성한다.

일반 로컬 코딩/계산/테스트는 자율 실행한다. GitHub push, 외부 연구 데이터 전송, 대규모 삭제, 유료 자원, 대규모 GPU는 별도 승인을 받아야 한다. 이전 PR 요청의 승인을 이번 연구의 push 승인으로 간주하지 않는다.
