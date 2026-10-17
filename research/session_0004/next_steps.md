# Session 0004 정확한 재개 지점 — 다음은 Session 0005 / Cycle24

## 저장 상태

- 신규 범용 Architecture 또는 검증된 Level3 발명: **없음**.
- 독립 가설 E/F/G3개, E-R 수정, G 방향성확장까지 Cycle19–23의5개 유의미한 Cycle. 12개를숫자맞추기로채우지않았다.
- 후속통제포함 모델51개, 확인예측183batch행, dataset24개. 기존추적파일1154개보존, pytest42통과. 원래45모델·159행과새6모델·24행은구분한다. 검증결과는 `results/session_0004/validation/integrity.json`.
- 제한된Level2표현probe F, Level3목표의경쟁mechanism G를유지하되검증실패를보존한다. E는fold식별정보를보강해야한다.
- 전체수식: `architectures/session_0004/candidate_E.md`, F/G. 한국어A–I보고: `reports/session_0004/research_summary.md`.
- 실제학습/확인은끝났으며확인결과후재학습하지않았다. 기존두세션파일을수정하지않았다. Push/외부연구데이터전송/유료/GPU/삭제없음.

## 1. 먼저 실행할 읽기·검증

```bash
cd /workspace/physics
source /workspace/.physics-venv/bin/activate
export XDG_CACHE_HOME=/workspace/.cache
export MPLCONFIGDIR=/workspace/.cache/matplotlib
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
python -m pytest -q
python -m scripts.verify_session0004
```

이verifier는이전결과를읽기만하고Session0004 validation만다시쓴다. Strict bitwise감사를원하면먼저canonical출력hash를보관하거나새로정의한snapshot검증방법을사용한다. Session0003의verifier도과거경로에출력을쓰므로Session0003전체를임시복사해그복사본을검증한다. 과거source/hash를rewrite해검증을통과시키지않는다.

## 2. 정확한 재현 명령

수치차이를검사할때만실행한다. 재현은새과학Cycle이나독립증거로세지않는다. 새output은아직존재하지않는경로여야한다.

```bash
python -m scripts.replay_session0004 --output /workspace/physics/results/session_0004_replay all
python -m pytest -q --junitxml=results/session_0004_replay/validation/pytest.xml
python -m scripts.replay_session0004 --output /workspace/physics/results/session_0004_replay verify
```

기존canonical결과를가리키면replay가실패하도록했다. all은label생성·counterexample·main36모델·transport9모델·capacity6모델·확인·고전비교를다시실행한다. 단계별로 `initialize`, `prepare`, `train`, `train_oracle`, `train_matched`, `confirm`, `transport`, `counterexamples`, `controls`, `capacity`, `audit`, `verify`도지원한다. 이세션에서는전체51모델재현을다시실행하지않았고 저장checkpoint183batch재현으로검증했다. Replay의initialize/counterexample만별도경로에서smoke검사했다. 원래run의환경metadata를복사하므로replay새환경은별도로검사한다.

## 3. 다음 연구의 우선순위

1. **정보흐름의효과를분리한다.** 이번추가통제에서active8415/width64/3000step MLP를새IID집합으로검증했다. 온도표현이득일부는남았지만G의PDE잔차는여전히나쁘다. `reports/session_0004/capacity_control.md`부터읽는다. 다음에는파라미터늘리기가아니라실제constitutive정보와일반추가feature가주는효과를분리할수있는통제를새확인집합에사전정의한다. 이것은새발명Cycle이아닌통제실험이다.
2. **그후새계산객체를정의한다.** 온도L2가작아도구성식mismatch가큰것은어떤공간scale의오차가상태에남는다는뜻이다. 구성식과balance의오차를함께운반하는객체와학습가능한상호작용규칙을기존residualMLP/GNN/learnedsolver와동일하지않게구체화할수있는지검토한다. 단순residualpenalty추가만으로새primitive라고하지않는다. 아직완전한새식/실행코드는없다.
3. **독립적인경쟁가설도유지한다.** F의finite response는bilinear/setpooling과가깝다. 이미알려진고차Volterra항을늘리는것말고,어떤상호작용을활성화할지물리적응답정보가정하는단순규칙이있는지검토한다. 고차항이나rank증가만으로새후보라고하지않는다.
4. **다른출발원리:** initial-condition/시간이력으로branch/basin을선택하는causal열응답을문제로정의한다. 현재E는해집합만표현하므로historyselector가아니다. RNN/implicit solver/mixture재표현이면그대로인정한다. 새primitive가정의되기전대규모데이터/학습부터시작하지않는다.
5. 작은열/방향성gate를통과한mechanism만비압축성Stokes/nonlinear momentum으로확장한다. 현재1Dscalartransport를incompressibleflow로이름바꾸지않는다. RANS/HVAC는미실행이며별도물리/데이터검증이필요하다.

## 4. 반복 금지·미해결

- C006재튜닝, Woodbury이름변경, source를잃는witness, coordinate-conjugate addition,고전transfer의scalar학습으로복귀하지않는다.
- foldoracle를학습된fold라보고하거나, F의반복T00를독립data로세거나, G의생성flux보존을정확한온도/운동량해라고하지않는다.
- 기존수학을썼다는이유로후보를자동기각하지않는다. 실제algorithm/정보흐름/학습/효과를비교한다.
- 방향성benchmark의양수source/영BC에서G도음수값을예측했다. 이는signed온도편차가허용되는일반문제와다른positivity요구다. 보존/방향성성공만보고이실패를삭제하지않는다.
- 원문403상태가바뀌었다는증거없이같은요청을반복하지않는다. 저장한공식source만읽은것과원문미열람을구분한다. Multi-solutionoperator/normal-formlearning, bilinear/Volterra/setlearning, hard-constrained/co-generation/learnedsolvers의최근접원문과특허검색은미완료다.
- 다음실행은새Session0005경로·새seed·새사전계획으로시작한다. 일반로컬코딩/CPU실험은자율, Push/외부업로드/유료/GPU/삭제는승인필요다.
