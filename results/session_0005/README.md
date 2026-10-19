# Session0005 실제 실행 산출물

- `environment.json`: 시작 시각, HEAD, 실제 Python/package/CPU/GPU/메모리 한도. GPU 미사용.
- `prior_file_sha256.json`: 시작 당시 기존 tracked1515개 파일. 기존 Session2–4 결과와 code hash를 보호한다.
- `prior_validation.json`: Session4 결과 복사본에서 실행한 기존 verifier 결과. 원본 validation에 쓰지 않았다.
- `data/`: 새25 datasets. C0 12files, F5 11files, C0b 2files. 총1024 input sets/rows이며 방향 쌍·반복 base/subset 등의 상관관계가 있어1024개의 독립 통계 표본이라는 뜻은 아니다.
- `*_dataset_manifest.json`: 실제 NPZ SHA와 생성 실패/준비 비용. F low-k 추가 준비 시간은 최초 실행에서 별도 기록하지 않았다.
- `models/`: C0 36, C0b 18, F encoder1+head15 =70 checkpoints. Learned Jacobi의3scalar checkpoint6개 포함. 서로 다른 architecture70개라는 뜻은 아니다.
- `*_frozen.json`: 학습 trace/validation 선택/θ/활성 θ/profile 연산/실제 시간/동결시각/SHA. F frozen head의 profile 비용에는 사전 계산한 공유 encoder가 빠지며 encoder 비용은 별도로 저장했다.
- `*_confirmation.json`: C0 144+C0b54+F135 =333 model-condition prediction batches. 고전 배열24 batches와 F exact 비용9조건은 별개다. 반복 prediction을 독립 데이터로 세지 않는다.
- `arrays/`: 각 result row의 `arrays` 경로가 입력·기준값·예측값·flux/mixed/higher 효과와 대응한다. 상대오차뿐 아니라 절대오차, effect 크기, 수지/구성식/PDE/positivity를 함께 기록했다.
- `compatibility.json`, `causal.json`, `closure_counterexample.json`: Cycle24/26/27의 원시 반증. NN를 학습하지 않은 경우를 명시했다.
- `closure_equation_audit.json`: 저장된 네 물리 조건에서 정확한 correction 식 `A_h delta−λexp(T0+S)expm1(delta)−H=0`을 사후 검증했다. 신규 solver나 학습된 연산이 아니다.
- `posthoc_audit.json`: 동결 이후 feature trace42, field 오류 분해135, pooled-set graph 재구성27 batches. 새 독립 확인이나 추가 fit이 아니다.
- `supplementary_costs.json`: 같은 저장 조건의 별도 재현 계측. 원래 J/K/closure 전체 runtime을 측정한 것처럼 소급하지 않는다. Python peak는 native/BLAS/tensor peak를 완전히 포함하지 않는다.
- `source_snapshot/`: phase 당시 소스·사전 계획과 SHA. Outcome-blind amendment가 있어 준비 단계와 훈련 단계의 소스가 일부 다를 수 있다. 각 복사본의 SHA를 검사하고 최신 확인 소스는 현 파일과도 대조한다.
- `plots/`: 실제 저장 결과에서 만든 PNG/PDF. [한국어 A–J 보고서](../../reports/session_0005/research_summary.md)의 그림이다.
- `validation/pytest.xml`: 전체54 passed의 실제 runner 결과.
- `validation/integrity.json`: 재학습 없는 checkpoint replay, 별도 dense/root 기준해2472records, metric/물리 반례/기존파일 검증. 중복 reference도 포함한 검사 수다.
- `validation/replay_routing_smoke.json`: 임시 출력 경로에서 J/K/closure 배열55개 재현. 전체 재학습 wrapper 자체는 다시 돌리지 않았다.
- `validation/source_hashes.json`, `validation/assets_sha256.json`: 종료 시 새 소스·문서 및 Session5 자산 SHA. 반복 실행 때 바뀌는 `integrity.json`과 manifest 자체를 제외한다.
- `run.json`: 실행 범위·실제 종료 시간·Cycle 수·중단 이유와 미실행 영역.

과거 pure functions를 재사용했다: `src/operators/session0004.py`, `src/physics/session0004.py`, `experiments/session0004.py`, `experiments/session0004_transport.py`. 이 파일들의 해시는 prior manifest에서 보호하며 과거 데이터를 학습에 재사용하지 않았다. 가중치·center/scale은 새 checkpoint에 저장했다.

검증과 새로운 경로에서의 선택적 재학습 명령은 [next_steps](../../research/session_0005/next_steps.md)에 있다. 새 확인 데이터 없이 기존 평가를 다시 해 독립적인 발견으로 보고하지 않는다. GitHub 전송은 수행하지 않았다.
