# Physics Neural Network Architecture Research

새로운 물리 신경망 구조를 탐구하는 장기 연구 프로젝트입니다. 조화진동자 환경 검증에 이어 열전달·유동의 primitive 탐색과 반증을 진행하고 있습니다. 현재 **신규성이 입증된 아키텍처는 없습니다**.

## 환경

Python 3.12, NumPy 2.2.6, SciPy 1.15.3, Matplotlib 3.10.3, pytest 8.4.2, PyTorch 2.8.0+cpu. 현재 클라우드 머신에는 GPU 장치가 없어 CPU wheel을 사용합니다. 별도 서비스나 비밀키는 필요하지 않습니다.

```bash
cd /workspace/physics
bash scripts/setup.sh
source /workspace/.physics-venv/bin/activate
export XDG_CACHE_HOME=/workspace/.cache
export MPLCONFIGDIR=/workspace/.cache/matplotlib
export OMP_NUM_THREADS=1
python -m pytest -q
python -m experiments.oscillator --output artifacts/oscillator
```

`uv`가 필요합니다. 설치는 체크아웃 밖의 가상환경을 사용하며, 전체 의존성 버전과 다운로드 해시는 `requirements-cpu.lock`으로 고정합니다. CPU wheel은 PyTorch 공식 인덱스, 일반 패키지는 PyPI를 이용합니다. TLS/해시 검증을 유지합니다. 기존 가상환경의 무관한 패키지를 삭제하지 않습니다. 캐시는 쓰기 가능한 `/workspace/.cache`에 둡니다.

## 첫 실험

질량과 각진동수가 1인 조화진동자에 대해 `H=(q²+p²)/2`, `dq/dt=p`, `dp/dt=-q`를 사용합니다. 작은 이차 해밀토니안 신경망의 세 계수를 128개 합성 상태의 도함수 관측으로 학습합니다. CPU float64, seed 824, Adam 400회이며 외부 데이터는 사용하지 않습니다.

필수 검증은 SciPy 해와 해석해 비교, PyTorch 자동미분 및 학습 수렴, Euler 에너지 증가와 leapfrog 에너지 오차의 비교, Matplotlib 그림 생성입니다. GPU 학습과 논문 조사는 이번 환경 검증 범위에 포함하지 않습니다.

출력은 `artifacts/oscillator/metrics.json`, `trajectories.npz`, `oscillator.png`입니다. 산출물은 Git에서 제외하고, 검증 기록은 `research/`에 보존합니다. 전체 실험은 로컬 CPU에서 실행되며 업로드하지 않습니다.

## 연구 원칙

가설·실패 수식·검증 결과를 기록하고 기존 연구 결과를 보존합니다. 논문을 조사할 때 원문 출처와 실제 확인 내용을 남깁니다. 문헌 검토 전에는 신규성을 주장하지 않습니다. 일반적인 로컬 코드 수정과 테스트는 자율적으로 진행하되, 외부 데이터 전송/Git push, 삭제, 유료 자원, 중대한 저장소 변경, 대규모 GPU 학습은 사전 승인을 받습니다. 자세한 작업 규칙은 `AGENTS.md`를 따릅니다.

## 현재 연구 기록

세션0002에서7개 가설과 수정안을12개 Cycle로 검토했습니다. 최종198개 결과 행과21개 통과 테스트를 보존했으며, 발명 후보를 확정하지 않았습니다.

- 종합 보고: [reports/research_summary.md](reports/research_summary.md)
- 현재 판정: [reports/current_best.md](reports/current_best.md)
- 수식/복잡도: [architectures](architectures)
- 실패와 신규성 확인 범위: [research/failures.md](research/failures.md), [research/novelty_review.md](research/novelty_review.md)
- 다음 실행: [research/next_steps.md](research/next_steps.md)

```bash
cd /workspace/physics
source /workspace/.physics-venv/bin/activate
export XDG_CACHE_HOME=/workspace/.cache MPLCONFIGDIR=/workspace/.cache/matplotlib
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m pytest -q
python scripts/verify_results.py results/session_0002_verified
```

전체 재현이 필요하면 `python -m experiments.run_research --output results/session_0003_replay`처럼 새 경로를 사용합니다. 기존 결과 경로는 거부합니다. 원래 repository root를 유지하고, 실제 물리 참조·후보·평가를 각각 `src/physics`, `src/operators`, 단일 실험 runner에 모았습니다. 비어 있는 domain별 폴더는 만들지 않았습니다. 연구 세션 종료 당시에는 GitHub 전송과 환경 publication을 수행하지 않았으며, 이후 반영 상태는 저장소의 PR과 환경 설정에서 확인합니다.
