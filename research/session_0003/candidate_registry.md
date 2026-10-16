# Session 0003 후보 등록

이전 C001–C007을 재명명하지 않았다. 새 가설4개 중 독립 신규 연산으로 확정한 것은 없다. Cycle0은 기존 결과의 감사/재현이며 새 연구 Cycle 수에 넣지 않는다.

| 후보 | 물리 출발 원리 | 계산 객체 | 실제 학습 | 물리/표현 반증 | 계산 동일성 | 최종 판정 |
|---|---|---|---|---|---|---|
| A | 유한 교란과 방향성 있는 response | 좌·우 response, compliance, 유한 변화 packet | 미실행; base inverse 공급 | frozen cross-response와 tangent 실패; full update는 정확 | Sherman–Morrison/Woodbury | REJECT as invention |
| B | 제약 관계의 충분성 | coarse balance 및 invariant witness | 미실행; 큰 decoder 전 정보 충돌 반증 | 동일 witness·norm의 다른 양수 source/해; 같은 mass/energy의 다른 위상 | 최소 norm closure/projection; input bottleneck | REJECT current witness-only architecture |
| C | 대칭/변환군의 변화 합성 | asinh 군좌표의 변화량 | gamma4 고정, 학습 미실행 | raw source superposition·질량 위반 | coordinate-conjugate addition | REJECT |
| D | 실제 공간의 ordered composition | (R,S,C) 구간 입출력 relation | theta1개를128label로 weighted LS | theta1은 exact; cross moment 제거 시 실패 | affine transfer matrix + prefix integration | REJECT as invention; classical control |

새 표현이라는 사실만으로 새로운 primitive는 아니다. A는 finite tangent를 넘어서는 분모가 있지만 이미 알려진 갱신식이다. B는 인증과 예측 표현의 정보량을 혼동하면 안 된다. C의 nonlinear appearance는 새 coupling이 아니다. D의 물리적 공간 순서는 C005의 허구적 물성 edit 순서와 다르지만 전체 알고리즘은 고전 transfer다.

**현재 최고 발명 후보 없음.** 정확한 numerical control과 다음 독립 질문은 `../../reports/session_0003/current_best.md`와 `next_steps.md`에 있다.

## 실행 연결

- 상세 수식 A–G: `../../architectures/session_0003/candidate_A.md`부터`candidate_D.md`.
- Cycle13–18: `../../reports/session_0003/cycles/`.
- 실제 입력/예측/기준값/오차/비용: `../../results/session_0003/metrics.json`, `arrays/`.
- 과거 보존/학습 재현 검증: `../../results/session_0003/validation/integrity.json`.
