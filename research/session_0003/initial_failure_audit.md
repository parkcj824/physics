# Session 0003 — 이전 실패의 재분류와 새 질문

기존 AGENTS, 후보 등록/실패/재개/문헌 기록, 종합 보고, 현재 판정, C001–C007 수식, 실행 코드 및 최종 결과를 읽은 뒤 작성했다. 기존 pytest21개가 통과했다. 기존 결과232개 파일의 SHA256을 `prior_result_sha256.json`에 기록했고 기존 source snapshot과 현재 source의 해시가 일치함을 확인했다. 기존 결과·모델·코드는 수정하지 않는다.

## 일곱 실패 축

| 후보 | 물리적 불가능성 | 표현력 | 분포 의존 | 해상도/물성 | 기존 계산과 동일성 | 비용 | 계산 객체 문제 |
|---|---|---|---|---|---|---|---|
| C001 | 현재 규칙은 reciprocity/heterogeneous response 불일치 | effective k 하나로 재료 배치 소실 | 균일 source의 특별해에 맞음; 학습 과적합은 아님 | 국소 source/BC/재료에 실패 | 정규화 kernel, 수정안은 고전 Green | dense N² | source-response 객체 전체가 불가능한 것은 아님 |
| C002 | 내부 보존은 가능, steady response 불충분 | finite-depth relaxation | 학습 실패가 아닌 step 조건 | CFL/condition number 의존 | Richardson | 반복 수 포함 시 불리 | 교환량과 정상해를 혼동 |
| C003 | 물리 불가능이 아니라 제한된 에너지 최적화 | rank4가 직교 source를 제거 | fixed span 문제 | 형상/새 mode 취약 | Ritz/Galerkin | setup 비용, rank tradeoff | variation 객체 자체의 불가능성은 아님 |
| C004 | signed deviation을 양수로 제한 | positive cone | 학습량으로 해결 불가 | gauge/온도 원점 불일치 | exponentiated-gradient 계열 | stiffness/overflow 위험 | 물리 변수 종류 선택 오류 |
| C005 | 이 유일 정상 문제에서 허구적 경로 의존 | 일정한 noncommuting generators | 미학습 generator 반례, 모든 event 모델 반증 아님 | endpoint 일관성 없음 | exponential flow composition | dense exponential 비용 | 물성 편집 순서와 실제 공간/시간 순서를 혼동 |
| C006 | reciprocal positive work는 타당 | 유한 age의 spectral support 한계 | 낮은 k에서 장시간 응답 미표현 | scale/range, supplied eigenbasis 의존 | positive exponential inverse approximation | eigensetup N³ | age 객체가 완전히 불가능한 것은 아님 |
| C007 | source 상수 변화가 물리 gauge라는 가정 오류 | 상수 source를 nullspace로 제거 | 파라미터 튜닝과 무관 | 모든 해상도에서 동일한 결함 | source Laplacian | dense N² | potential gauge를 source에 적용 |

## 이전 판정에서 더 엄격히 구분할 점

1. 고정 선형 PDE의 유일한 해는 반드시 `A^-1 b`다. **같은 해를 계산한다는 이유만으로 모든 다른 알고리즘을 동일 발명으로 볼 수는 없다.** 실제 연산 순서, 학습 표현, 재매개변수화, 준비/추론 비용을 비교해야 한다. 반대로 Sherman–Morrison처럼 갱신 수식까지 같으면 명확한 연산 중복이다.
2. 유한 지수합의 `g(0)<infinity`는 낮은 k 실패를 설명하는 필요 구조지만 관측된80.4%를 정량적으로 설명하려면 source의 고유모드 가중치를 계산해야 한다. 새 실험에서 `sum[(1-lambda*g)^2*(V^Tb/lambda)^2]`와 저장 오차를 비교한다. **C006 재학습/튜닝은 하지 않는다.**
3. 모든 물성을 동일한 양의 scalar로 곱하고 영 Dirichlet을 유지하면 비균질한 *고정 모양*의 k에서도 정확한 역스케일링이 성립한다. 임의로 다른 비균질 패턴을 scalar 평균으로 대체할 수 있다는 뜻은 아니다.
4. 비영 Dirichlet에서는 `T(s k,Q,g)=H_k g + (T(k,Q,0)/s)`다. 전체 T의 단순1/s가 아니라 source-induced 성분의1/s다. 수치로 두 경우를 분리한다.

## 새 최소 객체와 빠른 배제 기준

- A: **유한 물성 교란의 양방향 response packet**. 도함수만 쓰는 것과 유한 변화의 차이를 검사한다. 포화 분모를 전개하면 Sherman–Morrison/Woodbury이면 신규 후보에서 제외한다. Frozen packets의 누락된 cross-reaction도 검사한다.
- B: **관계만 보존하는 constraint witness**. witness가 다른 해를 구별하는지부터 확인한다. 동일 witness의 다른 source/해를 구성할 수 있으면 decoder를 크게 학습하지 않는다. 완전한 witness를 풀어야 한다면 기존 선형 제약 solve와 비교한다.
- C: **변환군의 원소로 표현한 변화량**. associative nonlinear 합성이 물리 source superposition과 양립하는지, 좌표를 풀면 기존 덧셈뿐인지 검사한다. A/B 조합이 아닌 대칭/변환 법칙 출발점이다.

이 셋이 모두 독립 primitive로 부적합하면 이름을 바꾸지 않고 다른 물리적 순서/합성 원리로 전환한다. 유효하지 않은 규칙에 대규모 학습을 투입하지 않는다. 이번 세션은 최대12Cycle이며 의미 없는 수를 채우지 않는다.
