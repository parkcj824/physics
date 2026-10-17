# Cycle 23 — G 방향성 확장 — constitutive law의 역할

1. **출발 원리:** 상태와방향성있는flux의공동생성.
2. **최소 계산 객체:** 기존Gblock에upwind이류구성식. 단위/입력은 architectures/session_0004의 해당 후보 수식 참조.
3. **구현 연산:** `j=j0+h cumsum(S); r=j-(v+uL+v-uR-nu grad u)`. 내부MLP와고정/학습변수는후보파일에완전히전개했다.
4. **기존 계산과 차이:** H의학습정보흐름을일반MLP/mixture/bilinear/gated residual구조와대조했다. 기존수학사용만으로기각하지않았다.
5. **검증 방정식:** Bratu/1D정상열반응/variable-k 열전도중해당후보범위. 실제lambda,k,source,BC와정답은data/arrays에있다.
6. **실제 결과:** seed3개의 sample평균 상대오차는 아래와 같다. 전체수치는 `results/session_0004/cycle_23_evidence.json`이며 실제입력/예측/정답/절대오차/보존/비용은confirmation 또는transport_confirmation의arrays경로와1:1대응한다. 개발결과→수학반례→동결모델의사전정의확인평가를분리했다.

| 모델 | 조건 | mean_relative_l2 |
|---|---|---|
| direct | direction | 0.27644 ± 0.0404 |
| direct | iid | 0.43682 ± 0.0719 |
| direct | lowdiff | 0.7637 ± 0.0605 |
| feedback | direction | 0.093155 ± 0.0383 |
| feedback | iid | 0.14599 ± 0.0251 |
| feedback | lowdiff | 0.33939 ± 0.039 |
| no_feedback | direction | 0.23366 ± 0.0374 |
| no_feedback | iid | 0.39982 ± 0.084 |
| no_feedback | lowdiff | 0.94399 ± 0.155 |

7. **자기반박:** 모든모델이centroid방향을맞혔지만feedback도14.6%IID오차와33.9%낮은확산오차.
8. **신규성 검토:** 기존수학/신경연산과의구체관계를기록했다. 저장된공식source만확인했고full paper/특허미검토이므로신규성미확인이다.
9. **판정:** REVISE; 부분적인architecture재사용. 새로운범용architecture의입증은없다.
10. **다음 가설:** 구성식잔차를줄이는새오차표현을solver재현과구분. 미실행가설을실행결과로세지않는다.
