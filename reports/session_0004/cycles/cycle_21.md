# Cycle 21 — G — state/balance co-generation

1. **출발 원리:** 보존과구성식의공동계산.
2. **최소 계산 객체:** latent→(T,j)→constitutive mismatch. 단위/입력은 architectures/session_0004의 해당 후보 수식 참조.
3. **구현 연산:** `h+=tanh(Wh+Ur+tanh(Gh)*tanh(Vr))`. 내부MLP와고정/학습변수는후보파일에완전히전개했다.
4. **기존 계산과 차이:** G의학습정보흐름을일반MLP/mixture/bilinear/gated residual구조와대조했다. 기존수학사용만으로기각하지않았다.
5. **검증 방정식:** Bratu/1D정상열반응/variable-k 열전도중해당후보범위. 실제lambda,k,source,BC와정답은data/arrays에있다.
6. **실제 결과:** seed3개의 sample평균 상대오차는 아래와 같다. 전체수치는 `results/session_0004/cycle_21_evidence.json`이며 실제입력/예측/정답/절대오차/보존/비용은confirmation 또는transport_confirmation의arrays경로와1:1대응한다. 개발결과→수학반례→동결모델의사전정의확인평가를분리했다.

| 모델 | 조건 | mean_relative_l2 |
|---|---|---|
| direct | boundary | 0.082783 ± 0.00996 |
| direct | iid | 0.079218 ± 0.0065 |
| direct | lowk | 0.93961 ± 0.0839 |
| direct | source | 0.18199 ± 0.0409 |
| feedback | boundary | 0.03241 ± 0.00836 |
| feedback | iid | 0.074527 ± 0.0146 |
| feedback | lowk | 0.69344 ± 0.0112 |
| feedback | source | 0.14461 ± 0.0197 |
| no_feedback | boundary | 0.096428 ± 0.00894 |
| no_feedback | iid | 0.12747 ± 0.0228 |
| no_feedback | lowk | 0.82139 ± 0.0308 |
| no_feedback | source | 0.2927 ± 0.0523 |

7. **자기반박:** 생성flux 보존은 constitutive/PDE 정확도를 보장하지 않음.
8. **신규성 검토:** 기존수학/신경연산과의구체관계를기록했다. 저장된공식source만확인했고full paper/특허미검토이므로신규성미확인이다.
9. **판정:** REVISE/UNRESOLVED. 새로운범용architecture의입증은없다.
10. **다음 가설:** jointdecoder와feedback의추가효과를분리. 미실행가설을실행결과로세지않는다.
