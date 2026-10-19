# 초기 모델 확인 전에 정한 실행 세부사항

- F의최소low-k검사는seed52011,k0.1–0.2,lambda0의16조건을추가한다. Reaction/nonlinear low-k는steady해존재부터검증해야하므로선형limit와혼동하지않는다. 이dataset은interaction모델학습·확인전에고정한다. 원래확인조건을대체하지않는다.
- F는모든subset의기준해를저장하지만fit은full-set field/mixed target과공유encoder용base/singleton만사용한다. 추가pair/triple subset은고차audit에사용한다. 생성label수와학습에쓴label수를구분한다.
- F의kernel은공통mean에표준편차.001의seed별초기화차이를넣는다. Gate/no-gate는같은seed에서같은kernel초기화를공유한다. Encoder는공유동결checkpoint다.
- ClassicalJacobi비용은매추론마다PyTorch model을새로만드는비용을포함하지않도록batch NumPy로측정한다. 실제source/flux/BC전처리는포함한다. 이는확인결과를관측하기전의계측수정이다.
- C0 LearnedJacobi의alpha초기값은모든seed에서동일한0이다. 따라서그3회는동일한결정론적fitting의재현이며독립초기화분산으로해석하지않는다.
