# 확인용 평가 전 지표 명확화

확인용 결과를 모델 개발에 사용하기 전에 정했다. Two-head의 집합 평가는 sample별 **field 전체**의 두 permutation 중 작은 L2를 사용한다. 점마다 정렬해 가상 branch를 만들지 않고 raw prediction과 assignment를 저장한다.

존재 gate가 선언한 branch 수와 참조해의 상대 L2 오차5% 안에 있는 branch 수를 구별한다. 이5%는 결과 확인 전에 정한 진단 기준이지 보편적인 물리 허용오차가 아니다.

안정성 판정은 N15 이산 오차와 fold 근방을 혼동하지 않도록 field를 N255로 query해 선형화 최대 고유값을 계산한다. 이는 평가용 계산이며 field 추론 시간에 숨기지 않는다.

추론 비용은 F의 최종 field 하나, G의 온도와 flux, E의 두 branch를 동일 출력의 고전 기준선과 비교한다. F quartet의 추가 예측은 정확도 진단용이며 timed inference와 분리한다. Tensor 변환/정규화 또는 source/matrix 조립은 포함하고, disk I/O와 안정성 진단은 제외한다.

F의 기본 direct MLP(width48)가 pair 모델보다 parameter수가 적으므로, 확인 평가 전에 width64 direct MLP를 추가한다. 선정 근거는 parameter budget이며 확인 결과를 보지 않았다. 같은 quartet/정규화/1500steps/validation/3seed를 사용한다. 추가 width64 control과 사전 계획의 정확한 fold oracle 모두 checkpoint 동결 이후에 확인 평가한다. 모델 수는 총36개가 된다.
