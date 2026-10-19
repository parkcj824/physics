# F 확인 평가 전 고차·작은효과 지표 명확화

Interaction 모델의학습도중,첫확인평가전에정했다. full−base−single만보면pair효과가dominant할수있으므로 **beyond-pair = full mixed − 모든pair mixed의합**을별도로검사한다. 3-source에서는순수3차Möbius효과이고4-source에서는3/4차의합이다. Pair모델진단을위해pair입력forward를추가하지만그비용은fullfield추론latency와분리한다. Predictor와reference모두동일한inclusion–exclusion으로계산한다.

Relative mixed error는분모floor1e-12의전체평균을유지하고,참조mixed norm>1e-8인조건수/조건부평균도함께제시한다. Beyond-pair는absolute norm/error와참조norm>1e-10의조건부relative를기록한다. 이기준은무차원설명의진단threshold이지보편적허용오차가아니다. Zero/미세효과에서상대비율만으로순위를정하지않는다. 원래선언한지표/확인조건을삭제하지않는다.
