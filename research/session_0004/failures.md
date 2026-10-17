# Session 0004 반증과 실패 원인

과거 실패를 이름만 바꾸지 않았다. 새로운 수학을 쓰지 않았다는 이유만으로 배제하지 않는다. 다음은 이번 실제 반증이다.

| 대상 | 실제 증거 | 진단·분류 | 다음에 반복하지 말 것 |
|---|---|---|---|
| E의 fold 추정 | learned fold3.629–3.649 대정답3.51383; 없는해조건에서평균9.33개허위slot | g와fold의식별불충분;더큰fold도같은trainingfield로표현가능 | away-from-fold 온도라벨정확도만으로존재학습성공주장 |
| E의 near-fold | pair13.75%오차, 정확fold추가시1.07% | 추가물리정보효과;완전한dataset-only발명아님 | oracle정보를학습능력으로숨기기 |
| 단일 mean field | lambda2상대PDE잔차0.7175,수지오류3.4438 | 평균정상해는정상해가아님 | 다중정상해를단일MSE평균으로선택 |
| F의 temp 이점 과장 | IID pair0.630% 대additive0.693% | finite cross는작아온도오차이득제한적 | cross학습이좋다는이유로모든field의큰우월성주장 |
| F의 pair 완전합성 | lambda1.5 triple norm9.52e-5;lambda0~4e-17 | 실제고차반응존재,2차truncation오류 | 임의edit수·near-critical상태의정확합성보장 |
| F source OOD | pair온도약3.49%,cross약13.39% | center가학습중고정;field정보의모든변화를배운것아님 | 제한된parameter분포를geometry일반화라부르기 |
| G의 보존/정확도혼동 | 생성수지~1e-16이나IID PDE잔차평균5.00;direct1.57 | 온도L2보다curvature/구성식오류가큼 | 생성flux의보존만으로정확물리성공선언 |
| G low k | meanrelative69.34% | 분포/scale확장실패 | 큰GPU학습이나기존C006으로복귀 |
| G 방향성확장 | feedbackIID14.60%,lowdiff33.94%;모든모델16/16centroid방향맞음 | 방향은필요조건;정확solution/operator신규성은미입증 | 방향맞춤을단독gate효과로귀속 |
| G positivity | 양수source/영BC에서seed별최소-0.02998/-0.0001012/-0.004914,음수sample10/1/10 | 생성balance와최대원리도서로다름 | signedgeneric온도와필요한positivity조건을혼동 |
| G/H ablation의원인귀속 | feedback에활성parameter/compute가더많음 | 메커니즘효과와capacity효과미분리 | 총parameter수가같으므로공정하다고주장 |

이는 원리적불가능성/정보식별문제/분포실패/계산비교한계를 구분한 기록이다. G/F전체가 영구불가능하다는 결론은 아니다. 기존수학과연관되어있다는이유만으로 REJECT하지 않았고, 실제실패를근거로 E/G는REVISE, F는제한표현probe로KEEP했다. 신규성은모두미확인이다.
