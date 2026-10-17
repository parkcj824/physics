# 확인 결과 후 발견한 confound의 후속 통제 — 새 확인 집합

최초 확인 결과/사후 gradient audit를 본 뒤, G/H feedback8704 대 direct5551 활성 parameter 차이를 발견했다. 이는 새 primitive Cycle이 아니라 Cycle21/23의추가통제다. 원래확인집합을모델개발에사용한뒤독립검증이라고재명명하지않는다.

이문서를새통제모델학습/새확인평가전에고정한다. 새directMLP는width64,같은depth3,활성parameter8415로feedback8704에가깝게한다. Encoder/BCdecoder/labels/normalization/loss/Adam.003/3seed는같다. Feedback의층당계산과walltime이더커서directcontrol은3000step(기존feedback1500)으로정한다. 정확한FLOPs동등은주장하지않고actualtime을기록한다. 기존G/H checkpoint는그대로동결유지하며다시학습하지않는다.

새평가는G IID32(seed45003), H IID32(seed45004). 기존train128/validation32만모델학습/선택에사용한다. 새두확인파일은학습전에생성/해시저장하고6개control모델동결후에한번평가한다. 원래G/H3종모델도같은새입력으로재평가한다. 조건별온도오차/flux/PDE/수지/추론비용을모두기록한다. 새OOD/geometry검증이나새데이터효율곡선은아니다. 추가Hyperparameter선택/모델수정은이평가후하지않는다.
