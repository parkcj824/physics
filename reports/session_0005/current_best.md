# 현재 최고 후보와 미해결 과제

검증된 신규 범용 architecture나 Level3 primitive는 없다. L/F5는 양수 source/zero BC의 finite reaction overlap을 표현하는 제한된 Level2 probe다. 일부 pair mixed 지표에서 bilinear보다 좋았지만 cubic이 더 나은 고차 지표가 있고, 작은 실제 효과를 지우며 source 위치 OOD field 오차가 남는다. 보존·구성식·PDE 정확성은 보장하지 않는다.

다음 연구에 가장 구체적인 객체는 **집단 상태 S와 nonlinear finite defect H의 쌍**이다. H가 같고 실제 필요한 correction이 다른 양수-source 반례 때문에 S를 버릴 수 없었다. 그러나 `(S,H)`를 사용하는 새 학습 규칙은 아직 없으며, 지금 실행한 수리는 고전 1-step Newton이다.

G의 물리 feature 고유 이점은 제한된 통제에서 확립되지 않았다. J의 least-squares gradient, K의 AR2를 신규 primitive로 채택하지 않는다. 가장 정확한 수치 기준선은 여전히 고전 banded/Newton solver다.

실제 수식·성능·범위: [A–J 종합 보고서](research_summary.md), [Cycle 보고](cycles.md). 다음 실행: [Session0006/Cycle28 재개 지침](../../research/session_0005/next_steps.md).
