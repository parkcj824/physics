# C0b — 관측한 C0 결과에 따른 추가 conditioning 통제

C0 확인 결과를 관측한 후 정한 추가 가설이다. 열에서일반feature/staticphysical모델이original보다좋아물리feedback의고유이득을확인하지못했다. 그러나raw physical residual과learned feature의크기가다르면gate conditioning이달라질수있다. 기존확인집합을다시독립검증이라부르지않는다.

새확인조건 G IID64(seed51203),H IID64(seed51303)를지금고정한다. Train/validation은C0의같은128/32라벨을사용한다. Seed2005/2006/2007,Adam.003/1000steps. 세신규통제는모두P projection을포함해활성parameter9472로맞춘다.

- norm-physical: s=normalize(normalize(P h)+normalize(r_current)).
- norm-generic: s=normalize(P h).
- norm-static: s=normalize(normalize(P h)+normalize(r_initial));물리feature는처음한번만계산한다.

Normalize(v)=v/sqrt(mean(v²)+1e-12)는sample별feature norm을맞추며zero vector에서정의된다. Residual방향정보와반복갱신의효과를검사한다. 세모델은같은latentgate/decoder/초기seed를사용한다. 원래C0모델도이새입력으로재평가하지만재학습하지않는다. 모델18개동결후새평가1회. Profiler지원FLOPs/활성parameter/실제시간/오차/물리잔차를모두기록한다. Normalize는잔차의크기정보를제거하므로이실험도모든가능한물리정보효과의최종판정이아니다.

이는새독립발명Cycle이아닌C0의명시적후속인과통제다. 원래C0 결과/모델/프로토콜을그대로보존한다. 신규Architecture주장을위한오차튜닝으로해석하지않는다.
