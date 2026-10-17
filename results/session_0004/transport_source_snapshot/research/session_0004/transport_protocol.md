# Cycle 23 — 확인용 결과를 보기 전 결정한 G의 방향성 반증

G feedback의 개발 validation loss0.00473은 no-feedback0.0229/direct0.0120보다 작았다. 이 결과와 물리적 공동생성 구조를 근거로, 확인 결과를 보기 전에 **같은 3-layer 공동생성 mechanism**을 정상 이류–확산에서 검사하기로 했다. 새 독립 후보가 아니라 G의 명시적 domain extension이다. Domain 간 weights는 공유하지 않는다.

PDE: `(v u - nu u_x)_x=S`, 1D Dirichlet. Generated flux는 `j0+h cumsum(S)`, mismatch는 upwind constitutive flux와의 차이다. 온도/상태 envelope, latent gate, supervised state/flux loss, depth3/width48/1500step/Adam.003/seed1004–1006는 유지한다. Nonlinear reaction에서 source-dependent flux가 들어간 곳을 이류 constitutive law로 바꾸며 full raw input의 마지막 scalar는 velocity다.

Train128(seed44001), validation32(seed44002), n15, v는양/음 uniform(.5,1.5), nu는.05–.15의 부드러운양수field, sourceGaussian와작은균일성분, BC[-.1,.2]. 확인은 iid32(seed44003), 방향쌍16×2(v±1, sourcecenter.3, BC0, nu.05–.15), low-diffusion16(nu.015–.025, seed44004). 모두 준비 시고정하고 모델동결후평가한다. feedback/no_feedback/direct 세모델을같은label/loss/3seed로비교한다.

독립고전upwind/banded reference, flux수지/constitutive/PDE오차와방향반전후centroid를검사한다. Upwind reference는인공확산을포함한이산기준선이며연속해와동일하다고주장하지않는다. 이벤치마크는이류–확산이며비압축성Navier–Stokes 또는RANS의성공을뜻하지않는다. 비선형유동확장은별도구조/검증이필요해이번실험으로대체하지않는다.
