# Session 0005 사전 계획 — Cycle24부터

모델·새 데이터 생성 전에 고정한다. Session0002–0004 파일1515개를 SHA256으로 보호한다. 현재 검증된 Level3 primitive/범용 architecture는 없다. 새 수학 사용 여부와 계산 알고리즘/정보 흐름/실증 이점을 구분한다. CPU float64, torch thread1, 학습seed2005/2006/2007. 최대12Cycle/90분이며 의미 없는 수 채우기는 하지 않는다.

## 통제실험 C0 — G의 물리 정보 효과 (새 발명 Cycle 아님)

두 물리군: G는 -(kT')'=Q+lambda exp(T), H는 (vu-nu u')'=S의upwind이산식. N15. 이전 생성법의분포를사용하되새seed/새파일을사용한다. 과거checkpoint/확인label은학습/선택하지않는다.

모델: original G(width48,depth3), physics-plus-learned-feature(s=r+P h), learned-feature-only(s=P h), static-physics(s=r0를한번계산하고latent3층에공급), direct width64, learned Jacobi3step. Original/static은같은parameter구조다. Physics-plus/learned-only는정확히같은활성parameter와행렬계산구조를갖고physics residual주입유무만다르다. Generic feature도full raw physical input의latent에서생성한다. Direct는활성parameter가가까운width64,2000steps; 나머지는1000steps. 학습시간과GEMM profiler FLOPs를기록하고정확한FLOPs/완전한walltime동등을주장하지않는다. Pure original 대 generic 차이만으로인과결론을내리지않는다.

각물리train128/validation32. Gseed51001/51002, H51101/51102. Adam.003, validation100step마다선택. State/flux normalized supervisedMSE, residual penalty없음. Trainnormalize만사용. 확인은모든checkpoint동결후1회: G iid64(51003), lowk16(51004,lambda0), source16(51005), BC16(51006,lambda0), positive-source/BC0 16(51007,lambda0). H iid64(51103),lowdiff16(51104),direction32=16쌍(51105). Classical banded/Newton,3/32stepJacobi를동일입력으로비교한다.

## Cycle24 — 상태/수지/구성식의 유한 변화 객체

객체 (T,j;b,c;deltaT,deltaj,delta b,delta c). b=Dj-Q-lambda expT, c=j-J(T). 유한증분관계와 b-Dc=PDEresidual을직접검증한다. 동시감소를위한첫규칙이least-squares gradient/defect correction과동일하면그사실을입증하고새NN를억지학습하지않는다. 보존정확/구성식오류,작은field L2/큰PDE잔차,lowk,positive-source/BC0,유속반전의최소반례를구성한다. 전처리/condition cost와고전solver비용을분리한다.

## Cycle25 — 반응 중첩으로 활성화하는 상호작용

PDE: -kT''=lambda expT+sum a_i Gaussian(c_i,w_i), BC0, N15, scalar k∈[.9,1.2],lambda∈{0,1.5},a_i∈[.1,1.5]. 여러edit의singleton response m_i를공유학습encoder로예측한다. Pooledfinite reaction H=lambda exp(T0)[expm1(sum m_i)-sum expm1(m_i)]에서학습threshold를넘는공간response만켜는가설이다. 가변source수를허용하되정해진Polynomial Order/Rank증가를새후보로세지않는다. 알려진nonlinear pooledset/defect-correction과계산동일성이있는지감사한다.

Train128sets(seed52001),validation32(52002),active source수1/2/3;각set의모든2^M subset해를label로보존한다. Singleton encoder는train의base/singleton만2000Adamsteps(.003)학습,validation선택,seed2005하나를모든비교모델에공유한다. Interaction모델은thresholded reaction,unthresholded ablation,bilinear E2,Volterra E2+E3,anchored attention. 같은train자료/target/공유encoder/정규화,1000Adamsteps(.003),3seed. Kernel은field15x15이며임의resolution일반화는주장하지않는다. Absolute cross error와cross 크기,전체field오차를모두측정한다.

새확인: singleton16(52003),pair32(52004),triple32(52005),four-source16(52006),far16(52007,centers.12/.88),shift16(52008,centers가train범위밖),tiny16(52009,amplitude1e-3–5e-3),linear16(52010,lambda0). 확인은학습전seed/범위를고정하고모든interaction모델동결후1회만평가한다. 매우작은cross의상대오차는absolute norm과함께보고하고지표선택으로우월성을꾸미지않는다. Selection on/off 및strong/weak activation도기록한다.

## Cycle26 — 독립 원리: 관측 가능한 causal heat state

Cattaneo열식 T_t=-j_x, tau j_t+j=-kT_x에서한공간mode의(T,j)와온도history를비교한다. 온도만같고flux가다른물리반례를먼저확인한다. 완전한(T,j)가Markov이면history가본질적으로필요하다고주장하지않는다. 두온도관측으로미래를예측하는계산이AR(2)/observability/exponential propagation과동일하면독립primitive로채택하지않는다. Snapshot간격aliasing도검사한다. 소수계수최소제곱학습은신경망학습으로부르지않는다.

## 기록·확장·중단

표현/알고리즘보장/신규성을별도평가한다. J/F/K 세출발가설을서로독립적으로관리한다. 새수식이기존solver/attention/GRU와같으면새이름으로유지하지않는다. 1D잔차/positivity/OOD를통과하기전2D/CFD/GPU로확장하지않는다. 모델수정이확인결과후라면새가설과새확인set을별도등록한다. 과거403요청을반복하지않고저장된공식source만확인한범위를명시한다. GitHub/연구데이터전송/유료/GPU/삭제없음.
