# L / F5 — 반응 중첩으로 활성화하는 finite interaction

## 객체·입출력

개별source token의ID가아니라각유한source가만드는 **물리response footprint** m_i(x)를객체로삼는다. Source `(a_i,c_i,w_i)`,scalar k>0,lambda≥0,BC0에서 `-kT''=lambda expT+sum a_i s_i`를무차원화했다. Gaussian shape는고정된공식이며 amplitude/위치/width는연속값이다. N15field query와M1–4source를사용했다. Arbitrary geometry/비균질k/비영BC를처리한다고주장하지않는다.

## 실제 학습 가능한 식

각MLP는 `W3 tanh(W2 tanh(W1z+b1)+b2)+b3`다. 공유encoder는

\[
T_0(x)={\lambda\over k}x(1-x)\operatorname{softplus}(f_0(\lambda/k,x)),\\
m_i(x)={a_i\over k}x(1-x)\operatorname{softplus}(f_1(\lambda/k,\lambda a_i/k,c_i,w_i,x)).
\]

Train128sets의base/singleton labels만으로2000Adamstep을학습하고validation32로선택한encoder하나를모든비교모델에동결공유한다. 확인시singleton기준해를입력하지않는다. Footprint생성시간도추론비용에포함한다.

S=sum_i m_i에대해

\[
H=\lambda e^{T_0}[\operatorname{expm1}(S)-\sum_i\operatorname{expm1}(m_i)],\\
\tau_\theta=\operatorname{softplus}(t_0+t_1\lambda/k),\quad a(x)=1_{H(x)>\tau_\theta},\quad h=[H-\tau_\theta]_+,\\
K_\theta=\operatorname{softplus}(W_0+(\lambda/k)W_1+\log k\,W_2),\\
\hat T=T_0+S+x(1-x)\odot K_\theta h.
\]

Wi는15×15학습행렬, t0/t1는학습scalar다. 출력변화는cross reaction이threshold를넘는위치에서활성화된다. Explicit pair/triple enumeration 없이모든source수의exp mixed term을합친다. Attention query-key경쟁/softmax로source를선택하지않는다. **하지만선택대상은공간overlap영역이지개별source graph edge/부분집합의식별자가아니다.**

Interaction loss는true full−base−single mixed MSE/trainRMS² + 전체field MSE/trainRMS²다. 모든head는같은자료와동결encoder를사용하고1000Adam.003/3seed,validation만checkpoint를선택했다. Fullcounterfactualsubsetlabel은저장했지만fit에는full/base/singleton만사용한다. 사용한자료와생성reference수를구별한다.

## 보장과 빠진 성질

mi≥0이면 `exp(sum mi)-1≥sum(expm1 mi)`이므로H≥0이다. Softplus kernel/threshold/envelope와함께BC0,permutation invariance,nonnegative source영역의positivity를보장한다(유한산술범위). Single source면H0,lambda0이면H0이다. 일반signed온도/냉각source에같은positivity제약을무조건적용하지않는다.

보존수지/constitutive/PDE잔차/branch/안정성은보장하지않는다. Low-k확인은lambda0선형limit다. source위치변화의encoder오류와interaction head오류를분리해야한다. Threshold가작은실제효과도제거한다. Dense K를계산하므로zero activation이실제skip/FLOPs절감을뜻하지않는다.

## 기존 연산과 수식 대조

Phi(mi)=(mi,expm1mi), sum pooling 후 Psi로H를만들고threshold+positive operator를적용한 **nonlinear pooled-set/learned defect-response 구조**로정확히전개된다. 이graph를저장checkpoint에서NumPy로재구성해비교한다. 기존수학때문에자동기각하는것이아니라이구체적graph동일성과baseline효과를검사한다.

Baseline bilinear: H2=lambda expT0*(S²−sum mi²)/2. Volterra3는H2와H3=lambda expT0*(S³−sum mi³)/6의별도학습kernel을사용한다. 이것은response-coordinate의local polynomial Volterra이며모든공간좌표의arbitrary full tensor를조사한것은아니다. Attention baseline은source/query의MLP Q,K,V/softmax와decoder를쓰고 `f(all)-f(empty)-sum(f(single)-f(empty))`로single-anchor를맞춘다. 논문Transformer전체를재현했다고부르지않는다.

## 비용·재사용과 판정

Footprints O(MNw²),pooling O(MN),densekernel O(N²),저장O(MN+N²). 모든차수의계수를독립학습하는것이아니라물리반응법칙으로묶어싼비용을얻는다. Bilinear/Volterra도pooled power identity로M에선형이므로그들대비독점적복잡도이점은없다. B개 training set에서 encoder 학습 E회는 O(EBMNw²), frozen footprint head 학습 L회는 O(LB(MN+N²))이며 별도 footprint 캐시는 O(BMN)이다. Label생성은set마다2^M reference(최대16),실제훈련에쓰지않은subset도포함하며비용을기록한다.

**잠정Level2표현/선택probe;Level3미입증.** 다른reaction law에서는exp finite overlap을해당constitutive function의finite mixed relation으로다시정의해야한다. 모든물리도메인/공유weights로확장한것은아니다. Cycle27의새양수source반례에서는 **H만같고S가다르면필요correction이다르다**. 따라서물리적으로충실한footprint라는의미를유지하는H-only closure는범용적으로불충분하다. 상태정보를추가한1step Newton은기존solver이며새primitive로승격하지않았다.
