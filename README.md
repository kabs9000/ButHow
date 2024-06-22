A General Theoretical Paradigm to Understand Learning from Human Preferences

Mohammad Gheshlaghi Azar Mark Rowland Bilal Piot Daniel Guo Daniele Calandriello Michal Valko R´emi Munos Google DeepMind

Abstract

The prevalent deployment of learning from human preferences through reinforcement learning (RLHF) relies on two important ap- proximations: the first assumes that pairwise preferences can be substituted with point- wise rewards. The second assumes that a reward model trained on these pointwise re- wards can generalize from collected data to out-of-distribution data sampled by the pol- icy. Recently, Direct Preference Optimisa- tion (DPO) has been proposed as an approach that bypasses the second approximation and learn directly a policy from collected data without the reward modelling stage. How- ever, this method still heavily relies on the first approximation.

In this paper we try to gain a deeper theo- retical understanding of these practical algo- rithms. In particular we derive a new general objective called ΨPO for learning from hu- man preferences that is expressed in terms of pairwise preferences and therefore bypasses both approximations. This new general ob- jective allows us to perform an in-depth anal- ysis of the behavior of RLHF and DPO (as spe- cial cases of ΨPO) and to identify their poten- tial pitfalls. We then consider another special case for ΨPO by setting Ψ simply to Identity, for which we can derive an efficient optimi- sation procedure, prove performance guaran- tees and demonstrate its empirical superior- ity to DPO on some illustrative examples.

Under review.

1 Introduction

Learning from human preferences (Christiano et al., 2017) is a paradigm adopted in the natural language processing literature to better align pretrained (Rad- ford et al., 2018; Ramachandran et al., 2016) and instruction-tuned (Wei et al., 2022) generative lan- guage models to human desiderata. It consists in first collecting large amounts of data where each datum is composed of a context, pairs of continuations of the context, also called generations, and a pairwise human preference that indicates which generation is the best. Then, a policy generating good generations given a context is learnt from the collected data. We frame the problem of learning from human preferences as an of- fline contextual bandit problem (Lu et al., 2010). The goal of this bandit problem is that given a context to choose an action (playing the role of the generation) which is most preferred by a human rater under the constraint that the resulting bandit policy should be close to some known reference policy. The constraint of staying close to a known reference policy can be satisfied e.g., by using KL regularisation (Geist et al., 2019) and its role is to avoid model drift (Lazaridou et al., 2020; Lu et al., 2020).

A prominent approach to tackle the problem of learn- ing from human preferences is through reinforcement learning from human feedback (RLHF, Ouyang et al., 2022; Stiennon et al., 2020) in which first a reward model is trained in the form of a classifier of preferred and dispreferred actions. Then the bandit policy is trained through RL to maximize this learned reward model while minimizing the distance with the refer- ence policy. Recently RLHF has been used successfully in solving the problem of aligning generative language models with human preferences (Ouyang et al., 2022). Furthermore recent works such as direct preference op- timisation (DPO, Rafailov et al., 2023) and (SLiC-HF, Zhao et al., 2023) have shown that it is possible to optimize the bandit policy directly from human pref- erences without learning a reward model. They also have shown that on a selection of standard language

arXiv:2310.12036v2 [cs.AI] 22 Nov 2023

---

Understand Learning from Human Preferences

tasks they are competitive with the state of the art RLHF while they are simpler to implement and require less resources.

Despite this practical success, little is known regard- ing theoretical foundations of these practical meth- ods. Notable exceptions, that consider specific special cases, are (Wang et al., 2023; Chen et al., 2022) and prior work on preference-based (Busa-Fekete et al., 2014, 2013) and dueling bandits and RL (Novoseller et al., 2020; Pacchiano et al., 2023). However, these theoretical works focus on providing theoretical guar- antees in terms of regret bounds in the standard bandit setting and they do not deal with the practical setting of RLHF, DPO and SLiC-HF.

In this work, our focus is on bridging the gap between theory and practice by introducing a simple and gen- eral theoretical representation of the practical algo- rithms for learning from human preferences. In par- ticular, we show that it is possible to characterise the objective functions of RLHF and DPO as special cases of a more general objective exclusively expressed in terms of pairwise preferences. We call this objective Ψ-preference optimisation (ΨPO) objective, where Ψ is an arbitrary non-deceasing mapping. We then ana- lyze this objective function in the special cases of RLHF and DPO and investigate its potential pitfalls. Our the- oretical investigation of RLHF and DPO reveals that in principle they can be both vulnerable to overfitting. This is due to the fact that those methods rely on the strong assumption that pairwise preferences can be substituted with ELo-score (pointwise rewards) via a Bradley-Terry (BT) modelisation (Bradley and Terry, 1952). In particular, this assumption could be prob- lematic when the (sampled) preferences are determin- istic or nearly deterministic as it leads to over-fitting to the preference dataset at the expense of ignoring the KL-regularisation term (see Sec. 4.2). We then present a simple solution to avoid the problem of over- fitting, namely by setting Ψ to identity in the ΨPO. This method is called Identity-PO (IPO) and by con- struction bypasses the BT modelisation assumption for preferences (see Sec. 5). Finally, we propose a practi- cal solution, via a sampled loss function (see Sec. 5.2), to optimize this simplified version of ΨPO empirically and, we compare its performance with DPO on simple bandit examples, providing empirical support for our theoretical findings (see Sec. 5.3 and Sec. 5.4).

2 Notations

In the remaining, we build on the notations of DPO (Rafailov et al., 2023). Given a context x ∈ X, where X is the finite space of contexts, we assume a finite action space Y. A policy π ∈ ∆X Y associates

to each context x ∈ X a discrete probability distri- bution π(.|x) ∈ ∆Y where ∆Y is the set of discrete distributions over Y. We denote µ ∈ ∆X Y the behav- ior policy. From a given context x, let y, y′ ∼ µ(x) be two actions generated independently by the refer- ence policy. These are then presented to human raters who express preferences for one of the generations, de- noted as yw ≻ yl where yw and yl denote the preferred and dispreferred actions amongst {y, y′} respectively. We then write true human preference p∗(y ≻ y′|x) the probability of y being preferred to y′ knowing the con- text x. The probability comes from the randomness of the choice of the human we ask for their preference. So p∗(y ≻ y′|x) = Eh[I{h prefers y to y′ given x}], where the expectation is over humans h. We also introduce the expected preference of a generation y over a dis- tribution µ knowing x, noted p∗(y ≻ µ|x), via the following equation:

p∗(y ≻ µ|x) = E y′∼µ(.|x)[p∗(y ≻ y′|x)] .

For any two policy π, µ ∈ ∆X Y and a context distribu- tion ρ we denote the total preference of policy π to µ as

p∗ ρ(π ≻ µ|x) = E x∼ρ y∼π(.|x) [p∗(y ≻ µ|x)] .

In practice, we do not observe p∗ directly, but sam- ples I(y, y′|x) from a Bernoulli distribution with mean p∗(y ≻ y′|x) (i.e., I(y, y′|x) is 1 with probability p∗(y ≻ y′|x) and 0 otherwise). In particular, we assume we have access to the preferences through a dataset of rated generations D = (xi, yi, y′ i)N i=1 = (xi, yw,i ≻ yl,i)N i=i, where N is the dataset size. In addition, for a general finite set S, a discrete proba- bility distribution η ∈ ∆S and a real function f ∈ RS, we note the expectation of f under η as Es∼η[f(s)] = � s∈S f(s)η(s). For a finite dataset D = (si)N i=1, with si ∈ S for each i, and a real function f ∈ RS, we denote the empirical expectation of f under D as Es∼D[f(s)] = 1

N �N i=1 f(si).

3 Background

3.1 Reinforcement Learning from Human Feedback (RLHF)

The standard RLHF paradigm (Christiano et al., 2017; Stiennon et al., 2020) consists of two main stages: (i) learning the reward model; (ii) policy optimisation us- ing the learned reward. Here we provide a recap of these stages.

---

Gheshlaghi Azar, Rowland, Piot, Guo, Calandriello, Valko, Munos

3.1.1 Learning the Reward Model

Learning a reward model consists in training a binary classifier to discriminate between the preferred and dis- preferred actions using a logistic regression loss. For the classifier, a popular choice is Bradley-Terry model: for a given context x and action y, we denote the point- wise reward, which can also be interpreted as an Elo score, of y given x by r(x, y). The Bradley-Terry model represents the preference function p(y ≻ y′|x) (classi- fier) as a sigmoid of the difference of rewards:

p(y ≻ y′|x) = σ � r(x, y) − r(x, y′) � , (1)

where σ(·) denotes the sigmoid function and plays the role of normalisation. Given the dataset D = (xi, yw,i ≻ yl,i)N i=1 one can learn the reward function by optimizing the following logistic regression loss

L(r) = −E(x,yw,yl)∼D

� log (p(yw ≻ yl|x)) � . (2)

Assuming that p∗(y ≻ y′|x) conforms to the Bradley- Terry model, one can show that as the size of the dataset D grows, p(y ≻ y′|x) becomes a more and more accurate estimate of true p∗(y ≻ y′|x) and in the limit converges to p∗(y ≻ y′|x).

3.1.2 Policy Optimisation with the Learned Reward

Using the reward (Elo-score) r(x, y) the RLHF objective is simply to optimize for the policy π ∈ ∆X Y that max- imizes the expected reward while minimizing the dis- tance between π and some reference policy πref ∈ ∆X Y through the following KL-regularized objective func- tion:

J(π) = Eπ[r(x, y)] − τDKL(π || πref) , (3)

in which the context x is drawn from ρ and the action y is drawn from π(.|x). The divergence DKL(π||πref) is defined as follows:

DKL(π || πref) = Ex∼ρ[KL(π(.|x) || πref(.|x))] .

where:

KL(π(.|x) || πref(.|x)) = Ey∼π(.|x)

� log � π(y|x)

πref(y|x)

�� .

The objective in Equation (3) is essentially optimized by PPO (Schulman et al., 2017) or similar approaches.

The combination of RLHF +PPO has been used with great success in practice (e.g., InsturctGPT and GPT- 4 Ouyang et al., 2022; OpenAI, 2023).

3.2 Direct Preference Optimisation

An alternative approach to the RL paradigm described above is direct preference optimisation (DPO; Rafailov et al., 2023), which avoids the training of a reward model altogether. The loss that DPO optimises, given an empirical dataset D, as a function of π, is given by

min π E(x,yw,yl)∼D

�

− log σ

�

τ log �π(yw|x)

π(yl|x)

� −

τ log �πref(yw|x)

πref(yl|x)

� ��

.

(4)

In its population form, the loss takes on the form

min π E x∼ρ y,y′∼µ

�

− p∗(y ≻ y′|x) log σ

�

τ log � π(y|x)

π(y′|x)

� −

τ log � πref(y|x)

πref(y′|x)

� ��

.

(5)

Rafailov et al. (2023) show that when (i) the Bradley- Terry model in Equation (1) perfectly fits the pref- erence data and (ii) the optimal reward function r is obtained from the loss in Equation (2), then the global optimisers of the RLHF objective in Equation (3) and the DPO objective in Equation (5) perfectly coincide. In fact, this correspondence is true more generally; see Proposition 4 in Appendix B.

4 A General Objective for Preference Optimisation

A central conceptual contribution of the paper is to propose a general objective for RLHF, based on maxi- mizing a non-linear function of preferences. To this end, we consider a general non-decreasing function Ψ : [0, 1] → R, a reference policy πref ∈ ∆X Y , and a real positive regularisation parameter τ ∈ R∗ +, and de- fine the Ψ-preference optimisation objective (ΨPO) as

max π E x∼ρ y∼π(.|x) y′∼µ(.|x)

[Ψ(p∗(y ≻ y′|x))] − τDKL(π || πref) . (6)

This objective balances the maximisation of a po- tentially non-linear function of preference probabili- ties with the KL regularisation term which encourages policies to be close to the reference πref. This is mo- tivated by the form of Equation (3), and we will see in the next subsection that it strictly generalises both RLHF and DPO, when the BT model holds.

---

Understand Learning from Human Preferences

4.1 A Deeper Analysis of DPO and RLHF

In the remaining, we omit the dependency on x for the ease of notations. This is without losing generality and all the following results are true for all x ∈ Supp(ρ).

We first connect DPO and RLHF with the Ψ-preference objective in Equation (6), under the special choice of Ψ(q) = log(q/(1 − q)). More precisely, the following proposition establishes this connection.

Proposition 1. Suppose Ψ(q) = log(q/(1 − q)). When the Bradley-Terry model holds for p∗, that is, there exists r : Y → R such that

p∗(y ≻ y′) = σ(r(y) − r(y′)) ,

then the optimal policy for Equation (6), for the RLHF objective in Equation (3), and for the standard DPO objective in Equation (5) are identical.

Proof. Note that under the assumption that the Bradley-Terry model holds, we have

E y′∼µ[Ψ(p∗(y ≻ y′))] = E y′∼µ

� Ψ � er(y)

er(y) + er(y′)

��

= E y′∼µ[log(er(y)/er(y′))]

= E y′∼µ[r(y) − r(y′)]

= r(y) − E y′∼µ[r(y′)] .

This is equal to the reward in Equation (3), up to an additive constant, and so it therefore follows that the optimal policy for Equation (6) and for optimizing the objective in Equation (3) are identical. Further, as shown by Rafailov et al. (2023), the optimal policy for the DPO objective in Equation (5) and the objective in Equation (3) are identical, which gives the statement of the proposition.

Applying this proposition to the objective function of Equation (6), for which there exists an analytical solu- tion, reveals that under the BT assumption the closed- form solution to DPO and RLHF can be written as

π∗(y) ∝ πref(y) exp � τ −1Ey′∼µ[Ψ(p∗(y ≻ y′))] � . (7)

The derivations leading to Equation 7 is a well known result and is provided in App. A.1 for completeness.

4.2 Weak Regularisation and Overfitting

It is worth taking a step back, and asking what kinds of policies the above objective leads us to discover. This highly non-linear transformation of the preference probabilities means that small increases in preference

probabilities already close to 1 are just as incentivized as larger increases in preference probabilities around 50%, which may be undesirable. The maximisation of logit-preferences, or Elo score in game-theoretic termi- nology, can also have counter-intuitive effects, even in transitive settings (Bertrand et al., 2023).

Consider the simple example where we have two ac- tions y and y′ such that p∗(y ≻ y′) = 1, i.e., y is always preferred to y′. Then the Bradley-Terry model would require that (r(y) − r(y′)) → +∞ to satisfy (1). If we plug this into the optimal policy (7) then we would get that π∗(y′)

π∗(y) = 0 (i.e., π∗(y′) = 0) irrespective of what constant τ is used for the KL-regularisation. Thus the strength of the KL-regularisation becomes weaker and weaker the more deterministic the preferences.

The weakness of the KL-regularisation becomes even more pronounced in the finite data regime, where we only have access to a sample estimate of the preference ˆp(y ≻ y′). Even if the true preference is, e.g., p∗(y ≻ y′) = 0.8, empirically it can be very possible when we only have a few data points to estimate ˆp(y ≻ y′) = 1, in which case the empirical optimal policy would make π(y′) = 0 for any τ. This means that overfitting can be a substantial empirical issue, especially when the context and action spaces are extremely large as it is for large language models.

Why may standard RLHF be more robust to this problem in practice? While a purported advantage of DPO is that it avoids the need to fit a reward function, we ob- serve that in practice when empirical preference prob- abilities are in the set {0, 1}, the reward function ends up being underfit. The optimal rewards in the presence of {0, 1} preference probabilities are infinite, but these values are avoided, and indeed regularisation of the re- ward function has been observed to be an important aspect of RLHF training in practice (Christiano et al., 2017). This underfitting of the reward function is thus crucial in obtaining a final policy that is sufficiently regularised towards the reference policy πref, and DPO, in avoiding the training of the reward function, loses the regularisation of the policy that the underfitted reward function affords.

While standard empirical practices such as early- stopping can still be used as an additional form of regularisation to curtail this kind of overfitting, in the next section, we will introduce a modification of the ΨPO objective such that the optimal empirical policy can be close to πref even when preferences are deter- ministic.

---

Gheshlaghi Azar, Rowland, Piot, Guo, Calandriello, Valko, Munos

5 IPO: ΨPO with identity mapping

We have observed in the previous section that DPO is prone to overfitting, and this stems from a combination of the unboundedness of Ψ, together with not training an explicit reward function. Not training a reward function directly is a clear advantage of DPO, but we would like to avoid the problems of overfitting as well.

This analysis of DPO motivates choices of Ψ which are bounded, ensuring that the KL regularisation in Equa- tion 6 remains effective even in the regime of {0, 1}- valued preferences, as it is often the case when working with empirical datasets. A particularly natural form of objective to consider is given by taking Ψ to be the identity mapping in Equation (6), leading to direct regularized optimisation of total preferences:

max π p∗ ρ(π ≻ µ) − τDKL(π || πref) . (8)

The standard approach to optimize the objective func- tion of Equation (8) is through RLHF with the choice of reward r(y) = p∗(y ≻ µ). However both using RL and estimating the reward model r(y) can be costly. Inspired by DPO one would like to devise an empirical solution for the optimisation problem of Equation (8) which can directly learn from the preference dataset. Thus it would be able to avoid RL and reward model- ing altogether.

5.1 Derivations and Computationally Efficient Algorithm

As with DPO, it will be beneficial to re-express Equa- tion (8) as an offline learning objective. To derive such an expression, we begin by following the derivation of Rafailov et al. (2023), manipulating the analytic ex- pression for the optimal policy into a system of root- finding problems. As in the previous section, we drop dependence on the context x from our notation, as all arguments can be applied on a per-context basis.

Root-finding problems. Let g(y) = Ey′∼µ[Ψ(p∗(y ≻ y′))]. Then we have

π∗(y) ∝ πref(y) exp(τ −1g(y)) . (9)

For any y, y′ ∈ Supp(πref), we therefore have

π∗(y) π∗(y′) = πref(y)

πref(y′) exp(τ −1(g(y) − g(y′))) . (10)

By letting

h∗(y, y′) = log �π∗(y)πref(y′)

π∗(y′)πref(y)

�

and rearranging Equation (10), we obtain

h∗(y, y′) = τ −1� g(y) − g(y′) � . (11)

The core idea now is to consider a policy π, define

hπ(y, y′) = log � π(y)πref(y′

π(y′)πref(y)

� ,

and aim to solve the equations:

hπ(y, y′) = τ −1� g(y) − g(y′) � . (12)

Loss for IPO. We now depart from the approach to the analysis employed by Rafailov et al. (2023), to ob- tain a novel offline formulation of Equation (6), in the specific case of Ψ as the identity function. In this case, Equation (12) reduces to

hπ(y, y′) = τ −1� p∗(y ≻ µ) − p∗(y′ ≻ µ) � .

We begin by re-expressing these root-finding problems as a single optimisation problem L(π):

L(π) = E y,y′∼µ

�� hπ(y, y′) − p∗(y ≻ µ) − p∗(y′ ≻ µ)

τ

�2�

.

(13)

One can easily show that for the choice of π∗ we have L(π∗) = 0. Thus π∗ is a global minimizer of L(π). The following theorem establishes the uniqueness of this solution.

Theorem 2 (Uniqueness of Global/Local Optima). Assume that Supp(µ) = Supp(πref) and define Π to be the set of policies π such that Supp(π) = Supp(µ). Then π �→ L(π) has a unique local/global minimum in Π, which is π∗.

Proof. By assumption, π∗ ∈ Π, and by definition ∀π ∈ Π, L(π) ≥ 0 as L(π) is an expectation of squared terms. Further, from Equation (11), it follows imme- diately that L(π∗) = 0, and so we deduce that π∗ is a global optimum for L. We now show that there are no other local/global minima for L in Π.

We write J = Supp(µ). We parametrise the set Π via vectors of logits s ∈ RJ, setting πs(y) = exp(s(y))/ �

y′∈J exp(s(y′)) for y ∈ J, and πs(y) = 0 otherwise. Let us write L(s) = L(πs) for the objective as a function of the logits s.

L(s) = Ey,y′∼µ

��p∗(y ≻ µ) − p∗(y′ ≻ µ)

τ (14)

− (s(y) − s(y′)) − log �πref(y′)

πref(y)

� �2� .

The objective is quadratic as a function of the logits s. Further, by expanding the quadratic above, we see that the loss can be expressed as a sum of squares �

y,y′∈J µ(y)µ(y′)(s(y) − s(y′))2 (15)

---

Understand Learning from Human Preferences

plus linear and constant terms. This is therefore a positive-semidefinite quadratic, and hence is convex. We thus deduce that all local minimisers of the loss L(s) are global minimisers as well (Boyd and Van- denberghe, 2004, Chap. 4). We now notice since πs is a surjective continuous mapping from s to π one can easily show from the definition of local minimum that every local minimiser π of L corresponds to a set of local minimisers Sπ of L. Thus all local minimums of L are also global minimums as well.

Finally, the only direction s in which the quadratic in Equation (15) does not increase away from 0 is when all bracketed terms remain 0; that is, in the direction (1, . . . , 1) ∈ RJ. Thus, L(s) is strictly convex, except in the direction (1, . . . , 1). (Boyd and Vandenberghe, 2004, Chap. 3). However, modifying logits in the di- rection e = (1, . . . , 1) does not modify the resulting policy πs, since, for y ∈ J,

πs+λe(y) = es(y)+λ

�

y′∈J es(y′)+λ = es(y)

�

y′∈J es(y′) = πs(y) .

The strict convexity combined with the fact that π∗ is a global minima proves that π∗ is the unique global/local minima in Π (Boyd and Vandenberghe, 2004, Chap. 4).

5.2 Sampled Loss for IPO

In order to obtain the sampled loss for IPO we need to show that we can build an unbiased estimate of the right-hand side of the equation (13). To this end, we consider the Population IPO Loss:

E y,y′∼µ

�� hπ(y, y′) − τ −1I(y, y′) �2� , (16)

where I(y, y′) is drawn from a Bernoulli distribution with mean p∗(y ≻ y′), i.e., I(y, y′) is 1 if y is pre- ferred to y′ (which happens with probability p∗(y ≻ y′)), and 0 otherwise. This straightforwardly yields a sample-based loss that can be used, by sampling a pair (y, y′) from the preference dataset, and consult- ing the recorded preference to obtain a sample from I(y, y′). The following proposition justifies the switch from Equation (13) to Equation (16), by demonstrat- ing their equality.

Proposition 3. The expressions in Equation (13) and Equation (16) are equal, up to an additive constant independent of π.

Proof. This equivalence is not completely trivial, since in general the conditional expectation

E[hπ(Y, Y ′) − τ −1I(Y, Y ′) | Y = y, Y ′ = y′]

is not equal to the corresponding quantity appearing in Equation (13), namely

hπ(y, y′) − τ −1� p∗(y ≻ µ) − p∗(y′ ≻ µ) � .

We instead need to exploit some symmetry between the distributions of y and y′, and use the fact that hπ(y, y′) decomposes as an additive function of y and y′. To show this equality of losses, it is enough to focus on the “cross-terms” obtained when expanding the quadratics in Equations (13) and (16); that is, to show

E y,y′∼µ

� hπ(y, y′)I(y, y′) �

= E y,y′∼µ

� hπ(y, y′)(p∗(y ≻ µ) − p∗(y′ ≻ µ)) � .

Now, starting with the right-hand side, and using the shorthand πy = log(π(y)), πR y = log(πref(y)), py = p∗(y ≻ µ), and similarly for y′, we have

E y,y′∼µ

� hπ(y, y′)(p∗(y ≻ µ) − p∗(y′ ≻ µ)) �

= E y,y′∼µ

� (πy − πy′ + πR y′ − πR y )(py − py′) �

= E y,y′∼µ

� πypy − πypy′ − πy′py + πy′

+ py′ + πR y′py − πR y′py′ − πR y py + πR y py′ �

= E y,y′∼µ

� (2py − 1)πy − (2py − 1)πR y � ,

where we have used iid-ness of y and y′, and Ey∼µ[py] = 1/2. Turning to the left-hand side, we have

E y,y′∼µ

� hπ(y, y′)I(y, y′) �

= E y,y′∼µ

�� πy − πy′ + πR y′ − πR y � I(y, y′) �

= E y∼µ

�� πy − πR y � E y′∼µ[I(y, y′) | y] �

+ E y′∼µ

�� − πy′ + πR y′ � E y∼µ[I(y, y′) | y′] �

= E y,y′∼µ

� πypy − πy′(1 − py′) + πR y′(1 − py′) − πR y py �

= E y,y′∼µ

� (2py − 1)πy − (2py − 1)πR y � ,

where we use the fact that Ey′∼µ I(y, y′) = py and Ey∼µ I(y, y′) = 1 − py′. This demonstrates equality of the losses, as required.

We now discuss how to approximate the loss in Equa- tion (16) with an empirical dataset. As in our ear- lier discussion, the empirical dataset D takes the form

---

Gheshlaghi Azar, Rowland, Piot, Guo, Calandriello, Valko, Munos

(yw,i, yl,i)N i=i. Note that each datapoint (yw,i, yl,i) con- tributes two terms to an empirical approximation of Equation (16), with (y, y′, I(y, y′)) = (yw,i, yl,i, 1), and also (y, y′, I(y, y′)) = (yl,i, yw,i, 0). This symmetry is important to exploit, and leads to a reduction in the variance of the loss. The overall empirical loss is there- fore given by

1 2 E (yw,yl)∼D

� (hπ(yw, yl) − τ −1)2 + hπ(yl, yw)2� =

1 2 E (yw,yl)∼D

� (hπ(yw, yl) − τ −1)2 + hπ(yw, yl)2� ,

which up to a constant equals:

E (yw,yl)∼D

�� hπ(yw, yl) − τ −1

2

�2�

. (17)

This simplified form of the loss provides some valu- able insights on the way in which IPO optimizes the policy π: IPO learns from preferences dataset simply by regressing the gap between log-likelihood ratios log(π(yw)/π(yl)) and log(πref(yw)/πref(yl)) to τ −1

2 . So the weaker the regularisation becomes, the higher would be the log-likelihood ratio of yw to yl. In other words IPO, unlike DPO, always regularizes its solution towards πref by controlling the gap be- tween the log-likelihood ratios log(π(yw)/π(yl)) and log(πref(yw)/πref(yl)), thus avoiding the over-fitting to the preference dataset. We summarize the sampled IPO in Algorithm 1:

Algorithm 1 Sampled IPO Require: Dataset D of prompts, preferred and dis- preferred generations x, yw and yl, respectively. A reference policy πref

1: Define

hπ(y, y′, x) = log �π(y|x)πref(y′|x)

π(y′|x)πref(y|x)

�

2: Starting from π = πref minimize

E (yw,yl,x)∼D

� hπ(yw, yl, x) − τ −1

2

�2 .

5.3 Illustrative Examples

To illustrate the qualitative difference between our al- gorithm and DPO we will consider a few simple cases. For simplicity we assume there is no context x, i.e., we are in the bandit setting.

5.3.1 Asymptotic Setting

We first consider the simple case where we have 2 ac- tions only, y1 and y2, and a deterministic preference between them: p∗(y1 ≻ y2) = 1. Suppose we start with a uniform πref and µ. We know from Section

4.2 that DPO will converge to the deterministic policy π∗(y1) = 1, π∗(y2) = 0 regardless of the value of τ. Thus even when the regularisation coefficient τ is very large, this is very different from the uniform πref.

Now, let us derive the optimal policy for IPO. We have p∗(y1 ≻ µ) = 3/4 and p∗(y2 ≻ µ) = 1/4. Plug- ging this into equation (9) with Ψ = I we get that π∗(y1) = exp(0.75τ −1)

exp(0.75τ −1)+exp(0.25τ −1) = σ(0.5τ −1), and π∗(y2) = σ(−0.5τ −1), where σ is the sigmoid func- tion. Hence we see that if we have large regularisation as τ → +∞, then π∗ converges to the uniform policy πref, and on the flip side as τ → +0, then π∗(y1) → 1 and π∗(y2) → 0, which is the deterministic optimal policy. The regularisation parameter τ can now actu- ally be used to control how close to πref we are.

5.4 Sampled Preferences

So far we relied on the closed-form optimal policy from Eq. (9) to study DPO and IPO’s stability, but this equa- tion is not applicable to more complex settings where we only have access to sampled preference instead of p⋆. We can still however find accurate approximations of the optimal policy by choosing a parametrisation πθ and optimize θ with an empirical loss over a dataset and iterative gradient-based updates. We will use this approach to show two non-asymptotic examples where DPO over-fits the dataset of preferences and ignore πref: when one action y wins against all others DPO pushes πθ(y) to 1 regardless of τ, and conversely when one ac- tion y never wins against the others DPO pushes πθ(y) to 0 again regardless of τ. In the same scenarios, IPO does not converge to these degenerate solutions but instead remains close to πref based on the strength of the regularisation τ.

For both scenarios we consider a discrete space Y = {ya, yb, yc} with 3 actions, and select a dataset of pairs D = {(yw,i, yl,j)}. Given D, we leverage the empiri- cal losses from Eq. 4 and Eq. 13 to find DPO’s and IPO’s optimal policy. We encode policies as πθ(yi) = softmax(θ)i using a vector θ ∈ R3, and optimize them for 18000 steps using Adam (Kingma and Ba, 2014) with learning rate 0.01 and mini-batch size 9. Mini- batches are constructed using uniform sampling with replacement from D. Both policies and losses are im- plemented using the flax python framework (Brad- bury et al., 2018; Heek et al., 2023), and the Adam im- plementation is from optax (Babuschkin et al., 2020).

---

Understand Learning from Human Preferences

Figure 1: Comparison Between the Learning Curves of Action Probabilities of IPO and DPO for D1

For each set of hyper-parameters we repeat the exper- iment 10 times with different seeds, and report mean and 95% confidence intervals. All experiments are ex- ecuted on a modern cloud virtual machine with 4 cores and 32GB of ram.

IPO Avoids Greedy Policies For the first exam- ple we sample each unique action pair once to collect a dataset D containing 3 observed preferences. Due to symmetries of pairwise preferences sampling only 3 preferences can results in only two outcomes (up to permutations of the actions):

D1 = {(ya, yb), (yb, yc), (ya, yc)},

D2 = {(ya, yb), (yb, yc), (yc, ya)},

where we focus on D1, which represent a total ordering, rather than D2, which represent a cycle. The outcome of the experiment is reported in Fig. 1 in which, we report the learning curves for varying values of τ. We observe that DPO always converges to the deterministic policy for all values of τ. In other word DPO completely ignores the reference policy, no matter how strong is the regularisation term, and converges to the action which is preferred in the dataset. On the other hand, IPO prevent the policy from becoming greedy when the regularisation is strong.

IPO Does not Exclude Actions In the first ex- ample DPO converges to a deterministic policy because one action strictly dominates all others and the loss continues to push up its likelihood until it saturates. The opposite effect happens for the logical opposite

Figure 2: Comparison Between the Learning Curves of Action Probabilities of IPO and DPO for D3

condition, i.e., when one action does not have at least a victory in the dataset DPO will sets its probability to 0 regardless of τ. While this is less disruptive than the first example (a single probability is perturbed whereas previously the whole policy was warped by an over-achieving action) it is also much more com- mon in real-world data. In particular, whenever the action space is large but the dataset small, some ac- tions will necessarily be sampled rarely or only once, making it likely to never observe a victory. Especially because we do not have data on their performance π should stick close to πref for safety, but DPO’s objective does not promote this.

In the final example the dataset consists of two ob- served preferences D3 = {(ya, yb), (yb, ya)} and leave the pair (ya, yc) completely unobserved. We compute solutions using Adam once again, and report the re- sults in Fig. 2 for varying values of τ. We observe again here that DPO ignores the prior πref completely, no mat- ter how strong we regularize the objective, whereas IPO gradually decreases the probability of unobserved action with τ.

6 Conclusion and Future Work

We presented a unified objective, called ΨPO, for learn- ing from preferences. It unifies RLHF and DPO methods. In addition, we introduced a particular case of ΨPO, called IPO, that allows to learn directly from prefer- ences without a reward modelling stage and without relying on the Bradley-Terry modelisation assumption

---

Gheshlaghi Azar, Rowland, Piot, Guo, Calandriello, Valko, Munos

that assumes that pairwise preferences can be substi- tuted with pointwise rewards. This is important be- cause it allows to avoid the overfitting problem. This theoretical contribution is only useful in practice if an empirical sampled loss function can be derived. This is what we have done in Sec 5 where we show that IPO can be formulated as a root-finding problem from which an empirical sampled loss function can be de- rived. The IPO loss function is simple, easy to im- plement and theoretically justified. Finally, in Sec. 5.3 and Sec. 5.4, we provide illustrative examples where we highlight the instabilities of DPO when the preferences are fully-known as well as when they are sampled. Those minimal experiments are sufficient to prove that IPO is better suited to learn from sampled preferences than DPO. Future works should scale those experiments to more complex settings such as training language models on human preferences data.

---

Understand Learning from Human Preferences

References

Igor Babuschkin, Kate Baumli, Alison Bell, Surya Bhupatiraju, Jake Bruce, Peter Buchlovsky, David Budden, Trevor Cai, Aidan Clark, Ivo Danihelka, et al. The DeepMind JAX ecosystem, 2020, 2020. URL http://github.com/deepmind.

Quentin Bertrand, Wojciech Marian Czarnecki, and Gauthier Gidel. On the limitations of the Elo: Real- world games are transitive, not additive. In Proceed- ings of the International Conference on Artificial In- telligence and Statistics, 2023.

Stephen P. Boyd and Lieven Vandenberghe. Convex optimization. Cambridge University Press, 2004.

James Bradbury, Roy Frostig, Peter Hawkins, Matthew James Johnson, Chris Leary, Dougal Maclaurin, George Necula, Adam Paszke, Jake VanderPlas, Skye Wanderman-Milne, and Qiao Zhang. JAX: composable transformations of Python+NumPy programs, 2018. URL http:// github.com/google/jax.

Ralph Allan Bradley and Milton E Terry. Rank anal- ysis of incomplete block designs: I. The method of paired comparisons. Biometrika, 39(3/4):324–345, 1952.

R´obert Busa-Fekete, Bal´azs Sz¨or´enyi, Paul Weng, Wei- wei Cheng, and Eyke H¨ullermeier. Preference-based reinforcement learning: Evolutionary direct policy search using a preference-based racing algorithm. Machine Learning, (3):327–351, 2014.

R´obert Busa-Fekete, Bal´azs Sz¨orenyi, Paul Weng, Wei- wei Cheng, and Eyke H¨ullermeier. Preference-based evolutionary direct policy search. In Autonomous Learning Workshop @ ICRA, 2013.

Xiaoyu Chen, Han Zhong, Zhuoran Yang, Zhaoran Wang, and Liwei Wang. Human-in-the-loop: Prov- ably efficient preference-based reinforcement learn- ing with general function approximation. In Pro- ceedings of the International Conference on Machine Learning, 2022.

Paul F. Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep rein- forcement learning from human preferences. In Ad- vances in Neural Information Processing Systems, 2017.

Matthieu Geist, Bruno Scherrer, and Olivier Pietquin. A theory of regularized Markov decision processes. In Proceedings of the International Conference on Machine Learning, 2019.

Jonathan Heek, Anselm Levskaya, Avital Oliver, Mar- vin Ritter, Bertrand Rondepierre, Andreas Steiner, and Marc van Zee. Flax: A neural network li-

brary and ecosystem for JAX, 2023. URL http: //github.com/google/flax.

Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In Proceedings of the International Conference on Learning Representa- tions, 2014.

Angeliki Lazaridou, Anna Potapenko, and Olivier Tieleman. Multi-agent communication meets nat- ural language: Synergies between functional and structural language learning. In Proceedings of the Annual Meeting of Association for Computational Linguistics, 2020.

Tyler Lu, D´avid P´al, and Martin P´al. Contextual multi-armed bandits. In Proceedings of the Inter- national Conference on Artificial Intelligence and Statistics, 2010.

Yuchen Lu, Soumye Singhal, Florian Strub, Aaron Courville, and Olivier Pietquin. Countering lan- guage drift with seeded iterated learning. In Pro- ceedings of the International Conference on Machine Learning, 2020.

Ellen Novoseller, Yibing Wei, Yanan Sui, Yisong Yue, and Joel Burdick. Dueling posterior sampling for preference-based reinforcement learning. In Proceed- ings of the Conference on Uncertainty in Artificial Intelligence, 2020.

OpenAI. Gpt-4 technical report, 2023.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller amd Maddie Simens, Amanda Askell, Peter Welin- der, Paul Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback. In Advances in Neural Information Processing Systems, 2022.

Aldo Pacchiano, Aadirupa Saha, and Jonathan Lee. Dueling RL: Reinforcement learning with trajectory preferences. arXiv, 2023.

Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. Improving language understand- ing by generative pre-training. 2018.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Ste- fano Ermon, Christopher D. Manning, and Chelsea Finn. Direct preference optimization: Your lan- guage model is secretly a reward model. arXiv, 2023.

Prajit Ramachandran, Peter J. Liu, and Quoc V. Le. Unsupervised pretraining for sequence to sequence learning. In Proceedings of the Conference on Em- pirical Methods in Natural Language Processings, 2016.

---

Gheshlaghi Azar, Rowland, Piot, Guo, Calandriello, Valko, Munos

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy opti- mization algorithms. arXiv, 2017.

Nisan Stiennon, Long Ouyang, Jeffrey Wu, Daniel Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul F. Christiano. Learning to summarize with human feedback. Advances in Neural Information Processing Systems, 2020.

Yuanhao Wang, Qinghua Liu, and Chi Jin. Is RLHF more difficult than standard RL? arXiv, 2023.

Jason Wei, Maarten Bosma, Vincent Y. Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, An- drew M. Dai, and Quoc V. Le. Finetuned language models are zero-shot learners. In Proceedings of the International Conference on Learning Representa- tions, 2022.

Yao Zhao, Rishabh Joshi, Tianqi Liu, Misha Khalman, Mohammad Saleh, and Peter J Liu. SLiC-HF: Se- quence likelihood calibration with human feedback. arXiv, 2023.

---

Understand Learning from Human Preferences

## APPENDICES

A Proofs

A.1 Existence and uniqueness of the regularized argmaximum

For completeness, we briefly recall the proof of existence and uniqueness of the argmaximum of the following regularized criterion that can also be found in the work of Rafailov et al. (2023):

Lτ(δ) = Es∈δ[f(s)] − τKL(δ || η),

= �

s∈S δ(s)f(s) − τKL(δ || η),

where S is a finite set, f ∈ RS a function mapping elements of S to real numbers, τ ∈ R∗ + a strictly positive real number, δ ∈ ∆S and η ∈ ∆S are discrete probability distributions over S. In particular, we recall that a discrete probability distribution δ ∈ ∆S can be identified as a positive real function δ ∈ RS + verifying: �

s∈S δ(s) = 1.

Now, if we define the softmax probability δ∗ ∈ ∆S as:

∀s ∈ S, δ∗(s) = η(s) exp(τ −1f(s)) �

s′∈S η(s′) exp(τ −1f(s′)),

then, under the previous definitions, we have the following result:

δ∗ = arg max δ∈∆S Lτ(δ)

Proof.

Lτ(δ)

τ = �

s∈S δ(s)f(s)

τ − KL(δ || η),

= �

s∈S δ(s)f(s)

τ − �

s∈S δ(s) log �δ(s)

η(s) � ,

= �

s∈S δ(s) �f(s)

τ − log �δ(s)

η(s) �� ,

= �

s∈S δ(s) � log � exp(τ −1f(s)) � − log �δ(s)

η(s) �� ,

= �

s∈S δ(s) � log �η(s) exp(τ −1f(s))

δ(s) �� ,

= �

s∈S δ(s) � log �η(s) exp(τ −1f(s))

�

s′∈S η(s′) exp(τ −1f(s′)) �

s′∈S η(s′) exp(τ −1f(s′))

δ(s) �� ,

= �

s∈S δ(s) � log � η(s) exp(τ −1f(s)) �

s′∈S η(s′) exp(τ −1f(s′))

δ(s) �� + �

s∈S δ(s) log � �

s′∈S η(s′) exp(τ −1f(s′)) � ,

= �

s∈S δ(s) � log �δ∗(s)

δ(s) �� + log � �

s′∈S η(s′) exp(τ −1f(s′)) � ,

= −KL(δ || δ∗) + log � �

s′∈S η(s′) exp(τ −1f(s′)) � .

---

Gheshlaghi Azar, Rowland, Piot, Guo, Calandriello, Valko, Munos

By definition of the KL, we now that δ∗ = arg maxδ∈∆S

� − KL(δ || δ∗) � and as:

−KL(δ || δ∗) = Lτ(δ)

τ − log � �

s′∈S η(s′) exp(τ −1f(s′)) �

where log � �

s′∈S η(s′) exp(τ −1f(s′)) � is a constant (does not depend on δ) and τ a positive multiplicative term, then −KL(δ || δ∗) and Lτ(δ) share the same argmaximum. This concludes the proof.

A.2 Non-uniqueness when Supp(π(·)) ̸= Supp(µ):

Notice that if we search for a solution where the support of π is strictly larger than that of µ then there could be multiple solutions. Let us illustrate this case with a simple example. Consider a single state x and 3 actions y1, y2, y3. The reference policy πref is uniform over {y1, y2, y3} and the policy µ assigns a probability 1/2 to both y1 and y2 and 0 probability to y3.

Thus the loss is L(π) = 2 � τ −1� p∗(y1 ≻ µ) − p∗(y2 ≻ µ) � − log π(y1)

π(y2) �2 . We deduce that any policy π =

(p, q, 1 − p − q) such that p

q = eτ −1(p∗(y1≻µ)−p∗(y2≻µ)) is a global minimum of L(π).

In particular there are an infinity of solutions different from the optimal solution π∗. The problem comes from the fact that when the support of µ does not cover the whole action space there are not enough constraints to uniquely characterize π∗. Assuming that the supports of πref and µ coincide enables us to recover uniqueness of the solution, as proven in Theorem 2.

B Additional results

In this section, we show the equivalence of DPO and RLHF, regardless of whether the preference model p∗ corre- sponds to a Bradley-Terry model. Note that the assumption of the existence of a minimizer is to exclude cases where the loss is minimized by taking the rewards of certain actions to +/ − ∞.

Proposition 4. Consider a preference model p∗ such that there exists a minimizer to the Bradley-Terry loss

arg min r − E x∼ρ y∼µ(·|x) y′∼µ(·|x)

[p∗(y ≻ y′|x) log σ(r(x, y) − r(x, y′))] .

Then, the optimal policy for the DPO objective in Equation (4) and for the RLHF objective in Equation (3) with reward model given as the minimizer to the Bradley-Terry loss above are identical, regardless of whether or not p∗ corresponds to a Bradley-Terry preference model.

Proof. Recall that the optimal policy π∗ r for a given reward function r for the objective in Equation (3) is given by π∗ r(y|x) ∝ πref(y|x) exp(τ −1r(x, y)). It therefore follows that

− E x∼ρ y,y′∼µ(·|x) [p(y ≻ y′|x) log σ(r(x, y) − r(x, y′))]

= − E x∼ρ y,y′∼µ(·|x)

� p(y ≻ y′|x) log σ � τ log � π∗ r(y|x)

π∗r(y′|x)

� − τ log � πref(y|x)

πref(y′|x)

��� .

In words, the value of the Bradley-Terry reward objective for r is the value of the DPO objective for π∗ r. We recall also that the map r �→ π∗ r is surjective.

Now, suppose r is optimal for the Bradley-Terry reward objective, meaning that π∗ r is optimal for the RLHF objective. If π∗ r is not optimal for the DPO objective, then there exists another policy π′ that obtains a strictly lower value for the DPO loss. But then there exists a reward function r′ such that π′ = π∗ r′, such as r′(x, y) = τ log(π′(y|x)/πref(y|x)), and this r′ therefore obtains a lower Bradley-Terry loss than r, a contradiction.

Similarly, if π∗ is optimal for the DPO objective, the corresponding reward function r(x, y) = τ log(π∗(y|x)/πref(y|x)) must be optimal for the Bradley-Terry reward loss. The corresponding optimizer for the RLHF objective is then given by π(y|x) ∝ πref(y|x) exp(τ −1τ log(π∗(y|x)/πref(y|x))) = π∗(y|x), as required.

---
