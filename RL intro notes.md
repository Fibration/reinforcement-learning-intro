# Notes for Reinforcement Learning: An Introduction

## ch 1
A reinforcement learning (RL) problem has four parts:
* Policy
* Reward signal
* Value function
* (Optional) Model

# ch 2: k-bandit problem
In RL, training data is used to *evaluate* rather than to *instruct*.

K-armed bandit: k actions A_1,...,A_k with reward `q_*(a) = E[R_t|A_t=a]`

Estimate q_* with Q_t.

Balance exploration and exploitation: epsilon-greedy method, epsilon probability that a random move is selected

**ex 2.1** epsilon=0.5, greedy move is made 0.75 of the time

**ex 2.2** definite: A4, A5; maybe: all


Incremental update rule:
```
New estimate <- Old estimate + step size (target - old estimate)
```

Nonstationary when the rewards change with time, use constant step size *exponential recency weighted average*. Problem: bias of the starting choice lives on far into the future.

Stochastic Approximation theory says that the sum converges when the sequence of coefficients `{alpha_n(a)}_n` obeys

```
sum_n alpha_n(a) = infinity    sum_n alpha_n^2(a) < infinity 
```

*Optimistic Initial Values* is the choice of high initial value functions which forces the agent to prioritise exploration at the start (since unexplored options have the initialised high value). But has limited bearing on nonstationary problems.

**ex 2.7** *exponential recency weighted average without bias*
```beta_n = alpha/o_n```

```o_n = o_{n-1} + alpha(1-o_{n-1})```

Gradient Bandit Algorithm: probability of taking action A is the softmax `exp(H(A)) / prod{a} exp(H(a))` which is the policy `pi(A)`.

At time t, update by 

```
H_t(A) = H_t-1(A) + alpha(R - mean R)(1 - pi(A))
```

and 

```
H_t(a) = H_t-1(a) - alpha(R - mean R)pi(a).
```

It redistributes some probability based on and relative to performance.

## ch 3: Markov Decision Processes

p: SxRxSxA -> [0,1]
p(s',r,s,a) = Pr(state=s', reward=r | state=s, action=a)

**ex 3.12** 
```
v(s) = sum{a} pi(a|s) q(s,a)
```

**ex 3.13** 
```
q(s,a) = sum{s',r} p(s',r|s,a) (r + gamma v(s'))
```

**ex 3.17** 
```
q(s,a) = E[G_t | S_t=s, A_t=a]
        = E[R_t+1 + gamma G_t+1 | S_t=s, A_t=a]
        = sum{s',r} p(s',r|s,a) [r + gamma sum{a'} pi(a'|s') q(s',a')] 

 v(s) = sum{a} pi(a|s) sum{s', r} p(s',r|s,a) (r + gamma v(s'))
```

In the gridworld example, the optimal policy is a dynamic system flow/phase diagram with the edges as a 1D repelling surface and the points A and B as point attractors.

**Are optimal policies solutions of PDEs?!**

"In classical physics, Hamilton’s principal function is
an action-value function; Newtonian dynamics are greedy with respect to this
function (e.g., Goldstein, 1957)."

"The counterpart of the
Bellman optimality equation for continuous time and state problems is known
as the Hamilton–Jacobi–Bellman equation"

Need to look into it.

## Ch 4: Dynamic Programming

**Theorem.** (Policy Improvement Theorem) Let pi and pi' be deterministic policies. If `q{pi}(s, pi'(s)) >= v{pi}(s)` then `v{pi'}(s) >= v{pi}(s)`. Strict inequality implies strict inequality.

**Ex 4.4** In part 3, check that v(pi(s)) > v(old action) in addition to checking that old action != pi(s).

**Algorithm** (Action value policy iteration) 
1. Initialisation: Choose arbitrary q(s,a) in Q, arbitrary pi(s) in A
2. Q-evaluation:
```
Loop:
        d = 0
        for s in S:
                q = Q(s,pi(s))
                Q(s,pi(s)) = sum{s', r} p(s',r|s,pi(s)) (r + gamma Q(s', pi(s')))
                d = max(d, |q - Q(s,pi(s))|)
        until d < epsilon
```
3. Policy improvement:
```
stable = true
for s in S:
        old = pi(s)
        pi(s) = argmax{a} sum{s', r} p(s',r|s,a) (r + gamma Q(s', pi(s')))
        if old != pi(s) then stable = false
if stable then stop and return q,pi else go to 2
```

**Algorithm** (Value iteration)
```
Choose for each s in S, choose arbitrary V(s). Choose precision epsilon > 0.

d = 0
Loop:
        for s in S:
                v = V(s)
                V(s) = max{a} p(s',r|s,a) (r + gamma V(s'))
                d = max(d, |v-V(s))
until d < epsilon

Output pi(s) as any action in argmax{a} p(s',r|s,a) (r + gamma V(s'))
```

## Ch 5: Monte Carlo

### Off-Policy Prediction

In importance sampling, there is a behaviour policy `b` which is stochastic and covers the target policy `pi`, that is, when `pi(a|s)>0`, then `b(a|s)>0`. The returns for the target policy `pi` is calculated by converting the returns `G_t` of the behaviour policy `b` with the importance sampling ratio `rho`.

The importance sampling ratio is
```
rho_{t:T-1} = prod{k,t,T-1} [pi(A_k | S_k)/b(A_k | S_k)
```

In the every-state regime, `T(s)` is a list of all time steps where `s` is visited.
In the first visit regime, `T(s) ` is a list of first visits to `s`.

Ordinary importance sampling is given by
```
V(s) = sum{t in T(s)} rho_{t:T-1} G_t / |T(s)|
```

Weighted importance sampling is given by
```
V(s)  = sum{t in T(s)} rho_{t:T-1} G_t / sum{t in T(s)} rho_{t:T-1}
```

In the first visit regime, ordinary importance sampling is biased and weighted importance sampling has high variance. In the every visit regime, both sampling methods are biased. To me, it seems that weighted importance sampling is better because it preserves the magnitude of the returns.

To implement incremental updates for weighted importance sampling, 
```
V_n+1 = V_n + (W_n/C_n) (G_n - V_n)
C_n+1 = C_n + W_n+1
```

Off-policy prediction algorithm:
```
Input(target policy pi)
Let b be a policy covering pi
Initialise for s in S, a in A, Q(s,a) random and C(s,a)=0
Loop:
        Generate episode for b: S_0,A_0,R_1,...,S_{T-1},A_{T-1},R_T
        G = 0
        W = 1
        For t=T-1,...,0 and W!=0:
                G = gamma G + R_{t+1}
                C(S_t,A_t) += W
                Q(S_t,A_t) = Q(S_t,A_t) + (W/C(S_t,A_t)) (G - Q(S_t,A_t))
                W(S_t,A_t) = W * (pi(S_t,A_t)/b(S_t,A_t))
```
Note that (S_t,A_t) refers to the actual state-action pair, not to the current time step.

Can reduce the variance with discounting-aware and per decision importance sampling. The discounting-aware version is as follows:
```
V(s) = sum{t in T(s)} [(1-gamma)sum{h,t+1,T-1} rho{t:h-1} gamma^{h-t-1} G{t:h} + gamma^{T-t-1} rho{t:T-1} G{t:T}]/
sum{t in T(s)} [(1-gamma)sum{h,t+1,T-1} rho{t:h-1} gamma^{h-t-1} + gamma^{T-t-1} rho{t:T-1}]
```

## Ch 6 Temporal Difference

Monte Carlo: `V(S_t) = V(S_t) + alpha (G_t - V(S_t))`

Temporal Difference: `V(S_t) = V(S_t) + alpha (del R_{t+1} - V(S_t))`

where `del R{t+1} = R_{t+1} + gamma V(S_t+1)` is the incremental reward estimate

MC method minimises the RMS error on the training set. TD method estimates the maximum likelihood.

Tabular TD(0) for estimating Q:
```
Input: policy for evaluation pi
Choose step size alpha
Initialise Q(s,a) arbitrarily, Q(terminal,a) = 0

For each episode:
        Initialise S
        Choose A from pi(A|S)
        For each step:
                take action A, observe S', R
                get A' from pi(A'|S')
                Q(S,A) = Q(S,A) + alpha [R + gamma Q(S',A') - Q(S,A)]
                S = S'
                A = A'
                end if S is terminal
```

**SARSA/ On-policy TD(0)**:
```
Choose step size alpha and epsilon > 0
Initialise Q(s,a) arbitrarily, Q(terminal,.) = 0

For each episode:
        initialise S
        get A as argmax{A,epsilon} Q(S,A) 
        for each step:
                Take action A, observe R,S'
                Choose A' as argmax{A',epsilon} Q(S',A')
                Q(S,A) = Q(S,A) + alpha [R + Q(S',A') - Q(S,A)]
                S = S'
                A = A'
                end if S is terminal
```

**Q-Learning**:
```
Choose step size alpha and epsilon > 0
Initialise Q(s,a) arbitrarily, Q(terminal, .) = 0

For each episode:
        initialise S
        for each step:
                Get action A from Q(S,A) by policy (epsilon greedy)
                take action A, observe R,S'
                Q(S,A) = Q(S,A) + alpha [R + gamma max{a}Q(S',a) - Q(S,A)]
                S = S'
                end if S is terminal
```

Q-learning is considered off-policy because the future value from state `S'` is greedy rather than epsilon greedy and may not be the actual step taken.

Expected Sarsa:
```
R + gamma E[Q(S',A')|S']
= R + gamma sum{a} pi(a|S') Q(S',a)
```

## Ch 9 On-policy prediction with approximation

Gradient Monte-Carlo for approximating v_pi:
```
Input: policy pi
Input: differentiable function v