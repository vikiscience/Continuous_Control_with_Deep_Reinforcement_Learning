# Problem Setting

The problem of moving a double-jointed arm to reach for moving object locations can be formulated as a Reinforcement Learning (**RL**) problem, where an Agent learns through interaction with Environment to achieve their goal - maintaining the arm's position at the object location for as many time steps as possible. The Agent can observe the state which the Environment is in, and takes actions that affect this state. In turn, the Environment gives the Agent feedback ("rewards") based on the actions.

This setting can be formulated as Markov Decision Process (**MDP**), where:

* Action space `A`, which is continuous with 4 dimensions (corresponding to torque applicable to two joints)
* State space `S`, which is defined to be 33-dimensional (eg. position, rotation, velocity, and angular velocities of the two arm Rigidbodies)
* The transition to the next state `s_{t+1}` and the resulting reward `r_{t+1}` are defined by the Environment and depend only on the current state `s_t` and the Agent's chosen action `a_t` ("one-step dynamics")
* Discount rate `gamma`, which is used by the Agent to prioritize current rewards over future rewards.

The Agent's goal is to find the optimal _policy_ `pi: S -> A` that maximizes the expected discounted return `J` (weighted sum of all future rewards per episode). Thus, the optimal policy is a function that gives the best possible action for each Environment state. In order to achieve this goal, the Agent can utilize the estimated values of `(s, a)` pairs by learning the so-called _action-value function_ `Q(s, a)`.

In our case, both state and action space are continuous and high-dimensional, which means that Deep Neural Networks (**DNN**s) can be used to represent the action-value function `Q` as well as policy `pi`. The dependency of both functions on the network weights is denoted by `Q_θ` and `pi_φ`, where `θ` and `φ` are the respective DNN weights.

Well-established RL algorithms for discrete action spaces are eg. Deep Q-Networks (**DQN**) and Double Deep Q-Networks (**Double DQN**). Both are based on a Q-Learning algorithm, which updates the Q-values iteratively as follows:

`Q_new(s_t, a_t) := Q(s_t, a_t) + alpha * (r_{t+1} + gamma * max(Q(s_{t+1}, a)) - Q(s_t, a_t))`

In other words, `<Q_new> = <Q_old> + alpha * (<target> - <Q_old>)`

In Q-Learning, the Q-values are utilized by the Agent to choose his actions with "epsilon-greedy policy", where for each state the best action according to `Q` is chosen with probability `(1 - epsilon)`, and a random action - with probability `epsilon`. Traditionally, `epsilon` is decayed after each episode during training.


# Learning Algorithm

Generally, RL algorithms can be divided into 3 categories:

| Category        | Description     | Examples  |
| :-------------: | :-------------: | :-----: |
| Value-Based (_Critic-Only_) | Use a DNN to learn action-value function `Q` and deriving epsilon-greedy policy `pi` based on `Q` | [DQN](https://www.nature.com/articles/nature14236), [Double DQN](https://arxiv.org/abs/1509.06461) |
| Policy-Based (_Actor-Only_) | Use a DNN to directly learn policy `pi` | [PPO](https://arxiv.org/abs/1707.06347), [TRPO](https://arxiv.org/abs/1502.05477) |
| _Actor-Critic_ |  Use `Q` as a baseline for learning `pi` | [DDPG](https://arxiv.org/abs/1509.02971), [ACKTR](https://arxiv.org/abs/1708.05144), [SAC](https://arxiv.org/abs/1801.01290) |


Actor-Critic methods combine the advantages of the other two categories of methods. In this case, an **Actor** represents the learned policy `pi` and is used to select actions, while the **Critic** represents the action-value function `Q` and evaluates `pi` by gathering experience and learning action values. Actor's training is then based on the learned action values.

In this project, an Actor-Critic method called "Twin Delayed Deep Deterministic policy gradient algorithm" (**TD3**) was used (see this [paper](https://arxiv.org/abs/1802.09477)). It was shown to outperform such methods as PPO, TRPO, DDPG, ACKTR and SAC on MuJoCo suite of continuous control problems in [OpenAI gym](https://github.com/openai/gym).


## TD3

TD3 algorithm addresses well-known problems of overestimation bias and high variance.

_Overestimation bias_ is a natural result of function approximation errors in Q-Learning, where the maximization of a noisy value estimate induces a consistent overestimation. The effect of these errors is further accumulated, because Q-values for the current state are updated using the Q-value estimate of a subsequent state (see formula above). This accumulated error causes _high variance_, where any bad states can be estimated with too high Q-value, resulting in suboptimal policy updates and even divergence.

While it is an established fact that methods applied to discrete action spaces are susceptible to overestimation bias, it was shown that the problem occurs likewise in continuous action spaces (with Actor-Critic methods).

In order to solve these two major problems, TD3 algorithm builds upon DDPG and introduces and puts together different techniques:

1) TD3 utilizes the clipped variant of Double Q-learning to reduce overestimation bias.

2) It is suggested to use target networks, a common approach in DQN methods, which turns out to be critical for variance reduction by reducing the accumulation of errors. 

3) Policy updates are delayed until the value estimate has converged. This technique couples value function `Q` and policy `pi` more effectively.

4) The authors introduce a novel regularization strategy, where a SARSA-style update bootstraps similar action estimates to further reduce variance. 


#### Experience replay buffer

It is a notable fact that training DNNs in RL settings is instable due to correlations in sequences of Environment observations. Traditionally, this instability is overcome by letting the Agent re-learn from its long-passed experience.

Hence, the Agent maintains a replay buffer of capacity `M` where he stores his previous experience in form of tuples `(s_t, a_t, r_{t+1}, s_{t+1})`. Every now and then, the Agent samples a mini-batch of tuples randomly from the buffer and uses these to update `Q`. Thus, the sequence correlation gets eliminated, and the learned policy is more robust.


#### Target networks

To solve a problem of "moving target" (correlations between action-values and target values), we don't change the DNN weights during training step, because they are used for estimating next best action. We achieve this by maintaining two DNNs - one is used for training, the other ("_target network_") is fixed and only updated with current weights after each `d` steps. 

Thus, target networks are frozen copies of Actor and Critic network `Q` and `pi`, correspondingly. These are used for estimating the target value as follows:

`<target> = r_{t+1} + gamma * Q_target(s_{t+1}, a_{t+1})`, where `a_{t+1} = pi_target(s_{t+1})`

The weights `θ'` of a target network `Q_target` are either updated periodically to exactly match the weights `θ` of the current network `Q`, or by some proportion `τ` at each time step: 

`θ' := τ * θ + (1 − τ) * θ'`

This update can be applied while sampling random mini-batches of transitions from an experience replay buffer.


#### Clipped Double Q-Learning

In Actor-Critic methods, the current and target networks may in practice be too similar to make an independent estimation, and so offer little improvement. Instead, the original Double Q-learning formulation can be used, with a pair of Actors `(pi_φ1, pi_φ2)` and Critics `(Q_θ1, Q_θ2)`, where `pi_φ1` is optimized with respect to `Q_θ1` and `pi_φ2` with respect to `Q_θ2`. At the same time, the target update of `Q_θ1` and `Q_θ2` are done with an independent estimate of the corresponding "opposite" target networks `Q_target2` and `Q_target1`:

`<target_1> = r_{t+1} + gamma * Q_target2(s_{t+1}, pi_φ1(s_{t+1}))`

`<target_2> = r_{t+1} + gamma * Q_target1(s_{t+1}, pi_φ2(s_{t+1}))`

However, the Critics are not entirely independent, due to the use of the opposite Critic in the learning targets as well as the same replay buffer. Consequently, opposite Critic can give even higher Q-value than the already overestimated one. To avoid further exaggerated overestimation, only one Actor `pi_φ` and one (clipped) target value for both Critics should be considered:

`<target> = r_{t+1} + gamma * min [ Q_target1(s_{t+1}, pi_φ(s_{t+1})); Q_target2(s_{t+1}, pi_φ(s_{t+1})) ]`

The single Actor is optimized with respect to `Q_θ1`.


#### Delayed Policy Updates

Because policy updates on high-error states lead to divergent behavior, the Actor `pi_φ` should be updated at a lower frequency than the Critic (eg. each `d` iterations), so that the Q-value error is minimized before the policy update. 

At the same time, each `d` iterations, both target Actor and the target Critics `pi_target`, `Q_target1` and `Q_target2` are updated using the same proportion `τ` of their current networks.


#### Target Policy Smoothing Regularization

Deterministic policies are known to overfit to narrow peaks in the value estimate, which increases the variance of the target in Critic updates. We can apply regularization to the target update according to the notion that similar actions should have similar values. Adding noise to the target policy `pi_target` and using its average over a mini-batch instead of the current policy in order to find target action `a'` was shown to smooth the value estimate.


#### Algorithm

1. Initialize: `N` - number of training episodes, replay memory with capacity `M`, mini-batch size `B`, `gamma` - discount rate; policy and target update frequency `d`, `τ` - proportion of current weights in a target weight update; `σ` - standard deviation of the explorative gaussian noise for action selection; `σ'` and `c` - standard deviation and clipping boundary of the gaussian noise for target action selection

2. Initialize at random: Actor `pi_φ`, Critics `Q_θ1` and `Q_θ2`; create their copies: target Actor `pi_target` and target Critics `Q_target1` and `Q_target2`

3. For each episode out of `N`:
   
   3.1. `t := 0`

   3.2. While not done:

      3.2.1. Observe `s_t`

      3.2.2. Choose `a_t` using current policy: `a_t ~ pi_φ(s_t) + noise`, where `noise ~ N(0, σ)`
   
      3.2.3. Take action `a_t`, observe reward `r_{t+1}` and next state `s_{t+1}` of the Environment 
   
      3.2.4. Store tuple `(s_t, a_t, r_{t+1}, s_{t+1}, done_{t+1})` in replay memory, where `done_{t+1} = 1` if the episode ended at timestep `t+1`, else `0`
   
      3.2.5. Sample random mini-batch of size `B` from memory
      
      3.2.6. Compute target action `a' = pi_target(s_{t+1}) + noise`, where `noise ~ clip(N(0, σ'), -c, c)`
      
      3.2.7. `<target> = r_{t+1} + gamma * (1 - done_{t+1}) * min [ Q_target1(s_{t+1}, a'); Q_target2(s_{t+1}, a') ]`
   
      3.2.6. Perform gradient descent on both current Critics' weights w.r.t. `<target>` with MSE as loss function
   
      3.2.7. Every `d` steps: update current Critic `pi_φ` by the deterministic policy gradient w.r.t. `Q_θ1` and update all target networks:
      
      `θ_target1 := τ * θ1 + (1 − τ) * θ_target1`
      
      `θ_target2 := τ * θ2 + (1 − τ) * θ_target2`
      
      `φ_target := τ * φ + (1 − τ) * φ_target`
   
      3.2.8. `t := t + 1`


## Implementation

The interaction between the Agent and the Environment is implemented in `algo.py`. The Agent's internal logic is placed in `agent.py`, including action selection according to a current policy, using replay buffer and a call to train the policy.

`models.py` contains the Actor and Critic DNNs and their interaction implemented for TD3 (with minor changes of the [source](https://github.com/sfujim/TD3)). 

The Actor network architecture is sequential with 3 linear neuron layers and RELU as an activation function. State vector is fed directly to the input layer, and the last layer's activation function is `tanh`. While the number of neurons in the first hidden layer is configurable (see variable `num_fc_actor` in `const.py`), the second hidden layer has the same width, and the output layer has the dimensionality of the action space.

Each Critic network receives the concatenated state and action vector as input. Both Critic networks have the same architecture as Actor, except that the width of hidden layers is configured by `num_fc_actor` in `const.py`, and the output layer has 1 neuron without activation function.

Finally, Grid Search is implemented in `hyperparameter_search.py` in order to select the best Agent solving the given Environment. The script also documents what hyperparameter values were tested so far. Best resulting hyperparameters are already listed in `const.py`. Such hyperparameters from `models.py` as `d = 2`, `τ = 0.005`, σ`' = 0.2` and `c = 0.5` were kept as they were in the source code.


# Hyperparameter optimization

As mentioned above, the current best hyperparameters of the algorithm found by the Grid Search are the following:

`N = 250`

`M = 40000`

`gamma = 0.95`

`B = 32`

`σ = 0.3`

`model_learning_rate = 0.001`

`num_fc_actor = 128`

`num_fc_critic = 128`


# Future Work

Other Actor-Critic RL algorithms can be implemented such as A3C, DDPG or SAC.
