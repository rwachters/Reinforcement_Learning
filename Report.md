[//]: # (Image References)

[image1]: ./plot.png

# Project 3: Collaboration and Competition
## Learning Algorithm
The learning algorithm used for this project is [Deep Deterministic Policy Gradient (DDPG)](https://arxiv.org/abs/1509.02971). DDPG is known as an Actor-Critic method, and it can be used for continuous action spaces. Just like DQN (from project 1) it uses [Experience Replay](https://paperswithcode.com/method/experience-replay) and a [Target Network](https://towardsdatascience.com/deep-q-network-dqn-ii-b6bf911b6b2c). The Actor learns a deterministic policy function, and the Critic learns a Q value function. They both interact with each other when learning. The Critic uses the deterministic action from the Actor when calculating the Q value. Because the Actor learns a deterministic policy, some noise must be added to the action values, to help with exploration. This algorithm uses a noise decay, so that the noise at the start of the learning process is high and much lower at the end of it.

Two types of neural networks are used in this project, one for the Actor and one for the Critic. They both have two hidden layers with 256 and 128 linear units. The Actor network has 24 inputs, and 2 outputs. That's because each state has 24 parameters and there are 2 action parameters. The Critic has 26 (24 + 2) inputs and only one output, the Q value.

In this project there are two agents, so there is an Actor and a Critic neural network for each agent. Both agents learn independently of each other. The Critic only uses the state that the agent sees and not the global state like in the [MADDPG](https://proceedings.neurips.cc/paper/2017/file/68a9750337a418a86fe06c1991a1d64c-Paper.pdf) algorithm.

The hyperparameters used for this algorithm are:

- `buffer_size=100000` replay buffer size
- `batch_size=1000` minibatch size
- `gamma=0.99` discount factor
- `tau=1e-3` for soft update of the target network parameters
- `lr_actor=1e-4` learning rate of the actor
- `lr_critic=1e-3` learning rate of the critic
- `weight_decay=0.0` L2 weight decay
- `update_every=20` how often to update the networks
- `noise_decay=3e-6` the noise decay used for the action values

## Plot of Rewards
![plot][image1]

The environment was solved in 23746 episodes.

## Ideas for Future Work
The performance of the agent could be improved in several ways:

- [MADDPG](https://proceedings.neurips.cc/paper/2017/file/68a9750337a418a86fe06c1991a1d64c-Paper.pdf)
- [Twin Delayed DDPG](https://spinningup.openai.com/en/latest/algorithms/td3.html)
- [Soft Actor Critic (SAC)](https://spinningup.openai.com/en/latest/algorithms/sac.html)
- [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)

