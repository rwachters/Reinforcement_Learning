from model import Actor, Critic
from pytorch_device import pytorch_device
import torch
import torch.nn.functional as f
import torch.optim as optim
from typing import Tuple, List
import copy


class DDPGAgent:
    """Interacts with and learns from the environment."""

    def __init__(self, actor: Actor, critic: Critic, gamma=0.99, tau=1e-3,
                 lr_actor=1e-4, lr_critic=1e-3, weight_decay=1e-2):
        """Initialize a DDPG Agent object.

            :param actor:
            :param critic:
            :param gamma: discount factor
            :param tau: for soft update of target parameters
            :param lr_actor: learning rate of the actor
            :param lr_critic: learning rate of the critic
            :param weight_decay: L2 weight decay
        """
        self.action_size = actor.action_size
        self.gamma = gamma
        self.tau = tau

        # Actor Network (w/ Target Network)
        self.actor = actor.to(pytorch_device)
        self.actor_target = copy.deepcopy(actor).to(pytorch_device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        # Critic Network (w/ Target Network)
        self.critic = critic.to(pytorch_device)
        self.critic_target = copy.deepcopy(critic).to(pytorch_device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic, weight_decay=weight_decay)

    def act(self, state) -> torch.Tensor:
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state)
        self.actor.train()
        return action

    def step(self, samples: Tuple[torch.Tensor, ...]):
        """Update policy and value parameters using given batch of experience tuples.
                Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
                where:
                    actor_target(state) -> action
                    critic_target(state, action) -> Q-value

                    :param samples: tuple of (s, a, r, s', done)
                """
        states, actions, rewards, next_states, dones = samples

        # ---------------------------- update critic ---------------------------- #
        with torch.no_grad():
            # Get predicted next-state actions and Q values from target models
            actions_next = self.actor_target(next_states) # + \
            #                (torch.rand(*actions.shape, device=pytorch_device) * 0.1 - 0.05)
            # torch.clamp_(actions_next, min=-1.0, max=1.0)
            q_targets_next = self.critic_target(next_states, actions_next)
            # Compute Q targets for current states
            q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))
        # Compute critic loss
        q_expected = self.critic(states, actions)
        critic_loss = f.mse_loss(q_expected, q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor(states) # + \
        #                (torch.rand(*actions.shape, device=pytorch_device) * 0.1 - 0.05)
        # torch.clamp_(actions_pred, min=-1.0, max=1.0)
        actor_loss = -self.critic(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
        self.actor_optimizer.step()

    def update_target_networks(self):
        soft_update(self.critic, self.critic_target, self.tau)
        soft_update(self.actor, self.actor_target, self.tau)

    def get_state_dicts(self):
        return {'actor_params': self.actor.state_dict(),
                'actor_optim_params': self.actor_optimizer.state_dict(),
                'critic_params': self.critic.state_dict(),
                'critic_optim_params': self.critic_optimizer.state_dict()}

    def load_state_dicts(self, state_dicts):
        self.actor.load_state_dict(state_dicts['actor_params'])
        self.actor_optimizer.load_state_dict(state_dicts['actor_optim_params'])
        self.critic.load_state_dict(state_dicts['critic_params'])
        self.critic_optimizer.load_state_dict(state_dicts['critic_optim_params'])


def soft_update(local_model, target_model, tau):
    """Soft update model parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target

    Params
    ======
        local_model: PyTorch model (weights will be copied from)
        target_model: PyTorch model (weights will be copied to)
        tau (float): interpolation parameter
    """
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
