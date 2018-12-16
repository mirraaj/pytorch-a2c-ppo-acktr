import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math, random
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

def _poisson(lmbd):
  L, k, p = math.exp(-lmbd), 0, 1
  while p > L:
    k += 1
    p *= random.uniform(0, 1)
  return max(k - 1, 0)

class SEPPO():
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=True,
                 c=10.,
                 replay_ratio=3,
                 bt_size=4):

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef


        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)


        self.c = c
        self.replay_ratio = replay_ratio
        self.bt_size = bt_size

    def update(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                   value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ, Q_value_batch = sample

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, _, Qs_batch, _ = self.actor_critic.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch,
                    masks_batch, actions_batch)

                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                           1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                Qs_batch = Qs_batch.view(-1, rollouts.action_space.n)
                Qs_batch = Qs_batch.gather(1, actions_batch)

                if self.use_clipped_value_loss:
                    value_pred_clipped = Qs_batch + \
                        (Qs_batch - Q_value_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (Qs_batch - return_batch).pow(2)
                    value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                self.optimizer.zero_grad()
                (value_loss * self.value_loss_coef + action_loss -
                 dist_entropy * self.entropy_coef).backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        # if len(mm) > 1:
        #     print (mm[0].state)
        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch

    # Check if working
    def update_off_policy(self, rollouts, gamma=0.99):
        n = self.replay_ratio
        if n == 0: 
            n=1
        print(n)
        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        counter = 1
        for _ in range(n):
        
            mem = rollouts.sample_batch_from_memory(self.bt_size)
            for trajectory in mem:
                states, actions, rewards, probs, masks = trajectory
                recurrent_hidden_states = torch.zeros(rollouts.num_steps + 1, rollouts.num_processes, rollouts.recurrent_hidden_state_size)
                Q = torch.zeros(rollouts.num_steps + 1, rollouts.num_processes, rollouts.action_space.n)
                probs_new = torch.zeros(rollouts.num_steps + 1, rollouts.num_processes, rollouts.action_space.n)
                V = torch.zeros(rollouts.num_steps + 1, rollouts.num_processes, 1)
                returns = torch.zeros(rollouts.num_steps + 1, rollouts.num_processes, 1)

                # Copy to Device
                Q = Q.to(rollouts.device)
                V = V.to(rollouts.device)
                probs_new = probs_new.to(rollouts.device)
                returns = returns.to(rollouts.device)
                # Got new value
                for step in range(rollouts.num_steps):
                    with torch.no_grad():
                        V[step+1], _, _, recurrent_hidden_states[step+1], Q[step+1], probs_new[step+1] = self.actor_critic.act(
                                states[step],
                                recurrent_hidden_states[step],
                                masks[step])
                # Compute returns off policy
                with torch.no_grad():
                    next_value = self.actor_critic.get_value(states[-1],
                                                        recurrent_hidden_states[-1],
                                                        masks[-1]).detach()
                
                returns[-1] = next_value
                Qret = next_value

                Qs = Q[:-1]

                # print (Qs.size(), actions.size())
                Q_action_value = Qs.gather(-1, actions)

                for step in reversed(range(rewards.size(0))):
                    Qret = Qret * \
                        gamma * masks[step + 1] + rewards[step]
                    returns[step] = Qret
                    rho = probs_new[step] / (probs[step] + 1e-5)

                    truncated_rho = rho.gather(-1, actions[step]).clamp(max=self.c)
                    Qret = truncated_rho * (Qret - Q_action_value[step].detach()) + V[step].detach()

                advantages = returns[:-1] - V[:-1]
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-5)
                value_loss_epoch = 0
                action_loss_epoch = 0
                dist_entropy_epoch = 0  
                # # DATA GENERATOR
                # print(probs.size(), probs_new.size(), 111)
                def feed_forward_generator(advantages):
                    num_steps, num_processes = rewards.size()[0:2]
                    batch_size = rollouts.num_processes * rollouts.num_steps
                    assert batch_size >= rollouts.num_mini_batch, (
                        "PPO requires the number of processes ({}) "
                        "* number of steps ({}) = {} "
                        "to be greater than or equal to the number of PPO mini batches ({})."
                        "".format(rollouts.num_processes, rollouts.num_steps, rollouts.num_processes * rollouts.num_steps, \
                            rollouts.num_mini_batch))
                    mini_batch_size = batch_size // rollouts.num_mini_batch
                    sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=False)
                    for indices in sampler:
                        obs_batch = states[:-1].view(-1, *states.size()[2:])[indices]
                        recurrent_hidden_states_batch = recurrent_hidden_states[:-1].view(-1,
                            recurrent_hidden_states.size(-1))[indices]
                        actions_batch = actions.view(-1, actions.size(-1))[indices]
                        value_preds_batch = V[:-1].view(-1, 1)[indices]
                        return_batch = returns[:-1].view(-1, 1)[indices]
                        masks_batch = masks[:-1].view(-1, 1)[indices]
                        old_probs_batch = probs[:-1].view(-1, rollouts.action_space.n)[indices]
                        new_probs_batch = probs_new[:-1].view(-1, rollouts.action_space.n)[indices]
                        adv_targ = advantages.view(-1, 1)[indices]
                        # print (advantages)
                        Q_value = Q[:-1].view(-1,rollouts.action_space.n)[indices]
                        Q_value = Q_value.gather(1, actions_batch)

                        yield obs_batch, recurrent_hidden_states_batch, actions_batch, \
                            value_preds_batch, return_batch, masks_batch, old_probs_batch.detach(), new_probs_batch.detach(), adv_targ, Q_value
                
                # Final generate experience and learn 
                ppo_epoch = 1
                for e in range(ppo_epoch):
                    data_generator = feed_forward_generator(advantages)
                    for sample in data_generator:
                        obs_batch, recurrent_hidden_states_batch, actions_batch, \
                           value_preds_batch, return_batch, masks_batch, behaviour_prob, old_prob, \
                                adv_targ, Q_value_batch = sample

                        # Reshape to do in a single forward pass for all steps
                        values, action_log_probs, dist_entropy, _, Qs_batch, probs_batch = self.actor_critic.evaluate_actions(
                            obs_batch, recurrent_hidden_states_batch,
                            masks_batch, actions_batch)

                        # Changing shape
                        values = values.view(-1, 1)
                        Qs_batch = Qs_batch.view(-1, rollouts.action_space.n)
                        probs_batch = probs_batch.view(-1, rollouts.action_space.n)
                        # print (probs_batch.size(), old_prob.size())
                        assert probs_batch.size() == old_prob.size()

                        ratio = probs_batch / (old_prob + 1e-5)
                        surr1 = ratio.gather(1, actions_batch) * adv_targ
                        surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                                   1.0 + self.clip_param) * adv_targ
                        # Computes the first term for off policy SEPPO
                        R = old_prob / (behaviour_prob + 1e-5)
                        action_loss = ((R.gather(1, actions_batch)).clamp(max=self.c).detach() *\
                            -torch.min(surr1, surr2)).mean()
                        # print (action_loss)
                        adv_tg = (Qs_batch - values)
                        adv = (adv_tg - adv_tg.mean()) / (adv_tg.std() + 1e-5)

                        surr1_offpolicycorrection = ratio * adv
                        surr2_offpolicycorrection = torch.clamp(ratio, 1.0 - self.clip_param,
                                                                       1.0 + self.clip_param) * adv
                        truncated_action_loss = (1. - self.c / (R + 1e-5)).clamp(min=0.).detach()* -torch.min(surr1_offpolicycorrection,\
                                                                                                              surr2_offpolicycorrection)
                        # print (truncated_action_loss)
                        truncated_action_loss = truncated_action_loss.mean(-1)
                        # print (truncated_action_loss.mean())
                        action_loss += truncated_action_loss.mean()
                        # PPO Loss 1


                        Qs_batch = Qs_batch.gather(1, actions_batch)

                        if self.use_clipped_value_loss:
                            value_pred_clipped = Qs_batch + \
                                (Qs_batch - Q_value_batch).clamp(-self.clip_param, self.clip_param)
                            value_losses = (Qs_batch - return_batch).pow(2)
                            value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                            value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                        else:
                            value_loss = 0.5 * (return_batch - values).pow(2).mean()

                        value_loss_epoch += value_loss.item()
                        action_loss_epoch += action_loss.item()
                        dist_entropy_epoch += dist_entropy.item()
                        counter += 1.
        self.optimizer.zero_grad()
        # print (value_loss, dist_entropy, action_loss)
        ((value_loss * self.value_loss_coef + action_loss -
         dist_entropy * self.entropy_coef)/(counter)).backward()
        nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                 self.max_grad_norm)
        self.optimizer.step()

        num_updates = n * self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        print (value_loss_epoch, action_loss_epoch, dist_entropy_epoch)
        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch