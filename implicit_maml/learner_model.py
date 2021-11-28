import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import implicit_maml.utils as utils
import datetime
from collections import deque
from logger import Logger
from tensorboardX import SummaryWriter
import os
import time
from   torch.nn import functional as F
from iq_learn.make_envs import make_env
from iq_learn.agent import make_agent
from iq_learn.utils import eval_mode, get_concat_samples, evaluate, soft_update, hard_update
from iq_learn.memory import Memory
from iq_learn.logger import Logger
import hydra
import types
from iq_learn.train_iq import irl_update, save, get_buffers, irl_update_inner, ilr_update_critic2, ilr_update_critic
from wrappers.atari_wrapper import LazyFrames
import wandb



class Learner:
    def __init__(self, model, loss_function, inner_lr=1e-3, outer_lr=1e-2, GPU=False, inner_alg='gradient', outer_alg='adam'):
        self.model = model
        self.use_gpu = GPU
        if GPU:
            self.model.cuda()
        assert outer_alg == 'sgd' or 'adam'
        self.inner_opt = torch.optim.SGD(self.model.parameters(), lr=inner_lr)
        if outer_alg == 'adam':
            self.outer_opt = torch.optim.Adam(self.model.parameters(), lr=outer_lr, eps=1e-3)
        else:
            self.outer_opt = torch.optim.SGD(self.model.parameters(), lr=outer_lr)
        self.loss_function = loss_function
        assert inner_alg == 'gradient' # sqp unsupported in this version
        self.inner_alg = inner_alg

    def get_params(self):
        return torch.cat([param.data.view(-1) for param in self.model.parameters()], 0).clone()

    def set_params(self, param_vals):
        offset = 0
        for param in self.model.parameters():
            param.data.copy_(param_vals[offset:offset + param.nelement()].view(param.size()))
            offset += param.nelement()
            
    def set_outer_lr(self, lr):
        for param_group in self.outer_opt.param_groups:
            param_group['lr'] = lr
            
    def set_inner_lr(self, lr):
        for param_group in self.inner_opt.param_groups:
            param_group['lr'] = lr

    def regularization_loss(self, w_0, lam=0.0):
        """
        Add a regularization loss onto the weights
        The proximal term regularizes around the point w_0
        Strength of regularization is lambda
        lambda can either be scalar (type float) or ndarray (numpy.ndarray)
        """
        regu_loss = 0.0
        offset = 0
        regu_lam = lam if type(lam) == float or np.float64 else utils.to_tensor(lam)
        if w_0.dtype == torch.float16:
            try:
                regu_lam = regu_lam.half()
            except:
                regu_lam = np.float16(regu_lam)
        for param in self.model.parameters():
            delta = param.view(-1) - w_0[offset:offset + param.nelement()].view(-1)
            if type(regu_lam) == float or np.float64:
                regu_loss += 0.5 * regu_lam * torch.sum(delta ** 2)
            else:
                # import ipdb; ipdb.set_trace()
                param_lam = regu_lam[offset:offset + param.nelement()].view(-1)
                param_delta = delta * param_lam
                regu_loss += 0.5 * torch.sum(param_delta ** 2)
            offset += param.nelement()
        return regu_loss

    def get_loss(self, args, expert_batch, policy_batch, return_numpy=False):
        logger = Logger(args.log_dir)
        step =0
        self.model.ilr_update_critic = types.MethodType(ilr_update_critic2, self.model)
        losses = self.ilr_update_critic2(policy_batch, expert_batch, logger, step)

        if self.actor and step % self.actor_update_frequency == 0:
            if not self.args.agent.vdice_actor:

                if self.args.offline:
                    obs = expert_batch[0]
                else:
                    # Use both policy and expert observations
                    obs = torch.cat([policy_batch[0], expert_batch[0]], dim=0)

                if self.args.num_actor_updates:
                    for i in range(self.args.num_actor_updates):
                        actor_alpha_losses = self.model.update_actor_and_alpha(obs, logger, step)

                # actor_alpha_losses = self.update_actor_and_alpha(obs, logger, step)
                losses.update(actor_alpha_losses)

        passedlosses = losses["total_loss"]    
        if return_numpy:
                passedlosses = utils.to_numpy(losses).ravel()[0]
        return passedlosses

    def predict(self, x, return_numpy=False):
        yhat = self.model.forward(utils.to_device(x, self.use_gpu))
        if return_numpy:
            yhat = utils.to_numpy(yhat)
        return yhat

    def learn_on_data(self, env_args, env,eval_env,args,expert_memory_replay, num_steps=10,
                      add_regularization=False,
                      w_0=None, lam=0.0):
        REPLAY_MEMORY = int(env_args.replay_mem)
        INITIAL_MEMORY = int(env_args.initial_mem)
        UPDATE_STEPS = int(env_args.update_steps)
        EPISODE_STEPS = int(env_args.eps_steps)
        EPISODE_WINDOW = int(env_args.eps_window)
        LEARN_STEPS = int(env_args.learn_steps)

        INITIAL_STATES = 128
        #assert self.inner_alg == 'gradient' # or 'sqp' or 'adam' # TODO(Aravind): support sqp and adam 
        #rain_loss = []
        if self.inner_alg == 'gradient':

            online_memory_replay = Memory(REPLAY_MEMORY//2, args.seed+1)

            ts_str = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")
            log_dir = os.path.join(args.log_dir, args.env.name, args.exp_name, ts_str)
            writer = SummaryWriter(log_dir=log_dir)
            print(f'--> Saving logs at: {log_dir}')
    #        TODO: Fix logging
            logger = Logger(args.log_dir)

            steps = 0

    # track avg. reward and scores
            scores_window = deque(maxlen=EPISODE_WINDOW)  # last N scores
            rewards_window = deque(maxlen=EPISODE_WINDOW)  # last N rewards
            best_eval_returns = -np.inf

            learn_steps = 0
            begin_learn = False
            episode_reward = 0

            state_0 = [env.reset()] * INITIAL_STATES
            if isinstance(state_0[0], LazyFrames):
                state_0 = np.array(state_0) / 255.0
                state_0 = torch.FloatTensor(state_0).to(args.device)
                print(state_0.shape)

            for epoch in range(num_steps):
                state = env.reset()
                episode_reward = 0
                done = False
                for episode_step in range(EPISODE_STEPS):
                    if steps < args.num_seed_steps:
                        action = env.action_space.sample()  # Sample random action
                    else:
                        with eval_mode(self.model):
                            action = self.model.choose_action(state, sample=True)
                    next_state, reward, done, _ = env.step(action)
                    episode_reward += reward
                    steps += 1

                    if learn_steps % args.env.eval_interval == 0:
                        render_path = None
                        if hasattr(args.eval, 'render') and args.eval.render:
                            render_dir = 'movies' if not hasattr(args.eval, 'render_dir') or args.eval.render_dir is None else args.eval.render_dir
                            render_path = os.path.join(render_dir, f'eval_epoch_{epoch:06d}.mp4')
                        eval_returns, eval_timesteps, eval_successes = evaluate(
                            self.model,
                            eval_env,
                            num_episodes=args.eval.eps,
                            max_steps=args.env.eps_steps,
                            render_path=render_path)
                        returns = np.mean(eval_returns)
                        success_rate = np.mean(eval_successes)
                        learn_steps += 1  # To prevent repeated eval at timestep 0
                        writer.add_scalar('Rewards/eval_rewards', returns,
                                        global_step=learn_steps)
                        writer.add_scalar('Success/eval_MW_success', np.mean(success_rate),
                                        global_step=learn_steps)
                        print('EVAL\tEp {}\tAverage reward: {:.2f}\tsuccess: {:.2f}'.format(epoch, returns, success_rate))
                        writer.add_scalar(
                            'Success/eval', np.mean((np.array(eval_returns) > 200)), global_step=epoch)
                        
                        if render_path is not None:
                            wandb.log({"video": wandb.Video(render_path, fps=4, format="gif")})

                        if returns > best_eval_returns:
                            best_eval_returns = returns
                            wandb.run.summary["best_returns"] = best_eval_returns
                            save(self.model, epoch, args, output_dir='results_best')

                    # allow infinite bootstrap
                    done_no_lim = done
                    if str(env.__class__.__name__).find('TimeLimit') >= 0 and episode_step + 1 == env._max_episode_steps:
                        done_no_lim = 0
                    online_memory_replay.add((state, next_state, action, reward, done_no_lim))

                    if online_memory_replay.size() > INITIAL_MEMORY:
                        if begin_learn is False:
                            print('learn begin!')
                            begin_learn = True

                        learn_steps += 1
                        if learn_steps == LEARN_STEPS:
                            print('Finished!')
                            wandb.finish()
                            return

                        ######
                        # IRL Modification
                        self.model.irl_update = types.MethodType(irl_update, self.model)
                        self.model.ilr_update_critic = types.MethodType(ilr_update_critic, self.model)
                        losses = self.model.irl_update(online_memory_replay,
                                                expert_memory_replay, logger, learn_steps)
                        ######

                        if learn_steps % args.log_interval == 0:
                            for key, loss in losses.items():
                                writer.add_scalar(key, loss, global_step=learn_steps)

                    if done:
                        break
                    state = next_state

                writer.add_scalar('episodes', epoch, global_step=learn_steps)

                rewards_window.append(episode_reward)
                scores_window.append(float(episode_reward > 200))
                writer.add_scalar('Rewards/train_reward', np.mean(rewards_window), global_step=epoch)
                writer.add_scalar('Success/train', np.mean(scores_window), global_step=epoch)

                print('TRAIN\tEp {}\tAverage reward: {:.2f}\t'.format(epoch, np.mean(rewards_window)))
                save(self.model, epoch, args, output_dir='results')
                
        return loss



    def learn_task(self, task, env_args, env, eval_env, args, num_steps=10, add_regularization=False, w_0=None, lam=0.0):
        REPLAY_MEMORY = int(env_args.replay_mem)
        expert_memory_replay = Memory(REPLAY_MEMORY//2, args.seed)
        expert_memory_replay.load(hydra.utils.to_absolute_path(f'experts/{args.env.demo}'),
                              num_trajs=args.eval.demos,
                              sample_freq=args.eval.subsample_freq,
                              seed=args.seed + 42)
        print(f'--> Expert memory size: {expert_memory_replay.size()}')
        return self.learn_on_data(self, env_args, env,eval_env,args, expert_memory_replay, num_steps=10,
                      add_regularization=False,
                      w_0=None, lam=0.0)

    def move_toward_target(self, target, lam=2.0):
        """
        Move slowly towards the target parameter value
        Default value for lam assumes learning rate determined by optimizer
        Useful for implementing Reptile
        """
        # we can implement this with the regularization loss, but regularize around the target point
        # and with specific choice of lam=2.0 to preserve the learning rate of inner_opt
        self.outer_opt.zero_grad()
        loss = self.regularization_loss(target, lam=lam)
        loss.backward()
        self.outer_opt.step()

    def outer_step_with_grad(self, grad, flat_grad=False):
        """
        Given the gradient, step with the outer optimizer using the gradient.
        Assumed that the gradient is a tuple/list of size compatible with model.parameters()
        If flat_grad, then the gradient is a flattened vector
        """
        check = 0
        for p in self.model.parameters():
            check = check + 1 if type(p.grad) == type(None) else check
        if check > 0:
            # initialize the grad fields properly
            dummy_loss = self.regularization_loss(self.get_params())
            dummy_loss.backward()  # this would initialize required variables
        if flat_grad:
            offset = 0
            grad = utils.to_device(grad, self.use_gpu)
            for p in self.model.parameters():
                this_grad = grad[offset:offset + p.nelement()].view(p.size())
                p.grad.copy_(this_grad)
                offset += p.nelement()
        else:
            for i, p in enumerate(self.model.parameters()):
                p.grad = grad[i]
        self.outer_opt.step()

    def matrix_evaluator(self, task, lam, regu_coef=1.0, lam_damping=10.0, x=None, y=None):
        """
        Constructor function that can be given to CG optimizer
        Works for both type(lam) == float and type(lam) == np.ndarray
        """
        if type(lam) == np.ndarray:
            lam = utils.to_device(lam, self.use_gpu)
        def evaluator(v):
            hvp = self.hessian_vector_product(task, v, x=x, y=y)
            Av = (1.0 + regu_coef) * v + hvp / (lam + lam_damping)
            return Av
        return evaluator

    def hessian_vector_product(self, task, vector, params=None, x=None, y=None):
        """
        Performs hessian vector product on the train set in task with the provided vector
        """
        if x is not None and y is not None:
            xt, yt = x, y
        else:
            expert_memory_replay, online_memory_replay = get_buffers(env, args,EPISODE_STEPS, REPLAY_MEMORY, INITIAL_MEMORY, fast_learner)
            
        if params is not None:
            self.set_params(params)
        tloss = self.get_loss(args, expert_memory_replay, online_memory_replay)
        grad_ft = torch.autograd.grad(tloss, self.model.parameters(), create_graph=True)
        flat_grad = torch.cat([g.contiguous().view(-1) for g in grad_ft])
        vec = utils.to_device(vector, self.use_gpu)
        h = torch.sum(flat_grad * vec)
        hvp = torch.autograd.grad(h, self.model.parameters())
        hvp_flat = torch.cat([g.contiguous().view(-1) for g in hvp])
        return hvp_flat


def make_fc_network(in_dim=1, out_dim=1, hidden_sizes=(40,40), float16=False):
    non_linearity = nn.ReLU()
    model = nn.Sequential()
    model.add_module('fc_0', nn.Linear(in_dim, hidden_sizes[0]))
    model.add_module('nl_0', non_linearity)
    model.add_module('fc_1', nn.Linear(hidden_sizes[0], hidden_sizes[1]))
    model.add_module('nl_1', non_linearity)
    model.add_module('fc_2', nn.Linear(hidden_sizes[1], out_dim))
    if float16:
        return model.half()
    else:
        return model

    
def make_SAC_network(args, task='PickPlaceMetaWorld'):
    assert task == 'PickPlaceMetaWorld'
    
    if task == 'PickPlaceMetaWorld':
        env = make_env(args)
        model= make_agent(env, args)
    
    return model


class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x

    
def model_imagenet_arch(in_channels, out_dim, num_filters=32, batch_norm=True, bias=True):
    raise NotImplementedError

