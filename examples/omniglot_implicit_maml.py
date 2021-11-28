import numpy as np
import torch
import torch.nn as nn
import implicit_maml.utils as utils
import random
import time 
import pickle
import argparse
import pathlib
import datetime
from tensorboardx import SummaryWriter
import os
from collections import deque

#local imports
from tqdm import tqdm
from implicit_maml.learner_model import Learner
from implicit_maml.learner_model import make_fc_network, make_SAC_network
from implicit_maml.utils import DataLog
from iq_learn.train_iq import get_buffers
from iq_learn.memory import Memory
from iq_learn.logger import Logger
from garage.envs import MetaWorldSetTaskEnv
from garage.experiment import (MetaEvaluator, MetaWorldTaskSampler,
                               SetTaskSampler)
from iq_learn.mw.record_demos import DemosRepository
import metaworld
import matplotlib.pyplot as plt

np.random.seed(123)
torch.manual_seed(123)
random.seed(123)
logger = DataLog()

# ===================
# hyperparameters
# ===================
parser = argparse.ArgumentParser(description='Implicit MAML + iQLearn on MetaWorld Tasks')
parser.add_argument('--data_dir', type=str, default='/home/aravind/data/omniglot-py/',
                    help='location of the dataset')
parser.add_argument('--N_way', type=int, default=5, help='number of classes for few shot learning tasks')
parser.add_argument('--K_shot', type=int, default=1, help='number of instances for few shot learning tasks')
parser.add_argument('--inner_lr', type=float, default=1e-2, help='inner loop learning rate')
parser.add_argument('--outer_lr', type=float, default=1e-2, help='outer loop learning rate')
parser.add_argument('--n_steps', type=int, default=16, help='number of steps in inner loop')
parser.add_argument('--meta_steps', type=int, default=1000, help='number of meta steps')
parser.add_argument('--task_mb_size', type=int, default=16)
parser.add_argument('--lam', type=float, default=1.0, help='regularization in inner steps')
parser.add_argument('--cg_steps', type=int, default=5)
parser.add_argument('--cg_damping', type=float, default=1.0)
parser.add_argument('--use_gpu', type=bool, default=True)
parser.add_argument('--num_tasks', type=int, default=20000)
parser.add_argument('--save_dir', type=str, default='/tmp')
parser.add_argument('--lam_lr', type=float, default=0.0)
parser.add_argument('--lam_min', type=float, default=0.0)
parser.add_argument('--scalar_lam', type=bool, default=True, help='keep regularization as a scalar or diagonal matrix (vector)')
parser.add_argument('--taylor_approx', type=bool, default=False, help='Use Neumann approximation for (I+eps*A)^-1')
parser.add_argument('--inner_alg', type=str, default='gradient', help='gradient or sqp for inner solve')
parser.add_argument('--load_agent', type=str, default=None)
parser.add_argument('--load_tasks', type=str, default=None)
parser.add_argument('--method_loss', type = str, default = 'value_expert')
args = parser.parse_args()
logger.log_exp_args(args)

print("Generating Creating Models ")
if args.load_agent is None:
    learner_net = make_SAC_network(args)
    fast_net = make_SAC_network(args)
    meta_learner = Learner(model=learner_net, loss_function=args.method_loss, inner_alg=args.inner_alg,
                           inner_lr=args.inner_lr, outer_lr=args.outer_lr, GPU=args.use_gpu)
    fast_learner = Learner(model=fast_net, loss_function=args.method_loss, inner_alg=args.inner_alg,
                           inner_lr=args.inner_lr, outer_lr=args.outer_lr, GPU=args.use_gpu)
else:
    meta_learner = pickle.load(open(args.load_agent, 'rb'))
    meta_learner.set_params(meta_learner.get_params())
    fast_learner = pickle.load(open(args.load_agent, 'rb'))
    fast_learner.set_params(fast_learner.get_params())
    for learner in [meta_learner, fast_learner]:
        learner.inner_alg = args.inner_alg
        learner.inner_lr = args.inner_lr
        learner.outer_lr = args.outer_lr
    
init_params = meta_learner.get_params()
device = 'cuda' if args.use_gpu is True else 'cpu'
lam = torch.tensor(args.lam) if args.scalar_lam is True else torch.ones(init_params.shape[0])*args.lam
lam = lam.to(device)

pathlib.Path(args.save_dir).mkdir(parents=True, exist_ok=True)


#===============================
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

env_args = args.env

REPLAY_MEMORY = int(env_args.replay_mem)
INITIAL_MEMORY = int(env_args.initial_mem)
EPISODE_STEPS = int(env_args.eps_steps)
EPISODE_WINDOW = int(env_args.eps_window)
LEARN_STEPS = int(env_args.learn_steps)

online_memory_replay = Memory(REPLAY_MEMORY//2, args.seed+1)

ts_str = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")
log_dir = os.path.join(args.log_dir, args.env.name, args.exp_name, ts_str)
writer = SummaryWriter(log_dir=log_dir)
print(f'--> Saving logs at: {log_dir}')
# TODO: Fix logging
logger = Logger(args.log_dir)

steps = 0

# track avg. reward and scores
scores_window = deque(maxlen=EPISODE_WINDOW)  # last N scores
rewards_window = deque(maxlen=EPISODE_WINDOW)  # last N rewards
best_eval_returns = -np.inf

learn_steps = 0
begin_learn = False
episode_reward = 0

benchmark = None
if env_args.mw_benchmark == 'ml1':
    benchmark = metaworld.ML1(env_args.mw_task)
elif env_args.mw_benchmark == 'ml10':
    benchmark = metaworld.ML10()
else:
    raise Exception(f'Unknown Meta-World benchmark {env_args.mw_benchmark}')

train_sampler = MetaWorldTaskSampler(benchmark, 'train')
test_sampler = SetTaskSampler(MetaWorldSetTaskEnv,
                                env=MetaWorldSetTaskEnv(benchmark, 'test'))

demos_repo = DemosRepository(args.meta.demos_per_task)

sample_env = train_sampler.sample(args.meta.meta_batch_size)[0]()


# ===================
# Train
# ===================
print("Training model ......")
losses = np.zeros((args.meta_steps, 4))
accuracy = np.zeros((args.meta_steps, 2))
for outstep in tqdm(range(args.meta_steps)):
    tasks = train_sampler.sample(args.meta.meta_batch_size)
    print(f'Epoch {outstep}, tasks: {",".join(task._task.task_name for task in tasks)}')
    w_k = meta_learner.get_params()
    meta_grad = 0.0
    lam_grad = 0.0
    
    for task in tasks:
        fast_learner.set_params(w_k.clone()) # sync weights
        #task = dataset.__getitem__(idx) # get task
        env = task._env_type()
        env.set_task(task._task)
        eval_env = task._env_type()
        eval_env.set_task(task._task)
        # TODO: do we need the separate eval env? Should we set different seeds?
        # Seed envs
        #env.seed(args.seed)
        #eval_env.seed(args.seed + 10)
        
        # Obtain demonstrations for the current task.
        demos = demos_repo.get_demos(task)
        expert_memory_replay = Memory(REPLAY_MEMORY//2, args.seed)
        for i in range(len(demos["states"])):
                # For each step...
                for j in range(len(demos["states"][i])):
                    state = demos["states"][i][j]
                    next_state = demos["next_states"][i][j]
                    action = demos["actions"][i][j]
                    reward = demos["rewards"][i][j]
                    done_no_lim = demos["dones"][i][j]
                    expert_memory_replay.add((state, next_state, action, reward, done_no_lim))
                    print(f'--> Expert memory size: {expert_memory_replay.size()}')
        online_memory_replay = get_buffers(env,args,EPISODE_STEPS, REPLAY_MEMORY, INITIAL_MEMORY, fast_learner)
        policy_batch = online_memory_replay.get_samples(args.train.batch, args.device)
        expert_batch = expert_memory_replay.get_samples(args.train.batch, args.device)
        vl_before = fast_learner.get_loss(args, expert_batch, policy_batch,return_numpy=True)
        tl = fast_learner.learn_task(task, num_steps=args.n_steps)
        # pull back for regularization
        fast_learner.inner_opt.zero_grad()
        regu_loss = fast_learner.regularization_loss(w_k, lam)
        regu_loss.backward()
        fast_learner.inner_opt.step()
        vl_after = fast_learner.get_loss(args, expert_batch, policy_batch,return_numpy=True)
        #tacc = utils.measure_accuracy(task, fast_learner, train=True)
        #vacc = utils.measure_accuracy(task, fast_learner, train=False)
        
        
        #Validation Set Batch - refresh the expert memory replay?
        demos = demos_repo.get_demos(task)
        expert_memory_replay = Memory(REPLAY_MEMORY//2, args.seed)
        for i in range(len(demos["states"])):
                # For each step...
                for j in range(len(demos["states"][i])):
                    state = demos["states"][i][j]
                    next_state = demos["next_states"][i][j]
                    action = demos["actions"][i][j]
                    reward = demos["rewards"][i][j]
                    done_no_lim = demos["dones"][i][j]
                    expert_memory_replay.add((state, next_state, action, reward, done_no_lim))
                    print(f'--> Expert memory size: {expert_memory_replay.size()}')
        online_memory_replay = get_buffers(eval_env,args,EPISODE_STEPS, REPLAY_MEMORY, INITIAL_MEMORY, fast_learner)
        policy_batch = online_memory_replay.get_samples(args.train.batch, args.device)
        expert_batch = expert_memory_replay.get_samples(args.train.batch, args.device)
        #Validation Set Batch
        valid_loss = fast_learner.get_loss(args, expert_batch, policy_batch,return_numpy=True)
        valid_grad = torch.autograd.grad(valid_loss, fast_learner.model.parameters())
        flat_grad = torch.cat([g.contiguous().view(-1) for g in valid_grad])
        
        if args.cg_steps <= 1:
            task_outer_grad = flat_grad
        else:
            task_matrix_evaluator = fast_learner.matrix_evaluator(task, lam, args.cg_damping)
            task_outer_grad = utils.cg_solve(task_matrix_evaluator, flat_grad, args.cg_steps, x_init=None)
        
        meta_grad += (task_outer_grad/args.task_mb_size)
        losses[outstep] += (np.array([tl[0], vl_before, tl[-1], vl_after])/args.meta.meta_batch_size)
        #accuracy[outstep] += np.array([tacc, vacc]) / args.task_mb_size
              
        if args.lam_lr <= 0.0:
            task_lam_grad = 0.0
        else:
            print("Warning: lambda learning is not tested for this version of code")
            train_loss = fast_learner.get_loss(task['x_train'], task['y_train'])
            train_grad = torch.autograd.grad(train_loss, fast_learner.model.parameters())
            train_grad = torch.cat([g.contiguous().view(-1) for g in train_grad])
            inner_prod = train_grad.dot(task_outer_grad)
            task_lam_grad = inner_prod / (lam**2 + 0.1)

        lam_grad += (task_lam_grad / args.meta.meta_batch_size)
    
    meta_learner.outer_step_with_grad(meta_grad, flat_grad=True)
    lam_delta = - args.lam_lr * lam_grad
    lam = torch.clamp(lam + lam_delta, args.lam_min, 5000.0) # clips each element individually if vector
    logger.log_kv('train_pre', losses[outstep,0])
    logger.log_kv('test_pre', losses[outstep,1])
    logger.log_kv('train_post', losses[outstep,2])
    logger.log_kv('test_post', losses[outstep,3])
    logger.log_kv('train_acc', accuracy[outstep, 0])
    logger.log_kv('val_acc', accuracy[outstep, 1])
    
    if (outstep % 50 == 0 and outstep > 0) or outstep == args.meta_steps-1:
        smoothed_losses = utils.smooth_vector(losses[:outstep], window_size=10)
        plt.figure(figsize=(10,6))
        plt.plot(smoothed_losses)
        plt.ylim([0, 2.0])
        plt.xlim([0, args.meta_steps])
        plt.grid(True)
        plt.legend(['Train pre', 'Test pre', 'Train post', 'Test post'], loc=1)
        plt.savefig(args.save_dir+'/learn_curve.png', dpi=100)
        plt.clf()
        plt.close('all')
        
        smoothed_acc = utils.smooth_vector(accuracy[:outstep], window_size=25)
        plt.figure(figsize=(10,6))
        plt.plot(smoothed_acc)
        plt.ylim([50.0, 100.0])
        plt.xlim([0, args.meta_steps])
        plt.grid(True)
        plt.legend(['Train post', 'Test post'], loc=4)
        plt.savefig(args.save_dir+'/accuracy.png', dpi=100)
        plt.clf()
        plt.close('all')

        pickle.dump(meta_learner, open(args.save_dir+'/agent.pickle', 'wb'))
        logger.save_log()

    if (outstep % 500 == 0 and outstep > 0):
        checkpoint_file = args.save_dir + '/checkpoint_' + str(outstep) + '.pickle'
        pickle.dump(meta_learner, open(checkpoint_file, 'wb'))
    if outstep == args.meta_steps-1:
        checkpoint_file = args.save_dir + '/final_model.pickle'
        pickle.dump(meta_learner, open(checkpoint_file, 'wb'))