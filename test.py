import sys
sys.path.append("./common")
sys.path.append("./auto_LiRPA")
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from argparser import argparser
import numpy as np
from read_config import load_config
from attacks import attack
from common.wrappers import make_atari, wrap_deepmind, wrap_pytorch, make_atari_cart
from models import QNetwork, model_setup
import torch.optim as optim
import torch
import torch.autograd as autograd
import time
import os
import argparse
import random
from datetime import datetime
from utils import Logger, get_acrobot_eps, test_plot 
from async_env import AsyncEnv
from train import get_logits_lower_bound

UINTS=[np.uint8, np.uint16, np.uint32, np.uint64]
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)


def main(args):
    config = load_config(args)
    prefix = config['env_id']
    training_config = config['training_config']
    test_config = config['test_config']
    attack_config = test_config["attack_config"]
    if config['name_suffix']:
        prefix += config['name_suffix']
    if config['path_prefix']:
        prefix = os.path.join(config['path_prefix'], prefix)
    if 'load_model_path' in test_config and os.path.isfile(test_config['load_model_path']):
        if not os.path.exists(prefix):
            os.makedirs(prefix)
        test_log = os.path.join(prefix, test_config['log_name'])
    else:
        if os.path.exists(prefix):
            test_log = os.path.join(prefix, test_config['log_name'])
        else:
            raise ValueError('Path {} not exists, please specify test model path.')
    logger = Logger(open(test_log, "w"))
    logger.log('Command line:', " ".join(sys.argv[:]))
    logger.log(args)
    logger.log(config)
    certify = test_config.get('certify', False)
    env_params = training_config['env_params']
    env_params['clip_rewards'] = False
    env_params['episode_life'] = False
    env_id = config['env_id']

    if "NoFrameskip" not in env_id:
        env = make_atari_cart(env_id)
    else:
        env = make_atari(env_id)
        env = wrap_deepmind(env, **env_params)
        env = wrap_pytorch(env)

    state = env.reset()
    dtype = state.dtype
    logger.log("env_shape: {}, num of actions: {}".format(env.observation_space.shape, env.action_space.n))

    model_width = training_config['model_width']
    robust_model = certify
    dueling = training_config.get('dueling', True)

    model = model_setup(env_id, env, robust_model, logger, USE_CUDA, dueling, model_width)

    if 'load_model_path' in test_config and os.path.isfile(test_config['load_model_path']):
        model_path = test_config['load_model_path']
    else:
        logger.log("choosing the best model from " + prefix)
        all_idx = [int(f[6:-4]) for f in os.listdir(prefix) if os.path.isfile(os.path.join(prefix, f)) and os.path.splitext(f)[1]=='.pth' and 'best' not in f]
        all_best_idx = [int(f[11:-4]) for f in os.listdir(prefix) if os.path.isfile(os.path.join(prefix, f)) and os.path.splitext(f)[1]=='.pth' and 'best' in f]
        if all_best_idx:
            model_frame_idx = max(all_best_idx)
            model_name = 'best_frame_{}.pth'.format(model_frame_idx)
        else:
            model_frame_idx = max(all_idx)
            model_name = 'frame_{}.pth'.format(model_frame_idx)
        model_path = os.path.join(prefix, model_name)

    logger.log('model loaded from ' + model_path)
    model.features.load_state_dict(torch.load(model_path))
    num_episodes = test_config['num_episodes']
    max_frames_per_episode = test_config['max_frames_per_episode']

    all_rewards = []
    episode_reward = 0

    seed = random.randint(0, sys.maxsize)
    logger.log('reseting env with seed', seed)
    env.seed(seed)
    state = env.reset()
    start_time = time.time()
    if training_config['use_async_env']:
        # Create an environment in a separate process, run asychronously
        async_env = AsyncEnv(env_id, result_path=prefix, draw=training_config['show_game'], record=training_config['record_game'], save_frames=test_config['save_frames'], env_params=env_params, seed=args.seed)

    episode_idx = 1
    this_episode_frame = 1

    if certify:
        certified = 0

    if dtype in UINTS:
        state_max = 1.0
        state_min = 0.0
    else:
        state_max = float('inf')
        state_min = float('-inf')

    for frame_idx in range(1, num_episodes * max_frames_per_episode + 1):

        state_tensor = torch.from_numpy(np.ascontiguousarray(state)).unsqueeze(0).cuda().to(torch.float32)
        # Normalize input pixel to 0-1
        if dtype in UINTS:
            state_tensor /= 255

        if test_config['attack']:
            attack_config['params']['robust_model'] = certify
            state_tensor = attack(model, state_tensor, attack_config)

        if certify:
            beta = training_config.get('convex_final_beta',0)
            eps = attack_config['params']['epsilon']
            if env_id == 'Acrobot-v1':
                eps_v = get_acrobot_eps(eps)
                if USE_CUDA:
                    eps_v = eps_v.cuda()
            else:
                eps_v = eps
            state_ub = torch.clamp(state_tensor + eps_v, max=state_max)
            state_lb = torch.clamp(state_tensor - eps_v, min=state_min)

        action = model.act(state_tensor)[0]

        if certify:
            max_logit = torch.tensor([action])
            c = torch.eye(model.num_actions).type_as(state_tensor)[max_logit].unsqueeze(1) - torch.eye(model.num_actions).type_as(state_tensor).unsqueeze(0)
            I = (~(max_logit.data.unsqueeze(1) == torch.arange(model.num_actions).type_as(max_logit.data).unsqueeze(0)))
            c = (c[I].view(state_tensor.size(0), model.num_actions-1, model.num_actions))
            logits_diff_lb = get_logits_lower_bound(model, state_tensor, state_ub, state_lb, eps_v, c, beta)
            if torch.min(logits_diff_lb[0], 0)[0].data.cpu().numpy() > 0:
                certified += 1


        if training_config['use_async_env']:
            async_env.async_step(action)
            next_state, reward, done, _ = async_env.wait_step()
        else:
            next_state, reward, done, _ = env.step(action)

        state = next_state
        episode_reward += reward

        if frame_idx % test_config['print_frame']==0:
            logger.log('\ntotal frame {}/{}, episode {}/{}, episode frame{}/{}, latest episode reward: {:.6g}, avg 10 episode reward: {:.6g}'.format(frame_idx, num_episodes*max_frames_per_episode, episode_idx, num_episodes, this_episode_frame, max_frames_per_episode,
                all_rewards[-1] if all_rewards else np.nan,
                np.average(all_rewards[:-11:-1]) if all_rewards else np.nan))
            if certify:
                logger.log('certified action: {}, certified action ratio: {:.6g}'.format(certified, certified*1.0/frame_idx))

        if this_episode_frame == max_frames_per_episode:
            logger.log('maximum number of frames reached in this episode, reset environment!')
            done = True
            if training_config['use_async_env']:
                async_env.epi_reward = 0

        if done:
            logger.log('reseting env with seed', seed)
            if training_config['use_async_env']:
                state = async_env.reset()
            else:
                state = env.reset()
            all_rewards.append(episode_reward)
            episode_reward = 0
            this_episode_frame = 1
            episode_idx += 1
            if episode_idx > num_episodes:
                break
        else:
            this_episode_frame += 1

    logger.log('\navg reward' + (' and avg certify:' if certify else ':'))
    logger.log(np.mean(all_rewards),'+-',np.std(all_rewards))
    if certify:
        logger.log(certified*1.0/frame_idx)


if __name__ == "__main__":
    args=  argparser()
    main(args)
