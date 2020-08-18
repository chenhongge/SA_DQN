import math, random
import numpy as np
import torch
import torch.nn as nn
from torch import autograd
import torch.optim as optim
from IPython.display import clear_output
import matplotlib
import time
matplotlib.use('Agg')
import matplotlib.pyplot as plt

USE_CUDA = torch.cuda.is_available()
ACROBOT_STD=[0.36641926, 0.65119815, 0.6835106, 0.67652863, 2.0165246, 3.0202584]
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)


class Logger(object):
    def __init__(self, log_file = None):
        self.log_file = log_file

    def log(self, *args, **kwargs):
        print(*args, **kwargs)
        if self.log_file:
            print(*args, **kwargs, file = self.log_file)
            self.log_file.flush()


class ActEpsilonScheduler(object):
    def __init__(self, epsilon_start = 1.0, epsilon_final = 0.01, epsilon_decay = 30000, method = 'linear', start_frame = 0, decay_zero = None):
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.method = method
        self.start_frame = start_frame
        self.decay_zero = decay_zero
    def get(self, frame_idx):
        if frame_idx < self.start_frame:
            return self.epsilon_start
        if self.method == 'exponential':
            return self.epsilon_final + (self.epsilon_start - self.epsilon_final) * math.exp(-1. * (frame_idx - self.start_frame) / self.epsilon_decay)
        else:
            # linear decay
            if self.decay_zero == None or self.decay_zero <= self.start_frame + self.epsilon_decay or frame_idx <= self.start_frame + self.epsilon_decay:
                return max(self.epsilon_final, self.epsilon_start + (self.epsilon_final - self.epsilon_start) * (frame_idx - self.start_frame) * 1. / self.epsilon_decay)
            else:
                # second stage linear decay to 0
                return max(0, self.epsilon_final * (self.decay_zero - frame_idx) / (self.decay_zero - self.start_frame - self.epsilon_decay))


class BufferBetaScheduler(object):
    def __init__(self, beta_start = 0.4, beta_frames = 1000, start_frame = 0):
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.start_frame = start_frame
    def get(self, frame_idx):
        return max(self.beta_start, min(1.0, self.beta_start + (frame_idx - self.start_frame) * (1.0 - self.beta_start) / self.beta_frames))


class CudaTensorManager(object):
    def __init__(self, state_shape, batch_size, per, use_cuda=True, dtype=np.uint8):
        # Allocate pinned memory at once
        # states and pinned states are allocated as uint8 to save transfer time
        self.dtype = dtype
        if dtype == np.uint8:
            self.pinned_next_state = torch.empty(batch_size, *state_shape, dtype=torch.uint8, pin_memory=True)
            self.pinned_state      = torch.empty(batch_size, *state_shape, dtype=torch.uint8, pin_memory=True)
        else:
            self.pinned_next_state = torch.empty(batch_size, *state_shape, dtype=torch.float32, pin_memory=True)
            self.pinned_state      = torch.empty(batch_size, *state_shape, dtype=torch.float32, pin_memory=True)

        self.pinned_reward  = torch.empty(batch_size, dtype=torch.float32, pin_memory=True)
        self.pinned_done    = torch.empty(batch_size, dtype=torch.float32, pin_memory=True)
        self.pinned_action  = torch.empty(batch_size, dtype=torch.int64, pin_memory=True)
        self.per = per
        self.use_cuda = use_cuda
        if self.per:
            self.pinned_weights    = torch.empty(batch_size, dtype=torch.float32, pin_memory=True)
        self.ncall = 0

    def get_cuda_tensors(self, state, next_state, action, reward, done, weights = None):
        """
        state      = torch.cuda.FloatTensor(state)
        next_state = torch.cuda.FloatTensor(next_state)
        action     = torch.cuda.LongTensor(action)
        reward     = torch.cuda.FloatTensor(reward)
        done       = torch.cuda.FloatTensor(done)
        if self.per:
            weights    = torch.cuda.FloatTensor(weights)
        return state, next_state, action, reward, done, weights
        """
        # Copy numpy array to pinned memory
        t = time.time()
        if self.dtype == np.uint8:
            self.pinned_next_state.copy_(torch.from_numpy(next_state.astype(np.uint8)))
            self.pinned_state.copy_(torch.from_numpy(state.astype(np.uint8)))
        else:
            self.pinned_next_state.copy_(torch.from_numpy(next_state.astype(self.dtype)))
            self.pinned_state.copy_(torch.from_numpy(state.astype(self.dtype)))

        self.pinned_reward.copy_(torch.from_numpy(reward))
        self.pinned_done.copy_(torch.from_numpy(done))
        self.pinned_action.copy_(torch.from_numpy(action))
        if self.per:
            self.pinned_weights.copy_(torch.from_numpy(weights))
        if self.use_cuda:
            # Use asychronous transfer. The order is important, start with the first tensor we will need to use.
            cuda_next_state = self.pinned_next_state.cuda(non_blocking=True)
            cuda_state      = self.pinned_state.cuda(non_blocking=True)
            cuda_reward     = self.pinned_reward.cuda(non_blocking=True)
            cuda_done       = self.pinned_done.cuda(non_blocking=True)
            cuda_action     = self.pinned_action.cuda(non_blocking=True)
            if self.per:
                cuda_weights    = self.pinned_weights.cuda(non_blocking=True)
        else:
            cuda_next_state = self.pinned_next_state
            cuda_state      = self.pinned_state
            cuda_reward     = self.pinned_reward
            cuda_done       = self.pinned_done
            cuda_action     = self.pinned_action
            if self.per:
                cuda_weights    = self.pinned_weights
        if self.per:
            return cuda_state, cuda_next_state, cuda_action, cuda_reward, cuda_done, cuda_weights
        else:
            return cuda_state, cuda_next_state, cuda_reward, cuda_reward, cuda_done


def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())


def plot(frame_idx, rewards, losses, prefix='.'):
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
    plt.plot(rewards)
    plt.subplot(132)
    plt.title('loss')
    plt.plot(losses)
    plt.savefig('{}/rewards_losses_so_far.pdf'.format(prefix))
    np.save('{}/frame_{}_losses.npy'.format(prefix, frame_idx), losses)
    np.save('{}/frame_{}_rewards.npy'.format(prefix, frame_idx), rewards)
    plt.close('all')


def test_plot(model_frame, frame_idx, rewards, prefix='.'):
    clear_output(True)
    plt.figure(figsize=(15,5))
    plt.subplot(121)
    plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
    plt.plot(rewards)
    plt.savefig('{}/model_frame_{}_test_frame_{}.pdf'.format(prefix, model_frame, frame_idx))
    plt.close('all')


def torch_arctanh(x, eps=1e-6):
    x *= (1 - eps)
    return (torch.log((1 + x) / (1 - x))) * 0.5


def tanh_rescale(x, x_min=-1., x_max=1.):
    return (torch.tanh(x)) * 0.5 * (x_max - x_min) + (x_max + x_min) * 0.5


def arctanh_rescale(y, x_min=-1., x_max=1.):
    return torch_arctanh((2*y-x_max-x_min)/(x_max-x_min))


def to_one_hot(y, num_classes):
    """
    Take a batch of label y with n dims and convert it to
    1-hot representation with n+1 dims.
    Link: https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/24
    """
    # y = y.detach().clone().view(-1, 1)
    # y_onehot = y.new_zeros((y.size()[0], num_classes)).scatter_(1, y, 1)
    y_onehot = torch.FloatTensor(1, num_classes)
    y_onehot.zero_()
    y_onehot.scatter_(1, torch.tensor([[y]]), 1)
    return Variable(y_onehot)


def get_acrobot_eps(eps):
    return eps * torch.Tensor(ACROBOT_STD)

