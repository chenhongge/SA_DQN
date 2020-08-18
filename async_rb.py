from multiprocessing import Pool
import numpy as np
from shmemarray import NpShmemArray
import os


def initializer(name_prefix, state_shape, state_type, kwargs):
    import cpprb
    from setproctitle import setproctitle
    global replay_buffer
    global numpy_buffers
    setproctitle('replay-buf')
    # Create replay buffer
    replay_buffer = cpprb.PrioritizedReplayBuffer(**kwargs)
    state = NpShmemArray(state_type, state_shape, name_prefix + "_state", create=False)
    next_state = NpShmemArray(state_type, state_shape, name_prefix + "_next_state", create=False)
    # Create byte tensors
    b_state = NpShmemArray(state_type, state_shape[1:], name_prefix + "_stateb", create=False)
    b_next_state = NpShmemArray(state_type, state_shape[1:], name_prefix + "_next_stateb", create=False)
    # Create shared memory for reweight
    indices = NpShmemArray(np.uint64, state_shape[:1], name_prefix + "_indicies", create=False)
    priorities = NpShmemArray(np.float32, state_shape[:1], name_prefix + "_priorities", create=False)
    numpy_buffers = state, next_state, b_state, b_next_state, indices, priorities

def worker_add(args):
    import time
    global replay_buffer
    global numpy_buffers
    action, reward, done = args
    # states already copied to shared memory, avoid IPC
    _, _, state, next_state, _, _ = numpy_buffers
    replay_buffer.add(obs=state, act=action, rew=reward, next_obs=next_state, done=done)
    return None

def worker_sample(args):
    global replay_buffer
    global numpy_buffers
    batch_size, buffer_beta = args
    res = replay_buffer.sample(batch_size, buffer_beta)
    res['act'] = res['act'].transpose()[0].astype(int)
    res['rew'] = res['rew'].transpose()[0].astype(int)
    res['done'] = res['done'].transpose()[0].astype(int)
    # copy states to shared memory, avoid IPC
    state, next_state, _, _, _, _ = numpy_buffers
    state[:] = res['obs']
    next_state[:] = res['next_obs']
    del res['obs']
    del res['next_obs']
    return res


def worker_get_size():
    global replay_buffer
    global numpy_buffers
    return replay_buffer.get_stored_size()

def worker_update_priorities():
    global replay_buffer
    global numpy_buffers
    _, _, _, _, indices, priorities = numpy_buffers
    return replay_buffer.update_priorities(indices, priorities)


class AsyncReplayBuffer(object):
    def __init__(self, initial_state=None, batch_size=None):
        self.total_items = 0
        self.synchronous = False
        if self.synchronous:
            self.update_priorities_args = None
            self.add_args = None
            self.sample_args = None
        else:
            # Create shared memory for state variables, as they are too large for IPC
            name_prefix = "{}".format(os.getpid())
            state_type = initial_state.dtype
            state_shape = (batch_size,) + initial_state.shape
            # print('shape is', state_shape)
            self.state = NpShmemArray(state_type, state_shape, name_prefix + "_state", create=True)
            self.next_state = NpShmemArray(state_type, state_shape, name_prefix + "_next_state", create=True)
            # Create byte tensors
            self.b_state = NpShmemArray(state_type, state_shape[1:], name_prefix + "_stateb", create=True)
            self.b_next_state = NpShmemArray(state_type, state_shape[1:], name_prefix + "_next_stateb", create=True)
            # Create shared memory for reweight
            self.indices = NpShmemArray(np.uint64, state_shape[:1], name_prefix + "_indicies", create=True)
            self.priorities = NpShmemArray(np.float32, state_shape[:1], name_prefix + "_priorities", create=True)
            self.name_prefix = name_prefix
            self.state_shape = state_shape
            self.state_type = state_type
            self.add_res = None
            self.prio_res = None
            self.sample_res = None

    def __call__(self, **kwargs):
        if self.synchronous:
            import cpprb
            self.replay_buffer = cpprb.PrioritizedReplayBuffer(**kwargs)
            return self
        else:
            # Create subprocess
            self.pool = Pool(processes=1, initializer=initializer, initargs=(
                self.name_prefix, self.state_shape, self.state_type, kwargs,))
            return self


    def add(self, obs, act, rew, next_obs, done):
        self.async_add(obs, act, rew, next_obs, done)

    def async_add(self, obs, act, rew, next_obs, done):
        self.total_items += 1
        if self.synchronous:
            """
            if self.add_args is not None:
                obs_o, act_o, rew_o, next_obs_o, done_o = self.add_args
                self.replay_buffer.add(obs=obs_o, act=act_o, rew=rew_o, next_obs=next_obs_o, done=done_o)
            self.add_args = obs, act, rew, next_obs, done
            """
            self.add_res = self.replay_buffer.add(obs=obs, act=act, rew=rew, next_obs=next_obs, done=done)
            return self.add_res
        else:
            self.b_state[:] = obs
            self.b_next_state[:] = next_obs
            self.add_res = self.pool.apply_async(worker_add, [(act, rew, done)])
            return

    def wait_add(self):
        if self.synchronous:
            return self.add_res
        else:
            if self.add_res is None:
                return None
            res = self.add_res.get()
            self.add_res = None
            return res

    def get_stored_size(self):
        if self.synchronous:
            return self.replay_buffer.get_stored_size()
        else:
            # a fast path
            return self.total_items
            # return self.pool.apply(worker_get_size)

    def update_priorities(self, indices, priorities):
        self.async_update_priorities(indices, priorities)

    def async_update_priorities(self, indices, priorities):
        if self.synchronous:
            self.prio_res = self.replay_buffer.update_priorities(indices, priorities)
            return self.prio_res
        else:
            self.indices[:] = indices
            self.priorities[:] = priorities
            self.prio_res = self.pool.apply_async(worker_update_priorities)
        return

    def wait_update_priorities(self):
        if self.synchronous:
            return self.prio_res
        else:
            if self.prio_res is None:
                return None
            return self.prio_res.get()

    def sample(self, batch_size, buffer_beta):
        if self.synchronous:
            return self.replay_buffer.sample(batch_size, buffer_beta)
        else:
            raise RuntimeError('function not available in asynchronous mode. Use async_sample()')

    def async_sample(self, batch_size, buffer_beta):
        if self.synchronous:
            self.sample_args = batch_size, buffer_beta
            return
        else:
            self.sample_res = self.pool.apply_async(worker_sample, [(batch_size, buffer_beta)])
        return

    def sample_available(self):
        if self.synchronous:
            return self.sample_args is not None
        return self.sample_res is not None

    def wait_sample(self):
        if self.synchronous:
            res = self.replay_buffer.sample(*self.sample_args)
            res['act'] = res['act'].transpose()[0].astype(int)
            res['rew'] = res['rew'].transpose()[0].astype(int)
            res['done'] = res['done'].transpose()[0].astype(int)
            self.sample_args = None
            return res
        else:
            if self.sample_res is None:
                return None
            res = self.sample_res.get()
            self.sample_res = None
            # Add shared memory arrays
            assert 'obs' not in res
            assert 'next_obs' not in res
            res['obs'] = self.state
            res['next_obs'] = self.next_state
            return res


