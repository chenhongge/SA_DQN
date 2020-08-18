from multiprocessing import Pool
import numpy as np
from PIL import Image
import random
import time
import pygame
import cv2
import sys
import os

ImageScale = 2

def worker_initializer(env_id, env_params, seed, save_frames=False):
    from common.wrappers import make_atari, make_atari_cart, wrap_deepmind, wrap_pytorch
    from setproctitle import setproctitle
    global env, return_unprocessed
    return_unprocessed = save_frames
    setproctitle('atari-env')
    if "NoFrameskip" not in env_id:
        env = make_atari_cart(env_id)
    else:
        env = make_atari(env_id)
        env = wrap_deepmind(env, **env_params)
        env = wrap_pytorch(env)
    random.seed(seed)
    seed = random.randint(0, sys.maxsize)
    print('reseting env with seed', seed, 'in initializer')
    env.seed(seed)
    state = env.reset()
    env.seed(seed)
    print('state shape', state.shape)


def recorder_initializer(path):
    from setproctitle import setproctitle
    setproctitle('atari-recorder')
    global recorder
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    recorder = cv2.VideoWriter(os.path.join(path, 'record.avi'),fourcc, 60.0, (84,84), isColor=True)


def display_initializer():
    global screen, font, timer, last_reward, last_done
    from setproctitle import setproctitle
    setproctitle('atari-display')
    # enlarge image 2 times
    width = 84 * ImageScale
    height = 84 * ImageScale
    try:
        pygame.display.init()
        pygame.font.init()
        pygame.display.set_caption('game')
        font = pygame.font.SysFont('dejavusansmono', 13)
        screen = pygame.display.set_mode((width, height))
        screen.fill((255, 255, 255))
        pygame.display.update()
    except Exception as e:
        print(e)
        time.sleep(120)
        raise e
    timer = time.time(), 0
    last_reward = [0, timer[0]]
    last_done = [0, timer[0]]

def worker_step(action):
    global env, return_unprocessed
    next_state, reward, done, info = env.step(action)
    if return_unprocessed:
        return next_state, reward, done, info, env.unprocessed_frame
    else:
        return next_state, reward, done, info

def display_step(pixels, action, reward, epi_reward, done, info):
    try:
        global screen, font, timer, last_reward, last_done
        if pixels.ndim == 3:
            pixels = np.transpose(pixels, (1, 2, 0))
        # enlarge image by 2X
        pixels = np.repeat(pixels, ImageScale, axis=0)
        pixels = np.repeat(pixels, ImageScale, axis=1)
        # create gray image with extra dimension
        if pixels.ndim == 2:
            pixels = np.stack((pixels, pixels, pixels), axis=-1)
        pygame.surfarray.blit_array(screen, pixels)
        # compute fps
        start_time, Nframes = timer
        Nframes += 1
        period = time.time() - start_time
        fps = Nframes / period
        if period > 1.0:
            start_time = time.time()
            Nframes = 0
        timer = start_time, Nframes
        # show information
        text_surface = font.render('fps={:6.2f} action={}'.format(fps, action), False, (0,0,255))
        screen.blit(text_surface, (0,0))
        text_surface = font.render('life={} epi_r={}'.format(info["ale.lives"], epi_reward), False, (0,255,0))
        screen.blit(text_surface, (0,20))
        def fade_text(text, cur_val, last_val, show_value=True):
            if not cur_val:
                cur_val = last_val[0]
                alpha = int(255 * max(0.75 - (time.time() - last_val[1]), 0) / 0.75)
            else:
                last_val[0] = cur_val
                last_val[1] = time.time()
                alpha = 255
            if show_value:
                if cur_val > 0:
                    text_surface = font.render("{}={}".format(text, cur_val), False, (0,255,0))
                else:
                    text_surface = font.render("{}={}".format(text, cur_val), False, (255,0,0))
            else:
                text_surface = font.render("{}".format(text), False, (255,0,0))
            text_surface.set_alpha(alpha)
            return text_surface
        text_surface = fade_text("done", done, last_done, False)
        screen.blit(text_surface, (0,40))
        text_surface = fade_text("reward", reward, last_reward, True)
        screen.blit(text_surface, (50,40))
        # update display
        pygame.display.flip()
    except Exception as e:
        # Avoid too many error messages when X is disconnected
        print('Error during displaying game play:')
        print(e)
        import traceback
        traceback.print_exc()
        print('pixels shape is', pixels.shape)
        time.sleep(120)


def recorder_step(pixels, action, reward, epi_reward, done, info):
    global recorder
    if pixels.ndim == 3:
        pixels = np.transpose(pixels, (2, 1, 0))
        # RGB to BGR
        pixels = pixels[:,:,::-1]
    # create gray image with extra dimension
    if pixels.ndim == 2:
        pixels = np.transpose(pixels)
        pixels = np.stack((pixels, pixels, pixels), axis=-1)
    pixels = np.ascontiguousarray(pixels)
    cv2.putText(pixels, "{} {}".format(action, int(reward)), (0,15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0,0))
    cv2.putText(pixels, "{} {}".format(int(epi_reward), "D" if done else ""), (0,30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0,0))
    recorder.write(pixels)


def worker_reset():
    global env
    return env.reset()

def worker_seed(seed):
    global env
    return env.seed(seed)


class AsyncEnv(object):
    def __init__(self, env_id, result_path, draw=False, record=False, save_frames=False, env_params=None, seed=2019):
        self.env_id = env_id
        if env_params is None:
            env_params = {}
        self.env_params = env_params
        self.pool = Pool(processes=1, initializer=worker_initializer, initargs=(env_id, env_params, seed, save_frames))
        self.draw = draw
        self.record = record
        self.res = None
        self.result_path = result_path
        self.save_frames = save_frames
        self.episode_counter = 0
        self.steps_counter = 0
        if draw:
            self.display_pool = Pool(processes=1, initializer=display_initializer)
            self.draw_res = None
            self.last_reward = None
            self.last_done = None
            self.epi_reward = 0
        if record:
            self.record_pool = Pool(processes=1, initializer=recorder_initializer, initargs=(result_path,))
        if save_frames:
            os.makedirs(os.path.join(result_path, "frames/000"), exist_ok=True)

    def async_step(self, action):
        self.action = action
        self.res = self.pool.apply_async(worker_step, (action,))

    def wait_step(self):
        if self.save_frames:
            next_state, reward, done, info, unprocessed_frame = self.res.get()
            im = Image.fromarray(unprocessed_frame)
            im.save(os.path.join(self.result_path, "frames", "{:03d}".format(self.episode_counter), '{:05d}.bmp'.format(self.steps_counter)))
        else:
            next_state, reward, done, info = self.res.get()
        res = next_state, reward, done, info
        if self.draw or self.record:
            self.epi_reward += reward
            if next_state.shape[0] == 1:
                pixels = next_state[0]
            elif next_state.shape[0] == 4:
                # Frame stack + gray image
                pixels = next_state[0]
            elif next_state.shape[0] > 4:
                # Frame stack + colored image
                pixels = next_state[0:3]
            else:
                pixels = next_state
        if self.draw:
            if self.draw_res is None or self.draw_res.ready():
                # Don't miss any reward or done event even we missed that frame
                if reward == 0 and self.last_reward is not None:
                    reward = self.last_reward
                    self.last_reward = None
                if not done and self.last_done is not None:
                    done = self.last_done
                    self.last_done = None
                self.draw_res = self.display_pool.apply_async(display_step, (pixels, self.action, reward, self.epi_reward, done, info))
            else:
                # This frame was skipped, but we save its reward and done states if they are non-zero
                if reward != 0:
                    self.last_reward = reward
                if done:
                    self.last_done = done
            if done:
                self.epi_reward = 0
        if self.record:
            self.record_pool.apply_async(recorder_step, (pixels, self.action, reward, self.epi_reward, done, info))
        self.steps_counter += 1
        return res

    def reset(self):
        self.episode_counter += 1
        self.steps_counter = 0
        if self.save_frames:
            os.makedirs(os.path.join(self.result_path, "frames", "{:03d}".format(self.episode_counter)), exist_ok=True)
        return self.pool.apply(worker_reset)

    def seed(self, seed):
        return self.pool.apply(worker_seed, (seed, ))

