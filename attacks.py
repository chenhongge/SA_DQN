import torch
from torch import autograd
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import numpy as np
from utils import arctanh_rescale, tanh_rescale, to_one_hot
TARGET_MULT = 10000.0

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

CARTPOLE_STD=[0.7322321, 1.0629482, 0.12236707, 0.43851405]
ACROBOT_STD=[0.36641926, 0.65119815, 0.6835106, 0.67652863, 2.0165246, 3.0202584]



def pgd(model, X, y, verbose=False, params={}, env_id=""):
    epsilon = params.get('epsilon', 0.00392)
    if env_id == "CartPole-v0":
        epsilon = torch.from_numpy(CARTPOLE_STD) * epsilon
    if env_id == "Acrobot-v1":
        epsilon = torch.from_numpy(ACROBOT_STD) * epsilon
    niters = params.get('niters', 4)
    img_min = params.get('img_min', 0.0)
    img_max = params.get('img_max', 1.0)
    loss_func = params.get('loss_func', nn.CrossEntropyLoss())
    step_size = epsilon * 1.0 / niters
    y = Variable(torch.tensor(y))
    if verbose:
        print('epislon: {}, step size: {}, target label: {}'.format(epsilon, step_size, y))
    rand = params.get('random_start', True)
    if rand:
        noise = 2 * epsilon * torch.rand(X.data.size()) - epsilon
        if USE_CUDA:
            noise = noise.cuda()
        X_adv = torch.clamp(X.data + noise, img_min, img_max)
        X_adv = Variable(X_adv.data, requires_grad=True)
        if verbose:
            print('linf diff after adding noise: ', np.max(abs(X_adv.data.cpu().numpy()-X.data.cpu().numpy())))
    else:
        X_adv = Variable(X.data, requires_grad=True)
    for i in range(niters):
        logits = model.forward(X_adv)
        loss = loss_func(logits, y)
        if verbose:
            print('current loss: ', loss.data.cpu().numpy())
        model.features.zero_grad()
        loss.backward()
        eta = step_size * X_adv.grad.data.sign()
        X_adv = Variable(X_adv.data + eta, requires_grad=True)
        # adjust to be within [-epsilon, epsilon]
        if env_id == "CartPole-v0" or env_id == "Acrobot-v1":
            eta = torch.max(X_adv.data-X.data, -epsilon)
            eta = torch.min(eta, epsilon)
        else:
            eta = torch.clamp(X_adv.data - X.data, -epsilon, epsilon)
        X_adv.data = X.data + eta
        if verbose:
            print('max eta: ', np.max(abs(eta.data.cpu().numpy())))
            print('linf diff before clamp: ', np.max(abs(X_adv.data.cpu().numpy()-X.data.cpu().numpy())))
        X_adv.data = torch.clamp(X_adv.data, img_min, img_max)
        if verbose:
            print('linf diff after clamp: ',np.max(abs(X_adv.data.cpu().numpy()-X.data.cpu().numpy())))
    if verbose:
        print('{} iterations'.format(i+1))
    return X_adv.data


def fgsm(model, X, y, verbose=False, params={}):
    epsilon=params.get('epsilon', 1)
    img_min=params.get('img_min', 0.0)
    img_max=params.get('img_max', 1.0)
    X_adv = Variable(X.data, requires_grad=True)
    logits = model.forward(X_adv)
    loss = F.nll_loss(logits, y)
    model.features.zero_grad()
    loss.backward()
    eta = epsilon*X_adv.grad.data.sign()
    X_adv = Variable(X_adv.data + eta, requires_grad=True)
    X_adv.data = torch.clamp(X_adv.data, img_min, img_max)
    return X_adv.data


def cw(model, X, y, verbose=False, params={}):
    C=params.get('C', 0.0001)
    niters=params.get('niters', 50)
    step_size=params.get('step_size', 0.01)
    confidence=params.get('confidence', 0.0001)
    img_min=params.get('img_min', 0.0)
    img_max=params.get('img_max', 1.0)
    Xt = arctanh_rescale(X, img_min, img_max)
    Xt_adv = Variable(Xt.data, requires_grad=True)
    y_onehot = to_one_hot(y, model.num_actions).float()
    optimizer = optim.Adam([Xt_adv], lr=step_size)
    for i in range(niters):
        logits = model.forward(tanh_rescale(Xt_adv, img_min, img_max))
        real = (y_onehot * logits).sum(dim=1)
        other = ((1.0 - y_onehot) * logits - (y_onehot * TARGET_MULT)).max(1)[0]
        loss1 = torch.clamp(real - other + confidence, min=0.)
        loss2 = torch.sum((X - tanh_rescale(Xt_adv, img_min, img_max)).pow(2), dim=[1,2,3])
        loss = loss1 + loss2 * C

        optimizer.zero_grad()
        model.features.zero_grad()
        loss.backward()
        optimizer.step()
        # if verbose:
        #    print('loss1: {}, loss2: {}'.format(loss1, loss2))
    return tanh_rescale(Xt_adv, img_min, img_max).data


def attack(model, X, attack_config, loss_func=nn.CrossEntropyLoss()):
    method = attack_config.get('method', 'pgd')
    verbose = attack_config.get('verbose', False)
    params = attack_config.get('params', {})
    params['loss_func'] = loss_func
    y = model.act(X)
    if method == 'cw':
        atk = cw
    elif method == 'fgsm':
        atk = fgsm
    else:
        atk = pgd
    adv_X = atk(model, X, y, verbose=verbose, params=params)
    abs_diff = abs(adv_X.cpu().numpy()-X.cpu().numpy())
    if verbose:
        print('adv image range: {}-{}, ori action: {}, adv action: {}, l1 norm: {}, l2 norm: {}, linf norm: {}'.format(torch.min(adv_X).cpu().numpy(), torch.max(adv_X).cpu().numpy(), model.act(X)[0], model.act(adv_X)[0], np.sum(abs_diff), np.linalg.norm(abs_diff), np.max(abs_diff)))
    return adv_X

