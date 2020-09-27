# SA-DQN Reference Implementation: State-adversarial DQN for robust deep reinforcement learning

This repository contains a reference implementation for State-Adversarial Deep
Q Networks (SA-DQN).  SA-DQN includes a theoretically principled (based on
SA-MDP) regularizer to obtain a DQN agent robust to noises and adversarial
perturbations on state observations. See our paper ["Robust Deep Reinforcement
Learning against Adversarial Perturbations on State
Observations"](https://arxiv.org/pdf/2003.08938) for more details. This paper has been accepted by **NeurIPS 2020 ** as a **spotlight** presentation.

Our DQN implementation is mostly a proof of concept, and does not include many
advanced training techniques like Rainbow or C51. It is based on an implement of [RL-Adventure](https://github.com/higgsfield/RL-Adventure) We use
[auto_LiRPA](https://github.com/KaidiXu/auto_LiRPA). as a sub-module to compute
convex relaxations of neural networks (which supports forward, backward mode
perturbation analysis and interval bound propagation (IBP)).

If you are looking for robust policy gradient algorithms for continous action
space environments (e.g., agents for MuJoCo control tasks) please see our
[SA-PPO repository](https://github.com/huanzhang12/SA_PPO).

## SA-DQN Demo

We attack vanilla DQN and SA-DQN agents on 4 Atari games under untargeted
projected gradient decent (PGD) attacks on the state observations (pixel inputs
to the Q network). SA-DQN (right figure of each pair) achieves high rewards under
attacks, where vanilla DQN has very low rewards.

| ![Pong-attack-natural.gif](/gifs/Pong-attack-natural.gif) | ![Pong-attack-natural.gif](/gifs/Pong-attack-convex.gif) | ![RoadRunner-attack-natural.gif](/gifs/RoadRunner-attack-natural.gif) | ![RoadRunner-attack-natural.gif](/gifs/RoadRunner-attack-convex.gif) | 
|:--:| :--:| :--:| :--:| 
| **Pong**, *Vanilla DQN* <br> reward under attack: **-21** <br> (trained agent: right paddle) | **Pong**, *SA-DQN* <br> reward under attack: **21** <br> (trained agent: right paddle) |**RoadRunner**, *Vanilla DQN* <br> reward under attack: **0** |**RoadRunner**, *SA-DQN* <br> reward under attack: **49900** |
| ![Freeway-attack-natural.gif](/gifs/Freeway-attack-natural.gif) | ![Freeway-attack-natural.gif](/gifs/Freeway-attack-convex.gif) | ![BankHeist-attack-natural.gif](/gifs/BankHeist-attack-natural.gif) | ![BankHeist-attack-natural.gif](/gifs/BankHeist-attack-convex.gif) | 
| **Freeway**, *Vanilla DQN* <br> reward under attack: **0** <br> (trained agent: left chicken) | **Freeway**, *SA-DQN* <br> reward under attack: **30** <br> (trained agent: left chicken) | **BankHeist**, *Vanilla DQN* <br> reward under attack: **0** | **BankHeist**,  *SA-DQN* <br> reward under attack: **1240** |


## Setup

First clone this repository and install necessary Python packages.
```bash
git submodule update --init
pip install -r requirements.txt
```

## Pretrained models

We release pretrained models for four Atari environments. Note that due to
limited computing resource, we train all these models for only **6 million
frames in total** (for both vanilla DQN and SA-DQN; **SA-DQN requires no
additional training frames**), where most DQN papers train them for 20 million
frames.  Also, our implementation is basic and do not include many tricks in
more advanced DQN training methods (such as Rainbow and C51). It is possible to
further improve SA-DQN performance by using more advanced training baselines or
train more frames.

We list the performance of our pretrained models in the Table below.

| Environment | Evaluation         | Vanilla DQN | SA-DQN (convex relaxation) |
|-------------|--------------------|:-----------:|:--------------------------:|
| Pong        | No attack          |   21.0±0.0  |         21.0±0.0           |
|             | PGD 10-step attack |  -21.0±0.0  |       **21.0±0.0**         |
|             | PGD 50-step attack |  -21.0±0.0  |       **21.0±0.0**         |
| RoadRunner  | No attack          |  45534.0±7066.0  |   44638.0±7367.0      |
|             | PGD 10-step attack |   0.0±0.0   |     **44732.0±8059.5**     |
|             | PGD 50-step attack |   0.0±0.0   |     **44678.0±6954.0**     |
| BankHeist   | No attack          |  1308.4±24.1|      1235.4±9.8            |
|             | PGD 10-step attack |   56.4±21.2 |       **1232.4±16.2**      | 
|             | PGD 50-step attack |   31.0±32.6 |       **1234.6±16.6**      |
| Freeway     | No attack          |   34.0±0.2  |         30.0±0.0           |
|             | PGD 10-step attack |   0.0±0.0   |        **30.0±0.0**        |
|             | PGD 50-step attack |   0.0±0.0   |        **30.0±0.0**        |


Our pretrained models can be obtained
[here](http://download.huan-zhang.com/models/robust-drl/dqn/sa-dqn-models.tar.gz).
Decompress the models after downloading them:

```bash
wget http://download.huan-zhang.com/models/robust-drl/dqn/sa-dqn-models.tar.gz
tar xvf sa-dqn-models.tar.gz
```

In the `models` directory, we provided a few pretrain models which can be
evaluated using `test.py`. For example, to test the RoadRunner pretrained
SA-DQN model, simply run:

```bash
# No attacks
python test.py --config config/RoadRunner_cov.json test_config:load_model_path=models/RoadRunner-convex.model
# 10-step PGD attack
python test.py --config config/RoadRunner_cov.json test_config:load_model_path=models/RoadRunner-convex.model test_config:attack=true
# 50-step PGD attack
python test.py --config config/RoadRunner_cov.json test_config:load_model_path=models/RoadRunner-convex.model test_config:attack=true test_config:attack_config:params:niters=50
```

## Training

We provide configuration files for vanilla DQN training and robust DQN training in the `config/` folder.

To train a model, simply run

```bash
python train.py --config <config file>
```

There are some config files in `config/`.
* Files with suffix `_nat` are for vanilla DQN models.
* Files with suffix `_cov` are for SA-DQN trained by convex relaxation.
* Files with suffix `_pgd` are for SA-DQN trained by PGD.
* Files with suffix `_adv` are for DQN with adversarial training ([Behzadan and Munir](https://arxiv.org/pdf/1712.09344.pdf)).

For example, to train an SA-DQN with convex relaxation on the Pong environment, simply run

```bash
python train.py --config config/Pong_cov.json
```

**Tips on training:** DQN for Atari games are generally tricky to train and
converge, especially for complicated environments like RoadRunner. Especially,
we train for only **6 million frames** due to limited computation resources,
and our implementation is only a proof of concept and does not include many
training tricks. For training new environments, we recommend that first finding
a set of hyperparameters that works well for vanilla DQN training, and then try
SA-DQN training. Generally, the only necessary parameters for tuning SA-DQN is
`kappa` (regularization parameters), where we searched among {0.005, 0.01,
0.02} in our experiments. It is possible to further improve SA-DQN performance
by using more advanced training baselines or train more frames.

Additionally, we use [auto_lirpa](https://github.com/KaidiXu/auto_LiRPA) which
provides a wide of range of possibilities for the convex relaxation based
method, including forward and backward mode bound analysis, and interval bound
propagation (IBP).  In our paper, we only evaluated the IBP+backward scheme,
which is known to be efficient and stable. We did not report results using
other possible relaxations as this is not our main focus. If you are interested
in trying other relaxation schemes, e.g., if you want to use the cheapest IBP
methods (at the cost of potential stability issue), you can try this:

```bash
# The below training should run faster (per frame) than the original due to the use of cheaper relaxations.
# However, you probably need to train more frames to compensate the instability of IBP.
python train.py --config config/Pong_cov.json training_config:convex_start_beta=0.0
```

## Performance Evaluation

### If you want to test the best model from your training directory

After training, models will be saved to folders with the environment name as
prefix. Run the following command to test the model's performance (it will
automatically load the best checkpoint, if available):

```bash
python test.py --config config/Pong_cov.json
```

To apply PGD attacks, run this command:

```bash
python test.py --config config/Pong_cov.json test_config:attack=true
```

To run more steps of PGD attacks (e.g., 50 steps PGD attack):

```bash
python test.py --config config/Pong_cov.json test_config:attack=true test_config:attack_config:params:niters=50
```

If you want to test the natural reward together with action certification rate, do


```bash
python test.py --config config/Pong_cov.json test_config:certify=true
```

### If you want to test a specific model

If you would like to test a specific model (for example, our pretrained model), you can give
the model path as `test_config:load_model_path=`, for example:

```bash
python test.py --config config/RoadRunner_cov.json test_config:load_model_path=models/RoadRunner-convex.model
```

This will create a directory with with the environment name as
prefix and save a ```test.log``` in it.

