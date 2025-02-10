# tdmpc_square_public

[Paper](https://arxiv.org/abs/2502.03550) [Website](https://darthutopian.github.io/tdmpc_square/)

We present <u>T</u>emporal <u>D</u>ifference Learning for <u>M</u>odel <u>P</u>redictive <u>C</u>ontrol with <u>P</u>olicy <u>C</u>onstraint (<strong>TD-M(PC)\(^2\)</strong>), a simple yet effective approach built on TD-MPC2 that allows a planning-based MBRL algorithm to better exploit complete off-policy data. This repo is built on top of <a href="https://humanoid-bench.github.io" target="_blank"><code>HumanoidBench</code></a>, it contains example code to implement TD-M(PC)^2 and reproduce results in the paper.

![image](humanoid_bench.jpg)

## Remark
This is a customized repository for MBRL development. One can enforce modifications in the paper 


## Directories
Structure of the repository:
* `data`: Weights of the low-level skill policies
* `dreamerv3`: Training code for dreamerv3
* `humanoid_bench`: Core benchmark code
    * `assets`: Simulation assets
    * `envs`: Environment files
    * `mjx`: MuJoCo MJX training code
* `jaxrl_m`: Training code for SAC
* `ppo`: Training code for PPO
* `tdmpc2`: Training code for TD-MPC2
* `tdmpc_square`: Training code for TD-MPC-square

## Installation
Create a clean conda environment:
```
conda create -n tdmpc-square python=3.11
conda activate tdmpc-square
```

Then, install the required packages:
```
# Install HumanoidBench
pip install -e .

# jax GPU version
pip install "jax[cuda12]==0.4.28"
# Or, jax CPU version
pip install "jax[cpu]==0.4.28"

# Install tdmpc-square, minimal requirement
pip install -r requirements_tdmpc_square.txt
```

To run the baselines:
```
# Install jaxrl
pip install -r requirements_jaxrl.txt

# Install dreamer
pip install -r requirements_dreamer.txt
```


## HumanoidBench Env Test
This section is to check

### Test Environments with Random Actions
```
python -m humanoid_bench.test_env --env h1hand-walk-v0
```

### Test Environments with Hierarchical Policy and Random Actions
```
# Define checkpoints to pre-trained low-level policy and obs normalization
export POLICY_PATH="data/reach_two_hands/torch_model.pt"
export MEAN_PATH="data/reach_two_hands/mean.npy"
export VAR_PATH="data/reach_two_hands/var.npy"

# Test the environment
python -m humanoid_bench.test_env --env h1hand-push-v0 --policy_path ${POLICY_PATH} --mean_path ${MEAN_PATH} --var_path ${VAR_PATH} --policy_type "reach_double_relative"
```

### Test Low-Level Reaching Policy (trained with MJX, testing on classical MuJoCo)
```
# One-hand reaching
python -m humanoid_bench.mjx.mjx_test --with_full_model 

# Two-hand reaching
python -m humanoid_bench.mjx.mjx_test --with_full_model --task=reach_two_hands --folder=./data/reach_two_hands
```

### Change Observations
As a default, the environment returns a privileged state of the environment (e.g., robot state + environment state). To get proprio, visual, and tactile sensing, set `obs_wrapper=True` and accordingly select the required sensors, e.g. `sensors="proprio,image,tactile"`. When using tactile sensing, make sure to use `h1touch` in place of `h1hand`.
Full test instruction:
```
python -m humanoid_bench.test_env --env h1touch-stand-v0 --obs_wrapper True --sensors "proprio,image,tactile"
```

## Training
```
# Define TASK
export TASK="h1hand-sit_simple-v0"

# Train TD-MPC2
python -m tdmpc_square.train disable_wandb=False wandb_entity=[WANDB_ENTITY] exp_name=tdmpc task=humanoid_${TASK} seed=0
```

### Baseline 

For TD-MPC, one can set actor_mode to 'sac' in `tdmpc_square/config.yaml`.
```
# Train DreamerV3
python -m embodied.agents.dreamerv3.train --configs humanoid_benchmark --run.wandb True --run.wandb_entity [WANDB_ENTITY] --method dreamer --logdir logs --task humanoid_${TASK} --seed 0

# Train SAC
python ./jaxrl_m/examples/mujoco/run_mujoco_sac.py --env_name ${TASK} --wandb_entity [WANDB_ENTITY] --seed 0

# Train PPO (not using MJX)
python ./ppo/run_sb3_ppo.py --env_name ${TASK} --wandb_entity [WANDB_ENTITY] --seed 0
```


Remarks: For Humanoid-Bench only consider handed tasks (high-dimentional under-actuated). For DMC suite experiments, please following README instruction `tdmpc2/README.md`.
Please ENABLE wandb log, however wandb_entity is not required in configuration. Run the following commandlines to set off training:
```
CUDA_VISIBLE_DEVICES=1 python -m tdmpc_square.train disable_wandb=False wandb_entity=summer_research exp_name=public task=humanoid_h1hand-slide-v0 seed=1
CUDA_VISIBLE_DEVICES=1 python -m tdmpc2.train disable_wandb=False wandb_entity=summer_research exp_name=residual_actdim task=dog-stand seed=3
```


## Citation
```
@article{lin2025tdmpc2improve,
            title={TD-M(PC)$^2$: Improving Temporal Difference MPC Through Policy Constraint}, 
            author={Haotian Lin and Pengcheng Wang and Jeff Schneider and Guanya Shi},
            journal={arXiv preprint arXiv:2502.03550},
            year={2025}, 
        }
```


## References
This codebase contains some files adapted from other sources:
* humanoidbench: https://github.com/carlosferrazza/humanoid-bench/tree/main
* jaxrl_m: https://github.com/dibyaghosh/jaxrl_m/tree/main
* DreamerV3: https://github.com/danijar/dreamerv3
* TD-MPC2: https://github.com/nicklashansen/tdmpc2
* purejaxrl (JAX-PPO traning): https://github.com/luchris429/purejaxrl/tree/main