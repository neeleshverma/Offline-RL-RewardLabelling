# Offline RL Reward Labelling using Optimal Transport Distances
The underlying algorithm is IQL. The code for IQL is taken from https://github.com/ikostrikov/implicit_q_learning .

The code can be simply run as -

```
python train.py --workdir /tmp/ot \
        --config configs/mujoco.py \
        --config.expert_dataset_name='hopper-medium-v2' \
        --config.k=1 \
        --config.offline_dataset_name='hopper-medium-v2' \
        --config.use_dataset_reward=True
```

(There might be some code issues).

Here, k is the number of expert trajectories.

For solving the Optimal Transport problem, JAX implementation is used https://github.com/ott-jax/ott  

The project report is available at [Report](https://github.com/neeleshverma/Offline-RL-RewardLabelling/blob/main/report/CSE_525_Project_Report.pdf)

Contact Info: neverma@cs.stonybrook.edu
