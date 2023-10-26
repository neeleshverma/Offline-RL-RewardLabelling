from typing import Any, Iterator, Optional

from absl import logging
from acme import types
from acme import wrappers
import d4rl
import gym
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
import tree


def load_dataset(env):
    dataset = d4rl.qlearning_dataset(env)
    return types.Transition(
        observation = dataset["observations"],
        action = dataset["actions"],
        reward = dataset["rewards"],
        next_observation = dataset["next_observations"],
        discount = 1.0 - dataset["terminals"].astype(np.float32),
    )

# Create a new environment
def create_environment(env_name, seed = None):
    environment = gym.make(env_name)
    if seed is not None:
        environment.seed(seed)
    environment = wrappers.GymWrapper(environment)
    environment = wrappers.SinglePrecisionWrapper(environment)
    environment = wrappers.CanonicalSpecWrapper(environment)
    return environment


def make_trajectories_list(observations, actions, rewards, masks, dones_float, next_observations):
    trajs = [[]]

    for i in tqdm.tqdm(range(len(observations))):
        trajs[-1].append(
            types.Transition(
                observation=observations[i],
                action=actions[i],
                reward=rewards[i],
                discount=masks[i],
                next_observation=next_observations[i])
                )
        if dones_float[i] == 1.0 and i + 1 < len(observations):
            trajs.append([])

    return trajs

# Merge trajectories to make an array
def merge_trajectories(trajs):
    flat = []
    for traj in trajs:
        for transition in traj:
            flat.append(transition)
    return tree.map_structure(lambda *xs: np.stack(xs), *flat)


# Episode reached the maximum running time
def timeout_episodes(env, dataset=None, terminate_on_end=False, disable_goal=True, **kwargs):
    if dataset is None:
        dataset = env.get_dataset(**kwargs)

    N = dataset['rewards'].shape[0]
    observation_list = []
    next_observation_list = []
    actions_list = []
    rewards_list = []
    terminals_list = []
    terminate_state_list = []
    if "infos/goal" in dataset:
        if not disable_goal:
            dataset["observations"] = np.concatenate([dataset["observations"], dataset['infos/goal']], axis=1)

    episode_step = 0
    for i in range(N - 1):
        observation = dataset['observations'][i]
        next_observation = dataset['observations'][i + 1]
        action = dataset['actions'][i]
        reward = dataset['rewards'][i]
        terminal = bool(dataset['terminals'][i])
        terminate_state = bool(dataset['terminals'][i])

        if "infos/goal" in dataset:
            final_timestep = True if (dataset['infos/goal'][i] != dataset['infos/goal'][i + 1]).any() else False
        else:
            final_timestep = dataset['timeouts'][i]

        if i < N - 1:
            terminal += final_timestep

        if (not terminate_on_end) and final_timestep:
            episode_step = 0
            continue
        if terminal or final_timestep:
            episode_step = 0

        observation_list.append(observation)
        next_observation_list.append(next_observation)
        actions_list.append(action)
        rewards_list.append(reward)
        terminals_list.append(terminal)
        terminate_state_list.append(terminate_state)
        episode_step += 1

    return {
        'observations': np.array(observation_list),
        'actions': np.array(actions_list),
        'next_observations': np.array(next_observation_list),
        'rewards': np.array(rewards_list)[:],
        'terminals': np.array(terminals_list)[:],
        'realterminals': np.array(terminate_state)[:],
    }


def get_trajectories(dataset_name, fix_antmaze_timeout=True):
    env = gym.make(dataset_name)
    if "antmaze" in dataset_name and fix_antmaze_timeout:
        dataset = timeout_episodes(env)
    else:
        dataset = d4rl.qlearning_dataset(env)
    
    dones_float = np.zeros_like(dataset['rewards'])

    for i in range(len(dones_float) - 1):
        if np.linalg.norm(dataset['observations'][i + 1] - dataset['next_observations'][i]) > 1e-6 or dataset['terminals'][i] == 1.0:
            dones_float[i] = 1
        else:
            dones_float[i] = 0
    dones_float[-1] = 1

    if 'realterminals' in dataset:
        masks = 1.0 - dataset['realterminals'].astype(np.float32)
    else:
        masks = 1.0 - dataset['terminals'].astype(np.float32)
        traj = make_trajectories_list(
            observations=dataset['observations'].astype(np.float32),
            actions=dataset['actions'].astype(np.float32),
            rewards=dataset['rewards'].astype(np.float32),
            masks=masks,
            dones_float=dones_float.astype(np.float32),
            next_observations=dataset['next_observations'].astype(np.float32)
            )
    return traj


def load_dataset_as_demo(name: str, num_top_episodes: int = 10):
    trajs = get_trajectories(name)
    if num_top_episodes < 0:
        logging.info("Loading the entire dataset as demonstrations")
        return trajs

    def compute_returns(traj):
        episode_return = 0
        for transition in traj:
            episode_return += transition.reward
        return episode_return

    trajs.sort(key=compute_returns)
    return trajs[-num_top_episodes:]


class JaxInMemorySampler(Iterator[Any]):

    def __init__(self, dataset, key, batch_size):
        self._dataset_size = jax.tree_util.tree_leaves(dataset)[0].shape[0]
        self._jax_dataset = jax.tree_map(jax.device_put, dataset)

        def sample(data, key: jnp.ndarray):
            key1, key2 = jax.random.split(key)
            indices = jax.random.randint(
                key1, (batch_size,), minval=0, maxval=self._dataset_size)
            data_sample = jax.tree_map(lambda d: jnp.take(d, indices, axis=0), data)
            return data_sample, key2

        self._sample = jax.jit(lambda key: sample(self._jax_dataset, key))
        self._key = key

    def __next__(self) -> Any:
        data, self._key = self._sample(self._key)
        return data
