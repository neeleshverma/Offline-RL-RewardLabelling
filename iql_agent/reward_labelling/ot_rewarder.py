from typing import Any, Callable, Optional, Protocol, Sequence

from acme import types
from acme.jax import networks as networks_lib
from acme.jax import running_statistics
from acme.jax import variable_utils
import chex
import jax
import jax.numpy as jnp
import numpy as onp
import ott
from ott.core import sinkhorn
from ott.geometry import pointcloud
import ott.geometry.costs

EncoderFn = Callable[[networks_lib.Params, networks_lib.Observation],
                     types.NestedArray]
PreprocessorState = Any


class Preprocessor(Protocol):
  """ Extracts state-level information from the observations
  """

  def init(self):
    ...

  def update(self, state: PreprocessorState, atoms) -> PreprocessorState:
    ...

  def preprocess(self, params: networks_lib.Params, state: PreprocessorState,
                 atoms):
    ...


class EncoderPreprocessor(Preprocessor):
  def __init__(self, encoder_fn: EncoderFn):
    self._encoder_fn = encoder_fn

  def init(self):
    return ()

  def update(self, state, atoms):
    del state, atoms
    return ()

  def preprocess(self, params, state, atoms):
    del state
    return self._encoder_fn(params, atoms)


class StatsPreprocessor(Preprocessor):
  """Computing different stats, currently mean and std are implemented"""

  def __init__(self, spec: types.NestedArray, partial_update: bool = False):
    self._observation_spec = spec
    self._partial_update = partial_update

  def init(self):
    return running_statistics.init_state(self._observation_spec)

  def update(self, state, atoms):
    assert atoms.ndim == 2
    if self._partial_update:
      state = running_statistics.init_state(self._observation_spec)
      state = running_statistics.update(state, atoms)
      return state
    else:
      state = running_statistics.update(state, atoms)
      return state

  def preprocess(self, params, state, atoms):
    del params
    return running_statistics.normalize(atoms, state)


class DefaultPreprocessor(Preprocessor):
  """Used when none other preprocessor is not being employed"""

  def init(self):
    return ()

  def update(self, state, atoms):
    del state, atoms
    return ()

  def preprocess(self, params, state, atoms):
    del params, state
    return atoms


# TODO - Not working currently
class AggregateFn(Protocol):
  """When multiple exerpt trajectories are available instead of single"""

  def __call__(self, rewards: chex.Array, **kwargs) -> chex.Array:
    ...


# Aggregate the rewards using top K expert demos
def aggregate_top_k(rewards, k=1):
  scores = jnp.sum(rewards, axis=-1)
  _, indices = jax.lax.top_k(scores, k=k)
  return jnp.mean(rewards[indices], axis=0)


# Simply taking the mean of expert demos
def aggregate_mean(rewards):
  """Aggregate rewards from multiple expert demonstrations by taking the mean"""
  return jnp.mean(rewards, axis=0)


# Squash the computed rewards (for stable training)
class SquashingFn(Protocol):
  def __call__(self, rewards: chex.Array, **kwargs) -> chex.Array:
    ...

# Linear Squashing (alpha * rewards)
def squashing_linear(rewards, alpha: float = 10.):
  return alpha * rewards

# Exponential Squashing (more useful)
def squashing_exponential(rewards, alpha: float = 5., beta: float = 5.):
  return alpha * jnp.exp(beta * rewards)


class OptimalTransportRewarder:
  """Main class which computes the optimal transport distance to
  the expert demo(s)
  """

  def __init__(self,
               demonstrations: Sequence[Sequence[types.Transition]],
               episode_length: int,
               preprocessor: Optional[Preprocessor] = None,
               aggregate_fn: AggregateFn = aggregate_top_k,
               squashing_fn: SquashingFn = squashing_linear,
               max_iterations: float = 100,
               threshold: float = 1e-9,
               epsilon: float = 1e-2,
               preprocessor_update_period: int = 1,
               use_actions_for_distance: bool = False,
               variable_client: Optional[variable_utils.VariableClient] = None):
    if use_actions_for_distance and preprocessor is not None:
      raise NotImplementedError(
          "Use actions with preprocessor not yet supported")

    self._episode_length = episode_length
    self._aggregate_fn = aggregate_fn
    self._squashing_fn = squashing_fn
    self._max_iterations = max_iterations
    self._threshold = threshold
    self._epsilon = epsilon
    self._preprocessor_update_period = preprocessor_update_period
    self._variable_client = variable_client
    self._params = variable_client.params if variable_client is not None else None
    self._num_episodes_seen = 0
    self._use_actions_for_distance = use_actions_for_distance

    # Prepare expert atoms
    self._expert_atoms = []
    self._expert_weights = []
    # Vectorize atoms, pad the atoms and compute weights
    for demo in demonstrations:
      atoms, weights, _, _ = get_trajectory_array(demo, self._episode_length,
                                              self._use_actions_for_distance)
      self._expert_atoms.append(atoms)
      self._expert_weights.append(weights)

    self._expert_atoms = onp.stack(self._expert_atoms)
    self._expert_weights = onp.stack(self._expert_weights)

    self._preprocessor = preprocessor or DefaultPreprocessor()
    self._preprocessor_state = self._preprocessor.init()

    self._batched_ot_solve = jax.jit(
        jax.vmap(self._solve_ot, in_axes=(None, None, 0, 0, None, None)))
    self._compute_rewards = jax.jit(self.get_ot_rewards)
    self._update_preprocessor = jax.jit(self._preprocessor.update)


  # OT solver (based on ott library based sinkhorn distance)
  def _solve_ot(self, params, state, expert_atoms, expert_weights, agent_atoms,
                agent_weights):
    agent_atoms = self._preprocessor.preprocess(params, state, agent_atoms)
    expert_atoms = self._preprocessor.preprocess(params, state, expert_atoms)
    cost_fn = ott.geometry.costs.Cosine()
    geom = pointcloud.PointCloud(
        agent_atoms, expert_atoms, cost_fn=cost_fn, epsilon=self._epsilon)
    sinkhorn_output = sinkhorn.sinkhorn(
        geom,
        a=agent_weights,
        b=expert_weights,
        threshold=self._threshold,
        max_iterations=self._max_iterations,
        jit=False)
    coupling_matrix = geom.transport_from_potentials(sinkhorn_output.f,
                                                     sinkhorn_output.g)
    cost_matrix = cost_fn.all_pairs(agent_atoms, expert_atoms)
    ot_costs = jnp.einsum('ij,ij->i', coupling_matrix, cost_matrix)
    return ot_costs
  

  # Compute the Optimal transport based rewards
  # We can assume that the reward is negative of the OT distance between given episode and expert demo
  def get_ot_rewards(self, params, preprocessor_state, all_expert_atoms,
                            all_expert_weights, agent_atoms, agent_weights,
                            agent_mask):
    ot_costs = self._batched_ot_solve(params, preprocessor_state,
                                      all_expert_atoms, all_expert_weights,
                                      agent_atoms, agent_weights)
    
    # Reward is simply negative of the OT distance
    pseudo_rewards = -ot_costs
    rewards = self._squashing_fn(pseudo_rewards)
    rewards = jnp.where(agent_mask, rewards, 0.)
    rewards = self._aggregate_fn(rewards)
    return rewards


  # Main function which given the number of agent steps, compute the OT based rewards
  def compute_offline_rewards(self, agent_steps):
    agent_atoms, agent_weights, num_agent_atoms, agent_mask = get_trajectory_array(
        agent_steps, self._episode_length, self._use_actions_for_distance)

    # Update preprocessor state
    if self._num_episodes_seen % self._preprocessor_update_period == 0:
      if self._variable_client is not None:
        self._variable_client.update_and_wait()
        self._params = self._variable_client.params

      self._preprocessor_state = self._update_preprocessor(
          self._preprocessor_state, agent_atoms[:num_agent_atoms])

    rewards = self.get_ot_rewards(self._params, self._preprocessor_state,
                                         self._expert_atoms,
                                         self._expert_weights, agent_atoms,
                                         agent_weights, agent_mask)
    rewards = rewards[:num_agent_atoms]

    self._num_episodes_seen += 1

    return jax.device_get(rewards)


# TODO - Merge this function in reward computation
# Convert observations inside the list in an array
def get_trajectory_array(trajectory: Sequence[types.Transition],
                     max_sequence_length: int,
                     use_actions: bool = False):
  num_atoms = len(trajectory)
  if use_actions:
    observations = [
        onp.concatenate([atom.observation, atom.action], axis=-1)
        for atom in trajectory
    ]
  else:
    observations = [atom.observation for atom in trajectory]

  atoms = onp.stack(observations, axis=0)
  atoms = padding(atoms, max_sequence_length)
  weights = onp.ones((num_atoms,)) / max_sequence_length
  weights = padding(weights, max_sequence_length)
  mask = padding(onp.ones(num_atoms, dtype=bool), max_sequence_length)
  return atoms, weights, num_atoms, mask


def padding(x, max_sequence_length: int):
  paddings = [(0, max_sequence_length - x.shape[0])]
  paddings.extend([(0, 0) for _ in range(x.ndim - 1)])
  return onp.pad(x, paddings, mode='constant', constant_values=0.)
