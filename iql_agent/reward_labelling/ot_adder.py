from acme import adders
from acme import types
import dm_env

from iql_agent.reward_labelling import ot_rewarder


class OptimalTransportAdder(adders.Adder):
  """A wrapper class to substitute OT based rewards"""

  def __init__(self, direct_rl_adder: adders.Adder, ot_rewarder: ot_rewarder.OptimalTransportRewarder):
    self._adder = direct_rl_adder
    self._rewarder = ot_rewarder
    self._steps = []
    self._timesteps = []


  def add_first(self, timestep: dm_env.TimeStep):
    self._steps = []
    self._timesteps = []
    self._timesteps.append(timestep)


  def add(self, action: types.NestedArray, next_timestep: dm_env.TimeStep, extras: types.NestedArray = ()):
    del extras
    self._steps.append(
        types.Transition(self._timesteps[-1].observation, action, (), (), ()))
    self._timesteps.append(next_timestep)
    if next_timestep.last():
      self.append_episode()


  def append_episode(self):
    psuedo_rewards = self._rewarder.compute_offline_rewards(self._steps[:])
    first_timestep = self._timesteps[0]
    self._adder.add_first(first_timestep._replace(reward=psuedo_rewards[0]))
    actions = [step.action for step in self._steps]
    assert len(actions) == len(self._timesteps) - 1 == len(psuedo_rewards)
    for action, next_ts, pr in zip(actions, self._timesteps[1:],
                                   psuedo_rewards):
      self._adder.add(action, next_ts._replace(reward=pr))
    self._adder.reset()


  def reset_episode(self):
    self._adder.reset()
    self._steps = []
    self._timesteps = []
