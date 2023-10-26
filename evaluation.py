import time
from typing import Optional

import acme
from acme import core
from acme.utils import counting
from acme.utils import loggers
import dm_env

class Validate(core.Worker):
  def __init__(self, environment, actor, label = "evaluation", counter = None, logger = None):
    self._env = environment 
    self.env = environment
    self.actor = actor
    self.counter = counter or counting.Counter()
    self.logger = logger or loggers.make_default_logger(label)

  def run(self, num_episodes: int):
    self.actor.update(wait=True)
    total_episode_return = 0.0
    total_episode_steps = 0
    start_time = time.time()

    for _ in range(num_episodes):
      timestep = self.env.reset()
      self.actor.observe_first(timestep)
      while not timestep.last():
        action = self.actor.select_action(timestep.observation)
        timestep = self.env.step(action)
        self.actor.observe(action, timestep)
        total_episode_steps += 1
        total_episode_return += timestep.reward

    steps_per_second = total_episode_steps / (time.time() - start_time)
    counts = self.counter.increment(
        steps=total_episode_steps, episodes=num_episodes)
    average_episode_return = total_episode_return / num_episodes
    average_episode_steps = total_episode_steps / num_episodes
    average_normalized_return = self.env.get_normalized_score(
        average_episode_return)
    result = {
        "average_episode_return": average_episode_return,
        "average_normalized_return": average_normalized_return,
        "average_episode_length": average_episode_steps,
        "steps_per_second": steps_per_second,
    }
    result.update(counts)
    self.logger.write(result)
