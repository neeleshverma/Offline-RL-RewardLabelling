import os
from typing import Any, Dict, Mapping, Optional

from absl import logging
from acme.jax import utils as jax_utils
from acme.utils import loggers as loggers_lib
from acme.utils.loggers import base

import logger


def time_delta(default_time_delta, time_delta = None):
  if time_delta is not None:
    return time_delta
  else:
    return default_time_delta

# Default Logger if wandb is not used:
class Logger:
  def __init__(self,
               workdir = None,
               log_to_wandb = False,
               wandb_kwargs = None,
               time_delta = 1.0,
               actor_time_delta = None,
               learner_time_delta = None,
               evaluator_time_delta = None,
               async_learner_logger = False):

    wandb_kwargs = wandb_kwargs or {}
    self._log_to_wandb = log_to_wandb
    self._run = None
    if log_to_wandb and self._run is None:
      import wandb
      wandb.require('service')
      self._run = wandb.init(**wandb_kwargs)
    if workdir is not None:
      os.makedirs(workdir, exist_ok=True)
    self._workdir = workdir
    self._time_delta = time_delta
    self._actor_time_delta = actor_time_delta
    self._learner_time_delta = learner_time_delta
    self._evaluator_time_delta = evaluator_time_delta
    self._async_learner_logger = async_learner_logger

  @property
  def run(self):
    return self._run

  def __call__(self,
               label,
               steps_key = None,
               task_instance = 0):
    if steps_key is None:
      steps_key = f'{label}_steps'

    if label == 'learner':
      return self.default_logger(
          label=label,
          asynchronous=self._async_learner_logger,
          time_delta=time_delta(self._time_delta, self._learner_time_delta),
          serialize_fn=jax_utils.fetch_devicearray,
          workdir=self._workdir,
          log_to_wandb=self._log_to_wandb,
          steps_key=steps_key,
          wandb_run=self._run)
    elif label in ('evaluator', 'eval_loop', 'evaluation', 'eval'):
      return self.default_logger(
          label=label,
          time_delta=time_delta(self._time_delta, self._evaluator_time_delta),
          steps_key=steps_key,
          workdir=self._workdir,
          log_to_wandb=self._log_to_wandb,
          wandb_run=self._run)
    elif label in ('actor', 'train_loop', 'train'):
      return self.default_logger(
          label=label,
          save_data=task_instance == 0,
          time_delta=time_delta(self._time_delta, self._evaluator_time_delta),
          steps_key=steps_key,
          workdir=self._workdir,
          log_to_wandb=self._log_to_wandb,
          wandb_run=self._run)
    else:
      logging.warning('Unknown label %s. Fallback to default.', label)
      return self.default_logger(
          label=label,
          steps_key=steps_key,
          time_delta=self._time_delta,
          workdir=self._workdir,
          log_to_wandb=self._log_to_wandb,
          wandb_run=self._run,
      )

  @staticmethod
  def default_logger(
      label,
      save_data = True,
      time_delta = 1.0,
      asynchronous = False,
      print_fn = None,
      workdir = None,
      serialize_fn = base.to_numpy,
      steps_key = 'steps',
      log_to_wandb = False,
      wandb_kwargs = None,
      add_uid = False,
      wandb_run = None):

    if not print_fn:
      print_fn = logging.info
    terminal_logger = loggers_lib.TerminalLogger(label=label, print_fn=print_fn)

    loggers = [terminal_logger]
    if save_data and workdir is not None:
      loggers.append(
          loggers_lib.CSVLogger(workdir, label=label, add_uid=add_uid))
    if save_data and log_to_wandb:
      if wandb_kwargs is None:
        wandb_kwargs = {}
      loggers.append(
          logger.WandbLogger(
              label=label, steps_key=steps_key, run=wandb_run, **wandb_kwargs))

    logger = loggers_lib.Dispatcher(loggers, serialize_fn)
    logger = loggers_lib.NoneFilter(logger)
    if asynchronous:
      logger = loggers_lib.AsyncLogger(logger)
    logger = loggers_lib.TimeFilter(logger, time_delta)
    return loggers_lib.AutoCloseLogger(logger)
