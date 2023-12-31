from typing import Any, Optional

from absl import logging
from acme.utils.loggers import base

try:
  import wandb
except ImportError:
  wandb = None


class WandbLogger(base.Logger):
  """A Wandb based logger"""

  def __init__(
      self,
      label: Optional[str] = None,
      steps_key: Optional[str] = None,
      *,
      project: Optional[str] = None,
      entity: Optional[str] = None,
      dir: Optional[str] = None,
      name: Optional[str] = None,
      group: Optional[str] = None,
      config: Optional[Any] = None,
      run=None,
      **wandb_kwargs,
  ):
    if wandb is None:
      raise ImportError("Wandb Logger is not installed")
    self._label = label
    self._iter = 0
    self._steps_key = steps_key
    if run is None:
      if wandb.run is None:
        self._run = wandb.init(
            project=project,
            dir=dir,
            entity=entity,
            name=name,
            group=group,
            config=config,
            **wandb_kwargs,
        )
      else:
        self._run = wandb.run
    else:
      self._run = run

    if steps_key and getattr(self._run, "define_metric", None):
      prefix = f"{self._label}/*" if self._label else "*"
      self._run.define_metric(
          prefix, step_metric=f"{self._label}/{self._steps_key}")

  @property
  def run(self):
    """Return the current wandb run."""
    return self._run

  def write(self, data: base.LoggingData):
    data = base.to_numpy(data)
    if self._steps_key is not None and self._steps_key not in data:
      logging.warn("steps key %s not found. Skip logging.", self._steps_key)
      return
    if self._label:
      stats = {f"{self._label}/{k}": v for k, v in data.items()}
    else:
      stats = data
    self._run.log(stats)

    self._iter += 1

  def close(self):
    pass
