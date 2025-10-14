# hyperparameter_schedulers.py

import numpy as np
from typing import Dict, Any

class ConfigScheduler:
    """
    Manages and schedules hyperparameter values over a series of steps.

    This class allows for defining a warmup period with an initial value,
    a ramp-up period to a sustained value, and then holds that sustained value.
    It's useful for preventing pipeline bubbles by starting with conservative
    hyperparameters and gradually increasing them as statistical models warm up.

    Args:
        schedule_configs (Dict[str, Dict[str, Any]]): A dictionary where each key
            is a hyperparameter name and the value is its configuration dict.
            Example config:
            {
                "initial_value": 0.05,
                "sustain_value": 0.33,
                "warmup_steps": 2,
                "ramp_steps": 24,
            }
    """
    def __init__(self, schedule_configs: Dict[str, Dict[str, Any]]):
        self.schedules = schedule_configs
        self.current_step = 0

    def get_current_values(self) -> Dict[str, Any]:
        """
        Calculates the value for each scheduled hyperparameter at the current step.

        Returns:
            Dict[str, Any]: A dictionary of hyperparameter names to their
                            calculated values for the current step.
        """
        current_values = {}
        for param, config in self.schedules.items():
            warmup_end = config.get("warmup_steps", 0)
            ramp_end = warmup_end + config.get("ramp_steps", 0)
            
            if self.current_step < warmup_end:
                # In warmup phase, hold the initial value
                value = config["initial_value"]
            elif self.current_step < ramp_end:
                # In ramp phase, linearly interpolate
                ramp_duration = config["ramp_steps"]
                progress = (self.current_step - warmup_end) / ramp_duration
                value = np.interp(
                    progress,
                    [0, 1],
                    [config["initial_value"], config["sustain_value"]]
                )
            else:
                # In sustain phase, hold the final value
                value = config["sustain_value"]

            current_values[param] = value
        
        return current_values

    def step(self):
        """Advances the scheduler to the next step."""
        self.current_step += 1