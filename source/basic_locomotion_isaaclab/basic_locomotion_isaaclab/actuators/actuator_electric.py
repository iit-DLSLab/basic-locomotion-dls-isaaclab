# Copyright (c) 2022-2024, The Berkeley Humanoid Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.types import ArticulationActions

from isaaclab.actuators import DCMotor, resolve_joint_parameter
from isaaclab.utils import DelayBuffer, LinearInterpolation

if TYPE_CHECKING:
    from .actuator_cfg import IdentifiedActuatorElectricCfg


class IdentifiedActuatorElectric(DCMotor):
    cfg: IdentifiedActuatorElectricCfg

    def __init__(self, cfg: IdentifiedActuatorElectricCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.friction_static = resolve_joint_parameter(
            self.cfg.friction_static, 0., self.joint_names, self._num_envs, self._device
        )
        self.activation_vel = resolve_joint_parameter(
            self.cfg.activation_vel, torch.inf, self.joint_names, self._num_envs, self._device
        )
        self.friction_dynamic = resolve_joint_parameter(
            self.cfg.friction_dynamic, 0., self.joint_names, self._num_envs, self._device
        )


    def compute(
            self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # call the base method
        control_action = super().compute(control_action, joint_pos, joint_vel)

        # apply friction model on the torque
        control_action.joint_efforts = control_action.joint_efforts - (self.friction_static * torch.tanh(
            joint_vel / self.activation_vel) + self.friction_dynamic * joint_vel)

        self.applied_effort = control_action.joint_efforts
        control_action.joint_positions = None
        control_action.joint_velocities = None

        return control_action
