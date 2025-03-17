# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import os
import torch
from typing import Optional
import yaml


class TocabiMotionLoader:
    """
    Helper class to load and sample motion data from NumPy-file format.
    """

    def __init__(self, motion_file: str, device: torch.device) -> None:
        """Load a motion file and initialize the internal variables.

        Args:
            motion_file: Motion file path to load.
            device: The device to which to load the data.

        Raises:
            AssertionError: If the specified motion file doesn't exist.
        """
        # assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        # data = np.load(motion_file)
        
        # self._dof_names = data["dof_names"].tolist()
        # self._body_names = data["body_names"].tolist()

        # self.dof_positions = torch.tensor(data["dof_positions"], dtype=torch.float32, device=self.device)
        # self.dof_velocities = torch.tensor(data["dof_velocities"], dtype=torch.float32, device=self.device)
        # self.body_positions = torch.tensor(data["body_positions"], dtype=torch.float32, device=self.device)
        # self.body_rotations = torch.tensor(data["body_rotations"], dtype=torch.float32, device=self.device)
        # self.body_linear_velocities = torch.tensor(
        #     data["body_linear_velocities"], dtype=torch.float32, device=self.device
        # )
        # self.body_angular_velocities = torch.tensor(
        #     data["body_angular_velocities"], dtype=torch.float32, device=self.device
        # )

        # self.dt = 1.0 / data["fps"]
        # self.num_frames = self.dof_positions.shape[0]
        # self.duration = self.dt * (self.num_frames - 1)
        # print(f"Motion loaded ({motion_file}): duration: {self.duration} sec, frames: {self.num_frames}")

        self.device = device
        self._load_motions(motion_file)

        self.motion_ids = torch.arange(len(self._motions), dtype=torch.float32, device=self.device)
        print("Motion IDs: ", self.motion_ids)
        # self.motion_dt = torch.tensor(self._motion_dt, dtype=torch.float32, device=self.device)
        # self.dof_positions = torch.tensor([motion[:, 1:13] for motion in self._motions], dtype=torch.float32, device=self.device)
        # self.dof_velocities = torch.tensor([motion[:, 13:25] for motion in self._motions], dtype=torch.float32, device=self.device) * 0.0005 / self.motion_dt.squeeze(0).unsqueeze(-1).unsqueeze(-1)

        # self.root_positions = torch.tensor([motion[:, 25:28] for motion in self._motions], dtype=torch.float32, device=self.device)
        # self.root_rotations = torch.tensor([motion[:, 28:32] for motion in self._motions], dtype=torch.float32, device=self.device)
        # self.root_linear_velocities = torch.tensor([motion[:, 32:35] for motion in self._motions], dtype=torch.float32, device=self.device) * 0.0005 / self.motion_dt.squeeze(0).unsqueeze(-1).unsqueeze(-1)
        # self.root_angular_velocities = torch.tensor([motion[:, 35:38] for motion in self._motions], dtype=torch.float32, device=self.device) * 0.0005 / self.motion_dt.squeeze(0).unsqueeze(-1).unsqueeze(-1)
        
        # self.key_positions = torch.tensor([motion[:, 38:44] for motion in self._motions], dtype=torch.float32, device=self.device)        
        return



    # @property
    # def dof_names(self) -> list[str]:
    #     """Skeleton DOF names."""
    #     return self._dof_names

    # @property
    # def body_names(self) -> list[str]:
    #     """Skeleton rigid body names."""
    #     return self._body_names

    # @property
    # def num_dofs(self) -> int:
    #     """Number of skeleton's DOFs."""
    #     return len(self._dof_names)

    # @property
    # def num_bodies(self) -> int:
    #     """Number of skeleton's rigid bodies."""
    #     return len(self._body_names)

    def _interpolate(
        self,
        a: torch.Tensor,
        *,
        b: Optional[torch.Tensor] = None,
        blend: Optional[torch.Tensor] = None,
        start: Optional[np.ndarray] = None,
        end: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """Linear interpolation between consecutive values.

        Args:
            a: The first value. Shape is (N, X) or (N, M, X).
            b: The second value. Shape is (N, X) or (N, M, X).
            blend: Interpolation coefficient between 0 (a) and 1 (b).
            start: Indexes to fetch the first value. If both, ``start`` and ``end` are specified,
                the first and second values will be fetches from the argument ``a`` (dimension 0).
            end: Indexes to fetch the second value. If both, ``start`` and ``end` are specified,
                the first and second values will be fetches from the argument ``a`` (dimension 0).

        Returns:
            Interpolated values. Shape is (N, X) or (N, M, X).
        """
        if start is not None and end is not None:
            return self._interpolate(a=a[start], b=a[end], blend=blend)
        if a.ndim >= 2:
            blend = blend.unsqueeze(-1)
        if a.ndim >= 3:
            blend = blend.unsqueeze(-1)
        return (1.0 - blend) * a + blend * b

    def _slerp(
        self,
        q0: torch.Tensor,
        *,
        q1: Optional[torch.Tensor] = None,
        blend: Optional[torch.Tensor] = None,
        start: Optional[np.ndarray] = None,
        end: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """Interpolation between consecutive rotations (Spherical Linear Interpolation).

        Args:
            q0: The first quaternion (wxyz). Shape is (N, 4) or (N, M, 4).
            q1: The second quaternion (wxyz). Shape is (N, 4) or (N, M, 4).
            blend: Interpolation coefficient between 0 (q0) and 1 (q1).
            start: Indexes to fetch the first quaternion. If both, ``start`` and ``end` are specified,
                the first and second quaternions will be fetches from the argument ``q0`` (dimension 0).
            end: Indexes to fetch the second quaternion. If both, ``start`` and ``end` are specified,
                the first and second quaternions will be fetches from the argument ``q0`` (dimension 0).

        Returns:
            Interpolated quaternions. Shape is (N, 4) or (N, M, 4).
        """
        if start is not None and end is not None:
            return self._slerp(q0=q0[start], q1=q0[end], blend=blend)
        # if q0.ndim >= 2:
        #     blend = blend.unsqueeze(-1)
        # if q0.ndim >= 3:
        #     blend = blend.unsqueeze(-1)
        if blend.ndim == 1:
            blend = blend.unsqueeze(-1)

        qw, qx, qy, qz = 0, 1, 2, 3  # wxyz
        cos_half_theta = (
            q0[..., qw] * q1[..., qw]
            + q0[..., qx] * q1[..., qx]
            + q0[..., qy] * q1[..., qy]
            + q0[..., qz] * q1[..., qz]
        )

        neg_mask = cos_half_theta < 0
        q1 = q1.clone()
        q1[neg_mask] = -q1[neg_mask]
        cos_half_theta = torch.abs(cos_half_theta)
        cos_half_theta = torch.unsqueeze(cos_half_theta, dim=-1)

        half_theta = torch.acos(cos_half_theta)
        sin_half_theta = torch.sqrt(1.0 - cos_half_theta * cos_half_theta)

        ratio_a = torch.sin((1 - blend) * half_theta) / sin_half_theta
        ratio_b = torch.sin(blend * half_theta) / sin_half_theta

        new_q_x = ratio_a * q0[..., qx : qx + 1] + ratio_b * q1[..., qx : qx + 1]
        new_q_y = ratio_a * q0[..., qy : qy + 1] + ratio_b * q1[..., qy : qy + 1]
        new_q_z = ratio_a * q0[..., qz : qz + 1] + ratio_b * q1[..., qz : qz + 1]
        new_q_w = ratio_a * q0[..., qw : qw + 1] + ratio_b * q1[..., qw : qw + 1]

        new_q = torch.cat([new_q_w, new_q_x, new_q_y, new_q_z], dim=len(new_q_w.shape) - 1)
        new_q = torch.where(torch.abs(sin_half_theta) < 0.001, 0.5 * q0 + 0.5 * q1, new_q)
        new_q = torch.where(torch.abs(cos_half_theta) >= 1, q0, new_q)
        return new_q

    def _compute_frame_blend(self, times: np.ndarray, motion_len, num_frames, dt) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the indexes of the first and second values, as well as the blending time
        to interpolate between them and the given times.

        Args:
            times: Times, between 0 and motion duration, to sample motion values.
                Specified times will be clipped to fall within the range of the motion duration.

        Returns:
            First value indexes, Second value indexes, and blending time between 0 (first value) and 1 (second value).
        """
        phase = np.clip(times / motion_len, 0.0, 1.0)
        index_0 = (phase * (num_frames - 1)).round(decimals=0).astype(int)
        index_1 = np.minimum(index_0 + 1, num_frames - 1)
        blend = ((times - index_0 * dt) / dt).round(decimals=5)
        return index_0, index_1, blend
    
    def sample_motions(self, n):
        m = len(self._motions)
        motion_ids = np.random.choice(m, size=n, replace=True, p=self._motion_weights)
        return motion_ids

    def sample_times(self, motion_ids, duration = None) -> np.ndarray:
        """Sample random motion times uniformly.

        Args:
            num_samples: Number of time samples to generate.
            duration: Maximum motion duration to sample.
                If not defined samples will be within the range of the motion duration.

        Raises:
            AssertionError: If the specified duration is longer than the motion duration.

        Returns:
            Time samples, between 0 and the specified/motion duration.
        """
        phase = np.random.uniform(low=0.0, high=1.0, size=motion_ids.shape)

        motion_len = self._motion_lengths[motion_ids]
        duration = motion_len if duration is None else duration
        # assert (
        #     duration <= motion_len
        # ), f"The specified duration ({duration}) is longer than the motion duration ({motion_len})"
        
        return duration * phase

    def sample(
        self, motion_ids, times: Optional[np.ndarray] = None, duration: float | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample motion data.

        Args:
            num_samples: Number of time samples to generate. If ``times`` is defined, this parameter is ignored.
            times: Motion time used for sampling.
                If not defined, motion data will be random sampled uniformly in time.
            duration: Maximum motion duration to sample.
                If not defined, samples will be within the range of the motion duration.
                If ``times`` is defined, this parameter is ignored.

        Returns:
            Sampled motion DOF positions (with shape (N, num_dofs)), DOF velocities (with shape (N, num_dofs)),
            body positions (with shape (N, num_bodies, 3)), body rotations (with shape (N, num_bodies, 4), as wxyz quaternion),
            body linear velocities (with shape (N, num_bodies, 3)) and body angular velocities (with shape (N, num_bodies, 3)).
        """

        n = len(motion_ids)
        root_pos0 = np.empty([n, 3])
        root_pos1 = np.empty([n, 3])
        root_rot = np.empty([n, 4])
        root_rot0 = np.empty([n, 4])
        root_rot1 = np.empty([n, 4])
        root_vel = np.empty([n, 3])
        root_ang_vel = np.empty([n, 3])
        dof_pos = np.empty([n, 12])
        dof_vel = np.empty([n, 12])
        key_pos0 = np.empty([n, 2, 3])
        key_pos1 = np.empty([n, 2, 3])    
        
        times = self.sample_times(motion_ids, duration) if times is None else times
        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]

        index_0, index_1, blend = self._compute_frame_blend(times, motion_len, num_frames, dt)
        blend = torch.tensor(np.expand_dims(blend, axis=-1), device=self.device)

        unique_ids = np.unique(motion_ids)
        for uid in unique_ids:
            ids = np.where(motion_ids == uid)
            curr_motion = self._motions[uid]

            root_pos0[ids, :]  = curr_motion[index_0[ids], 25:28]
            root_pos1[ids, :]  = curr_motion[index_1[ids], 25:28]

            # root_rot0[ids, :] = curr_motion[index_0[ids], 28:32]
            # root_rot1[ids, :]  = curr_motion[index_1[ids], 28:32]
            root_rot0[ids, 0] = curr_motion[index_0[ids], 31]
            root_rot0[ids, 1:] = curr_motion[index_0[ids], 28:31]
            root_rot1[ids, 0] = curr_motion[index_1[ids], 31]
            root_rot1[ids, 1:] = curr_motion[index_1[ids], 28:31]

            root_vel[ids, :] = curr_motion[index_0[ids], 32:35] * 0.0005 / dt[ids][:,np.newaxis]
            root_ang_vel[ids, :] = curr_motion[index_0[ids], 35:38] * 0.0005 / dt[ids][:,np.newaxis]
            
            key_pos0[ids, 0, :] = curr_motion[index_0[ids], 38:41] 
            key_pos0[ids, 1, :] = curr_motion[index_0[ids], 41:44]

            key_pos1[ids, 0, :] = curr_motion[index_1[ids], 38:41]
            key_pos1[ids, 1, :] = curr_motion[index_1[ids], 41:44]

            dof_pos[ids, :] = curr_motion[index_0[ids], 1:13]
            dof_vel[ids, :] = curr_motion[index_0[ids], 13:25] * 0.0005 / dt[ids][:,np.newaxis]

        root_pos0 = torch.tensor(root_pos0, device=self.device)
        root_pos1 = torch.tensor(root_pos1, device=self.device)
        
        root_rot0 = torch.tensor(root_rot0, device=self.device)
        root_rot1 = torch.tensor(root_rot1, device=self.device)
        
        root_vel = torch.tensor(root_vel, device=self.device)
        root_ang_vel = torch.tensor(root_ang_vel, device=self.device)

        key_pos0 = torch.tensor(key_pos0, device=self.device)
        key_pos1 = torch.tensor(key_pos1, device=self.device)

        dof_pos = torch.tensor(dof_pos, device=self.device)
        dof_vel = torch.tensor(dof_vel, device=self.device)

        root_pos = (1.0 - blend) * root_pos0 + blend * root_pos1

        # print("need shape: ", root_rot0.shape, root_rot1.shape, blend.shape)
        root_rot = self._slerp(q0=root_rot0, q1=root_rot1, blend=blend)
        # print("root_rot shape: ", root_rot.shape)

        blend_expand = blend.unsqueeze(-1)
        key_pos = (1.0 - blend_expand) * key_pos0 + blend_expand * key_pos1

        return (
            dof_pos,
            dof_vel,
            root_pos,
            root_rot,
            root_vel,
            root_ang_vel,
            key_pos,
        )
        # return (
        #     self._interpolate(self.dof_positions, blend=blend, start=index_0, end=index_1),
        #     self._interpolate(self.dof_velocities, blend=blend, start=index_0, end=index_1),
        #     self._interpolate(self.body_positions, blend=blend, start=index_0, end=index_1),
        #     self._slerp(self.body_rotations, blend=blend, start=index_0, end=index_1),
        #     self._interpolate(self.body_linear_velocities, blend=blend, start=index_0, end=index_1),
        #     self._interpolate(self.body_angular_velocities, blend=blend, start=index_0, end=index_1),
        # )

    # def get_dof_index(self, dof_names: list[str]) -> list[int]:
    #     """Get skeleton DOFs indexes by DOFs names.

    #     Args:
    #         dof_names: List of DOFs names.

    #     Raises:
    #         AssertionError: If the specified DOFs name doesn't exist.

    #     Returns:
    #         List of DOFs indexes.
    #     """
    #     indexes = []
    #     for name in dof_names:
    #         assert name in self._dof_names, f"The specified DOF name ({name}) doesn't exist: {self._dof_names}"
    #         indexes.append(self._dof_names.index(name))
    #     return indexes

    # def get_body_index(self, body_names: list[str]) -> list[int]:
    #     """Get skeleton body indexes by body names.

    #     Args:
    #         dof_names: List of body names.

    #     Raises:
    #         AssertionError: If the specified body name doesn't exist.

    #     Returns:
    #         List of body indexes.
    #     """
    #     indexes = []
    #     for name in body_names:
    #         assert name in self._body_names, f"The specified body name ({name}) doesn't exist: {self._body_names}"
    #         indexes.append(self._body_names.index(name))
    #     return indexes
    
    def _load_motions(self, motion_file):
        '''
        load motion files from txt
        0 : time
        1 ~ 12 : q pos
        13 ~ 24 : q vel
        25 ~ 27 : root pos
        28 ~ 31 : root rot
        32 ~ 34 : root vel
        35 ~ 37 : root ang vel
        38 ~ 43 : key pos
        '''
        self._motions = []
        self._motion_lengths = []
        self._motion_weights = []
        self._motion_fps = []
        self._motion_dt = []
        self._motion_num_frames = []
        self._motion_files = []

        self._motion_classes = []

        total_len = 0.0

        motion_files, motion_weights, step_times, play_speeds = self._fetch_motion_files(motion_file)
        num_motion_files = len(motion_files)
        for f in range(num_motion_files):
            curr_file = motion_files[f]
            print("Loading {:d}/{:d} motion files: {:s}".format(f + 1, num_motion_files, curr_file))
            curr_motion = np.loadtxt(curr_file)
            motion_hz = 1.0 / (curr_motion[1,0] - curr_motion[0,0])

            if step_times[f] == 0.6:
                start_idx = int((0.6 * 2 + 1) * motion_hz)
                end_idx = int(start_idx + 0.6 * 4 * motion_hz + 1)
                curr_motion = curr_motion[start_idx:end_idx, :]
            elif step_times[f] == 0.9:
                start_idx = int((0.9 * 2 + 1) * motion_hz)
                end_idx = int(start_idx + 0.9 * 4 * motion_hz + 1) # 4 steps
                curr_motion = curr_motion[start_idx:end_idx, :]
            elif step_times[f] == "yaw":
                start_idx = int((1 + 0.9 * 10) * motion_hz) # start after 10 steps
                end_idx = int(start_idx + 0.9 * 19 * motion_hz + 1) # 19 steps
                curr_motion = curr_motion[start_idx:end_idx, :]

            if play_speeds[f] is None:
                curr_dt = (curr_motion[1,0] - curr_motion[0,0]) 
                motion_fps = 1.0 / curr_dt

                num_frames = curr_motion.shape[0]
                curr_len = 1.0 / motion_fps * (num_frames - 1)
                print("Motion length: {:.3f}s".format(curr_len))
            else:
                if play_speeds[f] < 0.0:
                    curr_motion = np.flip(curr_motion, axis=0)
                    play_speeds[f] = -play_speeds[f]

                curr_dt = (curr_motion[1,0] - curr_motion[0,0]) / play_speeds[f]
                motion_fps = np.abs(1.0 / curr_dt)

                num_frames = curr_motion.shape[0]
                curr_len = 1.0 / motion_fps * (num_frames - 1)
                print("Motion length: {:.3f}s".format(curr_len))

            self._motion_fps.append(motion_fps)
            self._motion_dt.append(curr_dt)
            self._motion_num_frames.append(num_frames)
 
            self._motions.append(curr_motion)
            self._motion_lengths.append(curr_len)
            
            curr_weight = motion_weights[f]
            self._motion_weights.append(curr_weight)
            self._motion_files.append(curr_file)


        self._motion_lengths = np.array(self._motion_lengths)
        self._motion_weights = np.array(self._motion_weights)
        self._motion_weights /= np.sum(self._motion_weights)

        self._motion_fps = np.array(self._motion_fps)
        self._motion_dt = np.array(self._motion_dt)
        self._motion_num_frames = np.array(self._motion_num_frames)

        num_motions = len(self._motions)
        total_len = sum(self._motion_lengths)

        print("Loaded {:d} motions with a total length of {:.3f}s.".format(num_motions, total_len))

        return
    
    def _fetch_motion_files(self, motion_file):
        ext = os.path.splitext(motion_file)[1]
        if (ext == ".yaml"):
            dir_name = os.path.dirname(motion_file)
            motion_files = []
            motion_weights = []
            step_times = []
            play_speeds = []

            with open(os.path.join(os.getcwd(), motion_file), 'r') as f:
                motion_config = yaml.load(f, Loader=yaml.SafeLoader)

            motion_list = motion_config['motions']
            for motion_entry in motion_list:
                curr_file = motion_entry['file']
                curr_weight = motion_entry['weight']
                assert(curr_weight >= 0)

                curr_file = os.path.join(dir_name, curr_file)
                motion_weights.append(curr_weight)
                motion_files.append(curr_file)

                step_time = motion_entry.get('step_time', None)
                if (step_time is not None):
                    step_times.append(step_time)
                else:
                    step_times.append(None)
                
                play_speed = motion_entry.get('play_speed', None)
                if (play_speed is not None):
                    play_speeds.append(play_speed)
                else:
                    play_speeds.append(None)
        else:
            motion_files = [motion_file]
            motion_weights = [1.0]
            step_times = [0.9]
            play_speeds = [1.0]

        return motion_files, motion_weights, step_times, play_speeds


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Motion file")
    args, _ = parser.parse_known_args()

    motion = TocabiMotionLoader(args.file, "cpu")

    # print("- number of frames:", motion.num_frames)
    # print("- number of DOFs:", motion.num_dofs)
    # print("- number of bodies:", motion.num_bodies)
