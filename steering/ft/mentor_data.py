"""Our preprocessed radio demos (steering/ft/preprocess.py) as LeRobot-style rows for the MENTOR's openpi fork
(github.com/wensi-ai/openpi branch behavior, config pi05_b1k / robot b1k/R1Pro), so that their own transform
pipeline (RepackTransform -> B1KInputs -> MappedDeltaActions -> Normalize -> model transforms) runs unchanged.

Row = the keys their LeRobotB1KDataConfig repacks:
    observation.rgb.{zed_link,left_realsense_link,right_realsense_link}_camera_0 : (224, 224, 3) uint8
        (openpi_client resize_with_pad of the full-res demo frame = exactly what their B1KPolicyWrapper does to the
        sim images at serve time; at 224 their ResizeImages is a no-op)
    observation.state : (61,) float32, the fork's eval proprio layout, rebuilt from our 23-d state:
        base_qvel 0:3 (robot frame, as in the fork eval proprio and the 2026 demos), left arm 3:10,
        left fingers 24:26, right arm 28:35, right fingers 49:51, trunk 53:57. The gripper width (= sum of the two
        finger positions, preprocess.py) goes into the first finger and 0 into the second, so their
        extract_state_from_proprio (sum of fingers) returns our 23-d state bit-exactly. Every other index is 0 and
        is never read by their transforms.
    action : (H, 23) float32 absolute (base vel, torso, L arm, L grip cmd, R arm, R grip cmd)
    prompt : their TASK_REGISTRY radio prompt (== radio_data.RADIO_PROMPT, asserted by mentor_ft.py)

Kept free of openpi imports so spawned torch workers import it cheaply.
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import radio_data  # noqa: E402

HOLDOUT = (2880, 2960, 2970, 2990, 3000)  # == ft_config.HOLDOUT (not imported: ft_config needs openpi-comet)

CAM_DATASET_KEYS = {
    "head": "observation.rgb.zed_link_camera_0",
    "left_wrist": "observation.rgb.left_realsense_link_camera_0",
    "right_wrist": "observation.rgb.right_realsense_link_camera_0",
}
# fork eval obs keys after mentor_server's robot_r1 -> robot rename (their R1Pro ObservationConfig.obs_key)
CAM_OBS_KEYS = {
    "head": "robot::robot:zed_link:Camera:0::rgb",
    "left_wrist": "robot::robot:left_realsense_link:Camera:0::rgb",
    "right_wrist": "robot::robot:right_realsense_link:Camera:0::rgb",
}


def proprio61_from_state23(s):
    s = np.asarray(s, dtype=np.float32)
    p = np.zeros(s.shape[:-1] + (61,), dtype=np.float32)
    p[..., 0:3] = s[..., 0:3]
    p[..., 53:57] = s[..., 3:7]
    p[..., 3:10] = s[..., 7:14]
    p[..., 24] = s[..., 14]
    p[..., 28:35] = s[..., 15:22]
    p[..., 49] = s[..., 22]
    return p


class MentorRadioDataset:
    """Map-style dataset (torch-compatible) over every frame of the preprocessed episodes."""

    def __init__(self, prompt, action_horizon=32, exclude=HOLDOUT, episodes=None, seed=0, root=radio_data.DATA):
        self.prompt = prompt
        self.ds = radio_data.RadioDataset(
            root=root, action_horizon=action_horizon, episodes=episodes, exclude=tuple(exclude), seed=seed
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        x = self.ds[i]
        row = {CAM_DATASET_KEYS[c]: x[radio_data.CAM_KEYS[c]] for c in radio_data.CAMS}
        row["observation.state"] = proprio61_from_state23(x["observation/state"])
        row["action"] = np.asarray(x["actions"], dtype=np.float32)
        row["prompt"] = self.prompt
        return row

    def as_sim_obs(self, i):
        """The same frame as the (renamed) observation dict the sim sends mentor_server, + the demo action chunk."""
        x = self.ds[i]
        obs = {CAM_OBS_KEYS[c]: x[radio_data.CAM_KEYS[c]] for c in radio_data.CAMS}
        obs["robot::proprio"] = proprio61_from_state23(x["observation/state"])
        return obs, np.asarray(x["actions"], dtype=np.float32), np.asarray(x["observation/state"], dtype=np.float32)
