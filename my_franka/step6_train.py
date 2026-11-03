"""
Step 6: behavior cloning. Learn  obs (13 numbers) -> expert action (dx, dy, dz, gripper)  from the demos.

No simulator needed - this is ordinary supervised learning on the HDF5 files from step 5.
    python my_franka/step6_train.py                    # uses my_franka/data/demos_w*.hdf5
Output: my_franka/data/policy.pt (weights + the normalization stats the policy needs at run time)
"""
import argparse
import glob
import os

import h5py
import numpy as np
import torch as th
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
MAX_STEP = 0.03  # must match pick_common.MAX_STEP: position deltas are divided by this so they're ~[-1, 1]


class Policy(nn.Module):
    """MLP: normalized obs (13) -> (3 scaled position deltas, 1 gripper logit)."""

    def __init__(self, obs_dim=13, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 4),
        )
        self.register_buffer("obs_mean", th.zeros(obs_dim))
        self.register_buffer("obs_std", th.ones(obs_dim))

    def forward(self, obs):
        return self.net((obs - self.obs_mean) / self.obs_std)

    @th.no_grad()
    def act(self, obs):
        """Raw obs (13,) -> env-ready policy action [dx, dy, dz (metres), gripper (+1 open / -1 close)]."""
        out = self(obs[None].to(self.obs_mean.device))[0].cpu()
        return th.cat([out[:3].clamp(-1, 1) * MAX_STEP, th.tensor([1.0 if out[3] > 0 else -1.0])])


def load_demos(pattern):
    episodes, n_total = [], 0
    for path in sorted(glob.glob(pattern)):
        with h5py.File(path, "r") as f:
            for k in f.keys():
                try:
                    success = bool(f[k].attrs["success"])
                    ep = (f[k]["obs"][:], f[k]["act"][:])
                except (KeyError, OSError):  # episode cut off mid-write when a worker was stopped
                    continue
                n_total += 1
                if success:  # only train on episodes that really picked the apple
                    episodes.append(ep)
    print(f"loaded {len(episodes)} successful episodes (of {n_total}) from {pattern}")
    return episodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=os.path.join(DATA_DIR, "demos_w*.hdf5"))
    parser.add_argument("--out", default=os.path.join(DATA_DIR, "policy.pt"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    th.manual_seed(0)
    dev = "cuda" if th.cuda.is_available() else "cpu"

    # Split by EPISODE (not by step) so validation measures generalization to unseen layouts.
    episodes = load_demos(args.data)
    rng = np.random.default_rng(0)
    order = rng.permutation(len(episodes))
    n_val = max(1, len(episodes) // 10)
    split = {"val": [episodes[i] for i in order[:n_val]], "train": [episodes[i] for i in order[n_val:]]}
    data = {}
    for name, eps in split.items():
        obs = th.from_numpy(np.concatenate([o for o, _ in eps]))
        act = th.from_numpy(np.concatenate([a for _, a in eps]))
        data[name] = (obs.to(dev), (act[:, :3] / MAX_STEP).to(dev), (act[:, 3] < 0).float().to(dev))
        print(f"{name}: {len(eps)} episodes, {len(obs)} steps")

    policy = Policy().to(dev)
    policy.obs_mean.copy_(data["train"][0].mean(0))
    policy.obs_std.copy_(data["train"][0].std(0) + 1e-6)
    opt = th.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = th.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    bce = nn.BCEWithLogitsLoss()

    def losses(obs, dpos, close):
        out = policy(obs)
        # gripper target: 1 = close. Logit sign convention: act() opens when out[3] > 0, so use -logit for "close"
        return nn.functional.mse_loss(out[:, :3], dpos), bce(-out[:, 3], close)

    best = float("inf")
    for epoch in range(args.epochs):
        policy.train()
        obs, dpos, close = data["train"]
        perm = th.randperm(len(obs), device=dev)
        for i in range(0, len(obs), args.batch_size):
            idx = perm[i : i + args.batch_size]
            l_pos, l_grip = losses(obs[idx], dpos[idx], close[idx])
            opt.zero_grad()
            (l_pos + l_grip).backward()
            opt.step()
        sched.step()

        policy.eval()
        with th.no_grad():
            v_pos, v_grip = losses(*data["val"])
            out = policy(data["val"][0])
            grip_acc = ((out[:, 3] < 0).float() == data["val"][2]).float().mean().item()
        val = v_pos.item() + v_grip.item()
        if val < best:
            best = val
            th.save({"state_dict": policy.state_dict()}, args.out)
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch:3d}  train pos {l_pos.item():.4f} grip {l_grip.item():.4f} | "
                  f"val pos {v_pos.item():.4f} grip {v_grip.item():.4f} grip-acc {grip_acc * 100:.2f}%")
    print(f"saved best policy (val loss {best:.4f}) to {args.out}")


if __name__ == "__main__":
    main()
