"""
Step 8: RL fine-tuning (PPO) starting from the behavior-cloned policy.

Behavior cloning can only copy the teacher, so it inherits the teacher's failures. RL instead optimizes for the
outcome (did the apple get picked?), so it can in principle discover behavior the teacher never showed.
Starting from the BC policy matters: from a random network, PPO would need millions of steps just to find
the apple; from BC, most episodes already succeed and RL only has to improve the rest.

    OMNIGIBSON_HEADLESS=1 python my_franka/step8_rl_finetune.py --iters 40
Output: my_franka/data/policy_rl.pt (same format as policy.pt, so step7_eval.py / watch_policy.py can load it)

PPO in one paragraph: run the current policy (with some random noise added to its actions) for a batch of
steps, recording obs, action, reward. Estimate how much better than expected each action turned out to be (the
"advantage", using a learned value function / critic). Then nudge the policy to make good actions more likely
and bad ones less likely - but "clip" the nudge so one batch can't change the policy too much. Repeat.

v2 changes (v1 is in data/v1/step8_rl_finetune_v1.py). Plain PPO made the deterministic policy WORSE: with ~10
episodes per update and a 0/1 reward the gradient is mostly noise, and it eroded the precise "hold still while
the fingers close" behavior until the arm dragged the apple off the table. Fixes:
  - BC anchor loss: pulls the policy back toward the BC policy's actions unless the reward says otherwise
  - no exploration (and no policy gradient) while the gripper is closing
  - every few iterations, a noise-free check on fixed layouts; the best checkpoint is kept as *_best.pt
"""
import argparse
import copy
import os
import time

import numpy as np
import torch as th
import torch.nn as nn

import omnigibson as og
from pick_common import (
    MAX_STEP, apple_z, apply_layout, build_cfg, check_success, get_obs, get_state, sample_layout, to_env_action,
)
from step6_train import DATA_DIR, Policy

RL_SEED = 70_000_000  # training layouts for RL: disjoint from demo seeds and eval seeds (50,000,000+)
CHECK_SEED = 60_000_000  # layouts for the periodic deterministic check during training (also disjoint)
EP_STEPS = 300  # episode length cap during RL
LIFT_Z_SUCCESS = 0.17  # also require the gripper to be up near LIFT_Z, so the episode ends after a real lift


# ---------------------------------------------------------------------------------------------
# 1. A minimal gym-style wrapper: reset() -> obs, step(action) -> obs, reward, done
# ---------------------------------------------------------------------------------------------
class PickEnv:
    """
    Reward is SPARSE: +1 when the apple is grasped and lifted (episode ends), 0 otherwise. No hand-made shaping:
    since BC already succeeds most of the time, "did it work" is enough signal, and it can't be gamed.
    Episode also ends (reward 0) if the apple falls off the table or after EP_STEPS steps.
    """

    def __init__(self):
        self.env = og.Environment(configs=build_cfg())
        self.episode = 0

    def reset(self, seed=None):
        if seed is None:
            seed = RL_SEED + self.episode
            self.episode += 1
        layout = sample_layout(np.random.default_rng(seed))
        apply_layout(self.env, layout)
        self.rest_z, self.t, self.prev_grip = apple_z(self.env), 0, 1.0
        self.state = get_state(self.env, self.prev_grip)
        return get_obs(self.state)

    def step(self, raw_action):
        """@raw_action is the policy's raw output space: 3 scaled deltas in [-1, 1] + 1 gripper logit."""
        grip = 1.0 if raw_action[3] > 0 else -1.0
        act = th.cat([raw_action[:3].clamp(-1, 1) * MAX_STEP, th.tensor([grip])])
        self.env.step(to_env_action(self.env, self.state, act))
        self.prev_grip, self.t = grip, self.t + 1
        self.state = get_state(self.env, self.prev_grip)
        success, lifted = check_success(self.env, self.rest_z)
        success = success and self.state["eef"][2].item() > LIFT_Z_SUCCESS
        fell = lifted < -0.1
        done = success or fell or self.t >= EP_STEPS
        return get_obs(self.state), float(success), done, success


# ---------------------------------------------------------------------------------------------
# 2. Actor (initialized from BC) + critic
# ---------------------------------------------------------------------------------------------
class ActorCritic(nn.Module):
    def __init__(self, bc_policy):
        super().__init__()
        self.actor = copy.deepcopy(bc_policy)  # outputs the MEAN action (same network as BC)
        # Exploration noise std per output, learned. Start small so the BC behavior is barely disturbed:
        # 0.15 in scaled units = 4.5 mm extra wiggle per step; the gripper logit gets a bit more.
        self.log_std = nn.Parameter(th.log(th.tensor([0.15, 0.15, 0.15, 0.5])))
        self.critic = nn.Sequential(nn.Linear(13, 256), nn.Tanh(), nn.Linear(256, 256), nn.Tanh(), nn.Linear(256, 1))

    def value(self, obs):
        return self.critic((obs - self.actor.obs_mean) / self.actor.obs_std).squeeze(-1)

    def dist(self, obs):
        return th.distributions.Normal(self.actor(obs), self.log_std.exp())


def is_closing(obs):
    """Gripper closed but the grasp hasn't registered yet: the expert holds perfectly still here."""
    return (obs[..., 11] < 0) & (obs[..., 12] < 0.5)


def deterministic_check(env, actor, n_episodes):
    """Success rate of the noise-free policy (what watch_policy / step7_eval actually run)."""
    ok = 0
    for i in range(n_episodes):
        obs, done = env.reset(seed=CHECK_SEED + i), False
        while not done:
            with th.no_grad():
                obs, _, done, success = env.step(actor(obs[None])[0])
        ok += success
    return ok / n_episodes


# ---------------------------------------------------------------------------------------------
# 3. PPO loop
# ---------------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", default=os.path.join(DATA_DIR, "policy.pt"))
    parser.add_argument("--out", default=os.path.join(DATA_DIR, "policy_rl.pt"))
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--rollout", type=int, default=2048, help="env steps per PPO iteration")
    parser.add_argument("--critic-warmup", type=int, default=2, help="iterations that only train the critic")
    parser.add_argument("--bc-coef", type=float, default=1.0, help="weight of the 'stay close to BC' anchor loss")
    parser.add_argument("--eval-every", type=int, default=5, help="iterations between deterministic checks")
    parser.add_argument("--eval-episodes", type=int, default=10)
    args = parser.parse_args()
    th.manual_seed(0)

    bc = Policy()
    bc.load_state_dict(th.load(args.init, map_location="cpu")["state_dict"])
    ac = ActorCritic(bc)
    bc.eval()  # frozen copy of the BC policy: the anchor the RL policy is pulled back toward
    best_out = args.out.replace(".pt", "_best.pt")
    best = -1.0
    # Small actor learning rate: we want to refine the BC policy, not overwrite it.
    opt = th.optim.Adam([
        {"params": list(ac.actor.parameters()) + [ac.log_std], "lr": 3e-5},
        {"params": ac.critic.parameters(), "lr": 3e-4},
    ])
    gamma, lam, clip, epochs, minibatch = 0.99, 0.95, 0.2, 10, 256

    env = PickEnv()
    obs = env.reset()
    ep_results, ep_len = [], 0
    for it in range(args.iters):
        t0 = time.time()
        # ---- collect a rollout with the current (noisy) policy ----
        buf = {k: [] for k in ("obs", "act", "logp", "rew", "done", "val")}
        iter_results = []
        for _ in range(args.rollout):
            with th.no_grad():
                d = ac.dist(obs[None])
                # No exploration while the fingers close: there is nothing to gain and drift breaks the grasp.
                a = d.mean[0] if is_closing(obs) else d.sample()[0]
                buf["logp"].append(d.log_prob(a[None]).sum().item())
                buf["val"].append(ac.value(obs[None]).item())
            next_obs, r, done, success = env.step(a)
            for k, v in (("obs", obs), ("act", a), ("rew", r), ("done", done)):
                buf[k].append(v)
            ep_len += 1
            if done:
                iter_results.append((success, ep_len))
                obs, ep_len = env.reset(), 0
            else:
                obs = next_obs
        with th.no_grad():
            last_val = 0.0 if buf["done"][-1] else ac.value(obs[None]).item()

        # ---- advantages with GAE: "how much better than the critic expected was each action?" ----
        rew, val, done = (np.array(buf[k], dtype=np.float32) for k in ("rew", "val", "done"))
        adv, gae = np.zeros_like(rew), 0.0
        for t in reversed(range(len(rew))):
            next_v = last_val if t == len(rew) - 1 else val[t + 1]
            nonterminal = 1.0 - done[t]
            delta = rew[t] + gamma * next_v * nonterminal - val[t]
            gae = delta + gamma * lam * nonterminal * gae
            adv[t] = gae
        ret = th.tensor(adv + val)
        adv = th.tensor((adv - adv.mean()) / (adv.std() + 1e-8))
        obs_b, act_b = th.stack(buf["obs"]), th.stack(buf["act"])
        logp_old = th.tensor(buf["logp"])
        explored = ~is_closing(obs_b)  # only these steps carry a policy-gradient signal
        with th.no_grad():
            bc_target = bc(obs_b)[:, :3]

        # ---- PPO update ----
        train_actor = it >= args.critic_warmup
        for _ in range(epochs):
            for idx in th.randperm(len(obs_b)).split(minibatch):
                d = ac.dist(obs_b[idx])
                logp = d.log_prob(act_b[idx]).sum(-1)
                ratio = (logp - logp_old[idx]).exp()
                m = explored[idx].float()
                pg_terms = th.min(ratio * adv[idx], ratio.clamp(1 - clip, 1 + clip) * adv[idx])
                pg_loss = -(pg_terms * m).sum() / m.sum().clamp(min=1)
                # BC anchor: penalize moving away from what BC would do (in scaled action units)
                bc_loss = (ac.actor(obs_b[idx])[:, :3] - bc_target[idx]).pow(2).mean()
                v_loss = (ac.value(obs_b[idx]) - ret[idx]).pow(2).mean()
                loss = v_loss + ((pg_loss + args.bc_coef * bc_loss) if train_actor else 0.0)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), 0.5)
                opt.step()

        th.save({"state_dict": ac.actor.state_dict()}, args.out)
        check = ""
        if train_actor and (it + 1) % args.eval_every == 0:
            rate = deterministic_check(env, ac.actor, args.eval_episodes)
            check = f" | DETERMINISTIC check {rate * 100:.0f}%"
            if rate > best:
                best = rate
                th.save({"state_dict": ac.actor.state_dict()}, best_out)
                check += " (new best, saved)"
            obs, ep_len = env.reset(), 0  # the check interrupted the training episode
        ep_results += iter_results
        succ = np.mean([s for s, _ in iter_results]) if iter_results else float("nan")
        mean_len = np.mean([n for _, n in iter_results]) if iter_results else float("nan")
        print(f"iter {it:3d} | {'actor+critic' if train_actor else 'critic only '} | episodes {len(iter_results):3d} "
              f"success {succ * 100:5.1f}% len {mean_len:5.0f} | v_loss {v_loss.item():.4f} "
              f"bc_loss {bc_loss.item():.4f} | {time.time() - t0:.0f}s{check}", flush=True)
    print(f"saved fine-tuned policy to {args.out} (best deterministic check {best * 100:.0f}% -> {best_out})", flush=True)
    og.shutdown()


if __name__ == "__main__":
    main()
