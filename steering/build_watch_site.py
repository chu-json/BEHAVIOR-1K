"""Build the shareable "watch the prompts" page: example rollouts per prompt (Comet A / pt50 / base pi0.5 side by side,
prompt above the videos) plus side-by-side comparison tables. Output: <out>/index.html + <out>/videos/*.mp4.

    python steering/build_watch_site.py --out /home/behavior/ext/site
"""

import argparse
import csv
import html
import json
import os
import statistics as st
import sys

import av

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from motor_report import metrics  # noqa: E402
from probe_prompts import PROMPTS  # noqa: E402

RUNS = "/home/behavior/ext/runs"
MODELS = [("A_radio_sft", "Comet radio SFT"), ("B_pt50", "Comet pt50"), ("C_pi05_base", "Base pi0.5")]
SWEEP = ["P0", "P1", "P2", "P3", "P4"]
MOTOR = ["P0", "M_left_arm_up", "M_right_arm_up", "M_turn_left", "M_turn_right", "M_forward", "M_still", "X_nonsense"]
LABEL = {
    "P0": "original task prompt",
    "P1": "paraphrase",
    "P2": "asks for the left hand",
    "P3": "asks for the right hand",
    "P4": "asks not to lift the radio",
    "M_left_arm_up": "motor command",
    "M_right_arm_up": "motor command",
    "M_turn_left": "motor command",
    "M_turn_right": "motor command",
    "M_forward": "motor command",
    "M_still": "motor command",
    "X_nonsense": "control: unrelated sentence",
}
SETUP, SEED = "242", "0"


def sweep_dir(model, pid):
    if model == "C_pi05_base":
        return f"{RUNS}/day/base_sweep/C_pi05_base_{pid}_s0-1"
    return f"{RUNS}/prompt_sweep/{model}_{pid}_s0"


def motor_dir(model, pid):
    return f"{RUNS}/day/motor/{model}_{pid}_s0"


def find_rollout(run_dir):
    for r in csv.DictReader(open(f"{run_dir}/rollouts.csv")):
        if r["instance_id"] == SETUP and r["seed"] == SEED:
            return r
    return None


def reencode(src, dst, crf="32"):
    if os.path.exists(dst):
        return
    inp = av.open(src)
    vs = inp.streams.video[0]
    out = av.open(dst, "w")
    s = out.add_stream("libx264", rate=30)
    s.width, s.height, s.pix_fmt = vs.width, vs.height, "yuv420p"
    s.options = {"crf": crf, "preset": "slow", "movflags": "+faststart"}
    for f in inp.decode(video=0):
        for p in s.encode(av.VideoFrame.from_ndarray(f.to_ndarray(format="rgb24"), format="rgb24")):
            out.mux(p)
    for p in s.encode():
        out.mux(p)
    out.close()


def sweep_stats():
    """Per (model, prompt): seeds 0+1 x 5 setups = 10 rollouts, same settings for all three models."""
    rows = list(csv.DictReader(open(f"{RUNS}/prompt_sweep/results.csv")))
    rows += list(csv.DictReader(open(f"{RUNS}/day/base_sweep/results.csv")))
    out = {}
    for m, _ in MODELS:
        for pid in SWEEP:
            rs = [r for r in rows if r["checkpoint"] == m and r["prompt_id"] == pid]
            g = [r for r in rs if r["grasped"] == "1"]
            succ = [r for r in rs if r["success"] == "1"]
            out[(m, pid)] = {
                "n": len(rs),
                "success": len(succ),
                "pressed_left": sum(r["pressing_hand"] == "left" for r in succ),
                "pressed_right": sum(r["pressing_hand"] == "right" for r in succ),
                "grasped": len(g),
                "lifted_when_grasped": sum(r["lifted"] == "1" for r in g),
                "closest_cm": st.median(min(float(r["min_dist_left_cm"]), float(r["min_dist_right_cm"])) for r in rs),
            }
    return out


def motor_contrasts():
    groups = json.load(open(f"{RUNS}/day/motor/groups.json"))
    D = {}
    for g in groups:
        f = f"{RUNS}/day/motor/{g['name']}/rollouts.csv"
        if not os.path.exists(f):
            continue
        for r in csv.DictReader(open(f)):
            D.setdefault((g["checkpoint"], g["prompt_id"]), {})[f"{r['instance_id']}_s{r['seed']}"] = metrics(
                r["trace_path"]
            )
    tests = [
        ("M_still", "X_nonsense", "total_motion", -1, "Stay still vs. unrelated sentence", "less total motion", "m"),
        (
            "M_turn_left",
            "M_turn_right",
            "yaw_change",
            +1,
            "Turn left vs. turn right",
            "more left (CCW) rotation",
            "deg",
        ),
        (
            "M_left_arm_up",
            "M_right_arm_up",
            "raise_bias_left",
            +1,
            "Raise left vs. raise right arm",
            "left hand rises more",
            "m",
        ),
        ("M_forward", "M_still", "forward", +1, "Move forward vs. stay still", "more forward travel", "m"),
    ]
    out = []
    for a, b, key, sgn, label, expect, unit in tests:
        cells = {}
        for m, _ in MODELS:
            A, B = D.get((m, a), {}), D.get((m, b), {})
            ks = sorted(set(A) & set(B))
            v = [sgn * (A[k][key] - B[k][key]) for k in ks]
            cells[m] = (sum(x > 0 for x in v), len(v), st.median(v) if v else float("nan"))
        out.append((label, expect, unit, cells))
    return out


def video_card(model_label, r, src_key):
    ok = r["success"] == "1"
    outcome = f"turned the radio on at step {r['steps']}" if ok else f"timed out at {r['max_steps']} steps"
    closest = ""
    if r.get("min_finger_button_dist"):
        closest = f" · closest finger {100 * float(r['min_finger_button_dist']):.0f} cm from the button"
    return (
        f"<figure class=clip><figcaption><span class=model>{html.escape(model_label)}</span>"
        f"<span class='outcome {'ok' if ok else 'bad'}'>{outcome}</span></figcaption>"
        f"<video controls muted playsinline preload=metadata src='videos/{src_key}.mp4'></video>"
        f"<p class=meta>setup {SETUP}, seed {SEED}{closest}</p></figure>"
    )


def section(title, intro, prompts, dir_fn, prefix, out_dir):
    parts = [f"<section class=block><h2>{html.escape(title)}</h2><p class=lede>{intro}</p>"]
    for pid in prompts:
        cards = []
        for m, mlabel in MODELS:
            d = dir_fn(m, pid)
            r = find_rollout(d) if os.path.exists(f"{d}/rollouts.csv") else None
            if r is None:
                cards.append(
                    f"<figure class=clip><figcaption><span class=model>{mlabel}</span></figcaption>"
                    "<p class=meta>no rollout for this setup</p></figure>"
                )
                continue
            key = f"{prefix}_{m}_{pid}"
            reencode(r["video_path"], f"{out_dir}/videos/{key}.mp4")
            cards.append(video_card(mlabel, r, key))
        parts.append(
            f"<article class=prompt id='{prefix}-{pid}'><header><span class=pid>{pid}</span>"
            f"<span class=kind>{html.escape(LABEL[pid])}</span></header>"
            f"<blockquote>{html.escape(PROMPTS[pid])}</blockquote><div class=clips>{''.join(cards)}</div></article>"
        )
    parts.append("</section>")
    return "".join(parts)


CSS = """
:root{
  /* layout: one reading column; prompt rows hold three clips that wrap to one column on phones */
  --bg:#eef1f4; --panel:#ffffff; --fg:#17202b; --muted:#566273; --rule:#cfd6de; --accent:#1d5bbf;
  --ok:#1b7a47; --bad:#ad3a2d; --mid:#8a6a0e; --quote:#e5ebf3;
  --display:"Archivo",system-ui,sans-serif; --body:"Public Sans",system-ui,sans-serif;
  --mono:"JetBrains Mono",ui-monospace,Menlo,monospace;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#12161c; --panel:#1a2029; --fg:#e4e9ef; --muted:#9aa6b5; --rule:#2c3542; --accent:#7aa7f0;
  --ok:#5cc28a; --bad:#ec8576; --mid:#d8b552; --quote:#202a37; color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#12161c; --panel:#1a2029; --fg:#e4e9ef; --muted:#9aa6b5; --rule:#2c3542; --accent:#7aa7f0;
  --ok:#5cc28a; --bad:#ec8576; --mid:#d8b552; --quote:#202a37; color-scheme:dark}
body{background:var(--bg);color:var(--fg);font:15px/1.55 var(--body)}
.wrap{max-width:1180px;margin:0 auto;padding-inline:16px;padding-block:28px 64px}
h1,h2,h3{font-family:var(--display);text-wrap:balance;line-height:1.15}
h1{font-size:2rem;font-weight:800;letter-spacing:-.01em;margin:0 0 .4rem}
h2{font-size:1.35rem;font-weight:700;margin:0 0 .4rem}
h3{font-size:1.02rem;font-weight:700;margin:1.1rem 0 .3rem}
p{margin:.3rem 0 .7rem;max-width:72ch}
.lede{color:var(--muted)}
.sub{color:var(--muted);max-width:80ch}
.block{margin-top:40px}
.qa{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,330px),1fr));gap:14px;margin-top:20px}
.qa div{background:var(--panel);border:1px solid var(--rule);border-radius:8px;padding:12px 14px;min-width:0}
.qa h3{margin-top:0}
.tablewrap{overflow-x:auto;margin:10px 0 4px;border:1px solid var(--rule);border-radius:8px;background:var(--panel)}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:.92rem}
th,td{padding:7px 10px;text-align:left;border-bottom:1px solid var(--rule);vertical-align:top}
th{font-family:var(--display);font-weight:700;font-size:.82rem;letter-spacing:.03em;text-transform:uppercase;color:var(--muted)}
tr:last-child td{border-bottom:0}
td.num{text-align:right;white-space:nowrap}
.ok{color:var(--ok);font-weight:600}.bad{color:var(--bad);font-weight:600}.midc{color:var(--mid);font-weight:600}
code{font-family:var(--mono);font-size:.85em;background:var(--quote);padding:1px 4px;border-radius:4px}
.speed{position:sticky;top:env(safe-area-inset-top,0px);z-index:5;display:flex;flex-wrap:wrap;gap:8px;align-items:center;
  background:var(--bg);padding-block:10px;border-bottom:1px solid var(--rule);margin-top:24px}
.speed span{font-family:var(--display);font-weight:700;font-size:.85rem;letter-spacing:.03em;text-transform:uppercase;color:var(--muted)}
.speed button,.jump a{font:600 .88rem var(--body);color:var(--fg);background:var(--panel);border:1px solid var(--rule);
  border-radius:999px;padding:4px 12px;cursor:pointer;text-decoration:none}
.speed button[aria-pressed="true"]{background:var(--accent);color:var(--bg);border-color:var(--accent)}
.speed button:focus-visible,.jump a:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.jump{display:flex;flex-wrap:wrap;gap:6px}
.prompt{margin-top:22px;padding-top:16px;border-top:1px solid var(--rule)}
.prompt header{display:flex;gap:10px;align-items:baseline;flex-wrap:wrap}
.pid{font-family:var(--mono);font-weight:700;color:var(--accent)}
.kind{font-size:.85rem;letter-spacing:.04em;text-transform:uppercase;color:var(--muted)}
blockquote{margin:.4rem 0 .8rem;padding:10px 14px;background:var(--quote);border-radius:6px;font:500 1rem/1.45 var(--mono)}
.clips{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,300px),1fr));gap:14px}
.clip{margin:0;min-width:0}
.clip figcaption{display:flex;flex-direction:column;gap:2px;margin-bottom:6px}
.model{font-family:var(--display);font-weight:700}
.outcome{font-size:.88rem}
.clip video{width:100%;max-width:100%;aspect-ratio:672/448;background:#000;border-radius:6px;display:block}
.meta{font-size:.82rem;color:var(--muted);margin:.35rem 0 0}
@media (prefers-reduced-motion: reduce){*{scroll-behavior:auto}}
"""

JS = """
<script>
(function(){
  var rate = 2;
  try { var saved = parseFloat(localStorage.getItem('rate')); if (saved) rate = saved; } catch (e) {}
  function apply(){ document.querySelectorAll('video').forEach(function(v){ v.playbackRate = rate; v.defaultPlaybackRate = rate; });
    document.querySelectorAll('.speed button').forEach(function(b){ b.setAttribute('aria-pressed', String(parseFloat(b.dataset.rate) === rate)); }); }
  document.querySelectorAll('.speed button').forEach(function(b){ b.addEventListener('click', function(){ rate = parseFloat(b.dataset.rate);
    try { localStorage.setItem('rate', String(rate)); } catch (e) {} apply(); }); });
  document.querySelectorAll('video').forEach(function(v){ v.addEventListener('loadedmetadata', function(){ v.playbackRate = rate; }); });
  apply();
})();
</script>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(f"{args.out}/videos", exist_ok=True)

    S = sweep_stats()
    rows = []
    for pid in SWEEP:
        cells = []
        for m, _ in MODELS:
            s = S[(m, pid)]
            cls = "ok" if s["success"] >= 7 else ("midc" if s["success"] >= 3 else "bad")
            hand = f"{s['pressed_left']}L / {s['pressed_right']}R" if s["success"] else "–"
            lift = f"{s['lifted_when_grasped']}/{s['grasped']}" if s["grasped"] else "never grasped"
            cells.append(
                f"<td class=num><span class={cls}>{s['success']}/{s['n']}</span></td><td>{hand}</td>"
                f"<td>{lift}</td><td class=num>{s['closest_cm']:.0f} cm</td>"
            )
        rows.append(f"<tr><td><code>{pid}</code> {html.escape(LABEL[pid])}</td>{''.join(cells)}</tr>")
    sweep_table = (
        "<div class=tablewrap><table><tr><th rowspan=2>prompt</th>"
        + "".join(f"<th colspan=4>{lbl}</th>" for _, lbl in MODELS)
        + "</tr><tr>"
        + "<th>success</th><th>pressing hand</th><th>lifted when grasped</th><th>closest finger</th>" * 3
        + "</tr>"
        + "".join(rows)
        + "</table></div>"
    )

    mrows = []
    for label, expect, unit, cells in motor_contrasts():
        tds = []
        for m, _ in MODELS:
            k, n, med = cells[m]
            if not n:
                tds.append("<td>–</td>")
                continue
            cls = "ok" if k == n else ("midc" if k / n >= 0.75 else "bad")
            tds.append(f"<td class=num><span class={cls}>{k}/{n}</span> <small>median {med:+.2f} {unit}</small></td>")
        mrows.append(
            f"<tr><td>{html.escape(label)}<br><small>expected: {html.escape(expect)}</small></td>{''.join(tds)}</tr>"
        )
    motor_table = (
        "<div class=tablewrap><table><tr><th>paired test</th>"
        + "".join(f"<th>{lbl}</th>" for _, lbl in MODELS)
        + "</tr>"
        + "".join(mrows)
        + "</table></div>"
    )

    sweep_sec = section(
        "The task prompt and four rewordings",
        "Same five prompts for all three models, same setup and seed, whole-house scene, 3000-step limit. The prompt is the "
        "only thing that changes down a column; the model is the only thing that changes across a row.",
        SWEEP,
        sweep_dir,
        "sweep",
        args.out,
    )
    motor_sec = section(
        "Simple motor commands",
        "Plain instructions with an obvious expected effect, plus an unrelated sentence as a control. 1000-step episodes, "
        "so the Comet models usually run out of time before pressing the button.",
        MOTOR,
        motor_dir,
        "motor",
        args.out,
    )

    jump = "".join(f"<a href='#sweep-{p}'>{p}</a>" for p in SWEEP) + "".join(
        f"<a href='#motor-{p}'>{p.replace('M_', '').replace('X_', '').replace('_', ' ')}</a>" for p in MOTOR[1:]
    )
    page = f"""<title>Radio Prompt Rollouts</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@600;700;800&family=JetBrains+Mono:wght@500;700&family=Public+Sans:wght@400;600&display=swap">
<style>{CSS}</style>
<div class=wrap>
<h1>Radio prompt rollouts</h1>
<p class=sub>BEHAVIOR <code>turning_on_radio</code>, R1Pro robot. Three policies get the same instructions:
Comet's radio-specific checkpoint, Comet's 50-task pretrained checkpoint (pt50), and Physical Intelligence's base pi0.5,
which was never trained on BEHAVIOR. Each prompt below is shown above one example rollout per policy.</p>

<div class=qa>
<div><h3>How many checkpoints does Comet use?</h3><p>One per task, picked from 26. Comet's 2025 challenge release has 12
task-specific fine-tunes (SFT or RFT) and 14 snapshots of shared pretrained models (pt7, pt10, pt12 and pt50 at
different training steps). For each of the 50 tasks they submitted whichever scored best on that task's validation
set. For this task that is <code>pi05-turning_on_radio-sft</code>. After the challenge they also released two
multi-task models, pt12 and pt50. This page uses the radio checkpoint and pt50.</p></div>
<div><h3>Were the exact same prompts run on Comet and base pi0.5?</h3><p>Yes. Prompts P0 to P4 ran on all three policies with
identical setups (242, 203, 109, 181, 187), seeds (0 and 1), scene, robot settings and 3000-step limit: 10 rollouts per
policy per prompt. The motor commands ran on all three too (3 setups x 2 seeds for the radio checkpoint and base pi0.5,
3 setups x 1 seed for pt50).</p></div>
<div><h3>How to read the videos</h3><p>Each video shows the two wrist cameras on the left and the head camera on the right.
Videos start at 2x speed; change it with the buttons below. "Closest finger" is the nearest any finger link got to the
radio's button (presses register at about 5 to 6 cm).</p></div>
</div>

<section class=block><h2>Side by side: same prompts, three policies</h2>
<p class=lede>10 rollouts per cell (5 setups x 2 seeds). "Lifted when grasped" counts rollouts that raised the radio more
than 3 cm, out of those that grasped it. Closest finger is the median over the 10 rollouts.</p>
{sweep_table}
<p>The radio checkpoint succeeds on almost every prompt. On each setup it always presses with the same hand (left on 242, 181 and 187; right on 203 and 109), whatever the prompt asks.
pt50 is weaker and does not follow the hand or lifting instructions either. Base pi0.5 never grasps the radio.</p>
<h3>Motor commands, paired by setup and seed</h3>
{motor_table}
<p>Base pi0.5 reliably moves less when told to stay still and shows a noisy turning response; its left and right arm
commands come out on the wrong arm. The radio checkpoint does not respond to any of these commands. pt50's turning
result is based on only three pairs.</p>
</section>

<nav class=speed aria-label="Playback speed"><span>Speed</span>
<button type=button data-rate=1 id=speed1>1x</button><button type=button data-rate=2 id=speed2>2x</button>
<button type=button data-rate=4 id=speed4>4x</button><span>Jump to</span><div class=jump>{jump}</div></nav>
{sweep_sec}
{motor_sec}
</div>
{JS}"""
    with open(f"{args.out}/index.html", "w") as f:
        f.write(page)
    total = sum(os.path.getsize(f"{args.out}/videos/{v}") for v in os.listdir(f"{args.out}/videos"))
    print(f"wrote {args.out}/index.html; {len(os.listdir(args.out + '/videos'))} videos, {total / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
