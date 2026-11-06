"""Shared prompt set and model registry for the base-pi0.5 / Comet steerability experiments."""

EXT = "/home/behavior/ext"
BASE_PROMPT = "Turn on the radio receiver that's on the table in the living room."
_STEM = BASE_PROMPT.rstrip(".")

PROMPTS = {
    # The overnight sweep's prompts (identical strings).
    "P0": BASE_PROMPT,
    "P1": "Switch on the radio receiver sitting on the living room table.",
    "P2": _STEM + " using your left hand.",
    "P3": _STEM + " using your right hand.",
    "P4": _STEM + " without picking the radio up.",
    # Simple motor commands with a measurable expected effect (does the model follow ANY instruction?).
    "M_left_arm_up": "Raise your left arm.",
    "M_right_arm_up": "Raise your right arm.",
    "M_turn_left": "Turn left.",
    "M_turn_right": "Turn right.",
    "M_forward": "Move forward.",
    "M_still": "Stay still and do nothing.",
    # Controls: does the model read the prompt at all?
    "X_empty": "",
    "X_nonsense": "Purple elephants dance quietly on the moon.",
}

MODELS = {
    "A_radio_sft": {
        "ckpt": f"{EXT}/ckpt/pi05-turning_on_radio-sft",
        "config": "pi05_b1k-turning_on_radio_lr2.5e-6_step20k_sft",
    },
    "B_pt50": {
        "ckpt": f"{EXT}/ckpt/openpi_comet/pi05-b1kpt50-cs32",
        "config": "pi05_b1k-pt50_cs32_bs64_lr2.5e-5_step50k",
    },
    # Physical Intelligence's generalist pi0.5 base checkpoint (never trained on BEHAVIOR / R1Pro), wired with the
    # same B1K input/output transforms and Comet's shared B1K norm stats (see steering/NOTES.md).
    "C_pi05_base": {
        "ckpt": f"{EXT}/ckpt/pi05_base_b1k",
        "config": "pi05_b1k-base",
    },
}
