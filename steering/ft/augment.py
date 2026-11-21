"""Per-sample augmentations for the radio fine-tune (selected by name: train_radio.py --prompt-aug / --image-aug).

Must be module-level callables (pickled into the torch workers).
  prompt fns:  f(ctx) -> str
  image fns:   f(images: dict cam -> (224,224,3) uint8, ctx) -> images; ctx.seg(cam) -> radio/button masks + boxes
"""

import cv2
import numpy as np

from radio_data import RADIO_PROMPT, default_prompt

PARAPHRASES = (
    RADIO_PROMPT,
    "Turn on the radio.",
    "Switch on the radio on the living room table.",
    "Press the power button of the radio.",
    "Go to the radio on the table and turn it on.",
)


def paraphrase_prompt(ctx):
    """Comet prompt half the time, otherwise a random paraphrase (language-robustness baseline)."""
    if ctx.rng.random() < 0.5:
        return RADIO_PROMPT
    return PARAPHRASES[int(ctx.rng.integers(1, len(PARAPHRASES)))]


def _outline(img, mask, color=(0, 255, 0), thickness=2):
    if not mask.any():
        return img
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img = np.ascontiguousarray(img).copy()
    cv2.drawContours(img, contours, -1, color, thickness)
    return img


def outline_radio(images, ctx, cams=("head",)):
    """Draw a green outline around the radio (from the demo's instance segmentation) in the given cameras."""
    for cam in cams:
        images[cam] = _outline(images[cam], ctx.seg(cam)["radio_mask"])
    return images


def outline_radio_p50(images, ctx):
    return outline_radio(images, ctx) if ctx.rng.random() < 0.5 else images


def box_radio(images, ctx, cams=("head",)):
    """Draw the radio's 2-D bounding box (green) in the given cameras."""
    for cam in cams:
        x0, y0, x1, y1 = (int(v) for v in ctx.seg(cam)["radio_box"])
        if x0 >= 0:
            images[cam] = cv2.rectangle(np.ascontiguousarray(images[cam]).copy(), (x0, y0), (x1, y1), (0, 255, 0), 2)
    return images


PROMPT_AUGS = {"none": default_prompt, "paraphrase": paraphrase_prompt}
IMAGE_AUGS = {"none": None, "outline": outline_radio, "outline_p50": outline_radio_p50, "box": box_radio}
