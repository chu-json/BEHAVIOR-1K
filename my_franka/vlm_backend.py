"""
Local VLM backend: Qwen2.5-VL-7B-Instruct, frozen (no fine-tuning), used through plain text prompts.

    vlm = QwenVLM()
    text = vlm.query([image_np], "Locate the apple ...")          # raw text answer
    box  = vlm.locate(image_np, "apple")                            # (x1, y1, x2, y2) in image pixels, or None
    yes  = vlm.yes_no([img1, img2], "Is the robot holding the apple?")

Qwen2.5-VL was trained to ground objects with ABSOLUTE pixel coordinates of the (internally resized) input image, in a
JSON format like [{"bbox_2d": [x1, y1, x2, y2], "label": "apple"}]. We feed images whose sides are multiples of its
28-px patch grid so no resizing happens, and still rescale by the grid size the processor reports, to be safe.
"""
import json
import os
import re
import time

import numpy as np
import torch as th
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "data", "vlm", "Qwen2.5-VL-7B-Instruct")


class QwenVLM:
    def __init__(self, model_dir=MODEL_DIR, device="cuda"):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_dir, dtype=th.bfloat16, device_map=device, attn_implementation="sdpa")
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_dir)
        self.calls, self.seconds = 0, 0.0
        self.log = []  # (prompt, answer) of every call, for debugging and the write-up

    @th.inference_mode()
    def query(self, images, prompt, max_new_tokens=128):
        pils = [Image.fromarray(np.ascontiguousarray(im)) for im in images]
        content = [{"type": "image", "image": p} for p in pils] + [{"type": "text", "text": prompt}]
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=pils, return_tensors="pt").to(self.model.device)
        t0 = time.time()
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        answer = self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        self.calls += 1
        self.seconds += time.time() - t0
        self._last_grid = inputs.get("image_grid_thw")
        self.last_input_hw = None if self._last_grid is None else (int(self._last_grid[0][1]) * 14, int(self._last_grid[0][2]) * 14)
        self.log.append((prompt, answer))
        return answer.strip()

    def _scale(self, image):
        """(sx, sy): multiply the model's coordinates by these to get pixels of @image."""
        h, w = image.shape[:2]
        grid = self._last_grid
        if grid is None:
            return 1.0, 1.0
        gh, gw = int(grid[0][1]), int(grid[0][2])  # in 14-px patches
        return w / (gw * 14), h / (gh * 14)

    def locate(self, image, name):
        """Bounding box of @name in @image as (x1, y1, x2, y2) pixels, or None if the model does not find it."""
        prompt = (f"Locate the {name} in the image and output its bounding box coordinates in JSON format, "
                  f'like [{{"bbox_2d": [x1, y1, x2, y2], "label": "{name}"}}].')
        answer = self.query([image], prompt)
        nums = None
        try:
            m = re.search(r"\[.*\]", answer, re.S)
            data = json.loads(m.group(0)) if m else None
            if isinstance(data, list) and data and "bbox_2d" in data[0]:
                nums = data[0]["bbox_2d"]
        except (json.JSONDecodeError, TypeError, KeyError):
            pass
        if nums is None:  # fall back to the first four numbers in the answer
            found = re.findall(r"-?\d+(?:\.\d+)?", answer)
            nums = [float(x) for x in found[:4]] if len(found) >= 4 else None
        if nums is None:
            return None
        sx, sy = self._scale(image)
        x1, y1, x2, y2 = (float(v) for v in nums[:4])
        return (x1 * sx, y1 * sy, x2 * sx, y2 * sy)

    def yes_no(self, images, question):
        answer = self.query(images, question + " Answer with a single word: yes or no.", max_new_tokens=8)
        return answer.lower().startswith("yes"), answer
