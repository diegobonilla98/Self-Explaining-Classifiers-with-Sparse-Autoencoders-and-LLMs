import json
import re
import time
import random
from PIL import Image, ImageDraw, ImageFont
import io
import math
import numpy as np
import base64
import replicate
import tqdm
import dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
dotenv.load_dotenv()

SHORT_ANSWER_RE = re.compile(r'(?mi)^\s*SHORT\s*ANSWER:\s*(.+?)\s*$')

def build_user_prompt(llm_obj_str: str) -> str:
    # Minimal wrapper; the SYSTEM prompt contains the instructions.
    return (
        "NEURON JSON (single neuron):\n"
        "```json\n" + llm_obj_str + "\n```\n"
        "IMAGES: Top-5 thumbnails in descending z; each image is annotated (class, z)."
    )

def replicate_run(json_input: str, image_uris, max_retries: int = 3):
    prompt = build_user_prompt(json_input)
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            chunks = replicate.run(
                "openai/gpt-4.1-mini",
                input={
                    "system_prompt": system_prompt,
                    "prompt": prompt,
                    "image_input": image_uris,
                    "temperature": 0.2,
                    "top_p": 0.9,
                    "presence_penalty": 0,
                    "frequency_penalty": 0,
                    "max_completion_tokens": 512
                }
            )
            response = "".join(chunks).strip()
            m = SHORT_ANSWER_RE.search(response)
            if not m:
                # Try a very lenient fallback (grab last line)
                last_line = response.splitlines()[-1] if response else ""
                candidate = last_line.replace(""","\"").replace(""","\"")
                m = SHORT_ANSWER_RE.search(candidate)
            if m:
                ans = m.group(1).strip().strip('"').strip()
                return ans
            else:
                print("WARN: SHORT ANSWER not found.\n---\n", response, "\n---")
                return None
        except Exception as e:
            last_err = e
            sleep_s = 0.6 * attempt + random.random() * 0.4
            time.sleep(sleep_s)
    print("ERROR: replicate_run failed:", last_err)
    return None

def annotate_thumb(path_or_img, label: str, z: float, side: int = 224):
    img = Image.open(path_or_img) if isinstance(path_or_img, str) else path_or_img
    img = img.convert("L").resize((side, side), Image.BICUBIC)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    text = f"{label}\n(z={z:.2f})"
    # simple white box behind text for readability
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=2)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.rectangle([3, 3, 7 + tw, 7 + th], fill=255)
    draw.multiline_text((5, 5), text, fill=0, font=font, spacing=2)
    return img

def image_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return "data:image/png;base64," + b64

def process_single_neuron(neuron_info):
    """Process a single neuron and return (neuron_id, answer)"""
    try:
        neuron_id = int(neuron_info["neuron_id"])
        threshold = float(neuron_info.get("threshold", 0.0))
        sparsity = float(neuron_info.get("sparsity", 0.0))

        # ----- Top classes block (map labels; carry new fields if present; else None) -----
        top_classes_in = neuron_info.get("top_classes", [])
        top_classes_out = []
        S_vals = []
        for tc in top_classes_in:
            cls_name = decode_label(str(tc.get("class", "")))
            S_val = float(tc.get("S", 0.0))
            S_vals.append(S_val)
            top_classes_out.append({
                "class": cls_name,
                "S": S_val,
                "PMI": float(tc.get("PMI", 0.0)),
                "delta_minus": float(tc.get("delta_minus", 0.0)),
                "w": float(tc.get("w", 0.0)),
                # These might already exist in your JSON; if not, keep as None
                "coverage": (None if tc.get("coverage") is None else float(tc.get("coverage"))),
                "precision": (None if tc.get("precision") is None else float(tc.get("precision")))
            })

        # ----- Anti-classes: prefer explicit field; otherwise derive from positives in provided list -----
        anti_in = neuron_info.get("anti_classes")
        if isinstance(anti_in, list) and len(anti_in) > 0:
            anti_classes = [{"class": decode_label(str(a.get("class", ""))),
                                "delta_minus": float(a.get("delta_minus", 0.0))}
                            for a in anti_in]
        else:
            # derive (best-effort) from the classes we have: Δ- > 0 means removal helps that class
            anti_candidates = [tc for tc in top_classes_out if tc["delta_minus"] > 0]
            anti_candidates.sort(key=lambda d: d["delta_minus"], reverse=True)
            anti_classes = [{"class": d["class"], "delta_minus": d["delta_minus"]}
                            for d in anti_candidates[:3]]

        # ----- Mono score: pass through if present; else compute from S we have -----
        mono_score = neuron_info.get("mono_score")
        if mono_score is None:
            mono_score = mono_from_S(S_vals)

        # ----- Build final JSON object (exact shape you requested) -----
        llm_obj = {
            "neuron_id": neuron_id,
            "sparsity": sparsity,
            "threshold": threshold,
            "top_classes": top_classes_out,
            "anti_classes": anti_classes,
            "mono_score": float(mono_score)
        }

        # ====== FASTER PIL-ONLY IMAGE PROCESSING ======
        top_examples_info = neuron_info.get("top_examples", [])
        top_examples_images = []
        for te in top_examples_info:
            label = decode_label(str(te.get("label","")))
            z = float(te.get("z", 0.0))
            img = annotate_thumb(te["thumb"], label, z, side=224)
            top_examples_images.append(image_to_data_url(img))

        llm_obj_str = json.dumps(llm_obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        answer = replicate_run(llm_obj_str, top_examples_images)
        return (neuron_id, answer)
    
    except Exception as e:
        print(f"ERROR processing neuron {neuron_info.get('neuron_id', 'unknown')}: {e}")
        return (neuron_info.get("neuron_id", -1), None)

# ---- system prompt ----
with open(r"neuron_report\system_prompt.txt", "r", encoding="utf-8") as f:
    system_prompt = f.read()

# ---- label id -> name mapping ----
with open(r"runs\standard_cnn\label_decoder.json", "r", encoding="utf-8") as f:
    label_decoder = json.load(f)

def decode_label(s: str) -> str:
    """Map 'class_123' -> real name if available; otherwise return s unchanged."""
    m = re.fullmatch(r"class_(\d+)", s)
    if m:
        return label_decoder.get(str(int(m.group(1))), s)
    return s

# ---- output file ----
output_file = r"neuron_report\llm_output.jsonl"
output_dict = {}

# ---- load neuron summaries produced by your pipeline ----
json_path_in = r"neuron_report\neurons.jsonl"
with open(json_path_in, "r", encoding="utf-8") as f:
    neuron_summary = [json.loads(line) for line in f]

# ---- utility: safe softmax -> mono_score if missing ----
def softmax(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    x = x - np.max(x)
    e = np.exp(x)
    return e / (e.sum() + 1e-12)

def mono_from_S(S_vals):
    """1 - H(p)/log K using only the available S entries (top-3..5)."""
    if not S_vals:
        return 0.0
    p = softmax(np.array(S_vals))
    H = -(p * (np.log(p + 1e-12))).sum()
    return float(max(0.0, min(1.0, 1.0 - H / (math.log(len(p) + 1e-12)))))

# ====== PARALLEL PROCESSING ======
print(f"Processing {len(neuron_summary)} neurons in parallel...")

# Determine optimal number of workers (don't overwhelm the API)
max_workers = min(10, len(neuron_summary))  # Cap at 10 concurrent requests

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    # Submit all tasks
    future_to_neuron = {executor.submit(process_single_neuron, neuron_info): neuron_info 
                       for neuron_info in neuron_summary}
    
    # Process results as they complete with progress bar
    with tqdm.tqdm(total=len(neuron_summary), desc="Processing neurons") as pbar:
        for future in as_completed(future_to_neuron):
            neuron_id, answer = future.result()
            if answer is not None:
                output_dict[neuron_id] = answer
            pbar.update(1)


with open(output_file, "w", encoding="utf-8") as f:
    for neuron_id, answer in output_dict.items():
        f.write(json.dumps({"neuron_id": neuron_id, "answer": answer}, ensure_ascii=False) + "\n")
