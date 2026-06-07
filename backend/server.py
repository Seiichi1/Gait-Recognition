"""
Flask app used for Vercel deployment.
"""
from __future__ import annotations

import base64
import io
import os
from typing import Dict

from flask import Flask, jsonify, render_template, request
from PIL import Image

from src.demo import infer_identity, make_montage, prepare_sequence, to_uint8

base_dir = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, 
            template_folder=os.path.join(base_dir, 'templates'),
            static_folder=os.path.join(base_dir, 'static'))


def image_to_data_url(array) -> str:
    image = Image.fromarray(to_uint8(array))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def build_payload(occlusion: str, identity_seed: int) -> Dict[str, object]:
    raw_frames, processed_frames, gei, geni = prepare_sequence(occlusion, identity_seed - 1)
    result = infer_identity(processed_frames)
    sample_indices = [0, len(raw_frames) // 2, len(raw_frames) - 1]
    return {
        "occlusion": occlusion,
        "result": result,
        "images": {
            "montage": image_to_data_url(make_montage(processed_frames, every=5)),
            "gei": image_to_data_url(gei),
            "geni": image_to_data_url(geni),
            "raw": [image_to_data_url(raw_frames[idx]) for idx in sample_indices],
        },
    }

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/demo", methods=["POST"])
def demo_api():
    payload = request.get_json(silent=True) or {}
    occlusion = str(payload.get("occlusion", "CO")).upper()
    if occlusion not in {"NO", "CA", "CO", "ST"}:
        occlusion = "CO"
    identity_seed = int(payload.get("identity_seed", 2))
    identity_seed = max(1, min(4, identity_seed))
    return jsonify(build_payload(occlusion, identity_seed))
@app.route("/api/recognize", methods=["POST"])
def recognize_api():
    occ_type = request.form.get("occ_type", "NO")
    
    # We need to expose the silhouette_sequence in build_payload
    raw_frames, processed_frames, gei, geni = prepare_sequence(occ_type, 0)
    result = infer_identity(processed_frames)
    
    # Map result["top_matches"] to what main.js expects
    formatted_matches = [
        {"identity": m["label"], "confidence": m["score"] / 100.0}
        for m in result["top_matches"]
    ]
    
    return jsonify({
        "top_matches": formatted_matches,
        "silhouette_sequence": processed_frames.tolist(),
        "occ_type": occ_type
    })

if __name__ == "__main__":
    app.run(port=5000, debug=True)
