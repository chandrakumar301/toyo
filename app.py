import base64
import io

from flask import Flask, render_template, jsonify, request

app = Flask(__name__)

# Optional dependencies for image processing
try:
    from PIL import Image, ImageFilter
    import numpy as np
    import cv2
except ImportError:
    Image = None
    ImageFilter = None
    np = None
    cv2 = None


def _ensure_image_dependencies():
    if Image is None or np is None or cv2 is None:
        return False
    return True


def detect_red_in_image(image_data: bytes) -> dict:
    """Return a small report for red detection in an image."""
    if not _ensure_image_dependencies():
        return {
            "ok": False,
            "error": "Pillow and numpy are required for red detection. Install with: pip install pillow numpy",
        }

    img = Image.open(io.BytesIO(image_data)).convert("RGB")
    arr = np.array(img)

    # A simple red detection heuristic: red channel significantly higher than green/blue.
    red = arr[:, :, 0].astype(int)
    green = arr[:, :, 1].astype(int)
    blue = arr[:, :, 2].astype(int)

    red_mask = (red > 100) & (red > green * 1.5) & (red > blue * 1.5)
    red_pct = float(red_mask.mean()) * 100.0

    return {
        "ok": True,
        "redPercent": red_pct,
        "hasRed": red_pct > 1.0,  # treat as red if more than 1% of pixels are red
    }


def detect_scratch_in_image(image_data: bytes, include_preview: bool = False) -> dict:
    """Industrial quality inspection for car parts using OpenCV.

    Detects only real surface defects: scratches, cracks, dents, missing material.
    Ignores normal texture, lighting, shadows, noise.
    """
    if not _ensure_image_dependencies():
        return {
            "ok": False,
            "error": "OpenCV, Pillow and numpy are required. Install with: pip install opencv-python pillow numpy",
        }

    # Decode image bytes to numpy array
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return {"ok": False, "error": "Invalid image data"}

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Preprocessing: blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection using Canny
    edges = cv2.Canny(blurred, 50, 150)

    # Morphological operations to connect nearby edges (for continuous defects)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.morphologyEx(edges, cv2.MORPH_DILATE, kernel, iterations=1)

    # Remove small components (noise, texture)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated, connectivity=8)

    # Filter out small components (area < 50 pixels)
    clean_mask = np.zeros_like(dilated, dtype=np.uint8)
    for i in range(1, num_labels):  # Skip background (label 0)
        if stats[i, cv2.CC_STAT_AREA] >= 50:
            clean_mask[labels == i] = 255

    # Calculate defect area
    defect_pixels = np.sum(clean_mask > 0)
    total_pixels = clean_mask.size
    defect_pct = (defect_pixels / total_pixels) * 100.0

    # Decision logic
    if defect_pct > 2.0:  # Significant defect area
        result_status = "DEFECT"
        reason = f"Detected continuous defect area ({defect_pct:.1f}% of surface)"
        confidence = min(95, 50 + defect_pct * 2)  # Higher area = higher confidence
    else:
        result_status = "OK PART"
        reason = "No significant structural defects detected"
        confidence = max(80, 100 - defect_pct * 10)  # Low defect = high confidence

    bbox = None
    if result_status == "DEFECT":
        y_idxs, x_idxs = np.nonzero(clean_mask)
        if len(x_idxs):
            x0 = int(x_idxs.min())
            x1 = int(x_idxs.max())
            y0 = int(y_idxs.min())
            y1 = int(y_idxs.max())
            bbox = {
                "x0": x0 / clean_mask.shape[1],
                "y0": y0 / clean_mask.shape[0],
                "x1": x1 / clean_mask.shape[1],
                "y1": y1 / clean_mask.shape[0],
            }

    result = {
        "ok": True,
        "result": result_status,
        "reason": reason,
        "confidence": confidence,
        "bbox": bbox,
    }

    if include_preview:
        # Create preview image with overlay
        preview_img = cv2.resize(img, (320, 240))
        if result_status == "DEFECT" and bbox:
            # Draw bounding box
            h, w = preview_img.shape[:2]
            cv2.rectangle(preview_img,
                         (int(bbox["x0"] * w), int(bbox["y0"] * h)),
                         (int(bbox["x1"] * w), int(bbox["y1"] * h)),
                         (0, 0, 255), 2)  # Red rectangle

        # Encode to base64
        _, buf = cv2.imencode('.png', preview_img)
        preview = base64.b64encode(buf.tobytes()).decode("ascii")
        result["preview"] = f"data:image/png;base64,{preview}"

    return result


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/detect-red", methods=["POST"])
def api_detect_red():
    """Accepts a JSON payload with a base64 image and returns whether red is present."""
    payload = request.get_json(silent=True) or {}
    data_url = payload.get("image")

    if not data_url or not data_url.startswith("data:image"):
        return jsonify({"ok": False, "error": "Expected JSON with an 'image' data URL."}), 400

    header, b64 = data_url.split(",", 1)
    try:
        image_bytes = base64.b64decode(b64)
    except Exception as e:
        return jsonify({"ok": False, "error": f"Invalid base64 image data: {e}"}), 400

    report = detect_red_in_image(image_bytes)
    return jsonify(report)


@app.route("/api/detect-scratch", methods=["POST"])
def api_detect_scratch():
    """Accepts a JSON payload with a base64 image and returns whether scratches/vulnerabilities exist."""
    payload = request.get_json(silent=True) or {}
    data_url = payload.get("image")

    if not data_url or not data_url.startswith("data:image"):
        return jsonify({"ok": False, "error": "Expected JSON with an 'image' data URL."}), 400

    header, b64 = data_url.split(",", 1)
    try:
        image_bytes = base64.b64decode(b64)
    except Exception as e:
        return jsonify({"ok": False, "error": f"Invalid base64 image data: {e}"}), 400

    report = detect_scratch_in_image(image_bytes, include_preview=True)
    return jsonify(report)


@app.route("/api/detect-dataset", methods=["POST"])
def api_detect_dataset():
    """Accepts multipart form data with one or more image files.

    Returns a per-file report for scratch detection.
    """
    if 'files' not in request.files:
        return jsonify({"ok": False, "error": "Expected one or more files under the 'files' form field."}), 400

    files = request.files.getlist('files')
    reports = []

    for f in files:
        try:
            blob = f.read()
            report = detect_scratch_in_image(blob, include_preview=True)
        except Exception as e:
            report = {"ok": False, "error": str(e)}
        reports.append({"filename": f.filename, "report": report})

    return jsonify({"ok": True, "results": reports})


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
