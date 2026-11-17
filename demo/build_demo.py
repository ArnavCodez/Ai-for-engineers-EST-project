"""Build demo/index.html: one offline page holding M1, M3, the 270 test images and the saved results.

Run from the project folder with the project venv:
    python demo/build_demo.py
Needs outputs/models/*.keras and data/raw/NEU-DET/ (both come with the submission zip).
"""
import base64
import io
import json
import os
import re
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODELS = ROOT / "outputs" / "models"
RESULTS = ROOT / "outputs" / "results"
DATA_RAW = ROOT / "data" / "raw"
CHECK_IMAGES = [0, 110, 200]
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


def pack(arrays, dtype):
    manifest, chunks, offset = [], [], 0
    for name, arr in arrays.items():
        flat = np.ascontiguousarray(arr, dtype=dtype).ravel()
        manifest.append({"name": name, "shape": list(arr.shape), "offset": offset, "size": int(flat.size)})
        chunks.append(flat.tobytes())
        offset += flat.size
    return manifest, base64.b64encode(b"".join(chunks)).decode("ascii")


def inbound_names(layer_cfg):
    names = []

    def walk(obj):
        if isinstance(obj, dict):
            if obj.get("class_name") == "__keras_tensor__":
                names.append(obj["config"]["keras_history"][0])
                return
            for value in obj.values():
                walk(value)
        elif isinstance(obj, (list, tuple)):
            for value in obj:
                walk(value)

    walk(layer_cfg["inbound_nodes"])
    return names


def export_m1(model):
    weights = {}
    for name in ["conv2d", "conv2d_1", "conv2d_2", "dense", "dense_1"]:
        kernel, bias = model.get_layer(name).get_weights()
        weights[f"{name}/kernel"], weights[f"{name}/bias"] = kernel, bias
    return weights


def export_m3(model):
    """MobileNetV2 as a flat graph with every BatchNormalization folded into the conv before it."""
    base = model.get_layer("mobilenetv2_1.00_224")
    cfg_by_name = {c["config"]["name"]: c for c in base.get_config()["layers"]}
    nodes, weights, alias, consumers = [], {}, {}, {}

    for layer in base.layers:
        cfg = cfg_by_name[layer.name]
        inputs = [alias.get(n, n) for n in inbound_names(cfg)]
        for n in inputs:
            consumers[n] = consumers.get(n, 0) + 1
        kind = layer.__class__.__name__
        node = {"name": layer.name, "inputs": inputs}
        if kind == "InputLayer":
            node["op"] = "input"
        elif kind in ("Conv2D", "DepthwiseConv2D"):
            assert not layer.use_bias
            node.update(op="conv" if kind == "Conv2D" else "dw", strides=layer.strides[0], padding=layer.padding)
            weights[f"{layer.name}/kernel"] = layer.get_weights()[0].astype(np.float64)
        elif kind == "BatchNormalization":
            gamma, beta, mean, var = layer.get_weights()
            scale = gamma / np.sqrt(var + layer.epsilon)
            producer = inputs[0]
            assert consumers[producer] == 1, producer
            kernel = weights[f"{producer}/kernel"]
            weights[f"{producer}/kernel"] = kernel * (scale if kernel.shape[-1] == scale.size else scale[:, None])
            weights[f"{producer}/bias"] = beta - mean * scale
            alias[layer.name] = producer
            continue
        elif kind == "ReLU":
            assert float(layer.max_value) == 6.0 and float(layer.negative_slope) == 0.0
            node["op"] = "relu6"
        elif kind == "ZeroPadding2D":
            node.update(op="pad", pad=[list(p) for p in layer.padding])
        elif kind == "Add":
            node["op"] = "add"
        else:
            raise ValueError(f"unexpected layer {kind}")
        nodes.append(node)

    kernel, bias = model.get_layer("dense").get_weights()
    weights["dense/kernel"], weights["dense/bias"] = kernel, bias
    return nodes, weights


def run_m3_graph(nodes, weights, x):
    t = {}
    for n in nodes:
        ins = [t[i] for i in n["inputs"]]
        if n["op"] == "input":
            y = x
        elif n["op"] == "conv":
            y = tf.nn.conv2d(ins[0], weights[n["name"] + "/kernel"], n["strides"], n["padding"].upper())
            y = y + weights[n["name"] + "/bias"]
        elif n["op"] == "dw":
            s = n["strides"]
            y = tf.nn.depthwise_conv2d(ins[0], weights[n["name"] + "/kernel"], [1, s, s, 1], n["padding"].upper())
            y = y + weights[n["name"] + "/bias"]
        elif n["op"] == "relu6":
            y = tf.nn.relu6(ins[0])
        elif n["op"] == "pad":
            (t0, b0), (l0, r0) = n["pad"]
            y = tf.pad(ins[0], [[0, 0], [t0, b0], [l0, r0], [0, 0]])
        else:
            y = ins[0] + ins[1]
        t[n["name"]] = y
    feats = tf.reduce_mean(t[nodes[-1]["name"]], axis=[1, 2])
    return tf.nn.softmax(feats @ weights["dense/kernel"] + weights["dense/bias"]).numpy()


def load_test_images(test):
    """Decode exactly as the notebooks did; browsers decode these colour JPEGs to slightly different gray values."""
    lossless, gray = [], []
    for rel in test["filepath"]:
        pixels = tf.io.decode_jpeg((DATA_RAW / rel).read_bytes(), channels=1).numpy()
        buf = io.BytesIO()
        Image.fromarray(pixels[..., 0]).save(buf, "PNG", optimize=True)
        lossless.append(buf.getvalue())
        gray.append(pixels.astype(np.float32) / 255.0)
    return lossless, np.stack(gray)


def main():
    test = pd.read_csv(ROOT / "data" / "splits" / "test.csv")
    saved = pd.read_csv(RESULTS / "test_predictions.csv")
    assert (saved["filepath"] == test["filepath"]).all()
    lossless, gray = load_test_images(test)

    m1 = keras.models.load_model(MODELS / "m1_baseline.keras")
    m3 = keras.models.load_model(MODELS / "m3_mobilenetv2.keras")
    m1_weights = export_m1(m1)
    m3_nodes, m3_weights = export_m3(m3)
    m3_half = {k: v.astype(np.float16).astype(np.float32) for k, v in m3_weights.items()}

    x224 = tf.image.resize(gray, (224, 224)).numpy()
    x224 = np.repeat(x224 * 2.0 - 1.0, 3, axis=-1)
    keras_probs = m3.predict(x224, verbose=0)
    graph_probs = np.concatenate([run_m3_graph(m3_nodes, m3_half, x224[i:i + 30]) for i in range(0, len(x224), 30)])
    agree = float((graph_probs.argmax(1) == keras_probs.argmax(1)).mean())
    print(f"M3 folded float16 graph vs Keras: same class on {agree:.2%} of test images, "
          f"largest probability difference {np.abs(graph_probs - keras_probs).max():.4f}")
    assert agree == 1.0
    m1_probs = m1.predict(gray, verbose=0)
    m1_saved = (saved["pred_m1_baseline"] == np.array(CLASSES)[m1_probs.argmax(1)]).mean()
    print(f"M1 Keras predictions match the saved test predictions on {m1_saved:.2%} of images")

    m1_manifest, m1_b64 = pack(m1_weights, np.float32)
    m3_manifest, m3_b64 = pack(m3_half, np.float16)

    metrics = pd.read_csv(RESULTS / "test_metrics.csv")
    robustness = pd.read_csv(RESULTS / "robustness.csv")
    split = json.loads((RESULTS / "split_summary.json").read_text(encoding="utf-8"))
    images = []
    for i, row in saved.iterrows():
        images.append({
            "file": Path(row["filepath"]).name,
            "label": CLASSES.index(row["label"]),
            "m1": [CLASSES.index(row["pred_m1_baseline"]), round(float(row["conf_m1_baseline"]), 4)],
            "m3": [CLASSES.index(row["pred_m3_mobilenetv2"]), round(float(row["conf_m3_mobilenetv2"]), 4)],
            "img": base64.b64encode(lossless[i]).decode("ascii"),
        })
    meta = {
        "classes": CLASSES,
        "imageType": "image/png",
        "check": {
            "idx": CHECK_IMAGES,
            "m1": m1_probs[CHECK_IMAGES].round(6).tolist(),
            "m3": keras_probs[CHECK_IMAGES].round(6).tolist(),
        },
        "m1": {"manifest": m1_manifest},
        "m3": {"manifest": m3_manifest, "graph": m3_nodes},
        "images": images,
        "metrics": metrics.to_dict(orient="records"),
        "robustness": robustness.to_dict(orient="records"),
        "split": split["images"],
    }

    src = HERE / "src"
    tfjs = (HERE / "vendor" / "tf.min.js").read_text(encoding="utf-8")
    page = (src / "index.html").read_text(encoding="utf-8")
    parts = {
        "/*CSS*/": (src / "style.css").read_text(encoding="utf-8"),
        "/*TFJS*/": tfjs,
        "/*APP*/": (src / "app.js").read_text(encoding="utf-8"),
        "M1_WEIGHTS": m1_b64,
        "M3_WEIGHTS": m3_b64,
        "/*META*/": json.dumps(meta, separators=(",", ":")),
    }
    for key, value in parts.items():
        assert page.count(key) == 1, key
        page = page.replace(key, re.sub(r"</(script)", r"<\\/\1", value, flags=re.I))
    out = HERE / "index.html"
    out.write_text(page, encoding="utf-8", newline="\n")
    print(f"wrote {out.relative_to(ROOT)} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
