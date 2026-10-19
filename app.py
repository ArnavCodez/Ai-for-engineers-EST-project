"""Streamlit demo: steel surface defect classifier (NEU-DET, 6 classes).

Run from this folder (venv active):
    streamlit run app.py

Two tabs:
- Classify: pick a test-split image or upload your own, optionally change its
  brightness, contrast, noise, blur or rotation, and see what the baseline CNN (M1)
  and the MobileNetV2 transfer model (M3) predict.
- Model performance: test metrics, confusion matrices, training curves,
  robustness and the TFLite export, from outputs/results.
"""
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "outputs" / "models"
RESULTS_DIR = ROOT / "outputs" / "results"
DATA_ROOT = ROOT / "data" / "raw"
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]

MODEL_SPECS = {
    "M1 baseline CNN": ("m1_baseline", 200, False),
    "M3 MobileNetV2 (transfer)": ("m3_mobilenetv2", 224, True),
}
ALL_MODELS = {"m1_baseline": "M1 baseline CNN", "m2_improved": "M2 improved CNN",
              "m3_mobilenetv2": "M3 MobileNetV2"}


@st.cache_resource
def load_model(name):
    import keras
    return keras.models.load_model(MODELS_DIR / f"{name}.keras")


@st.cache_data
def load_csv(name):
    return pd.read_csv(RESULTS_DIR / name)


@st.cache_data
def test_split():
    return pd.read_csv(ROOT / "data" / "splits" / "test.csv")


def prepare(img, size, three_channels):
    from keras.applications.mobilenet_v2 import preprocess_input
    arr = np.asarray(img.convert("L").resize((size, size), Image.BILINEAR), dtype="float32")[None, ..., None]
    if three_channels:
        return preprocess_input(np.repeat(arr, 3, axis=-1))
    return arr / 255.0


def distort(img, brightness, contrast, noise, blur, rotation, seed=0):
    img = img.convert("L")
    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    if blur:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    if rotation:
        img = img.rotate(rotation, resample=Image.BILINEAR, fillcolor=128)
    if noise:
        arr = np.asarray(img, dtype="float32") / 255.0
        arr += np.random.default_rng(seed).normal(0, noise, arr.shape)
        img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype("uint8"))
    return img


st.set_page_config(page_title="Steel defect classifier", layout="wide")
st.title("Steel surface defect classifier")
st.caption("NEU-DET, six defect classes, 270 held-out test images. Models trained in the project notebooks.")

tab_classify, tab_perf = st.tabs(["Classify an image", "Model performance"])

# ---------------------------------------------------------------- classify
with tab_classify:
    left, mid, right = st.columns([1, 1, 2])
    with left:
        st.markdown("**Image**")
        source = st.radio("Image source", ["Test split", "Upload"], horizontal=True)
        img, truth = None, None
        if source == "Test split":
            df = test_split()
            label = st.selectbox("Class", CLASSES)
            rows = df[df["label"] == label].reset_index(drop=True)
            idx = st.slider("Image", 0, len(rows) - 1, 0)
            img = Image.open(DATA_ROOT / rows.loc[idx, "filepath"])
            truth = label
        else:
            up = st.file_uploader("Steel surface image", type=["jpg", "jpeg", "png", "bmp"])
            if up:
                img = Image.open(up)

        st.markdown("**Change the image** (1.0 / 0 = unchanged)")
        brightness = st.slider("Brightness", 0.3, 1.7, 1.0, 0.05)
        contrast = st.slider("Contrast", 0.3, 1.7, 1.0, 0.05)
        noise = st.slider("Gaussian noise (std)", 0.0, 0.2, 0.0, 0.01)
        blur = st.slider("Blur radius (px)", 0.0, 4.0, 0.0, 0.25)
        rotation = st.slider("Rotation (degrees)", -45, 45, 0, 5)

    if img is None:
        mid.info("Upload an image to classify it.")
    else:
        shown = distort(img, brightness, contrast, noise, blur, rotation)
        mid.image(shown, caption=f"True class: {truth}" if truth else "Uploaded image", width="stretch")
        with right:
            probs_all = {}
            for name, (fname, size, rgb) in MODEL_SPECS.items():
                probs = load_model(fname).predict(prepare(shown, size, rgb), verbose=0)[0]
                probs_all[name] = probs
                top = int(np.argmax(probs))
                if truth is None:
                    st.metric(name, CLASSES[top], f"{probs[top]:.1%} confidence", delta_color="off")
                else:
                    ok = CLASSES[top] == truth
                    st.metric(name, CLASSES[top], f"{'correct' if ok else 'wrong'}, {probs[top]:.1%} confidence",
                              delta_color="normal" if ok else "inverse")
            probs_df = pd.DataFrame(probs_all, index=CLASSES).reset_index(names="class").melt(
                id_vars="class", var_name="model", value_name="probability")
            st.altair_chart(alt.Chart(probs_df).mark_bar().encode(
                y=alt.Y("class:N", title=None), x=alt.X("probability:Q", scale=alt.Scale(domain=[0, 1]),
                                                        axis=alt.Axis(format="%")),
                color="model:N", yOffset="model:N", tooltip=["model", "class", alt.Tooltip("probability:Q", format=".1%")]),
                width="stretch")

# ---------------------------------------------------------------- performance
with tab_perf:
    metrics = load_csv("test_metrics.csv")
    best = metrics.set_index("model").loc["m3_mobilenetv2"]
    st.subheader("Final model: M3 MobileNetV2 on the 270 test images")
    c = st.columns(5)
    c[0].metric("Accuracy", f"{best['accuracy']:.1%}")
    c[1].metric("Macro F1", f"{best['macro_f1']:.3f}")
    c[2].metric("Crack (crazing) recall", f"{best['crack_recall']:.0%}")
    c[3].metric("Errors", f"{int(best['errors'])} / 270")
    c[4].metric("TFLite size", f"{load_csv('tflite.csv').set_index('model').loc['m3_mobilenetv2', 'tflite_size_mb']:.1f} MB")

    left, right = st.columns(2)
    with left:
        st.markdown("**Test metrics by model**")
        long = metrics.assign(model=metrics["model"].map(ALL_MODELS)).melt(
            id_vars="model", value_vars=["accuracy", "macro_precision", "macro_recall", "macro_f1"],
            var_name="metric", value_name="score")
        st.altair_chart(alt.Chart(long).mark_bar().encode(
            x=alt.X("metric:N", title=None), xOffset="model:N",
            y=alt.Y("score:Q", scale=alt.Scale(domain=[0.9, 1.0]), axis=alt.Axis(format="%")),
            color="model:N", tooltip=["model", "metric", alt.Tooltip("score:Q", format=".2%")]),
            width="stretch")
    with right:
        st.markdown("**Size and speed** (CPU, one image)")
        st.dataframe(metrics.assign(model=metrics["model"].map(ALL_MODELS))[
            ["model", "params", "model_size_mb", "cpu_ms_per_image", "errors"]].rename(columns={
                "model_size_mb": "size (MB)", "cpu_ms_per_image": "ms / image"}),
            hide_index=True, width="stretch")
        st.dataframe(load_csv("tflite.csv").assign(model=lambda d: d["model"].map(ALL_MODELS))[
            ["model", "keras_size_mb", "tflite_size_mb", "tflite_val_accuracy", "same_prediction_share",
             "tflite_cpu_ms_per_image"]].rename(columns={
                "keras_size_mb": "Keras MB", "tflite_size_mb": "TFLite MB", "tflite_val_accuracy": "TFLite val acc",
                "same_prediction_share": "same prediction", "tflite_cpu_ms_per_image": "TFLite ms"}),
            hide_index=True, width="stretch")

    preds = load_csv("test_predictions.csv")
    chosen = st.selectbox("Model for the confusion matrix and per-class scores", list(ALL_MODELS),
                          index=2, format_func=ALL_MODELS.get)
    pred_col = f"pred_{chosen}"
    left, right = st.columns(2)
    with left:
        st.markdown(f"**Confusion matrix: {ALL_MODELS[chosen]}**")
        cm = preds.groupby(["label", pred_col]).size().rename("count").reset_index()
        grid = pd.MultiIndex.from_product([CLASSES, CLASSES], names=["label", pred_col]).to_frame(index=False)
        cm = grid.merge(cm, how="left").fillna(0)
        base = alt.Chart(cm).encode(x=alt.X(f"{pred_col}:N", title="predicted", sort=CLASSES),
                                    y=alt.Y("label:N", title="true", sort=CLASSES))
        st.altair_chart(
            base.mark_rect().encode(color=alt.Color("count:Q", scale=alt.Scale(scheme="blues"), legend=None))
            + base.mark_text().encode(text=alt.Text("count:Q", format=".0f"),
                                      color=alt.condition(alt.datum.count > 22, alt.value("white"), alt.value("black"))),
            width="stretch")
    with right:
        st.markdown("**Per-class precision, recall and F1**")
        rows = []
        for cls in CLASSES:
            tp = ((preds[pred_col] == cls) & (preds["label"] == cls)).sum()
            p = tp / max((preds[pred_col] == cls).sum(), 1)
            r = tp / max((preds["label"] == cls).sum(), 1)
            rows += [(cls, "precision", p), (cls, "recall", r), (cls, "F1", 2 * p * r / max(p + r, 1e-9))]
        pc = pd.DataFrame(rows, columns=["class", "metric", "score"])
        st.altair_chart(alt.Chart(pc).mark_bar().encode(
            y=alt.Y("class:N", title=None), yOffset="metric:N",
            x=alt.X("score:Q", scale=alt.Scale(domain=[0.8, 1.0]), axis=alt.Axis(format="%")),
            color="metric:N", tooltip=["class", "metric", alt.Tooltip("score:Q", format=".1%")]),
            width="stretch")

    st.markdown("**Training curves**")
    hist = pd.concat([load_csv(f"history_{m}.csv").assign(model=ALL_MODELS[m]) for m in ALL_MODELS])
    hist = hist.melt(id_vars=["epoch", "model"], value_vars=["accuracy", "val_accuracy", "loss", "val_loss"],
                     var_name="series", value_name="value")
    hist["split"] = np.where(hist["series"].str.startswith("val_"), "validation", "train")
    hist["kind"] = hist["series"].str.replace("val_", "")
    left, right = st.columns(2)
    for col, kind in [(left, "accuracy"), (right, "loss")]:
        col.altair_chart(alt.Chart(hist[hist["kind"] == kind]).mark_line().encode(
            x="epoch:Q", y=alt.Y("value:Q", title=kind), color="model:N",
            strokeDash=alt.StrokeDash("split:N", sort=["validation", "train"])), width="stretch")

    left, right = st.columns(2)
    with left:
        st.markdown("**Robustness: test accuracy under corruption**")
        rob = load_csv("robustness.csv").assign(model=lambda d: d["model"].map(ALL_MODELS))
        st.altair_chart(alt.Chart(rob).mark_line(point=True).encode(
            x=alt.X("severity:O"), y=alt.Y("accuracy:Q", axis=alt.Axis(format="%")), color="model:N",
            column=alt.Column("corruption:N", title=None),
            tooltip=["model", "corruption", "level", alt.Tooltip("accuracy:Q", format=".1%")]).properties(width=110))
    with right:
        st.markdown("**Augmentation ablation** (validation accuracy)")
        st.altair_chart(alt.Chart(load_csv("ablation.csv")).mark_bar().encode(
            y=alt.Y("augmentation:N", sort="-x", title=None),
            x=alt.X("val_accuracy:Q", scale=alt.Scale(domain=[0.9, 1.0]), axis=alt.Axis(format="%")),
            tooltip=["augmentation", "val_accuracy", "epochs"]), width="stretch")

    wrong = preds[preds[pred_col] != preds["label"]]
    st.markdown(f"**Misclassified test images: {ALL_MODELS[chosen]}** ({len(wrong)})")
    cols = st.columns(6)
    for i, (_, r) in enumerate(wrong.head(18).iterrows()):
        cols[i % 6].image(str(DATA_ROOT / r["filepath"]), width="stretch",
                          caption=f"true {r['label']} → {r[pred_col]} ({r[f'conf_{chosen}']:.0%})")
