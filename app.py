"""
Brain MRI Tumor Classification Using a Hybrid CNN–Quantum Machine Learning Model
--------------------------------------------------------------------------------
Pipeline (matches the project methodology):
  1. Dataset collection        – Training/ and Testing/ folders, one subfolder per class
  2. Image preprocessing       – 224x224 resize, grayscale -> 3 channels, ImageNet norm
  3. CNN feature extraction    – pretrained ResNet18, 512-d feature vector
  4. Dimensionality reduction  – PCA, 512 -> 4 features (fit on training data only)
  5. Feature scaling           – MinMaxScaler to [-pi, +pi]
  6. Quantum encoding          – AngleEmbedding on 4 qubits
  7. Quantum processing        – StronglyEntanglingLayers (trainable, L layers)
  8. Quantum measurement       – Pauli-Z expectation values (4 outputs)
  9. Classical classification  – fully connected layer + Softmax (4 classes)
 10. Model training            – Cross-Entropy loss, Adam, backpropagation
 11. Evaluation                – accuracy, precision, recall, F1, confusion matrix
 12. Final prediction          – upload an MRI image, get class + probability scores

Run:
    pip install -r requirements.txt
    streamlit run app.py

Educational/research prototype, NOT a medical diagnostic system.
"""

import io
import json
import random
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pennylane as qml
import seaborn as sns
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, models, transforms

warnings.filterwarnings("ignore")

# ----------------------------- Page configuration ------------------------------

st.set_page_config(
    page_title="Hybrid QML Brain MRI Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        background: linear-gradient(135deg, #1a2a6c 0%, #2a4d9b 55%, #4b6cb7 100%);
        padding: 1.6rem 2rem; border-radius: 14px; margin-bottom: 1.2rem;
    }
    .main-header h1 { color: #ffffff; margin: 0; font-size: 1.9rem; }
    .main-header p  { color: #d7e3ff; margin: 0.35rem 0 0 0; font-size: 1.0rem; }
    .step-chip {
        display: inline-block; background: #eef3ff; border: 1px solid #c9d8ff;
        border-radius: 8px; padding: 4px 10px; margin: 3px; font-size: 0.82rem;
    }
    .pred-card {
        border-radius: 12px; padding: 1rem 1.4rem; margin: 0.6rem 0;
        border: 1px solid #e3e3e3;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="main-header">
        <h1>🧠 Brain MRI Tumor Classification — Hybrid CNN + Quantum ML</h1>
        <p>ResNet18 → PCA → 4-Qubit Variational Quantum Circuit → Softmax classifier</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.warning(
    "Research / educational demonstration only. This application is not validated "
    "for clinical use and must not be used to diagnose or treat patients."
)

# --------------------------------- Constants -----------------------------------

APP_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = APP_DIR / "Dataset"
MODEL_DIR = APP_DIR / "saved_model"
CACHE_DIR = APP_DIR / "cache"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
N_QUBITS = 4
N_LAYERS = 2
PRETTY_NAMES = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No Tumor",
    "pituitary": "Pituitary Tumor",
}
CLASS_COLORS = {
    "glioma": "#e74c3c",
    "meningioma": "#f39c12",
    "notumor": "#2ecc71",
    "pituitary": "#8e44ad",
}

IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def pretty(name: str) -> str:
    return PRETTY_NAMES.get(name.lower(), name.title())


# ----------------------------- Model components --------------------------------

@st.cache_resource(show_spinner=False)
def load_cnn():
    """Step 3 — pretrained ResNet18 with the classification head removed (512-d)."""
    try:
        cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    except Exception:
        cnn = models.resnet18(weights=None)
        st.warning("Could not download ImageNet weights — using an untrained ResNet18.")
    cnn.fc = nn.Identity()
    cnn.eval().to(DEVICE)
    for p in cnn.parameters():
        p.requires_grad = False
    return cnn


def build_quantum_layer():
    """Steps 6–8 — AngleEmbedding + StronglyEntanglingLayers + Pauli-Z readout."""
    dev = qml.device("default.qubit", wires=N_QUBITS)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
        return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

    return qml.qnn.TorchLayer(circuit, {"weights": (N_LAYERS, N_QUBITS, 3)})


class HybridCNNQML(nn.Module):
    """Step 9 — quantum layer feeding a classical fully-connected classifier."""

    def __init__(self, n_classes):
        super().__init__()
        self.quantum_layer = build_quantum_layer()
        self.classifier = nn.Linear(N_QUBITS, n_classes)

    def forward(self, x):
        return self.classifier(self.quantum_layer(x))


def draw_quantum_circuit():
    """Render the 4-qubit circuit used by the model."""
    dev = qml.device("default.qubit", wires=N_QUBITS)

    @qml.qnode(dev)
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
        return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

    fig, _ = qml.draw_mpl(circuit, style="pennylane", decimals=1)(
        np.zeros(N_QUBITS), np.zeros((N_LAYERS, N_QUBITS, 3))
    )
    return fig


# --------------------------- Dataset & feature cache ---------------------------

def _class_counts(dataset):
    return {dataset.classes[c]: int(sum(1 for t in dataset.targets if t == c))
            for c in range(len(dataset.classes))}


def _subsample(dataset, max_per_class, seed):
    """Optionally limit images per class for quick experimentation."""
    if not max_per_class:
        return dataset
    rng = np.random.default_rng(seed)
    idx = []
    for c in range(len(dataset.classes)):
        cls_idx = [i for i, t in enumerate(dataset.targets) if t == c]
        rng.shuffle(cls_idx)
        idx.extend(cls_idx[:max_per_class])
    return Subset(dataset, sorted(idx))


def _feature_cache_key(dataset_path, max_per_class):
    counts = []
    for split in ("Training", "Testing"):
        split_dir = dataset_path / split
        if split_dir.is_dir():
            for cls_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
                counts.append(f"{split}/{cls_dir.name}:{len(list(cls_dir.iterdir()))}")
    return "_".join(counts) + f"_cap{max_per_class}"


@st.cache_data(show_spinner=False)
def prepare_features(dataset_path_str, batch_size, max_per_class, seed):
    """Steps 1–5: load images, extract CNN features (disk-cached), PCA, scaling."""
    dataset_path = Path(dataset_path_str)
    train_path, test_path = dataset_path / "Training", dataset_path / "Testing"
    if not train_path.is_dir() or not test_path.is_dir():
        raise FileNotFoundError(
            "Dataset folder must contain 'Training' and 'Testing' subfolders."
        )

    train_ds = datasets.ImageFolder(str(train_path), transform=IMAGE_TRANSFORM)
    test_ds = datasets.ImageFolder(str(test_path), transform=IMAGE_TRANSFORM)
    if set(train_ds.classes) != set(test_ds.classes):
        raise ValueError("Training and Testing class folders do not match.")
    if train_ds.classes != test_ds.classes:
        raise ValueError("Class order mismatch between splits.")

    train_view = _subsample(train_ds, max_per_class, seed)
    test_view = _subsample(test_ds, max_per_class, seed)

    cache_file = CACHE_DIR / f"features_{abs(hash(_feature_cache_key(dataset_path, max_per_class)))}.npz"
    if cache_file.exists():
        data = np.load(cache_file)
        train_f, y_train = data["train_f"], data["y_train"]
        test_f, y_test = data["test_f"], data["y_test"]
        cache_hit = True
    else:
        cnn = load_cnn()
        train_f, y_train = _extract(train_view, batch_size, cnn, "Training")
        test_f, y_test = _extract(test_view, batch_size, cnn, "Testing")
        CACHE_DIR.mkdir(exist_ok=True)
        np.savez_compressed(cache_file, train_f=train_f, y_train=y_train,
                            test_f=test_f, y_test=y_test)
        cache_hit = False

    pca = PCA(n_components=N_QUBITS, random_state=seed)
    train_red = pca.fit_transform(train_f)
    test_red = pca.transform(test_f)
    scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
    train_s = scaler.fit_transform(train_red)
    test_s = scaler.transform(test_red)

    return {
        "x_train": train_s, "y_train": y_train,
        "x_test": test_s, "y_test": y_test,
        "pca": pca, "scaler": scaler, "classes": train_ds.classes,
        "train_counts": _class_counts(train_ds), "test_counts": _class_counts(test_ds),
        "train_used": int(len(y_train)), "test_used": int(len(y_test)),
        "pca_var": pca.explained_variance_ratio_,
        "cache_hit": cache_hit,
    }


def _extract(view, batch_size, cnn, label):
    loader = DataLoader(view, batch_size=batch_size, shuffle=False, num_workers=0)
    feats, labels = [], []
    progress = st.progress(0.0)
    status = st.empty()
    total = len(loader.dataset)
    done = 0
    with torch.no_grad():
        for images, y in loader:
            f = cnn(images.to(DEVICE))
            feats.append(f.cpu().numpy())
            labels.append(y.numpy())
            done += images.size(0)
            progress.progress(min(done / total, 1.0))
            status.caption(f"Extracting ResNet18 features — {label}: {done}/{total} images")
    status.empty()
    progress.empty()
    return np.concatenate(feats), np.concatenate(labels)


# ------------------------------- Training (10) ---------------------------------

def train_hybrid(prep, epochs, batch_size, lr, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    x_train = torch.tensor(prep["x_train"], dtype=torch.float32)
    y_train = torch.tensor(prep["y_train"], dtype=torch.long)
    x_test = torch.tensor(prep["x_test"], dtype=torch.float32)
    n_classes = len(prep["classes"])

    loader = DataLoader(TensorDataset(x_train, y_train),
                        batch_size=batch_size, shuffle=True)
    model = HybridCNNQML(n_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history, progress, status = [], st.progress(0.0), st.empty()
    for epoch in range(epochs):
        model.train()
        loss_sum, correct, total = 0.0, 0, 0
        for bx, by in loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * bx.size(0)
            correct += (out.argmax(1) == by).sum().item()
            total += bx.size(0)
        history.append({"Epoch": epoch + 1,
                        "Training Loss": loss_sum / max(total, 1),
                        "Training Accuracy": correct / max(total, 1)})
        progress.progress((epoch + 1) / epochs)
        status.caption(
            f"Epoch {epoch + 1}/{epochs} — loss {history[-1]['Training Loss']:.4f} "
            f"— acc {history[-1]['Training Accuracy']:.3f}"
        )
    status.empty()
    progress.empty()

    # --- Evaluation on held-out Testing set (step 11) ---
    model.eval()
    logits = []
    with torch.no_grad():
        for i in range(0, len(x_test), batch_size):
            logits.append(model(x_test[i:i + batch_size].to(DEVICE)).cpu())
    logits = torch.cat(logits)
    y_pred = logits.argmax(1).numpy()
    probs = torch.softmax(logits, 1).numpy()
    y_true = prep["y_test"]

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(
        y_true, y_pred, labels=list(range(n_classes)),
        target_names=[pretty(c) for c in prep["classes"]],
        output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))

    return {
        "model": model,
        "history": pd.DataFrame(history),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision, "recall": recall, "f1": f1,
        "report": pd.DataFrame(report).T,
        "cm": cm, "y_true": y_true, "y_pred": y_pred, "probs": probs,
    }


# ----------------------------- Save / load model --------------------------------

def save_artifacts(prep, result):
    MODEL_DIR.mkdir(exist_ok=True)
    torch.save({
        "state_dict": result["model"].cpu().state_dict(),
        "classes": prep["classes"], "n_qubits": N_QUBITS, "n_layers": N_LAYERS,
    }, MODEL_DIR / "hybrid_qml_model.pth")
    result["model"].to(DEVICE)
    joblib.dump(prep["pca"], MODEL_DIR / "pca.joblib")
    joblib.dump(prep["scaler"], MODEL_DIR / "scaler.joblib")
    (MODEL_DIR / "meta.json").write_text(json.dumps({
        "classes": list(prep["classes"]),
        "accuracy": float(result["accuracy"]),
        "f1": float(result["f1"]),
    }, indent=2))


def load_artifacts():
    ckpt = torch.load(MODEL_DIR / "hybrid_qml_model.pth", map_location=DEVICE)
    classes = ckpt["classes"]
    model = HybridCNNQML(len(classes)).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    meta = json.loads((MODEL_DIR / "meta.json").read_text()) \
        if (MODEL_DIR / "meta.json").exists() else {}
    return {
        "model": model, "classes": classes,
        "pca": joblib.load(MODEL_DIR / "pca.joblib"),
        "scaler": joblib.load(MODEL_DIR / "scaler.joblib"),
        "meta": meta, "loaded_from_disk": True,
    }


def predict(image: Image.Image, pipe):
    """Step 12 — full inference chain for one uploaded image."""
    tensor = IMAGE_TRANSFORM(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feats = load_cnn()(tensor).cpu().numpy()
    scaled = pipe["scaler"].transform(pipe["pca"].transform(feats))
    x = torch.tensor(scaled, dtype=torch.float32).to(DEVICE)
    pipe["model"].eval()
    with torch.no_grad():
        probs = torch.softmax(pipe["model"](x), dim=1).cpu().numpy()[0]
    idx = int(np.argmax(probs))
    return pipe["classes"][idx], probs


# ----------------------------------- Sidebar ------------------------------------

with st.sidebar:
    st.header("⚙️ Configuration")
    dataset_path = st.text_input(
        "Dataset folder", value=str(DEFAULT_DATASET),
        help="Folder containing Training/ and Testing/ subfolders.")
    epochs = st.slider("Training epochs", 1, 30, 8)
    batch_size = st.selectbox("Batch size", [8, 16, 32, 64], index=2)
    lr = st.select_slider("Learning rate",
                          options=[0.0001, 0.0005, 0.001, 0.005, 0.01], value=0.001)
    seed = st.number_input("Random seed", 0, 9999, 42)
    max_per_class = st.select_slider(
        "Max images per class (0 = all)", options=[0, 50, 100, 200, 400, 800], value=0,
        help="Use a small value for a quick demo run.")
    st.caption(f"Device: **{DEVICE}** · Qubits: **{N_QUBITS}** · Layers: **{N_LAYERS}**")
    st.divider()
    train_btn = st.button("🚀 Train / Evaluate", type="primary",
                          use_container_width=True)
    saved_exists = (MODEL_DIR / "hybrid_qml_model.pth").exists()
    load_btn = st.button("📂 Load saved model", use_container_width=True,
                         disabled=not saved_exists)
    if not saved_exists:
        st.caption("No saved model yet — train once to enable loading.")

if "pipe" not in st.session_state:
    st.session_state.pipe = None
if "result" not in st.session_state:
    st.session_state.result = None

if train_btn:
    try:
        with st.spinner("Preparing features (CNN → PCA → scaling)..."):
            prep = prepare_features(dataset_path, int(batch_size),
                                    int(max_per_class), int(seed))
        if prep["cache_hit"]:
            st.toast("CNN features loaded from cache — skipping extraction.", icon="⚡")
        with st.spinner("Training the hybrid quantum model..."):
            result = train_hybrid(prep, int(epochs), int(batch_size), float(lr), int(seed))
        save_artifacts(prep, result)
        st.session_state.pipe = {
            "model": result["model"], "classes": prep["classes"],
            "pca": prep["pca"], "scaler": prep["scaler"],
            "meta": {"classes": list(prep["classes"]),
                     "accuracy": float(result["accuracy"]),
                     "f1": float(result["f1"])},
            "loaded_from_disk": False,
        }
        st.session_state.result = {**result, "prep": prep}
        st.success("Training finished — model saved to `saved_model/`.")
    except Exception as e:
        st.session_state.pipe = st.session_state.result = None
        st.error(f"Training failed: {e}")

if load_btn and st.session_state.pipe is None:
    try:
        st.session_state.pipe = load_artifacts()
        st.toast("Saved model loaded — ready for prediction.", icon="✅")
    except Exception as e:
        st.error(f"Could not load saved model: {e}")

pipe = st.session_state.pipe
result = st.session_state.result

# ---------------------------------- Tabs ----------------------------------------

tab_overview, tab_train, tab_circuit, tab_predict, tab_about = st.tabs(
    ["🏠 Overview", "📊 Training & Evaluation", "⚛️ Quantum Circuit",
     "🔍 Predict", "ℹ️ About"])

# --- Tab 1: dataset overview ----------------------------------------------------
with tab_overview:
    st.subheader("Pipeline overview")
    steps = ["Dataset", "Preprocess 224×224", "ResNet18 (512-d)", "PCA → 4",
             "Scale [-π, π]", "AngleEmbedding", "StronglyEntanglingLayers",
             "Pauli-Z ⟨Z⟩", "FC + Softmax", "Adam / CE loss",
             "Metrics & CM", "Predict"]
    st.markdown("".join(
        f'<span class="step-chip">{i + 1}. {s}</span>' for i, s in enumerate(steps)),
        unsafe_allow_html=True)

    ds_path = Path(dataset_path)
    if (ds_path / "Training").is_dir():
        st.divider()
        st.subheader("Dataset — class distribution")
        counts = {}
        for split in ("Training", "Testing"):
            split_dir = ds_path / split
            if split_dir.is_dir():
                counts[split] = {
                    pretty(p.name): len(list(p.iterdir()))
                    for p in sorted(split_dir.iterdir()) if p.is_dir()
                }
        if counts:
            df = pd.DataFrame(counts).fillna(0).astype(int)
            c1, c2 = st.columns([1, 2])
            c1.dataframe(df, use_container_width=True)
            c2.bar_chart(df)

        st.divider()
        st.subheader("Sample MRI images")
        n_samples = st.slider("Samples per class", 1, 6, 3, key="n_samples")
        rng = random.Random(7)
        for cls_dir in sorted(p for p in (ds_path / "Training").iterdir() if p.is_dir()):
            st.markdown(f"**{pretty(cls_dir.name)}**")
            files = [f for f in cls_dir.iterdir()
                     if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
            cols = st.columns(n_samples)
            for col, f in zip(cols, rng.sample(files, min(n_samples, len(files)))):
                col.image(str(f), use_column_width=True)
    else:
        st.info("Set a valid dataset folder in the sidebar to preview images.")

# --- Tab 2: training & evaluation ----------------------------------------------
with tab_train:
    if result is None:
        st.info(
            "No training results yet — configure parameters in the sidebar and "
            "press **Train / Evaluate**. Feature extraction is cached on disk, so "
            "re-training with different epochs/LR is fast."
        )
    else:
        prep = result["prep"]
        st.subheader("Held-out test performance")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{result['accuracy']:.2%}")
        c2.metric("Macro Precision", f"{result['precision']:.3f}")
        c3.metric("Macro Recall", f"{result['recall']:.3f}")
        c4.metric("Macro F1-score", f"{result['f1']:.3f}")
        st.caption(
            f"Train images used: {prep['train_used']} · Test images used: "
            f"{prep['test_used']} · PCA variance retained (4 comps): "
            f"{sum(prep['pca_var']):.2%} · Device: {DEVICE}"
        )

        left, right = st.columns(2)
        with left:
            st.markdown("**Training history**")
            st.line_chart(
                result["history"].set_index("Epoch")
                [["Training Loss", "Training Accuracy"]])
            st.dataframe(result["history"].round(4), use_container_width=True)
        with right:
            st.markdown("**Confusion matrix**")
            fig, ax = plt.subplots(figsize=(6, 4.6))
            names = [pretty(c) for c in prep["classes"]]
            sns.heatmap(result["cm"], annot=True, fmt="d", cmap="Blues",
                        xticklabels=names, yticklabels=names, ax=ax)
            ax.set_xlabel("Predicted label")
            ax.set_ylabel("True label")
            plt.xticks(rotation=30, ha="right")
            st.pyplot(fig)
            plt.close(fig)

        st.markdown("**Per-class classification report**")
        st.dataframe(result["report"].round(4), use_container_width=True)

        with st.expander("PCA explained variance (4 components)"):
            st.bar_chart(pd.DataFrame(
                {"Component": [f"PC{i + 1}" for i in range(N_QUBITS)],
                 "Explained variance ratio": prep["pca_var"]}
            ).set_index("Component"))

# --- Tab 3: quantum circuit -----------------------------------------------------
with tab_circuit:
    st.subheader("Variational quantum circuit (4 qubits)")
    st.caption(
        "AngleEmbedding maps the 4 scaled PCA features to RY rotations; "
        "StronglyEntanglingLayers adds trainable rotations + CNOT entanglement; "
        "each qubit is measured via Pauli-Z expectation."
    )
    try:
        fig = draw_quantum_circuit()
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.error(f"Could not draw the circuit: {e}")
    st.code(
        "q0: ──RY(x1)──╭StronglyEntangling──⟨Z⟩\n"
        "q1: ──RY(x2)──┤StronglyEntangling──⟨Z⟩\n"
        "q2: ──RY(x3)──┤StronglyEntangling──⟨Z⟩\n"
        "q3: ──RY(x4)──╰StronglyEntangling──⟨Z⟩",
        language="text",
    )
    st.markdown(
        f"- **Trainable parameters:** {N_LAYERS} layers × {N_QUBITS} qubits × "
        f"3 rotations = **{N_LAYERS * N_QUBITS * 3}** quantum weights\n"
        f"- **Simulator:** `default.qubit` (PennyLane), backprop differentiation\n"
        f"- **Readout:** 4 Pauli-Z expectation values → `nn.Linear(4, 4)` → Softmax"
    )

# --- Tab 4: prediction ----------------------------------------------------------
with tab_predict:
    st.subheader("Classify a new brain MRI image")
    uploaded = st.file_uploader(
        "Upload an MRI scan", type=["jpg", "jpeg", "png", "bmp"])

    if pipe is None:
        st.info("Train a model (sidebar → Train) or click **Load saved model**.")
    else:
        meta = pipe.get("meta", {})
        if meta:
            st.caption(
                f"Active model — classes: {', '.join(pretty(c) for c in pipe['classes'])}"
                + (f" · last test accuracy: {meta.get('accuracy', 0):.1%}"
                   if "accuracy" in meta else "")
                + (" · loaded from disk" if pipe.get("loaded_from_disk") else "")
            )

    if uploaded is not None:
        image = Image.open(io.BytesIO(uploaded.getvalue()))
        col_img, col_res = st.columns([1, 1.3])
        col_img.image(image, caption="Uploaded MRI", width=320)
        if pipe is None:
            col_res.warning("No model available.")
        elif col_res.button("🔎 Classify", type="primary"):
            with st.spinner("Running CNN → PCA → quantum circuit..."):
                pred_class, probs = predict(image, pipe)
            st.session_state.last_pred = (uploaded.name, pred_class, probs)

        last = st.session_state.get("last_pred")
        if last and last[0] == uploaded.name and pipe:
            _, pred_class, probs = last
            color = CLASS_COLORS.get(pred_class, "#2a4d9b")
            col_res.markdown(
                f'<div class="pred-card" style="background:{color}18;'
                f'border-color:{color};">'
                f'<h3 style="color:{color};margin:0;">'
                f'{pretty(pred_class)}</h3>'
                f'<p style="margin:4px 0 0 0;">Confidence: '
                f'<b>{probs[list(pipe["classes"]).index(pred_class)]:.1%}</b></p>'
                f"</div>",
                unsafe_allow_html=True,
            )
            prob_df = pd.DataFrame(
                {"Class": [pretty(c) for c in pipe["classes"]],
                 "Probability": probs}).set_index("Class")
            fig, ax = plt.subplots(figsize=(5, 2.6))
            colors = [CLASS_COLORS.get(c, "#2a4d9b") for c in pipe["classes"]]
            ax.barh(prob_df.index, prob_df["Probability"], color=colors)
            ax.set_xlim(0, 1)
            ax.set_xlabel("Probability")
            ax.invert_yaxis()
            for i, v in enumerate(prob_df["Probability"]):
                ax.text(v + 0.02, i, f"{v:.1%}", va="center", fontsize=9)
            plt.tight_layout()
            col_res.pyplot(fig)
            plt.close(fig)
            col_res.caption(
                "Softmax scores are not clinically calibrated probabilities.")

# --- Tab 5: about ---------------------------------------------------------------
with tab_about:
    st.subheader("Methodology")
    st.markdown(
        """
        | # | Stage | Implementation |
        |---|-------|----------------|
        | 1 | Dataset | 4 classes: Glioma, Meningioma, No Tumor, Pituitary |
        | 2 | Preprocessing | 224×224 resize, grayscale→3ch, ImageNet normalization |
        | 3 | CNN features | Pretrained ResNet18, frozen → 512-d vector |
        | 4 | PCA | 512 → 4 components (fit on training split only) |
        | 5 | Scaling | MinMaxScaler → [-π, +π] |
        | 6 | Encoding | `qml.AngleEmbedding` (RY) on 4 qubits |
        | 7 | Ansatz | `qml.StronglyEntanglingLayers` (2 layers, trainable) |
        | 8 | Measurement | Pauli-Z expectation values ×4 |
        | 9 | Classifier | `nn.Linear(4→4)` + Softmax |
        | 10 | Training | Cross-Entropy loss, Adam, backprop through the QNode |
        | 11 | Evaluation | Accuracy, precision, recall, F1, confusion matrix |
        | 12 | Prediction | Image upload → full pipeline → class + probabilities |
        """
    )
    st.markdown(
        """
        **Notes**
        - The quantum circuit runs on PennyLane's `default.qubit` simulator — no
          hardware advantage is claimed.
        - The CNN is frozen; only quantum weights + the linear head are trained.
        - Artifacts are saved to `saved_model/` (`.pth`, `pca.joblib`,
          `scaler.joblib`, `meta.json`) after each training run.
        - Extracted ResNet features are cached in `cache/` so hyperparameter
          sweeps skip re-extraction.
        """
    )

st.divider()
st.caption(
    "Hybrid CNN–QML educational prototype · Not for clinical diagnosis · "
    "Respect the dataset license."
)
