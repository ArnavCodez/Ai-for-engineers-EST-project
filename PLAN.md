# EST Project: Steel Surface Defect & Crack Recognition with a CNN

**UCS321 AI for Engineers, EST mini project, statement #3**

## Phase checklist (tick when acceptance criteria are met)

- [x] Phase 0: Setup & data check (2026-10-18)
- [x] Phase 1: Problem understanding (2026-10-18)
- [x] Phase 2: Exploratory data analysis (EDA) (2026-10-18)
- [x] Phase 3: Pre-processing, split & augmentation (2026-10-18)
- [x] Phase 4: Model development (CNN from scratch + transfer learning) (2026-10-18)
- [x] Phase 5: Evaluation & interpretation (the test set is used here, once) (2026-10-18)
- [x] Phase 6: Innovation extras (Grad-CAM, robustness, demo…) (2026-10-18)
- [ ] Phase 7: Deliverables & viva prep

> Before every phase, read `CONTRIBUTING.md` (especially section 4: logging) and the last entries of `LOG.md`.

---

## The problem statement (verbatim)

> Tata Steel Ltd. is automating quality inspection in its manufacturing process. Develop a CNN-based image recognition system to identify surface cracks in steel plates from high-resolution images, enabling predictive maintenance and reducing defects in production.

**What we are building, in one line:** a CNN that looks at a grayscale image of a hot-rolled steel surface and says which defect it shows (including **crazing**, a network of fine surface cracks), with heatmaps showing *where* it saw the defect.

### Rubric
The EST rubric was **"shared on email"**. **Paste it here as soon as you have it**, and map each item to a phase as in the MST plan. Until then, assume the same structure as the MST rubric:
- problem understanding
- data & pre-processing
- model development
- evaluation & interpretation
- innovation
- presentation / viva

**Required by the instructions:** a **flow diagram** in which the pre-processing and visualisation steps are clearly documented (Phase 7).

---

## Dataset

### Primary: NEU Surface Defect Database (NEU-CLS), Northeastern University, China
- **1,800 grayscale images, 200 × 200 px:** 6 classes × 300 images of hot-rolled steel strip surfaces.

| Class | Short code | What it looks like (verify with sources in Phase 1) |
|---|---|---|
| Crazing | Cr | A network of fine surface **cracks**, our "crack" class |
| Inclusion | In | Non-metallic particles trapped in the surface |
| Patches | Pa | Irregular blotchy areas |
| Pitted surface | PS | Small pits / craters |
| Rolled-in scale | RS | Oxide scale pressed into the surface during rolling |
| Scratches | Sc | Lines from mechanical contact |

- **Where to get it:**
  - the official NEU page (faculty.neu.edu.cn/songkechen);
  - figshare "NEU-CLS";
  - several Kaggle mirrors (search "NEU surface defect database").
- **Mirror caveat:** some mirrors are the **NEU-DET** detection version (train/validation folders, XML bounding boxes). For classification we only need each image and its class, which comes from the folder name or filename prefix. If you use NEU-DET, **merge its folders and make our own split**. Log which mirror was used and its URL.
- **Verify after download:**
  - 6 classes × 300 images;
  - 200 × 200 px;
  - grayscale (some mirrors save them as 3-channel JPG/BMP; log what you find).

### Stretch only (Phase 6, I6): Severstal Steel Defect Detection (Kaggle competition)
- 12,568 training images of **1600 × 256 px**: real line-camera, "high-resolution" images.
- 4 defect classes (unnamed), with pixel masks stored in RLE format.
- Roughly half the training images have no defect.
- You must accept the competition rules on Kaggle to download it, and it is several GB.
- Use it only if time allows, to demonstrate handling high-resolution images with a sliding window.

### Related datasets (mention in the report only, don't use)
- GC10-DET (10 steel defect types)
- X-SDD (7 classes)

---

## Key nuances (read these before writing any code; log each one as a `[NUANCE]` when it shows up)

- **N1. Cracks vs defect types (`[DECISION]` in Phase 1).** The statement says "cracks". In NEU, the **crazing** class is exactly a network of fine surface cracks. We build a **6-class defect recogniser**, with crazing as the crack class. We **also report crack-vs-rest metrics** (precision/recall for crazing) so we answer the statement directly.
- **N2. "High-resolution" images.** NEU images are 200×200 crops. A real line camera produces long, high-resolution images, and a real system would cut them into tiles and classify each tile. Explain this on the slides, and optionally demonstrate it with Severstal (I6).
- **N3. Small dataset (300 images per class), so overfitting is the main danger.** Fight it with augmentation, dropout, BatchNorm, early stopping and transfer learning. Show the train vs validation curves to prove it.
- **N4. Split BEFORE augmenting, and augment the training set only.** Validation and test images are never augmented: they must look like real unseen images. Use a stratified 70/15/15 split with `SEED = 42`, and save it to `data/splits/*.csv`.
- **N5. Duplicates.** Check for exact duplicate images using file hashes. A duplicate that lands in both train and test inflates accuracy. Log the count, even if it is 0.
- **N6. Accuracy will probably be very high (around 95–100%).** The teacher may ask "isn't that too good?". Our answers:
  - NEU is clean, balanced and taken in controlled conditions;
  - we checked for duplicates and leakage;
  - our robustness test (I2) shows how accuracy drops with noise or blur;
  - a real plant would need its own data.
- **N7. Augmentation must make physical sense.**
  - Flips are fine.
  - Large rotations (90°) change the defect's direction relative to the rolling direction, which matters for scratches and rolled-in scale. Test with and without rotation (ablation I3).
  - Brightness and contrast changes mimic lighting variation.
  - No colour jitter, because the images are grayscale.
- **N8. Pretrained (ImageNet) models expect 3-channel 224×224 inputs and their own preprocessing.** For example, MobileNetV2 scales pixels to [-1, 1] with `tf.keras.applications.mobilenet_v2.preprocess_input`. So:
  - repeat the grayscale channel 3 times;
  - resize to 224;
  - use the matching `preprocess_input`.

  Forgetting this is a classic silent bug: the model still trains, just badly.
- **N9. Fine-tuning and BatchNorm.** When unfreezing pretrained layers:
  - call the base model with `training=False`, so the BatchNorm layers stay in inference mode;
  - use a very low learning rate (about 1e-5).

  Otherwise the pretrained weights get destroyed (see the Keras transfer-learning guide).
- **N10. Classes that look alike.** Crazing ↔ rolled-in scale and inclusion ↔ scratches/pitted surface often get confused. Look at the confusion matrix and the misclassified images.
- **N11. Use the test set once only**, in Phase 5. Choose models using validation data only.
- **N12. Reproducibility.** Set seeds (`tf.keras.utils.set_random_seed(42)`). Results can still vary slightly between runs. If time allows, report the final models as mean ± std over 3 runs.

---

## Tech stack

`python 3.11`, `tensorflow` (Keras 3; runs on CPU on native Windows, with Colab GPU optional), `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn` (split + metrics), `pillow` or `opencv-python`, `jupyter`, `nbconvert`. Optional: `gradio` (demo).

## Folder layout (create in Phase 0)

```
EST_Steel_Defect_CNN/
  PLAN.md  LOG.md  CONCEPTS.md  VIVA_QA.md
  requirements.txt  .gitignore
  data/raw/                 NEU images as downloaded (never modified)
  data/splits/              train.csv, val.csv, test.csv (filepath, label)
  notebooks/                01_data_check, 02_eda, 03_preprocessing, 04_models, 05_evaluation, 06_innovation
  src/                      data.py (loading/pipeline), models.py (model builders), gradcam.py, plotting.py
  outputs/figures/  outputs/results/  outputs/models/ (git-ignored)
  docs/                     problem.md, flow_diagram.md (+ .png), report/, slides/
  app/                      demo app (optional)
```

---

## Phase 0: Setup & data check

**Tasks**
1. Create the folder layout, `.gitignore` (`.venv/`, `data/raw/`, `outputs/models/`, `.ipynb_checkpoints/`) and `requirements.txt`. Then create `.venv`, install, and save `pip freeze`.
2. Check TensorFlow: log `tf.__version__` and `tf.config.list_physical_devices()`, and state whether this machine is CPU-only.
3. Download NEU into `data/raw/`, and log which source/mirror and its URL.
4. Build `notebooks/01_data_check.ipynb`:
   - count the images per class;
   - list the image sizes, modes (L/RGB) and file formats found;
   - show 1 sample per class;
   - hash every file (md5) to find exact duplicates;
   - build one master table `data/all_images.csv` with columns `filepath, label`.

**Acceptance criteria**
- [ ] The environment and TensorFlow work; versions and devices are logged.
- [ ] The class counts, sizes and modes are logged (expected 6 × 300, 200×200; if different, log why).
- [ ] The duplicate check is done and logged.
- [ ] Concepts added: digital image as a matrix of pixels, grayscale vs RGB, image resolution.

---

## Phase 1: Problem understanding

**Task:** write `docs/problem.md` (about 1–1.5 pages):
- **Context:**
  - how hot-rolled steel strip is made, in 3–4 lines;
  - why surface defects matter (customer rejection, cracks growing under load, scrap cost);
  - why manual inspection is weak (slow, tiring, inconsistent between inspectors, very fast-moving strip).
- **The six defect types:** what each looks like and a likely cause. Verify the causes from 1–2 sources and cite them in `LOG.md`.
- **Link to "predictive maintenance":** trends in defect types point to equipment problems before quality collapses. For example:
  - more rolled-in scale → check the descaling system;
  - more scratches → check the rolls and guides.

  Our classifier feeds such a trend dashboard.
- **Objective:** a 6-class image classifier, plus crack-specific (crazing) metrics (N1).
- **Success criteria:**
  - high macro-F1;
  - high **recall on cracks**, because a missed crack ships to the customer;
  - fast inference per image, because the strip moves fast;
  - a small model that can run on an edge device.
- **Constraints:** NEU is lab-style data; a real line has high-resolution images (N2).

**Acceptance criteria**
- [ ] `docs/problem.md` is written, with sources logged.
- [ ] The N1 `[DECISION]` (cracks = crazing + 6-class) is logged.
- [ ] At least 5 motivation and domain Q&As are in `VIVA_QA.md`.

---

## Phase 2: EDA

**Notebook:** `notebooks/02_split_and_eda.ipynb`. Save every figure to `outputs/figures/eda_XX_*.png`.

> **Changed 2026-10-18:** the train/validation/test split (task 1 of Phase 3) now happens at the start of this notebook, and every EDA plot uses training images only, so the test images stay unseen until Phase 5. See the `[DECISION]` in `LOG.md`.

**Required figures**
1. Class balance bar chart.
2. A sample grid of 5 random images per class (6 × 5).
3. The **mean image** per class (average of all 300). Shows typical patterns, e.g. the direction of scratches.
4. Pixel intensity histograms per class (overlaid or as subplots).
5. Boxplots of per-image mean brightness and contrast (std) for each class: could brightness alone separate the classes?
6. Optional: PCA or t-SNE of the raw pixels, coloured by class. It shows the classes overlap in raw pixel space, which is *why* we need a CNN to learn features.

**Acceptance criteria**
- [ ] At least 5 figures are saved, each with 1–2 observation bullets in `LOG.md`.
- [ ] A "Top 5 EDA insights" list is written.
- [ ] Concepts added: pixel intensity histogram, mean image, PCA/t-SNE (if used).

---

## Phase 3: Pre-processing, split & augmentation

**Notebook:** `notebooks/03_pipeline_and_augmentation.ipynb`. (Changed 2026-10-18: the split moved to notebook 02, and helper code lives in each notebook instead of `src/`, because Kaggle runs one notebook file at a time.)

1. **Split (`[DECISION]`):**
   - stratified 70/15/15 split with `SEED = 42`, saved to `data/splits/train.csv`, `val.csv` and `test.csv`;
   - if any duplicates were found, keep each duplicate group inside the same split;
   - log the counts per class per split.
2. **Loading pipeline:** `tf.data` from the CSVs:
   - read the image;
   - decode it as grayscale;
   - resize (200 for the scratch CNN, 224 for pretrained models);
   - batch (32) and prefetch.
3. **Normalisation:**
   - scratch CNN: divide by 255, giving [0, 1];
   - pretrained models: repeat to 3 channels and apply that model's `preprocess_input` (N8).
4. **Augmentation:** use Keras preprocessing layers *inside the model*, which are only active during training:
   - `RandomFlip("horizontal_and_vertical")`
   - `RandomRotation` (small, e.g. 0.05; see N7)
   - `RandomZoom(0.1)`
   - `RandomContrast(0.2)`
   - `RandomBrightness(0.2)`

   Log every choice and its reason.
5. **Figure:** one image shown with 8 augmented versions.

**Leakage checklist** (copy it into `LOG.md` and tick each item)
- [ ] The split was made before any augmentation.
- [ ] Validation and test images are never augmented.
- [ ] No duplicate image appears in two different splits.
- [ ] The test set has not been used for any decision.

**Acceptance criteria**
- [ ] The split CSVs are saved.
- [ ] The pipeline works (one batch shown, shapes logged).
- [ ] The augmentation figure is saved.
- [ ] The checklist is ticked.
- [ ] Concepts added: stratified split, normalisation, data augmentation, batch, tf.data/prefetch.

---

## Phase 4: Model development

**Notebooks:** `notebooks/04_cnn_from_scratch.ipynb` (M1, M2) and `notebooks/05_transfer_learning.ipynb` (M3, plus the comparison of all three). (Changed 2026-10-18: split in two so the scratch CNNs can run without internet, and model code lives in the notebooks because Kaggle runs one notebook file at a time.)

**Common training setup**
- Optimiser: Adam; loss: sparse categorical cross-entropy.
- Batch 32; up to 50 epochs.
- Callbacks:
  - `EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)`
  - `ReduceLROnPlateau(factor=0.5, patience=4)`
  - `ModelCheckpoint` (best `val_loss`)
- Log for every model:
  - `model.summary()` (parameter count);
  - epochs actually trained;
  - training time;
  - best validation accuracy and loss.
- Save the training-curve plots (loss and accuracy, train vs validation).

| ID | Model | Details | Purpose |
|---|---|---|---|
| M1 | Baseline CNN from scratch, **no augmentation** | 3 conv blocks [Conv2D 32→64→128, 3×3, ReLU, MaxPool 2×2] → GlobalAveragePooling → Dense 128 ReLU → Dropout 0.5 → Dense 6 softmax | Simplest CNN. Explain every layer. Expect visible overfitting |
| M2 | Improved CNN | M1 + BatchNorm + the augmentation layers + dropout | Shows the effect of regularisation and augmentation |
| M3 | Transfer learning, MobileNetV2 (ImageNet weights) | Stage A: freeze the base, train the head (GAP → Dropout 0.3 → Dense 6). Stage B: unfreeze the top ~30 layers, lr = 1e-5 (N9) | Small and fast, good for a factory edge device |
| M4 (optional) | EfficientNetB0 or ResNet50 | Same two-stage recipe | Comparison: bigger model vs accuracy vs speed |

- Save the comparison table to `outputs/results/val_results.csv` with columns: model, params, val accuracy, val loss, epochs, train time, model size (MB).
- **`[DECISION]`:** choose 2 finalists, using validation results only.

**Acceptance criteria**
- [ ] At least 3 models are trained, with curves and the table saved.
- [ ] The overfitting in M1 vs M2 is discussed, using the curves.
- [ ] The finalist decision is logged.
- [ ] Concepts added: convolution (kernel, stride, padding, feature map), ReLU, pooling, global average pooling, dense layer, softmax, cross-entropy, Adam and learning rate, epoch/batch, overfitting, dropout, BatchNorm, early stopping, transfer learning, fine-tuning, MobileNetV2 (depthwise-separable convolutions in simple words).

---

## Phase 5: Evaluation & interpretation: the test set is used here, once

**Notebook:** `notebooks/06_evaluation.ipynb` (renumbered 2026-10-18). Log the exact time the test set was first evaluated.

**Metrics on the test set, for both finalists**
- Accuracy, macro precision/recall/F1, and the per-class classification report.
- **Crack-vs-rest metrics (N1):** precision, recall and F1 for crazing.
- Inference time per image on CPU (average over the test set) and model file size (MB).
- Save them to `outputs/results/test_metrics.csv` and `classification_report.txt`.

**Required figures**
1. Confusion matrix (counts and row-normalised) for each finalist.
2. A gallery of misclassified images, each labelled with true class, predicted class and confidence.
3. Bar chart comparing all models (test accuracy and macro-F1).
4. Final training curves of the chosen model.

**Interpretation** (plain English, in the notebook and `LOG.md`)
- Which classes get confused, and why (they look visually similar; see N10).
- Is the crack class (crazing) reliable?
- Scratch CNN vs transfer learning: accuracy, speed and size trade-off.
- If accuracy is about 100%, explain why honestly (N6).

**Acceptance criteria**
- [ ] The test set is evaluated once, with the time logged.
- [ ] The metrics files are saved.
- [ ] The figures are saved.
- [ ] The interpretation is written.
- [ ] Concepts added: accuracy vs precision vs recall vs F1, macro average, confusion matrix, inference latency.

---

## Phase 6: Innovation / creativity

Do **at least 3** of these (I1 and I2 are strongly recommended). Log each with numbers and figures.

- **I1. Grad-CAM explainability.**
  - Implement Grad-CAM by hand with `tf.GradientTape`, in about 20 lines in `src/gradcam.py`, so we can explain every line.
  - Show heatmap overlays for 2 correct images per class and for the misclassified ones.
  - Viva value: it *proves* the CNN looks at the defect, not at the background.
- **I2. Robustness test (dirty camera / lighting in a steel plant).**
  - Corrupt the test images with Gaussian noise, Gaussian blur, a brightness shift and a contrast drop, each at 3 severity levels.
  - Plot accuracy vs severity for both finalists.
  - Save it to `outputs/results/robustness.csv`.
- **I3. Augmentation ablation:** M2 trained with no augmentation, with augmentation but no rotation, and with full augmentation. Compare validation and test accuracy.
- **I4. Data-efficiency curve:** train M2 and M3 on 25%, 50%, 75% and 100% of the training set. Shows that transfer learning needs much less data.
- **I5. Demo app (great for the viva):** a Gradio app where you upload an image and get the class, confidence and Grad-CAM overlay. Put it in `app/`.
- **I6. Stretch, high-resolution:**
  - train a defect/no-defect classifier on 256×256 patches cut from Severstal images (labels from the masks);
  - slide it across full 1600×256 images and draw a defect-location heatmap.

  This directly addresses "high-resolution images" (N2). Only attempt it if time allows: it needs a big download and more compute (use Colab).
- **I7. Edge deployment:** convert the final model to TensorFlow Lite and compare its size and speed with the Keras model. Shows it could run on a small device on the production line.

**Acceptance criteria**
- [ ] At least 3 items are done, each with a results file, a figure and a `LOG.md` entry.
- [ ] Each is explainable in 2 lines in `VIVA_QA.md`.

---

## Phase 7: Deliverables & viva prep

1. **Flow diagram (required):** `docs/flow_diagram.md` in Mermaid, plus an exported PNG.
   - It must show: dataset → data check (counts, duplicates) → EDA (list the plots) → split → resize/normalise → augmentation (train only) → models M1–M3 → training with callbacks → choose finalists on validation → test evaluation → Grad-CAM / robustness / demo.
   - Export with `npx -y @mermaid-js/mermaid-cli -i docs/flow_diagram.md -o docs/flow_diagram.png`, or draw it in draw.io.
2. **Final submission notebook (if the teacher wants one file):** `notebooks/EST_final.ipynb`. It must run top to bottom (it also works on Colab).
3. **Report (if asked):** in the rubric order.
4. **Slides (10–12):**
   1. Title + team
   2. Problem (Tata Steel inspection)
   3. Dataset & the 6 defects
   4. Flow diagram
   5. EDA highlights
   6. Pre-processing & augmentation
   7. CNN architecture (layer diagram)
   8. Transfer learning
   9. Results (table + confusion matrix)
   10. Grad-CAM + robustness
   11. Demo / limitations & future work
   12. Thank you
5. **Viva prep:**
   - `VIVA_QA.md` should have at least 40 Q&As.
   - Each teammate does a mock viva: pair up and quiz each other with 10 random questions from `VIVA_QA.md`, one at a time, grading against the answers.
6. **Final `LOG.md` entry:** a project summary plus the final numbers table, cross-checked against `outputs/results/`.

**Acceptance criteria**
- [ ] The flow diagram PNG exists.
- [ ] The notebooks run top to bottom.
- [ ] The slides (and report if asked) are done.
- [ ] Every number in the slides matches the results files.
- [ ] Every teammate has done a mock viva.

---

## Team split (fill in names; everyone still reads the whole LOG)

| Member | Owns | Presents |
|---|---|---|
| A: ____ | Phases 0–1, `docs/problem.md` | Problem & dataset |
| B: ____ | Phases 2–3 EDA, split, augmentation | EDA & pre-processing |
| C: ____ | Phase 4 M1/M2 (scratch CNNs) | CNN architecture |
| D: ____ | Phase 4 M3 + Phase 5 evaluation | Transfer learning & results |
| E: ____ | Phases 6–7 Grad-CAM, robustness, demo, slides | Innovation & demo |

## Deadlines (fill in)
- EST project submission / presentation: ____

## References
- NEU surface defect database (official): http://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/index.htm
- NEU dataset summary: https://github.com/jonathanwvd/awesome-industrial-datasets/blob/master/markdown/neu_surface_defect_dataset.md
- NEU-CLS on figshare: https://figshare.com/articles/dataset/NEU-CLS/28903550
- GC10-DET paper, "Deep Metallic Surface Defect Detection: The New Benchmark and Detection Network": https://pmc.ncbi.nlm.nih.gov/articles/PMC7146379/
- Severstal dataset overview: https://datasetninja.com/severstal
- Keras transfer learning & fine-tuning guide (BatchNorm nuance): https://keras.io/guides/transfer_learning/
- Before using a reference in the report, open and read it, and log what we took from it.
