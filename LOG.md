# LOG: EST Project (Steel Surface Defect CNN)

This is an **append-only** team journal. The format and rules are in `CONTRIBUTING.md` section 4.
Search tags with Ctrl+F: `[DECISION]` `[NUANCE]` `[BUG]` `[RESULT]` `[IDEA]` `[SETUP]` `[DATA]` `[EDA]`

---

## 2026-10-06 | Phase — | Project chosen and plan written  (run by: Arnav)
**Tags:** [DECISION]
**What we did:** Chose EST statement #3 (Tata Steel, CNN for surface cracks/defects in steel) and wrote `PLAN.md`.
**Why:** The team decided on #3.
**How:** Looked for public steel-surface datasets that match the statement.
**Result:** The primary dataset is NEU-CLS (6 defect classes × 300 grayscale images, 200×200). Severstal (1600×256, Kaggle) is kept as a high-resolution stretch goal.
**Nuances / gotchas:**
- The popular Kaggle "surface crack detection" dataset is **concrete**, not steel. Don't use it, because the teacher may catch that.
- The statement says "cracks": in NEU, the crazing class is the crack class (N1 in `PLAN.md`).
**Alternatives considered:**
- Severstal as the main dataset: realistic, but big, its classes are unnamed, and it needs mask (RLE) decoding. Too heavy as the base.
- GC10-DET / X-SDD: less standard.
**Viva angle:** Q: "Your images are 200×200, but the statement says high-resolution?" A: NEU images are crops of real strip images. A real system tiles high-resolution line-camera images into patches like these (N2), and our Severstal sliding-window extension shows this if we got to it.

---

## 2026-10-18 08:47 | Phase 0.1 | Project skeleton, and notebooks run on Kaggle  (run by: Arnav)
**Tags:** [SETUP] [DECISION]
**What we did:** Created the folder layout (`notebooks/`, `data/`, `outputs/`, `docs/`, `kaggle/`), `.gitignore`, `README.md` and `data/README.md`. Installed the Kaggle CLI and signed in. Wrote `kaggle/run_on_kaggle.py`, which pushes a notebook to Kaggle, waits for the run and copies its output files back into `data/` and `outputs/`.
**Why:** Kaggle gives a free GPU, already hosts the NEU dataset (no download onto the laptop), and shows each notebook with its outputs, which is how we present the work. TensorFlow on native Windows is CPU-only.
**How:** `uv tool install kaggle` (Kaggle CLI 2.2.4), `kaggle auth login` (browser sign-in, no key file handled by hand). Kernel settings per notebook (slug, GPU on/off, attached datasets) live in `kaggle/kernels.json`. Run with `python kaggle/run_on_kaggle.py 01_data_check`.
**Result:** Dataset found on Kaggle: `kaustubhdikshit/neu-surface-defect-database` (27.7 MB, NEU-DET layout). Notebooks are private on Kaggle until the team decides to make them public.
**Nuances / gotchas:**
- Every notebook is self-contained (its own setup cell, no imports from `src/`), because Kaggle runs one notebook file at a time. A notebook that needs an earlier notebook's files (e.g. `all_images.csv`) gets that notebook's output attached as an input.
- Each notebook writes to `/kaggle/working/data` and `/kaggle/working/outputs`, the same folder names as this project, so the downloaded files drop straight into place.
- The same notebooks also run on a laptop if the images are in `data/raw/NEU-DET/`.
**Alternatives considered:** Local CPU training (works for this small dataset but slow for MobileNetV2 and repeated runs); Google Colab (GPU too, but sessions disconnect and the dataset must be uploaded each time).
**Viva angle:** Q: "Where did you train?" A: On Kaggle's free GPU, using the NEU dataset hosted there. The notebooks run unchanged on a laptop too.

---

## 2026-10-18 08:52 | Phase 1 | Problem understanding, crack decision and domain notes  (run by: Arnav)
**Tags:** [DECISION] [NUANCE]
**What we did:** Wrote `docs/problem.md` (context, the six NEU defect types, link to predictive maintenance, objective, success criteria, constraints, sources). Added 5 domain concepts to `CONCEPTS.md` and 8 Q&As to `VIVA_QA.md`. Decided that cracks = the crazing class, with a 6-class recogniser plus crack-vs-rest metrics.
**Why:** The problem statement asks for "surface cracks", but NEU has six defect classes. Crazing is the crack-like class. A 6-class recogniser plus crack-vs-rest metrics answers the statement and keeps all the data.
**How:** Read web sources on hot rolling, defect causes and NEU class descriptions (12 sources, listed in `docs/problem.md`). A source was used for a fact only after we opened and read it. Where no source gave a cause, the table says "Not verified".
**Result:** Decision N1 recorded: one CNN on all 6 classes, plus precision, recall and F1 for crazing against the other five classes. Success criteria: macro-F1, recall on cracks, fast inference per image, a small model for an edge device. Target numbers are not set yet. The dataset numbers in `problem.md` (1,800 images, 6 × 300, 200 × 200 px, grayscale) come from the sources and are re-checked by notebook 01.
**Nuances / gotchas:**
- We found no source for the cause of crazing or patches, so the docs make no claim.
- Some dataset pages call the crazing class "cracks".
- Rolled-in scale and crazing look alike, so expect confusion between them.
- NEU images are 200 × 200 crops. A real line-scan camera gives long images that a real system would tile (N2). That NEU is a crop of line-scan output is our assumption; we found no source saying so.
- One defect type can have several causes, so a defect trend is a hint to inspect, not a diagnosis.
- The OXMAINT guide is an industry page, not peer reviewed; we used it only for causes, not for its percentages.
- We could not open the original NEU paper (Song and Yan, 2013) or the NEU faculty page, so the journal name is left out until someone checks it.
**Alternatives considered:** A binary crack-vs-no-crack classifier only (rejected: it throws away the other five classes). A detection model with bounding boxes (out of scope: our task is classification).
**Viva angle:** Q: "Where are the cracks in your project?" A: The crazing class is a network of fine surface cracks, and we report crack-specific recall. Q: "How is this predictive maintenance?" A: Rising rolled-in scale points to the descaling system and rising scratches point to the rolls. It is a hint to inspect, not proof.

---

## 2026-10-18 08:57 | Phase 0.2 | Data check: 1,800 images, 6 × 300, one duplicate pair  (run by: Arnav)
**Tags:** [DATA] [RESULT] [NUANCE]
**What we did:** Ran `notebooks/01_data_check.ipynb` on Kaggle. It lists every image, records size, colour mode, format and two fingerprints (hash of the file bytes and hash of the decoded pixels), shows one image per class and saves the master table.
**Why:** Before building anything we need to know the data is complete, what format it is in, and whether any picture is repeated (a repeat split across train and test would inflate the test score).
**How:** `python kaggle/run_on_kaggle.py 01_data_check`. Kaggle kernel `neu-steel-01-data-check-run`, dataset `kaustubhdikshit/neu-surface-defect-database` attached. MD5 hashes via `hashlib`; images opened with Pillow.
**Result:** (`outputs/results/data_check_summary.json`, `data/all_images.csv`)
- 1,800 images: 300 per class for crazing, inclusion, patches, pitted surface, rolled-in scale and scratches. The mirror's folders hold 1,440 (train) + 360 (validation); we ignore that split.
- All 1,800 are 200 × 200 px JPEG files stored as RGB, and in all 1,800 the three colour channels are identical. So they are grayscale images saved in a colour format.
- 1,800 XML box annotations (for detection, not used).
- 1 duplicate group: `patches_101.jpg` and `patches_105.jpg` are byte-identical, same class. The pixel hash finds the same single group.
- Environment on Kaggle: Python 3.13.15, TensorFlow 2.20.0, NumPy 2.1.3, pandas 2.3.3; versions of every package in `outputs/results/pip_freeze_kaggle.txt`.
- `outputs/figures/data_01_one_per_class.png`: the first image of each class. The six defects already look visibly different (dark blobs for patches, a bright line for scratches, small dark marks for inclusion).
**Nuances / gotchas:**
- "Grayscale stored as RGB" matters: our CNN from scratch reads one channel (`decode_jpeg(channels=1)`); MobileNetV2 gets that channel repeated three times.
- The duplicate pair is handled in notebook 02 by splitting groups of identical pixels, so both copies land in the same part.
**Alternatives considered:** Using the mirror's train/validation folders as our split (no test set, not stratified by us, and the duplicate pair could still cross over).
**Viva angle:** Q: "Did you check for duplicates?" A: Yes, by hashing every file and every decoded image. One pair (patches_101 and patches_105) was identical; we keep both copies in the same part of the split.

---

## 2026-10-18 08:58 | Phase 0.3 | Two problems with running on Kaggle  (run by: Arnav)
**Tags:** [BUG] [SETUP]
**What we did:** Fixed how notebooks run on Kaggle, after two problems in the first runs.
**Why:** We need (1) the notebook copy in our repo to contain its outputs, and (2) a GPU for training.
**How / Result:**
- **Bug 1, notebook outputs missing.** Symptom: `kaggle kernels pull` returns the notebook without outputs, and the run log only has printed text (no tables or figures). Cause: Kaggle keeps the executed notebook on its website, not in the files we can download. Fix: `kaggle/run_on_kaggle.py` now sends a short script that runs the notebook with `jupyter nbconvert --execute --inplace` inside `/kaggle/working`, so the executed `.ipynb` is one of the output files. Checked: the returned `notebooks/01_data_check.ipynb` has 13 outputs, including the figure.
- **Bug 2, no GPU.** Symptom: TensorFlow logged `failed call to cuInit: UNKNOWN ERROR (303)` and listed no GPU; in the next run `nvidia-smi` did not exist on the machine. The kernel settings did ask for a GPU (`machine_shape: NvidiaTeslaT4`). Likely cause: Kaggle only gives GPUs (and internet inside notebooks) to phone-verified accounts. Fix: pending, Arnav to check phone verification. Until then notebooks 01 to 03 run on CPU, which they do not need anything more for.
**Nuances / gotchas:** Kaggle did not report an error for the missing GPU; it silently ran on CPU. So notebook 04 prints the GPU list at the start, to make this visible.
**Alternatives considered:** Training on the laptop CPU (possible, slower); Google Colab.
**Viva angle:** —

---

## 2026-10-18 09:00 | Phase 0 summary  (run by: Arnav)
**Tags:** [RESULT]
1. **Recap:**
   - The project folder is set up like the MST repo, and every notebook runs on Kaggle through `kaggle/run_on_kaggle.py`.
   - The dataset is the Kaggle mirror of NEU (NEU-DET layout). We use only the images and their class names.
   - 1,800 images, 300 per class, all 200 × 200 px, stored as RGB but truly grayscale.
   - One duplicate pair (`patches_101.jpg` = `patches_105.jpg`).
   - Kaggle runs on CPU for now; GPU and internet wait on phone verification of the Kaggle account.
2. **Acceptance checklist:**
   - [x] Environment and TensorFlow work; versions and devices logged (Python 3.13.15, TensorFlow 2.20.0, CPU only so far).
   - [x] Class counts, sizes and modes logged.
   - [x] Duplicate check done and logged.
   - [x] Concepts added: image as a pixel matrix, grayscale vs RGB, image resolution (plus MD5 hashing).
3. **What every teammate must be able to explain:**
   - Why the files are RGB but we call the images grayscale (three identical channels).
   - How MD5 hashes find duplicates, and why one duplicate pair matters for the test score.
   - Why we ignore the mirror's own train/validation folders.
   - How a notebook gets from this folder to Kaggle and back (`kaggle/run_on_kaggle.py`).
4. **Key numbers:**

| Item | Value |
|---|---|
| Images | 1,800 (6 × 300) |
| Size | 200 × 200 px, JPEG |
| Channels identical | 1,800 of 1,800 |
| Duplicate groups | 1 (2 images) |

---

## 2026-10-18 09:01 | Phase 1 summary  (run by: Arnav)
**Tags:** [RESULT]
1. **Recap:** `docs/problem.md` explains hot rolling, the six defects with sourced causes (crazing and patches marked "Not verified"), the link to predictive maintenance and our success criteria. Crack = crazing class (N1). 12 sources read.
2. **Acceptance checklist:** [x] `docs/problem.md` written with sources; [x] N1 `[DECISION]` logged; [x] at least 5 motivation Q&As in `VIVA_QA.md` (8 added).
3. **What every teammate must be able to explain:** where the cracks are in our data; why recall matters more than precision for cracks; how defect trends point at equipment (and why that is a hint, not a diagnosis); why NEU images are small crops of what a line camera sees.
4. **Key numbers:** none measured in this phase.

---

## 2026-10-18 09:03 | Phase 2.1 | Split first, then EDA on training images only  (run by: Arnav)
**Tags:** [DECISION] [DATA] [EDA] [RESULT]
**What we did:** Ran `notebooks/02_split_and_eda.ipynb`: made the train/validation/test split, then six EDA figures on the training images. Ran it a second time to add one check (PC1 vs brightness); the split files came out byte-identical (same MD5 for all three CSVs).
**Why:** The plan had EDA before the split. We moved the split first, as in our MST project, so the test images are never looked at before the final evaluation.
**How:** Stratified 70/15/15 split with `train_test_split` (seed 42), done on groups of identical pixels (`pixel_md5`) so the duplicate pair stays together. Saved to `data/splits/{train,val,test}.csv`. EDA on 1,260 training images loaded as grayscale arrays. PCA on images shrunk to 50 × 50.
**Result:** (`outputs/results/split_summary.json`, `eda_summary.json`)
- Split: 1,260 / 270 / 270 images; 210 / 45 / 45 per class; 0 pictures in two parts (an `assert` checks this).
- `eda_01_class_balance.png`: every class has the same count in each part.
- `eda_02_sample_grid.png`: five random training images per class; patches and pitted surface vary a lot in brightness within the class.
- `eda_03_mean_image_per_class.png`: the mean images of crazing, inclusion, pitted surface and scratches get darker towards the right edge (uneven lighting); the scratches mean shows both vertical and horizontal lines; patches show dark blobs near the centre.
- `eda_04_intensity_histograms.png`: heavy overlap between classes; patches and pitted surface have a spike at 255 (over-exposed pixels).
- `eda_05_brightness_contrast_boxplots.png`: median brightness from 90.36 (scratches) to 186.05 (pitted surface), but the ranges overlap; median contrast (pixel std) 57.75 for patches, 11.18 to 28.55 for the others.
- `eda_06_pca_raw_pixels.png`: PC1 explains 70.63 % of the pixel variance and is overall brightness (correlation 1.0); PC2 8.28 %. Crazing, inclusion and rolled-in scale overlap in the middle.

**Top 5 EDA insights:**
1. The classes are perfectly balanced, so accuracy is a fair first metric.
2. Brightness alone cannot separate the classes; texture and shape must.
3. Patches stand out by contrast; inclusion and rolled-in scale are the flattest images.
4. Lighting is uneven (darker right edge), a camera effect the model should not learn; horizontal flips help.
5. In raw pixel space the biggest difference between images is brightness, which says nothing about the defect. This is why we need a CNN.

**Nuances / gotchas:**
- The PCA plot uses no labels; colours are added afterwards.
- Overlap in two PCA components does not prove the classes are inseparable, only that raw pixels are a poor representation.

**Alternatives considered:** EDA on all 1,800 images (rejected: it lets the test images influence our choices).
**Viva angle:** Q: "Why did you split before doing EDA?" A: So that nothing we decide is based on the test images; they stay unseen until the final evaluation.

---

## 2026-10-18 09:04 | Phase 2 summary  (run by: Arnav)
**Tags:** [RESULT]
1. **Recap:** fixed split (210/45/45 per class, reproducible); six EDA figures on training images; brightness differs between classes but overlaps; patches have the highest contrast; uneven lighting and over-exposed pixels exist; raw pixels separate mainly by brightness.
2. **Acceptance checklist:** [x] at least 5 figures saved, each with observations above; [x] Top 5 EDA insights written; [x] concepts added: pixel intensity histogram, mean image, PCA (plus stratified and group-aware split, data leakage).
3. **What every teammate must be able to explain:** stratified vs random split; why we split groups of identical pixels; what a mean image and a histogram show; what PCA does and why PC1 here is just brightness.
4. **Key numbers:**

| Item | Value |
|---|---|
| Train / val / test | 1,260 / 270 / 270 |
| Median contrast, patches vs rest | 57.75 vs 11.18 to 28.55 |
| PC1 variance, correlation with brightness | 70.63 %, 1.0 |

---

## 2026-10-18 09:05 | Phase 3.1 | Loading pipeline and augmentation  (run by: Arnav)
**Tags:** [DECISION] [RESULT]
**What we did:** Ran `notebooks/03_pipeline_and_augmentation.ipynb`: a `tf.data` pipeline for both input formats, the augmentation layers, a figure of augmented images and the leakage checklist.
**Why:** The models need images in the right size, channel count and pixel scale, and augmentation is our main tool against overfitting on 210 images per class.
**How:**
- `make_dataset(part, image_size, three_channels, shuffle)`: `decode_jpeg(channels=1)`, resize, then either divide by 255 (our CNN, 0 to 1) or grayscale-to-RGB plus `mobilenet_v2.preprocess_input` (MobileNetV2, −1 to 1). Batch 32, `prefetch`.
- Augmentation (Keras layers inside the model, active only in training): `RandomFlip("horizontal_and_vertical")`, `RandomRotation(0.05, fill_mode="reflect")` (about ±18°), `RandomZoom(0.1)`, `RandomContrast(0.2)`, `RandomBrightness(0.2)`.

**Result:** (`outputs/results/pipeline_check.json`)
- Scratch-CNN batch: 32 × 200 × 200 × 1, pixels 0.0 to 1.0. MobileNetV2 batch: 32 × 224 × 224 × 3, pixels −0.997 to 1.0. Both first batches contain all 6 classes.
- `prep_01_augmentation_examples.png`: crazing, rolled-in scale and scratches with eight random versions each; all still look like real steel.

**Leakage checklist:** [x] split made before any augmentation; [x] validation and test never augmented (the layers run only in training); [x] no picture in two parts; [x] test set not used for any decision (this notebook never loads it).
**Nuances / gotchas:**
- Small rotations only: scratches and rolled-in scale follow the rolling direction. EDA showed scratches in both directions, so flips are safe; a 90° turn is still left out, and the augmentation ablation (Phase 6) can test it.
- `fill_mode="reflect"` avoids black corners that never occur on a real strip.
- `RandomBrightness` needs `value_range=(0, 1)` for our CNN and `(-1, 1)` for MobileNetV2, otherwise it assumes 0 to 255.

**Alternatives considered:** Augmenting the image files on disk (would multiply the dataset and risk augmented copies leaking into validation); colour jitter (meaningless for grayscale).
**Viva angle:** Q: "Why are the augmentation layers inside the model?" A: Keras runs them only during training, so validation and test images are always the real, unchanged images.

---

## 2026-10-18 09:06 | Phase 3 summary  (run by: Arnav)
**Tags:** [RESULT]
1. **Recap:** one pipeline, two input formats (200 × 200 × 1 on 0 to 1; 224 × 224 × 3 on −1 to 1); five physically sensible augmentations inside the model; leakage checklist ticked.
2. **Acceptance checklist:** [x] split CSVs saved (notebook 02); [x] pipeline works, batch shapes logged; [x] augmentation figure saved; [x] leakage checklist ticked; [x] concepts added: stratified split, normalisation, data augmentation, batch, tf.data / prefetch / cache.
3. **What every teammate must be able to explain:** why MobileNetV2 needs `preprocess_input`; why each augmentation makes physical sense (and why no colour jitter or 90° turns); why augmentation never touches validation or test; what a batch and `prefetch` are.
4. **Key numbers:**

| Item | Value |
|---|---|
| Batch size | 32 |
| Training images per class | 210 |
| Rotation range | ±0.05 turn (about ±18°) |

---

## 2026-10-18 09:30 | Phase 4.0 | Moved to Arnav's main Kaggle account; models notebook split in two  (run by: Arnav)
**Tags:** [SETUP] [DECISION] [BUG]
**What we did:** Signed the Kaggle CLI into Arnav's main account (`arnavaggarwal07`), which he phone-verified so notebooks can use a GPU and the internet. Re-ran notebooks 01 to 03 there. Split the planned `04_models` notebook into `04_cnn_from_scratch` (M1, M2) and `05_transfer_learning` (M3 and the comparison of all three); evaluation becomes notebook 06 and the extras notebook 07.
**Why:** The first account was not phone-verified, so Kaggle silently ran everything on CPU and blocked internet, and MobileNetV2 needs internet once to download its ImageNet weights. Notebooks can only read outputs of the same account's private notebooks, so 01 to 03 had to run again under the new account. Splitting the models notebook lets the scratch CNNs run without internet and keeps each notebook short enough to present.
**How:** `kaggle auth login --force` (browser sign-in). `kaggle/run_on_kaggle.py` now reads the signed-in username instead of a fixed one. New kernel settings in `kaggle/kernels.json`.
**Result:** Same notebooks, same seed, so the split is unchanged; the GPU check is in notebook 01's environment cell.
**Nuances / gotchas:**
- **Bug:** pushing notebook 02 failed with `400 Bad Request` on `SaveKernel`. Cause: the repo copy now holds its executed outputs (six embedded figures), so the upload was too big. Fix: `run_on_kaggle.py` strips all outputs before uploading; Kaggle recomputes them anyway.
- The first run on the new account started before the phone verification went through and still had no GPU.
**Alternatives considered:** Training the scratch CNNs on CPU under the first account (about 20 to 30 minutes per model; MobileNetV2 would still need internet).
**Viva angle:** —

---

## 2026-10-18 09:31 | Phase 7.1 | Flow diagram  (run by: Arnav)
**Tags:** [RESULT]
**What we did:** Wrote `docs/flow_diagram.md` (Mermaid) and exported `docs/flow_diagram.png` (3136 × 972 px), in the same style as our MST diagram.
**Why:** The project instructions require a flow diagram with the pre-processing and visualisation steps clearly documented.
**How:** Four blocks left to right: 1 data (source, data check, split first), 2 visualisation and pre-processing (EDA figures, loading pipeline, augmentation), 3 modelling (own CNNs, MobileNetV2, training setup, choose finalists on validation), 4 evaluation and extras (test set once, Grad-CAM, robustness, ablation, TFLite). A dotted arrow links the EDA findings (uneven lighting, scratches in both directions) to the choice of flips. Exported with `npx -y @mermaid-js/mermaid-cli -i docs/flow_diagram.md -o docs/flow_diagram.png -b white -s 4` (mermaid-cli 12.0.0, already installed).
**Result:** `docs/flow_diagram.png`, readable on one slide. Every number in it comes from the Phase 0 to 3 results.
**Nuances / gotchas:** mermaid-cli writes `flow_diagram-1.png` for Markdown input; it is renamed to `flow_diagram.png`. The model results are not in the diagram, so it does not need redrawing after training.
**Alternatives considered:** Drawing it in draw.io (harder to keep in sync with the text version).
**Viva angle:** Q: "Walk us through your pipeline." A: Follow the diagram left to right: check the data, split first, explore only training images, load and augment, train three models, choose two on validation, test once, then explain and stress-test the result.

---

## 2026-10-18 09:40 | Phase 4.1 | Bug: M2 predicted one class for every validation image  (run by: Arnav)
**Tags:** [BUG] [NUANCE] [DECISION]
**What we did:** First run of `notebooks/04_cnn_from_scratch.ipynb` on the GPU. M1 worked; M2 (M1 + batch norm + augmentation) failed. Found the cause, fixed it and re-ran.
**Why:** M2 is our regularised CNN; a broken M2 would make the M1 vs M2 comparison meaningless.
**How:** Read M2's per-epoch history, then opened the saved `m2_improved.keras` file and read the batch-norm statistics with h5py.
**Result:**
- First run: M1 96.30 % validation accuracy (50 epochs, best epoch 47, still improving). M2 16.67 % validation accuracy at every epoch (exactly 1 in 6, so one class for every image) while its training accuracy rose from 49.5 % to 88.5 %; early stopping ended it after 9 epochs and restored epoch 1. Files: `outputs/results/first_run/`.
- Cause: batch normalisation uses each batch's own statistics in training but running averages at prediction time. Keras updates those averages with momentum 0.99 (1 % per step). With only 40 batches per epoch they lagged far behind: in the first BN layer the running variance was about 0.67 in every channel, while the real variance of those activations is about 0.01. So every validation image was normalised wrongly.
- Fix: `BatchNormalization(momentum=0.9)`, so the averages catch up within one epoch. Also more epochs (M1 was still improving at 50).
- After the fix: M2 95.19 % validation accuracy, M1 96.67 %.
**Nuances / gotchas:**
- Exactly 16.67 % on a balanced 6-class set is the signature of a model that outputs one class for everything.
- The training curve looked fine; only the validation side was broken. That is what pointed at something that behaves differently in training and prediction, which batch norm does.
- The same fix goes into the augmentation ablation in notebook 07, which retrains the M2 network.
**Alternatives considered:** Removing batch norm from M2 (would hide the problem instead of understanding it); a lower learning rate (does not fix lagging averages).
**Viva angle:** Q: "Why did your batch-norm CNN first score 16.7 %?" A: Batch norm predicts with running averages that Keras updates slowly (momentum 0.99); with 40 batches per epoch they had not caught up, so every validation image was normalised wrongly. Momentum 0.9 fixed it (95.2 %).

---

## 2026-10-18 09:50 | Phase 4.2 | MobileNetV2: fine-tuning broke batch norm until we froze it  (run by: Arnav)
**Tags:** [BUG] [NUANCE] [DECISION]
**What we did:** Three tries of the MobileNetV2 fine-tuning (stage B), all judged on the validation set.
**Why:** The first try scored below our small CNN, which is unusual for an ImageNet model on this dataset.
**How:** `notebooks/05_transfer_learning.ipynb`. Stage A (frozen base, head only, lr 1e-3) was identical in every run (best 81.11 % validation accuracy at epoch 10), so only stage B changed.
**Result:**
- Try 1: top 30 layers unfrozen, lr 1e-5: 88.89 % validation accuracy, still improving at the last epoch.
- Try 2: unfrozen from layer 100, lr 1e-4: worse; validation accuracy jumped between 50 % and 80 %.
- In both, the *training* accuracy fell at the first fine-tuning epoch (0.90 to 0.51 in try 1), although a learning rate of 1e-5 barely changes the weights. Cause: once unfrozen, the base's batch-norm layers normalised with each steel batch's statistics instead of ImageNet's, even though the base is called with `training=False` (plan nuance N9). In Keras 3 the reliable fix is to keep every `BatchNormalization` layer frozen.
- Try 3: unfrozen from layer 100, lr 1e-4, all batch-norm layers frozen: 97.41 % validation accuracy.
- Also fixed: the results table now reports the epoch the saved weights come from, and keeps stage A's weights if stage B ends up worse.
- Histories of the earlier tries: `outputs/results/first_run/` and `second_run/`.
**Nuances / gotchas:** Frozen ImageNet features alone reach only about 81 %: steel textures look nothing like ImageNet photos, so fine-tuning matters here.
**Alternatives considered:** Unfreezing the whole base (more risk of overfitting with 1,260 images); a smaller learning rate (try 1 showed it learns too slowly).
**Viva angle:** Q: "What can go wrong when fine-tuning a pretrained CNN?" A: Its batch-norm layers can switch to the new data's statistics and wreck the pretrained features. We saw training accuracy drop at the start of fine-tuning; freezing every batch-norm layer fixed it (88.9 % to 97.4 %).

---

## 2026-10-18 10:07 | Phase 4.3 | Final training run and the two finalists  (run by: Arnav)
**Tags:** [RESULT] [DECISION]
**What we did:** Ran the final versions of notebooks 04 and 05 on Kaggle (Tesla T4 GPU) and chose two finalists for the test, using the validation set only.
**Why:** The plan asks for at least three models compared on validation, and two finalists for a single test.
**How:** M1 and M2: Adam lr 1e-3, up to 100 epochs, EarlyStopping patience 12 (raised from 8 because M2's validation curve is jumpy and stopped before the learning-rate cuts could calm it), ReduceLROnPlateau (factor 0.5, patience 4), batch-norm momentum 0.9 in M2. M3: stage A lr 1e-3 up to 20 epochs; stage B from layer 100, lr 1e-4, batch norm frozen, up to 40 epochs. Table: `outputs/results/val_results.csv`; histories `history_<model>.csv`; curves `model_01` to `model_04` figures.
**Result:**

| Model | Val accuracy | Val loss | Params | File | Epochs (best) | Train time |
|---|---|---|---|---|---|---|
| M1 baseline CNN | 97.04 % | 0.1056 | 109,958 | 1.38 MB | 100 (100) | 182.1 s |
| M2 + BN + augmentation | 95.93 % | 0.1310 | 110,630 | 1.41 MB | 41 (29) | 158.6 s |
| M3 MobileNetV2 | 97.04 % | 0.0832 | 2,265,670 | 24.46 MB | 15 + 14 (21) | 121.2 s |

- M1 is not overfitting: its training accuracy (dropout on) was below its validation accuracy in 91 of 100 epochs; it was still improving at the 100-epoch limit.
- M3 stage A alone (frozen ImageNet features) peaked at 81.11 %.
- **Finalists (decided before the test is opened): M3**, lowest validation loss; **M1**, best of our own CNNs and about 18 times smaller. M2 is scored on the test too, for the comparison chart only.

**Nuances / gotchas:**
- M1 and M3 tie on validation accuracy; with 270 validation images one image is 0.37 percentage points, so validation loss is the tie-breaker.
- M3's re-run of the same settings gave 97.41 % in the check run and 97.04 % in the final run: stage A is reproducible on Kaggle's GPU but fine-tuning is not exactly. Only the final run counts.
- M2 doing worse than M1 is a real result on this dataset, not a bug: the bug run (16.67 %) is documented in Phase 4.1.

**Alternatives considered:** Choosing by validation accuracy alone (a tie); taking all three models as finalists.
**Viva angle:** Q: "Why didn't augmentation and batch norm help?" A: The dataset is small and clean, and M1 is tiny (110 k parameters) with dropout and global average pooling, so it was not overfitting in the first place. M2 scored 3 validation images fewer, within the noise of a 270-image validation set.

---

## 2026-10-18 10:07 | Phase 4 summary  (run by: Arnav)
**Tags:** [RESULT]
1. **Recap:** three CNNs trained on 1,260 images; two batch-norm bugs found and fixed (M2 momentum, MobileNetV2 fine-tuning); M1 and M3 tie at 97.04 % validation accuracy, M3 has the lowest loss; finalists M3 and M1.
2. **Acceptance checklist:** [x] at least 3 models trained, curves and table saved; [x] overfitting in M1 vs M2 discussed with the curves; [x] finalist decision logged; [x] concepts to add: convolution, ReLU, pooling, global average pooling, dense, softmax, cross-entropy, Adam, epoch/batch, overfitting, dropout, batch norm, early stopping, transfer learning, fine-tuning, MobileNetV2.
3. **What every teammate must be able to explain:** each layer of M1; why batch norm behaves differently in training and prediction (and both bugs it caused); stage A vs stage B of transfer learning; why validation loss broke the tie.
4. **Key numbers:** see the table in Phase 4.3.

---

## 2026-10-18 10:09 | Phase 5.1 | Test set used once  (run by: Arnav)
**Tags:** [RESULT] [NUANCE]
**What we did:** Ran `notebooks/06_evaluation.ipynb` on Kaggle. The test set (270 images, 45 per class) was opened once, at **2026-10-18 04:38:53 UTC (10:08:53 IST)**, after the finalists were logged (Phase 4.3). All three models were scored in that single pass; M3 and M1 are the finalists.
**Why:** The one honest measurement of how the models do on images they have never seen.
**How:** Accuracy, macro precision / recall / F1 and per-class reports with scikit-learn; crack metrics = crazing vs the other five classes; CPU time per image = mean of 100 single-image calls on `/CPU:0` after 10 warm-up calls. Files: `outputs/results/test_metrics.csv`, `classification_report_<model>.txt`, `test_predictions.csv`, `evaluation_summary.json`.
**Result:**

| Model | Test accuracy | Macro F1 | Crack precision | Crack recall | Errors | CPU ms / image | File |
|---|---|---|---|---|---|---|---|
| **M3 MobileNetV2 (final)** | 98.52 % | 0.9852 | 1.0000 | 1.0000 | 4 | 223.96 | 24.46 MB |
| **M1 baseline CNN** | 94.07 % | 0.9400 | 0.9783 | 1.0000 | 16 | 21.57 | 1.38 MB |
| M2 (not a finalist) | 96.67 % | 0.9665 | 0.9375 | 1.0000 | 9 | 29.31 | 1.41 MB |

- `eval_01_confusion_m3_mobilenetv2.png`: M3's mistakes are three scratches called inclusion and one rolled-in scale called pitted surface.
- `eval_02_confusion_m1_baseline.png`: M1's weak class is pitted surface (recall 77.78 %: 6 called inclusion, 3 scratches, 1 rolled-in scale); scratches recall 88.89 % (4 called inclusion).
- `eval_03_misclassified.png`: the 19 test images missed by at least one finalist (one of them by both), with each model's prediction and confidence. 9 of M1's 10 pitted-surface mistakes have a confidence below 0.7.
- `eval_04_test_comparison.png`: test accuracy and macro F1 of all three models.

**Nuances / gotchas:**
- Every model found all 45 crazing (crack) images: crack recall 100 % for all three.
- M1 fell from 97.04 % (validation) to 94.07 % (test), while M2 rose from 95.93 % to 96.67 %. Validation (270 images) could not separate models this close; we keep the finalists chosen beforehand and say so (the same lesson as in our MST project).
- Accuracy is high, as expected for NEU (plan nuance N6): the images are clean, balanced and taken under controlled conditions; we checked duplicates (one pair, kept in one split) and leakage; the robustness test (Phase 6) shows how accuracy holds up when images get worse.
- CPU times are on Kaggle's CPU, one image at a time; a laptop or an edge board will differ.
- The title of `eval_03` slightly overlaps its top row. We did not re-run notebook 06 to fix it, because that would mean opening the test set again.
**Alternatives considered:** Reporting the best test model regardless of the plan (would be choosing on the test set).
**Viva angle:** Q: "Your validation said M1 = M3, but the test says M3 is much better. Why?" A: Validation has only 270 images; one image is 0.37 points, and M1's mistakes were mostly low-confidence. The test confirms M3 generalises better; its validation loss (0.083 vs 0.106) already hinted at that.

---

## 2026-10-18 10:12 | Phase 5 summary  (run by: Arnav)
**Tags:** [RESULT]
1. **Recap:** test used once at 04:38:53 UTC; final model M3 at 98.52 % (4 errors), M1 at 94.07 %; crack recall 100 % for all; M1 struggles with pitted surface; M3 is 10 times slower on CPU and 18 times bigger.
2. **Acceptance checklist:** [x] test set evaluated once, time logged; [x] metrics files saved; [x] figures saved; [x] interpretation written; [ ] concepts to add: accuracy vs precision vs recall vs F1, macro average, confusion matrix, inference latency.
3. **What every teammate must be able to explain:** precision vs recall (and why crack recall matters most); how to read the confusion matrix; why M1 and M3 tied on validation but not on test; the speed / size / accuracy trade-off.
4. **Key numbers:** see the table in Phase 5.1.

---

## 2026-10-18 11:03 | Phase 6.1 | Extras: Grad-CAM, robustness, augmentation ablation, TensorFlow Lite  (run by: Arnav)
**Tags:** [RESULT] [NUANCE] [BUG] [DECISION]
**What we did:** Ran `notebooks/07_extras.ipynb` on Kaggle (four experiments, I1 to I4 in the notebook).
**Why:** The plan asks for at least three extras; these four answer "where does the model look", "does it survive a real plant", "what is augmentation worth" and "can it run on a small device".
**How:**
- I1 Grad-CAM written by hand with `tf.GradientTape` (last conv ReLU for M1, output of the pretrained base for M3), on 2 validation images per class and on the 19 missed test images.
- I2 robustness on the **validation** set (the test set stays single-use): Gaussian noise std 0.02 / 0.05 / 0.10, blur kernel 3 / 5 / 9 px, darker by 0.15 / 0.30 / 0.45, contrast × 0.8 / 0.6 / 0.4. M2 was added to the two finalists because it is the CNN trained with brightness and contrast augmentation.
- I3 ablation: the M2 network retrained with no augmentation, without rotation, and with the full set; same seed and callbacks as notebook 04; validation only.
- I4 TensorFlow Lite with dynamic-range quantisation, from an inference copy of each model (augmentation block removed).
**Result:** (`outputs/results/robustness.csv`, `ablation.csv`, `tflite.csv`, `extras_summary.json`; figures `extra_01` to `extra_05`)
- I1: M3's heatmaps sit on the defects (scratch line, inclusion streak, dark patches). M1's heatmaps often sit on the background: for inclusion the flat steel around the defect lights up, and on a missed scratch the background is red and the scratch is not.
- I2 (validation accuracy, clean then three severities):

| Damage | M1 | M2 | M3 |
|---|---|---|---|
| noise | 97.0 / 58.2 / 29.3 / 24.4 | 95.9 / 46.3 / 31.1 / 25.2 | 97.0 / 96.3 / 82.6 / 48.1 |
| blur | 97.0 / 38.5 / 32.6 / 25.6 | 95.9 / 72.2 / 47.4 / 38.5 | 97.0 / 95.9 / 92.6 / 84.8 |
| darker | 97.0 / 65.9 / 31.9 / 25.6 | 95.9 / 89.6 / 52.6 / 42.6 | 97.0 / 96.7 / 96.7 / 91.9 |
| low contrast | 97.0 / 80.0 / 38.1 / 24.8 | 95.9 / 97.8 / 94.1 / 52.6 | 97.0 / 95.9 / 87.0 / 65.2 |

- I3 (validation accuracy / loss): no augmentation 100.00 % / 0.0147; no rotation 99.26 % / 0.0558; full 95.56 % / 0.1677. An earlier run gave the same order: 100.00 %, 98.89 %, 94.07 % (`outputs/results/extras_run1/`).
- I4: M3 24.46 MB to 2.51 MB, 97.41 % validation accuracy, same answer as Keras on 99.63 % of images, 18.03 ms per image on Kaggle's CPU (Keras: 223.96 ms in notebook 06). M1 1.38 MB to 0.12 MB, 97.04 %, 100 % same answers, 10.14 ms.
**Nuances / gotchas:**
- **Bug:** the first TFLite conversion failed on MobileNetV2: its augmentation layers contain random operations (`StatelessRandomUniformV2`, `Tan` from the rotation) that TensorFlow Lite cannot convert. Fix: convert an inference copy without the augmentation block, which does nothing at prediction time anyway.
- On clean NEU images augmentation costs accuracy and rotation costs the most (plan nuance N7: scratches and rolled-in scale follow the rolling direction). But augmentation is what keeps M2 working under lighting changes, which a real line has. The best clean score (100 % on validation) came from the M2 network without augmentation; we did not test it, because the test set was already used.
- Noise breaks every model at std 0.10; no augmentation we used adds noise, which is a clear next step.
- M1's collapse plus its background heatmaps show it learned a shortcut from brightness and texture of the background, which also explains its test drop on pitted surface.
- Ablation training is not exactly repeatable on the GPU; two runs agree on the order, not on the exact numbers.
**Alternatives considered:** Robustness on the test set (would re-use it); full integer quantisation for TFLite (needs a calibration set, more work for a small gain).
**Viva angle:** Q: "How do you know the CNN looks at the defect?" A: Grad-CAM. MobileNetV2's heatmaps sit on the scratch, inclusion and patches; our small CNN often looked at the background, which is why it broke when we darkened the images.

---

## 2026-10-18 11:03 | Phase 6 summary  (run by: Arnav)
**Tags:** [RESULT]
1. **Recap:** four extras done (Grad-CAM, robustness, augmentation ablation, TensorFlow Lite); M3 looks at the defects and is the most robust; M1 learned a background shortcut; augmentation trades clean accuracy for lighting robustness; MobileNetV2 fits in 2.51 MB and runs in 18 ms per image on CPU.
2. **Acceptance checklist:** [x] at least 3 items done, each with a results file, a figure and a LOG entry (4 done); [ ] each explainable in 2 lines in `VIVA_QA.md` (to add).
3. **What every teammate must be able to explain:** how Grad-CAM works; why M1 broke under darker images; the augmentation trade-off (clean accuracy vs robustness); what quantisation does to size and speed.
4. **Key numbers:** see Phase 6.1.

---

## 2026-10-18 11:06 | Handover  (run by: Arnav)
**Tags:** [SETUP]
- **Where we stopped:** Phases 0 to 6 are done and ticked in `PLAN.md`. All seven notebooks ran on Kaggle (account `arnavaggarwal07`) and are saved here with their outputs. Final model: MobileNetV2, 98.52 % test accuracy, crack recall 100 %. `CONCEPTS.md` has 48 concepts and `VIVA_QA.md` 95 questions. Flow diagram: `docs/flow_diagram.png`.
- **What's next (Phase 7):** the report (`Report2_XX.pdf`), about 12 EST slides added to the team's Canva deck so `PPT_XX` covers both projects, the GitHub repo `Ai-for-engineers-EST-project` (git is set up locally, nothing pushed yet), and `Code2_XX.zip` for the submission form. Then a mock viva for every member.
- **Open questions:** the group number (XX in every file name); who owned which stage, for the contributions section.
- **Know before starting:**
  - Run a notebook with `python kaggle/run_on_kaggle.py <name>` after `kaggle auth login`. The account must be phone-verified for a GPU.
  - Do not re-run `06_evaluation`: the test set has been used once (2026-10-18 04:38:53 UTC).
  - Training on the GPU is not exactly repeatable; the numbers in the notebooks and `outputs/results/` are the ones to quote. Earlier tries are kept in `outputs/results/first_run/`, `second_run/` and `extras_run1/`.

---
