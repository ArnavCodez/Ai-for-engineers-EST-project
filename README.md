# Steel surface defect recognition with a CNN

UCS321 AI for Engineers, EST mini project, statement #3 (Tata Steel scenario). TIET Patiala, B.E. ECE, Semester III.

We train a convolutional neural network (CNN) that looks at a grayscale image of hot-rolled steel strip and says which of 6 surface defects it shows. One of them, **crazing**, is a network of fine surface cracks, so we also report how well the model catches cracks on their own. The data is the public NEU surface defect database (Northeastern University): 6 classes of 200 × 200 px images. Details in `data/README.md` and `docs/problem.md`.

## Status

| Phase | What | State |
|---|---|---|
| 0 | Setup and data check (`notebooks/01_data_check.ipynb`) | done |
| 1 | Problem understanding (`docs/problem.md`) | done |
| 2 | Split and exploratory data analysis (`notebooks/02_split_and_eda.ipynb`) | done |
| 3 | Loading pipeline and augmentation (`notebooks/03_pipeline_and_augmentation.ipynb`) | done |
| 4 | Models: our own CNNs (`notebooks/04_cnn_from_scratch.ipynb`) and MobileNetV2 transfer learning (`notebooks/05_transfer_learning.ipynb`) | done |
| 5 | Evaluation on the test set, used once (`notebooks/06_evaluation.ipynb`) | done |
| 6 | Extras: Grad-CAM, robustness, augmentation ablation, TensorFlow Lite (`notebooks/07_extras.ipynb`) | done |
| 7 | Flow diagram, report, slides, viva prep | in progress (diagram and report done) |

## Where the notebooks run

The notebooks run on **Kaggle**, with the NEU dataset attached and a free GPU for training. Each one also runs on a laptop if the dataset is in `data/raw/NEU-DET/` and the packages in `requirements.txt` are installed (`pip install -r requirements.txt`). Training without a GPU is slow, mainly for MobileNetV2.

To run a notebook on Kaggle from this folder and pull its results back:
```
python kaggle/run_on_kaggle.py 01_data_check
```
This pushes `notebooks/01_data_check.ipynb` to Kaggle (settings in `kaggle/kernels.json`), waits for the run, and copies the files it wrote into `data/` and `outputs/`. It needs the Kaggle CLI (`pip install kaggle`), signed in once with `kaggle auth login`.

## Folders

```
notebooks/   01 data check, 02 split + EDA, 03 pipeline, 04 own CNNs, 05 MobileNetV2, then evaluation and extras
data/        README (source), all_images.csv, splits/ (raw images are not in git)
outputs/     figures/ (PNG), results/ (CSV, JSON), models/ (not in git)
docs/        problem.md, flow_diagram.md (+ .png), report/ (report.md and report.pdf)
kaggle/      run_on_kaggle.py and kernels.json (Kaggle settings per notebook)
app.py       Streamlit demo (run_demo.bat starts it)
```

## The demo

```
streamlit run app.py
```
Or double-click `run_demo.bat`: it makes the virtual environment on the first run, then opens the demo at http://localhost:8502. It needs the trained models in `outputs/models/` and the images in `data/raw/NEU-DET/` (both are in the submission zip, not in git).

Two tabs:
- **Classify an image:** pick a test image or upload one, change its brightness, contrast, noise, blur or rotation with sliders, and see the class and confidence from M1 and M3 side by side.
- **Model performance:** test metrics for all three models, size and speed (Keras and TFLite), confusion matrix and per-class scores, training curves, robustness, the augmentation ablation and the misclassified test images, all read from `outputs/results/`.

## How to learn this project

Read in this order: this README, `PLAN.md`, `LOG.md` top to bottom, `CONCEPTS.md`, then `VIVA_QA.md`. The team rules (logging format, test set used once, seed 42) are in `CONTRIBUTING.md`.
