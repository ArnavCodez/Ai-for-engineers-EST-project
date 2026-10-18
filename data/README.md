# Data: where it comes from

## Source
- **Dataset:** NEU surface defect database (Northeastern University, China): grayscale images of hot-rolled steel strip, 6 defect classes.
- **Copy we use:** the Kaggle mirror [`kaustubhdikshit/neu-surface-defect-database`](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database) (27.7 MB). It is the NEU-DET version: images in `train/` and `validation/` folders plus XML bounding boxes for detection.
- **What we take from it:** only each image and its class. The class is the start of the file name (`crazing_12.jpg` is crazing). We ignore the boxes and the mirror's own train/validation folders, and make our own split (`data/splits/`).
- **Original dataset page:** http://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/index.htm

## How to get it
- **On Kaggle (how we run the notebooks):** nothing to download. Each notebook has the dataset attached and reads it from `/kaggle/input`.
- **On a laptop:** download the mirror from the Kaggle page above and unzip it so the folder `data/raw/NEU-DET/` exists. The notebooks find it there.

## Files in this folder
- `raw/`: the dataset as downloaded. Never edited, not in git.
- `all_images.csv`: one row per image (path, class, size, mode, file hashes), made by `notebooks/01_data_check.ipynb`.
- `splits/`: our train/validation/test lists, made once in notebook 03 and reused by every later notebook.
