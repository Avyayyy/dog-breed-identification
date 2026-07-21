# Dog Breed Identification 

Classifying dog breeds from images using transfer learning, built with TensorFlow in Google Colab.

## Problem Statement

Given an image of a dog, predict which of 120 breeds it belongs to. This is a **multi-class image classification** problem 

**Evaluation goal:** 

Submissions are evaluated on Multi-Class Log Loss between the predicted probability and the observed target. - www.kaggle.com/competitions/dog-breed-identification/overview/evaluation

## Dataset

- Source: Dog Breed Identification (Kaggle) - https://www.kaggle.com/competitions/dog-breed-identification/overview
- ~10,000+ labeled training images across **120 dog breeds**, plus a separate unlabeled test set for competition submission
- Relatively few images per breed (roughly 100 on average) 
- `labels.csv` maps each training image ID to its breed

## Approach

- Loaded `labels.csv`, converted image IDs into full file pathnames, and one-hot encoded the 120 breed labels
- Split the training data into train/validation sets (since Kaggle only provides train and test, with test labels withheld for competition scoring)
- Preprocessed images into Tensors (resize, normalise) and batched them using `tf.data`
- Built a transfer learning model in Keras using a pretrained **MobileNetV2** (via Kaggle Models) with a custom Dense output layer for the 120 breed classes
- Added callbacks: TensorBoard (tracking training progress) and early stopping (preventing overfitting)
- **Experimented on a ~1,000 image subset first** to validate the pipeline before committing to a full training run — this surfaced clear overfitting (100% training accuracy, ~68% validation accuracy), an expected and useful signal given how few images exist per breed
- Trained the final model on the full training dataset
- Evaluated predictions, visualised top predictions per image, and generated predictions on the held-out Kaggle test set for submission
- Saved/reloaded the trained model, and ran predictions on my unseen dog images

## Results

**Experiment (1,000-image subset):** ~100% training accuracy, ~68% validation accuracy — early stopping triggered, revealing overfitting on a small subset. This result informed the decision to train on the full dataset rather than trust the subset model.

**Final model (trained on full training set):** ~99.8% training accuracy after 17 epochs (early stopping). No held-out validation score for this run, since all available training data was used to maximise performance for the Kaggle test set submission.

**Kaggle competition submission:** scored **0.88412** on the official evaluation metric, multi-class log loss (lower is better). 


## Tools & Libraries

- Python
- TensorFlow / Keras
- TensorFlow Hub (Kaggle Models) — pretrained MobileNetV2 for transfer learning
- Google Colab (GPU runtime)
- pandas, NumPy
- Matplotlib (visualisation)

  
## Project Structure

```
.
├── README.md
├── .gitignore
├── dog-breed-identification.ipynb   # main notebook (Colab)
└── data/                             # (not tracked in git — download separately)
    ├── train/
    ├── test/
    └── labels.csv
```
