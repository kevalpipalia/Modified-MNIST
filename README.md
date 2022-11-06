# Modified-MNIST
Classifying Handwritten Digits (Modified MNIST)

This repository is in addition to the Kaggle Competition for classification of MNIST digits in IFT6390 Fall 2022. (Community Prediction Competition)

Team Name: Keval Pipalia

Rank - 4/113

Competition Page: https://www.kaggle.com/competitions/classification-of-mnist-digits/overview

Leaderboard: https://www.kaggle.com/competitions/classification-of-mnist-digits/leaderboard


The goal is to design a machine learning algorithm that can automatically classify images. All the images (including held out test samples) are of the dimension 56 * 28. The class for a image is denoted by the sum of digits in the image. We have generated this dataset from the popular MNIST dataset. The training and held-out test set for this data consists of 50,000 samples and 10,000 samples respectively.The goal is to automatically classify the digits, into one of 19 classes from 0 to 18.

The main goal of this competition is for you to gain knowledge on various machine learning algorithms.

The various features in the data corresponds to the pixel values of the images. The training and held-out test set
for this data consists of 50,000 samples and 10,000 samples respectively.

To regenerate the results:
1. Download the data from Kaggle page and put inside data folder
2. Install Dependencies using 
```
pip install -r requirements.txt
```

3. To retrain the model / follow the complete training pipeline, use train.py inside each model folder,
e.g. to retrain CNN, use:
```
python CNN/train.py
```
    OR
```
python Ensembled\ CNN/train.py
```

4. To generate results, use:
```
python CNN/predict.py
```