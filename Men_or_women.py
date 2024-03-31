from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import KMeans
import pickle
import glob
import sys
import random
import collections.abc

collections.Iterable = collections.abc.Iterable
collections.Mapping = collections.abc.Mapping
collections.MutableSet = collections.abc.MutableSet
collections.MutableMapping = collections.abc.MutableMapping
from tqdm import tqdm
import glob
import random
import cv2
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dropout, Dense, Flatten
from keras.models import Sequential
from keras.applications import MobileNet
from keras.utils import to_categorical

Images = []

men = []
for file in tqdm(glob.glob(
    "men/*jpg"
)[:1000]):
    imd = Image.open(file)
    imd = imd.resize((224, 224))
    imd = np.ravel(imd)
    men.append(imd)
men = np.array(men)
yd = np.full((len(men), 1), 0)
# print(men, yd)
men_ = np.concatenate([men, yd], axis=1)

Images.extend(men_)

women = []
i = 0
for file in tqdm(glob.glob(
    "women/*jpg"
)[:1000]):
    imd = Image.open(file)
    imd = imd.resize((224, 224))
    imd = np.ravel(imd)
    women.append(imd)
women = np.array(women)
yd = np.full((len(women), 1), 1)
women_ = np.concatenate([women, yd], axis=1)

Images.extend(women_)
Images = np.array(Images)

X, y = Images[:, :-1], Images[:, -1]
X = X.reshape((-1, 224, 224, 3))
y = y.reshape((-1, 1))
y = to_categorical(y, dtype="uint8")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=61
)

def get_model():

    model = Sequential()

    base_model = MobileNet(
        weights="imagenet",  # Load weights pre-trained on ImageNet.
        input_shape=(224, 224, 3),
        classes=2,
        pooling="avg",
        include_top=False,
    )
    # freeezing the weights of the final layer
    for layer in base_model.layers:
        layer.trainable = False

    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation="softmax"))  # final op layer

    return model


model = get_model()
model.compile(
    optimizer=Adam(learning_rate=0.01),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
print(model.summary)

model.fit(
    x=np.array(X_train),
    y=np.array(y_train),
    batch_size=32,
    validation_data=(np.array(X_test), np.array(y_test)),
    epochs=20,
)
model.save("saved_model/model_t")
