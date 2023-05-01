import argparse
import json
import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from smexperiments.tracker import Tracker
from tensorflow.python.keras.applications.xception import Xception
from tensorflow.keras import layers, models, optimizers

def dnn_model(x_train, y_train, x_test, y_test):
    model = tf.keras.models.Sequential(
        [
            layers.Flatten(),
            layers.Dense(1024, activation=tf.nn.relu),
            layers.Dropout(0.4),
            layers.Dense(100, activation=tf.nn.softmax),
        ]
    )

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=32)
    model.evaluate(x_test, y_test)

    return model


def xception_model(x_train, y_train, x_test, y_test):
    base_model = Xception(input_tensor=layers.Input(shape=(32,32,3)), weights="imagenet", include_top=False)
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation="relu")(x)
    predictions = layers.Dense(100, activation="softmax")(x)
    
    model = models.Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    
    model.fit(x_train, y_train, epochs=32)
    
    model.evaluate(x_test, y_test)
    
    return model


def _load_training_data(base_dir):
    x_train = np.load(os.path.join(base_dir, "train_data.npy"))
    y_train = np.load(os.path.join(base_dir, "train_labels.npy"))
    return x_train, y_train


def _load_testing_data(base_dir):
    x_test = np.load(os.path.join(base_dir, "eval_data.npy"))
    y_test = np.load(os.path.join(base_dir, "eval_labels.npy"))
    return x_test, y_test


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--sm-model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAINING"))
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ.get("SM_HOSTS")))
    parser.add_argument("--current-host", type=str, default=os.environ.get("SM_CURRENT_HOST"))

    return parser.parse_known_args()


if __name__ == "__main__":
    args, unknown = _parse_args()

    train_data, train_labels = _load_training_data(args.train)
    eval_data, eval_labels = _load_testing_data(args.train)

    model = xception_model(train_data, train_labels, eval_data, eval_labels)
    if args.current_host == args.hosts[0]:
        model.save(os.path.join(args.sm_model_dir, "000000001"))
