from datetime import datetime
from pathlib import Path
from typing import Union, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard


class Trainer:
    """
    Fits the model to a datasets

    :param name: Name of the model, used to build the target log directory if no explicit path is given
    :param log_dir_path: Path to the directory where the model and tensorboard summaries should be stored
    """

    def __init__(self,
                 name: Optional[str]="audio_unet",
                 log_dir_path: Optional[Union[Path, str]]=None,
                 ):

        if log_dir_path is None:
            log_dir_path = build_log_dir_path(name)
        if isinstance(log_dir_path, Path):
            log_dir_path = str(log_dir_path)

        self.log_dir_path = log_dir_path

    def shift(self, inputs, shape):
        left_cutoff = shape[0] // 4
        right_cutoff = shape[0] // 2

        for i in range(shape[0]):
            if i <= left_cutoff:
                inputs[i] = np.reshape(np.append([0], inputs[i][:-1]), newshape=(len(inputs[i]), 1))
            if i > left_cutoff and i <= right_cutoff:
                inputs[i] = np.reshape(np.append(inputs[i][1:], [0]), newshape=(len(inputs[i]), 1))

        inputs = inputs.reshape(1, shape[0], shape[1], shape[2])

        return inputs

    def fit(self,
            model: Model,
            train_dataset: tf.data.Dataset,
            validation_dataset: Optional[tf.data.Dataset]=None,
            test_dataset: Optional[tf.data.Dataset]=None,
            epochs=10,
            batch_size=1,
            is_tsconv=False,
            shape=(256, 10, 1),
            **fit_kwargs):
        """
        Fits the model to the given data

        :param model: The model to be fit
        :param train_dataset: The dataset used for training
        :param validation_dataset: (Optional) The dataset used for validation
        :param test_dataset:  (Optional) The dataset used for test
        :param epochs: Number of epochs
        :param batch_size: Size of minibatches
        :param fit_kwargs: Further kwargs passd to `model.fit`
        """

        if is_tsconv:
            train_data = train_dataset.get_single_element("data")["data"].numpy()
            train_labels = train_dataset.get_single_element("labels")["labels"].numpy()
            validation_data = validation_dataset.get_single_element("data")["data"].numpy()
            validation_labels = validation_dataset.get_single_element("labels")["labels"].numpy()
            test_data = test_dataset.get_single_element("data")["data"].numpy()
            test_labels = test_dataset.get_single_element("labels")["labels"].numpy()

            train_dataset = tf.data.Dataset.from_tensor_slices({"inputs": self.shift(train_data, train_data.shape), "labels": train_labels.reshape(1, shape[0], shape[1], shape[2])})
            validation_dataset = tf.data.Dataset.from_tensor_slices({"inputs": self.shift(validation_data, validation_data.shape), "labels": validation_labels.reshape(1, shape[0], shape[1], shape[2])})
            test_dataset = tf.data.Dataset.from_tensor_slices({"inputs": self.shift(test_data, test_data.shape), "labels": test_labels.reshape(1, shape[0], shape[1], shape[2])})

        prediction_shape = model.predict(train_dataset)[1:].take(count=1).batch(batch_size=1).shape

        train_dataset = train_dataset.map(prediction_shape).batch(batch_size)

        if validation_dataset:
            validation_dataset = validation_dataset.map(prediction_shape).batch(batch_size)


        history = model.fit(train_dataset,
                            validation_data=validation_dataset,
                            epochs=epochs,
                            **fit_kwargs)

        self.evaluate(model, test_dataset, prediction_shape)

        return history

    def evaluate(self,
                 model:Model,
                 test_dataset: Optional[tf.data.Dataset]=None,
                 shape:Tuple[int, int, int]=None):

        if test_dataset:
            model.evaluate(test_dataset
                           .map(utils.crop_labels_to_shape(shape))
                           .batch(batch_size=1)
                           )


def build_log_dir_path(root: Optional[str]= "unet") -> str:
    return str(Path(root) / datetime.now().strftime("%Y-%m-%dT%H-%M_%S"))