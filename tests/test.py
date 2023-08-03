import numpy as np
import tensorflow as tf

from src import generic_conv_utils, training, unet_with_tsconv, unet_without_tsconv


def create_mock_data(batch_size=1, frequency_bins=256, timesteps=10, channels=1):
    audio_data = np.random.rand(batch_size, frequency_bins, timesteps, channels)
    labels = np.random.randint(low=0, high=2, size=(batch_size, frequency_bins, timesteps, channels))
    return tf.data.Dataset.from_tensor_slices(audio_data, labels)

class TestTraining:
    def test_train(self, tmp_path):
        output_shape = (8, 8, 2)
        image_shape = (10, 10, 3)
        epochs = 5
        shuffle = True
        batch_size = 10

        filters = 2
        frequency_bins = 256
        time_steps = 1
        channels = 1
        is_ts_conv = True

        model = unet_with_tsconv.build_model(
            filters=filters,
            frequency_bins=frequency_bins,
            time_steps=time_steps,
            channels=channels,
            is_ts_conv=is_ts_conv)

        model.predict().shape = (None, *output_shape)

        trainer = training.Trainer(name="test", log_dir_path=str(tmp_path))

        train_dataset = create_mock_data()
        validation_dataset = create_mock_data()
        test_dataset = create_mock_data()

        trainer.fit(model,
                    train_dataset=train_dataset,
                    validation_dataset=validation_dataset,
                    test_dataset=test_dataset,
                    epochs=epochs,
                    batch_size=batch_size,
                    shuffle=shuffle)

        args, kwargs = model.fit.call_args
        train_dataset = args[0]
        validation_dataset = kwargs["validation_data"]

        assert tuple(train_dataset.element_spec[0].shape) == (None, *image_shape)
        assert tuple(train_dataset.element_spec[1].shape) == (None, *output_shape)
        assert train_dataset._batch_size.numpy() == batch_size

        assert validation_dataset._batch_size.numpy() == batch_size
        assert tuple(validation_dataset.element_spec[0].shape) == (None, *image_shape)
        assert tuple(validation_dataset.element_spec[1].shape) == (None, *output_shape)

        assert kwargs["epochs"] == epochs
        assert kwargs["shuffle"] == shuffle

        args, kwargs = model.evaluate.call_args
        test_dataset = args[0]
        assert tuple(test_dataset.element_spec[0].shape) == (None, *image_shape)
        assert tuple(test_dataset.element_spec[1].shape) == (None, *output_shape)



class TestCropConcatBlock():

    def test_uneven_concat(self):
        layer = generic_conv_utils.ConcatBlock()
        down_tensor = np.ones([1, 61, 61, 32])
        up_tensor = np.ones([1, 52, 52, 32])

        concat_tensor = layer(up_tensor, down_tensor)

        assert concat_tensor.shape == (1, 52, 52, 64)


class TestNonTSConvModel:

    def test_serialization(self, tmpdir):
        save_path = str(tmpdir / "unet_model")
        unet_model = unet_without_tsconv.build_model()
        unet_model.save(save_path)

        reconstructed_model = tf.keras.models.load_model(save_path)
        assert reconstructed_model is not None

    def test_build_model(self):

        filters = 2
        frequency_bins = 256
        time_steps = 1
        channels = 1
        kernel_size = (3,3)

        model = unet_without_tsconv.build_model(filters=filters,
                                                frequency_bins=frequency_bins,
                                                time_steps=time_steps,
                                                channels=channels)

        input_shape = model.get_layer("inputs").output.shape
        assert tuple(input_shape) == (None, frequency_bins, time_steps, channels)
        output_shape = model.get_layer("outputs").output.shape
        assert tuple(output_shape) == (None, frequency_bins, time_steps, channels)

        conv2D_layers = _collect_conv2d_layers(model)


        for conv2D_layer in conv2D_layers[:-1]:
            assert conv2D_layer.kernel_size == kernel_size

class TestTSConvModel:

    def test_serialization(self, tmpdir):
        save_path = str(tmpdir / "unet_model")
        unet_model = unet_with_tsconv.build_model()
        unet_model.save(save_path)

        reconstructed_model = tf.keras.models.load_model(save_path)
        assert reconstructed_model is not None

    def test_build_model(self):

        filters = 2
        frequency_bins = 256
        time_steps = 1
        channels = 1
        kernel_size = (3,3)

        model = unet_with_tsconv.build_model(filters=filters,
                                             frequency_bins=frequency_bins,
                                             time_steps=time_steps,
                                             channels=channels)

        input_shape = model.get_layer("inputs").output.shape
        assert tuple(input_shape) == (None, frequency_bins, time_steps, channels)
        output_shape = model.get_layer("outputs").output.shape
        assert tuple(output_shape) == (None, frequency_bins, time_steps, channels)

        conv2D_layers = _collect_conv2d_layers(model)


        for conv2D_layer in conv2D_layers[:-1]:
            assert conv2D_layer.kernel_size == kernel_size


def _collect_conv2d_layers(model):
    conv2d_layers = []
    for layer in model.layers:
        if type(layer) == tf.keras.layers.Conv2D:
            conv2d_layers.append(layer)
        elif type(layer) == generic_conv_utils.Conv2DBlock:
            conv2d_layers.append(layer.conv2d_1)
            conv2d_layers.append(layer.conv2d_2)

    return conv2d_layers