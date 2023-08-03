import numpy as np
import tensorflow as tf

import unet_with_tsconv
import unet_without_tsconv
import training

def create_mock_data(batch_size=1, frequency_bins=256, timesteps=10, channels=1):
    audio_data = np.random.rand(batch_size, frequency_bins, timesteps, channels)
    labels = np.random.randint(low=0, high=2, size=(batch_size, frequency_bins, timesteps, channels))
    return tf.data.Dataset.from_tensor_slices({"data": audio_data, "labels": labels})

if __name__ == '__main__':
    print("Hello!")
    print("Do you want to use TSConv?")
    tsconv = 0
    while bool(tsconv) != True:
        tsconv = input("Please enter True or False. (Default is True) ")
    filters = int(input("How many filters do you want to use? (Default is 2) "))
    frequency_bins = int(input("How many frequencies are we covering? (Default is 256) "))
    time_steps = int(input("How many timesteps is your data? (default is 10) "))
    channels = int(input("How many channels are there? (Default is 1) "))

    if tsconv == "True":
        model = unet_with_tsconv.build_model(
            filters=filters,
            frequency_bins=frequency_bins,
            time_steps=time_steps,
            channels=channels,
            is_ts_conv=tsconv)
        unet_with_tsconv.make_model(model)
    else:
        model = unet_without_tsconv.build_model(
            filters=filters,
            frequency_bins=frequency_bins,
            time_steps=time_steps,
            channels=channels)
        unet_without_tsconv.make_model(model)

    train_dataset = create_mock_data(batch_size=1, frequency_bins=256, timesteps=10, channels=1)
    validation_dataset = create_mock_data(batch_size=1, frequency_bins=256, timesteps=10, channels=1)
    test_dataset = create_mock_data(batch_size=1, frequency_bins=256, timesteps=10, channels=1)

    trainer = training.Trainer(name="test_unet")

    trainer.fit(model=model,
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                validation_dataset=validation_dataset,
                epochs=10,
                batch_size=1,
                is_tsconv=tsconv,
                shape=(frequency_bins, time_steps, channels))
