import numpy as np

import unet_with_tsconv
import unet_without_tsconv

if __name__ == '__main__':
    print("Hello!\n")
    tsconv = input("Do you want to use TSConv?")
    filters = input("How many filters do you want to use?")
    frequency_bins = input("How many frequencies are we covering?")
    time_steps = input("How many timesteps is your data?")
    channels = input("How many channels are there?")

    if tsconv == True:
        model = unet_with_tsconv.build_model(
            filters=filters,
            frequency_bins=frequency_bins,
            time_steps=time_steps,
            channels=channels,
            is_ts_conv=tsconv)
    else:
        model = unet_without_tsconv.build_model(
            filters=filters,
            frequency_bins=frequency_bins,
            time_steps=time_steps,
            channels=channels)

    dataset = np.zeros(shape=[frequency_bins, time_steps, channels])

    # I am unclear on what the input data looks like, so this call won't work, thanks to the lack of input data. Sorry!
    model.fit(dataset, epochs=10, steps_per_epoch=10)