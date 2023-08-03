## How to use this model

This model is to be trained on multi-dimensional sound data. Importing the data directly by modifying main.py, then calling main.py through the command line, is the quickest way to use the code found in unet_with_tsconv.py and unet_without_tsconv.py. You may also import them directly, following the example found in main.py.

## Explanation of structure
This project is divided into several files, all of which are in src; main.py is fairly simple, and just has a user interface to allow a user to call for the creation, training, and use of a model. The meat of the project is in generic_conv_utils.py, unet_without_tsconv.py, unet_with_tsconv.py, and training.py.

unet_with_tsconv.py and unet_without_tsconv.py are nearly identical; the only real difference is the existence of the TSConv class and it's use in the GenericConvBlock class.  Otherwise, build_model is where the meat of the setup is done, calling abstracted layer classes to create the model object throughout the call, compile it at the end, and return it to the user.  This will allow the user to use the model in their own code or in test cases however they see fit.

generic_conv_utils.py is just a holder for model objects that both unet_without_tsconv.py and unet_with_tsconv.py are going to use.

training.py is where the work of training an existing model is done. It's fairly barebones, lacking plugs into callbacks, modifiable hyperparameters, and other such niceties, but it gets the job done of training a model using the provided data.

## Issues

Shifting does not work as part of the TSConv block itself, so the shift operation has been moved to the Trainer.fit calls. This is not ideal, but it's the best I have for now pending further research.

There's an issue with the shapes being actually input into the model itself I can't resolve for now; getting the model this far took significant time, and so I'm calling it here as enough.

