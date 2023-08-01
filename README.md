## Shortcomings
Due to time constraints, I could not figure out the details of TSConv, so the TSConv implementation is currently incomplete, matching the Conv2D implementation directly. 

I also did not complete the test cases due to aforementioned time constraints.


## How to use this model

This model is to be trained on multi-dimensional sound data. Importing the data directly by modifying main.py, then calling main.py through the command line, is the quickest way to use the code found in unet_with_tsconv.py and unet_without_tsconv.py. You may also import them directly, following the example found in main.py.

## Explanation of structure
unet_with_tsconv.py and unet_without_tsconv.py are nearly identical; the only real difference is the existence of the TSConv class (incomplete as it is) and it's use in the 2DConv blocks.  Otherwise, build_model is where the meat of the setup is done, calling abstracted layer classes to create the model object throughout the call, compile it at the end, and return it to the user.  This will allow the user to use the model in their own code or in test cases however they see fit.