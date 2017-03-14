"""
A simple designer for Torch 7 models.

(c) Aleksei Tiulpin, MIPT, University of Oulu, 2017

"""

from designer import ModelDesigner

if __name__ == '__main__':
    # Corresponds to feature maps or neurons on the prevous layers for FC blocks
    # image size

    designer = ModelDesigner()
    designer.set_channels(3)
    designer.set_imsize((32,32))
    # Here we are adding the features block
    designer.add_features()
    # Here we specify the layers
    # No need to specify input planes because they are controlled automatically
    # If the outputs are not integers, then the script will print an error message
    designer.add_conv_block(64, 3, 3, 1, 1, 1, 1)
    designer.add_drop(0.3)
    designer.add_conv_block(64, 3, 3, 1, 1, 1, 1)
    designer.add_pool(2,2)
    designer.add_conv_block(128, 3, 3, 1, 1, 1, 1)
    designer.add_drop(0.4)
    designer.add_conv_block(128, 3, 3, 1, 1, 1, 1)
    designer.add_pool(2,2)
    # Here  we are adding the classifier block
    designer.add_regressor()
    designer.add_view() # Don't forget to make a View
    designer.add_drop(0.4)
    designer.add_fc_block(-1) # Here -1 indicates that we take the same number of neurons as it was in the previous layer
    designer.add_fc_block(1024)
    designer.add_fc(10)
    # Eventually we render the model. We can print the debug information as well
    designer.render(onlymodel=True)
