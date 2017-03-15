"""
A simple designer for Torch 7 models.

(c) Aleksei Tiulpin, MIPT, University of Oulu, 2017

"""

from designer import ModelDesigner

if __name__ == '__main__':
    # First we have to create an instance of model designer

    # The essential step is to specify the number of channels and the image size
    # You can instantiate the model designer and then specify them, or to specify them
    # right in the object constructor. If the image size and the number of input channels
    # are have not been specified, then any of the methods will throw an exception

    # One way:
    # ===========================
    # designer = ModelDesigner()
    # designer.set_channels(3)
    # designer.set_imsize((32,32))

    # Another way:
    # ===========================
    designer = ModelDesigner(3, (32,32))


    # Here we are adding the features block
    designer.add_block('features')
    # Here we specify the layers
    # No need to specify input planes because they are controlled automatically
    # If the outputs are not integers, then the script will print an error message
    designer.add_conv_block(32, 3, 3, 1, 1, 1, 1)
    designer.add_drop(0.3)
    designer.add_conv_block(32, 3, 3, 1, 1, 1, 1)
    designer.add_pool(2,2)

    designer.add_conv_block(64, 3, 3, 1, 1, 1, 1)
    designer.add_drop(0.4)
    designer.add_conv_block(64, 3, 3, 1, 1, 1, 1)
    designer.add_pool(2,2)

    designer.add_conv_block(128, 3, 3, 1, 1, 1, 1)
    designer.add_drop(0.4)
    designer.add_conv_block(128, 3, 3, 1, 1, 1, 1)
    designer.add_pool(2,2)

    # Here  we are adding the classifier block
    designer.add_block('classifier')
    designer.add_view() # Don't forget to make a View
    designer.add_drop(0.4)
    designer.add_fc_block(-1) # Here -1 indicates that we take the same number of neurons as it was in the previous layer
    designer.add_drop(0.4)
    designer.add_fc_block(-1)
    designer.add_fc(10)
    designer.add_softmax()
    # Eventually we render the model. We can print the debug information as well
    designer.render(onlymodel=True)
