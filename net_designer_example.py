"""
A simple designer for Torch 7 models.

(c) Aleksei Tiulpin, Research unit of Medical Imaging, Physics and Technology, University of Oulu, 2017

"""

from designer import ModelDesigner

if __name__ == '__main__':

    # First we have to create an instance of model designer.
    # Another essential step is to specify the number of channels and the image size.
    # You can instantiate the model designer and then specify them using the corresponding methods,
    # or specify them right in the object constructor. If the image size and the number of input channels
    # have not been specified, then call of any designer's methods will throw an error.

    # One way to start
    # ===========================
    # designer = ModelDesigner()
    # designer.set_channels(3)
    # designer.set_imsize((32,32))

    # Another way (short one):
    # ===========================
    designer = ModelDesigner(3, (32,32))

    # Here we are adding the features block
    designer.add_block('features')
    # Here we need to specify the layers
    # No need to specify the input planes, because they are controlled automatically
    # using private class attributes.

    # If the outputs are not integers, then the script will print an error message,
    # but not an exception
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
    # Don't forget to make a View
    designer.add_view()
    designer.add_drop(0.4)
    # Here -1 indicates that we take the same number of neurons as it was in the previous layer
    designer.add_fc_block(-1)
    designer.add_drop(0.4)
    designer.add_fc_block(-1)
    designer.add_fc(10)
    designer.add_softmax()
    # Eventually we render the model. We can print the debug information as well,
    # but here we print only model
    designer.render(onlymodel=True)
