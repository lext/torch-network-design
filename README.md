# torch-network-design
A simple python script to design convent architecture in torch. Very useful to design your network if you are a beginner or just lazy to calculate the inputs and outputs all the time.
## Basic pipeline and structures
### Convolutional blocks
First we have to create an instance of model designer and specify the number of channels and the image size. There are two ways for doing that: you can instantiate the model designer and then specify channels and image size using the corresponding methods, or just specify them in the object constructor. If the image size and the number of input channels have not been specified, then call of any designer's methods will throw an error. Here is an example of initializing the model for RGB image  256x256 pixels
```python
from designer import ModelDesigner

designer = ModelDesigner(3, (256,256))

```
By the default, batch normalization will be added after each convolution (or basically before ReLU). If you want to change this, set the corresponding parameter *bn_first=True* in teh constructor:
```python
designer = ModelDesigner(3, (256,256), bn_first=True)
```

Then we must add a sequential block. In this case I will call it *features*:

```python
designer.add_block('features')
```

Since we are dealing mostly with convnets, we will add a convolutional block:
```python
designer.add_conv_block(32, 3, 3, 1, 1, 1, 1, 1e-3, 0.01)
```

In torch this would be to add 3 layers: 32 feature maps maps 3x3 with stride 1 and padding 1 followed by batch normalization with *epsilon=1e-3* and Leaky ReLU with *a=1e-2*:

```lua
features:add(nn.SpatialConvolution(3, 32, 3, 3, 1, 1, 1, 1))
features:add(nn.SpatialBatchNormalization(32, 0.001))
features:add(nn.LeakyReLU(0.01,true))
```
Parameters for batch normalization and Leaky ReLu equal to 1e-3 and 1e-1 by default, so, if you want to use just these ones, then you can omit them when calling the class method:

```python
designer.add_conv_block(32, 3, 3, 1, 1, 1, 1)
```
If you want to have a net with normal ReLU, just specify *leakyrelu=0* and the normal ReLU will be adedd instead of LeakyReLU.

You can also add a droupout:

```python
designer.add_drop(0.5)
```

And max pooling (here 2x2):
```python
designer.add_pool(2,2)
```
You can also specify the stride and the padding exactly the same way as it is done for convolutions:
```python
designer.add_pool(3,3,2,2,0,0)

```


**ATTENTION:** The number of input dimensions is controlled automatically using private class attributes! The network designer controls inputs and outputs of every layer and prints an error, if for example, the outputs are not integers.

### Fully-connected blocks
Here there are 3 types of blocks that usually need to be used: *fc_block* -- linear layer followed by batch normalization and leaky ReLU, *fc* - just a linear layer and also *softmax* -- LogSoftMax layer in Torch. You need to specify the number of neurons on the given layer. This number can be -1. It will indicate that you take the same number of neurons as it was on the previous layer. Also, **when you instantiate the object, you need to set only the number of input dimensions as there is no any convolution involved!**

An example to build a simple MLP:

```python
from designer import ModelDesigner

designer = ModelDesigner(100)

# Adding a block
designer.add_block('classifier')
# 300 neurons
designer.add_fc_block(300)
# Dropout
designer.add_drop(0.5)
# 100 neurons
# you can use either mentioned way to specify the number of neurons for this block.
# If you use -1 as an argument, the layer will have 300 neurons, as in the previous layer.
designer.add_fc_block(100)
# Dropout
designer.add_drop(0.2)
# 3 classes classifier
designer.add_fc(3)
designer.add_softmax()
```

Output
```lua
local classifier = nn.Sequential()

classifier:add(nn.Linear(100, 100))
classifier:add(nn.BatchNormalization(100, 0.001))
classifier:add(nn.LeakyReLU(0.1,true))

classifier:add(nn.Dropout(0.4))

classifier:add(nn.Linear(100, 100))
classifier:add(nn.BatchNormalization(100, 0.001))
classifier:add(nn.LeakyReLU(0.1,true))

classifier:add(nn.Dropout(0.4))

classifier:add(nn.Linear(100, 3))

classifier:add(nn.LogSoftMax())

model = nn.Sequential():add(classifier)
```

## Full convnet example
See net_designer_example.py for the comments.
```python

from designer import ModelDesigner

if __name__ == '__main__':
    designer = ModelDesigner(3, (32,32))

    designer.add_block('features')

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

    designer.add_view()

    designer.add_drop(0.4)
    designer.add_fc_block(-1)
    designer.add_drop(0.4)
    designer.add_fc_block(-1)
    designer.add_fc(10)
    designer.add_softmax()

    designer.render(onlymodel=True)

```

This will produce the following output:

```lua

local features = nn.Sequential()

features:add(nn.SpatialConvolution(3, 32, 3, 3, 1, 1, 1, 1))
features:add(nn.SpatialBatchNormalization(32, 0.001))
features:add(nn.LeakyReLU(0.1,true)) -- 32x32 -> 32x32

features:add(nn.Dropout(0.3))

features:add(nn.SpatialConvolution(32, 32, 3, 3, 1, 1, 1, 1))
features:add(nn.SpatialBatchNormalization(32, 0.001))
features:add(nn.LeakyReLU(0.1,true)) -- 32x32 -> 32x32

features:add(nn.SpatialMaxPooling(2, 2)) -- 32x32 -> 16x16

features:add(nn.SpatialConvolution(32, 64, 3, 3, 1, 1, 1, 1))
features:add(nn.SpatialBatchNormalization(64, 0.001))
features:add(nn.LeakyReLU(0.1,true)) -- 16x16 -> 16x16

features:add(nn.Dropout(0.4))

features:add(nn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1))
features:add(nn.SpatialBatchNormalization(64, 0.001))
features:add(nn.LeakyReLU(0.1,true)) -- 16x16 -> 16x16

features:add(nn.SpatialMaxPooling(2, 2)) -- 16x16 -> 8x8

features:add(nn.SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1))
features:add(nn.SpatialBatchNormalization(128, 0.001))
features:add(nn.LeakyReLU(0.1,true)) -- 8x8 -> 8x8

features:add(nn.Dropout(0.4))

features:add(nn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1))
features:add(nn.SpatialBatchNormalization(128, 0.001))
features:add(nn.LeakyReLU(0.1,true)) -- 8x8 -> 8x8

features:add(nn.SpatialMaxPooling(2, 2)) -- 8x8 -> 4x4

local classifier = nn.Sequential()

classifier:add(nn.View(2048))

classifier:add(nn.Dropout(0.4))

classifier:add(nn.Linear(2048, 2048))
classifier:add(nn.BatchNormalization(2048, 0.001))
classifier:add(nn.LeakyReLU(0.1,true))

classifier:add(nn.Dropout(0.4))

classifier:add(nn.Linear(2048, 2048))
classifier:add(nn.BatchNormalization(2048, 0.001))
classifier:add(nn.LeakyReLU(0.1,true))

classifier:add(nn.Linear(2048, 10))

classifier:add(nn.LogSoftMax())

model = nn.Sequential():add(features):add(classifier)

```
