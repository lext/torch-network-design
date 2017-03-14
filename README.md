# torch-network-design
A simple python script to design convent architecture in torch. Very useful to design your network if you are a beginner or just lazy to calculate the inputs and outputs all the time.

Example of usage
```python

from designer import ModelDesigner

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

```

This will produce the following output:

```lua
features = nn.Sequential()

features:add(nn.SpatialConvolution(3, 64, 3, 3, 1, 1, 1, 1))
features:add(nn.SpatialBatchNormalization(64, 0.001))
features:add(nn.LeakyReLU(0.1,true)) -- 32x32 -> 32x32

features:add(nn.Dropout(0.3))

features:add(nn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1))
features:add(nn.SpatialBatchNormalization(64, 0.001))
features:add(nn.LeakyReLU(0.1,true)) -- 32x32 -> 32x32

features:add(nn.SpatialMaxPooling(2, 2)) -- 32x32 -> 16x16

features:add(nn.SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1))
features:add(nn.SpatialBatchNormalization(128, 0.001))
features:add(nn.LeakyReLU(0.1,true)) -- 16x16 -> 16x16

features:add(nn.Dropout(0.4))

features:add(nn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1))
features:add(nn.SpatialBatchNormalization(128, 0.001))
features:add(nn.LeakyReLU(0.1,true)) -- 16x16 -> 16x16

features:add(nn.SpatialMaxPooling(2, 2)) -- 16x16 -> 8x8

classifer = nn.Sequential()

classifier:add(nn.View(8192))

classifier:add(nn.Dropout(0.4))

classifier:add(nn.Linear(8192, 8192)
classifier:add(nn.BatchNormalization(8192, 0.001))
classifier:add(nn.LeakyReLU(0.1,true))

classifier:add(nn.Linear(8192, 1024)
classifier:add(nn.BatchNormalization(1024, 0.001))
classifier:add(nn.LeakyReLU(0.1,true))

classifier:add(nn.Linear(1024, 10)

model = nn.Sequential():add(features):add(classifier)
```
