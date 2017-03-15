# torch-network-design
A simple python script to design convent architecture in torch. Very useful to design your network if you are a beginner or just lazy to calculate the inputs and outputs all the time.

Example of usage (see net_designer_example.py for the comments).
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
