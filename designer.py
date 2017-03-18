class ModelDesigner:
    """
    This class allows painfully design Torch 7 sequential models.
    The implementation is based on few stacks.

    (c) Aleksei Tiulpin
    """
    def __init__(self, dim=None, imsize=None):
        self.__model = [] # Here we add teh model itself
        self.__fmaps = [] # This list controls the number of feature maps or neurons for FC layers
        self.__inps = [] # This list controls the number of inputs on the current layer
        self.__blocks = [] # Here we control the blocks: e.g. features followed by classifier

        if imsize is not None:
            self.set_imsize(imsize)

        if dim is not None:
            self.set_channels(dim)


    def set_channels(self, n_channels):
        """
        Channels has to be an integer number
        """
        self.__fmaps.append(n_channels)

    def set_imsize(self, imsize):
        """
        imsize is a tuple of integers WxH
        """
        self.__inps.append(imsize)

    def render(self, onlymodel=False):
        """
        Renders the model according to Torch 7 specifications.

        """
        if len(self.__inps) == 0 or len(self.__fmaps) == 0:
            raise ValueError('Specify the number of input dimensions and the image size!')
        print
        if not onlymodel:
            print "==> Torch model:"
            print "========================"
        # Adding all the blocks to the model
        self.__model = self.__model +['model = nn.Sequential()'+''.join([':add({})'.format(block_name) for block_name in self.__blocks])]
        print '\n\n'.join(self.__model)

        if not onlymodel:
            print "========================"
            for inp in self.__inps:
                print inp


    def add_block(self, block):
        """
        Declares network Sequential block

        """
        self.__model.append('local {} = nn.Sequential()'.format(block))
        self.__blocks.append(block)

    def add_conv_block(self,nout, kw, kh, sw, sh, pw, ph, bnpar=1e-3, leakyrelu=0.1):
        """
        One convolutional block with 3x3 filters and padding 1, followed by Batch normalization and Leaky ReLU.

        Args:
            nout: Number of output planes
            kw: Kernel width
            kh: Kernel height
            sw: Convolution horizontal stride
            sh: Convolution vertical stride
            pw: Convolution horizontal padding
            ph: Convolution vertical padding
            bnpar: Batch normalization eps
            leakyrelu: Leaky ReLU coefficient

        """
        if len(self.__inps) == 0 or len(self.__fmaps) == 0:
            raise ValueError('Specify the number of input dimensions and the image size!')
        out = (float(self.__inps[-1][0]+2*pw-kw)/sw+1, float(self.__inps[-1][0]+2*ph-kh)/sh+1)

        if (not out[0].is_integer()) or (not out[1].is_integer()):
            print "============================"
            print 'ERROR!!!!! Float outputs from layer', len(self.__inps)
            print "============================"

        self.__model.append("{0}:add(nn.SpatialConvolution({1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}))\n{0}:add(nn.SpatialBatchNormalization({2}, {13}))\n{0}:add(nn.LeakyReLU({14},true)) -- {9}x{10} -> {11}x{12}".format(self.__blocks[-1], self.__fmaps[-1], nout, kw,kh,sw,sh,pw,ph, int(self.__inps[-1][0]), int(self.__inps[-1][1]), int(out[0]), int(out[1]), bnpar, leakyrelu))
        self.__fmaps.append(nout)
        self.__inps.append(out)

    def add_pool(self, pw=2,ph=2):
        """
        Max Pooling Layer

        Args:
            pw: Horizontal pooling
            ph: Vertical pooling
        """
        if len(self.__inps) == 0 or len(self.__fmaps) == 0:
            raise ValueError('Specify the number of input dimensions and the image size!')

        out = (self.__inps[-1][0]*1./pw, self.__inps[-1][1]*1./ph)

        if (not out[0].is_integer()) or (not out[1].is_integer()):
            print "============================"
            print 'ERROR!!!!! Float outputs from layer', len(self.__inps)
            print "============================"


        self.__model.append('{0}:add(nn.SpatialMaxPooling({1}, {2})) -- {3}x{4} -> {5}x{6}'.format(self.__blocks[-1], pw, ph, int(self.__inps[-1][0]), int(self.__inps[-1][1]), int(out[0]), int(out[1])))
        self.__inps.append(out)

    def add_drop(self,prob):
        """
        Dropout layer. Can be added to features and FC blocks.

        Args:
            prob: Dropout probabilty parameter
        """
        if len(self.__inps) == 0 or len(self.__fmaps) == 0:
            raise ValueError('Specify the number of input dimensions and the image size!')
        self.__model.append('{}:add(nn.Dropout({}))'.format(self.__blocks[-1], prob))
        self.__fmaps.append(self.__fmaps[-1])
        self.__inps.append(self.__inps[-1])

    def add_view(self):
        """
        Vew layer. Automaticaly calculates the outputs.

        """
        if len(self.__inps) == 0 or len(self.__fmaps) == 0:
            raise ValueError('Specify the number of input dimensions and the image size!')
        nout = int(self.__fmaps[-1]*self.__inps[-1][0]*self.__inps[-1][1])
        self.__model.append('{}:add(nn.View({}))'.format(self.__blocks[-1], nout))
        self.__fmaps.append(nout)
        self.__inps.append(nout)

    def add_fc_block(self, nout, bnpar=1e-3, leakyrelu=0.1):
        """
        Fully connected block: FC layer followed by batch normlization and Leaky ReLU

        """

        if len(self.__fmaps) == 0:
            raise ValueError('Specify the number of input dimensions and the image size!')
        if nout == -1:
            nout = self.__fmaps[-1]
        self.__model.append("{0}:add(nn.Linear({1}, {2}))\n{0}:add(nn.BatchNormalization({2}, {3}))\n{0}:add(nn.LeakyReLU({4},true))".format(self.__blocks[-1], self.__fmaps[-1], nout, bnpar, leakyrelu))

        self.__fmaps.append(nout)
        self.__inps.append(nout)

    def add_fc(self, nout):
        """
        Fully connected layer without activation and batch normalization

        """
        if len(self.__fmaps) == 0:
            raise ValueError('Specify the number of input dimensions and the image size!')

        if nout == -1:
            nout = self.__fmaps[-1]
        self.__model.append("{0}:add(nn.Linear({1}, {2}))".format(self.__blocks[-1], self.__fmaps[-1], nout))

        self.__fmaps.append(nout)
        self.__inps.append(nout)

    def add_softmax(self):
        if len(self.__fmaps) == 0:
            raise ValueError('Specify the number of input dimensions and the image size!')
        self.__model.append("{0}:add(nn.LogSoftMax())".format(self.__blocks[-1]))

        self.__fmaps.append(1)
        self.__inps.append(1)
