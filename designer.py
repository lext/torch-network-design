class ModelDesigner:
    model = [] # Here we add teh model itself
    fmaps = [] # This list controls the number of feature maps or neurons for FC layers
    inps = [] # This list controls the number of inputs on the current layer
    blocks = [] # Here we control the blocks: e.g. features followed by classifier

    def set_channels(self, n_channels):
        """
        Channels has to be an integer number
        """
        self.fmaps.append(n_channels)

    def set_imsize(self, imsize):
        """
        imsize is a tuple of integers WxH
        """
        self.inps.append(imsize)

    def render(self, onlymodel=False):
        """
        Renders the model according to Torch 7 specifications.

        """
        print
        if not onlymodel:
            print "==> Torch model:"
            print "========================"
        # Adding all the blocks to the model
        self.model = self.model +['model = nn.Sequential()'+''.join([':add({})'.format(block_name) for block_name in self.blocks])]
        print '\n\n'.join(self.model)

        if not onlymodel:
            print "========================"
            for inp in self.inps:
                print inp

    def add_features(self):
        """
        Declares features block

        """
        self.model.append('features = nn.Sequential()')
        self.blocks.append('features')

    def add_classifier(self):
        """
        Declares classifier block

        """
        self.model.append('classifer = nn.Sequential()')
        self.blocks.append('classifier')

    def add_regressor(self):
        """
        Declares regressor block. Only the name is different from classifier

        """

        self.model.append('regressor = nn.Sequential()')
        self.blocks.append('regressor')

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
        out = (float(self.inps[-1][0]+2*pw-kw)/sw+1, float(self.inps[-1][0]+2*ph-kh)/sh+1)

        if (not out[0].is_integer()) or (not out[1].is_integer()):
            print "============================"
            print 'ERROR!!!!! Float outputs from layer', len(self.inps)
            print "============================"

        self.model.append("features:add(nn.SpatialConvolution({0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}))\nfeatures:add(nn.SpatialBatchNormalization({1}, {12}))\nfeatures:add(nn.LeakyReLU({13},true)) -- {8}x{9} -> {10}x{11}".format(self.fmaps[-1], nout, kw,kh,sw,sh,pw,ph, int(self.inps[-1][0]), int(self.inps[-1][1]), int(out[0]), int(out[1]), bnpar, leakyrelu))
        self.fmaps.append(nout)
        self.inps.append(out)

    def add_pool(self, pw=2,ph=2):
        """
        Max Pooling Layer

        Args:
            pw: Horizontal pooling
            ph: Vertical pooling
        """


        out = (self.inps[-1][0]*1./pw, self.inps[-1][1]*1./ph)

        if (not out[0].is_integer()) or (not out[1].is_integer()):
            print "============================"
            print 'ERROR!!!!! Float outputs from layer', len(self.inps)
            print "============================"


        self.model.append('features:add(nn.SpatialMaxPooling({0}, {1})) -- {2}x{3} -> {4}x{5}'.format(pw, ph, int(self.inps[-1][0]), int(self.inps[-1][1]), int(out[0]), int(out[1])))
        self.inps.append(out)

    def add_drop(self,prob):
        """
        Dropout layer. Can be added to features and FC blocks.

        Args:
            prob: Dropout probabilty parameter
        """
        self.model.append('{}:add(nn.Dropout({}))'.format(self.blocks[-1], prob))

    def add_view(self):
        """
        Vew layer. Automaticaly calculates the outputs.

        """
        nout = int(self.fmaps[-1]*self.inps[-1][0]*self.inps[-1][1])
        self.model.append('{}:add(nn.View({}))'.format(self.blocks[-1], nout))
        self.fmaps.append(nout)
        self.inps.append(nout)

    def add_fc_block(self, nout, bnpar=1e-3, leakyrelu=0.1):
        """
        Fully connected block: FC layer followed by batch normlization and Leaky ReLU
        """
        if nout == -1:
            nout = self.fmaps[-1]
        self.model.append("{0}:add(nn.Linear({1}, {2})\n{0}:add(nn.BatchNormalization({2}, {3}))\n{0}:add(nn.LeakyReLU({4},true))".format(self.blocks[-1], self.fmaps[-1], nout, bnpar, leakyrelu))

        self.fmaps.append(nout)
        self.inps.append(nout)

    def add_fc(self, nout):
        """
        Fully connected layer without activation and batch normalization

        """
        if nout == -1:
            nout = self.fmaps[-1]
        self.model.append("{0}:add(nn.Linear({1}, {2})".format(self.blocks[-1], self.fmaps[-1], nout))

        self.fmaps.append(nout)
        self.inps.append(nout)