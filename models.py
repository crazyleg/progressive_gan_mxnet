import math
import mxnet as mx
from mxnet.gluon import nn
from mxnet import nd, autograd, gluon, image

class Generator(gluon.nn.HybridBlock):
    """
    Generator neural template
    """

    def nf(self, stage):
        fmap_base = 8192  # Overall multiplier for the number of feature maps.
        fmap_decay = 1.0  # log2 feature map reduction when doubling the resolution.
        fmap_max = 512
        return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

    def __init__(self, k, scale,
                 activation='relu',
                 normalization='batch',
                 final_activation=None):
        super(Generator, self).__init__()
        self.alpha = 1

        with self.name_scope():
            self.input = nn.HybridSequential(prefix='input')
            self.input.add(nn.Conv2DTranspose(self.nf(1), 4, 2, 1, use_bias=False))
            if 'batch' in normalization: self.input.add(nn.BatchNorm())
            self.input.add(nn.Activation(activation))

            self.growth = nn.HybridSequential(prefix='growth')

            for i in range(scale):
                self.growth.add(nn.Conv2DTranspose(self.nf(i), 4, 2, 1, use_bias=False, prefix=f'growth_conv_{i}'))
                if 'batch' in normalization: self.growth.add(nn.BatchNorm(prefix=f'growth_batch_{i}'))
                self.growth.add(nn.Activation(activation))


            self.torgb = nn.HybridSequential(prefix='output')
            self.torgb.add(nn.Conv2DTranspose(3, 3, 1, 1, use_bias=True, prefix=f'torgb_{i}'))
            if final_activation != None: self.output.add(nn.Activation(final_activation))

            self.torgb_sidechain = nn.HybridSequential(prefix='output')
            self.torgb_sidechain.add(nn.Conv2DTranspose(3, 3, 1, 1, use_bias=True, prefix=f'torgb_{i+1}'))
            if final_activation != None: self.torgb_sidechain.add(nn.Activation(final_activation))
    def set_alpha(self, alpha):
        self.alpha = alpha

    def hybrid_forward(self, F, x, *args, **kwargs):
        main_path = self.input(x)

        for i in range(len(self.growth)-3):
            main_path = self.growth[i](main_path)
        side_chain = main_path

        for i in range(len(self.growth)-3, len(self.growth)):
            side_chain = self.growth[i](side_chain)

        main_path = self.torgb(main_path)

        main_path = F.UpSampling(main_path, scale=2, sample_type='nearest')
        side_chain = self.torgb_sidechain(side_chain)
        x = main_path*self.alpha + side_chain*(1-self.alpha)
        return x

class GeneratorWrapper():
    def __init__(self, init_scale=4, max_scale=16, features=8192):
        self.scale = init_scale
        self.features = features
        self.generator = Generator(self.features, self.scale)
        self.generator.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=mx.cpu())

    def forward(self, x):
        return self.generator(x)

    def increase_scale(self):
        self.generator.save_params('tmp')
        self.scale += 1
        self.generator = Generator(self.features, self.scale)
        self.generator.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=mx.cpu())
        self.generator.load_params('tmp', ctx=mx.cpu(), allow_missing=True, ignore_extra=True)

class Discriminator(gluon.nn.HybridBlock):
    """
    Generator neural template
    """

    def nf(self, stage):
        fmap_base = 8192  # Overall multiplier for the number of feature maps.
        fmap_decay = 1.0  # log2 feature map reduction when doubling the resolution.
        fmap_max = 512
        return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

    def __init__(self, k, scale,
                 activation='relu',
                 normalization='batch'):
        super(Generator, self).__init__()
        self.alpha = 1

        with self.name_scope():

            self.fromrgb = nn.HybridSequential(prefix='output')
            self.fromrgb.add(nn.Conv2D(self.nf(scale), 3, 1, 1, use_bias=True, prefix=f'fromrgb_{i}'))
            self.output.add(nn.Activation(activation))

            self.fromrgb_sidechain = nn.HybridSequential(prefix='output')
            self.fromrgb_sidechain.add(nn.Conv2D(self.nf(scale)+1, 3, 1, 1, use_bias=True, prefix=f'fromrgb_{i+1}'))
            self.torgb_sidechain.add(nn.Activation(activation))

            self.growth = nn.HybridSequential(prefix='growth')

            for i in range(scale):
                self.growth.add(nn.Conv2DTranspose(self.nf(i), 4, 2, 1, use_bias=False, prefix=f'growth_conv_{i}'))
                if 'batch' in normalization: self.growth.add(nn.BatchNorm(prefix=f'growth_batch_{i}'))
                self.growth.add(nn.Activation(activation))
                self.growth.add(nn.AvgPool2D())


            self.downscale = nn.HybridSequential(prefix='input')
            self.downscale.add(nn.AvgPool2D())

            self.result = nn.HybridSequential(prefix='input')
            self.result.add(nn.Dense(100, bias=True))
            self.result.add(nn.Dense(1, bias=True))

    def set_alpha(self, alpha):
        self.alpha = alpha

    def hybrid_forward(self, F, x, *args, **kwargs):
        main_path = self.fromrgb(x)
        side_chain = self.downscale(x)
        side_chain = self.fromrgb_sidechain(side_chain)

        for i in range(0,3):
            main_path = self.growth[i](main_path)

        x = side_chain*self.alpha + main_path*(1-self.alpha)

        for i in range(3,len(self.growth)):
            x = self.growth[i](x)

        x=self.result(x)
        return x

class DiscriminatorWrapper():
    def __init__(self, init_scale=4, max_scale=16, features=8192):
        self.scale = init_scale
        self.features = features
        self.generator = Discriminator(self.features, self.scale)
        self.generator.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=mx.cpu())

    def forward(self, x):
        return self.generator(x)

    def increase_scale(self):
        self.generator.save_params('tmp')
        self.scale += 1
        self.generator = Generator(self.features, self.scale)
        self.generator.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=mx.cpu())
        self.generator.load_params('tmp', ctx=mx.cpu(), allow_missing=True, ignore_extra=True)
