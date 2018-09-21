import math
import mxnet as mx
from mxnet.gluon import nn
from mxnet import nd, autograd, gluon, image


def nf(stage):
    """
    Function te get number of feaatures for each progressive scale
    Credit goes to original NVIDIA tensorflow implementation

    """
    fmap_base = 8192  # Overall multiplier for the number of feature maps.
    fmap_decay = 1.0  # log2 feature map reduction when doubling the resolution.
    fmap_max = 512
    return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)


class Generator(gluon.nn.HybridBlock):
    """
    Generator neural template
    """

    def __init__(self, k, scale,
                 activation='relu',
                 normalization='batch',
                 final_activation=None):
        super(Generator, self).__init__()
        self.alpha = 1
        self.scale = scale
        self.normalization = normalization
        self.activation = activation
        self.final_activation = final_activation
        self.network = dict()
        name = f'input'
        self.network[name] = nn.HybridSequential()
        with self.network[name].name_scope():
            self.network[name].add(nn.Conv2DTranspose(nf(1), 4, 2, 1, use_bias=False))
            if 'batch' in normalization:
                self.network[name].add(nn.BatchNorm())
            self.network[name].add(nn.Activation(activation))

        for i in range(self.scale+1):
            name = f'growth_{i}'
            self.network[name] = nn.HybridSequential()
            with self.network[name].name_scope():
                self.network[name].add(nn.Conv2DTranspose(nf(i), 4, 2, 1, use_bias=False))
                if 'batch' in normalization:
                    self.network[name].add(nn.BatchNorm())
                self.network[name].add(nn.Activation(activation))

        name = f'to_rgb_main'
        self.network[name] = nn.HybridSequential()
        with self.network[name].name_scope():
            self.network[name].add(nn.Conv2DTranspose(3, 3, 1, 1, use_bias=True))
            if final_activation != None:
                self.network[name].add(nn.Activation(final_activation))

        name = f'to_rgb_growth'
        self.network[name] = nn.HybridSequential()
        with self.network[name].name_scope():
            self.network[name].add(nn.Conv2DTranspose(3, 3, 1, 1, use_bias=True))
            if final_activation != None:
                self.network[name].add(nn.Activation(final_activation))

    def set_alpha(self, alpha):
        self.alpha = alpha

    def hybrid_forward(self, F, x, *args, **kwargs):
        main_path = self.network['input'](x)

        for i in range(0,self.scale):
            main_path = self.network[f'growth_{i}'](main_path)

        side_chain = main_path
        side_chain = self.network[f'growth_{self.scale}'](side_chain)

        main_path =  self.network[f'to_rgb_main'](main_path)

        main_path = F.UpSampling(main_path, scale=2, sample_type='nearest')
        side_chain = self.network[f'to_rgb_growth'](side_chain)
        x = main_path*(1-self.alpha) + side_chain*(self.alpha)
        return x

    def change_scale(self, ctx):
        self.scale += 1
        self.network[f'to_rgb_main'] = self.network[f'to_rgb_growth']
        self.network[f'to_rgb_growth'] = None
        name = f'growth_{self.scale}'
        self.network[name] = nn.HybridSequential()
        with self.network[name].name_scope():
            self.network[name].add(nn.Conv2DTranspose(nf(self.scale), 4, 2, 1, use_bias=False))
            if 'batch' in self.normalization:
                self.network[name].add(nn.BatchNorm())
            self.network[name].add(nn.Activation(self.activation))

        name = f'to_rgb_growth'
        self.network[name] = nn.HybridSequential()
        with self.network[name].name_scope():
            self.network[name].add(nn.Conv2DTranspose(3, 3, 1, 1, use_bias=True))
            if self.final_activation != None:
                self.network[name].add(nn.Activation(self.final_activation))


        self.network[f'to_rgb_growth'].collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
        self.network[f'growth_{self.scale}'].collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)

class GeneratorWrapper():
    """
    Generator training wrapper
    """


    def __init__(self, ctx, init_scale=3, max_scale=16, features=8192):
        self.scale = init_scale
        self.ctx = ctx
        self.features = features
        self.generator = Generator(self.features, self.scale, final_activation='tanh')
        [self.generator.register_child(self.generator.network[x]) for x in self.generator.network.keys()]
        self.generator.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=self.ctx)
        self.trainer = gluon.Trainer(self.generator.collect_params(), 'Adam', {'learning_rate': .0001, 'beta1':0.5})
        # self.generator.hybridize()

    def forward(self, x):
        return self.generator(x)


    def save(self, name):
        self.generator.save_parameters(name)
        # self.generator.export(name+'exp')

    def load(self, name):
        self.generator.load_parameters(name, allow_missing=True)


    def train(self, true_image, latent_z, discriminator):
        data_fake = gluon.utils.split_and_load(latent_z, self.ctx)
        data_true = gluon.utils.split_and_load(true_image, self.ctx)

        with autograd.record():
            data_fake = [self.generator(X) for X in data_fake]

            disc_fake = [discriminator.discriminator(X) for X in data_fake]
            # adv_loss = [nd.mean((X - 1) ** 2) for X in disc_fake]
            disc_real = [discriminator.discriminator(X) for X in data_true]

            d_loss_fake = [
                (X - nd.repeat(nd.expand_dims(nd.mean(Y, axis=0), axis=0), repeats=Y.shape[0], axis=0) + 1) ** 2
                for
                X, Y in zip(disc_real, disc_fake)]
            d_loss_real = [
                (Y - nd.repeat(nd.expand_dims(nd.mean(X, axis=0), axis=0), repeats=X.shape[0], axis=0) - 1) ** 2
                for
                X, Y in zip(disc_real, disc_fake)]

            # d_loss_fake = [X ** 2 for X in disc_fake]
            # d_loss_real = [(X - 1) ** 2 for X in disc_real]

            adv_loss = [nd.mean(X + Y) * 0.5 for X, Y in zip(d_loss_real, d_loss_fake)]
        for l in adv_loss:
            l.backward()

        self.trainer.step(data_true[0].shape[0])
        curr_gloss = nd.mean(sum(adv_loss) / len(self.ctx)).asscalar()

        return curr_gloss

    def increase_scale(self):
        self.generator.change_scale(self.ctx)
        # self.generator.hybridize()

class Discriminator(gluon.nn.HybridBlock):
    """
    Dicriminator neural template
    """

    def __init__(self, k, scale,
                 activation='relu',
                 normalization='batch'):
        super(Discriminator, self).__init__()
        self.alpha = 0
        self.scale=scale
        self.activation = activation
        self.normalization = normalization
        i=0
        self.network = dict()


        name = f'fromrgb_growth'
        self.network[name] = nn.HybridSequential(prefix=name)
        with self.network[name].name_scope():
            self.network[name].add(nn.Conv2D(nf(scale), 3, 1, 1, use_bias=True))
            self.network[name].add(nn.LeakyReLU(0.2))

        name = f'fromrgb_main'
        self.network[name] = nn.HybridSequential(prefix=name)
        with self.network[name].name_scope():
            self.network[name].add(nn.Conv2D(nf(scale-1), 3, 1, 1, use_bias=True))
            self.network[name].add(nn.LeakyReLU(0.2))



        for i in range(scale, 1, -1):
            name = f'growth_{i}'
            self.network[name] = nn.HybridSequential(prefix=name)
            with self.network[name].name_scope():
                self.network[name].add(nn.Conv2D(nf(i-1), 3, 1, 1, use_bias=False))
                if 'batch' in normalization: self.network[name].add(nn.BatchNorm())
                self.network[name].add(nn.LeakyReLU(0.2))
                self.network[name].add(nn.MaxPool2D())

        self.network['downscale'] = nn.HybridSequential()
        self.network['downscale'].add(nn.MaxPool2D())

        self.network['result'] = nn.HybridSequential()
        self.network['result'].add(nn.Conv2D(1000, 1, 1, 0, use_bias=True))
        self.network['result'].add(nn.Conv2D(1, 1, 1, 0, use_bias=True))

    def change_scale(self, ctx):

        self.scale += 1
        self.network[f'fromrgb_main'] = self.network[f'fromrgb_growth']

        name = f'fromrgb_growth'
        self.network[name] = nn.HybridSequential(prefix=name)
        with self.network[name].name_scope():
            self.network[name].add(nn.Conv2D(nf(self.scale), 3, 1, 1, use_bias=True))
            self.network[name].add(nn.LeakyReLU(0.2))

        name = f'growth_{self.scale}'
        self.network[name] = nn.HybridSequential()
        with self.network[name].name_scope():
            self.network[name].add(nn.Conv2D(nf(self.scale-1), 3, 1, 1, use_bias=False))
            if 'batch' in self.normalization: self.network[name].add(nn.BatchNorm())
            self.network[name].add(nn.LeakyReLU(0.2))
            self.network[name].add(nn.MaxPool2D())

        self.network[f'fromrgb_growth'].collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
        self.network[f'growth_{self.scale}'].collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)

    def set_alpha(self, alpha):
        self.alpha = alpha



    def hybrid_forward(self, F, x, *args, **kwargs):
        main_path = self.network[f'fromrgb_growth'](x)
        main_path = self.network[f'growth_{self.scale}'](main_path)

        side_chain = self.network['downscale'](x)
        side_chain = self.network[f'fromrgb_main'](side_chain)

        x = side_chain * (1-self.alpha) + main_path * (self.alpha)

        for i in range(self.scale-1, 1, -1):
            x = self.network[f'growth_{i}'](x)

        x=self.network['result'](x)
        return x

class DiscriminatorWrapper():
    """
    Discriminator training wrapper
    """
    def __init__(self, ctx, init_scale=5, max_scale=16, features=8192):
        self.scale = init_scale
        self.features = features
        self.ctx = ctx
        self.discriminator = Discriminator(self.features, self.scale)
        [self.discriminator.register_child(self.discriminator.network[x]) for x in self.discriminator.network.keys()]
        self.discriminator.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=self.ctx)
        self.trainer = gluon.Trainer(self.discriminator.collect_params(), 'Adam', {'learning_rate': .0001, 'beta1':0.5})
        # self.discriminator.hybridize()

    def forward(self, x):
        return self.discriminator(x)

    def save(self, name):
        self.discriminator.save_parameters(name)
        # self.discriminator.export(name+'exp')

    def load(self, name):
        self.discriminator.load_parameters(name, allow_missing=True)

    def train(self, true_image, latent_z, generator):
        data_fake = gluon.utils.split_and_load(latent_z, self.ctx)
        data_true = gluon.utils.split_and_load(true_image, self.ctx)

        result = [generator.generator(X) for X in data_fake]
        with autograd.record():

            disc_fake = [self.discriminator(X) for X in result]
            disc_real = [self.discriminator(X) for X in data_true]


            d_loss_fake = [
                (X - nd.repeat(nd.expand_dims(nd.mean(Y, axis=0), axis=0), repeats=Y.shape[0], axis=0) - 1) ** 2
                for
                X, Y in zip(disc_real, disc_fake)]
            d_loss_real = [
                (Y - nd.repeat(nd.expand_dims(nd.mean(X, axis=0), axis=0), repeats=X.shape[0], axis=0) + 1) ** 2
                for
                X, Y in zip(disc_real, disc_fake)]

            # d_loss_fake = [X ** 2 for X in disc_fake]
            # d_loss_real = [(X - 1) ** 2 for X in disc_real]

            d_loss_total = [nd.mean(X + Y) * 0.5 for X, Y in zip(d_loss_real, d_loss_fake)]

        for l in d_loss_total:
            l.backward()

        self.trainer.step(latent_z[0].shape[0])
        curr_dloss = nd.mean(sum(d_loss_total) / len(self.ctx)).asscalar()
        return curr_dloss

    def increase_scale(self):
        self.discriminator.change_scale(self.ctx)
        # self.discriminator.hybridize()


