import mxnet as mx
from tqdm import tqdm
from mxnet import nd, autograd, gluon, image
from models import GeneratorWrapper, DiscriminatorWrapper
from dataloaders import ArtLoader
from tensorboardX import SummaryWriter
import numpy as np
import sched, time

GPU_COUNT = 2
KIMAGE = 1_200_000
BATCH_SIZE = 64
EPS = 1e-8
ctx = [mx.gpu(i) for i in range(GPU_COUNT)]
generator = GeneratorWrapper(ctx, init_scale=2)
discriminator = DiscriminatorWrapper(ctx, init_scale=2)


config = {'resolutions': [16, 32, 64, 128,256, 512],
          'batch_sizes': [64, 64, 32, 16, 8,   4]}

#TODO LOGGER CONFIG
logger = SummaryWriter('/data/tensorboard-runs/art/')

#TODO FAKEPOOL
k = 0
for resolution, batch_size in zip(config['resolutions'], config['batch_sizes']):
    train_data = gluon.data.DataLoader(
        ArtLoader(folder='/data/art/wikiart-downloader/style/all/abstract-expressionism/',
                  final_resolution=resolution), batch_size=batch_size,
        num_workers=8, shuffle=True, last_batch='rollover')
    data_iterator = train_data.__iter__()

    i = 0
    timer = time.time()
    pbar = tqdm(total=KIMAGE)
    while i<KIMAGE:
        true_image = data_iterator.__next__()
        data_true = true_image
        latent_z = mx.nd.random_normal(0, 1, shape=(batch_size, 100, 1, 1))
        generator.generator.set_alpha(i / KIMAGE + EPS)
        discriminator.discriminator.set_alpha(i / KIMAGE + EPS)

        d_loss = discriminator.train(data_true, latent_z, generator)
        g_loss = generator.train(data_true, latent_z, discriminator)

        i += batch_size
        k += batch_size
        pbar.update(batch_size)

        if time.time()-timer>30:
            timer = time.time()
            logger.add_scalar('D_loss', np.mean(d_loss), k)
            logger.add_scalar('alpha', 1-i/KIMAGE, k)
            logger.add_scalar('G_loss', np.mean(g_loss), k)
            latent_z = mx.nd.random_normal(0, 1, shape=(batch_size, 100, 1, 1), ctx=ctx[0])
            images = generator.forward(latent_z)
            result_image = np.concatenate((images[0].asnumpy(),
                                           images[1].asnumpy(),
                                           images[2].asnumpy(),), axis=2)
            true_image = np.concatenate((data_true[0].asnumpy(),
                                         data_true[1].asnumpy(),
                                         data_true[2].asnumpy(),), axis=2)
            logger.add_image('true_data', np.transpose((np.clip(true_image * 255.0,0,255)).astype('int16'), axes=[1, 2, 0]), k)
            logger.add_image('fake_data', np.transpose((np.clip(result_image * 255.0,0,255)).astype('int16'), axes=[1, 2, 0]), k)
    generator.save(f'g_lsgan_{resolution}')
    discriminator.save(f'g_lsgan_{resolution}')
    generator.increase_scale()
    discriminator.increase_scale()

    # generator.generator.set_alpha(1)
    # discriminator.discriminator.set_alpha(1)



    pbar.close()



# q = generator.forward(data[0])
# print(q.shape)
# generator.increase_scale()
# q = generator.forward(data[0])
# print(q.shape)
# generator.increase_scale()
# q = generator.forward(data[0])
# print(q.shape)
# generator.increase_scale()
# q = generator.forward(data[0])
# print(q.shape)
# data = gluon.utils.split_and_load(mx.nd.random_normal(0, 1, shape=(BATCH_SIZE, 3, 16, 16)), ctx)
# q = discriminator.forward(data[0])
# print(q.shape)
# discriminator.increase_scale()
# data = gluon.utils.split_and_load(mx.nd.random_normal(0, 1, shape=(BATCH_SIZE, 3, 32, 32)), ctx)
# q = discriminator.forward(data[0])
# print(q.shape)
# discriminator.increase_scale()
# data = gluon.utils.split_and_load(mx.nd.random_normal(0, 1, shape=(BATCH_SIZE, 3, 64, 64)), ctx)
# q = discriminator.forward(data[0])
# print(q.shape)
# discriminator.increase_scale()
# data = gluon.utils.split_and_load(mx.nd.random_normal(0, 1, shape=(BATCH_SIZE, 3, 128, 128)), ctx)
# q = discriminator.forward(data[0])
# print(q.shape)
# discriminator.increase_scale()
# data = gluon.utils.split_and_load(mx.nd.random_normal(0, 1, shape=(BATCH_SIZE, 3, 256, 256)), ctx)
# q = discriminator.forward(data[0])
# print(q.shape)