import mxnet as mx
from tqdm import tqdm
from mxnet import nd, autograd, gluon, image
from models import GeneratorWrapper, DiscriminatorWrapper
from dataloaders import ArtLoader
from tensorboardX import SummaryWriter

GPU_COUNT = 2
KIMAGE = 120_000
BATCH_SIZE = 8
ctx = [mx.gpu(i) for i in range(GPU_COUNT)]
generator = GeneratorWrapper(ctx)
discriminator = DiscriminatorWrapper(ctx)
train_data = gluon.data.DataLoader(ArtLoader(folder='/data/art/wikiart-downloader/style/all/abstract-expressionism/',
                                             final_resolution=32), batch_size=BATCH_SIZE,
                                   num_workers=1, shuffle=True, last_batch='rollover')


logger = SummaryWriter('/data/tensorboard-runs/art/')

#TODO FAKEPOOL

for resolution in (32,64,128):
    for i, true_image in tqdm(enumerate(train_data)):

        latent_z = mx.nd.random_normal(0, 1, shape=(BATCH_SIZE, 100, 1, 1), ctx=mx.cpu())
        d_loss = discriminator.train(latent_z, true_image, generator)
        logger.add_scalar('D_loss', d_loss, i)


# latent_z = mx.nd.random_normal(0, 1, shape=(4, 100, 1, 1), ctx=mx.cpu())
#
# q = generator.forward(latent_z)
# generator.increase_scale()
# q = generator.forward(latent_z)
# generator.increase_scale()
# q = generator.forward(latent_z)
# generator.increase_scale()
# q = generator.forward(latent_z)
# print(q)
