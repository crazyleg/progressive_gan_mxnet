import mxnet as mx
from mxnet import nd, autograd, gluon, image
from models import GeneratorWrapper



# generator = GeneratorWrapper()
#
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
