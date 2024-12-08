# We referred to Haiku's ResNet implementation:
# https://github.com/deepmind/dm-haiku/blob/main/haiku/_src/nets/resnet.py

import flax.linen as nn
# import jax
import jax.numpy as jnp


def flatten(x):
    return x.reshape((x.shape[0], -1))

class BlockV1(nn.Module):
    num_channels: int

    @nn.compact
    def __call__(self, x, is_training):
        i = x
        x = nn.Conv(self.num_channels, kernel_size=3)(x)
        x = nn.BatchNorm(momentum=0.9)(x, use_running_average=not is_training)
        x = nn.relu(x)
        x = nn.Conv(self.num_channels, kernel_size=3)(x)
        x = nn.BatchNorm(momentum=0.9)(x, use_running_average=not is_training)
        return nn.relu(x + i)


class BlockV2(nn.Module):
    num_channels: int

    @nn.compact
    def __call__(self, x, is_training, ):
        i = x
        x = nn.BatchNorm(momentum=0.9)(x, use_running_average=not is_training)
        x = nn.relu(x)
        x = nn.Conv(self.num_channels, kernel_size=3)(x)
        x = nn.BatchNorm(momentum=0.9)(x, use_running_average=not is_training)
        x = nn.relu(x)
        x = nn.Conv(self.num_channels, kernel_size=3)(x)
        return x + i


class AZNet(nn.Module):
    """AlphaZero NN architecture."""
    num_actions: int
    num_channels: int = 64
    num_blocks: int = 5
    resnet_v2: bool = True
    name: str = "aznet"
    resnet_cls = BlockV2 if resnet_v2 else BlockV1
    # super().__init__(name=name)
    # self.num_actions = num_actions
    # self.num_channels = num_channels
    # self.num_blocks = num_blocks
    # self.resnet_v2 = resnet_v2
    # self.resnet_cls = BlockV2 if resnet_v2 else BlockV1
    @nn.compact
    def __call__(self, x, is_training=False):
        x = x.astype(jnp.float32)
        x = nn.Conv(self.num_channels, kernel_size=3)(x)

        if not self.resnet_v2:
            x = nn.BatchNorm(momentum=0.9)(x, use_running_average=not is_training)
            x = nn.relu(x)

        for i in range(self.num_blocks):
            x = self.resnet_cls(self.num_channels, name=f"block_{i}")(
                x, is_training
            )

        if self.resnet_v2:
            x = nn.BatchNorm(momentum=0.9)(x, use_running_average=not is_training)
            x = nn.relu(x)

        # policy head
        logits = nn.Conv(features=2, kernel_size=1)(x)
        logits = nn.BatchNorm(momentum=0.9)(logits, use_running_average=not is_training)
        logits = nn.relu(logits)
        # logits = nn.Flatten()(logits)
        logits = flatten(logits)
        logits = nn.Dense(self.num_actions)(logits)

        # value head
        v = nn.Conv(features=1, kernel_size=1)(x)
        v = nn.BatchNorm(momentum=0.9)(v, use_running_average=not is_training)
        v = nn.relu(v)
        # v = nn.Flatten()(v)
        v = flatten(v)
        v = nn.Dense(self.num_channels)(v)
        v = nn.relu(v)
        v = nn.Dense(1)(v)
        v = jnp.tanh(v)
        v = v.reshape((-1,))

        return logits, v