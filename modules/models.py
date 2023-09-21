import jax
import jax.numpy as jnp
from jax import random, vmap, jit, grad, value_and_grad, jacfwd, jacrev
import optax
import equinox as eqx
from functools import partial


class DeepONet(eqx.Module):
    branch_net: eqx.Module
    trunk_net: eqx.Module

    def __init__(self, branch_net, trunk_net):
        self.branch_net = branch_net
        self.trunk_net = trunk_net

    def __call__(self, sensors, inputs):
        b = self.branch_net(sensors)
        t = self.trunk_net(inputs)
        return jnp.sum(b * t, axis=-1, keepdims=True)


class FourierFeatures(eqx.Module):
    weights: jax.Array = eqx.field(static=True)
    frequency: float = eqx.field(static=True)
    scale: float = eqx.field(static=True)

    def __init__(self, weights=None, frequency=2*jnp.pi, scale=1., input_dim=None, num_features=None,
                 key=random.PRNGKey(0), dtype=jnp.float32):
        self.scale = scale
        if weights is None:
            key, subkey = random.split(key)
            weights = random.normal(subkey, (input_dim, num_features), dtype=dtype)
        self.weights = weights
        self.frequency = frequency

    def __call__(self, inputs, **kwargs):
        return jnp.concatenate([self.scale * jnp.sin(self.frequency * jnp.dot(inputs, self.weights)),
                                self.scale * jnp.cos(self.frequency * jnp.dot(inputs, self.weights))])

