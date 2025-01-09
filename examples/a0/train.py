# Copyright 2023 The Pgx Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import pickle
import time
from functools import partial
from typing import NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import mctx
import optax
import pgx
import wandb
from omegaconf import OmegaConf
from pgx.experimental import auto_reset
from pydantic import BaseModel

from network import AZNet

devices = jax.local_devices()
print(devices)
# devices = [devices[i] for i in range(2)]
# devices = [devices[i] for i in range(2, 4)]
num_devices = len(devices)


class Config(BaseModel):
    env_id: pgx.EnvId = "go_9x9"
    seed: int = 42
    max_num_iters: int = 400
    # network params
    num_channels: int = 128
    num_layers: int = 6
    resnet_v2: bool = True
    # selfplay params
    selfplay_batch_size: int = 1024
    num_simulations: int = 32
    num_samples: int = 4
    max_num_steps: int = 256
    temperature: float = 1.0
    fraction: float = 1.0
    # training params
    training_batch_size: int = 4096
    learning_rate: float = 0.001
    # eval params
    eval_interval: int = 10
    eval_simulation: int = 200
    eval_baseline_simulation: int = 32
    class Config:
        extra = "forbid"


conf_dict = OmegaConf.from_cli()
config: Config = Config(**conf_dict)
print(config)

env = pgx.make(config.env_id)
# baseline = pgx.make_baseline_model(config.env_id + "_v0")

# baseline = AZNet(
#         env.num_actions,
#         config.num_channels,
#         config.num_layers,
#         config.resnet_v2,
#     )


ckpt = "baseline/000200.ckpt"
with open(ckpt, "rb") as f:
    dic = pickle.load(f)
    baseline = dic["model"]
    baseline_model = {'params': baseline['params'], 'batch_stats': baseline['batch_stats']}
    # baseline_model = jax.device_put_replicated(baseline_model, devices)

# def forward_fn(x, is_training=True):
#     net = AZNet(
#         num_actions=env.num_actions,
#         num_channels=config.num_channels,
#         num_blocks=config.num_layers,
#         resnet_v2=config.resnet_v2,
#     )
#     policy_out, value_out = net(x, is_training=not is_eval)
#     return policy_out, value_out


forward = AZNet(
        env.num_actions,
        config.num_channels,
        config.num_layers,
        config.resnet_v2,
    )

optimizer = optax.adam(learning_rate=config.learning_rate)


def recurrent_fn(model, rng_key: jnp.ndarray, action: jnp.ndarray, state: pgx.State):
    # model: params
    # state: embedding
    # del rng_key
    model_params, model_state = model['params'], model['batch_stats']

    current_player = state.current_player
    state = jax.vmap(env.step)(state, action, rng_key)

    logits, value = forward.apply(
        # model_params, model_state,
        {'params': model_params, 'batch_stats': model_state}, 
        state.observation, is_training=False)
    # mask invalid actions
    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)

    reward = state.rewards[jnp.arange(state.rewards.shape[0]), current_player]
    value = jnp.where(state.terminated, 0.0, value)
    discount = -1.0 * jnp.ones_like(value)
    discount = jnp.where(state.terminated, 0.0, discount)

    recurrent_fn_output = mctx.RecurrentFnOutput(
        reward=reward,
        discount=discount,
        prior_logits=logits,
        value=value,
    )
    return recurrent_fn_output, state


class SelfplayOutput(NamedTuple):
    obs: jnp.ndarray
    reward: jnp.ndarray
    terminated: jnp.ndarray
    action_weights: jnp.ndarray
    discount: jnp.ndarray


@partial(jax.pmap, in_axes=(0, 0, None))
def selfplay(model, rng_key: jnp.ndarray, temperature=1.0) -> SelfplayOutput:
    if isinstance(model, tuple):
        model_params, model_state = model
    elif isinstance(model, dict):
        model_params, model_state = model['params'], model['batch_stats']
    else:
        raise ValueError("model should be a tuple or a dict, but got {}".format(type(model)))
    batch_size = config.selfplay_batch_size // num_devices

    def step_fn(state, key) -> SelfplayOutput:
        key1, key2 = jax.random.split(key)
        observation = state.observation

        logits, value= forward.apply(
            {'params': model_params, 'batch_stats': model_state},
            # model_params, model_state, 
            state.observation, is_training=False
        )
        root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=state)
        policy_output = mctx.sprites_gumbel_muzero_policy(
            params={'params': model_params, 'batch_stats': model_state},
            rng_key=key1,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=config.num_simulations,
            # num_samples=config.num_samples, 
            invalid_actions=~state.legal_action_mask,
            # qtransform=mctx.qtransform_by_parent_and_siblings,
            qtransform=mctx.qtransform_completed_by_mix_value,
            # gumbel_scale=1.0,
            # dirichlet_fraction= 0.0, 
            # fraction=config.fraction,
            # temperature=temperature,
        )
        actor = state.current_player
        keys = jax.random.split(key2, batch_size)
        state = jax.vmap(auto_reset(env.step, env.init))(state, policy_output.action, keys)
        discount = -1.0 * jnp.ones_like(value)
        discount = jnp.where(state.terminated, 0.0, discount)
        return state, SelfplayOutput(
            obs=observation,
            action_weights=policy_output.action_weights,
            reward=state.rewards[jnp.arange(state.rewards.shape[0]), actor],
            terminated=state.terminated,
            discount=discount,
        )

    # Run selfplay for max_num_steps by batch
    rng_key, sub_key = jax.random.split(rng_key)
    keys = jax.random.split(sub_key, batch_size)
    state = jax.vmap(env.init)(keys)
    key_seq = jax.random.split(rng_key, config.max_num_steps)
    _, data = jax.lax.scan(step_fn, state, key_seq)

    return data


class Sample(NamedTuple):
    obs: jnp.ndarray
    policy_tgt: jnp.ndarray
    value_tgt: jnp.ndarray
    mask: jnp.ndarray


@jax.pmap
def compute_loss_input(data: SelfplayOutput) -> Sample:
    batch_size = config.selfplay_batch_size // num_devices
    # If episode is truncated, there is no value target
    # So when we compute value loss, we need to mask it
    value_mask = jnp.cumsum(data.terminated[::-1, :], axis=0)[::-1, :] >= 1

    # Compute value target
    def body_fn(carry, i):
        ix = config.max_num_steps - i - 1
        v = data.reward[ix] + data.discount[ix] * carry
        return v, v

    _, value_tgt = jax.lax.scan(
        body_fn,
        jnp.zeros(batch_size),
        jnp.arange(config.max_num_steps),
    )
    value_tgt = value_tgt[::-1, :]

    return Sample(
        obs=data.obs,
        policy_tgt=data.action_weights,
        value_tgt=value_tgt,
        mask=value_mask,
    )


def loss_fn(model_params, model_state, samples: Sample):
    (logits, value), model_state = forward.apply(
        {'params': model_params, 'batch_stats': model_state},
        # model_params, model_state, 
        samples.obs, is_training=True, mutable='batch_stats',
    )

    policy_loss = optax.softmax_cross_entropy(logits, samples.policy_tgt)
    policy_loss = jnp.mean(policy_loss)

    l2_loss = optax.l2_loss(logits)
    l2_loss = jnp.mean(l2_loss)

    value_loss = optax.l2_loss(value, samples.value_tgt)
    value_loss = jnp.mean(value_loss * samples.mask)  # mask if the episode is truncated

    return policy_loss + value_loss , (model_state['batch_stats'], policy_loss, value_loss, l2_loss)


@partial(jax.pmap, axis_name="i")
def train(model, opt_state, data: Sample):
    if isinstance(model, tuple):
        model_params, model_state = model
    elif isinstance(model, dict):
        model_params, model_state = model['params'], model['batch_stats']
    else:
        raise ValueError("model should be a tuple or a dict, but got {}".format(type(model)))
    grads, (model_state, policy_loss, value_loss, l2_loss) = jax.grad(loss_fn, has_aux=True)(
        model_params, model_state, data
    )
    grads = jax.lax.pmean(grads, axis_name="i")
    updates, opt_state = optimizer.update(grads, opt_state)
    model_params = optax.apply_updates(model_params, updates)
    model = {'params': model_params, 'batch_stats': model_state}
    return model, opt_state, policy_loss, value_loss, l2_loss


@jax.pmap
def evaluate(rng_key, my_model):
    """A simplified evaluation by sampling. Only for debugging. 
    Please use MCTS and run tournaments for serious evaluation."""
    my_player = 0
    my_model_params, my_model_state = my_model['params'], my_model['batch_stats']

    key, subkey = jax.random.split(rng_key)
    batch_size = config.selfplay_batch_size // num_devices
    keys = jax.random.split(subkey, batch_size)
    state = jax.vmap(env.init)(keys)

    def body_fn(val):
        key, state, R = val
        my_logits, my_value= forward.apply(
            # my_model_params, my_model_state, 
            {'params': my_model_params, 'batch_stats': my_model_state},
            state.observation, is_training=False
        )

        key1, key = jax.random.split(key)
        root = mctx.RootFnOutput(prior_logits=my_logits, value=my_value, embedding=state)

        my_output = mctx.sprites_gumbel_muzero_policy(
            params={'params': my_model_params, 'batch_stats': my_model_state},
            rng_key=key1,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=config.eval_simulation,
            # num_samples=1, 
            # dirichlet_fraction=0.0,
            invalid_actions=~state.legal_action_mask,
            qtransform=mctx.qtransform_completed_by_mix_value,
            # fraction=config.fraction,
            gumbel_scale=1.0,
        )

        opp_logits, opp_value = forward.apply(
            # my_model_params, my_model_state, 
            {'params': baseline_model['params'], 'batch_stats': baseline_model['batch_stats']},
            state.observation, is_training=False
        )
        # opp_logits, _ = baseline(state.observation)
        # key, subkey = jax.random.split(key)
        # greedy_action = jax.random.categorical(subkey, opp_logits, axis=-1)
        root = mctx.RootFnOutput(prior_logits=opp_logits, value=opp_value, embedding=state)

        key1, key = jax.random.split(key)
        base_output = mctx.gumbel_muzero_policy(
            params={'params': baseline_model['params'], 'batch_stats': baseline_model['batch_stats']},
            rng_key=key1,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=config.eval_baseline_simulation,
            invalid_actions=~state.legal_action_mask,
            qtransform=mctx.qtransform_completed_by_mix_value,
            gumbel_scale=1.0,
        )


        is_my_turn = (state.current_player == my_player) # .reshape((-1, 1))
        # action = jnp.where(is_my_turn, my_output.action, greedy_action, )
        # logits = jnp.where(is_my_turn, my_logits, opp_logits)
        # action = jax.random.categorical(subkey, logits, axis=-1)
        action = jnp.where(is_my_turn, my_output.action, base_output.action, )
        # print(action.shape)
        subkey, key = jax.random.split(key)
        subkeys = jax.random.split(subkey, batch_size)
        state = jax.vmap(env.step)(state, action, subkeys)
        R = R + state.rewards[jnp.arange(batch_size), my_player]
        return (key, state, R)

    _, _, R = jax.lax.while_loop(
        lambda x: ~(x[1].terminated.all()), body_fn, (key, state, jnp.zeros(batch_size))
    )
    return R


if __name__ == "__main__":
    wandb.init(project="pgx-az",\
            config=config.model_dump())

    # Initialize model and opt_state
    dummy_state = jax.vmap(env.init)(jax.random.split(jax.random.PRNGKey(0), 2))
    dummy_input = dummy_state.observation
    model = forward.init(jax.random.PRNGKey(0), dummy_input, 
                          is_training=False)  # (params, state)
    params = model['params']
    opt_state = optimizer.init(params=params)
    # replicates to all devices
    model, opt_state = jax.device_put_replicated((model, opt_state), devices)

    # Prepare checkpoint dir
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
    now = now.strftime("%Y%m%d%H%M%S")
    ckpt_dir = os.path.join("checkpoints", f"{config.env_id}_{now}_sprites_seed_{config.seed}")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Initialize logging dict
    iteration: int = 0
    hours: float = 0.0
    frames: int = 0
    log = {"iteration": iteration, "hours": hours, "frames": frames}
    temperature = config.temperature
    # num_samples = config.num_samples

    rng_key = jax.random.PRNGKey(config.seed)
    while True:
        # if iteration % 133 == 132:
        #     temperature = temperature / 3
            # num_samples = max(num_samples // 4, 1)

        if iteration % config.eval_interval == 0:
            # Evaluation
            rng_key, subkey = jax.random.split(rng_key)
            keys = jax.random.split(subkey, num_devices)
            R = evaluate(keys, model)
            win_rate = ((R == 1).sum() / R.size).item()
            log.update(
                {
                    f"eval/vs_baseline/avg_R": R.mean().item(),
                    f"eval/vs_baseline/win_rate": win_rate,
                    f"eval/vs_baseline/Elo": 1000 + 400 * jnp.log10(win_rate / (1 - win_rate)).item(),
                    f"eval/vs_baseline/draw_rate": ((R == 0).sum() / R.size).item(),
                    f"eval/vs_baseline/lose_rate": ((R == -1).sum() / R.size).item(),
                }
            )

            # Store checkpoints
            model_0, opt_state_0 = jax.tree_util.tree_map(lambda x: x[0], (model, opt_state))
            with open(os.path.join(ckpt_dir, f"{iteration:06d}.ckpt"), "wb") as f:
                dic = {
                    "config": config,
                    "rng_key": rng_key,
                    "model": jax.device_get(model_0),
                    "opt_state": jax.device_get(opt_state_0),
                    "iteration": iteration,
                    "frames": frames,
                    "hours": hours,
                    "pgx.__version__": pgx.__version__,
                    "env_id": env.id,
                    "env_version": env.version,
                }
                pickle.dump(dic, f)

        print(log)
        wandb.log(log)

        if iteration >= config.max_num_iters:
            break

        iteration += 1
        log = {"iteration": iteration}
        st = time.time()

        # Selfplay
        rng_key, subkey = jax.random.split(rng_key)
        keys = jax.random.split(subkey, num_devices)
        data: SelfplayOutput = selfplay(model, keys, temperature)
        samples: Sample = compute_loss_input(data)

        # Shuffle samples and make minibatches
        samples = jax.device_get(samples)  # (#devices, batch, max_num_steps, ...)
        frames += samples.obs.shape[0] * samples.obs.shape[1] * samples.obs.shape[2]
        samples = jax.tree_util.tree_map(lambda x: x.reshape((-1, *x.shape[3:])), samples)
        rng_key, subkey = jax.random.split(rng_key)
        ixs = jax.random.permutation(subkey, jnp.arange(samples.obs.shape[0]))
        samples = jax.tree_util.tree_map(lambda x: x[ixs], samples)  # shuffle
        num_updates = samples.obs.shape[0] // config.training_batch_size
        minibatches = jax.tree_util.tree_map(
            lambda x: x.reshape((num_updates, num_devices, -1) + x.shape[1:]), samples
        )

        # Training
        policy_losses, value_losses, l2_losses = [], [], []
        for i in range(num_updates):
            minibatch: Sample = jax.tree_util.tree_map(lambda x: x[i], minibatches)
            model, opt_state, policy_loss, value_loss, l2_loss = train(model, opt_state, minibatch)
            policy_losses.append(policy_loss.mean().item())
            value_losses.append(value_loss.mean().item())
            l2_losses.append(l2_loss.mean().item()) 
        policy_loss = sum(policy_losses) / len(policy_losses)
        value_loss = sum(value_losses) / len(value_losses)
        l2_loss = sum(l2_losses) / len(l2_losses)

        et = time.time()
        hours += (et - st) / 3600
        log.update(
            {
                "train/policy_loss": policy_loss,
                "train/value_loss": value_loss,
                "train/l2_loss": l2_loss,
                "hours": hours,
                "frames": frames,
            }
        )