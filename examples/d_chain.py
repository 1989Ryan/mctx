import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import functools
from typing import Tuple

import chex
import jax
import jax.numpy as jnp
import mctx
import functools



@chex.dataclass
class Env:
    state: chex.Array
    end_state: chex.Array
    depth: chex.Array
    num_actions: chex.Array
    done: chex.Array
    reward_map: chex.Array
    reward: chex.Array
    uniform_logits: chex.Array 

def env_reset(start,  depth, num_actions):
    reward_map = [d / depth for d in range(depth + 1)] 
    print(reward_map)
    end = [[1, i - 1] for i in range(depth, 0, -1)] + [[0, depth]]
    print(end)
    print(jax.nn.softmax(jnp.ones(num_actions)))
    return Env(
        state = jnp.array(start),
        end_state = jnp.array(end),
        depth = jnp.array(depth),
        num_actions = jnp.int8(num_actions),
        done = jnp.bool_(False),
        reward_map = jnp.array(reward_map),
        reward = jnp.float32(0.0),
        uniform_logits = jnp.ones(num_actions)
    )


def valid_action_mask(env: Env) -> chex.Array:
    return jnp.where(env.done, False, True)


def env_step(env: Env, action: chex.Array) -> tuple[Env, chex.Array, chex.Array]:
    # move state according to action
    action = jnp.where(valid_action_mask(env), jnp.array([action, 1-action]), 
                       jnp.array([0, 0]))

    next_state = env.state + action
    # calculate reward if the next state is in the end state
    # the end state is a list of states, and the reward is a list of rewards
    # the reward_map and end_state has the same indexing
    reward = jnp.sum(jnp.where(jnp.all(next_state == env.end_state, axis=1), 
                        env.reward_map, 0))


    # calculate done if the next state is in the end state
    # jax.debug.print("{info}", info=env.end_state)
    # jax.debug.print("{info}", info=jnp.all(next_state == env.end_state, axis=-1))
    done = jnp.any(jnp.all(next_state == env.end_state, axis=-1))

    # calculate depth
    # return the next state, depth, done, reward
    new_env = Env(
        state=next_state, 
        end_state=env.end_state, 
        depth=env.depth, 
        num_actions=env.num_actions, 
        done=done, 
        reward_map=env.reward_map, 
        reward=reward,
        uniform_logits=env.uniform_logits)
    return new_env, reward, done

@jax.jit
def rollout(env: Env, rng_key: chex.PRNGKey) -> chex.Array:
    def cond(inputs):
        env, key = inputs
        return ~env.done
    
    def step(inputs):
        env, key = inputs
        key, subkey = jax.random.split(key)
        action = jax.random.categorical(subkey, logits=env.uniform_logits)
        env, reward, done = env_step(env, action)
        # jax.debug.print("{info}", info=env.state)
        return env, key
    
    leaf, key = jax.lax.while_loop(cond, step, (env, rng_key))
    return leaf.reward

def value_fn(env: Env, rng_key: chex.PRNGKey) -> chex.Array:
    return rollout(env, rng_key).astype(jnp.float32)


def root_fn(env: Env, rng_key: chex.PRNGKey) -> mctx.RootFnOutput:
    return mctx.RootFnOutput(
        prior_logits = env.uniform_logits,
        value = value_fn(env, rng_key),
        embedding = env
    )


def recurrent_fn(params, rng_key, action, embedding):
    env = embedding
    env, reward, done = env_step(env, action)
    recurrent_fn_output = mctx.RecurrentFnOutput(
        reward=reward,
        discount=jnp.where(done, 0, 1).astype(jnp.float32),
        prior_logits=env.uniform_logits,
        value=jnp.where(done, 0, value_fn(env, rng_key)).astype(jnp.float32),
    )
    return recurrent_fn_output, env

available_devices = jax.devices()
@functools.partial(jax.jit, static_argnums=(2,))
def run_mcts(rng_key: chex.PRNGKey, env: Env, num_simulations: int) -> chex.Array:
    batch_size = 1024
    # jax.debug.print("{info}", info=env.depth)
    key1, key2 = jax.random.split(rng_key)
    # policy_output = mctx.gumbel_muzero_policy(
    # policy_output = mctx.muzero_policy(
    # policy_output = mctx.sprites_muzero_policy(
    # policy_output = mctx.sprites_gumbel_muzero_policy_baseline(
    policy_output = mctx.sprites_gumbel_muzero_policy(
    # policy_output = mctx.parallel_pimct_policy(
        params=None,
        rng_key=key1,
        root=jax.vmap(root_fn, (None, 0), 0)(env, jax.random.split(key2, batch_size)),
        recurrent_fn=jax.vmap(recurrent_fn, (None, 0, 0, 0)),
        num_simulations=num_simulations,
        max_depth=env.depth,
        # qtransform=mctx.qtransform_by_parent_and_siblings,
        # qtransform=functools.partial(mctx.qtransform_by_min_max, min_value=0, max_value=1),
        qtransform=functools.partial(mctx.qtransform_completed_by_mix_value,
                                    rescale_values=False, 
                                     ),
        # dirichlet_fraction=0.0,
        # c_param=1.414,
        # pb_c_init = 1.414,
        # num_samples=1, 
    )
    return policy_output


env = env_reset(
    start = [0, 0],
    depth = 5,
    num_actions = 2,
)

policy_output = run_mcts(jax.random.PRNGKey(0), env, 1000)
# print(policy_output.search_tree.summary().qvalues)
# print(policy_output.search_tree.children_rewards[:, 0, :].mean(axis=0))
# print(policy_output.search_tree.children_discounts[:, 0, :].mean(axis=0))
# print(policy_output.search_tree.children_values[:, 0, :].mean(axis=0))
print(policy_output.search_tree.summary().qvalues.mean(axis=0))
print(policy_output.search_tree.summary().qvalues.max(axis=0))
print(policy_output.search_tree.summary().qvalues.min(axis=0))
print(policy_output.search_tree.summary().value.mean(axis=0))
print(policy_output.search_tree.summary().value.max(axis=0))
print(policy_output.search_tree.summary().value.min(axis=0))
print(policy_output.action_weights.mean(axis=0))
