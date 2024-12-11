# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
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
# ==============================================================================
"""A JAX implementation of batched MCTS."""
import functools
from typing import Any, NamedTuple, Optional, Tuple, TypeVar

import chex
import jax
import jax.numpy as jnp

from mctx._src import action_selection
from mctx._src import base
from mctx._src import tree as tree_lib
from mctx._src import qtransforms

Tree = tree_lib.Tree
T = TypeVar("T")


def assign_unique_indices(tree, batch_range, node_index, action_index, sim):
    """
    Automatically assign unique next_node_index for unique node-action pairs within each batch.

    Args:
        tree: A data structure with tree.children_index to query next_node_index.
        batch_range: The batch indices, shape [batch_size, 1, 1].
        node_index: Node indices, shape [batch_size, sample_size, 1].
        action_index: Action indices, shape [batch_size, sample_size, 1].
        sim: Current simulation index to assign new indices.
    
    Returns:
        Updated next_node_index, shape [batch_size, sample_size, 1].
    """
    # Query the current next_node_index
    next_node_index = tree.children_index[batch_range, node_index, action_index]  # [batch_size, sample_size]

    # Flatten the sample dimension for uniqueness checks within each batch
    flat_indices = jnp.stack([node_index, action_index], axis=-1)  # [batch_size, sample_size, 2]

    # expand sim 

    def process_batch(batch_flat_indices, batch_next_node_index, batch_sim):
        # Identify unique node-action pairs within the batch with fixed output size
        max_unique_pairs = batch_flat_indices.shape[0]  # Maximum number of unique pairs (sample_size)
        unique_flat_indices, unique_indices = jnp.unique(
            batch_flat_indices, axis=0, return_inverse=True, size=max_unique_pairs, fill_value=-1
        )

        # Mask for valid unique entries (i.e., not padded by `fill_value=-1`)
        valid_mask = unique_flat_indices[:, 0] != -1
        num_valid_unique = jnp.sum(valid_mask)

        # Generate a static array for new indices
        max_possible_indices = jnp.arange(1, 1 + max_unique_pairs) + batch_sim

        # Mask out invalid indices (set to -1 where valid_mask is False)
        new_node_indices = jnp.where(
            jnp.arange(max_unique_pairs) < num_valid_unique,
            max_possible_indices,
            -1
        )

        # Map unique node-action pairs back to the original flat shape
        expanded_new_node_indices = new_node_indices[unique_indices]

        # Ensure shapes match for broadcasting
        expanded_new_node_indices = expanded_new_node_indices.reshape(batch_next_node_index.shape)

        # Update next_node_index for unvisited nodes (-1) with the newly assigned unique indices
        return jnp.where(batch_next_node_index == Tree.UNVISITED, expanded_new_node_indices, batch_next_node_index), num_valid_unique

    # Vectorized processing for each batch using vmap
    updated_next_node_index, num_valid_unique = jax.vmap(process_batch, in_axes=(0, 0, 0))(
        flat_indices, next_node_index, sim
    )

    return updated_next_node_index.astype(jnp.int32), num_valid_unique + sim

def sprites_search(
    params: base.Params,
    rng_key: chex.PRNGKey,
    *,
    root: base.RootFnOutput,
    recurrent_fn: base.RecurrentFn,
    root_action_selection_fn: base.RootActionSelectionFn,
    interior_action_selection_fn: base.InteriorActionSelectionFn,
    num_simulations: int,
    max_depth: Optional[int] = None,
    invalid_actions: Optional[chex.Array] = None,
    extra_data: Any = None,
    pb_c_base: chex.Numeric = 1.25,
    pb_c_init: chex.Numeric = 19652,
    num_choices: int = 1,
    loop_fn: base.LoopFn = jax.lax.fori_loop,
    qtransform: base.QTransform = qtransforms.qtransform_by_parent_and_siblings) -> Tree:
  """Performs a full search and returns sampled actions.

  In the shape descriptions, `B` denotes the batch dimension.

  Args:
    params: params to be forwarded to root and recurrent functions.
    rng_key: random number generator state, the key is consumed.
    root: a `(prior_logits, value, embedding)` `RootFnOutput`. The
      `prior_logits` are from a policy network. The shapes are
      `([B, num_actions], [B], [B, ...])`, respectively.
    recurrent_fn: a callable to be called on the leaf nodes and unvisited
      actions retrieved by the simulation step, which takes as args
      `(params, rng_key, action, embedding)` and returns a `RecurrentFnOutput`
      and the new state embedding. The `rng_key` argument is consumed.
    root_action_selection_fn: function used to select an action at the root.
    interior_action_selection_fn: function used to select an action during
      simulation.
    num_simulations: the number of simulations.
    max_depth: maximum search tree depth allowed during simulation, defined as
      the number of edges from the root to a leaf node.
    invalid_actions: a mask with invalid actions at the root. In the
      mask, invalid actions have ones, and valid actions have zeros.
      Shape `[B, num_actions]`.
    extra_data: extra data passed to `tree.extra_data`. Shape `[B, ...]`.
    loop_fn: Function used to run the simulations. It may be required to pass
      hk.fori_loop if using this function inside a Haiku module.

  Returns:
    `SearchResults` containing outcomes of the search, e.g. `visit_counts`
    `[B, num_actions]`.
  """
  action_selection_fn = action_selection.switching_action_selection_wrapper_parallel(
      root_action_selection_fn=root_action_selection_fn,
      interior_action_selection_fn=interior_action_selection_fn
  )

  # Do simulation, expansion, and backward steps.
  batch_size = root.value.shape[0]
  batch_range = jnp.arange(batch_size)
  batch_range = jnp.repeat(jnp.expand_dims(batch_range, axis=1), num_choices, axis=1)
  if max_depth is None:
    max_depth = num_simulations
  if invalid_actions is None:
    invalid_actions = jnp.zeros_like(root.prior_logits)

  def body_fun(sim, loop_state):
    # jax.debug.print("sim {sim} / {total}", sim=sim, total=num_simulations)
    rng_key, tree, last_node_index = loop_state
    rng_key, simulate_key, expand_key = jax.random.split(rng_key, 3)
    # simulate is vmapped and expects batched rng keys.
    simulate_keys = jax.random.split(simulate_key, batch_size)
    # jax.debug.print("simulate running")
    parent_index, action = parallel_sampling_simulate(
        simulate_keys, tree, num_choices, action_selection_fn, max_depth)
    # jax.debug.print("simulate done")
    # A node first expanded on simulation `i`, will have node index `i`.
    # assign unique next node index 

    next_node_index, last_node_index = assign_unique_indices(tree, batch_range, parent_index, action, last_node_index)
    # jax.debug.print("expand running")
    tree = parallel_expand(
        params, expand_key, tree, recurrent_fn, parent_index,
        action, next_node_index, num_choices)
    # jax.debug.print("expand done")
    def loop_backward(index, inputs):
      tree, next_node_index_ = inputs
      # jax.debug.print("backward {index}", index=index) 
      next_node_index = next_node_index_.at[:, index].get()
      # jax.debug.print("next_node_index shape {next_node_index_}", next_node_index_=next_node_index.shape)
      tree = pimct_backward(tree, next_node_index, pb_c_init, pb_c_base, qtransform)
      # jax.debug.print("tree shape {tree}", tree=tree.children_index.shape)
      return tree, next_node_index_
    # jax.debug.print("backward running")
    tree, _ = jax.lax.fori_loop(0, num_choices, loop_backward, (tree, next_node_index))
    # jax.debug.print("backward done")
    # tree = pimct_backward_parallel(tree, next_node_index, c_param, qtransform)
    # jax.debug.print("rng_key {rng_key}", rng_key=rng_key)
    # jax.debug.print("tree shape {tree}", tree=tree.children_index.shape)
    # jax.debug.print("last_node_index {last_node_index}", last_node_index=next_node_index)
    loop_state = rng_key, tree, last_node_index
    # jax.debug.print("return loop state")
    # jax.debug.print("Num simulations in total {sim}", sim=num_simulations)
    return loop_state

  # Allocate all necessary storage.
  tree = instantiate_tree_from_root(root, num_simulations * num_choices,
                                    root_invalid_actions=invalid_actions,
                                    extra_data=extra_data)
  _, tree, _ = loop_fn(
      0, num_simulations, body_fun, (rng_key, tree, jnp.zeros(batch_size)))

  return tree



# @functools.partial(jax.pmap, in_axes=(None, 0, 0, None, None, None, None, None), out_axes=0)
def parallel_pimct_search(
    params: base.Params,
    rng_key: chex.PRNGKey,
    *,
    root: base.RootFnOutput,
    recurrent_fn: base.RecurrentFn,
    root_action_selection_fn: base.RootActionSelectionFn,
    interior_action_selection_fn: base.InteriorActionSelectionFn,
    num_simulations: int,
    max_depth: Optional[int] = None,
    invalid_actions: Optional[chex.Array] = None,
    extra_data: Any = None,
    c_param: chex.Numeric = 1.414,
    num_choices: int = 1,
    loop_fn: base.LoopFn = jax.lax.fori_loop,
    qtransform: base.QTransform = qtransforms.qtransform_by_parent_and_siblings) -> Tree:
  """Performs a full search and returns sampled actions.

  In the shape descriptions, `B` denotes the batch dimension.

  Args:
    params: params to be forwarded to root and recurrent functions.
    rng_key: random number generator state, the key is consumed.
    root: a `(prior_logits, value, embedding)` `RootFnOutput`. The
      `prior_logits` are from a policy network. The shapes are
      `([B, num_actions], [B], [B, ...])`, respectively.
    recurrent_fn: a callable to be called on the leaf nodes and unvisited
      actions retrieved by the simulation step, which takes as args
      `(params, rng_key, action, embedding)` and returns a `RecurrentFnOutput`
      and the new state embedding. The `rng_key` argument is consumed.
    root_action_selection_fn: function used to select an action at the root.
    interior_action_selection_fn: function used to select an action during
      simulation.
    num_simulations: the number of simulations.
    max_depth: maximum search tree depth allowed during simulation, defined as
      the number of edges from the root to a leaf node.
    invalid_actions: a mask with invalid actions at the root. In the
      mask, invalid actions have ones, and valid actions have zeros.
      Shape `[B, num_actions]`.
    extra_data: extra data passed to `tree.extra_data`. Shape `[B, ...]`.
    loop_fn: Function used to run the simulations. It may be required to pass
      hk.fori_loop if using this function inside a Haiku module.

  Returns:
    `SearchResults` containing outcomes of the search, e.g. `visit_counts`
    `[B, num_actions]`.
  """
  action_selection_fn = action_selection.switching_action_selection_wrapper_parallel(
      root_action_selection_fn=root_action_selection_fn,
      interior_action_selection_fn=interior_action_selection_fn
  )

  # Do simulation, expansion, and backward steps.
  batch_size = root.value.shape[0]
  batch_range = jnp.arange(batch_size)
  batch_range = jnp.repeat(jnp.expand_dims(batch_range, axis=1), num_choices, axis=1)
  if max_depth is None:
    max_depth = num_simulations
  if invalid_actions is None:
    invalid_actions = jnp.zeros_like(root.prior_logits)

  def body_fun(sim, loop_state):
    # jax.debug.print("sim {sim} / {total}", sim=sim, total=num_simulations)
    rng_key, tree, last_node_index = loop_state
    rng_key, simulate_key, expand_key = jax.random.split(rng_key, 3)
    # simulate is vmapped and expects batched rng keys.
    simulate_keys = jax.random.split(simulate_key, batch_size)
    # jax.debug.print("simulate running")
    # print("running simulation")
    parent_index, action = parallel_sampling_simulate(
        simulate_keys, tree, num_choices, action_selection_fn, max_depth)
    # print("simulation done")
    # jax.debug.print("simulate done")
    # A node first expanded on simulation `i`, will have node index `i`.
    # assign unique next node index 

    next_node_index, last_node_index = assign_unique_indices(tree, batch_range, parent_index, action, last_node_index)
    # jax.debug.print("expand running")
    tree = parallel_expand(
        params, expand_key, tree, recurrent_fn, parent_index,
        action, next_node_index, num_choices)
    # jax.debug.print("expand done")
    def loop_backward(index, inputs):
      tree, next_node_index_ = inputs
      # jax.debug.print("backward {index}", index=index) 
      next_node_index = next_node_index_.at[:, index].get()
      # jax.debug.print("next_node_index shape {next_node_index_}", next_node_index_=next_node_index.shape)
      tree = pimct_backward_(tree, next_node_index, c_param, qtransform)
      # jax.debug.print("tree shape {tree}", tree=tree.children_index.shape)
      return tree, next_node_index_
    # jax.debug.print("backward running")
    tree, _ = jax.lax.fori_loop(0, num_choices, loop_backward, (tree, next_node_index))
    # jax.debug.print("backward done")
    # tree = pimct_backward_parallel(tree, next_node_index, c_param, qtransform)
    # jax.debug.print("rng_key {rng_key}", rng_key=rng_key)
    # jax.debug.print("tree shape {tree}", tree=tree.children_index.shape)
    # jax.debug.print("last_node_index {last_node_index}", last_node_index=next_node_index)
    loop_state = rng_key, tree, last_node_index
    # jax.debug.print("return loop state")
    # jax.debug.print("Num simulations in total {sim}", sim=num_simulations)
    return loop_state

  # Allocate all necessary storage.
  tree = instantiate_tree_from_root(root, num_simulations * num_choices,
                                    root_invalid_actions=invalid_actions,
                                    extra_data=extra_data)
  _, tree, _ = loop_fn(
      0, num_simulations, body_fun, (rng_key, tree, jnp.zeros(batch_size)))

  return tree

def pimct_search(
    params: base.Params,
    rng_key: chex.PRNGKey,
    *,
    root: base.RootFnOutput,
    recurrent_fn: base.RecurrentFn,
    root_action_selection_fn: base.RootActionSelectionFn,
    interior_action_selection_fn: base.InteriorActionSelectionFn,
    num_simulations: int,
    max_depth: Optional[int] = None,
    invalid_actions: Optional[chex.Array] = None,
    extra_data: Any = None,
    c_param: chex.Numeric = 1.414,
    loop_fn: base.LoopFn = jax.lax.fori_loop,
    qtransform: base.QTransform = qtransforms.qtransform_by_parent_and_siblings) -> Tree:
  """Performs a full search and returns sampled actions.

  In the shape descriptions, `B` denotes the batch dimension.

  Args:
    params: params to be forwarded to root and recurrent functions.
    rng_key: random number generator state, the key is consumed.
    root: a `(prior_logits, value, embedding)` `RootFnOutput`. The
      `prior_logits` are from a policy network. The shapes are
      `([B, num_actions], [B], [B, ...])`, respectively.
    recurrent_fn: a callable to be called on the leaf nodes and unvisited
      actions retrieved by the simulation step, which takes as args
      `(params, rng_key, action, embedding)` and returns a `RecurrentFnOutput`
      and the new state embedding. The `rng_key` argument is consumed.
    root_action_selection_fn: function used to select an action at the root.
    interior_action_selection_fn: function used to select an action during
      simulation.
    num_simulations: the number of simulations.
    max_depth: maximum search tree depth allowed during simulation, defined as
      the number of edges from the root to a leaf node.
    invalid_actions: a mask with invalid actions at the root. In the
      mask, invalid actions have ones, and valid actions have zeros.
      Shape `[B, num_actions]`.
    extra_data: extra data passed to `tree.extra_data`. Shape `[B, ...]`.
    loop_fn: Function used to run the simulations. It may be required to pass
      hk.fori_loop if using this function inside a Haiku module.

  Returns:
    `SearchResults` containing outcomes of the search, e.g. `visit_counts`
    `[B, num_actions]`.
  """
  action_selection_fn = action_selection.switching_action_selection_wrapper(
      root_action_selection_fn=root_action_selection_fn,
      interior_action_selection_fn=interior_action_selection_fn
  )

  # Do simulation, expansion, and backward steps.
  batch_size = root.value.shape[0]
  batch_range = jnp.arange(batch_size)
  if max_depth is None:
    max_depth = num_simulations
  if invalid_actions is None:
    invalid_actions = jnp.zeros_like(root.prior_logits)

  def body_fun(sim, loop_state):
    rng_key, tree = loop_state
    rng_key, simulate_key, expand_key = jax.random.split(rng_key, 3)
    # simulate is vmapped and expects batched rng keys.
    simulate_keys = jax.random.split(simulate_key, batch_size)
    parent_index, action = simulate(
        simulate_keys, tree, action_selection_fn, max_depth)
    # A node first expanded on simulation `i`, will have node index `i`.
    # Node 0 corresponds to the root node.
    next_node_index = tree.children_index[batch_range, parent_index, action]
    next_node_index = jnp.where(next_node_index == Tree.UNVISITED,
                                sim + 1, next_node_index)
  
    tree = expand(
        params, expand_key, tree, recurrent_fn, parent_index,
        action, next_node_index)
    tree = pimct_backward_(tree, next_node_index, c_param, qtransform)
    loop_state = rng_key, tree
    return loop_state

  # Allocate all necessary storage.
  tree = instantiate_tree_from_root(root, num_simulations,
                                    root_invalid_actions=invalid_actions,
                                    extra_data=extra_data)
  _, tree = loop_fn(
      0, num_simulations, body_fun, (rng_key, tree))

  return tree


def sprites_muzero_search(
    params: base.Params,
    rng_key: chex.PRNGKey,
    *,
    root: base.RootFnOutput,
    recurrent_fn: base.RecurrentFn,
    root_action_selection_fn: base.RootActionSelectionFn,
    interior_action_selection_fn: base.InteriorActionSelectionFn,
    num_simulations: int,
    max_depth: Optional[int] = None,
    invalid_actions: Optional[chex.Array] = None,
    extra_data: Any = None,
    pb_c_base: chex.Numeric = 1.25,
    pb_c_init: chex.Numeric = 19652,
    loop_fn: base.LoopFn = jax.lax.fori_loop,
    qtransform: base.QTransform = qtransforms.qtransform_by_parent_and_siblings) -> Tree:
  """Performs a full search and returns sampled actions.

  In the shape descriptions, `B` denotes the batch dimension.

  Args:
    params: params to be forwarded to root and recurrent functions.
    rng_key: random number generator state, the key is consumed.
    root: a `(prior_logits, value, embedding)` `RootFnOutput`. The
      `prior_logits` are from a policy network. The shapes are
      `([B, num_actions], [B], [B, ...])`, respectively.
    recurrent_fn: a callable to be called on the leaf nodes and unvisited
      actions retrieved by the simulation step, which takes as args
      `(params, rng_key, action, embedding)` and returns a `RecurrentFnOutput`
      and the new state embedding. The `rng_key` argument is consumed.
    root_action_selection_fn: function used to select an action at the root.
    interior_action_selection_fn: function used to select an action during
      simulation.
    num_simulations: the number of simulations.
    max_depth: maximum search tree depth allowed during simulation, defined as
      the number of edges from the root to a leaf node.
    invalid_actions: a mask with invalid actions at the root. In the
      mask, invalid actions have ones, and valid actions have zeros.
      Shape `[B, num_actions]`.
    extra_data: extra data passed to `tree.extra_data`. Shape `[B, ...]`.
    loop_fn: Function used to run the simulations. It may be required to pass
      hk.fori_loop if using this function inside a Haiku module.

  Returns:
    `SearchResults` containing outcomes of the search, e.g. `visit_counts`
    `[B, num_actions]`.
  """
  action_selection_fn = action_selection.switching_action_selection_wrapper(
      root_action_selection_fn=root_action_selection_fn,
      interior_action_selection_fn=interior_action_selection_fn
  )

  # Do simulation, expansion, and backward steps.
  batch_size = root.value.shape[0]
  batch_range = jnp.arange(batch_size)
  # batch_range = jnp.repeat(jnp.expand_dims(batch_range, axis=1), num_choices, axis=1)
  if max_depth is None:
    max_depth = num_simulations
  if invalid_actions is None:
    invalid_actions = jnp.zeros_like(root.prior_logits)

  def body_fun(sim, loop_state):
    # jax.debug.print("sim {sim} / {total}", sim=sim, total=num_simulations)
    rng_key, tree = loop_state
    # rng_key, tree, last_node_index = loop_state
    rng_key, simulate_key, expand_key = jax.random.split(rng_key, 3)
    # simulate is vmapped and expects batched rng keys.
    simulate_keys = jax.random.split(simulate_key, batch_size)
    # jax.debug.print("simulate running")
    parent_index, action = simulate(
        simulate_keys, tree, action_selection_fn, max_depth)
    # parent_index, action = parallel_sampling_simulate(
    #     simulate_keys, tree, num_choices, action_selection_fn, max_depth)
    # jax.debug.print("simulate done")
    # A node first expanded on simulation `i`, will have node index `i`.
    # assign unique next node index 
    next_node_index = tree.children_index[batch_range, parent_index, action]
    next_node_index = jnp.where(next_node_index == Tree.UNVISITED,
                                sim + 1, next_node_index)
    # next_node_index, last_node_index = assign_unique_indices(tree, batch_range, parent_index, action, last_node_index)
    # jax.debug.print("expand running")
    tree = expand(
        params, expand_key, tree, recurrent_fn, parent_index,
        action, next_node_index)
    tree = pimct_backward(tree, next_node_index, pb_c_init, pb_c_base, qtransform)

    # tree = backward(tree, next_node_index)
    loop_state = rng_key, tree
    # jax.debug.print("expand done")
    # def loop_backward(index, inputs):
    #   tree, next_node_index_ = inputs
    #   # jax.debug.print("backward {index}", index=index) 
    #   next_node_index = next_node_index_.at[:, index].get()
    #   # jax.debug.print("next_node_index shape {next_node_index_}", next_node_index_=next_node_index.shape)
    #   tree = pimct_backward(tree, next_node_index, pb_c_init, pb_c_base, qtransform)
    #   # jax.debug.print("tree shape {tree}", tree=tree.children_index.shape)
    #   return tree, next_node_index_
    # jax.debug.print("backward running")
    # tree, _ = jax.lax.fori_loop(0, num_choices, loop_backward, (tree, next_node_index))
    # jax.debug.print("backward done")
    # tree = pimct_backward_parallel(tree, next_node_index, c_param, qtransform)
    # jax.debug.print("rng_key {rng_key}", rng_key=rng_key)
    # jax.debug.print("tree shape {tree}", tree=tree.children_index.shape)
    # jax.debug.print("last_node_index {last_node_index}", last_node_index=next_node_index)
    # loop_state = rng_key, tree, last_node_index
    # jax.debug.print("return loop state")
    # jax.debug.print("Num simulations in total {sim}", sim=num_simulations)
    return loop_state

  # Allocate all necessary storage.
  tree = instantiate_tree_from_root(root, num_simulations,
                                    root_invalid_actions=invalid_actions,
                                    extra_data=extra_data)
  _, tree = loop_fn(
      0, num_simulations, body_fun, (rng_key, tree))

  return tree



def ments_search(
    params: base.Params,
    rng_key: chex.PRNGKey,
    *,
    root: base.RootFnOutput,
    recurrent_fn: base.RecurrentFn,
    root_action_selection_fn: base.RootActionSelectionFn,
    interior_action_selection_fn: base.InteriorActionSelectionFn,
    num_simulations: int,
    max_depth: Optional[int] = None,
    invalid_actions: Optional[chex.Array] = None,
    extra_data: Any = None,
    tau: chex.Numeric = 1e-1,
    loop_fn: base.LoopFn = jax.lax.fori_loop) -> Tree:
  """Performs a full search and returns sampled actions.

  In the shape descriptions, `B` denotes the batch dimension.

  Args:
    params: params to be forwarded to root and recurrent functions.
    rng_key: random number generator state, the key is consumed.
    root: a `(prior_logits, value, embedding)` `RootFnOutput`. The
      `prior_logits` are from a policy network. The shapes are
      `([B, num_actions], [B], [B, ...])`, respectively.
    recurrent_fn: a callable to be called on the leaf nodes and unvisited
      actions retrieved by the simulation step, which takes as args
      `(params, rng_key, action, embedding)` and returns a `RecurrentFnOutput`
      and the new state embedding. The `rng_key` argument is consumed.
    root_action_selection_fn: function used to select an action at the root.
    interior_action_selection_fn: function used to select an action during
      simulation.
    num_simulations: the number of simulations.
    max_depth: maximum search tree depth allowed during simulation, defined as
      the number of edges from the root to a leaf node.
    invalid_actions: a mask with invalid actions at the root. In the
      mask, invalid actions have ones, and valid actions have zeros.
      Shape `[B, num_actions]`.
    extra_data: extra data passed to `tree.extra_data`. Shape `[B, ...]`.
    loop_fn: Function used to run the simulations. It may be required to pass
      hk.fori_loop if using this function inside a Haiku module.

  Returns:
    `SearchResults` containing outcomes of the search, e.g. `visit_counts`
    `[B, num_actions]`.
  """
  action_selection_fn = action_selection.switching_action_selection_wrapper(
      root_action_selection_fn=root_action_selection_fn,
      interior_action_selection_fn=interior_action_selection_fn
  )

  # Do simulation, expansion, and backward steps.
  batch_size = root.value.shape[0]
  batch_range = jnp.arange(batch_size)
  if max_depth is None:
    max_depth = num_simulations
  if invalid_actions is None:
    invalid_actions = jnp.zeros_like(root.prior_logits)

  def body_fun(sim, loop_state):
    rng_key, tree = loop_state
    rng_key, simulate_key, expand_key = jax.random.split(rng_key, 3)
    # simulate is vmapped and expects batched rng keys.
    simulate_keys = jax.random.split(simulate_key, batch_size)
    parent_index, action = simulate(
        simulate_keys, tree, action_selection_fn, max_depth)
    # A node first expanded on simulation `i`, will have node index `i`.
    # Node 0 corresponds to the root node.
    next_node_index = tree.children_index[batch_range, parent_index, action]
    next_node_index = jnp.where(next_node_index == Tree.UNVISITED,
                                sim + 1, next_node_index)
  
    tree = expand(
        params, expand_key, tree, recurrent_fn, parent_index,
        action, next_node_index)
    tree = maximum_entropy_backward(tree, next_node_index, tau)
    loop_state = rng_key, tree
    return loop_state

  # Allocate all necessary storage.
  tree = instantiate_tree_from_root(root, num_simulations,
                                    root_invalid_actions=invalid_actions,
                                    extra_data=extra_data)
  _, tree = loop_fn(
      0, num_simulations, body_fun, (rng_key, tree))

  return tree





def search(
    params: base.Params,
    rng_key: chex.PRNGKey,
    *,
    root: base.RootFnOutput,
    recurrent_fn: base.RecurrentFn,
    root_action_selection_fn: base.RootActionSelectionFn,
    interior_action_selection_fn: base.InteriorActionSelectionFn,
    num_simulations: int,
    max_depth: Optional[int] = None,
    invalid_actions: Optional[chex.Array] = None,
    extra_data: Any = None,
    loop_fn: base.LoopFn = jax.lax.fori_loop) -> Tree:
  """Performs a full search and returns sampled actions.

  In the shape descriptions, `B` denotes the batch dimension.

  Args:
    params: params to be forwarded to root and recurrent functions.
    rng_key: random number generator state, the key is consumed.
    root: a `(prior_logits, value, embedding)` `RootFnOutput`. The
      `prior_logits` are from a policy network. The shapes are
      `([B, num_actions], [B], [B, ...])`, respectively.
    recurrent_fn: a callable to be called on the leaf nodes and unvisited
      actions retrieved by the simulation step, which takes as args
      `(params, rng_key, action, embedding)` and returns a `RecurrentFnOutput`
      and the new state embedding. The `rng_key` argument is consumed.
    root_action_selection_fn: function used to select an action at the root.
    interior_action_selection_fn: function used to select an action during
      simulation.
    num_simulations: the number of simulations.
    max_depth: maximum search tree depth allowed during simulation, defined as
      the number of edges from the root to a leaf node.
    invalid_actions: a mask with invalid actions at the root. In the
      mask, invalid actions have ones, and valid actions have zeros.
      Shape `[B, num_actions]`.
    extra_data: extra data passed to `tree.extra_data`. Shape `[B, ...]`.
    loop_fn: Function used to run the simulations. It may be required to pass
      hk.fori_loop if using this function inside a Haiku module.

  Returns:
    `SearchResults` containing outcomes of the search, e.g. `visit_counts`
    `[B, num_actions]`.
  """
  action_selection_fn = action_selection.switching_action_selection_wrapper(
      root_action_selection_fn=root_action_selection_fn,
      interior_action_selection_fn=interior_action_selection_fn
  )

  # Do simulation, expansion, and backward steps.
  batch_size = root.value.shape[0]
  batch_range = jnp.arange(batch_size)
  if max_depth is None:
    max_depth = num_simulations
  if invalid_actions is None:
    invalid_actions = jnp.zeros_like(root.prior_logits)

  def body_fun(sim, loop_state):
    # jax.debug.print("sim {sim} / {total}", sim=sim, total=num_simulations)
    rng_key, tree = loop_state
    rng_key, simulate_key, expand_key = jax.random.split(rng_key, 3)
    # simulate is vmapped and expects batched rng keys.
    simulate_keys = jax.random.split(simulate_key, batch_size)
    parent_index, action = simulate(
        simulate_keys, tree, action_selection_fn, max_depth)
    # A node first expanded on simulation `i`, will have node index `i`.
    # Node 0 corresponds to the root node.
    next_node_index = tree.children_index[batch_range, parent_index, action]
    next_node_index = jnp.where(next_node_index == Tree.UNVISITED,
                                sim + 1, next_node_index)
  
    tree = expand(
        params, expand_key, tree, recurrent_fn, parent_index,
        action, next_node_index)
    tree = backward(tree, next_node_index)
    # jax.debug.print("next node index {next_node_index}", next_node_index=next_node_index)
    loop_state = rng_key, tree
    return loop_state

  # Allocate all necessary storage.
  tree = instantiate_tree_from_root(root, num_simulations,
                                    root_invalid_actions=invalid_actions,
                                    extra_data=extra_data)
  _, tree = loop_fn(
      0, num_simulations, body_fun, (rng_key, tree))

  return tree


class _SimulationState(NamedTuple):
  """The state for the simulation while loop."""
  rng_key: chex.PRNGKey
  node_index: int
  action: int
  next_node_index: int
  depth: int
  is_continuing: bool


@functools.partial(jax.vmap, in_axes=[0, 0, None, None, None,], out_axes=0)
def parallel_sampling_simulate(
    rng_key: chex.PRNGKey,
    tree: Tree,
    num_choices: int,
    action_selection_fn: base.InteriorActionSelectionFn,
    max_depth: int) -> Tuple[chex.Array, chex.Array]:
  """Traverses the tree until reaching an unvisited action or `max_depth`.

  Each simulation starts from the root and keeps selecting actions traversing
  the tree until a leaf or `max_depth` is reached.

  Args:
    rng_key: random number generator state, the key is consumed.
    tree: _unbatched_ MCTS tree state.
    action_selection_fn: function used to select an action during simulation.
    max_depth: maximum search tree depth allowed during simulation.

  Returns:
    `(parent_index, action)` tuple, where `parent_index` is the index of the
    node reached at the end of the simulation, and the `action` is the action to
    evaluate from the `parent_index`.
  """
  def cond_fun(state):
    return jnp.any(state.is_continuing)


  def body_fun(state):
    # Preparing the next simulation state.
    node_index = jnp.where(
        state.is_continuing, state.next_node_index, state.node_index
    )
    # node_index = state.next_node_index
    # jax.debug.print("node_index {node_index}", node_index=node_index)
    rng_key, action_selection_key = jnp.hsplit(jax.vmap(jax.random.split)(state.rng_key), 2)
    rng_key, action_selection_key = jnp.squeeze(rng_key, axis=1), jnp.squeeze(action_selection_key, axis=1)
    # rng_key, action_selection_key = jax.vmap(jax.random.split)(state.rng_key)
    # rng_key, action_selection_key = jax.random.split(state.rng_key)
    # print(state.depth)
    action = action_selection_fn(action_selection_key, tree, node_index,
                                 depth=state.depth)
    next_node_index = tree.children_index[node_index, action]

    # The returned action will be visited.
    depth = state.depth + 1
    is_before_depth_cutoff = depth < max_depth
    is_visited = next_node_index != Tree.UNVISITED
    is_continuing = jnp.logical_and(is_visited, is_before_depth_cutoff)
    return _SimulationState(  # pytype: disable=wrong-arg-types  # jax-types
        rng_key=rng_key,
        node_index=node_index,
        action=action,
        next_node_index=next_node_index,
        depth=depth,
        is_continuing=is_continuing)

  node_index = jnp.array(Tree.ROOT_INDEX, dtype=jnp.int32)
  depth = jnp.zeros((), dtype=tree.children_prior_logits.dtype)
  # pytype: disable=wrong-arg-types  # jnp-type
  # initial_state = _SimulationState(
  #     rng_key=rng_key,
  #     node_index=tree.NO_PARENT,
  #     action=tree.NO_PARENT,
  #     next_node_index=node_index,
  #     depth=depth,
  #     is_continuing=jnp.array(True))
  # repeat for num_choices times
  initial_state = _SimulationState(
      rng_key=jax.random.split(rng_key, num_choices),
      node_index=jnp.repeat(jnp.expand_dims(tree.NO_PARENT, axis=0), num_choices),
      action=jnp.repeat(jnp.expand_dims(tree.NO_PARENT, axis=0), num_choices),
      next_node_index=jnp.repeat(jnp.expand_dims(node_index, axis=0), num_choices),
      depth=jnp.repeat(jnp.expand_dims(depth, axis=0), num_choices),
      is_continuing=jnp.repeat(jnp.expand_dims(jnp.array(True), axis=0), num_choices)
  )
  
  # pytype: enable=wrong-arg-types
  end_state = jax.lax.while_loop(cond_fun, body_fun, initial_state)
  # jax.debug.print("end_state {end_state}", end_state=end_state.node_index)
  # Returning a node with a selected action.
  # The action can be already visited, if the max_depth is reached.
  return end_state.node_index, end_state.action


@functools.partial(jax.vmap, in_axes=[0, 0, None, None], out_axes=0)
def simulate(
    rng_key: chex.PRNGKey,
    tree: Tree,
    action_selection_fn: base.InteriorActionSelectionFn,
    max_depth: int) -> Tuple[chex.Array, chex.Array]:
  """Traverses the tree until reaching an unvisited action or `max_depth`.

  Each simulation starts from the root and keeps selecting actions traversing
  the tree until a leaf or `max_depth` is reached.

  Args:
    rng_key: random number generator state, the key is consumed.
    tree: _unbatched_ MCTS tree state.
    action_selection_fn: function used to select an action during simulation.
    max_depth: maximum search tree depth allowed during simulation.

  Returns:
    `(parent_index, action)` tuple, where `parent_index` is the index of the
    node reached at the end of the simulation, and the `action` is the action to
    evaluate from the `parent_index`.
  """
  def cond_fun(state):
    return state.is_continuing

  def body_fun(state):
    # Preparing the next simulation state.
    node_index = state.next_node_index
    rng_key, action_selection_key = jax.random.split(state.rng_key)
    action = action_selection_fn(action_selection_key, tree, node_index,
                                 state.depth)
    next_node_index = tree.children_index[node_index, action]
    # The returned action will be visited.
    depth = state.depth + 1
    is_before_depth_cutoff = depth < max_depth
    is_visited = next_node_index != Tree.UNVISITED
    is_continuing = jnp.logical_and(is_visited, is_before_depth_cutoff)
    return _SimulationState(  # pytype: disable=wrong-arg-types  # jax-types
        rng_key=rng_key,
        node_index=node_index,
        action=action,
        next_node_index=next_node_index,
        depth=depth,
        is_continuing=is_continuing)

  node_index = jnp.array(Tree.ROOT_INDEX, dtype=jnp.int32)
  depth = jnp.zeros((), dtype=tree.children_prior_logits.dtype)
  # pytype: disable=wrong-arg-types  # jnp-type
  initial_state = _SimulationState(
      rng_key=rng_key,
      node_index=tree.NO_PARENT,
      action=tree.NO_PARENT,
      next_node_index=node_index,
      depth=depth,
      is_continuing=jnp.array(True))
  # pytype: enable=wrong-arg-types
  end_state = jax.lax.while_loop(cond_fun, body_fun, initial_state)

  # Returning a node with a selected action.
  # The action can be already visited, if the max_depth is reached.
  return end_state.node_index, end_state.action


def parallel_find_unique(value, node_index, num_samples):
  unique_flat_indices, unique_indices = jnp.unique(
            value, axis=0, return_inverse=True, size=num_samples, fill_value=-1
        )

def parallel_segment_average(value, node_index, num_nodes, batch_size, num_samples):
  # print(value.shape)
  step_values = jax.vmap(jax.ops.segment_sum, in_axes=(0, 0, None), out_axes=0)(
      value, node_index, num_nodes)
  # print(step_values.shape)
  index_nums = parallel_segment_counting(value, node_index, num_nodes)
  step_values = step_values / (index_nums + 1e-6) # [B, K], K = num_nodes, N = num_samples
  # averaged_values = jnp.zeros_like(value) # [B, N], value -> step_values[batch_range, node_index]
  averaged_values_in_original_shape = jax.vmap(
        lambda av, ni: av[ni], in_axes=(0, 0)
    )(step_values, node_index)
  
  return averaged_values_in_original_shape

def parallel_segment_counting(value, node_index, num_nodes):
  index_nums = jax.vmap(jax.ops.segment_sum, in_axes=(0, 0, None), out_axes=0)(
      jnp.ones_like(value), node_index, num_nodes)
  return index_nums

@functools.partial(jax.jit, static_argnames=["recurrent_fn", "num_choices"])
def parallel_expand(
    params: chex.Array,
    rng_key: chex.PRNGKey,
    tree: Tree[T],
    recurrent_fn: base.RecurrentFn,
    parent_index: chex.Array,
    action: chex.Array,
    next_node_index: chex.Array,
    num_choices: int) -> Tree[T]:
  """Create and evaluate child nodes from given nodes and unvisited actions.

  Args:
    params: params to be forwarded to recurrent function.
    rng_key: random number generator state.
    tree: the MCTS tree state to update.
    recurrent_fn: a callable to be called on the leaf nodes and unvisited
      actions retrieved by the simulation step, which takes as args
      `(params, rng_key, action, embedding)` and returns a `RecurrentFnOutput`
      and the new state embedding. The `rng_key` argument is consumed.
    parent_index: the index of the parent node, from which the action will be
      expanded. Shape `[B]`.
    action: the action to expand. Shape `[B]`.
    next_node_index: the index of the newly expanded node. This can be the index
      of an existing node, if `max_depth` is reached. Shape `[B]`.

  Returns:
    tree: updated MCTS tree state.
  """
  batch_size = tree_lib.infer_batch_size(tree)
  batch_range = jnp.arange(batch_size)
  # chex.assert_shape([parent_index, action, next_node_index], (batch_size, num_choices))
  # Retrieve states for nodes to be evaluated.
  # embedding = jax.tree.map(
  #     lambda x: x[batch_range, parent_index], tree.embeddings)
  # jax.debug.print("end states {end_state}", end_state=tree.embeddings.end_state)
  func = functools.partial(
    jax.vmap(lambda y, x: x[batch_range, y], 
             in_axes=(1, None), out_axes=1), parent_index)
  embedding = jax.tree.map(
          func,
          tree.embeddings)
  # Evaluate and create a new node.
  rng_keys = jax.random.split(rng_key, batch_size)
  # split rng_keys to [B, K, 1]
  rng_keys = jax.vmap(jax.random.split, in_axes=(0, None), out_axes=0)(rng_keys, num_choices)
  # jax.debug.print("end states {end_state}", end_state=embedding.end_state)
  step, embedding = jax.vmap(recurrent_fn, (None, 1, 1, 1), out_axes=1)(
      params, rng_keys, action, embedding)
  # chex.assert_shape(step.prior_logits, [batch_size, num_choices, tree.num_actions])
  # chex.assert_shape(step.reward, [batch_size, num_choices])
  # chex.assert_shape(step.discount, [batch_size, num_choices])
  # chex.assert_shape(step.value, [batch_size, num_choices])
  
  # step_values = jax.vmap(jax.ops.segment_sum, in_axes=(0, 0, None), out_axes=0)(
  #     step.value, next_node_index, tree.node_values.shape[1])
  # index_nums = jax.vmap(jax.ops.segment_sum, in_axes=(0, 0, None), out_axes=0)(
  #     jnp.ones_like(step.value), next_node_index, tree.node_values.shape[1])
  step_values = parallel_segment_average(step.value, next_node_index, tree.node_values.shape[1], batch_size, num_choices)
  def loop_body(index, inputs):
    tree, step_values, next_node_index = inputs
    tree = update_tree_node(tree, next_node_index.at[:, index].get(), 
                            step.prior_logits.at[:, index].get(), step_values.at[:, index].get(), 
                            jax.tree.map(lambda x: x.at[:, index].get(), embedding))
    # jax.debug.print("info {index}", index=embedding.end_state)
    return tree, step_values, next_node_index

  tree, _, _ = jax.lax.fori_loop(0, num_choices, loop_body, (tree, step_values, next_node_index))

  def loop_replace(index, inputs):
    tree, next_node_index_, parent_index_, action_, step_ = inputs
    next_node_index = next_node_index_.at[:, index].get()
    parent_index = parent_index_.at[:, index].get()
    action = action_.at[:, index].get()
    step_reward = step_.reward.at[:, index].get()
    step_discount = step_.discount.at[:, index].get()
    tree = tree.replace(
      children_index=batch_update(
          tree.children_index, next_node_index, parent_index, action),
      children_rewards=batch_update(
          tree.children_rewards, step_reward, parent_index, action),
      children_discounts=batch_update(
          tree.children_discounts, step_discount, parent_index, action),
      parents=batch_update(tree.parents, parent_index, next_node_index),
      action_from_parent=batch_update(
          tree.action_from_parent, action, next_node_index))
    return tree, next_node_index_, parent_index_, action_, step_

  tree, _, _, _, _ = jax.lax.fori_loop(0, num_choices, loop_replace, (tree, next_node_index, parent_index, action, step))
  # jax.debug.print("expanded tree") 
  return tree
  # tree = tree.replace(
  #     children_index=batch_update(
  #         tree.children_index, next_node_index, parent_index, action),
  #     children_rewards=batch_update(
  #         tree.children_rewards, step.reward, parent_index, action),
  #     children_discounts=batch_update(
  #         tree.children_discounts, step.discount, parent_index, action),
  #     parents=batch_update(tree.parents, parent_index, next_node_index),
  #     action_from_parent=batch_update(
  #         tree.action_from_parent, action, next_node_index))

def expand(
    params: chex.Array,
    rng_key: chex.PRNGKey,
    tree: Tree[T],
    recurrent_fn: base.RecurrentFn,
    parent_index: chex.Array,
    action: chex.Array,
    next_node_index: chex.Array) -> Tree[T]:
  """Create and evaluate child nodes from given nodes and unvisited actions.

  Args:
    params: params to be forwarded to recurrent function.
    rng_key: random number generator state.
    tree: the MCTS tree state to update.
    recurrent_fn: a callable to be called on the leaf nodes and unvisited
      actions retrieved by the simulation step, which takes as args
      `(params, rng_key, action, embedding)` and returns a `RecurrentFnOutput`
      and the new state embedding. The `rng_key` argument is consumed.
    parent_index: the index of the parent node, from which the action will be
      expanded. Shape `[B]`.
    action: the action to expand. Shape `[B]`.
    next_node_index: the index of the newly expanded node. This can be the index
      of an existing node, if `max_depth` is reached. Shape `[B]`.

  Returns:
    tree: updated MCTS tree state.
  """
  batch_size = tree_lib.infer_batch_size(tree)
  batch_range = jnp.arange(batch_size)
  chex.assert_shape([parent_index, action, next_node_index], (batch_size,))

  # Retrieve states for nodes to be evaluated.
  embedding = jax.tree.map(
      lambda x: x[batch_range, parent_index], tree.embeddings)

  # Evaluate and create a new node.
  rng_keys = jax.random.split(rng_key, batch_size)
  # step, embedding = jax.vmap(recurrent_fn, (None, 0, None, None), out_axes=0)(
  #     params, rng_keys, action, embedding)
  step, embedding = recurrent_fn(params, rng_keys, action, embedding)
  chex.assert_shape(step.prior_logits, [batch_size, tree.num_actions])
  chex.assert_shape(step.reward, [batch_size])
  chex.assert_shape(step.discount, [batch_size])
  chex.assert_shape(step.value, [batch_size])
  tree = update_tree_node(
      tree, next_node_index, step.prior_logits, step.value, embedding)

  # Return updated tree topology.
  return tree.replace(
      children_index=batch_update(
          tree.children_index, next_node_index, parent_index, action),
      children_rewards=batch_update(
          tree.children_rewards, step.reward, parent_index, action),
      children_discounts=batch_update(
          tree.children_discounts, step.discount, parent_index, action),
      parents=batch_update(tree.parents, parent_index, next_node_index),
      action_from_parent=batch_update(
          tree.action_from_parent, action, next_node_index))


@functools.partial(jax.vmap, in_axes=[0, 0, None], out_axes=0)
def maximum_entropy_backward(
    tree: Tree[T],
    leaf_index: chex.Numeric,
    tau: chex.Numeric = 1.0) -> Tree[T]:
  """Goes up and updates the tree until all nodes reached the root.

  Args:
    tree: the MCTS tree state to update, without the batch size.
    leaf_index: the node index from which to do the backward.

  Returns:
    Updated MCTS tree state.
  """
  def cond_fun(loop_state):
    _, _, index = loop_state
    return index != Tree.ROOT_INDEX

  def body_fun(loop_state):
    # Here we update the value of our parent, so we start by reversing.
    tree, leaf_value, index = loop_state
    parent = tree.parents[index]
    count = tree.node_visits[parent] # N(s)
    action = tree.action_from_parent[index] # a
    reward = tree.children_rewards[parent, action] # R(s, a)
    leaf_value = reward + tree.children_discounts[parent, action] * leaf_value  # R(s, a) + gamma * V(s')
    children_values = tree.node_values[index]
      
    parent_value = jax.lax.cond(
      count == 0,
      lambda _: leaf_value,
      lambda _: tau * jnp.log(jnp.sum(jnp.exp(children_values / tau))),
      operand=None
    )  
    
    children_counts = tree.children_visits[parent, action] + 1

    tree = tree.replace(
        node_values=update(tree.node_values, parent_value, parent),
        node_visits=update(tree.node_visits, count + 1, parent),
        children_values=update(
            tree.children_values, children_values, parent, action),
        children_visits=update(
            tree.children_visits, children_counts, parent, action))

    return tree, leaf_value, parent

  leaf_index = jnp.asarray(leaf_index, dtype=jnp.int32)
  loop_state = (tree, tree.node_values[leaf_index], leaf_index)
  tree, _, _ = jax.lax.while_loop(cond_fun, body_fun, loop_state)

  return tree


@functools.partial(jax.vmap, in_axes=[0, 0, None, None], out_axes=0)
def pimct_backward_parallel(
    tree: Tree[T],
    leaf_index: chex.Numeric,
    c_param: chex.Numeric = 1.414, 
    qtransform: base.QTransform = qtransforms.qtransform_by_parent_and_siblings,
    ) -> Tree[T]:
  """Goes up and updates the tree until all nodes reached the root.

  Args:
    tree: the MCTS tree state to update, without the batch size.
    leaf_index: the node index from which to do the backward.

  Returns:
    Updated MCTS tree state.
  """
  def cond_fun(loop_state):
    _, _, index = loop_state
    return index != Tree.ROOT_INDEX

  def body_fun(loop_state):
    # Here we update the value of our parent, so we start by reversing.
    tree, leaf_value, index = loop_state
    num_choices = leaf_value.shape[0]
    # preprocess: merge the identical indexes, average the values according to the indexes
    # leaf_value: [n, 1], index: [n, 1]
    # leaf_value = jax.jit(jax.ops.segment_sum, static_argnums=2)(leaf_value, index, num_choices)
    # index_nums = jax.jit(jax.ops.segment_sum, static_argnums=2)(jnp.ones_like(leaf_value), index, num_choices)
    # leaf_value = leaf_value / (index_nums + 1e-6)
    
    parent = tree.parents[index]
    count = tree.node_visits[parent] # N(s)
    action = tree.action_from_parent[index] # a
    reward = tree.children_rewards[parent, action] # R(s, a)
    leaf_value = reward + tree.children_discounts[parent, action] * leaf_value  # R(s, a) + gamma * V(s')
    child_count = tree.children_visits[parent, action] + 1
    child_value = tree.node_values[index]
    
    tree = tree.replace(
        children_values=update(
            tree.children_values, child_value, parent, action),
        children_visits=update(
            tree.children_visits, child_count, parent, action))

    children_values = tree.children_values[parent]
    # children_values = qtransform(tree, parent, min_value=0, max_value=1)
    policy_weights = action_selection.compute_pikl_weights(
      children_values, count, tree.num_actions, c_param, jnp.ones_like(children_values) / tree.num_actions)

    parent_value = count * jnp.dot(policy_weights, children_values) / (count + 1) \
      + tree.raw_values[parent] / (count + 1)

    tree = tree.replace(
        node_values=update(tree.node_values, parent_value, parent),
        node_visits=update(tree.node_visits, count + 1, parent),
    )

    return tree, leaf_value, parent

  leaf_index = jnp.asarray(leaf_index, dtype=jnp.int32)
  loop_state = (tree, tree.node_values[leaf_index], leaf_index)
  tree, _, _ = jax.lax.while_loop(cond_fun, body_fun, loop_state)
  # jax.debug.print("pimct_backward_parallel completed")
  return tree


@functools.partial(jax.vmap, in_axes=[0, 0, None, None], out_axes=0)
def pimct_backward_(
    tree: Tree[T],
    leaf_index: chex.Numeric,
    c_param: chex.Numeric = 1.414,
    qtransform: base.QTransform = qtransforms.qtransform_by_parent_and_siblings,
    ) -> Tree[T]:
  """Goes up and updates the tree until all nodes reached the root.

  Args:
    tree: the MCTS tree state to update, without the batch size.
    leaf_index: the node index from which to do the backward.

  Returns:
    Updated MCTS tree state.
  """
  def cond_fun(loop_state):
    _, _, index = loop_state
    return index != Tree.ROOT_INDEX

  def body_fun(loop_state):
    # Here we update the value of our parent, so we start by reversing.
    tree, leaf_value, index = loop_state
    parent = tree.parents[index]
    count = tree.node_visits[parent] # N(s)
    action = tree.action_from_parent[index] # a
    reward = tree.children_rewards[parent, action] # R(s, a)
    leaf_value = reward + tree.children_discounts[parent, action] * leaf_value  # R(s, a) + gamma * V(s')
    child_count = tree.children_visits[parent, action] + 1
    child_value = tree.node_values[index]
    prior_logits = tree.children_prior_logits[parent]
    prior_probs = jax.nn.softmax(prior_logits)
    tree = tree.replace(
        children_values=update(
            tree.children_values, child_value, parent, action),
        children_visits=update(
            tree.children_visits, child_count, parent, action))
    # pb_c = pb_c_init + jnp.log((count + pb_c_base + 1.) / pb_c_base)
    # print(children_values)
    # children_values = qtransform(tree, parent, min_value=0, max_value=1)
    children_values = tree.children_values[parent]
    policy_weights = action_selection.compute_pikl_weights(
      children_values, count, tree.num_actions, c_param, jnp.ones_like(children_values) / tree.num_actions)
    # to_print = children_values * tree.children_discounts[parent]
    # jax.debug.print("qvalue shape: {info}", info= to_print.shape)
    # jax.debug.print("discount shape: {info}", info=tree.children_discounts[parent])
    # values = tree.children_rewards[parent] + tree.children_discounts[parent] * children_values
    
    policy_weights = jnp.where(tree.children_visits[parent] > 0, policy_weights, jnp.zeros_like(policy_weights))
    policy_weights = policy_weights / jnp.sum(policy_weights, axis=-1)
    parent_value = count * jnp.dot(policy_weights,  \
      tree.children_rewards[parent] + tree.children_discounts[parent] * children_values) / (count + 1) \
      + tree.raw_values[parent]  / (count + 1)
    # parent_value = reward + parent_value
    # parent_value = jnp.dot(policy_weights, children_values)

    tree = tree.replace(
        node_values=update(tree.node_values, parent_value, parent),
        node_visits=update(tree.node_visits, count + 1, parent),
        # children_values=update(
        #     tree.children_values, children_values, parent, action),
        # children_visits=update(
        #     tree.children_visits, children_counts, parent, action)
            )

    return tree, leaf_value, parent

  leaf_index = jnp.asarray(leaf_index, dtype=jnp.int32)
  loop_state = (tree, tree.node_values[leaf_index], leaf_index)
  tree, _, _ = jax.lax.while_loop(cond_fun, body_fun, loop_state)

  return tree


@functools.partial(jax.vmap, in_axes=[0, 0, None, None, None], out_axes=0)
def pimct_backward(
    tree: Tree[T],
    leaf_index: chex.Numeric,
    pb_c_init: chex.Numeric = 1.25,
    pb_c_base: chex.Numeric = 19652.0,
    qtransform: base.QTransform = qtransforms.qtransform_by_parent_and_siblings,
    ) -> Tree[T]:
  """Goes up and updates the tree until all nodes reached the root.

  Args:
    tree: the MCTS tree state to update, without the batch size.
    leaf_index: the node index from which to do the backward.

  Returns:
    Updated MCTS tree state.
  """
  def cond_fun(loop_state):
    _, _, index = loop_state
    return index != Tree.ROOT_INDEX

  def body_fun(loop_state):
    # Here we update the value of our parent, so we start by reversing.
    tree, leaf_value, index = loop_state
    parent = tree.parents[index]
    count = tree.node_visits[parent] # N(s)
    action = tree.action_from_parent[index] # a
    reward = tree.children_rewards[parent, action] # R(s, a)
    leaf_value = reward + tree.children_discounts[parent, action] * leaf_value  # R(s, a) + gamma * V(s')
    child_count = tree.children_visits[parent, action] + 1
    child_value = tree.node_values[index]
    prior_logits = tree.children_prior_logits[parent]
    prior_probs = jax.nn.softmax(prior_logits)

    tree = tree.replace(
        children_values=update(
            tree.children_values, child_value, parent, action),
        children_visits=update(
            tree.children_visits, child_count, parent, action))
    pb_c = pb_c_init + jnp.log((count + pb_c_base + 1.) / pb_c_base)
    
    policy_weights = action_selection.compute_pikl_puct_weights(
      qtransform(tree, parent), prior_probs, count + 1, tree.num_actions, pb_c)
    visit_counts = tree.children_visits[parent]
    # policy_weights = prior_probs
    # policy_weights = visit_counts
    # policy_weights = jnp.where(visit_counts > 0, 1.0, 0.0)
    policy_weights = jnp.where(visit_counts > 0, policy_weights, jnp.zeros_like(policy_weights))
    # policy_weights = policy_weights / jnp.where(visit_counts> 0, jnp.sum(policy_weights, axis=-1), 1.0)
  
    # jax.debug.print("policy_weights {policy_weights}", policy_weights=policy_weights)
    qvalues = tree.qvalues(parent)
    # weighted_q = jnp.dot(policy_weights, tree.qvalues(parent))
    sum_probs = jnp.sum(policy_weights, axis=-1)
    policy_weights = policy_weights / (sum_probs + 1e-6)
    weighted_q = jnp.sum(jnp.where(
      visit_counts > 0,
      policy_weights * qvalues,
      0.0), axis=-1)
    parent_value = count * weighted_q / (count + 1.0) \
      + tree.raw_values[parent]  / (count + 1.0)
    # parent_value = count * jnp.dot(policy_weights, tree.qvalues(parent)) / (count + 1.0) \
    #   + tree.raw_values[parent]  / (count + 1.0)
    # jax.debug.print("parent_value {parent_value}", parent_value=parent_value)
    # parent_value = count * jnp.dot(policy_weights,  \
    #   tree.children_rewards[parent] + tree.children_discounts[parent] * children_values) / (count + 1) \
    #   + tree.raw_values[parent]  / (count + 1)


    tree = tree.replace(
        node_values=update(tree.node_values, parent_value, parent),
        node_visits=update(tree.node_visits, count + 1, parent),
        )

    return tree, leaf_value, parent

  leaf_index = jnp.asarray(leaf_index, dtype=jnp.int32)
  loop_state = (tree, tree.node_values[leaf_index], leaf_index)
  tree, _, _ = jax.lax.while_loop(cond_fun, body_fun, loop_state)

  return tree

@jax.vmap
def backward(
    tree: Tree[T],
    leaf_index: chex.Numeric) -> Tree[T]:
  """Goes up and updates the tree until all nodes reached the root.

  Args:
    tree: the MCTS tree state to update, without the batch size.
    leaf_index: the node index from which to do the backward.

  Returns:
    Updated MCTS tree state.
  """

  def cond_fun(loop_state):
    _, _, index = loop_state
    return index != Tree.ROOT_INDEX

  def body_fun(loop_state):
    # Here we update the value of our parent, so we start by reversing.
    tree, leaf_value, index = loop_state
    parent = tree.parents[index]
    count = tree.node_visits[parent]
    action = tree.action_from_parent[index]
    reward = tree.children_rewards[parent, action]
    leaf_value = reward + tree.children_discounts[parent, action] * leaf_value
    parent_value = (
        tree.node_values[parent] * count + leaf_value) / (count + 1.0)
    children_values = tree.node_values[index]
    children_counts = tree.children_visits[parent, action] + 1

    tree = tree.replace(
        node_values=update(tree.node_values, parent_value, parent),
        node_visits=update(tree.node_visits, count + 1, parent),
        children_values=update(
            tree.children_values, children_values, parent, action),
        children_visits=update(
            tree.children_visits, children_counts, parent, action))

    return tree, leaf_value, parent

  leaf_index = jnp.asarray(leaf_index, dtype=jnp.int32)
  loop_state = (tree, tree.node_values[leaf_index], leaf_index)
  tree, _, _ = jax.lax.while_loop(cond_fun, body_fun, loop_state)

  return tree


# Utility function to set the values of certain indices to prescribed values.
# This is vmapped to operate seamlessly on batches.
def update(x, vals, *indices):
  return x.at[indices].set(vals)


batch_parallel_update = jax.vmap(update, in_axes=(0, (0, 1), (0, 1), (0, 1)), out_axes=0)
batch_update = jax.vmap(update)


def merge_parallel_tree_nodes(
    tree: Tree[T],
) -> Tree[T]:
  """
  Merge the identical indexes, average the values according to the indexes, 
  sum the visits. The rewards in indentical indexes are the same. 
  """

def update_tree_node_parallel(
    tree: Tree[T],
    node_index: chex.Array,
    prior_logits: chex.Array,
    value: chex.Array,
    embedding: chex.Array) -> Tree[T]:
  """Updates the tree at node index.

  Args:
    tree: `Tree` to whose node is to be updated.
    node_index: the index of the expanded node. Shape `[B]`.
    prior_logits: the prior logits to fill in for the new node, of shape
      `[B, num_actions]`.
    value: the value to fill in for the new node. Shape `[B]`.
    embedding: the state embeddings for the node. Shape `[B, ...]`.

  Returns:
    The new tree with updated nodes.
  """
  batch_size = tree_lib.infer_batch_size(tree)
  batch_range = jnp.arange(batch_size)
  chex.assert_shape(prior_logits, (batch_size, tree.num_actions))

  # When using max_depth, a leaf can be expanded multiple times.
  new_visit = tree.node_visits[batch_range, node_index] + 1
  updates = dict(  # pylint: disable=use-dict-literal
      children_prior_logits=batch_update(
          tree.children_prior_logits, prior_logits, node_index), # has duplicated prior_logits for the same node
      raw_values=batch_update(
          tree.raw_values, value, node_index), # has duplicated values for the same node
      node_values=batch_update(
          tree.node_values, value, node_index),
      node_visits=batch_update(                 # need accumulate the visits in different samples
          tree.node_visits, new_visit, node_index),
      embeddings=jax.tree.map(                   # no stochasticity in the embeddings, remove duplicates
          lambda t, s: batch_update(t, s, node_index),
          tree.embeddings, embedding))

  return tree.replace(**updates)

@jax.jit
def update_tree_node(
    tree: Tree[T],
    node_index: chex.Array,
    prior_logits: chex.Array,
    value: chex.Array,
    embedding: chex.Array) -> Tree[T]:
  """Updates the tree at node index.

  Args:
    tree: `Tree` to whose node is to be updated.
    node_index: the index of the expanded node. Shape `[B]`.
    prior_logits: the prior logits to fill in for the new node, of shape
      `[B, num_actions]`.
    value: the value to fill in for the new node. Shape `[B]`.
    embedding: the state embeddings for the node. Shape `[B, ...]`.

  Returns:
    The new tree with updated nodes.
  """
  batch_size = tree_lib.infer_batch_size(tree)
  batch_range = jnp.arange(batch_size)
  chex.assert_shape(prior_logits, (batch_size, tree.num_actions))

  # When using max_depth, a leaf can be expanded multiple times.
  new_visit = tree.node_visits[batch_range, node_index] + 1
  updates = dict(  # pylint: disable=use-dict-literal
      children_prior_logits=batch_update(
          tree.children_prior_logits, prior_logits, node_index), # has duplicated prior_logits for the same node
      raw_values=batch_update(
          tree.raw_values, value, node_index), # has duplicated values for the same node
      node_values=batch_update(
          tree.node_values, value, node_index),
      node_visits=batch_update(                 # need accumulate the visits in different samples
          tree.node_visits, new_visit, node_index),
      embeddings=jax.tree.map(                   # no stochasticity in the embeddings, remove duplicates
          lambda t, s: batch_update(t, s, node_index),
          tree.embeddings, embedding))

  return tree.replace(**updates)


def instantiate_tree_from_root(
    root: base.RootFnOutput,
    num_simulations: int,
    root_invalid_actions: chex.Array,
    extra_data: Any) -> Tree:
  """Initializes tree state at search root."""
  chex.assert_rank(root.prior_logits, 2)
  batch_size, num_actions = root.prior_logits.shape
  # print("batch_size: ", batch_size)
  # print("value shape: ", root.value.shape)
  chex.assert_shape(root.value, [batch_size])
  num_nodes = num_simulations + 1
  data_dtype = root.value.dtype
  batch_node = (batch_size, num_nodes)
  batch_node_action = (batch_size, num_nodes, num_actions)

  def _zeros(x):
    return jnp.zeros(batch_node + x.shape[1:], dtype=x.dtype)

  # Create a new empty tree state and fill its root.
  tree = Tree(
      node_visits=jnp.zeros(batch_node, dtype=jnp.int32),
      raw_values=jnp.zeros(batch_node, dtype=data_dtype),
      node_values=jnp.zeros(batch_node, dtype=data_dtype),
      parents=jnp.full(batch_node, Tree.NO_PARENT, dtype=jnp.int32),
      action_from_parent=jnp.full(
          batch_node, Tree.NO_PARENT, dtype=jnp.int32),
      children_index=jnp.full(
          batch_node_action, Tree.UNVISITED, dtype=jnp.int32),
      children_prior_logits=jnp.zeros(
          batch_node_action, dtype=root.prior_logits.dtype),
      children_values=jnp.zeros(batch_node_action, dtype=data_dtype),
      children_visits=jnp.zeros(batch_node_action, dtype=jnp.int32),
      children_rewards=jnp.zeros(batch_node_action, dtype=data_dtype),
      children_discounts=jnp.zeros(batch_node_action, dtype=data_dtype),
      embeddings=jax.tree.map(_zeros, root.embedding),
      root_invalid_actions=root_invalid_actions,
      extra_data=extra_data)

  root_index = jnp.full([batch_size], Tree.ROOT_INDEX)
  tree = update_tree_node(
      tree, root_index, root.prior_logits, root.value, root.embedding)
  return tree
