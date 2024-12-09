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
"""A collection of action selection functions."""
from typing import Optional, TypeVar

import chex
import jax
import jax.numpy as jnp

from mctx._src import base
from mctx._src import qtransforms
from mctx._src import seq_halving
from mctx._src import tree as tree_lib


def switching_action_selection_wrapper_parallel(
    root_action_selection_fn: base.RootActionSelectionFn,
    interior_action_selection_fn: base.InteriorActionSelectionFn
) -> base.InteriorActionSelectionFn:
  """Wraps root and interior action selection fns in a conditional statement."""

  def switching_action_selection_fn_parallel(
      rng_key: chex.PRNGKey,
      tree: tree_lib.Tree,
      node_index: base.NodeIndices,
      depth: base.Depth,
      # num_samples: int,
      *param) -> chex.Array:
    return jnp.where(
        depth == 0,
        root_action_selection_fn(rng_key, tree, node_index, *param),
        interior_action_selection_fn(rng_key, tree, node_index, depth, *param)
    )
  return switching_action_selection_fn_parallel




def switching_action_selection_wrapper(
    root_action_selection_fn: base.RootActionSelectionFn,
    interior_action_selection_fn: base.InteriorActionSelectionFn
) -> base.InteriorActionSelectionFn:
  """Wraps root and interior action selection fns in a conditional statement."""

  def switching_action_selection_fn(
      rng_key: chex.PRNGKey,
      tree: tree_lib.Tree,
      node_index: base.NodeIndices,
      depth: base.Depth) -> chex.Array:
    return jax.lax.cond(
        depth == 0,
        lambda x: root_action_selection_fn(*x[:3]),
        lambda x: interior_action_selection_fn(*x),
        (rng_key, tree, node_index, depth))

  return switching_action_selection_fn



def uct_action_selection(
    rng_key: chex.PRNGKey,
    tree: tree_lib.Tree,
    node_index: chex.Numeric,
    depth: chex.Numeric,
    *,
    c_param: float = 1.414,
    qtransform: base.QTransform = qtransforms.qtransform_by_parent_and_siblings,
) -> chex.Array:
  """Returns the action selected for a node index.

  Args:
    rng_key: random number generator state.
    tree: _unbatched_ MCTS tree state.
    node_index: scalar index of the node from which to select an action.
    depth: the scalar depth of the current node. The root has depth zero.
    pb_c_init: constant c_1 in the PUCT formula.
    pb_c_base: constant c_2 in the PUCT formula.
    qtransform: a monotonic transformation to convert the Q-values to [0, 1].

  Returns:
    action: the action selected from the given node.
  """
  visit_counts = tree.children_visits[node_index]
  node_visit = tree.node_visits[node_index]
  # pb_c = pb_c_init + jnp.log((node_visit + pb_c_base + 1.) / pb_c_base)
  
  # UCT action selection
  policy_score = c_param * jnp.sqrt(jnp.log(node_visit + 1) / (visit_counts + 1e-4))
  chex.assert_shape([node_index, node_visit], ())
  chex.assert_equal_shape([visit_counts, policy_score])
  value_score = qtransform(tree, node_index)

  # Add tiny bit of randomness for tie break
  node_noise_score = 1e-7 * jax.random.uniform(
      rng_key, (tree.num_actions,))
  to_argmax = value_score + policy_score + node_noise_score

  # Masking the invalid actions at the root.
  return masked_argmax(to_argmax, tree.root_invalid_actions * (depth == 0))


@jax.jit
def compute_pikl_puct_weights(q, prior_logits, visits, num_children, c_param):
    lambda_N = c_param * jnp.sqrt(visits) / (visits + 1.0) 
    alpha_min = jnp.max(q + lambda_N * prior_logits)
    alpha_max = jnp.max(q + lambda_N)    

    def cond_fn(val):
        alpha = val
        mid = (alpha[0] + alpha[1]) / 2
        return jnp.logical_and(jnp.abs(jnp.sum(lambda_N * prior_logits / (mid - q), axis=-1) - 1) > 1e-3,
                            alpha[1] - alpha[0] > 1e-3)
    def body_fn(val):
        alpha = val
        mid = (alpha[0] + alpha[1]) / 2
        # alpha = jax.lax.cond(jnp.sum(lambda_N**2 / (num_children * (mid - q)**2)) > 1,
        alpha = jax.lax.cond(jnp.sum(lambda_N * prior_logits / (mid - q), axis=-1) > 1,
                            lambda _: jnp.array([(alpha[0] + alpha[1]) / 2, alpha[1]]),
                            lambda _: jnp.array([alpha[0], (alpha[0] + alpha[1]) / 2]),
                            operand=alpha) 
        return alpha

    init_val = jnp.array([alpha_min, alpha_max])

    alpha = jax.lax.cond(jnp.abs(alpha_max - alpha_min) < 1e-3,
                lambda _: alpha_min / 2 + alpha_max / 2,
                lambda _: jnp.sum(jax.lax.while_loop(cond_fn, body_fn, init_val)) / 2,
                operand=None)
    
    return lambda_N * prior_logits / (alpha - q)

@jax.jit
def compute_pikl_weights(q, visits, num_children, c_param, uniform=None):
    lambda_N = c_param * jnp.sqrt(jnp.log(visits) / (visits + num_children)) 

    alpha_min = jnp.max(q + lambda_N )
    alpha_max = jnp.max(q + lambda_N *jnp.sqrt(num_children) )    

    def cond_fn(val):
        alpha = val
        mid = (alpha[0] + alpha[1]) / 2
        return jnp.logical_and(jnp.abs(jnp.sum(lambda_N**2 / ((mid - q)**2)) - 1) > 1e-3,
                            alpha[1] - alpha[0] > 1e-3)
    def body_fn(val):
        alpha = val
        mid = (alpha[0] + alpha[1]) / 2
        # alpha = jax.lax.cond(jnp.sum(lambda_N**2 / (num_children * (mid - q)**2)) > 1,
        alpha = jax.lax.cond(jnp.sum(lambda_N**2 / ((mid - q)**2)) > 1,
                            lambda _: jnp.array([(alpha[0] + alpha[1]) / 2, alpha[1]]),
                            lambda _: jnp.array([alpha[0], (alpha[0] + alpha[1]) / 2]),
                            operand=alpha) 
        return alpha

    init_val = jnp.array([alpha_min, alpha_max])

    alpha = jax.lax.cond(jnp.abs(alpha_max - alpha_min) < 1e-3,
                lambda _: alpha_min / 2 + alpha_max / 2,
                lambda _: jnp.sum(jax.lax.while_loop(cond_fn, body_fn, init_val)) / 2,
                operand=None)

    choices_weights = jax.lax.cond(visits <= 1,
                lambda _: uniform,
                # lambda _: lambda_N**2 / (num_children * (alpha - q)**2),
                lambda _: lambda_N**2 / ((alpha - q)**2),
                operand=None)
    # print(choices_weights)
    return choices_weights

def softmax_policy(r, tau):
    return tau * jnp.log(jnp.sum(jnp.exp(r / tau)))

def softindmax(r, tau):
    return jnp.exp((r - softmax_policy(r, tau)) / tau)

def maximum_entropy_action_selection(
    rng_key: chex.PRNGKey,
    tree: tree_lib.Tree,
    node_index: chex.Numeric,
    depth: chex.Numeric,
    *,
    tau: float = 1.414,
    epsilon: float = 1e-2,
    qtransform: base.QTransform = qtransforms.qtransform_by_parent_and_siblings,
) -> chex.Array:
  """Returns the action selected for a node index.

  Args:
    rng_key: random number generator state.
    tree: _unbatched_ MCTS tree state.
    node_index: scalar index of the node from which to select an action.
    depth: the scalar depth of the current node. The root has depth zero.
    tau: temperature parameter for the softmax.
    qtransform: a monotonic transformation to convert the Q-values to [0, 1].

  Returns:
    action: the action selected from the given node.
  """
  visit_counts = tree.children_visits[node_index]
  node_visit = tree.node_visits[node_index]
  value = qtransform(tree, node_index)

  lambda_t = epsilon * tree.num_actions / jnp.log(depth + 1)
  policy_weights = (1 - lambda_t) * softindmax(value,  tau) + lambda_t * jnp.ones_like(value) / tree.num_actions
  
  return masked_choice(rng_key, policy_weights, tree.root_invalid_actions * (depth == 0))

@jax.jit
def sample_point_distribution(target_distribution, total_samples, visiting_counts, num_choices = 1):
    # Compute ideal counts for each category
    # print("target_distribution shape {}", target_distribution.shape)
    # print("total_samples shape {}", total_samples.shape)
    # num_choices = jnp.max(jnp.array([num_choices, 1]))
    # print(total_samples.shape)
    # print(num_choices.shape)
    # print(target_distribution.shape)  
    ideal_counts = jnp.dot(total_samples + num_choices, target_distribution)
    # jax.debug.print("ideal counts shape {info}", info=ideal_counts.shape)
    # Compute remaining samples needed
    remaining_samples = jnp.maximum(ideal_counts - visiting_counts, 0)
    
    # Compute adjusted sampling probabilities
    adjusted_probs = remaining_samples / jnp.sum(remaining_samples)
    return adjusted_probs

def count_elements_batched(input_array):
    # Vectorized computation for batched inputs
    def count_row(row):
        return jnp.array([jnp.sum(row == val) for val in row])
    
    return jnp.vstack([count_row(row) for row in input_array])

def count_elements(input_array):
    # Create a boolean mask for each element comparing it to all other elements
    unique_count = jnp.array([jnp.sum(input_array == val) for val in input_array])
    return unique_count

def delta_pikl_puct_action_sampling_parallel(
    rng_key: chex.PRNGKey,
    tree: tree_lib.Tree,
    node_index: chex.Numeric,
    depth: chex.Numeric,
    *,
    pb_c_init: float = 1.25,
    pb_c_base: float = 19652.0,
    num_samples: int = 1,
    qtransform: base.QTransform = qtransforms.qtransform_by_parent_and_siblings,
) -> chex.Array:
  """Returns the action selected for a node index.

  Args:
    rng_key: random number generator state.
    tree: _unbatched_ MCTS tree state.
    node_index: scalar index of the node from which to select an action.
    depth: the scalar depth of the current node. The root has depth zero.
    pb_c_init: constant c_1 in the PUCT formula.
    pb_c_base: constant c_2 in the PUCT formula.
    qtransform: a monotonic transformation to convert the Q-values to [0, 1].

  Returns:
    action: the action selected from the given node.
  """
  visit_counts = tree.children_visits[node_index]
  node_visit = tree.node_visits[node_index]
  # pb_c = pb_c_init + jnp.log((node_visit + pb_c_base + 1.) / pb_c_base)
  # UCT action selection
  pb_c = pb_c_init + jnp.log((node_visit + pb_c_base + 1.) / pb_c_base)
  # policy_score = c_param * jnp.sqrt(jnp.log(node_visit + 1) / (visit_counts + 1e-4))
  prior_logits = tree.children_prior_logits[node_index]
  prior_probs = jax.nn.softmax(prior_logits)
  num_samples = visit_counts.shape[0]
  values = jax.vmap(qtransform, in_axes=[None, 0])(tree, node_index)
  policy_weights = jax.vmap(compute_pikl_puct_weights, in_axes=[0, 0, 0, None, 0,])(values, prior_probs, 
            node_visit, tree.num_actions, pb_c)

  # jax.debug.print("policy weights shape {info}", info=policy_weights.shape)
  # to_sample =  policy_weights
  unique_count = count_elements(node_index) 
  adjust_probs = sample_point_distribution(policy_weights, node_visit, visit_counts, unique_count)
  # to_sample = policy_weights
  # to_sample = adjust_probs
  # node_noise_score = 1e-7 * jax.vmap(jax.random.uniform, in_axes=(0, None))(
  #     rng_key, (tree.num_actions,))
  # to_sample = policy_weights + node_noise_score
  # to_sample = policy_weights 
  to_sample = 0.5 * adjust_probs + 0.5 * policy_weights 
  invalid_actions_root = jnp.repeat(tree.root_invalid_actions[None, :], num_samples, axis=0)
  # mask = jnp.where(depth == 0, invalid_actions_root, jnp.zeros_like(invalid_actions_root))
  mask = invalid_actions_root * jnp.repeat((depth == 0)[:, None], tree.num_actions, axis=1)
  return jax.vmap(masked_choice, in_axes=[0, 0, None, 0])(rng_key, to_sample, tree.num_actions, mask)

def delta_pikl_action_sampling_parallel(
    rng_key: chex.PRNGKey,
    tree: tree_lib.Tree,
    node_index: chex.Numeric,
    depth: chex.Numeric,
    *,
    c_param: float = 1.414,
    num_samples: int = 1,
    qtransform: base.QTransform = qtransforms.qtransform_by_parent_and_siblings,
) -> chex.Array:
  """Returns the action selected for a node index.

  Args:
    rng_key: random number generator state.
    tree: _unbatched_ MCTS tree state.
    node_index: scalar index of the node from which to select an action.
    depth: the scalar depth of the current node. The root has depth zero.
    pb_c_init: constant c_1 in the PUCT formula.
    pb_c_base: constant c_2 in the PUCT formula.
    qtransform: a monotonic transformation to convert the Q-values to [0, 1].

  Returns:
    action: the action selected from the given node.
  """
  visit_counts = tree.children_visits[node_index]
  node_visit = tree.node_visits[node_index]
  # pb_c = pb_c_init + jnp.log((node_visit + pb_c_base + 1.) / pb_c_base)
  # UCT action selection
  # policy_score = c_param * jnp.sqrt(jnp.log(node_visit + 1) / (visit_counts + 1e-4))
  num_samples = visit_counts.shape[0]
  values = jax.vmap(qtransform, in_axes=[None, 0])(tree, node_index)
  policy_weights = jax.vmap(compute_pikl_weights, in_axes=[0, 0, None, None, 0])(values, 
            node_visit, tree.num_actions, c_param, jnp.ones_like(visit_counts) / len(visit_counts))
  # jax.debug.print("policy weights shape {info}", info=policy_weights.shape)
  # to_sample =  policy_weights
  # unique_counts = count_elements(node_index)
  # adjust_probs = sample_point_distribution(policy_weights, node_visit, visit_counts, unique_counts)
  to_sample = policy_weights
  # to_sample = adjust_probs
  # to_sample =  0.5 * adjust_probs + 0.5 * policy_weights
  # jax.debug.print("rng_key shape {info}", info=rng_key.shape)  
  invalid_actions_root = jnp.repeat(tree.root_invalid_actions[None, :], num_samples, axis=0)
  # mask = jnp.where(depth == 0, invalid_actions_root, jnp.zeros_like(invalid_actions_root))
  mask = invalid_actions_root * jnp.repeat((depth == 0)[:, None], 2, axis=1)
  return jax.vmap(masked_choice, in_axes=[0, 0, None, 0])(rng_key, to_sample, tree.num_actions, mask)



def delta_pikl_action_selection(
    rng_key: chex.PRNGKey,
    tree: tree_lib.Tree,
    node_index: chex.Numeric,
    depth: chex.Numeric,
    *,
    c_param: float = 1.414,
    qtransform: base.QTransform = qtransforms.qtransform_by_parent_and_siblings,
) -> chex.Array:
  """Returns the action selected for a node index.

  Args:
    rng_key: random number generator state.
    tree: _unbatched_ MCTS tree state.
    node_index: scalar index of the node from which to select an action.
    depth: the scalar depth of the current node. The root has depth zero.
    pb_c_init: constant c_1 in the PUCT formula.
    pb_c_base: constant c_2 in the PUCT formula.
    qtransform: a monotonic transformation to convert the Q-values to [0, 1].

  Returns:
    action: the action selected from the given node.
  """
  visit_counts = tree.children_visits[node_index]
  node_visit = tree.node_visits[node_index]
  # pb_c = pb_c_init + jnp.log((node_visit + pb_c_base + 1.) / pb_c_base)
  # UCT action selection
  # policy_score = c_param * jnp.sqrt(jnp.log(node_visit + 1) / (visit_counts + 1e-4))
  # values = jax.vmap(qtransform, in_axes=[None, 0])(tree, node_index)
  policy_weights = compute_pikl_weights(qtransform(tree, node_index), 
            node_visit, tree.num_actions, c_param, jnp.ones_like(visit_counts) / len(visit_counts))

  adjust_probs = sample_point_distribution(policy_weights, node_visit, visit_counts)
  # to_sample = policy_weights
  # to_sample = adjust_probs
  to_sample =  0.5 * adjust_probs + 0.5 * policy_weights
  return masked_choice(rng_key, to_sample, tree.num_actions, tree.root_invalid_actions * (depth == 0))





def pikl_action_selection(
    rng_key: chex.PRNGKey,
    tree: tree_lib.Tree,
    node_index: chex.Numeric,
    depth: chex.Numeric,
    *,
    c_param: float = 1.414,
    qtransform: base.QTransform = qtransforms.qtransform_by_parent_and_siblings,
) -> chex.Array:
  """Returns the action selected for a node index.

  Args:
    rng_key: random number generator state.
    tree: _unbatched_ MCTS tree state.
    node_index: scalar index of the node from which to select an action.
    depth: the scalar depth of the current node. The root has depth zero.
    pb_c_init: constant c_1 in the PUCT formula.
    pb_c_base: constant c_2 in the PUCT formula.
    qtransform: a monotonic transformation to convert the Q-values to [0, 1].

  Returns:
    action: the action selected from the given node.
  """
  visit_counts = tree.children_visits[node_index]
  node_visit = tree.node_visits[node_index]
  # pb_c = pb_c_init + jnp.log((node_visit + pb_c_base + 1.) / pb_c_base)
  # UCT action selection
  # policy_score = c_param * jnp.sqrt(jnp.log(node_visit + 1) / (visit_counts + 1e-4))
  value = qtransform(tree, node_index)
  
  policy_weights = compute_pikl_weights(value, 
            node_visit, tree.num_actions, c_param, jnp.ones_like(visit_counts) / len(visit_counts))

  return masked_choice(rng_key, policy_weights, tree.num_actions, tree.root_invalid_actions * (depth == 0))


def muzero_action_selection(
    rng_key: chex.PRNGKey,
    tree: tree_lib.Tree,
    node_index: chex.Numeric,
    depth: chex.Numeric,
    *,
    pb_c_init: float = 1.25,
    pb_c_base: float = 19652.0,
    qtransform: base.QTransform = qtransforms.qtransform_by_parent_and_siblings,
) -> chex.Array:
  """Returns the action selected for a node index.

  See Appendix B in https://arxiv.org/pdf/1911.08265.pdf for more details.

  Args:
    rng_key: random number generator state.
    tree: _unbatched_ MCTS tree state.
    node_index: scalar index of the node from which to select an action.
    depth: the scalar depth of the current node. The root has depth zero.
    pb_c_init: constant c_1 in the PUCT formula.
    pb_c_base: constant c_2 in the PUCT formula.
    qtransform: a monotonic transformation to convert the Q-values to [0, 1].

  Returns:
    action: the action selected from the given node.
  """
  visit_counts = tree.children_visits[node_index]
  node_visit = tree.node_visits[node_index]
  pb_c = pb_c_init + jnp.log((node_visit + pb_c_base + 1.) / pb_c_base)
  prior_logits = tree.children_prior_logits[node_index]
  prior_probs = jax.nn.softmax(prior_logits)
  policy_score = jnp.sqrt(node_visit) * pb_c * prior_probs / (visit_counts + 1)
  chex.assert_shape([node_index, node_visit], ())
  chex.assert_equal_shape([prior_probs, visit_counts, policy_score])
  value_score = qtransform(tree, node_index)

  # Add tiny bit of randomness for tie break
  node_noise_score = 1e-7 * jax.random.uniform(
      rng_key, (tree.num_actions,))
  to_argmax = value_score + policy_score + node_noise_score

  # Masking the invalid actions at the root.
  return masked_argmax(to_argmax, tree.root_invalid_actions * (depth == 0))


@chex.dataclass(frozen=True)
class GumbelMuZeroExtraData:
  """Extra data for Gumbel MuZero search."""
  root_gumbel: chex.Array


GumbelMuZeroExtraDataType = TypeVar(  # pylint: disable=invalid-name
    "GumbelMuZeroExtraDataType", bound=GumbelMuZeroExtraData)


def gumbel_muzero_root_action_selection(
    rng_key: chex.PRNGKey,
    tree: tree_lib.Tree[GumbelMuZeroExtraDataType],
    node_index: chex.Numeric,
    *,
    num_simulations: chex.Numeric,
    max_num_considered_actions: chex.Numeric,
    qtransform: base.QTransform = qtransforms.qtransform_completed_by_mix_value,
) -> chex.Array:
  """Returns the action selected by Sequential Halving with Gumbel.

  Initially, we sample `max_num_considered_actions` actions without replacement.
  From these, the actions with the highest `gumbel + logits + qvalues` are
  visited first.

  Args:
    rng_key: random number generator state.
    tree: _unbatched_ MCTS tree state.
    node_index: scalar index of the node from which to take an action.
    num_simulations: the simulation budget.
    max_num_considered_actions: the number of actions sampled without
      replacement.
    qtransform: a monotonic transformation for the Q-values.

  Returns:
    action: the action selected from the given node.
  """
  del rng_key
  chex.assert_shape([node_index], ())
  visit_counts = tree.children_visits[node_index]
  prior_logits = tree.children_prior_logits[node_index]
  chex.assert_equal_shape([visit_counts, prior_logits])
  completed_qvalues = qtransform(tree, node_index)

  table = jnp.array(seq_halving.get_table_of_considered_visits(
      max_num_considered_actions, num_simulations))
  num_valid_actions = jnp.sum(
      1 - tree.root_invalid_actions, axis=-1).astype(jnp.int32)
  num_considered = jnp.minimum(
      max_num_considered_actions, num_valid_actions)
  chex.assert_shape(num_considered, ())
  # At the root, the simulation_index is equal to the sum of visit counts.
  simulation_index = jnp.sum(visit_counts, -1)
  chex.assert_shape(simulation_index, ())
  considered_visit = table[num_considered, simulation_index]
  chex.assert_shape(considered_visit, ())
  gumbel = tree.extra_data.root_gumbel
  to_argmax = seq_halving.score_considered(
      considered_visit, gumbel, prior_logits, completed_qvalues,
      visit_counts)

  # Masking the invalid actions at the root.
  return masked_argmax(to_argmax, tree.root_invalid_actions)


def gumbel_muzero_interior_action_selection(
    rng_key: chex.PRNGKey,
    tree: tree_lib.Tree,
    node_index: chex.Numeric,
    depth: chex.Numeric,
    *,
    qtransform: base.QTransform = qtransforms.qtransform_completed_by_mix_value,
) -> chex.Array:
  """Selects the action with a deterministic action selection.

  The action is selected based on the visit counts to produce visitation
  frequencies similar to softmax(prior_logits + qvalues).

  Args:
    rng_key: random number generator state.
    tree: _unbatched_ MCTS tree state.
    node_index: scalar index of the node from which to take an action.
    depth: the scalar depth of the current node. The root has depth zero.
    qtransform: function to obtain completed Q-values for a node.

  Returns:
    action: the action selected from the given node.
  """
  del rng_key, depth
  chex.assert_shape([node_index], ())
  visit_counts = tree.children_visits[node_index]
  prior_logits = tree.children_prior_logits[node_index]
  chex.assert_equal_shape([visit_counts, prior_logits])
  completed_qvalues = qtransform(tree, node_index)

  # The `prior_logits + completed_qvalues` provide an improved policy,
  # because the missing qvalues are replaced by v_{prior_logits}(node).
  to_argmax = _prepare_argmax_input(
      probs=jax.nn.softmax(prior_logits + completed_qvalues),
      visit_counts=visit_counts)

  chex.assert_rank(to_argmax, 1)
  return jnp.argmax(to_argmax, axis=-1).astype(jnp.int32)


def masked_argmax(
    to_argmax: chex.Array,
    invalid_actions: Optional[chex.Array]) -> chex.Array:
  """Returns a valid action with the highest `to_argmax`."""
  if invalid_actions is not None:
    chex.assert_equal_shape([to_argmax, invalid_actions])
    # The usage of the -inf inside the argmax does not lead to NaN.
    # Do not use -inf inside softmax, logsoftmax or cross-entropy.
    to_argmax = jnp.where(invalid_actions, -jnp.inf, to_argmax)
  # If all actions are invalid, the argmax returns action 0.
  return jnp.argmax(to_argmax, axis=-1).astype(jnp.int32)


def masked_multiple_choice(
    prng_key: chex.PRNGKey,
    num_choices: int,
    to_sample: chex.Array,
    invalid_actions: Optional[chex.Array]) -> chex.Array:
  """Returns a valid action with the highest `to_argmax`."""
  if invalid_actions is not None:
    chex.assert_equal_shape([to_sample, invalid_actions])
    # The usage of the -inf inside the argmax does not lead to NaN.
    # Do not use -inf inside softmax, logsoftmax or cross-entropy.
    to_sample = jnp.where(invalid_actions, -jnp.inf, to_sample)
  # If all actions are invalid, the argmax returns action 0.
  return jax.random.categorical(prng_key, to_sample, shape=(num_choices,)).astype(jnp.int32)



def masked_choice(
    prng_key: chex.PRNGKey,
    to_sample: chex.Array,
    num_actions: int,
    invalid_actions: Optional[chex.Array]) -> chex.Array:
  """Returns a valid action with the highest `to_argmax`."""
  if invalid_actions is not None:
    chex.assert_equal_shape([to_sample, invalid_actions])
    # The usage of the -inf inside the argmax does not lead to NaN.
    # Do not use -inf inside softmax, logsoftmax or cross-entropy.
    to_sample = jnp.where(invalid_actions, 0.0, to_sample)
  # If all actions are invalid, the argmax returns action 0.
  return jax.random.choice(prng_key, a=num_actions, p=to_sample).astype(jnp.int32) 

  # return jax.random.categorical(prng_key, to_sample, 1).astype(jnp.int32)


def masked_sample(
    prng_key: chex.PRNGKey,
    to_sample: chex.Array,
    invalid_actions: Optional[chex.Array]) -> chex.Array:
  """Returns a valid action with the highest `to_argmax`."""
  if invalid_actions is not None:
    chex.assert_equal_shape([to_sample, invalid_actions])
    # The usage of the -inf inside the argmax does not lead to NaN.
    # Do not use -inf inside softmax, logsoftmax or cross-entropy.
    to_sample = jnp.where(invalid_actions, -jnp.inf, to_sample)
  # If all actions are invalid, the argmax returns action 0.
  return jax.random.categorical(prng_key, to_sample, 1).astype(jnp.int32)


def _prepare_argmax_input(probs, visit_counts):
  """Prepares the input for the deterministic selection.

  When calling argmax(_prepare_argmax_input(...)) multiple times
  with updated visit_counts, the produced visitation frequencies will
  approximate the probs.

  For the derivation, see Section 5 "Planning at non-root nodes" in
  "Policy improvement by planning with Gumbel":
  https://openreview.net/forum?id=bERaNdoegnO

  Args:
    probs: a policy or an improved policy. Shape `[num_actions]`.
    visit_counts: the existing visit counts. Shape `[num_actions]`.

  Returns:
    The input to an argmax. Shape `[num_actions]`.
  """
  chex.assert_equal_shape([probs, visit_counts])
  to_argmax = probs - visit_counts / (
      1 + jnp.sum(visit_counts, keepdims=True, axis=-1))
  return to_argmax
