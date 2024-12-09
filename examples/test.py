import jax.numpy as jnp

def count_elements(input_array):
    # Create a boolean mask for each element comparing it to all other elements
    unique_count = jnp.array([jnp.sum(input_array == val) for val in input_array])
    return unique_count

# Example usage
input_array = jnp.array([[1, 4, 3, 3, 3, 1]])
output_array = count_elements(input_array)

print(output_array)