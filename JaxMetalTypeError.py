# import jax
# from jax import numpy as jnp

# jax.print_environment_info()
# def fn(s):
#     ig = jax.lax.broadcast_in_dim(s, shape=(1024, 32, 1024), broadcast_dimensions=(1, 2))
#     ih = jax.lax.transpose(jax.lax.broadcast_in_dim(s, shape=(1024, 32, 1024), broadcast_dimensions=(1, 2)), permutation=(2, 1, 0))
#     return ih


# s = jnp.zeros((32, 1024))

# jax.jit(fn)(s)

# import jax
# from jax import numpy as jnp

# @jax.jit
# def do_broadcast(s):
#   return jax.lax.broadcast_in_dim(s, shape=(1024, 32, 1024), broadcast_dimensions=(1, 2))

# @jax.jit
# def do_transpose(ig):
#   return jax.lax.transpose(ig, permutation=(2, 1, 0))

# s = jnp.zeros((32, 1024))

# ig = do_broadcast(s)
# ih = do_transpose(ig)

# print(ih.shape)

import jax
from jax import numpy as jnp
import os

# Create a directory to store the XLA dumps
dump_dir = "/tmp/xla_dumps"
if not os.path.exists(dump_dir):
    os.makedirs(dump_dir)

print("--- JAX Environment Info ---")
jax.print_environment_info()
print("--------------------------\n")

def fn(s):
    # Broadcast 's' to a larger shape
    ig = jax.lax.broadcast_in_dim(s, shape=(1024, 32, 1024), broadcast_dimensions=(1, 2))
    # Broadcast 's' again and then transpose it
    ih = jax.lax.transpose(jax.lax.broadcast_in_dim(s, shape=(1024, 32, 1024), broadcast_dimensions=(1, 2)), permutation=(2, 1, 0))
    # A simple arithmetic operation to ensure the previous results are used
    return ig + ih


# JIT-compile the function
jit_fn = jax.jit(fn)

# Define the input data
s = jnp.zeros((32, 1024))

print(f"Input shape: {s.shape}")
print(f"Dumping XLA intermediate files to: {dump_dir}")

# Execute the JIT-compiled function to trigger XLA compilation
# block_until_ready() ensures the computation completes
result = jit_fn(s).block_until_ready()

print(f"\nExecution complete. Output shape: {result.shape}")
print("Check the specified directory for the dumped XLA files.")