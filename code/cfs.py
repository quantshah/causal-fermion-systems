import jax.numpy as jnp
import numpy as np
import jax
import optax
from tqdm.auto import tqdm
import matplotlib.pyplot as plt


def make_hermitian(mat, n: int) -> jnp.array:
    """Generates a Hermitian linear operator from a random input matrix
    with a fixed set of non-zero positive and negative eigenvalues.

    Args:
        mat (array[complex]): A random matrix.
        n (int, optional): The number of positive/negative eigenvalues.
                           Defaults to 2.
    """
    T = (mat + mat.T.conj()) / 2

    eigvals, eigvecs = jnp.linalg.eigh(T)

    eigvals = eigvals.at[2 * n :].set(0.0)
    eigvals = eigvals.at[:n].set(jnp.abs(eigvals[:n]))
    eigvals = eigvals.at[n:].set(-jnp.abs(eigvals[n:]))

    return jnp.dot(eigvecs, (eigvals * eigvecs.conj()).T)


def generate_random_mat(dim):
    """Generates a random matrix.

    Args:
        dim (int): The dimension of the Hilbert space

    Returns:
        array: A random array
    """
    return jnp.array(
        np.random.uniform(size=(dim, dim)) + 1j * np.random.uniform(size=(dim, dim))
    )


def diff(a, b):
    return jnp.abs(a) - jnp.abs(b)


vectorized_diff = jax.vmap(jax.vmap(diff, in_axes=(0, None)), in_axes=(None, 0))


@jax.jit
def lagrangian(x, y):
    """The Lagrangian between two points.

    Args:
        x (array[complex]): Hermitian matrix 1
        y (array[complex]): Hermitian matrix 2
    """
    eigvals = jnp.linalg.eigvals(x @ y)
    return (1 / 4) * jnp.sum(vectorized_diff(eigvals, eigvals) ** 2)/n


def cprod(c1, c2):
    """Simple product of two numbers that we will vectorize for both arrays
    instead of a double for loop.

    Args:
        c1, c2 (jnp.array): The coefficients

    Returns:
        float: The product
    """
    return c1 * c2


# The vectorized product of size dim(c1)*dim(c2).
vectorized_cprod = jax.jit(
    jax.vmap(jax.vmap(cprod, in_axes=[0, None]), in_axes=[None, 0])
)

# The vectorized version giving results of shape (n x n x N x N)
vectorized_make_hermitian = jax.jit(
    jax.vmap(lambda mat: make_hermitian(mat, n), in_axes=[0])
)

# The vectorized version giving results of shape (n x n)
vectorized_lagrangian = jax.jit(
    jax.vmap(jax.vmap(lagrangian, in_axes=[0, None]), in_axes=[None, 0])
)


def cost(hermitian_ops, cvals):
    """Cost function to optimize.

    Args:
        hermitian_ops (jnp.array): A (n, N x N) array of n Hermitian matrices
                                   each of dimension (N x N)
        cvals (jnp.array): A n-dimensional vector of the coefficients.

    Returns:
        float: The scalar cost
    """
    lvalues = vectorized_lagrangian(hermitian_ops, hermitian_ops)
    cprodvalues = vectorized_cprod(cvals, cvals)
    return jnp.sum(lvalues * cprodvalues)


dim = 2
n = 1
m = 5  # Number of operators


x = make_hermitian(generate_random_mat(dim), n)
y = make_hermitian(generate_random_mat(dim), n)


params = jnp.array([generate_random_mat(dim) for i in range(m)])
herm = vectorized_make_hermitian(params)

cvals = np.random.uniform(size=m)
cvals = jax.nn.softmax(cvals)

print(cost(herm, cvals))

@jax.jit
def loss_fn(params):
    """A loss function to optimize.

    Args:
        params (tuple of two arrays): The 

    Returns:
        _type_: _description_
    """
    mat, cvals = params
    herm = vectorized_make_hermitian(mat)
    herm = herm/jnp.trace(herm)
    cvals = jax.nn.softmax(cvals)
    # herm = vectorized_trace_normalization(herm)
    loss_val = cost(herm, cvals)
    return loss_val # + 0.01*jnp.sum(jnp.abs(params[0])) + 0.01*jnp.sum(jnp.abs(params[1]))


def get_optimized_matrices(dim, num_ops, num_iters = 1000):
    """Runs an optimization to generate Hermitian matrices by minimizing
    the Lagrangian/loss function.

    Args:
        dim (int): The dimensions for the Hermitian matrices to generate.
        num_ops (_type_): The number of Hermitian matrices to create
        num_iters (int, optional): Number of iterations. Defaults to 1000.

    Returns:
        array: The set of Hermitian matrices
    """
    mat = jnp.array([generate_random_mat(dim) for i in range(num_ops)])
    cvals = jnp.array(np.random.uniform(size=num_ops))
    params = [mat, cvals]

    optimizer = optax.adam(2e-3, 0.9, 0.9)
    opt_state = optimizer.init(params)

    @jax.jit
    def update(
        params,
        opt_state):
        """Single gradient descent update step."""
        grads = jax.grad(loss_fn)(params)
        herm_grads, cvals_grads = grads
        herm_grads = jnp.conj(herm_grads)
        # herm_grads = herm_grads/jnp.linalg.norm(herm_grads)

        grads = [herm_grads, cvals_grads]
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state

    loss_hist = []

    for i in tqdm(range(num_iters)):
        params, opt_state = update(params, opt_state)
        loss_hist.append(loss_fn(params))

    herm = vectorized_make_hermitian(params[0])
    cvals = jax.nn.softmax(params[1])
    # herm = vectorized_trace_normalization(herm)

    return herm, cvals, loss_hist


matrices, coeffs, loss_hist = get_optimized_matrices(2, 100, num_iters = 1000)

plt.plot(loss_hist)
print([np.trace(mat) for mat in matrices])