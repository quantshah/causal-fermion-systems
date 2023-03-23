import jax.numpy as jnp
import numpy as np
import jax


def make_hermitian(mat, n: int) -> jnp.array:
    """Generates a Hermitian linear operator from a random input matrix for a
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
    """Cost function to optimize

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
