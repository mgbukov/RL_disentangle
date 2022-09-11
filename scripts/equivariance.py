from itertools import permutations
import random
import numpy as np

random.seed(0)
np.random.seed(0)
np.set_printoptions(precision=5, suppress=True)


#------------------------------- Permutations Group setup -------------------------------#
d = 4
fact = lambda n: 1 if n <= 1 else n * fact(n-1)
size_G = fact(d)

all_perms = {id: perm for id, perm in enumerate(permutations(range(d)))}
# 0  : (0, 1, 2, 3)
# 1  : (0, 1, 3, 2)
# 2  : (0, 2, 1, 3)
# ...
# 23 : (3, 2, 1, 0)

perm_to_idx = {p: i for i, p in all_perms.items()}
# (0, 1, 2, 3) : 0
# (0, 1, 3, 2) : 1
# (0, 2, 1, 3) : 2
# ...
# (3, 2, 1, 0) : 23

def perm_inverse(p):
    """Compute the inverse of a permutation.

    Args:
        p (tuple[int]): A tuple of integers representing the permutation pi,
            e.g. (2, 0, 3, 1).

    Result:
        p_inv (tuple[int]): A tuple of integers representing the permutation inverse, i.e. 
            p_inv * p = id, e.g. (1, 3, 0, 2). 
    """
    p_inv = [None] * len(p)
    for i, t in enumerate(p):
        p_inv[t] = i
    return tuple(p_inv)

def perm_prod(pi, pj):
    """Compute the product of two permutations.

    Args:
        pi (tuple[int]): A tuple of integers representing the permutation pi,
            e.g. (3, 1, 2, 0).
        pi (tuple[int]): A tuple of integers representing the permutation pj,
            e.g. (2, 0, 1, 3).

    Result:
        p (tuple[int]): A tuple of integers representing the permutation p = pi * pj,
            e.g. (2, 3, 1, 0).        
    """
    return tuple(pi[i] for i in pj)

def perm_apply(x, g):
    """Applying permutation `g` to input `x`.

    Args:
        x (np.Array): A numpy array of shape (1, X), giving the input to be permuted.
        g (tuple[int]): A tuple of integers representing the permutation g.
    
    Result:
        x_p (np.Array): A numpy array of shape (1, X), giving the permuted input.
    """
    x_p = x[:, g]
    return x_p

# cayley_table[(i, j)] = k, where all_perms[k] = pi * pj
cayley_table = {(i, j): perm_to_idx[perm_prod(pi, pj)]
                for i, pi in all_perms.items()
                for j, pj in all_perms.items()}

# inverse_table[i] = j, where all_perms[j] = pi_inv
inverse_table = {i: perm_to_idx[perm_inverse(pi)] for i, pi in all_perms.items()}

def prod_with_inverse(u, g):
    """Return the permutation that corresponds to u * g_inv"""
    return all_perms[cayley_table[u, inverse_table[g]]]

# # These are helper functions defined to work on the indexes of the permutation group G.
# # All function accept the id of a permutation as input and return the id of a permutation
# # as an output.
# def inverse(g): return inverse_table[g]
# def prod(u, g): return cayley_table[(u, g)]
# def apply(x, g): return perm_apply(x, all_perms[g])


#---------------------------------- Forward pass setup ----------------------------------#
# The input to our network is a vector of shape (1, d). The output is also a vector which
# is of shape (1, o) that specifies the score associated to each class (each action).
# Here `o` is the number of output classes.
# For every permutation of the group G we need to associate a transformation of the output.
# E.g. we would like to specify what should be the result of permuting the input x with
# permutation g and forwarding it through the network.
#
# The easiest case is when the output has the same shape as the input, i.e. o = d.
# In this case we would like to permute the scores in the same way that we have permuted
# the input features.
#
#                          model(x)
# x   = [x0, x1, x2, x3]   ---------->   y   = [y0, y1, y2, y3]
# x_p = [x3, x2, x0, x1]   ---------->   y_p = [y3, y2, y0, y1]
#
# The cosets are a partitioning of G such that to every output class corresponds one coset.
# Choosing an arbitrary output class `y*` as our pivot point, the coset for class `y_i` is
# defined as follows: all the group elements g that transform `y*` into `y_i`.
# WLOG we can choose `y*` = `y_0`.
#
#                      g in coset of `y_2`
# y = [y0, ?, ?, ?]   --------------------->   y_p = [y2, ?, ?, ?]
#
# cosets[act_i] = {idx(g)| g in G, g*act_0 = act_i}
#
o = d
cosets ={i : [g_id for g_id, g in all_perms.items() if g[0] == i] for i in range(o)}
# cosets = {}
# for i in range(o):
#     cosets[i] = []
#     for g_id, g in enumerate(all_perms):
#         if g[0] == i:    # the permutation transforms class `y_0` into class `y_i`
#             cosets[i].append(g_id)

def forward(x, weights, o):
    """Forward pass of a permutation equivariant linear model.
    x(shape=(1, d)) @ w(shape=(d, o)) = y(shape=(1, o))

    Args:
        x (np.Array): A numpy array of shape (1, d) giving the input features.
        weights (np.Array): A numpy array of shape (size_G, d, o) giving the model weights.
            A different array of shape (d, o) corresponds to every member of the group G.
        o (int): Number of output classes.
    
    Result:
        y (np.Array): A numpy array of shape (1, o) giving the output scores for each class.
    """
    y = np.zeros(shape=(1, o))

    for i in range(o):
        y_i = []

        # Sum_u y(u) (formula 10.11)
        for u in cosets[i]:
            # (f*w)(u) = Sum_g f (ug-1) * w(g) (formula 10.6)
            for g in all_perms.keys():
                ug_inv = prod_with_inverse(u, g)

                # Permutation of x - f(ug-1)
                x_p = perm_apply(x, ug_inv)

                # W  ---->g    W'
                # Instead of permuting w we are associating a different w to every g.
                # Thus instead of w_p = apply(w, g), we use w_p = weights[g] 
                w_p = weights[g]
                out = x_p @ w_p # shape=(1, o)

                # After the forward pass we get output of shape (1, o).
                # We need to produce one single number for our current action y[i]. Thus,
                # we need to sum the outputs (or take mean w/e). The reason for this is
                # that the values in this output vector are not associated in any way with
                # the other output classes, but only with class `i`.
                out = out.mean()
                y_i.append(out)

        # After iterating over all the groups in the coset of action `y_i` we need to
        # "project down" to the action space.
        # y[i] = 1/|H| * Sum_u y^G(u) = 1/|H| * Sum_u Sum_G (f*w)(u)
        y_i = np.array(y_i).mean()
        y[0, i] = y_i

    return y


# Create random input and random weights.
x = np.random.randint(-10, 10, size=(1, d))
#
# Instead of permuting w we are associating a different w to every g.
# Thus apply(w, g) = weights[g]
weights = np.random.randint(-10, 10, size=(size_G, d, o))

print("\n### Linear model ###")
print("\tx_p\t\t\ty_p\n")
for perm in all_perms.values():
    x_p = perm_apply(x, perm)
    y_p = forward(x_p, weights, o)
    print(x_p, y_p)


#----------------------------- Multi-layer Perceptron (MLP) -----------------------------#
# If we would like to use a more complicated model, then we need to perform the same
# operation for all of the model's layers.
#

def mlp_forward(x, model_parameters, o):
    """Forward pass of a permutation equivariant multi-layer perceptron model.
    x (shape=(1,d)) --> FC_1 (shape(d, h1)) --> ... --> FC_n (shape(h, hn)) --> y (shape(1, o))

    Args:
        x (np.Array): A numpy array of shape (1, d) giving the input features.
        model_parameters (dict): A dictionary containing the weights for every layer:
            i (int): A numpy array of shape (size_G, h_(i-1), h_i) giving the weights of
                the i-th hidden layer.
        o (int): Number of output classes.
    
    Result:
        y (np.Array): A numpy array of shape (1, o) giving the output scores for each class.
    """

    y = np.zeros(shape=(1, o))

    for i in range(o):
        y_i = []

        # Sum_u y(u) (formula 10.11)
        for u in cosets[i]:
            # (f*w)(u) = Sum_g f (ug-1) * w(g) (formula 10.6)
            for g in all_perms.keys():
                ug_inv = prod_with_inverse(u, g)

                # Permutation of x - f(ug-1)
                x_p = perm_apply(x, ug_inv)

                # Now instead of a linear model we have a multi-layer perceptron.
                out = x_p
                num_layers = len(model_parameters)
                for j in range(num_layers):
                    w, b = model_parameters[j]

                    # Again, instead of permuting w we are associating a different w to every g.
                    w_p, b_p = w[g], b[g]
                    out = out @ w_p + b_p
                    if j < num_layers-1: # apply ReLU non-linearity on all but the last layer
                        out = np.maximum(0, out)

                # After the forward pass we get output of shape (1, o).
                # We need to produce one single number for our current action y[i]. Thus,
                # we need to sum the outputs (or take mean w/e). The reason for this is
                # that the values in this output vector are not associated in any way with
                # the other output classes, but only with class `i`.
                out = out.mean()
                y_i.append(out)

        # After iterating over all the groups in the coset of action `y_i` we need to
        # "project down" to the action space.
        # y[i] = 1/|H| * Sum_u y^G(u) = 1/|H| * Sum_u Sum_G (f*w)(u)
        y_i = np.array(y_i).mean()
        y[0, i] = y_i

    return y

hidden_sizes = [128, 64, 32] + [o]
model_parameters = {}
prev = d
for i, h in enumerate(hidden_sizes):
    #                                       w                       b
    model_parameters[i] = (np.random.randn(size_G, prev, h), np.random.randn(size_G, h) / 100)
    prev = h

print("\n### Multi-layer perceptron ###")
print("\tx_p\t\t\ty_p\n")
for perm in all_perms.values():
    x_p = perm_apply(x, perm)
    y_p = mlp_forward(x_p, model_parameters, o)
    print(x_p, y_p)

#------------------------------------ Vectorization -------------------------------------#

def forward_vectorized(x, weights, o):
    """A vectorized forward pass of a permutation equivariant linear model.
    x(shape=(1, d)) @ w(shape=(d, o)) = y(shape=(1, o))

    Args:
        x (np.Array): A numpy array of shape (1, d) giving the input features.
        weights (np.Array): A numpy array of shape (size_G, d, o) giving the model weights.
            A different array of shape (d, o) corresponds to every member of the group G.
        o (int): Number of output classes.
    
    Result:
        y (np.Array): A numpy array of shape (1, o) giving the output scores for each class.
    """
    y = np.zeros(shape=(1, o))

    # Pre-generate all permutations of the input vector.
    x_lift = np.ndarray(shape=(size_G, *x.shape))
    for i, g in all_perms.items():
        x_lift[i] = perm_apply(x, g)

    for i in range(o):
        y_i = []
        for u in cosets[i]:
            perms_to_consider = [cayley_table[u, inverse_table[g]] for g in all_perms.keys()]
            out = x_lift[perms_to_consider] @ weights   # shape = (size_G, 1, o)
            y_i.append(out.mean())
        y_i = np.array(y_i).mean()
        y[0, i] = y_i

    return y

def forward_vectorized_extra(x, weights, o):
    """A vectorized forward pass of a permutation equivariant linear model.
    x(shape=(1, d)) @ w(shape=(d, o)) = y(shape=(1, o))

    Args:
        x (np.Array): A numpy array of shape (1, d) giving the input features.
        weights (np.Array): A numpy array of shape (size_G, d, o) giving the model weights.
            A different array of shape (d, o) corresponds to every member of the group G.
        o (int): Number of output classes.
    
    Result:
        y (np.Array): A numpy array of shape (1, o) giving the output scores for each class.
    """
    # Pre-generate all permutations of the input vector.
    x_lift = np.ndarray(shape=(size_G, *x.shape))
    for i, g in all_perms.items():
        x_lift[i] = perm_apply(x, g)

    x_double_lift = np.ndarray(shape=(size_G, *x_lift.shape))
    j = 0
    for i in range(o):
        for u in cosets[i]:
            perms_to_consider = [cayley_table[u, inverse_table[g]] for g in all_perms.keys()]
            x_double_lift[j] = x_lift[perms_to_consider]
            j += 1

    out_lift = x_double_lift @ np.expand_dims(weights, axis=0)
    return out_lift.reshape(1, o, -1).mean(axis=-1)

#------------------------------------- Qubit system -------------------------------------#
# When considering our qubit system the shape of the input is (1, d) = (1, 2**L). However,
# we will only be considering a subset of all the possible permutations.
# Instead of looking at transformations that could permute every single feature of our
# input, this subset consists of the transformations that permute entire blocks of the
# system. Namely, the blocks that correspond to the contribution of a single qubit.
# NOTE: We can see that this subset of transformations also forms a group under the group
# operation. Let us call this subgroup G*.
L = 3
qubit_d = 2 ** L
size_sub_G = fact(L)
qubit_perms = {id: perm for id, perm in enumerate(permutations(range(L)))}
qubit_perm_to_idx = {p: i for i, p in qubit_perms.items()}
qubit_cayley_table = {(i, j): qubit_perm_to_idx[perm_prod(pi, pj)]
                for i, pi in qubit_perms.items()
                for j, pj in qubit_perms.items()}
qubit_inverse_table = {i: qubit_perm_to_idx[perm_inverse(pi)] for i, pi in qubit_perms.items()}

def qubit_prod_with_inverse(u, g):
    """Return the permutation that corresponds to u * g_inv"""
    return qubit_perms[qubit_cayley_table[u, qubit_inverse_table[g]]]

# We now need to redefine the function for applying the permutation transformation.
def qubit_perm_apply(x, g):
    """Applying permutation `g` to input `x`.
    Our input has a shape of (1, d), consisting of d "features".
    The input represents a quantum system of L qubits. Here d = 2 ** L.
    The permutation `g` should be applied at the `qubit-level` instead of at the
    `feature-level`.

    Args:
        x (np.Array): A numpy array of shape (1, 2**L), giving the input to be permuted.
        g (tuple[int]): A tuple of integers representing the permutation g.

    Result:
        x_p (np.Array): A numpy array of shape (1, 2**L), giving the permuted input.
    """
    _, d = x.shape
    L = int(np.log2(d))
    #                                  /------- L ------\
    # Reshape the input into the shape (2, 2, 2, ...., 2).
    x_p = x.reshape((2,)*L)
    #
    # Having the input reshaped in this form, we can perform the permutation transformation
    # using transposition of the tensor axes.
    x_p = x_p.transpose(*g)
    #
    # After transposing the tensor, reshape back into the original shape.
    return x_p.reshape(1, 2**L)


# Another thing that we need to consider is that we need to partition the subgroup into
# cosets. Again for every permutation of the subgroup G* we need to associate a
# transformation of the output, e.g. we would like to specify what should be the result of
# permuting the input `x` with permutation `g` and forwarding it through the network.
#
# In our specific case every output class corresponds to a tuple of qubits.
# y[0] = (q0, q1)
# y[1] = (q0, q2) ...
#
qubit_o = L * (L - 1)
ys = {idx : p for idx, p in enumerate(permutations(range(L), 2))}
#
# We would like to permute the scores of the classes in the same way that we have permuted
# the qubit blocks in the input. For example: permuting q1 and q2 should result in
# permuting y0 and y1, because y0 corresponds to the score of (q0, q1) and y1 corresponds
# to the score of (q0, q2)
#                                                             model(x)
# x   = [x0, x1, ..., x(2**L-1)] ~~> {q0, q1, q2, ..., q(L-1)} --------> y   = [y0, y1, ..., yL*(L-1)]
# x_p = [xi, xj, ...., ...., xk] ~~> {q0, q2, q1, ..., q(L-1)} --------> y_p = [y1, y0, ..., yL*(L-1)]
#
# The cosets are again constructed by considering all group elements `g` that transform
# the pivot point into `y_i` for all `i`.
# Again, `y*` = y_0 = (q0, q1)
qubit_cosets = {i: [g_id for g_id, g in qubit_perms.items() if (g[0], g[1])==ys[i]]
                for i in range(qubit_o)}
# qubit_cosets = {}
# for i in range(qubit_o):
#     idxs = []
#     for g_id, g in qubit_perms.items():
#         # Action `i` acts on qubits `qk` and `ql`. If permutation `g`` permutes the system
#         # in such a way that qubits `qk` and `ql` arrive at positions 0 and 1, i.e.
#         # {q0, q1, ..., q(L-1)} g~> {qk, ql, ?, ?, ..., ?}, then permutation `g` is in the
#         # coset of action `i`.
#         qi, qj = ys[i]
#         if g[0] == qi and g[1] == qj:
#             idxs.append(g_id)
#     qubit_cosets[i] = idxs

# Finally, redefine the forward function.
def qubit_forward(x, weights, o):
    """Forward pass of a permutation equivariant linear model.
    x(shape=(1, 2**L)) @ w(shape=(2**L, L*(L-1))) = y(shape=(1, L*(L-1)))

    Args:
        x (np.Array): A numpy array of shape (1, 2**L) giving the input quantum system.
        weights (np.Array): A numpy array of shape (size_sub_G, 2**L, L*(L-1)) giving the
            model weights.
        o (int): Number of output classes.
    
    Result:
        y (np.Array): A numpy array of shape (1, L*(L-1)) giving the output scores for
            each class.
    """
    y = np.zeros(shape=(1, o))
    for i in range(o):
        y_i = []
        for u in qubit_cosets[i]:
            for g in qubit_perms.keys():
                ug_inv = qubit_prod_with_inverse(u, g)
                x_p = qubit_perm_apply(x, ug_inv)
                w_p = weights[g]
                out = x_p @ w_p
                out = out.mean()
                y_i.append(out)
        y_i = np.array(y_i).mean()
        y[0, i] = y_i
    return y

# Create random input and random weights.
qubit_x = np.random.randint(-10, 10, size=(1, qubit_d))
qubit_weights = np.random.randint(-10, 10, size=(size_sub_G, qubit_d, qubit_o))
#
print("\n\n### QUBIT SYSTEM ###")
print("\tx_p\t\t\tperm\t\t\t\t\ty_p")
print(f"\t\t\t\t\t{'   '.join(map(str, ys.values()))}")
for perm in qubit_perms.values():
    x_p = qubit_perm_apply(qubit_x, perm)
    y_p = qubit_forward(x_p, qubit_weights, qubit_o)
    print(x_p, perm, y_p)

#