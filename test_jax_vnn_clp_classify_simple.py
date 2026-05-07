'''
Von Neumann network that mimics an MLP
'''
import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import optax
# import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
from typing import Tuple
import time

from sklearn import datasets
from sklearn.model_selection import train_test_split

from universe import UniverseLP, CoordinateCells #local
import utils #local

utils.jax_status()

#parameters
epochs = 60
length = 3
padding = 6
samples = 320
batch_size = 2
num_batches = samples//batch_size
num_classes = 3
epsilon = 1e-8
N = int(length/2+0.5)+padding
start_state = 1 #set None for random state

#===========
##data
X_preproc, Y_raw = datasets.make_classification(
    n_samples=samples, n_features=3, n_redundant=0, n_informative=2, n_clusters_per_class=1, n_classes=3, random_state=start_state
)
print("Data Shape:", X_preproc.shape, "Samples:", samples, "Input Length:", length, "Classes:", num_classes)

data_split = 0.2
X, X_test, Y, Y_test = train_test_split(X_preproc, Y_raw, test_size=data_split, random_state=42)

#reshape to number of batches
X = np.reshape(X, (int(num_batches*(1.0-data_split)),batch_size,length))
Y = np.reshape(Y, (int(num_batches*(1.0-data_split)),batch_size))
X_test = np.reshape(X_test, (int(num_batches*data_split),batch_size,length))
Y_test = np.reshape(Y_test, (int(num_batches*data_split),batch_size))
print("X shape:", X.shape, "X test shape:", X_test.shape)
print("Y shape:", Y.shape, "Y test shape:", Y_test.shape)

#===========
#load network architecture
#input coords
input_coords = []
input_coords.append([1,2])
input_coords.append([3,1])
input_coords.append([5,2])

#output
output_coords = []
output_coords.append([2,6])
output_coords.append([3,6])
output_coords.append([4,6])

output_size = (num_classes,1)
output_offset = (2,6)

input_coords = jnp.array(input_coords)
output_coords = jnp.array(output_coords)
print("input coords:", input_coords.shape, input_coords)
print("N:", N, "Epochs:", epochs)

#locations
#MLP with 1 input and 1 output, one layer inbetween
#setup
kernel_size = (3,3)
activation = jax.nn.tanh
final_activation = None #jax.nn.sigmoid
directions = 1 #how many filters, i.e. possible directions, to learn
#depth
# in_row_coord = input_coords[:, 0].min() #top most in
# out_row_coord = output_coords[:, 0].min() #top most out
# in_col_coord = input_coords[:, 1].min() #far left in
# out_col_coord = output_coords[:, 1].max() #far right out
# depth_row = out_row_coord - in_row_coord
# depth_col = out_col_coord - in_col_coord
# depth = max(depth_row, depth_col)
depth = 5
print("depth:", depth)

#===========
##create model
def _forward(batch) -> jnp.ndarray:
    '''
    Forward pass through network
    '''
    clp = UniverseLP(N, sharpen=False)
    clp.initialize_clp(input_coords, output_coords, output_offset, output_size, [], depth, kernel_size, directions=directions)
    return clp(batch, activation, final_activation, norm=False)

# Make the network and optimizer. Haiku standard
network = hk.without_apply_rng(hk.transform_with_state(_forward))
# opt = optax.sgd(learning_rate=0.005)
opt = optax.adam(learning_rate=0.01)

# Initialize network and optimiser; note we draw an input to get shapes.
x = X[0]
# x = x[jnp.newaxis, :] #fake batch dim
print("x", x.shape, x)
# x = jnp.reshape(x, (x.shape[0], length, 1)) #to col vector
params, state = network.init(jax.random.PRNGKey(42), x)
opt_state = opt.init(params)
# print(params)
# print(state)

#save architecture
#set input and output states
s_init = utils.initializer_to_function(CoordinateCells(input_coords, output_coords, output_offset, output_size, is_states=True, random_init=False))
arch = s_init(rng=jax.random.PRNGKey(42), shape=(N,N), dtype=jnp.float32)
print("arch", arch.shape, type(arch))

#loss function
def loss(params: hk.Params, state, batch, labels, l2_scaling=1e-3) -> jnp.ndarray:
    '''
    Cross entropy
    '''
    labels = jax.nn.one_hot(labels, num_classes)
    # print("labels", labels.shape)

    #CE loss
    logits, state = network.apply(params, state, batch)
    # print("logits", logits.shape)
    logits = jnp.squeeze(logits, axis=-1) #remove extra dim
    # ce = -jnp.sum(labels * jax.nn.log_softmax(logits))
    # ce = -1 * jnp.mean(labels * jnp.log(logits + epsilon) + (1 - labels) * jnp.log(1. - logits + epsilon))

    #L2 norm of weights loss
    # filters = params['universe_lp']['f']
    # l2 = jnp.mean(filters**2)

    # return ce + l2*l2_scaling, state
    # return ce, state

    return optax.softmax_cross_entropy(logits=logits, labels=labels).mean(), state

@jax.jit
def update(
        params: hk.Params,
        opt_state: optax.OptState,
        state,
        batch,
        labels
    ) -> Tuple[hk.Params, optax.OptState]:
    """Learning rule (stochastic gradient descent)."""
    (loss_value, state), grads = jax.value_and_grad(loss, has_aux=True)(params, state, batch, labels)
    updates, opt_state = opt.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, loss_value, state

# Train/eval loop.
start = time.time() #time generation
for step in range(1, epochs+1):
    # Do SGD on a batch of training examples.
    total = 0
    loss_value = 0
    for x, y in zip(X, Y):
        # x = x[jnp.newaxis, :] #fake batch dim
        # x = jnp.reshape(x, (x.shape[0], length, 1)) #to col vector
        params, opt_state, value, state = update(params, opt_state, state, x, y)
        loss_value += value
        total += 1
        # break
    loss_value /= total

    if step % 5 == 0:
        correct = 0
        total = 0
        for x, y in zip(X, Y):
            # x = x[jnp.newaxis, :] #fake batch dim
            # x = jnp.reshape(x, (x.shape[0], length, 1)) #to col vector
            logits, state_test = network.apply(params, state, x)
            predicted = jnp.squeeze(logits, axis=-1) #remove extra dim
            predicted = jnp.argmax(predicted, axis=-1)
            # print(predicted.shape, y.shape)
            correct += jnp.sum(predicted == y)
            total += batch_size
        # print(correct, total)
        accuracy = 100 * correct / total
        print(f"[Step {step}, Loss {loss_value}, Ave.Loss {loss_value/samples}, Train Accuracy {accuracy}]")
    else:
        print(f"[Step {step}, Loss {loss_value}, Ave.Loss {loss_value/samples}]")
end = time.time()
elapsed = end - start
print("Training took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")

jnp.savez("clp_classify_cells.npz",
            weights=params['universe_lp']['w'],
            biases=params['universe_lp']['b'],
            filters=params['universe_lp']['f'],
            states=params['universe_lp']['s'],
            N=N, length=length, num_classes=num_classes, input_coords=input_coords, output_coords=output_coords,
            neuron_coords=[], kernel_size=kernel_size,
            allow_pickle=False)

plt.imshow(arch)
plt.title("Initial Architecture")
plt.savefig("vnn_clp_classify_arch.png", dpi=150)
plt.show()

plt.imshow(params['universe_lp']['w'])
plt.title("Final Weights")
plt.savefig("vnn_clp_classify_weights.png", dpi=150)
plt.show()

plt.imshow(params['universe_lp']['b'])
plt.title("Final Biases")
plt.savefig("vnn_clp_classify_biases.png", dpi=150)
plt.show()

plt.imshow(params['universe_lp']['f'][0,:,:,0])
plt.title("Final Filter")
plt.savefig("vnn_clp_classify_filter.png", dpi=75)
plt.show()

plt.imshow(params['universe_lp']['s'])
plt.title("Final States")
plt.savefig("vnn_clp_classify_states.png", dpi=75)
plt.show()

plt.imshow(state['universe_lp']['universe'][0])
plt.title("Final Universe")
plt.savefig("vnn_clp_classify_universe.png", dpi=75)
plt.show()

#test performance
correct = 0
total = 0
for x, y in zip(X_test, Y_test):
    # x = x[jnp.newaxis, :] #fake batch dim
    # x = jnp.reshape(x, (x.shape[0], length, 1)) #to col vector
    logits, state_test = network.apply(params, state, x)
    predicted = jnp.squeeze(logits, axis=-1) #remove extra dim
    predicted = jnp.argmax(predicted, axis=-1)
    correct += jnp.sum(predicted == y)
    total += batch_size
accuracy = 100 * correct / total
print(f"[Final Accuracy {accuracy}]")

print("END")
