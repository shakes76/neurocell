'''
Von Neumann network that mimics an MLP
'''
import jax
import numpy as np
import jax.numpy as jnp
import haiku as hk
import optax
# import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
from typing import Tuple
import time

from sklearn.model_selection import train_test_split

from universe import UniverseLP, CoordinateCells #local
import utils #local

utils.jax_status()

int_dtype = np.int32
ds_dtype = np.float32

#parameters
epochs = 20
bits = 2
length = bits
padding = 6
samples = 160
num_classes = 1
length_data = bits*samples
N = int(length/2+0.5)+padding
delta = 0.2

#===========
##data
X_preproc = np.zeros((length_data), dtype=int_dtype)
X_preproc[:length_data//2] = 1 #equal number of 0 and 1
np.random.shuffle(X_preproc)
X_preproc = np.reshape(X_preproc, (samples, bits))
Y_raw = X_preproc[:,0] ^ X_preproc[:,1] #xor
# Y_raw = X_preproc[:,0] & X_preproc[:,1] #and

#add noise (necessary)
noise = np.random.uniform(-delta,+delta, size=(samples, bits))
X_preproc = X_preproc.astype(ds_dtype) + noise

print("Data Shape:", X_preproc.shape, "Samples:", samples, "Input Length:", length, "Classes:", num_classes)
# print(X_preproc, Y_raw)
# print(X_preproc.astype(ds_dtype), Y_raw.astype(ds_dtype))

X, X_test, Y, Y_test = train_test_split(X_preproc, Y_raw, test_size=0.2, random_state=42)
print("X shape", X.shape, "Y shape:", Y.shape, "Y test shape:", Y_test.shape)
print("X", X[:3])
print("Y", Y[:3])

#===========
#load network architecture
#input coords
input_coords = []
input_coords.append([2,1])
input_coords.append([4,1])

#output
output_coords = []
output_coords.append([3,4])

output_size = (num_classes,1)
output_offset = (3,4)

input_coords = jnp.array(input_coords)
output_coords = jnp.array(output_coords)
print("input coords:", input_coords.shape, input_coords)
print("N:", N, "Epochs:", epochs)

#locations
#MLP with 1 input and 1 output, one layer inbetween
#setup
kernel_size = (3,3)
activation = jax.nn.tanh
final_activation = None
#depth
depth = 4
print("depth:", depth)

#===========
##create model
def _forward(batch) -> jnp.ndarray:
    '''
    Forward pass through network
    '''
    clp = UniverseLP(N, sharpen=False)
    clp.initialize_clp(input_coords, output_coords, output_offset, output_size, [], depth, kernel_size)
    return clp(batch, activation, final_activation, norm=True)

# Make the network and optimizer. Haiku standard
network = hk.without_apply_rng(hk.transform_with_state(_forward))
# opt = optax.sgd(learning_rate=0.005)
opt = optax.adam(learning_rate=0.01)

# Initialize network and optimiser; note we draw an input to get shapes.
x = X[0]
x = x[jnp.newaxis, :] #fake batch dim
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
    # labels = jax.nn.one_hot(labels, num_classes)
    # print("labels", labels.shape)

    #CE loss
    logits, state = network.apply(params, state, batch)
    # print("logits", logits.shape)
    logits = jnp.squeeze(logits[0], axis=-1) #remove extra dim
    # ce = -jnp.sum(labels * jax.nn.log_softmax(logits))
    # print(logits, labels)
    # print(optax.softmax_cross_entropy(logits=logits, labels=labels))

    #L2 norm of weights loss
    # filters = params['universe_lp']['f']
    # l2 = jnp.mean(filters**2)

    # return ce + l2*l2_scaling, state
    # return ce, state

    # return optax.softmax_cross_entropy(logits=logits, labels=labels).mean(), state
    # return optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=labels).mean(), state
    return optax.l2_loss(predictions=logits[0], targets=labels).mean(), state

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
        x = x[jnp.newaxis, :] #fake batch dim
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
            x = x[jnp.newaxis, :] #fake batch dim
            # x = jnp.reshape(x, (x.shape[0], length, 1)) #to col vector
            logits, state_test = network.apply(params, state, x)
            # print("logits shape", logits.shape)
            predicted = jnp.squeeze(logits[0], axis=-1) #remove extra dim
            # print("pred shape", predicted.shape)
            predicted = jnp.round(predicted)
            # print("pred", predicted)
            correct += jnp.sum(predicted[0] == y)
            total += 1
        # print(correct, total)
        accuracy = 100 * correct / total
        print(f"[Step {step}, Loss {loss_value}, Ave.Loss {loss_value/samples}, Train Accuracy {accuracy}]")
    else:
        print(f"[Step {step}, Loss {loss_value}, Ave.Loss {loss_value/samples}]")
    # break
end = time.time()
elapsed = end - start
print("Training took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")

jnp.savez("clp_xor_cells.npz",
            weights=params['universe_lp']['w'],
            biases=params['universe_lp']['b'],
            filters=params['universe_lp']['f'],
            states=params['universe_lp']['s'],
            N=N, length=length, num_classes=num_classes, input_coords=input_coords, output_coords=output_coords,
            neuron_coords=[], kernel_size=kernel_size,
            allow_pickle=False)

plt.imshow(arch)
plt.title("Initial Architecture")
plt.savefig("vnn_clp_xor_arch.png", dpi=150)
plt.show()

plt.imshow(params['universe_lp']['w'])
plt.title("Final Weights")
plt.savefig("vnn_clp_xor_weights.png", dpi=150)
plt.show()

plt.imshow(params['universe_lp']['b'])
plt.title("Final Biases")
plt.savefig("vnn_clp_xor_biases.png", dpi=150)
plt.show()

plt.imshow(params['universe_lp']['f'][0,:,:,0])
plt.title("Final Filter")
plt.savefig("vnn_clp_xor_filter.png", dpi=75)
plt.show()

plt.imshow(params['universe_lp']['s'])
plt.title("Final States")
plt.savefig("vnn_clp_xor_states.png", dpi=75)
plt.show()

plt.imshow(state['universe_lp']['universe'][0])
plt.title("Final Universe")
plt.savefig("vnn_clp_xor_universe.png", dpi=75)
plt.show()

#test performance
correct = 0
total = 0
for x, y in zip(X_test, Y_test):
    x = x[jnp.newaxis, :] #fake batch dim
    logits, state_test = network.apply(params, state, x)
    predicted = jnp.squeeze(logits[0], axis=-1) #remove extra dim
    predicted = jnp.round(predicted)
    print("x", x, "y_pred", predicted, "y", y)
    correct += jnp.sum(predicted[0] == y)
    total += 1
accuracy = 100 * correct / total
print(f"[Final Accuracy {accuracy}]")

print("END")
