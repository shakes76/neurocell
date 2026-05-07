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

from universe import UniverseLP #local
import utils #local

utils.jax_status()

#parameters
N = 13
epochs = 50
samples = 128
batch_size = 8
num_batches = samples//batch_size
length = 3
num_classes = 3
epsilon = 1e-8
start_state = 1 #set None for random state
print("N:", N, "Epochs:", epochs)

#===========
##data
X, Y = datasets.make_classification(
    n_samples=samples, n_features=3, n_redundant=0, n_informative=2, n_clusters_per_class=1, n_classes=3, random_state=start_state
)
print("X_pre shape", X.shape, "Y_pre shape", Y.shape)

#reshape to number of batches
X = np.reshape(X, (num_batches,batch_size,length))
Y = np.reshape(Y, (num_batches,batch_size))

#locations
#MLP with 1 input and 1 output, one layer inbetween
#setup
input_size = (length,1)
input_offset = (5,4)
output_size = (num_classes,1)
output_offset = (5,8)
kernel_size = (7,3)
activation = jax.nn.tanh
final_activation = jax.nn.sigmoid
depth = 3
print("depth:", depth)

#===========
##create model
def _forward(batch) -> jnp.ndarray:
    '''
    Forward pass through network
    '''
    mlp = UniverseLP(N)
    mlp.initialize_mlp(input_size, output_size, input_offset, output_offset, depth, kernel_size)
    return mlp(batch, activation, final_activation, norm=False)

# Make the network and optimizer. Haiku standard
network = hk.without_apply_rng(hk.transform_with_state(_forward))
# opt = optax.sgd(learning_rate=0.005)
opt = optax.adam(learning_rate=0.01)

# Initialize network and optimiser; note we draw an input to get shapes.
x = X[0]
# print("X0 shape:", x.shape)
x = x[:, jnp.newaxis] #fake 2D
params, state = network.init(jax.random.PRNGKey(42), x)
opt_state = opt.init(params)
# print(params)
# print(state)

plt.imshow(params['universe_lp']['s'])
plt.title("Initial States")
plt.savefig("vnn_mlp_classify_states_init.png", dpi=75)
plt.show()

states = params['universe_lp']['s']
states = jnp.where(states>=0.6, 2, states) #activate if passive, more accurate
states = jnp.where(jnp.logical_and(states>0,states<0.6), 1, states) #weight if signal

plt.imshow(states)
plt.title("Initial States (Thresholded)")
plt.savefig("vnn_mlp_classify_states_init_thres.png", dpi=75)
plt.show()

#loss function
def loss(params: hk.Params, state, batch, labels, l2_scaling=1e-3) -> jnp.ndarray:
    '''
    Cross entropy 
    '''
    labels = jax.nn.one_hot(labels, num_classes)

    #CE loss
    logits, state = network.apply(params, state, batch)
    logits = jnp.squeeze(logits, axis=-1) #remove extra dim
    # print(logits.shape, labels.shape)
    ce = -1 * jnp.mean(labels * jnp.log(logits + epsilon) + (1 - labels) * jnp.log(1. - logits + epsilon))

    return ce, state

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
    loss_value = 0
    for x, y in zip(X, Y):
        x = x[:, :, jnp.newaxis] #fake 2D
        # y = y[:, :, jnp.newaxis] #fake 2D
        params, opt_state, value, state = update(params, opt_state, state, x, y)
        loss_value += value
        # break

    if step % 10 == 0:
        Y_pred = []
        for x, y in zip(X, Y):
            x = x[:, :, jnp.newaxis] #fake 2D
            logits, state_test = network.apply(params, state, x)
            predictions = jnp.squeeze(logits, axis=-1) #remove extra dim
            Y_pred.append(predictions)
        Y_pred = jnp.argmax(jnp.array(Y_pred), axis=-1)
        # Y_pred = Y_pred[:, jnp.newaxis] #fake 2D
        accuracy = jnp.mean(Y_pred == Y)
        print(f"[Step {step}, Loss {loss_value}, Accuracy {accuracy}]")
    else:
        print(f"[Step {step}, Loss {loss_value}]")
end = time.time()
elapsed = end - start
print("Training took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total") 

jnp.savez("mlp_classify_cells.npz", 
            weights=params['universe_lp']['w'], 
            biases=params['universe_lp']['b'], 
            filters=params['universe_lp']['f'], 
            states=params['universe_lp']['s'], 
            N=N, length=length, num_classes=num_classes, input_size=input_size, output_size=output_size,
            input_offset=input_offset, output_offset=output_offset,
            kernel_size=kernel_size,
            allow_pickle=False)

plt.imshow(params['universe_lp']['w'])
plt.title("Final Weights")
plt.savefig("vnn_mlp_classify_weights.png", dpi=150)
plt.show()

plt.imshow(params['universe_lp']['b'])
plt.title("Final Biases")
plt.savefig("vnn_mlp_classify_biases.png", dpi=150)
plt.show()

plt.imshow(params['universe_lp']['f'][0,:,:,0])
plt.title("Final Filter")
plt.savefig("vnn_mlp_classify_filter.png", dpi=75)
plt.show()

plt.imshow(params['universe_lp']['s'])
plt.title("Final States")
plt.savefig("vnn_mlp_classify_states_final.png", dpi=75)
plt.show()

states = params['universe_lp']['s']
states = jnp.where(states>=0.6, 2, states) #activate if passive, more accurate
states = jnp.where(jnp.logical_and(states>0,states<0.6), 1, states) #weight if signal

plt.imshow(states)
plt.title("Final States (Thresdolded)")
plt.savefig("vnn_mlp_classify_states_final_thres.png", dpi=75)
plt.show()

#function plot
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
Y_pred = []
for x, y in zip(X, Y):
    x = x[:, :, jnp.newaxis] #fake 2D
    logits, state_test = network.apply(params, state, x)
    Y_pred.append(jnp.squeeze(logits, axis=-1))
Y_pred = jnp.argmax(jnp.array(Y_pred), axis=-1)
print("After", Y_pred.shape, Y_pred)
accuracy = jnp.mean(Y_pred == Y)
print(f"[Final Accuracy {accuracy}]")
for x, y in zip(X, Y_pred):
    ax.scatter(x[:, 0], x[:, 1], x[:, 2], marker="o", c=y, s=25, edgecolor="k")
plt.savefig("vnn_mlp_classify_plot.png", dpi=150)
plt.show()

print("END")