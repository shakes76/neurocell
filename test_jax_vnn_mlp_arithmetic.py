'''
Von Neumann network that mimics an MLP for arithmetic
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

from sklearn.model_selection import train_test_split

from universe import UniverseLP #local
import data
import utils #local

utils.jax_status()

#parameters
epochs = 200
n = 8
# bit_depth = 2**(n-1)
length = 2*n
samples = 32000
batch_size = 128
kernel_size = (17,3)
num_batches = samples//batch_size
delta = 0.2
padding = 2*max(kernel_size[0], kernel_size[1]) #on one side
layers = 5
N = 2*n + padding
print("N:", N, "Epochs:", epochs)

#===========
#data
X1f, X2f, Y_raw = data.generate_arithmetic(samples, "add", n, batch_size, delta)

#stack
X_preproc = np.concatenate([X1f, X2f], axis=-1)
# print(X_preproc.shape)

X, X_test, Y, Y_test = train_test_split(X_preproc, Y_raw, test_size=0.2, random_state=42)
print("X shape", X.shape, "Y shape:", Y.shape, "Y test shape:", Y_test.shape)
# print("X", X[:1])
# print("Y", Y[:1])

Y_packed = np.packbits(Y,axis=2)
Y_test_packed = np.packbits(Y_test,axis=2)
print("Y_packed shape", Y_packed.shape)
# print("Y_packed", Y_packed[:1])

#locations
#MLP with inputs and outputs, depth layers inbetween
#setup
center = (N//2, N//2)
depth = 2*layers+1 #decision and signal layers, not including output
input_size = (length,1)
input_offset = (center[0]-length//2,center[1]-(depth-1)//2)
output_size = (n,1)
output_offset = (center[0]-output_size[0]//2,input_offset[1]+depth+1) #+1 output layer
activation = jax.nn.tanh
final_activation = None
directions = 8 #how many filters, i.e. possible directions, to learn
print("depth:", depth, "in offset", input_offset, "out offset", output_offset, "directions:", directions)

#===========
##create model
def _forward(batch) -> jnp.ndarray:
    '''
    Forward pass through network
    '''
    mlp = UniverseLP(N, sharpen=False)
    mlp.initialize_mlp(input_size, output_size, input_offset, output_offset, depth, kernel_size, directions=directions)
    return mlp(batch, activation, final_activation, norm=True)

# Make the network and optimizer. Haiku standard
network = hk.without_apply_rng(hk.transform_with_state(_forward))

#optimizer
total_steps = epochs*num_batches
schedule = optax.warmup_cosine_decay_schedule(init_value=0.001,peak_value=0.03,warmup_steps=0.025*total_steps,decay_steps=total_steps,end_value=0.0005)
# opt = optax.sgd(learning_rate=0.005)
# opt = optax.adam(learning_rate=schedule)
opt = optax.adamw(learning_rate=schedule)

# Initialize network and optimiser; note we draw an input to get shapes.
x = X[0]
x = x[:, :, jnp.newaxis] #fake 2D
print("X0 shape:", x.shape)
params, state = network.init(jax.random.PRNGKey(42), x)
opt_state = opt.init(params)
# print(params)
# print(state)

plt.imshow(params['universe_lp']['s'])
plt.title("Initial States")
plt.savefig("vnn_mlp_arith_states_init.png", dpi=75)
# plt.show()

states = params['universe_lp']['s']
states = jnp.where(states>=0.6, 2, states) #activate if passive, more accurate
states = jnp.where(jnp.logical_and(states>0,states<0.6), 1, states) #weight if signal

plt.imshow(states)
plt.title("Initial States (Thresholded)")
plt.savefig("vnn_mlp_arith_states_init_thres.png", dpi=75)
plt.show()

#loss function
def loss(params: hk.Params, state, batch, labels, l2_scaling=1e-3) -> jnp.ndarray:
    '''
    Cross entropy
    '''
    #MSE loss
    logits, state = network.apply(params, state, batch)
    logits = jnp.squeeze(logits, axis=-1) #remove extra dim

    return optax.squared_error(predictions=logits, targets=labels).mean(), state

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
    updates, opt_state = opt.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, loss_value, state

# Train/eval loop.
start = time.time() #time generation
for step in range(1, epochs+1):
    # Do SGD on a batch of training examples.
    loss_value = 0
    for x, y in zip(X, Y):
        x = x[:, :, jnp.newaxis] #fake 2D

        params, opt_state, value, state = update(params, opt_state, state, x, y)
        loss_value += value/batch_size

        # x1 = jnp.round(x[0,:n,0]).astype(np.uint8)
        # x2 = jnp.round(x[0,n:,0]).astype(np.uint8)
        # y = y[0].astype(np.uint8)
        # print(jnp.packbits(x1,axis=0),jnp.packbits(x2,axis=0), jnp.packbits(y,axis=0))
        # break

    if step % 10 == 0:
        Y_pred = []
        done = 0
        for x, y in zip(X, Y):
            x = x[:, :, jnp.newaxis] #fake 2D
            logits, state_test = network.apply(params, state, x)
            predictions = jnp.squeeze(logits, axis=-1) #remove extra dim
            predictions = jnp.round(predictions).astype(np.uint8)
            predictions = jnp.packbits(predictions,axis=1)
            Y_pred.append(predictions)
            if done < 3:
                x1 = jnp.round(x[0,:n]).astype(np.uint8)
                x2 = jnp.round(x[0,n:]).astype(np.uint8)
                y = y[0].astype(np.uint8)
                print(jnp.packbits(x1,axis=0),jnp.packbits(x2,axis=0), predictions[0], jnp.packbits(y,axis=0))
                done += 1
        # Y_pred = jnp.argmax(jnp.array(Y_pred), axis=-1)
        Y_pred = jnp.array(Y_pred)
        # print(Y_pred.shape, Y_packed.shape)
        accuracy = jnp.mean(Y_pred == Y_packed)
        print(f"[Step {step}, Loss {loss_value}, Accuracy {accuracy}]")
    else:
        print(f"[Step {step}, Loss {loss_value}]")
end = time.time()
elapsed = end - start
print("Training took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")

jnp.savez("mlp_arith_cells.npz",
            weights=params['universe_lp']['w'],
            biases=params['universe_lp']['b'],
            filters=params['universe_lp']['f'],
            states=params['universe_lp']['s'],
            N=N, length=length, n=n, input_size=input_size, output_size=output_size,
            input_offset=input_offset, output_offset=output_offset,
            kernel_size=kernel_size,
            allow_pickle=False)

plt.imshow(params['universe_lp']['w'])
plt.title("Final Weights")
plt.savefig("vnn_mlp_arith_weights.png", dpi=150)
plt.show()

plt.imshow(params['universe_lp']['b'])
plt.title("Final Biases")
plt.savefig("vnn_mlp_arith_biases.png", dpi=150)
plt.show()

fig = plt.figure(figsize=(2,2*directions))
for c in range(directions):
    plt.subplot(directions, 1, c+1)
    plt.imshow(params['universe_lp']['f'][0,:,:,c])
    plt.tight_layout()
    plt.axis('off') # Hides both x and y axes
    plt.title("Filter "+str(c))
plt.savefig("vnn_mlp_arith_filter.png", dpi=75)
plt.show()

plt.imshow(params['universe_lp']['s'])
plt.title("Final States")
plt.savefig("vnn_mlp_arith_states_final.png", dpi=75)
plt.show()

states = params['universe_lp']['s']
states = jnp.where(states>=0.6, 2, states) #activate if passive, more accurate
states = jnp.where(jnp.logical_and(states>0,states<0.6), 1, states) #weight if signal

plt.imshow(states)
plt.title("Final States (Thresholded)")
plt.savefig("vnn_mlp_arith_states_final_thres.png", dpi=75)
plt.show()

plt.imshow(state['universe_lp']['universe'][0])
plt.title("Final Universe")
plt.savefig("vnn_mlp_arith_universe.png", dpi=75)
plt.show()

#function plot
Y_pred = []
for x, y in zip(X_test, Y_test):
    x = x[:, :, jnp.newaxis] #fake 2D
    logits, state_test = network.apply(params, state, x)
    predictions = jnp.squeeze(logits, axis=-1) #remove extra dim
    predictions = jnp.round(predictions).astype(np.uint8)
    predictions = jnp.packbits(predictions,axis=1)
    Y_pred.append(predictions)
# Y_pred = jnp.argmax(jnp.array(Y_pred), axis=-1)
# print("After", Y_pred.shape, Y_pred)
Y_pred = jnp.array(Y_pred)
accuracy = jnp.mean(Y_pred == Y_test_packed)
print(f"[Final Accuracy {accuracy}]")

print("END")
