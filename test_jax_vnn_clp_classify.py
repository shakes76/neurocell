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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from universe import UniverseLP #local
import utils #local

utils.jax_status()

#parameters
epochs = 100

#===========
##data
data_dict = datasets.load_wine(return_X_y=False)
X_raw = data_dict['data']
Y_raw = data_dict['target']
length = len(data_dict['feature_names'])
samples = len(Y_raw)
num_classes = len(data_dict['target_names'])
print("Data Shape:", X_raw.shape, "Samples:", samples, "Input Length:", length, "Classes:", num_classes)

#scaling
scaler = StandardScaler()
X_preproc = scaler.fit_transform(X_raw)

X, X_test, Y, Y_test = train_test_split(X_preproc, Y_raw, test_size=0.2, random_state=42)
print("Y shape:", Y.shape, "Y test shape:", Y_test.shape)

#===========
#load network architecture
def circ_architecture(length, num_classes=3, padding=8, spc=2):
    '''
    padding 8 for steps 2, 12 for steps 3
    '''
    radius = jnp.ceil(length*spc/jnp.pi)
    radius_int = int(radius+0.5) #round and pad
    print("radius", radius)

    N = int(2.*radius+padding+0.5)
    print("N:", N)

    offset = [N//2-(radius_int), N//2+radius_int//2]
    slopes = [[0,-spc], [1,-spc], [spc,-1], [spc,0], [spc,1], [1,spc], [0,spc]]
    steps = 2 #3
    #input coords
    input_coords = []
    #draw anti-clockwise
    current = offset
    input_coords.append(current)
    for vec in slopes:
        for step in range(steps):
            new_point = [current[0]+vec[0], current[1]+vec[1]]
            input_coords.append(new_point)
            current = new_point
    final = current

    #output
    input_size = [final[0]-offset[0], final[1]-offset[1]]
    # print("Input Size:", input_size)
    output_offset = (offset[0]+input_size[0]//2-num_classes//2, N//2+radius_int//2)
    print(output_offset)
    slopes = [[1,0]]
    current = output_offset
    output_coords = []
    output_coords.append(current)
    for vec in slopes:
        for step in range(1,num_classes):
            new_point = [current[0]+vec[0], current[1]+vec[1]]
            output_coords.append(new_point)
            current = new_point

    return N, input_coords, output_coords, output_offset

N, input_coords, output_coords, output_offset = circ_architecture(length, num_classes=3, padding=8, spc=2)
output_size = (num_classes,1)

#remove points to match data points
del input_coords[0]
del input_coords[-1]

input_coords = jnp.array(input_coords)
output_coords = jnp.array(output_coords)
print("N:", N, "Epochs:", epochs)

cells = np.zeros((N,N))
for point in input_coords:
    cells[point[0], point[1]] = 128
for point in output_coords:
    cells[point[0], point[1]] = 255

plt.imshow(cells)
plt.title("Initial Architecture")
plt.savefig("vnn_clp_classify_coords.png", dpi=150)
plt.show()

#locations
#MLP with 1 input and 1 output, one layer inbetween
#setup
kernel_size = (3,3)
activation = jax.nn.tanh
final_activation = jax.nn.sigmoid
directions = 3 #how many filters, i.e. possible directions, to learn
#depth
in_row_coord = input_coords[:, 0].min() #top most in
out_row_coord = output_coords[:, 0].min() #top most out
in_col_coord = input_coords[:, 1].min() #far left in
out_col_coord = output_coords[:, 1].max() #far right out
depth_row = out_row_coord - in_row_coord
print("row depth:", depth_row)
depth_col = out_col_coord - in_col_coord
print("col depth:", depth_col)
depth = max(depth_row, depth_col)
print("max depth:", depth)
depth = 10
print("depth:", depth, "kernel:", kernel_size)

#===========
##create model
def _forward(batch) -> jnp.ndarray:
    '''
    Forward pass through network
    '''
    clp = UniverseLP(N, sharpen=True)
    clp.initialize_clp(input_coords, output_coords, output_offset, output_size, [], depth, kernel_size, directions=directions)
    return clp(batch, activation, final_activation, norm=True)

# Make the network and optimizer. Haiku standard
network = hk.without_apply_rng(hk.transform_with_state(_forward))
# opt = optax.sgd(learning_rate=0.005)
opt = optax.adam(learning_rate=0.01)

# Initialize network and optimiser; note we draw an input to get shapes.
params, state = network.init(jax.random.PRNGKey(42), X[0])
opt_state = opt.init(params)
# print(params)
# print(state)

#loss function
def loss(params: hk.Params, state, batch, labels, l2_scaling=1e-3) -> jnp.ndarray:
    '''
    Cross entropy
    '''
    labels = jax.nn.one_hot(labels, num_classes)

    #CE loss
    logits, state = network.apply(params, state, batch)
    logits = jnp.squeeze(logits[0], axis=-1) #remove extra dim
    # ce = -jnp.sum(labels * jax.nn.log_softmax(logits))

    # #L2 norm of weights loss
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
    loss_value = 0
    for x, y in zip(X, Y):
        x = x[jnp.newaxis, :] #fake batch dim
        params, opt_state, value, state = update(params, opt_state, state, x, y)
        loss_value += value
        # break

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
            predicted = jnp.argmax(jnp.array(predicted), axis=-1)
            # print("pred", predicted)
            correct += jnp.sum(predicted == y)
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

jnp.savez("clp_classify_cells.npz",
            weights=params['universe_lp']['w'],
            biases=params['universe_lp']['b'],
            filters=params['universe_lp']['f'],
            states=params['universe_lp']['s'],
            N=N, length=length, num_classes=num_classes, input_coords=input_coords, output_coords=output_coords,
            kernel_size=kernel_size,
            allow_pickle=False)

plt.imshow(params['universe_lp']['w'])
plt.title("Final Weights")
plt.savefig("vnn_clp_classify_weights.png", dpi=150)
plt.show()

plt.imshow(params['universe_lp']['b'])
plt.title("Final Biases")
plt.savefig("vnn_clp_classify_biases.png", dpi=150)
plt.show()

print("filters shape:", params['universe_lp']['f'].shape)
fig = plt.figure(figsize=(4,4))
for c in range(directions):
    plt.subplot(1, directions, c+1)
    plt.imshow(params['universe_lp']['f'][0,:,:,c])
    plt.tight_layout()
    plt.title("Filter "+str(c))
plt.savefig("vnn_clp_classify_filter.png", dpi=75)
plt.show()

plt.imshow(params['universe_lp']['s'])
plt.title("Final States")
plt.savefig("vnn_clp_classify_states.png", dpi=75)
plt.show()

states = params['universe_lp']['s']
states = jnp.where(states>=0.6, 2, states) #activate if passive, more accurate
states = jnp.where(jnp.logical_and(states>0,states<0.6), 1, states) #weight if signal

plt.imshow(states)
plt.title("Final States (Thresholded)")
plt.savefig("vnn_clp_classify_states_thres.png", dpi=75)
plt.show()

plt.imshow(state['universe_lp']['universe'][0])
plt.title("Final Universe")
plt.savefig("vnn_clp_classify_universe.png", dpi=75)
plt.show()

#test performance
correct = 0
total = 0
for x, y in zip(X_test, Y_test):
    x = x[jnp.newaxis, :] #fake batch dim
    # x = jnp.reshape(x, (x.shape[0], length, 1)) #to col vector
    logits, state_test = network.apply(params, state, x)
    predicted = jnp.squeeze(logits[0], axis=-1) #remove extra dim
    predicted = jnp.argmax(jnp.array(predicted), axis=-1)
    correct += jnp.sum(predicted == y)
    total += 1
accuracy = 100 * correct / total
print(f"[Final Test Accuracy {accuracy}]")

print("END")
