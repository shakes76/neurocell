# Neurocell Library
Cellular machines library for automated systems and self assembly with John von Neumann Networks (VNNs).

## Structure
In general, driver scripts are prefixed with tests_* and unit tests as unit_*.

The various modules are:
* data - for various data creation primitives
* convs - for various convolution and filtering operators
* filters - definitions of very simple filters and kernels
* forward - forward and propagator operators for VNNs
* universe - definition and classes of various different VNNs
* utils - misc helper functions

The descriptions of the various scripts can be found on the [experiments page](experiments.md)

## Implementation
JAX library based implementation of the cellular machines as JAX Numpy arrays

Includes implementation of the Von Neumann Networks and Game of Life (GoL) in JAX.

## Requirements
See the requirements file provided.

### Latest Versions and Environment
The latest versions of JAX, Grain and Haiku work OK as of 06/2026. You can follow the following environment creation steps:
* Create the environment and update PIP
```
python -m venv ~/scratch/envs/jax-grain 
source ~/scratch/envs/jax-grain/bin/activate 
pip install --upgrade pip 
```
* Install JAX for your preferred CUDA or ROCm version, for CUDA 12 which works best for HPCs depending on stable driver versions in 2026
```
pip install --upgrade -U "jax[cuda12]" 
```
* Install other dependencies that include Haiku and Optax. Grain for CIFAR10 scripts, Seaborn/Matplotlib for plotting and Scikit for simple experiments.
```
pip install optax einops flax dm-haiku 
pip install scikit-learn scikit-image seaborn tqdm 
pip install tensorflow_datasets grain opencv-python 
```
* Tensorflow datasets is needed by Grain for CIFAR10, you may encounter importlib error, so install it
```
pip install --upgrade importlib-resources setuptools 
```

### Stable Version
The following setup was needed for Ubuntu/Pop OS! 24.04. Firstly, we needed to install JAX with this particular version found to be stable
```
pip install -U "jax[cuda13]"==0.7.2
```
Then we need to install Haiku and other dependencies with stable versions found where indicated
```
pip install optax einops flax==0.10.7 dm-haiku==0.0.14
pip install scikit-learn scikit-image seaborn
```

## License
This software and notes are made available under the [GNU Lesser General Public License v. 3 (“LGPL”)](https://www.gnu.org/licenses/lgpl-3.0.html). This means you are able to use it as a library, while keeping your source code closed as long as LGPL requirements are met. Changes to the library or notes need to be released under the same license as per the LGPL.

Private licenses are available on request.
