'''
Misc data pipelines for training
'''
import numpy as np

def generate_square_pulses(
    length: int = 128,
    pulse_width: int = 8,
    amplitude: float = 1.0,
    batch_size: int = 1,
    noise_level: float = 0.0,
    z_score_standardize: bool = False
) -> np.ndarray:
    """
    Generates a batch of 1D square pulses, each with a random shift.
    Can optionally add noise and standardize each pulse using the z-score.

    Args:
        length (int): The total length of each 1D signal array.
        pulse_width (int): The width of the square pulse in data points.
        amplitude (float): The amplitude (height) of the pulse.
        batch_size (int): The number of signals to generate in the batch.
        noise_level (float): The standard deviation of the Gaussian noise to add.
        z_score_standardize (bool): If True, standardize each signal
                                    to have a mean of 0 and a variance of 1.

    Returns:
        np.ndarray: A 2D NumPy array of shape (batch_size, length).

    Raises:
        ValueError: If pulse_width is greater than the signal length.

    @author Gemini 2.5Pro
    """
    if pulse_width > length:
        raise ValueError("Pulse width cannot be greater than the total signal length.")

    # 1. Create a silent batch of signals (a 2D array of zeros)
    signals = np.zeros((batch_size, length))

    # 2. For each signal in the batch, create a pulse at a random location
    for i in range(batch_size):
        # Determine a random starting point for the pulse in this specific signal
        # The maximum possible start index is length - pulse_width to ensure
        # the pulse fits entirely within the signal array.
        max_start_index = length - pulse_width
        start_index = np.random.randint(0, max_start_index + 1)

        # "Turn on" the pulse for the calculated duration
        end_index = start_index + pulse_width
        signals[i, start_index:end_index] = amplitude

    # 3. Add Gaussian noise if specified
    if noise_level > 0:
        # Generate noise from a normal distribution and scale it by noise_level
        noise = np.random.randn(batch_size, length) * noise_level
        signals += noise

    # 4. Optionally standardize the signals (after adding noise)
    if z_score_standardize:
        # Calculate mean and std for each signal along the length axis (axis=1)
        # keepdims=True ensures the output shape is (batch_size, 1) for broadcasting
        mean = np.mean(signals, axis=1, keepdims=True)
        std = np.std(signals, axis=1, keepdims=True)

        # Add a small epsilon to the standard deviation to avoid division by zero
        # in case a signal is completely flat (has zero variance).
        epsilon = 1e-8
        signals = (signals - mean) / (std + epsilon)

    return signals

def generate_arithmetic(samples, operation: str = "add", n = 8, batch_size = 64, noise_level: float = 0.2):
    """
    Generate data that follows arithmetic operations such as addition, subtraction and exclusive OR

    The pairwise inputs are returned with the outputs. All the arrays returned are in binary.

    n is the bit depth of the numbers involved
    operations supported include "add", "minus" and "xor"
    """
    bit_depth = 2**(n-1)
    num_batches = samples//batch_size
    delta = noise_level

    numbers = np.random.randint(1, bit_depth, size=(num_batches,batch_size,1), dtype=np.uint8)
    operands = np.random.randint(1, bit_depth, size=(num_batches,batch_size,1), dtype=np.uint8)

    #operation
    operation = operation.lower()
    if operation == "minus":
        results = numbers - operands
    elif operation == "xor":
        results = numbers ^ operands
    else:
        results = numbers + operands

    #to bits
    X1 = np.unpackbits(numbers, axis=2, count=n)
    X2 = np.unpackbits(operands, axis=2, count=n)
    Y_raw = np.unpackbits(results, axis=2, count=n)
    # print(X1.shape)

    #add noise
    rand1 = np.random.uniform(-delta,+delta,size=X1.shape)
    rand2 = np.random.uniform(-delta,+delta,size=X2.shape)
    X1f = X1.astype(np.float32) + rand1
    X2f = X2.astype(np.float32) + rand2

    return  X1f, X2f, Y_raw
