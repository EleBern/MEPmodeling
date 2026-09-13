# Generate MUAPs
from gen_muaps import gen_muaps

a = # pygpc parameter
b = # pygpc parameter
lam = # pygpc parameter
delay = 2.5 * lam # fixed parameter

muaps, tmuap = gen_muaps(n_neurons=100, amplitude=[a, b], axonalDelay=delay, lam=lam)

# Load previously saved MUAP spike times

# Run spinal model