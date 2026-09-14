# Generate MUAPs
from gen_muaps import gen_muaps
from MEPmodel_bio import cal_error

a = # pygpc parameter
b = # pygpc parameter
lam = # pygpc parameter
delay = 2.5 * lam # fixed parameter

muaps, tmuap = gen_muaps(n_neurons=100, amplitude=[a, b], axonalDelay=delay, lam=lam)

# Load previously saved MUAP spike times

# Run spinal model

ref = cal_error(ref, sim)

# Output metric
R2 =  ref["R2"]
print(R2)