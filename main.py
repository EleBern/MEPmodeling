import os
import h5py
import pygpc
from collections import OrderedDict
from muap_gpc_model import MUAP_gpc

fn_results = "pygpc/test"

if os.path.exists(fn_results + ".hdf5"):
    os.remove(fn_results + ".hdf5")

if os.path.exists(fn_results + "_val.hdf5"):
    os.remove(fn_results + "_val.hdf5")

if os.path.exists(fn_results + "_mc.hdf5"):
    os.remove(fn_results + "_mc.hdf5")

# define model
model = MUAP_gpc()

# define problem
parameters = OrderedDict()
# Parameter distributions
# a   = 5      # pygpc parameter [2, 14] uniform
# b   = 100    # pygpc parameter [45, 425] uniform
# lam = 3      # pygpc parameter [1.111 - 5.706], normal distribution mu=3.619, std=0.774
parameters["a"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[2, 14]) # This is a uniform distribution
parameters["b"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[45, 425]) # pdf_limits - sampling range
parameters["delay_a"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[2.35-0.46, 2.35+0.46]) # pygpc.Norm(pdf_shape=[2.35, 0.46]) # Normal distribution. pdf share mu, std 
parameters["delay_b"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[-0.18-1.69, -0.18+1.69]) # pygpc.Norm(pdf_shape=[-0.18, 1.69]) # Normal distribution. pdf share mu, std 

# parameters["lam"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[1.111, 5.706]) 
# parameters["lam"] = pygpc.Norm(pdf_shape=[3.619, 0.774]) # Normal distribution. pdf share mu, std 
parameters["lam"] = 3.619

problem = pygpc.Problem(model, parameters)

# gPC options
options = dict()
options["order"] = [4] * problem.dim
options["order_max"] = 15
options["order_start"] = 2
options["method"] = 'reg'
options["solver"] = "Moore-Penrose"
options["interaction_order"] = 2
options["order_max_norm"] = 1.0
options["n_cpu"] = 0
options["eps"] = 0.1
options["fn_results"] = fn_results
options["basis_increment_strategy"] = None
options["plot_basis"] = False
options["n_grid"] = 1300
options["save_session_format"] = ".hdf5"
options["matrix_ratio"] = 2
options["grid"] = pygpc.LHS
options["error_type"] = "nrmsd"
options["grid_options"] = {"seed": 1, 'criterion': 'ese'}

# define algorithm
algorithm = pygpc.Static(problem=problem, options=options, grid=None)

# Initialize gPC Session
session = pygpc.Session(algorithm=algorithm)

# run gPC session
session, coeffs, results = session.run()


# Load results
with h5py.File(fn_results + ".hdf5", "r") as f:
    coeffs = f["coeffs"][:]
    # filename of associated gPC .pkl files
    fn_session = os.path.join(os.path.split(fn_results)[0], f["misc/fn_session"][0].astype(str))
    fn_session_folder = f["misc/fn_session_folder"][0].astype(str)

# print("> Loading gpc session object: {}".format(fn_session))
session = pygpc.io.read_session(fname=fn_session, folder=fn_session_folder)

# Validation
pygpc.validate_gpc_plot(session=session,
                        coeffs=coeffs,
                        random_vars=["delay_a", "delay_b"],
                        n_grid=[51, 51],
                        output_idx=0,
                        fn_out=session.fn_results + '_val',
                        n_cpu=session.n_cpu)

nrmsd = pygpc.validate_gpc_mc(session=session,
                                coeffs=coeffs,
                                n_samples=int(1e4),
                                output_idx=0,
                                n_cpu=session.n_cpu,
                                fn_out=session.fn_results + '_mc')


# Sensitivity analysis
pygpc.get_sensitivities_hdf5(fn_gpc=session.fn_results,
                                output_idx=None,
                                calc_sobol=True,
                                calc_global_sens=True,
                                calc_pdf=True,
                                n_samples=int(1e4))

sobol, gsens = pygpc.get_sens_summary(fn_results, parameters, fn_results + "_sens_summary.txt")
pygpc.plot_sens_summary(sobol=sobol, gsens=gsens, fn_plot=session.fn_results + "_sens_summary.pdf")