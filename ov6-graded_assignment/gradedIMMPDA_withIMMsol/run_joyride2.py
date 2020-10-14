# %%
import matplotlib.pyplot as plt
import numpy as np
from typing import List

import dynamicmodels
import measurementmodels
import ekf
import imm
import pda
#import load_track_and_plot_joyride as joyride
from gaussparams import GaussParams
from mixturedata import MixtureParameters

import utils

# set styles
try:
    # installed with "pip install SciencePLots" (https://github.com/garrettj403/SciencePlots.git)
    # gives quite nice plots
    plt_styles = ["science", "grid", "bright", "no-latex"]
    plt.style.use(plt_styles)
    print(f"pyplot using style set {plt_styles}")
except Exception as e:
    print(e)
    print("setting grid and only grid and legend manually")
    plt.rcParams.update(
        {
            # setgrid
            "axes.grid": True,
            "grid.linestyle": ":",
            "grid.color": "k",
            "grid.alpha": 0.5,
            "grid.linewidth": 0.5,
            # Legend
            "legend.frameon": True,
            "legend.framealpha": 1.0,
            "legend.fancybox": True,
            "legend.numpoints": 1,
        }
    )

# %% setup and track
# init values
mean_init = np.array([7000, 3500, 0, 0, 0])
cov_init = np.diag([100, 100, 10, 10, 0.1]) ** 2  


# %% PDAF using EKF with CV-model
# sensor
sigma_z = 8
clutter_intensity = 0.5e-4
PD = 0.95
gate_size = 4

# dynamic models
sigma_a_CV = 3

# make model
measurement_model = measurementmodels.CartesianPosition(sigma_z, state_dim=5)
dynamic_model = dynamicmodels.WhitenoiseAccelleration(sigma_a_CV, n=5)
ekf_filter = ekf.EKF(dynamic_model, measurement_model)

tracker = pda.PDA(ekf_filter, clutter_intensity, PD, gate_size)
init_state = tracker.init_filter_state({"mean": mean_init, "cov": cov_init})
#
# track
utils.evaluate_on_joyride(tracker, init_state, modes = ["CV"], prefix="cv")


# %% PDAF using EKF with CT-model
# sensor
sigma_z = 15
clutter_intensity = 0.5e-4
PD = 0.95
gate_size = 4

# dynamic models
sigma_a_CT = 3
sigma_omega = 0.005 * np.pi

# make model
measurement_model = measurementmodels.CartesianPosition(sigma_z, state_dim=5)
dynamic_model = dynamicmodels.ConstantTurnrate(sigma_a_CT, sigma_omega)
ekf_filter = ekf.EKF(dynamic_model, measurement_model)

tracker = pda.PDA(ekf_filter, clutter_intensity, PD, gate_size)

init_state = tracker.init_filter_state({"mean": mean_init, "cov": cov_init})

# track
utils.evaluate_on_joyride(tracker, init_state, modes=["CT"], prefix="ct")



# %% IMM-PDA with CV/CT-models
# sensor
sigma_z = 10
clutter_intensity = 0.5e-4
PD = 0.95
gate_size = 4

# dynamic models
sigma_a_CV = 1
sigma_a_CT = 0.2
sigma_omega = 0.005*np.pi

# markov chain
PI11 = 0.98
PI22 = 0.98

p10 = 0.5  # initvalue for mode probabilities

PI = np.array([[PI11, (1 - PI11)], [(1 - PI22), PI22]])
assert np.allclose(np.sum(PI, axis=1), 1), "rows of PI must sum to 1"

# init values
mode_probabilities_init = np.array([p10, (1 - p10)])
mode_states_init = GaussParams(mean_init, cov_init)
init_imm_state = MixtureParameters(mode_probabilities_init, [mode_states_init] * 2)

assert np.allclose(
    np.sum(mode_probabilities_init), 1
), "initial mode probabilities must sum to 1"

# make model
measurement_model = measurementmodels.CartesianPosition(sigma_z, state_dim=5)
dynamic_models: List[dynamicmodels.DynamicModel] = []
dynamic_models.append(dynamicmodels.WhitenoiseAccelleration(sigma_a_CV, n=5))
dynamic_models.append(dynamicmodels.ConstantTurnrate(sigma_a_CT, sigma_omega))
ekf_filters = []
ekf_filters.append(ekf.EKF(dynamic_models[0], measurement_model))
ekf_filters.append(ekf.EKF(dynamic_models[1], measurement_model))
imm_filter = imm.IMM(ekf_filters, PI)

tracker = pda.PDA(imm_filter, clutter_intensity, PD, gate_size)

utils.evaluate_on_joyride(tracker, init_imm_state, False, 60, 60+10, modes=["CV", "CT"], prefix="cvct")


# %% IMM-PDA with CV/CT/CV-models
# sensor
sigma_z = 10
clutter_intensity = 0.5e-4
PD = 0.95
gate_size = 4

# dynamic models
sigma_a_CV_low = 0.5
sigma_a_CV_high = 5
sigma_a_CT = 0.5

dt = 2.5
sigma_omega = (1/(2*3*dt**2))*np.pi

# markov chain
PI11 = 0.98
PI22 = 0.98
PI33 = 0.98

#PI = np.array([[PI11, (1 - PI11)/2, (1-PI11)/2], [(1 - PI22)/2, PI22, (1-PI22)/2], [(1-PI33)/2, (1-PI33)/2, PI33]])
PI = np.array([
    [PI11, 5*(1 - PI11)/6, (1-PI11)/6],
    [5*(1 - PI22)/6, PI22, (1-PI22)/6], 
    [5*(1-PI33)/6, (1-PI33)/6, PI33]])

assert np.allclose(np.sum(PI, axis=1), 1), "rows of PI must sum to 1"

# init values
mode_probabilities_init = np.array([1/3]*3)
mode_states_init = GaussParams(mean_init, cov_init)
init_imm_state = MixtureParameters(mode_probabilities_init, [mode_states_init] * 3)

assert np.allclose(
    np.sum(mode_probabilities_init), 1
), "initial mode probabilities must sum to 1"


# make model
measurement_model = measurementmodels.CartesianPosition(sigma_z, state_dim=5)
dynamic_models: List[dynamicmodels.DynamicModel] = []
dynamic_models.append(dynamicmodels.WhitenoiseAccelleration(sigma_a_CV_low, n=5))
dynamic_models.append(dynamicmodels.WhitenoiseAccelleration(sigma_a_CV_high, n=5))
dynamic_models.append(dynamicmodels.ConstantTurnrate(sigma_a_CT, sigma_omega))
ekf_filters = []
ekf_filters.append(ekf.EKF(dynamic_models[0], measurement_model))
ekf_filters.append(ekf.EKF(dynamic_models[1], measurement_model))
ekf_filters.append(ekf.EKF(dynamic_models[2], measurement_model))
imm_filter = imm.IMM(ekf_filters, PI)

tracker = pda.PDA(imm_filter, clutter_intensity, PD, gate_size)

utils.evaluate_on_joyride(tracker, init_imm_state, False, 60, 60+1, modes=["CV", "CVhigh", "CT"], prefix="cvcvct")


plt.show()