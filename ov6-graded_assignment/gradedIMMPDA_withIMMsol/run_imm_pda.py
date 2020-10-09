# %% imports
from typing import List

import scipy
import scipy.io
import scipy.stats

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from gaussparams import GaussParams
from mixturedata import MixtureParameters
import dynamicmodels
import measurementmodels
import ekf
import imm
import pda
import estimationstatistics as estats

# %% plot config check and style setup


# to see your plot config
print(f"matplotlib backend: {matplotlib.get_backend()}")
print(f"matplotlib config file: {matplotlib.matplotlib_fname()}")
print(f"matplotlib config dir: {matplotlib.get_configdir()}")
plt.close("all")

# try to set separate window ploting
if "inline" in matplotlib.get_backend():
    print("Plotting is set to inline at the moment:", end=" ")

    if "ipykernel" in matplotlib.get_backend():
        print("backend is ipykernel (IPython?)")
        print("Trying to set backend to separate window:", end=" ")
        import IPython

        IPython.get_ipython().run_line_magic("matplotlib", "")
    else:
        print("unknown inline backend")

print("continuing with this plotting backend", end="\n\n\n")


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


# %% load data and plot
filename_to_load = "data_for_imm_pda.mat"
loaded_data = scipy.io.loadmat(filename_to_load)
K = loaded_data["K"].item()
Ts = loaded_data["Ts"].item()
Xgt = loaded_data["Xgt"].T
Z = [zk.T for zk in loaded_data["Z"].ravel()]
true_association = loaded_data["a"].ravel()

# plot measurements close to the trajectory
fig1, ax1 = plt.subplots(num=1, clear=True)

Z_plot_data = np.empty((0, 2), dtype=float)
plot_measurement_distance = 45
for Zk, xgtk in zip(Z, Xgt):
    to_plot = np.linalg.norm(Zk - xgtk[None:2], axis=1) <= plot_measurement_distance
    Z_plot_data = np.append(Z_plot_data, Zk[to_plot], axis=0)

ax1.scatter(*Z_plot_data.T, s=5, color="C1")
ax1.plot(*Xgt.T[:2], color="C0", linewidth=1.5)
ax1.set_title("True trajectory and the nearby measurements")
plt.show(block=False)

# %% play measurement movie. Remember that you can cross out the window
play_movie = False
play_slice = slice(0, K)
if play_movie:
    if "inline" in matplotlib.get_backend():
        print("the movie might not play with inline plots")
    fig2, ax2 = plt.subplots(num=2, clear=True)
    sh = ax2.scatter(np.nan, np.nan, s=5)
    th = ax2.set_title(f"measurements at step 0")
    mins = np.vstack(Z).min(axis=0)
    maxes = np.vstack(Z).max(axis=0)
    ax2.axis([mins[0], maxes[0], mins[1], maxes[1]])
    plotpause = 0.1
    # sets a pause in between time steps if it goes to fast
    for k, Zk in enumerate(Z[play_slice]):
        sh.set_offsets(Zk)
        th.set_text(f"measurements at step {k}")
        fig2.canvas.draw_idle()
        plt.show(block=False)
        plt.pause(plotpause)


# %% IMM-PDA

# THE PRESET PARAMETERS AND INITIAL VALUES WILL CAUSE TRACK LOSS!
# Some reasoning and previous exercises should let you avoid track loss.
# No exceptions should be generated if PDA works correctly with IMM,
# but no exceptions do not guarantee correct implementation.

# sensor
sigma_z = 10
clutter_intensity = 1e-2
PD = 0.8
gate_size = 5

# dynamic models
sigma_a_CV = 0.5
sigma_a_CT = 0.5
sigma_omega = 0.3


# markov chain
PI11 = 0.9
PI22 = 0.9

p10 = 0.9  # initvalue for mode probabilities

PI = np.array([[PI11, (1 - PI11)], [(1 - PI22), PI22]])
assert np.allclose(np.sum(PI, axis=1), 1), "rows of PI must sum to 1"

mean_init = np.array([0, 0, 0, 0, 0])
cov_init = np.diag([1000, 1000, 30, 30, 0.1]) ** 2  # THIS WILL NOT BE GOOD
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

# init_imm_pda_state = tracker.init_filter_state(init__immstate)


NEES = np.zeros(K)
NEESpos = np.zeros(K)
NEESvel = np.zeros(K)

tracker_update = init_imm_state
tracker_update_list = []
tracker_predict_list = []
tracker_estimate_list = []
# estimate
for k, (Zk, x_true_k) in enumerate(zip(Z, Xgt)):
    tracker_predict = tracker.predict(tracker_update, Ts)
    tracker_update = tracker.update(Zk, tracker_predict)

    # You can look at the prediction estimate as well
    tracker_estimate = tracker.estimate(tracker_update)

    NEES[k] = estats.NEES(*tracker_estimate, x_true_k, idxs=np.arange(4))
    NEESpos[k] = estats.NEES(*tracker_estimate, x_true_k, idxs=np.arange(2))
    NEESvel[k] = estats.NEES(*tracker_estimate, x_true_k, idxs=np.arange(2, 4))

    tracker_predict_list.append(tracker_predict)
    tracker_update_list.append(tracker_update)
    tracker_estimate_list.append(tracker_estimate)


x_hat = np.array([est.mean for est in tracker_estimate_list])
prob_hat = np.array([upd.weights for upd in tracker_update_list])

# calculate a performance metrics
poserr = np.linalg.norm(x_hat[:, :2] - Xgt[:, :2], axis=0)
velerr = np.linalg.norm(x_hat[:, 2:4] - Xgt[:, 2:4], axis=0)
posRMSE = np.sqrt(
    np.mean(poserr ** 2)
)  # not true RMSE (which is over monte carlo simulations)
velRMSE = np.sqrt(np.mean(velerr ** 2))
# not true RMSE (which is over monte carlo simulations)
peak_pos_deviation = poserr.max()
peak_vel_deviation = velerr.max()


# consistency
confprob = 0.9
CI2 = np.array(scipy.stats.chi2.interval(confprob, 2))
CI4 = np.array(scipy.stats.chi2.interval(confprob, 4))

confprob = confprob
CI2K = np.array(scipy.stats.chi2.interval(confprob, 2 * K)) / K
CI4K = np.array(scipy.stats.chi2.interval(confprob, 4 * K)) / K
ANEESpos = np.mean(NEESpos)
ANEESvel = np.mean(NEESvel)
ANEES = np.mean(NEES)

# %% plots
# trajectory
fig3, axs3 = plt.subplots(1, 2, num=3, clear=True)
axs3[0].plot(*x_hat.T[:2], label=r"$\hat x$")
axs3[0].plot(*Xgt.T[:2], label="$x$")
axs3[0].set_title(
    f"RMSE(pos, vel) = ({posRMSE:.3f}, {velRMSE:.3f})\npeak_dev(pos, vel) = ({peak_pos_deviation:.3f}, {peak_vel_deviation:.3f})"
)
axs3[0].axis("equal")
# probabilities
axs3[1].plot(np.arange(K) * Ts, prob_hat)
axs3[1].set_ylim([0, 1])
axs3[1].set_ylabel("mode probability")
axs3[1].set_xlabel("time")

# NEES
fig4, axs4 = plt.subplots(3, sharex=True, num=4, clear=True)
axs4[0].plot(np.arange(K) * Ts, NEESpos)
axs4[0].plot([0, (K - 1) * Ts], np.repeat(CI2[None], 2, 0), "--r")
axs4[0].set_ylabel("NEES pos")
inCIpos = np.mean((CI2[0] <= NEESpos) * (NEESpos <= CI2[1]))
axs4[0].set_title(f"{inCIpos*100:.1f}% inside {confprob*100:.1f}% CI")

axs4[1].plot(np.arange(K) * Ts, NEESvel)
axs4[1].plot([0, (K - 1) * Ts], np.repeat(CI2[None], 2, 0), "--r")
axs4[1].set_ylabel("NEES vel")
inCIvel = np.mean((CI2[0] <= NEESvel) * (NEESvel <= CI2[1]))
axs4[1].set_title(f"{inCIvel*100:.1f}% inside {confprob*100:.1f}% CI")

axs4[2].plot(np.arange(K) * Ts, NEES)
axs4[2].plot([0, (K - 1) * Ts], np.repeat(CI4[None], 2, 0), "--r")
axs4[2].set_ylabel("NEES")
inCI = np.mean((CI2[0] <= NEES) * (NEES <= CI2[1]))
axs4[2].set_title(f"{inCI*100:.1f}% inside {confprob*100:.1f}% CI")

print(f"ANEESpos = {ANEESpos:.2f} with CI = [{CI2K[0]:.2f}, {CI2K[1]:.2f}]")
print(f"ANEESvel = {ANEESvel:.2f} with CI = [{CI2K[0]:.2f}, {CI2K[1]:.2f}]")
print(f"ANEES = {ANEES:.2f} with CI = [{CI4K[0]:.2f}, {CI4K[1]:.2f}]")

# errors
fig5, axs5 = plt.subplots(2, num=5, clear=True)
axs5[0].plot(np.arange(K) * Ts, np.linalg.norm(x_hat[:, :2] - Xgt[:, :2], axis=1))
axs5[0].set_ylabel("position error")

axs5[1].plot(np.arange(K) * Ts, np.linalg.norm(x_hat[:, 2:4] - Xgt[:, 2:4], axis=1))
axs5[1].set_ylabel("velocity error")

plt.show()

# %% TBD: estimation "movie"
# def plot_cov_ellipse2d(
#     ax: plt.Axes,
#     mean: np.ndarray = np.zeros(2),
#     cov: np.ndarray = np.eye(2),
#     n_sigma: float = 1,
#     *,
#     edgecolor: "Color" = "C0",
#     facecolor: "Color" = "none",
#     **kwargs,  # extra Ellipse keyword arguments
# ) -> matplotlib.patches.Ellipse:
#     """Plot a n_sigma covariance ellipse centered in mean into ax."""
#     ell_trans_mat = np.zeros((3, 3))
#     ell_trans_mat[:2, :2] = np.linalg.cholesky(cov)
#     ell_trans_mat[:2, 2] = mean
#     ell_trans_mat[2, 2] = 1

#     ell = matplotlib.patches.Ellipse(
#         (0.0, 0.0),
#         2.0 * n_sigma,
#         2.0 * n_sigma,
#         edgecolor=edgecolor,
#         facecolor=facecolor,
#         **kwargs,
#     )
#     trans = matplotlib.transforms.Affine2D(ell_trans_mat)
#     ell.set_transform(trans + ax.transData)
#     return ax.add_patch(ell)


# play_estimation_movie = False
# mTL = 0.2  # maximum transparancy (between 0 and 1);
# plot_pause = 1  # lenght to pause between time steps;
# start_k = 1
# end_k = 10
# plot_range = slice(start_k, end_k)  # the range to go through

# # %k = 31; assert(all([k > 1, k <= K]), 'K must be in proper range')
# fig6, axs6 = plt.subplots(1, 2, num=6, clear=True)
# mode_lines = [axs6[0].plot(np.nan, np.nan, color=f"C{s}")[0] for s in range(2)]
# meas_sc = axs6[0].scatter(np.nan, np.nan, color="r", marker="x")
# meas_sc_true = axs6[0].scatter(np.nan, np.nan, color="g", marker="x")
# min_ax = np.vstack(Z).min(axis=0)  # min(cell2mat(Z'));
# max_ax = np.vstack(Z).max(axis=0)  # max(cell2mat(Z'));
# axs6[0].axis([min_ax[0], max_ax[0], min_ax[1], max_ax[0]])

# for k, (Zk, pred_k, upd_k, ak) in enumerate(
#     zip(
#         Z[plot_range],
#         tracker_predict_list[plot_range],
#         tracker_update_list[plot_range],
#         true_association[plot_range],
#     ),
#     start_k,
# ):
#     # k, (Zk, pred_k, upd_k, ak) = data
#     (ax.cla() for ax in axs6)
#     pl = []
#     gated = tracker.gate(Zk, pred_k)  # probbar(:, k), xbar(:, :, k), Pbar(:, :, :, k));
#     minG = 1e20 * np.ones(2)
#     maxG = np.zeros(2)
#     cond_upd_k = tracker.conditional_update(Zk[gated], pred_k)
#     beta_k = tracker.association_probabilities(Zk[gated], pred_k)
#     for s in range(2):
#         mode_lines[s].set_data = (
#             np.array([u.components[s].mean[:2] for u in tracker_update_list[:k]]).T,
#         )
#         axs6[1].plot(prob_hat[: (k - 1), s], color=f"C{s}")
#         for j, cuj in enumerate(cond_upd_k):
#             alpha = 0.7 * beta_k[j] * cuj.weights[s] + 0.3
#             # csj = mTL * co(s, :) + (1 - mTL) * (beta(j)*skupd(s, j)*co(s, :) + (1 - beta(j)*skupd(s, j)) * ones(1, 3)); % transparancy
#             upd_km1_s = tracker_update_list[k - 1].components[s]
#             pl.append(
#                 axs6[0].plot(
#                     [upd_km1_s.mean[0], cuj.components[s].mean[0]],
#                     [upd_km1_s.mean[1], cuj.components[s].mean[1]],
#                     "--",
#                     color=f"C{s}",
#                     alpha=alpha,
#                 )
#             )

#             pl.append(
#                 axs6[1].plot(
#                     [k - 1, k],
#                     [prob_hat[k - 1, s], cuj.weights[s]],
#                     color=f"C{s}",
#                     alpha=alpha,
#                 )
#             )
#             # axis([minAx(1), maxAx(1), minAx(2), maxAx(2)])
#             #%alpha(pl, beta(j)*skupd(s, j));
#             # drawnow;
#             pl.append(
#                 plot_cov_ellipse2d(
#                     axs6[0],
#                     cuj.components[s].mean[:2],
#                     cuj.components[s].cov[:2, :2],
#                     edgecolor=f"C{s}",
#                     alpha=alpha,
#                 )
#             )

#         Sk = imm_filter.filters[s].innovation_cov([0, 0], pred_k.components[s])
#         # gateData = chol(Sk)' * [cos(thetas); sin(thetas)] * sqrt(tracker.gateSize) + squeeze(xbar(1:2, s, k));
#         # plot(gateData(1, :),gateData(2, :), '.--', 'Color', co(s,:))
#         pl.append(
#             plot_cov_ellipse2d(
#                 axs6[0],
#                 pred_k.components[s].mean[:2],
#                 Sk,
#                 n_sigma=tracker.gate_size,
#                 edgecolor=f"C{s}",
#             )
#         )
#         meas_sc.set_offsets(Zk)
#         pl.append(axs6[0].scatter(*Zk.T, color="r", marker="x"))
#         if ak > 0:
#             meas_sc_true.set_offsets(Zk[ak - 1])
#         else:
#             meas_sc_true.set_offsets(np.array([np.nan, np.nan]))

#         # for j = 1:size(xkupd, 3)
#         #     csj = mTL * co(s, :) + (1 - mTL) * (beta(j)*skupd(s, j)*co(s, :) + (1 - beta(j)*skupd(s, j)) * ones(1, 3)); % transparancy
#         #     plot([k-1, k], [probhat(s, k-1), skupd(s, j)], '--', 'color', csj)

#         # minGs = min(gateData, [], 2);
#         # minG = minGs .* (minGs < minG) + minG .* (minG < minGs);
#         # maxGs = max(gateData, [], 2);
#         # maxG = maxGs .* (maxGs > maxG) + maxG .* (maxG > maxGs);

#     # scale = 1
#     # minAx = minG - scale * (maxG - minG);
#     # maxAx = maxG + scale * (maxG - minG);
#     # axis([minAx(1), maxAx(1), minAx(2), maxAx(2)])
#     # %legend()

#     # mode probabilities

#     # axis([1, plotRange(end), 0, 1])
#     # drawnow;
#     plt.pause(plot_pause)

# %%
