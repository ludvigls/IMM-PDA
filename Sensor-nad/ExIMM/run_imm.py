# %% imports
import numpy as np
import scipy
import scipy.io
import scipy.stats
import matplotlib
import matplotlib.pyplot as plt


import dynamicmodels
import measurementmodels
import imm
import ekf

# %% plot config check and style setup

# to see your plot config
print(f"matplotlib backend: {matplotlib.get_backend()}")
print(f"matplotlib config file: {matplotlib.matplotlib_fname()}")
print(f"matplotlib config dir: {matplotlib.get_configdir()}")
plt.close("all")

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

# %% load data
use_pregen = True
# you can generate your own data if set to false
if use_pregen:
    data_filename = "data_for_imm.mat"
    loaded_data = scipy.io.loadmat(data_filename)
    Z = loaded_data["Z"].T
    K = loaded_data["K"].item()
    Ts = loaded_data["Ts"].item()
    Xgt = loaded_data["Xgt"].T
else:
    K = 100
    Ts = 2.5
    sigma_z = 2.25
    sigma_a = 0.7  # effective all the time
    sigma_omega = 5e-4 * np.pi  # effective only in turns

    init_x = [0, 0, 2, 0, 0]
    init_P = np.diag([25, 25, 3, 3, 0.0005]) ** 2
    # [Xgt, Z] = simulate_atc(q, r, K, init, false);
    raise NotImplementedError


fig1, ax1 = plt.subplots(num=1, clear=True)
ax1.plot(*Xgt.T[:2])
ax1.scatter(*Z.T[:2])


# %% tune single filters
# TODO
sigma_z = 1
sigma_a_CT = 1
sigma_omega = 1

# TODO
init_state = {"mean": [None], "cov": np.diag([None]) ** 2}

measurement_model = measurementmodels.CartesianPosition(
    sigma_z, state_dim=5
)  # note the 5
CV = dynamicmodels.WhitenoiseAccelleration(sigma_a_CV, n=5)  # note the 5
CT = dynamicmodels.ConstantTurnrate(sigma_a_CT, sigma_omega)

# make models
filters = []
filters.append(ekf.EKF(CV, measurement_model))
filters.append(None) # TODO, so you understand what is going on here

pred = []
upd = []
stats = []
for ekf_filter in filters:
    ekfpred_list, ekfupd_list = # TODO
    stats.append(
        ekf_filter.performance_stats_sequence(
            K=K,
            Z=Z,
            ekfpred_list=ekfpred_list,
            ekfupd_list=ekfupd_list,
            X_true=Xgt,
            NEES_idx=ekf_filter.dynamic_model._all_idx,  # HACK?
            norm_idxs=[[0, 1], [2, 3]],
            norms=[2, 2],
        )
    )
    pred.append(ekfpred_list)
    upd.append(ekfupd_list)

print(stats[0].dtype.names)

# errors
err_pred = np.array([st["dists_pred"] for st in stats])
err_upd = np.array([st["dists_upd"] for st in stats])
RMSE_pred = np.sqrt((err_pred ** 2).mean(axis=1)) # not RMSE over monte carlo trials but time
RMSE_upd = np.sqrt((err_upd ** 2).mean(axis=1)) # same as above


# measurement consistency
NIS = np.array([st["NIS"] for st in stats])
ANIS = # TODO, hint mean
CINIS = # TODO, hint scipy.stats.chi2.interval
CIANIS = # TODO, hint [...].inteval
print(f"ANIS={ANIS} with CIANIS={CIANIS}")


# plot
fig2, axs2 = plt.subplots(2, 2, num=2, clear=True)
for axu, axl, u_s, rmse_pred, rmse_upd in zip(
    axs2[0], axs2[1], upd, RMSE_pred, RMSE_upd
):
    # ax.scatter(*Z.T)
    axu.plot(*u_s.mean.T[:2])
    rmsestr = ", ".join(f"{num:.3f}" for num in (*rmse_upd, *rmse_pred))
    axu.set_title(f"RMSE(p_u, v_u, p_pr, v_pr)|\n{rmsestr}|")
    axu.axis("equal")
    axl.plot(np.arange(K) * Ts, u_s.mean.T[4])
    axl.plot(np.arange(K) * Ts, Xgt[:, 4])

axs2[1, 0].set_ylabel(r"$\omega$")

fig3, axs3 = plt.subplots(1, 3, num=3, clear=True)

axs3[0].plot(np.arange(K) * Ts, NIS[0])
axs3[0].plot(np.arange(K) * Ts, NIS[1])
axs3[0].set_title("NIS")

axs3[1].plot(np.arange(K) * Ts, err_upd[0, :, 0])
axs3[1].plot(np.arange(K) * Ts, err_upd[1, :, 0])
axs3[1].set_title("pos error")

axs3[2].plot(np.arange(K) * Ts, err_upd[0, :, 1])
axs3[2].plot(np.arange(K) * Ts, err_upd[1, :, 1])
axs3[2].set_title("vel error")

# %% tune IMM by only looking at the measurements
# TODO
sigma_z = 1
sigma_a_CV = 1
sigma_a_CT = 1
sigma_omega = 1
PI = np.array([[1, 0], [0, 1]])  # TODO
# Optional sanity check
assert np.allclose(PI.sum(axis=1), 1), "rows of PI must sum to 1" # RIGHT?? yes...

# make model
measurement_model = measurementmodels.CartesianPosition(sigma_z, state_dim=5)
CV = dynamicmodels.WhitenoiseAccelleration(sigma_a_CV, n=5)
CT = dynamicmodels.ConstantTurnrate(sigma_a_CT, sigma_omega)
ekf_filters = []
# TODO fill in you filters
# ekf_filters.append(...)
imm_filter = None # TODO

init_state = { # TODO
    "weight": np.array([0.5, 0.5]),  # Mode probs: optional
    "mean": [0, 0, 0, 0, 0],
    "cov": np.diag([1, 1, 1, 1, 1]) ** 2,
}


imm_preds, imm_upds, imm_ests = # TODO: perform estimate

# extract some data
x_est = np.array([est.mean for est in imm_ests])
prob_est = np.array([upds.weights for upds in imm_upds])

# consistency
NISes_comb = [imm_filter.NISes(zk, pred_k) for zk, pred_k in zip(Z, imm_preds)]
NIS = np.array([n[0] for n in NISes_comb])
NISes = np.array([n[1] for n in NISes_comb])
ANIS = # TODO
CINIS = # TODO
CIANIS = # TODO
print(f"ANIS={ANIS} with CIANIS={CIANIS}")

# plot
fig4, axs4 = plt.subplots(2, 2, num=4, clear=True)
axs4[0, 0].plot(*x_est.T[:2], label="est", color="C0")
axs4[0, 0].scatter(*Z.T, label="z", color="C1")
axs4[0, 0].legend()
axs4[0, 1].plot(np.arange(K) * Ts, x_est[:, 4], label=r"$\omega$")
axs4[0, 1].legend()
axs4[1, 0].plot(np.arange(K) * Ts, prob_est, label=r"$Pr(s)$")
axs4[1, 0].legend()
axs4[1, 1].plot(np.arange(K) * Ts, NIS, label="NIS")
axs4[1, 1].plot(np.arange(K) * Ts, NISes)

ratio_in_CI = np.sum(np.less_equal(CINIS[0], NIS) * np.less_equal(NIS, CINIS[1])) / K
CI_LABELS = ["CI0", "CI1"]
for ci, cilbl in zip(CINIS, CI_LABELS):
    axs4[1, 1].plot([1, K * Ts], np.ones(2) * ci, "--r", label=cilbl)
axs4[1, 1].text(K * Ts * 1.1, 1, f"{ratio_in_CI} inside CI", rotation=90)
axs4[1, 1].legend()

# %% tune IMM by looking at ground truth
# TODO
sigma_z = 1
sigma_a_CV = 1
sigma_a_CT = 1
sigma_omega = 1
PI = np.array([[1, 0], [0, 1]])
# Optional sanity check
assert np.allclose(PI.sum(axis=1), 1), "rows of PI must sum to 1"

# make model
measurement_model = measurementmodels.CartesianPosition(sigma_z, state_dim=5)
CV = dynamicmodels.WhitenoiseAccelleration(sigma_a_CV, n=5)
CT = dynamicmodels.ConstantTurnrate(sigma_a_CT, sigma_omega)
ekf_filters = []
# TODO: add and create the filters
# ekf_filters.append ...
imm_filter = None # TODO

init_state = { # TODO, pick something reasonoable
    "weight": np.array([0.5, 0.5]),  # Mode probs: optional
    "mean": [0, 0, 0, 0, 0],
    "cov": np.diag([1, 1, 1, 1, 1]) ** 2,
}


imm_preds, imm_upds, imm_ests = # TODO

# extract some data
x_est = np.array([est.mean for est in imm_ests])
mode_prob = np.array([upds.weights for upds in imm_upds])

# consistency: NIS
NISes_comb = (imm_filter.NISes(zk, pred_k) for zk, pred_k in zip(Z, imm_preds))
NIS, NISes = [np.array(n) for n in zip(*NISes_comb)]
ANIS = # TODO
CINIS = # TODO
CIANIS = # TODO

# consistency: NEES
NEESes_comb_pred = (
    imm_filter.NEESes(pred_k, x_true, idx=np.arange(4))
    for pred_k, x_true in zip(imm_preds, Xgt)
)
NEES_pred, NEESes_pred = (np.array(n) for n in zip(*NEESes_comb_pred))
ANEES_pred = # TODO
NEESes_comb_upd = (
    imm_filter.NEESes(upd_k, x_true, idx=np.arange(4))
    for upd_k, x_true in zip(imm_upds, Xgt)
)
NEES_upd, NEESes_upd = (np.array(n) for n in zip(*NEESes_comb_upd))
ANEES_upd = # TODO
CINEES = # TODO
CIANEES = # TODO
print(f"ANIS={ANIS} and CIANEES={CIANIS}")
print(f"ANEES_upd={ANEES_upd}, and CIANEES={CIANEES}")

#  errors
pos_err = np.sqrt(np.sum((x_est[:, :2] - Xgt[:, :2]) ** 2, axis=1))
vel_err = np.sqrt(np.sum((x_est[:, 2:4] - Xgt[:, 2:4]) ** 2, axis=1))
pos_RMSE = np.sqrt(
    np.mean(pos_err ** 2)
)  # not true RMSE (which is over monte carlo simulations)
vel_RMSE = np.sqrt(
    np.mean(vel_err ** 2)
)  # not true RMSE (which is over monte carlo simulations)
pos_peak_deviation = # TODO
vel_peak_deviation = # TODO

# plot
rmsestr = ", ".join(f"{num:.3f}" for num in (pos_RMSE, vel_RMSE))
devstr = ", ".join(f"{num:.3f}" for num in (pos_peak_deviation, vel_peak_deviation))

fig5, axs5 = plt.subplots(2, 2, num=5, clear=True)
axs5[0, 0].plot(*x_est.T[:2], label="est", color="C0")
axs5[0, 0].scatter(*Z.T, label="z", color="C1")
axs5[0, 0].legend()
axs5[0, 0].set_title(f"RMSE(p, v) = {rmsestr}\npeak_dev(p, v) = {devstr}.0")
axs5[0, 1].plot(np.arange(K) * Ts, x_est[:, 4], label=r"$\hat{\omega}$")
axs5[0, 1].plot(np.arange(K) * Ts, Xgt[:, 4], label=r"$\omega$")
axs5[0, 1].legend()
for s in range(len(ekf_filters)):
    axs5[1, 0].plot(np.arange(K) * Ts, prob_est[:, s], label=rf"$Pr(s={s})$")
axs5[1, 0].legend()
axs5[1, 1].plot(np.arange(K) * Ts, NIS, label="NIS")
axs5[1, 1].plot(np.arange(K) * Ts, NISes)

ratio_in_CI = np.sum(np.less_equal(CINIS[0], NIS) * np.less_equal(NIS, CINIS[1])) / K
CI_LABELS = ["CI0", "CI1"]
for ci, cilbl in zip(CINIS, CI_LABELS):
    axs5[1, 1].plot([1, K * Ts], np.ones(2) * ci, "--r", label=cilbl)
axs5[1, 1].text(K * Ts * 1.1, 1, f"{ratio_in_CI} inside CI", rotation=90)
axs5[1, 1].legend()

fig6, axs6 = plt.subplots(2, 2, sharex=True, num=6, clear=True)
axs6[0, 0].plot(np.arange(K) * Ts, pos_err)
axs6[0, 0].set_ylabel("position error")
axs6[0, 1].plot(np.arange(K) * Ts, vel_err)
axs6[0, 1].yaxis.set_label_position("right")
axs6[0, 1].set_ylabel("velocity error")
axs6[1, 0].plot(np.arange(K) * Ts, NIS)
axs6[1, 0].plot(np.arange(K) * Ts, NISes)
ratio_in_CI = np.mean(np.less_equal(CINIS[0], NIS) * np.less_equal(NIS, CINIS[1]))
axs6[1, 0].set_ylabel(f"NIS: {ratio_in_CI}% in CI")
axs6[1, 0].plot([0, Ts * (K - 1)], np.repeat(CINIS[None], 2, 0), "r--")
# axs6[1, 0].text(K * Ts * 1.1, -2, f"{ratio_in_CI}% inside CI", rotation=90)

axs6[1, 1].plot(np.arange(K) * Ts, NEES_pred)
axs6[1, 1].plot(np.arange(K) * Ts, NEES_upd)
# axs6[1, 1].plot(np.arange(K) * Ts, NISes)
ratio_in_CI_nees = np.mean(
    np.less_equal(CINEES[0], NEES_upd) * np.less_equal(NEES_upd, CINEES[1])
)
# axs6[1, 1].text(K * Ts * 1.1, -2, f"{ratio_in_CI_nees}% inside CI", rotation=90)
axs6[1, 1].yaxis.set_label_position("right")
axs6[1, 1].set_ylabel(f"NEES: {ratio_in_CI_nees}% in CI")

axs6[1, 1].plot([0, Ts * (K - 1)], np.repeat(CINEES[None], 2, 0), "r--")
# axs6[1, 1].text(K * Ts * 1.1, -2, f"{ratio_in_CI_nees}% inside CI", rotation=90)
