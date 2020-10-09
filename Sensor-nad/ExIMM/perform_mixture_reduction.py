# %% imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mixturereduction import gaussian_mixture_moments

# %% setup and show initial
# TODO: fill in values
mus = np.array([1, 1, 1]).reshape(3, 1)
sigmas = np.array([1, 1, 1]).reshape(
    3, 1, 1
)  # note std and not var as in gaussian_mixture_moments
w = np.array([1, 1, 1])
w = w.ravel() / np.sum(w)
assert np.allclose(w.sum(), 1), "weights must sum to one"

totMean, totSigma2 = (
    elem.squeeze() for elem in gaussian_mixture_moments(w, mus, sigmas ** 2)
)
plotNsigmas = 3
x = totMean + plotNsigmas * np.sqrt(totSigma2) * np.arange(-1, 1 + 1e-10, 5e-2)

fig1, ax1 = plt.subplots(num=1, clear=True)
pdf_comp_vals = np.array(
    [
        multivariate_normal.pdf(x, mean=mus[i].item(), cov=sigmas[i].item() ** 2)
        for i in range(len(mus))
    ]
)
pdf_mix_vals = np.average(pdf_comp_vals, axis=0, weights=w)

for i in range(len(mus)):
    ax1.plot(x, pdf_comp_vals[i], label=f"comp {i}")
ax1.legend()

# %% merge and show combinations
fi2, ax2 = plt.subplots(num=2, clear=True)
ax2.plot(x, pdf_mix_vals, label="original")
k = 0
wcomb = np.zeros_like(w)
mucomb = np.zeros_like(w)

sigma2comb = np.zeros_like(w)
pdf_mix_comb_vals = np.zeros_like(pdf_comp_vals)
for i in range(2):  # index of first to merge
    for j in range(i + 1, 3):  # index of second to merge
        k_other = 2 - k  # the index of the non merging (only works for 3 components)

        # merge components
        wcomb[k] = w[i] + w[j]
        mucomb[k], sigma2comb[k] = gaussian_mixture_moments(
            w[[i, j]] / wcomb[k], mus[[i, j]], sigmas[[i, j]] ** 2
        )

        # plot
        pdf_mix_comb_vals[k] = (
            wcomb[k] * multivariate_normal.pdf(x, mucomb[k], sigma2comb[k])
            + w[k_other] * pdf_comp_vals[k_other]
        )
        ax2.plot(x, pdf_mix_comb_vals[k], label=f"combining {i} {j}")
        k += 1

ax2.legend(loc=(1.05, 0))

print(mucomb)
print(sigma2comb)
sigmacomb = np.sqrt(sigma2comb)

# %%
