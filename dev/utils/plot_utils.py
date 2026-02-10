import torch
import torch.nn as nn

import numpy as np

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from matplotlib import cm

from utils.stats_utils import *



def plot_z_samples(latents_list, cfg, event_bin, bin_prd_start, align_event,
                   n_trials_to_plot=5, title=None, titles=["filtering\n", "smoothing\n", "forecasting\n"]):

    assert isinstance(latents_list, (list, tuple)) and len(latents_list) == 3, \
        "Pass a list of 3 latent tensors."

    if titles is None:
        titles = [f"Set {i+1}" for i in range(len(latents_list))]
    else:
        assert len(titles) == len(latents_list), "Length of titles must match number of latent sets."

    latents = latents_list[0]
    np.random.seed(42)
    trial_indcs = np.random.choice(range(0, latents.shape[1]), size=n_trials_to_plot, replace=False)

    n_sets = len(latents_list)
    n_samples, _, n_bins, n_latents = latents.shape

    fig, axs = plt.subplots(len(trial_indcs), n_sets, figsize=(4 * n_sets, 6))
    fig.subplots_adjust(hspace=0, wspace=0.3)

    if n_trials_to_plot == 1:
        axs = np.expand_dims(axs, 0)
    if n_sets == 1:
        axs = np.expand_dims(axs, 1)

    blues = cm.get_cmap("winter", cfg.n_samples)
    reds = cm.get_cmap("summer", cfg.n_samples)
    springs = cm.get_cmap("spring", cfg.n_samples)
    color_map_list = [blues, reds, springs]
    mean_colors = ['navy', 'green', 'coral']

    for i, trial_idx in enumerate(trial_indcs):
        for k, latents in enumerate(latents_list):
            samples = latents[:, trial_idx:trial_idx+1, :, :3].squeeze(1)
            ax = axs[i, k]

            # Plot samples
            for j in range(samples.shape[0]):
                for n in range(samples.shape[2]):
                    ax.plot(np.arange(n_bins) * cfg.bin_sz_ms,
                            samples[j, :, n],
                            color=color_map_list[n](j),
                            linewidth=0.5, alpha=0.4)

            # Plot means
            for n in range(samples.shape[2]):
                ax.plot(np.arange(n_bins) * cfg.bin_sz_ms,
                        samples[:, :, n].mean(axis=0),
                        color=mean_colors[n], linewidth=1.5, alpha=0.8)

            # Event lines
            if k == 2:
                ax.axvline(bin_prd_start * cfg.bin_sz_ms, linestyle='--', color='gold')
            ax.axvline(event_bin * cfg.bin_sz_ms, linestyle='--', color='purple', alpha=0.6)

            # Remove axes for all but bottom row
            if i < n_trials_to_plot - 1:
                ax.axis('off')
            else:
                ax.yaxis.set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.set_xlabel("time (ms)", fontsize=16)

            if k == 0:
                ax.set_ylabel(f"trial {trial_idx+1}", fontsize=16)

            ax.set_xlim(0, n_bins * cfg.bin_sz_ms)

            if i == 0:
                ax.set_title(titles[k], fontsize=24)

    # Add annotation only to top-left plot
    axs[0, 0].annotate(f"{align_event}",
                       xy=(event_bin * cfg.bin_sz_ms, axs[0, 0].get_ylim()[1]),
                       xytext=(event_bin * cfg.bin_sz_ms - (n_bins * cfg.bin_sz_ms * 0.1),
                               axs[0, 0].get_ylim()[1] * 1.1),
                       arrowprops=dict(facecolor='black', alpha=0.4, arrowstyle='->'),
                       fontsize=12, alpha=0.6, ha='center')

    # Add annotation only to top-left plot
    axs[0, 2].annotate(f"prediction\nstarts",
                       xy=(bin_prd_start * cfg.bin_sz_ms, axs[0, 2].get_ylim()[1]),
                       xytext=(bin_prd_start * cfg.bin_sz_ms - (n_bins * cfg.bin_sz_ms * 0.1),
                               axs[0, 0].get_ylim()[1] * 1.1),
                       arrowprops=dict(facecolor='black', alpha=0.4, arrowstyle='->'),
                       fontsize=12, alpha=0.6, ha='center')

    fig.suptitle(f"{title}" if title is not None else "", fontsize=26)
    fig.tight_layout()

    return fig, axs


def plot_z_3d(z, vel):

    z = z.mean(dim=0).cpu()
    trials, time_bins, latents = z.shape
    data_reshaped = z.view(-1, latents).cpu().numpy()

    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(data_reshaped)
    pca_result_reshaped = pca_result.reshape(trials, time_bins, 3)

    fig = plt.figure(figsize=(8, 6))
    fig.suptitle("Top 3 principal components of single-trial\nlatent trajectories", fontsize=24)
    ax = fig.add_subplot(111, projection='3d')

    for i, traj in enumerate(pca_result_reshaped):
        pos = torch.cumsum(vel[i], dim=0).cpu()
        reach_angle = torch.atan2(pos[-1, 0], pos[-1, 1])
        reach_color = plt.cm.hsv(reach_angle / (2 * np.pi) + 0.5)

        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], linewidth=0.8, alpha=0.6, color=reach_color)

        ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], color='red', marker='o', s=10, alpha=0.1, label='start' if i == 0 else "")
        ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], color='gray', marker='x', s=10, alpha=0.6, label='end' if i == 0 else "")

    ax.set_xlabel("PC1", fontsize=16)
    ax.set_ylabel("PC2", fontsize=16)
    ax.set_zlabel("PC3", fontsize=16)
    ax.legend(loc='upper right', fontsize=16)

    fig.tight_layout()
    plt.show()


def plot_z_single_bins(z: torch.Tensor, vel: torch.Tensor,):

    if z.ndim == 4:
        z = z.mean(dim=0)
    trials, time_bins, latents = z.shape

    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(z.view(-1, latents).numpy())

    hand_pos = vel_to_pos(vel)
    angles = torch.atan2(hand_pos[:, :, 0], hand_pos[:, :, 1]).reshape(-1).cpu().numpy()
    colors = plt.cm.hsv(angles / (2 * np.pi) + 0.5)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3), sharex=True, sharey=True)
    fig.suptitle("Single time bins\n", fontsize=20)

    scatter_kwargs = dict(s=2, alpha=0.2, c=colors, cmap='hsv', label='single time bins')

    axes[0].scatter(pca_result[:, 0], pca_result[:, 1], **scatter_kwargs)
    axes[0].set_xlabel("PC1", fontsize=16)
    axes[0].set_ylabel("PC2", fontsize=16)

    axes[1].scatter(pca_result[:, 0], pca_result[:, 2], **scatter_kwargs)
    axes[1].set_xlabel("PC1", fontsize=16)
    axes[1].set_ylabel("PC3", fontsize=16)

    axes[2].scatter(pca_result[:, 1], pca_result[:, 2], **scatter_kwargs)
    axes[2].set_xlabel("PC2", fontsize=16)
    axes[2].set_ylabel("PC3", fontsize=16)

    plt.show()



def vis_loading_matrix(likelihood_pdf, U):

    if isinstance(likelihood_pdf.readout_fn, nn.Sequential):
        readout_fn = likelihood_pdf.readout_fn[-1]
    else:
        readout_fn = likelihood_pdf.readout_fn

    R = likelihood_pdf.delta * torch.exp(readout_fn.bias).cpu()
    C = readout_fn.weight.detach().cpu()
    C_rot = (R.unsqueeze(-1).sqrt() * U).detach()

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(C, aspect='auto', cmap='bwr', interpolation='none')
    plt.colorbar()
    plt.title('original loadings\n', fontsize=24)
    plt.xlabel('latent', fontsize=16)
    plt.ylabel('neuron', fontsize=16)

    plt.subplot(1, 2, 2)
    plt.imshow(C_rot, aspect='auto', cmap='bwr', interpolation='none')
    plt.colorbar()
    plt.title("rotated loadings\n", fontsize=24)
    plt.xlabel('latent', fontsize=16)
    plt.ylabel('neuron', fontsize=16)

    plt.tight_layout()
    plt.show()


def plot_reaching(true_pos, pos_hat_s, r2_s, pos_hat_f, r2_f, pos_hat_p, r2_p, n_trials_to_plot):

    trial_plt_dx = torch.randperm(true_pos.shape[0])[:n_trials_to_plot]
    reach_angle = torch.atan2(true_pos[:, -1, 0], true_pos[:, -1, 1])
    reach_colors = plt.cm.hsv(reach_angle / (2 * np.pi) + 0.5)

    def center_reach(ax, reach):
        max_val = max(abs(reach[:, :, 0]).max() * 1.2, abs(reach[:, :, 1]).max() * 1.2)
        ax.set_xlim(-max_val, max_val)
        ax.set_ylim(-max_val, max_val)

        ax.axhline(0, color='gray', linewidth=0.6, linestyle='--')
        ax.axvline(0, color='gray', linewidth=0.6, linestyle='--')

    def plot_reaching_regime(axs, pos, reach_colors):
        n_trials, n_bins, _ = pos.shape
        axs.axis('off')

        for n in range(n_trials):
            axs.plot(pos[n, :, 0], pos[n, :, 1], color=reach_colors[n])
        center_reach(axs, pos)

    with torch.no_grad():
        fig, axs = plt.subplots(1, 4, figsize=(16, 4))

        plot_reaching_regime(axs[0], true_pos[trial_plt_dx], reach_colors[trial_plt_dx])
        plot_reaching_regime(axs[1], pos_hat_s[trial_plt_dx], reach_colors[trial_plt_dx])
        plot_reaching_regime(axs[2], pos_hat_f[trial_plt_dx], reach_colors[trial_plt_dx])
        plot_reaching_regime(axs[3], pos_hat_p[trial_plt_dx], reach_colors[trial_plt_dx])

        axs[0].set_title('true', fontsize=22)
        axs[1].set_title(f'smoothed\nr2:{r2_s:.3f}', fontsize=22)
        axs[2].set_title(f'filtered\nr2:{r2_f:.3f}', fontsize=22)
        axs[3].set_title(f'predicted\nr2:{r2_p:.3f}', fontsize=22)

        fig.tight_layout()


def vis_cca_weights(w):

    w0 = np.abs(w[:, 0])
    w1 = np.abs(w[:, 1])

    idx0 = np.argsort(w0)[::-1]
    idx1 = np.argsort(w1)[::-1]

    fig, axs = plt.subplots(2, 2, figsize=(8, 6), sharey='row')

    # ---------- Original order ----------
    axs[0, 0].bar(range(len(w0)), w0, color='skyblue', alpha=0.8)
    axs[0, 0].set_title("1st Canonical Axis", fontsize=22)
    axs[0, 0].set_xlabel("latents", fontsize=18)
    axs[0, 0].set_ylabel("|weight|", fontsize=18)
    axs[0, 0].grid(True)

    axs[0, 1].bar(range(len(w1)), w1, color='coral', alpha=0.8)
    axs[0, 1].set_title("2nd Canonical Axis", fontsize=22)
    axs[0, 1].set_xlabel("latents", fontsize=18)
    axs[0, 1].grid(True)

    # ---------- Sorted ----------
    axs[1, 0].bar(range(len(w0)), w0[idx0], color='skyblue', alpha=0.8)
    axs[1, 0].set_xlabel("latents (sorted)", fontsize=18)
    axs[1, 0].set_ylabel("|weight|", fontsize=18)
    axs[1, 0].grid(True)

    axs[1, 1].bar(range(len(w1)), w1[idx1], color='coral', alpha=0.8)
    axs[1, 1].set_xlabel("latents (sorted)", fontsize=18)
    axs[1, 1].grid(True)

    plt.suptitle("Contribution of Latents to Canonical Axes", fontsize=24)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_latent_single_trials(z, reach_pos, latent_idx, event_bin, bin_size_ms=20.0, alpha=0.05):
    # time centered on event bin
    time = (np.arange(z.shape[1]) - event_bin) * bin_size_ms
    plt.figure(figsize=(5, 3))

    for trial_idx in range(z.shape[0]):
        x, y = reach_pos[trial_idx]
        reach_angle = torch.atan2(x, y)
        reach_color = plt.cm.hsv((reach_angle / (2 * np.pi)).item() + 0.5)

        plt.plot(time, z[trial_idx, :, latent_idx], color=reach_color, alpha=alpha)

    plt.axvline(0, linestyle='--', linewidth=2, color='red')
    # plt.axvline(-200, linestyle='--', linewidth=2, color='gray')

    plt.title(f"\nLatent {latent_idx}\n", fontsize=20)
    # plt.xlabel("time (ms)", fontsize=16)
    # plt.grid(True)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_cumulative_seq2seq_r2(cum_results):

    k_list = cum_results['k_list']
    order = cum_results.get('order', np.arange(len(k_list)))
    r2_mean = cum_results['r2_test_mean']
    r2_perdim = cum_results['r2_test_perdim']  # shape (F, D)

    plt.figure(figsize=(6, 4))

    plt.plot(k_list, r2_perdim[:, 0], lw=1.5, alpha=0.6, color='gold', label=f'vel x R²')
    plt.plot(k_list, r2_perdim[:, 1], lw=1.5, alpha=0.6, color='coral', label=f'vel y R²')

    plt.plot(k_list, r2_mean, lw=2.0, color='navy', label='mean R²')
    plt.scatter(k_list, r2_mean, s=15, color='navy')

    plt.xlabel('# latents used', fontsize=14)
    plt.ylabel('R²', fontsize=14)
    plt.title('noncausal seq2seq cumulative decoding from latents\nordered by weights in the first canonical axis\n', fontsize=18)

    plt.xticks(k_list[::4])
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(fontsize=12)

    plt.tight_layout()
    plt.show()


def plot_latent_condition_average(
    z,                     # (trials, time, latents)
    condition_ids,         # (trials,) → integer condition labels
    target_pos_per_cond,   # (n_conditions, 2) → (x,y) for each condition
    latent_idx,
    event_bin,
    bin_size_ms=20.0,
    alpha=0.8
):
    time = (np.arange(z.shape[1]) - event_bin) * bin_size_ms

    conds = torch.unique(condition_ids)
    plt.figure(figsize=(5, 3))

    for cond in conds:
        mask = (condition_ids == cond)
        z_avg = z[mask][:, :, latent_idx].mean(axis=0)

        x, y = target_pos_per_cond[cond]
        angle = torch.atan2(x, y)
        color = plt.cm.hsv((angle / (2 * np.pi)).item() + 0.5)

        plt.plot(time, z_avg, color=color, alpha=alpha, lw=1.5)

    plt.axvline(0, linestyle='--', linewidth=1.5, color='red')

    plt.title(f"Latent {latent_idx}, condition-averaged", fontsize=22)
    plt.xlabel("time (ms)", fontsize=16)
    #plt.grid(True)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.show()