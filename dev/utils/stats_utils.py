from operator import ifloordiv

import numpy as np

import torch
import torch.nn as nn

from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

from scipy.stats import pearsonr

import xfads.utils as utils



def compute_r2(pred: torch.Tensor, true: torch.Tensor) -> float:

    ss_residual = torch.sum((true - pred) ** 2)
    ss_total = torch.sum((true - torch.mean(true)) ** 2)
    r2 = 1 - (ss_residual / ss_total)

    return r2.item()


def get_ll_pdf_readouf_fn(ssm):
    likelihood_pdf = ssm.likelihood_pdf

    if isinstance(likelihood_pdf.readout_fn, nn.Sequential):
        readout_fn = likelihood_pdf.readout_fn[-1]
    else:
        readout_fn = likelihood_pdf.readout_fn

    return likelihood_pdf, readout_fn



move_to_cpu = lambda x: x.cpu().detach() if torch.is_tensor(x) else type(x)(map(move_to_cpu, x)) if isinstance(x, (list, tuple)) else x


def get_rotation_matrix(likelihood_pdf):

    rotation_matrix, S, U = move_to_cpu(utils.get_latent_rotation(likelihood_pdf))
    return rotation_matrix, S, U


def rotate_z(z, rotation_matrix, n_latents_read=None):
    if n_latents_read is None:
        z = z[..., :n_latents_read]
    z_rot = torch.einsum('stbd,df->stbf', z, rotation_matrix.T).cpu()
    return z_rot

def rotate_c(R, U):
    C_rot = (R.unsqueeze(-1).sqrt() * U).detach()
    return C_rot


def compute_latent_variance_contribution(C, z_samples):
    z = z_samples.reshape(-1, z_samples.shape[-1])  # (N, latents)
    z_centered = z - z.mean(0, keepdim=True)

    latent_contribs = []
    for i in range(C.shape[1]):
        yi = torch.outer(z_centered[:, i], C[:, i])
        proj = yi - yi.mean(0)
        var_i = (proj**2).sum()
        latent_contribs.append(var_i)

    latent_contribs = torch.tensor(latent_contribs)
    var_explained = latent_contribs / latent_contribs.sum()

    return var_explained


def n_latents_vs_r2_latents(z_train, z_test, vel_train, vel_test, alpha=0.01, max_latents=None):

    def flatten_latents(z): return z.reshape(-1, z.shape[-1]).detach().cpu().numpy()
    def flatten_vel(v):     return v.reshape(-1, 2).detach().cpu().numpy()

    X_train = flatten_latents(z_train)
    X_test  = flatten_latents(z_test)
    y_train = flatten_vel(vel_train)
    y_test  = flatten_vel(vel_test)

    n_latents = X_train.shape[1] if max_latents is None else min(max_latents, X_train.shape[1])
    r2_scores = []

    for k in range(1, n_latents+1):
        clf = Ridge(alpha=alpha)
        clf.fit(X_train[:, :k], y_train)
        y_pred = clf.predict(X_test[:, :k])
        r2 = r2_score(y_test, y_pred, multioutput='uniform_average')
        r2_scores.append(r2)
    return np.array(r2_scores)


def n_latents_vs_r2_rates(z_train, z_test, vel_train, vel_test, loadin_matrix, b, delta, alpha=0.01, max_latents=None):
    """
    Compute R² of velocity decoding from generated rates.
    Matches velocity_decoder: mean over samples, flatten trials x bins.
    """
    def flatten_vel(v):
        return v.reshape(-1, 2).detach().cpu().numpy()

    y_train = flatten_vel(vel_train)
    y_test  = flatten_vel(vel_test)

    total_latents = z_train.shape[-1]
    n_latents = total_latents if max_latents is None else min(max_latents, total_latents)
    r2_scores = []

    for k in range(1, n_latents + 1):
        # select first k latents
        z_train_k = z_train[..., :k]
        z_test_k  = z_test[..., :k]

        # select corresponding weights
        C_k = loadin_matrix[:, :k]

        # generate rates (mean over samples, like velocity_decoder)
        rates_train = (delta * torch.exp(
            torch.einsum('nd,stbd->stbn', C_k, z_train_k) + b.view(1,1,1,-1)
        )).mean(dim=0).detach().cpu()

        rates_test = (delta * torch.exp(
            torch.einsum('nd,stbd->stbn', C_k, z_test_k) + b.view(1,1,1,-1)
        )).mean(dim=0).detach().cpu()

        # flatten trials x bins
        X_train = rates_train.reshape(-1, rates_train.shape[-1]).numpy()
        X_test  = rates_test.reshape(-1,  rates_test.shape[-1]).numpy()

        # Ridge regression
        clf = Ridge(alpha=alpha)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        r2 = r2_score(y_test, y_pred, multioutput='uniform_average')
        r2_scores.append(r2)

    return np.array(r2_scores)


def reconstruct_rates(delta, readout_mx, bias, z):

    if z.ndim == 4:
        rate_hat = delta * torch.exp(torch.einsum('nd,stbd->stbn', readout_mx, z) + bias.view(1, 1, 1, -1)).mean(dim=0).detach()
    elif z.ndim == 3:
        rate_hat = delta * torch.exp(torch.einsum('nd,tbd->tbn', readout_mx, z) + bias.view(1, 1, -1))
    else:
        raise ValueError(f"Expected z to be 3D or 4D, got {z.ndim}D")

    return rate_hat


def residualize_velocity(vel_train, vel_valid, vel_test, cond_train, cond_valid, cond_test):
    cond_mean_dict = {}
    unique_conditions = cond_train.unique()

    # Compute mean velocity per condition using only training set
    for cond in unique_conditions:
        idx = (cond_train == cond)
        cond_mean_dict[int(cond)] = vel_train[idx].mean(dim=0, keepdim=True)  # (1, n_bins, 2)

    def subtract_cond_mean(vel, cond):
        vel_resid = torch.zeros_like(vel)
        for i, c in enumerate(cond):
            vel_resid[i] = vel[i] - cond_mean_dict[int(c)]
        return vel_resid

    vel_train_resid = subtract_cond_mean(vel_train, cond_train)
    vel_valid_resid = subtract_cond_mean(vel_valid, cond_valid)
    vel_test_resid  = subtract_cond_mean(vel_test, cond_test)

    return vel_train_resid, vel_valid_resid, vel_test_resid


def inst_velocity_decoder(cfg,
                          rate_hat_train_f, rate_hat_valid_f, rate_hat_test_f,
                          rate_hat_train_s, rate_hat_valid_s, rate_hat_test_s,
                          rate_hat_train_p, rate_hat_valid_p, rate_hat_test_p,
                          vel_train, vel_valid, vel_test,
                          conds_train, conds_valid,conds_test,
                          on_residuals=False):

    if on_residuals:
        vel_train, vel_valid, vel_test = residualize_velocity(
            vel_train, vel_valid, vel_test,
            conds_train, conds_valid,conds_test
        )

    train_output_shape = list(vel_train.shape)[:-1] + [2]
    valid_output_shape = list(vel_valid.shape)[:-1] + [2]
    test_output_shape = list(vel_test.shape)[:-1] + [2]

    reshape_detach = lambda x: x.reshape(-1, x.shape[2]).detach().cpu()

    rate_hat_train_f = reshape_detach(rate_hat_train_f)
    rate_hat_valid_f = reshape_detach(rate_hat_valid_f)
    rate_hat_test_f = reshape_detach(rate_hat_test_f)

    rate_hat_train_s = reshape_detach(rate_hat_train_s)
    rate_hat_valid_s = reshape_detach(rate_hat_valid_s)
    rate_hat_test_s = reshape_detach(rate_hat_test_s)

    rate_hat_train_p = reshape_detach(rate_hat_train_p)
    rate_hat_valid_p = reshape_detach(rate_hat_valid_p)
    rate_hat_test_p = reshape_detach(rate_hat_test_p)

    vel_train = vel_train.reshape(-1, 2).cpu()
    vel_valid = vel_valid.reshape(-1, 2).cpu()
    vel_test = vel_test.reshape(-1, 2).cpu()

    clf = Ridge(alpha=0.01)
    clf.fit(rate_hat_train_s, vel_train)

    with torch.no_grad():
        vel_hat_train_f = clf.predict(rate_hat_train_f).reshape(train_output_shape)
        vel_hat_valid_f = clf.predict(rate_hat_valid_f).reshape(valid_output_shape)
        vel_hat_test_f = clf.predict(rate_hat_test_f).reshape(test_output_shape)

        vel_hat_train_s = clf.predict(rate_hat_train_s).reshape(train_output_shape)
        vel_hat_valid_s = clf.predict(rate_hat_valid_s).reshape(valid_output_shape)
        vel_hat_test_s = clf.predict(rate_hat_test_s).reshape(test_output_shape)

        vel_hat_train_p = clf.predict(rate_hat_train_p).reshape(train_output_shape)
        vel_hat_valid_p = clf.predict(rate_hat_valid_p).reshape(valid_output_shape)
        vel_hat_test_p = clf.predict(rate_hat_test_p).reshape(test_output_shape)

    with torch.no_grad():
        r2_train_enc_f = clf.score(rate_hat_train_f, vel_train)
        r2_valid_enc_f = clf.score(rate_hat_valid_f, vel_valid)
        r2_test_enc_f = clf.score(rate_hat_test_f, vel_test)

        r2_train_enc_s = clf.score(rate_hat_train_s, vel_train)
        r2_valid_enc_s = clf.score(rate_hat_valid_s, vel_valid)
        r2_test_enc_s = clf.score(rate_hat_test_s, vel_test)

        r2_train_enc_p = clf.score(rate_hat_train_p, vel_train)
        r2_valid_enc_p = clf.score(rate_hat_valid_p, vel_valid)
        r2_test_enc_p = clf.score(rate_hat_test_p, vel_test)

    vel_hat_train_f = torch.tensor(vel_hat_train_f).type(torch.float32).to(cfg.data_device)
    vel_hat_valid_f = torch.tensor(vel_hat_valid_f).type(torch.float32).to(cfg.data_device)
    vel_hat_test_f = torch.tensor(vel_hat_test_f).type(torch.float32).to(cfg.data_device)

    vel_hat_train_s = torch.tensor(vel_hat_train_s).type(torch.float32).to(cfg.data_device)
    vel_hat_valid_s = torch.tensor(vel_hat_valid_s).type(torch.float32).to(cfg.data_device)
    vel_hat_test_s = torch.tensor(vel_hat_test_s).type(torch.float32).to(cfg.data_device)

    vel_hat_train_p = torch.tensor(vel_hat_train_p).type(torch.float32).to(cfg.data_device)
    vel_hat_valid_p = torch.tensor(vel_hat_valid_p).type(torch.float32).to(cfg.data_device)
    vel_hat_test_p = torch.tensor(vel_hat_test_p).type(torch.float32).to(cfg.data_device)

    return (clf,
            vel_hat_train_f, vel_hat_valid_f, vel_hat_test_f,
            vel_hat_train_s, vel_hat_valid_s, vel_hat_test_s,
            vel_hat_train_p, vel_hat_valid_p, vel_hat_test_p,
            r2_train_enc_f, r2_valid_enc_f, r2_test_enc_f,
            r2_train_enc_s, r2_valid_enc_s, r2_test_enc_s,
            r2_train_enc_p, r2_valid_enc_p, r2_test_enc_p)


vel_to_pos = lambda v: torch.cumsum(torch.tensor(v), dim=1)


def to_numpy(x):
    """Convert PyTorch tensors to NumPy arrays (safe no-op for NumPy)."""
    import torch
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        raise TypeError(f"Unsupported input type: {type(x)}")


def compute_vaf(y_true, y_pred, eps=1e-10):
    """y_true, y_pred: (N, T, D) -> returns (per-dim vaf array, mean_vaf)"""
    N, T, D = y_true.shape
    vafs = []
    for d in range(D):
        yt = y_true[:, :, d].ravel()
        yhat = y_pred[:, :, d].ravel()
        var_err = np.var(yt - yhat)
        var_y = np.var(yt)
        vafs.append(1.0 - var_err / (var_y + eps))
    return np.array(vafs), float(np.mean(vafs))


def compute_r2_per_ouput_dim(y_true, y_pred, eps=1e-10):
    """
    Compute coefficient of determination (R^2) per output dimension.
    y_true, y_pred: (N, T, D)
    Returns:
        r2s: array of R^2 per dimension
        mean_r2: average across dimensions
    """
    N, T, D = y_true.shape
    r2s = []
    for d in range(D):
        yt = y_true[:, :, d].ravel()
        yhat = y_pred[:, :, d].ravel()
        ss_res = np.sum((yt - yhat) ** 2)
        ss_tot = np.sum((yt - np.mean(yt)) ** 2) + eps
        r2 = 1 - ss_res / ss_tot
        r2s.append(r2)
    return np.array(r2s), float(np.mean(r2s))


def flatten_latents(latents):
    N, T, F = latents.shape
    return latents.reshape(N, T * F)


def flatten_targets(vel):
    N, T, D = vel.shape
    return vel.reshape(N, T * D)


def unflatten_targets(Yflat, T, D):
    return Yflat.reshape(-1, T, D)


def shift_latents_for_lag(latents, lag, fill=0.0):
    """
    Shift latents by integer lag for causal alignment.
      lag > 0 : latents lead behavior by 'lag' steps -> shift earlier
      lag < 0 : latents lag behavior -> shift later
    Pads with `fill`.
    """
    if lag == 0:
        return latents.copy()
    N, T, F = latents.shape
    out = np.full_like(latents, fill_value=fill, dtype=latents.dtype)
    if lag > 0:
        out[:, :T - lag, :] = latents[:, lag:, :]
    else:
        lag_abs = abs(lag)
        out[:, lag_abs:, :] = latents[:, :T - lag_abs, :]
    return out


def supervised_latent_rotation(X_train, Y_train, X_valid=None, X_test=None,
                               n_components=None, return_full_rotation=False,
                               scale_inputs=False):
    """
    Fit linear regression Y = X W on training set, SVD on W to get input-side directions U.
    Optionally z-score X before rotation.

    Parameters:
    - X_train: (N_train, M)
    - Y_train: (N_train, O)
    - X_valid, X_test: optional validation/test sets
    - n_components: how many rotated components to keep
    - scale_inputs: if True, z-score X_train (fit on train only)
    - return_full_rotation: if True, also return full MxM rotation matrix

    Returns:
    - X_train_rot, X_valid_rot, X_test_rot
    - U_used: rotation matrix (M, k)
    - optionally U_full
    """
    # --- optional z-score BEFORE rotation ---
    xscaler = None
    if scale_inputs:
        xscaler = StandardScaler().fit(X_train)
        X_train_scaled = xscaler.transform(X_train)
        X_valid_scaled = xscaler.transform(X_valid) if X_valid is not None else None
        X_test_scaled  = xscaler.transform(X_test)  if X_test is not None else None
    else:
        X_train_scaled = X_train
        X_valid_scaled = X_valid
        X_test_scaled  = X_test

    # Fit linear regression on scaled data
    lr = LinearRegression().fit(X_train_scaled, Y_train)
    W = lr.coef_.T  # shape (M, O)

    # SVD on W: W = U S Vt
    U, S, Vt = np.linalg.svd(W, full_matrices=False)  # U: (M, r)
    M = W.shape[0]
    r = U.shape[1]

    # Determine k (how many rotated components to keep)
    k = M if n_components is None else int(n_components)
    if k > M:
        raise ValueError(f"n_components ({k}) cannot exceed feature dim M ({M})")

    # Build U_used
    if k <= r:
        U_used = U[:, :k]
    else:
        Q, _ = np.linalg.qr(np.concatenate([U, np.eye(M)], axis=1))
        U_full = Q[:, :M]
        U_used = U_full[:, :k]

    # Rotate features
    X_train_rot = X_train_scaled.dot(U_used)
    X_valid_rot = X_valid_scaled.dot(U_used) if X_valid is not None else None
    X_test_rot  = X_test_scaled.dot(U_used)  if X_test is not None else None

    if return_full_rotation:
        if r == M:
            U_full = U
        else:
            Q, _ = np.linalg.qr(np.concatenate([U, np.eye(M)], axis=1))
            U_full = Q[:, :M]
        return X_train_rot, X_valid_rot, X_test_rot, U_used, U_full, xscaler

    return X_train_rot, X_valid_rot, X_test_rot, U_used


def train_eval_seq2seq_ridge(latents_train, latents_valid, latents_test,
                             vel_train, vel_valid, vel_test,
                             alphas=None,
                             use_pca=False, pca_n_components=None,
                             use_supervised_rotation=False, rotation_n_components=None,
                             scale_inputs=True, scale_outputs=False,
                             cv=5,
                             lag=0,
                             return_model=True):
    """
    Train/evaluate trial-level sequence-to-sequence Ridge.
    Inputs (all numpy arrays):
      latents_*(N, T, F), vel_*(N, T, D)
    Options:
      alphas: iterable of ridge alphas (if None, uses np.logspace(-6,6,25))
      use_pca: reduce flattened X via PCA (fit on training only)
      scale_inputs: z-score X (fit on training only)
      scale_outputs: z-score Y (rare; inverse trans before metrics)
      cv: folds for RidgeCV
      lag: integer; apply causal shift to latents before flattening
    Returns:
      results dict with keys:
        'model','xscaler','yscaler','pca','best_alpha',
        'pred_train','pred_valid','pred_test', 'mse','vaf','pearson','lag'
    """
    # Ensure inputs are numpy arrays
    latents_train = to_numpy(latents_train)
    latents_valid = to_numpy(latents_valid)
    latents_test  = to_numpy(latents_test)
    vel_train     = to_numpy(vel_train)
    vel_valid     = to_numpy(vel_valid)
    vel_test      = to_numpy(vel_test)

    # basic checks
    assert isinstance(latents_train, np.ndarray), "latents_train must be np.ndarray"
    assert isinstance(vel_train, np.ndarray), "vel_train must be np.ndarray"
    if alphas is None:
        alphas = np.logspace(-6, 6, 25)

    # apply lag shift
    if lag != 0:
        latents_train = shift_latents_for_lag(latents_train, lag)
        latents_valid = shift_latents_for_lag(latents_valid, lag)
        latents_test  = shift_latents_for_lag(latents_test, lag)

    N_train, T, F = latents_train.shape
    N_valid = latents_valid.shape[0]
    N_test  = latents_test.shape[0]
    _, _, D = vel_train.shape

    # --- Apply supervised rotation (if requested) ---
    if use_supervised_rotation:
        # reshape to (N*T, F) and (N*T, D)
        X_train_flat = latents_train.reshape(-1, F)
        X_valid_flat = latents_valid.reshape(-1, F)
        X_test_flat  = latents_test.reshape(-1, F)
        Y_train_flat = vel_train.reshape(-1, D)

        # rotate
        X_train_rot, X_valid_rot, X_test_rot, rotation_matrix = supervised_latent_rotation(
            X_train_flat, Y_train_flat,
            X_valid_flat, X_test_flat,
            n_components=rotation_n_components,
            scale_inputs=True
        )

        # reshape back to (N, T, F) for the rest of pipeline
        latents_train = X_train_rot.reshape(N_train, T, F)
        latents_valid = X_valid_rot.reshape(N_valid, T, F)
        latents_test  = X_test_rot.reshape(N_test, T, F)
    else:
        rotation_matrix = None

    # flatten
    X_train = flatten_latents(latents_train)
    X_valid = flatten_latents(latents_valid)
    X_test  = flatten_latents(latents_test)

    Y_train = flatten_targets(vel_train)
    Y_valid = flatten_targets(vel_valid)
    Y_test  = flatten_targets(vel_test)

    # scale inputs
    xscaler = None
    if scale_inputs:
        xscaler = StandardScaler().fit(X_train)
        X_train_s = xscaler.transform(X_train)
        X_valid_s = xscaler.transform(X_valid)
        X_test_s  = xscaler.transform(X_test)
    else:
        X_train_s, X_valid_s, X_test_s = X_train, X_valid, X_test

    # PCA (optional)
    pca_model = None
    if use_pca:
        n_feats = X_train_s.shape[1]
        if pca_n_components is None:
            pca_n_components = min(N_train, n_feats)
        pca_model = PCA(n_components=pca_n_components)
        X_train_s = pca_model.fit_transform(X_train_s)
        X_valid_s = pca_model.transform(X_valid_s)
        X_test_s  = pca_model.transform(X_test_s)

    # scale outputs (optional)
    yscaler = None
    if scale_outputs:
        yscaler = StandardScaler().fit(Y_train)
        Y_train_s = yscaler.transform(Y_train)
    else:
        Y_train_s = Y_train

    # RidgeCV
    ridgecv = RidgeCV(alphas=alphas, scoring='neg_mean_squared_error', cv=cv)
    ridgecv.fit(X_train_s, Y_train_s)
    best_alpha = ridgecv.alpha_

    model = Ridge(alpha=best_alpha).fit(X_train_s, Y_train_s)

    # predictions
    Yhat_train = model.predict(X_train_s)
    Yhat_valid = model.predict(X_valid_s)
    Yhat_test  = model.predict(X_test_s)

    if yscaler is not None:
        Yhat_train = yscaler.inverse_transform(Yhat_train)
        Yhat_valid = yscaler.inverse_transform(Yhat_valid)
        Yhat_test  = yscaler.inverse_transform(Yhat_test)

    Yhat_train_resh = unflatten_targets(Yhat_train, T, D)
    Yhat_valid_resh = unflatten_targets(Yhat_valid, T, D)
    Yhat_test_resh  = unflatten_targets(Yhat_test, T, D)

    # metrics
    mse_train = mean_squared_error(vel_train.reshape(-1, D), Yhat_train_resh.reshape(-1, D))
    mse_valid = mean_squared_error(vel_valid.reshape(-1, D), Yhat_valid_resh.reshape(-1, D))
    mse_test  = mean_squared_error(vel_test.reshape(-1, D),  Yhat_test_resh.reshape(-1, D))

    vaf_train_perdim, vaf_train_mean = compute_vaf(vel_train, Yhat_train_resh)
    vaf_valid_perdim, vaf_valid_mean = compute_vaf(vel_valid, Yhat_valid_resh)
    vaf_test_perdim,  vaf_test_mean  = compute_vaf(vel_test, Yhat_test_resh)

    r2_train_perdim, r2_train_mean = compute_r2_per_ouput_dim(vel_train, Yhat_train_resh)
    r2_valid_perdim, r2_valid_mean = compute_r2_per_ouput_dim(vel_valid, Yhat_valid_resh)
    r2_test_perdim,  r2_test_mean  = compute_r2_per_ouput_dim(vel_test, Yhat_test_resh)

    # pearson per-dim
    pearson_train = []
    pearson_valid = []
    pearson_test = []
    for d in range(D):
        yt = vel_train[:, :, d].ravel()
        yhat = Yhat_train_resh[:, :, d].ravel()
        pearson_train.append(np.nan if np.std(yt) < 1e-12 or np.std(yhat) < 1e-12 else pearsonr(yt, yhat)[0])

        yt = vel_valid[:, :, d].ravel()
        yhat = Yhat_valid_resh[:, :, d].ravel()
        pearson_valid.append(np.nan if np.std(yt) < 1e-12 or np.std(yhat) < 1e-12 else pearsonr(yt, yhat)[0])

        yt = vel_test[:, :, d].ravel()
        yhat = Yhat_test_resh[:, :, d].ravel()
        pearson_test.append(np.nan if np.std(yt) < 1e-12 or np.std(yhat) < 1e-12 else pearsonr(yt, yhat)[0])

    results = {
        'model': model,
        'xscaler': xscaler,
        'yscaler': yscaler,
        'pca': pca_model,
        'rotation_matrix': rotation_matrix,
        'use_supervised_rotation': use_supervised_rotation,
        'best_alpha': best_alpha,
        'pred_train': Yhat_train_resh,
        'pred_valid': Yhat_valid_resh,
        'pred_test': Yhat_test_resh,
        'mse': {'train': mse_train, 'valid': mse_valid, 'test': mse_test},
        'r2': {
            'train': {'perdim': r2_train_perdim, 'mean': r2_train_mean},
            'valid': {'perdim': r2_valid_perdim, 'mean': r2_valid_mean},
            'test':  {'perdim': r2_test_perdim,  'mean': r2_test_mean},
        },
        'vaf': {
            'train': {'perdim': vaf_train_perdim, 'mean': r2_train_mean},
            'valid': {'perdim': vaf_valid_perdim, 'mean': r2_valid_mean},
            'test':  {'perdim': vaf_test_perdim,  'mean': r2_test_mean},
        },
        'pearson': {
            'train': np.array(pearson_train),
            'valid': np.array(pearson_valid),
            'test': np.array(pearson_test),
        },
        'lag': lag,
    }

    if return_model:
        return results
    else:
        # drop model object if user only wants metrics/preds
        results.pop('model')
        return results


def lag_sweep(latents_train, latents_valid, latents_test,
              vel_train, vel_valid, vel_test,
              lag_range=range(-10, 11),
              **train_eval_kwargs):
    """
    Sweeps integer lags and returns best_lag, best_results, all_results dict.
    Selection criterion: validation mean VAF.
    """
    all_results = {}
    best_lag = None
    best_vaf = -np.inf
    best_results = None

    for lag in lag_range:
        res = train_eval_seq2seq_ridge(latents_train, latents_valid, latents_test,
                                       vel_train, vel_valid, vel_test,
                                       lag=lag, **train_eval_kwargs)
        all_results[lag] = res
        v = res['r2']['valid']['mean']
        print(f"lag={lag:3d} | valid mean Rˆ2 = {v:.4f} | alpha={res['best_alpha']}")
        if v > best_vaf:
            best_vaf = v
            best_lag = lag
            best_results = res

    return best_lag, best_results, all_results


def extract_weight_matrix(model, T, F, D, pca_model=None, xscaler=None):
    """
    model: trained sklearn Ridge with coef_ shape (T*D, M)
    Returns:
      W_raw: (T, D, T, F) such that y_flat = X_flat @ W_flat.T + bias_flat
      bias_raw: (T*D,)
    Notes: if PCA/xscaler were used during training, pass them to map back to original feature space.
    """
    C_z = model.coef_.copy()          # (T*D, M)
    intercept_z = model.intercept_.copy()  # (T*D,)

    # project back to original feature space if PCA used
    if pca_model is not None:
        C_s = C_z.dot(pca_model.components_)  # (T*D, T*F)
    else:
        C_s = C_z

    # adjust for input scaling if present
    if xscaler is not None:
        s = xscaler.scale_
        mu = xscaler.mean_
        C_raw = C_s / s.reshape(1, -1)
        bias_raw = intercept_z - ( (mu / s) @ C_s.T )
    else:
        C_raw = C_s
        bias_raw = intercept_z

    # reshape to (T, D, T, F)
    W_raw = C_raw.reshape(T, D, T, F)
    return W_raw, bias_raw


def extract_weight_matrix_per_dim(model, T, F, D, pca_model=None, xscaler=None):
    """
    Extract ridge weights per output dimension (e.g., vel X and Y separately).

    Returns:
        W_per_dim: list of length D, each entry is (T, T, F)
        bias_per_dim: list of length D, each entry is (T,)
    """
    W_raw, bias_raw = extract_weight_matrix(model, T, F, D, pca_model=pca_model, xscaler=xscaler)

    W_per_dim = []
    bias_per_dim = []

    for d in range(D):
        W_per_dim.append(W_raw[:, d, :, :])        # shape (T, T, F)
        bias_per_dim.append(bias_raw[d::D])       # take every D-th element starting at d -> shape (T,)

    return W_per_dim, bias_per_dim


def seq2seq_cumulative_decoder(latents_train, latents_valid, latents_test,
                      vel_train, vel_valid, vel_test,
                      order=None,
                      alphas=None,
                      use_pca=False,
                      pca_n_components=None,
                      use_supervised_rotation=False,
                      rotation_n_components=None,
                      scale_inputs=True,
                      scale_outputs=False,
                      cv=5,
                      lag=0,
                      return_models=False,
                      verbose=True):
    """
    Run cumulative decoding: add latents one-by-one (in `order`) and fit a seq2seq ridge
    for each k = 1..F. Returns R^2 per-dim and mean for each k.

    Parameters
    ----------
    latents_* : arrays with shape (N, T, F)
    vel_* : arrays with shape (N, T, D)
    order : array-like of length F, optional
        The order in which to add latents. If None, uses [0,1,...,F-1].
    alphas, use_pca, ... : forwarded to train_eval_seq2seq_ridge
    return_models : bool
        If True, include fitted model objects per k in the returned dict (heavy).
    verbose : bool
        Print progress.

    Returns
    -------
    results : dict with keys:
        'k_list' : list of ints (1..F)
        'r2_test_mean' : np.array shape (F,)
        'r2_test_perdim' : np.array shape (F, D)
        'r2_valid_mean' : np.array (optional)
        'models' (optional): list of model dicts returned by train_eval_seq2seq_ridge
    """
    # convert to numpy if not already (uses your to_numpy)
    latents_train = to_numpy(latents_train)
    latents_valid = to_numpy(latents_valid)
    latents_test  = to_numpy(latents_test)
    vel_train     = to_numpy(vel_train)
    vel_valid     = to_numpy(vel_valid)
    vel_test      = to_numpy(vel_test)

    N_train, T, F = latents_train.shape
    _, _, D = vel_train.shape

    if order is None:
        order = np.arange(F)
    else:
        order = np.asarray(order)
        if order.size != F:
            raise ValueError(f"order length {order.size} != F ({F})")

    # prepare outputs
    k_list = np.arange(1, F+1)
    r2_test_mean = np.zeros(F, dtype=float)
    r2_test_perdim = np.zeros((F, D), dtype=float)
    r2_valid_mean = np.zeros(F, dtype=float)
    r2_valid_perdim = np.zeros((F, D), dtype=float)

    models = [] if return_models else None

    # iterate cumulative
    for i, k in enumerate(k_list):
        selected = order[:k]
        if verbose:
            print(f"[cumulative] k={k} / {F}  -> using latents: {selected}")

        # create reduced latents (keep original trial/bin layout)
        lt_train_k = latents_train[:, :, selected]
        lt_valid_k = latents_valid[:, :, selected]
        lt_test_k  = latents_test[:, :, selected]

        # call existing train_eval_seq2seq_ridge (keeps the same pipeline)
        res = train_eval_seq2seq_ridge(
            lt_train_k, lt_valid_k, lt_test_k,
            vel_train, vel_valid, vel_test,
            alphas=alphas,
            use_pca=use_pca,
            pca_n_components=pca_n_components,
            use_supervised_rotation=use_supervised_rotation,
            rotation_n_components=rotation_n_components,
            scale_inputs=scale_inputs,
            scale_outputs=scale_outputs,
            cv=cv,
            lag=lag,
            return_model=True
        )

        # extract test r2
        r2_test_perdim[i, :] = np.array(res['r2']['test']['perdim'])
        r2_test_mean[i] = float(res['r2']['test']['mean'])

        # validation r2 (useful for overfitting checks)
        r2_valid_perdim[i, :] = np.array(res['r2']['valid']['perdim'])
        r2_valid_mean[i] = float(res['r2']['valid']['mean'])

        if return_models:
            models.append(res)
        # optional: free memory if not returning
        if not return_models:
            # delete big arrays inside res to save memory (if present)
            res.pop('pred_train', None)
            res.pop('pred_valid', None)
            res.pop('pred_test', None)
            models = None

    out = {
        'k_list': k_list,
        'order': order,
        'r2_test_mean': r2_test_mean,
        'r2_test_perdim': r2_test_perdim,
        'r2_valid_mean': r2_valid_mean,
        'r2_valid_perdim': r2_valid_perdim,
    }
    if return_models:
        out['models'] = models

    return out


def plot_cumulative_r2(cum_results, show_per_dim=False, per_dim_alpha=0.5,
                       xlabel='# latents used', ylabel='test mean R²',
                       title='Cumulative decoding: #latents vs test R²',
                       xticks_every=1):
    """
    Plot cumulative R^2 results returned by cumulative_decode().

    Parameters
    ----------
    cum_results : dict (output of cumulative_decode)
    show_per_dim : bool
        If True, plot each output-dimension R^2 as a lighter line.
    per_dim_alpha : float
        alpha for per-dim lines.
    """
    k_list = cum_results['k_list']
    order = cum_results.get('order', np.arange(len(k_list)))
    r2_mean = cum_results['r2_test_mean']
    r2_perdim = cum_results['r2_test_perdim']  # shape (F, D)

    plt.figure(figsize=(6,4))
    if show_per_dim:
        D = r2_perdim.shape[1]
        for d in range(D):
            plt.plot(k_list, r2_perdim[:, d], lw=1.2, alpha=per_dim_alpha, label=f'dim {d}' if d==0 else None)

    plt.plot(k_list, r2_mean, lw=2.5, color='C0', label='mean test R²')
    plt.scatter(k_list, r2_mean, s=20, color='C0')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if xticks_every is not None and xticks_every > 0:
        plt.xticks(k_list[::xticks_every])
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend() if show_per_dim else None
    plt.tight_layout()
    plt.show()


def flatten_for_cca(Z, V):
    """Z: (N_obs, n_latents), V: (N_obs, 2) -> standardized numpy arrays"""
    scalerZ = StandardScaler()
    scalerV = StandardScaler()
    Zs = scalerZ.fit_transform(Z)
    Vs = scalerV.fit_transform(V)
    return Zs, Vs


def apply_cca(z, vel):
    n_components = vel.shape[-1]
    Z = z.reshape(-1, z.shape[-1])
    V = vel.reshape(-1, 2)

    Zs, Vs = flatten_for_cca(Z, V)
    cca = CCA(n_components=n_components)
    Zc, Vc = cca.fit_transform(Zs, Vs)
    corrs = [np.corrcoef(Zc[:, i], Vc[:, i])[0, 1] for i in range(n_components)]
    return cca, np.array(corrs), Zc, Vc


import numpy as np
import matplotlib.pyplot as plt

def order_latents_by_variance(latents,
                              time_window,
                              move_onset_bin=None,
                              bin_size_ms=None,
                              plot=False):
    """
    Orders latent dimensions based on their average variance across trials
    within a given time window.

    Parameters
    ----------
    latents : np.ndarray or torch.Tensor
        Shape (N_trials, T_timebins, F_latents)
    time_window : tuple (t_start, t_end)
        Time bin indices specifying the window over which to compute variance.
        Example: (0, 15) for pre-movement window.
    move_onset_bin : int, optional
        Bin index for movement onset (used for x-axis labeling if plotting)
    bin_size_ms : float, optional
        Bin size in milliseconds (for labeling, optional)
    plot : bool, default False
        If True, plots the average variance per latent (sorted).

    Returns
    -------
    sorted_indices : np.ndarray
        Indices of latents sorted in descending order of average variance.
    avg_var_per_latent : np.ndarray
        Average variance per latent in the specified time window.
    """

    # Convert to numpy
    try:
        import torch
        if isinstance(latents, torch.Tensor):
            latents = latents.detach().cpu().numpy()
    except Exception:
        pass

    if latents.ndim != 3:
        raise ValueError("latents must have shape (N, T, F)")

    N, T, F = latents.shape
    t_start, t_end = time_window
    if not (0 <= t_start < T and 0 < t_end <= T):
        raise ValueError(f"time_window {time_window} must be within [0, {T}]")

    # Compute variance across trials -> shape (T, F)
    var_time_feature = np.var(latents, axis=0)  # (T, F)

    # Average variance within the window
    avg_var_per_latent = var_time_feature[t_start:t_end, :].mean(axis=0)  # (F,)

    # Sort by variance (descending)
    sorted_indices = np.argsort(avg_var_per_latent)[::-1]

    # Optional plotting
    if plot:
        plt.figure(figsize=(6, 4))
        plt.bar(np.arange(F), avg_var_per_latent[sorted_indices])
        plt.xlabel('Latent (sorted)')
        plt.ylabel('Average variance')
        title = f"Average latent variance from bin {t_start}–{t_end}"
        if move_onset_bin is not None and bin_size_ms is not None:
            t_start_ms = (t_start - move_onset_bin) * bin_size_ms
            t_end_ms = (t_end - move_onset_bin) * bin_size_ms
            title += f" ({t_start_ms:.0f}–{t_end_ms:.0f} ms)"
        plt.title(title)
        plt.tight_layout()
        plt.show()

    return sorted_indices, avg_var_per_latent

