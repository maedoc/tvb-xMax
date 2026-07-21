"""Visualization utilities for APVBT.

Common plotting functions extracted from notebooks for consistent,
reusable visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_latent_space(mvn, n_dims=None, save_path=None, figsize=(3, 5)):
    """Plot latent space distribution across parcellations.

    Shows how latent codes vary across parcellations and subjects.

    Args:
        mvn: MvNorm instance with latent codes
        n_dims: Number of latent dimensions to plot (default: all)
        save_path: Path to save figure (optional)
        figsize: Figure size tuple

    Returns:
        Figure and axes objects
    """
    nparc, tts, ld = mvn.us.shape
    n_dims = n_dims or ld

    fig = plt.figure(figsize=figsize)
    for i in range(min(n_dims, ld)):
        plt.plot(mvn.us[:, 0, i], i + np.random.randn(nparc)*0.05,
                'xk', alpha=0.4, label='Subj 0' if i == 0 else '')
        plt.plot(mvn.us[:, 1, i], i + np.random.randn(nparc)*0.05,
                'ro', alpha=0.4, label='Subj 1' if i == 0 else '')
        if tts > 5:
            plt.plot(mvn.us[:, 5, i], i + np.random.randn(nparc)*0.1,
                    'g+', alpha=0.4, label='Subj 5' if i == 0 else '')

    plt.grid(True, alpha=0.3)
    plt.ylabel('Latent dimension')
    plt.xlabel('Latent weight')
    plt.title(f'Latent Space ({nparc} parcs, {tts} subjects, {ld}D)')
    if n_dims <= 16:
        plt.legend(loc='best', framealpha=0.9)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig, plt.gca()


def plot_connectome_matrix(conn, title=None, log_scale=True,
                           save_path=None, figsize=(6, 5)):
    """Plot a single connectome matrix.

    Args:
        conn: Connectome matrix (n_roi, n_roi)
        title: Plot title
        log_scale: Whether to plot log(conn + 1)
        save_path: Path to save figure
        figsize: Figure size tuple

    Returns:
        Figure and axes objects
    """
    fig, ax = plt.subplots(figsize=figsize)

    if log_scale:
        data = np.log(conn + 1)
        label = 'log(weight + 1)'
    else:
        data = conn
        label = 'weight'

    im = ax.imshow(data, cmap='viridis', aspect='auto')
    cbar = plt.colorbar(im, ax=ax, label=label)

    ax.set_xlabel('ROI')
    ax.set_ylabel('ROI')
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Connectome ({conn.shape[0]} ROIs)')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig, ax


def plot_all_parcellations(xc, idx=0, save_path=None, figsize=(15, 15)):
    """Plot connectomes for all parcellations.

    Args:
        xc: XCode instance
        idx: Subject index to plot
        save_path: Path to save figure
        figsize: Figure size tuple

    Returns:
        Figure and axes objects
    """
    from .utils import triu_to_mat

    nparc = len(xc.parcs)
    ncols = 4
    nrows = int(np.ceil(nparc / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()

    for i, parc in enumerate(xc.parcs):
        w = triu_to_mat(xc.conns[i] + xc.means[i])[idx]
        axes[i].imshow(np.log(w + 1), cmap='viridis')
        axes[i].set_title(parc, fontsize=10)
        axes[i].axis('off')

    # Hide unused subplots
    for i in range(nparc, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig, axes


def plot_confusion_matrix(crs, parcs=None, save_path=None, figsize=(10, 8)):
    """Plot confusion rate matrix between parcellations.

    Args:
        crs: Confusion rate matrix (n_parc, n_parc)
        parcs: List of parcellation names (optional)
        save_path: Path to save figure
        figsize: Figure size tuple

    Returns:
        Figure and axes objects
    """
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(crs, vmin=0, vmax=1.0, cmap='RdYlGn_r')
    cbar = plt.colorbar(im, ax=ax, label='Confusion Rate')

    ax.set_xlabel('Decode to Parcellation')
    ax.set_ylabel('Encode from Parcellation')
    ax.set_title('Cross-Parcellation Confusion Rates')

    if parcs is not None and len(parcs) < 25:
        ax.set_xticks(range(len(parcs)))
        ax.set_yticks(range(len(parcs)))
        ax.set_xticklabels(parcs, rotation=90, ha='right', fontsize=8)
        ax.set_yticklabels(parcs, fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig, ax


def plot_sbi_diagnostics(shrinkage, z_scores, ci90, param_names=None,
                        save_path=None, figsize=(12, 4)):
    """Plot SBI posterior diagnostics.

    Args:
        shrinkage: Posterior shrinkage (n_params,)
        z_scores: Z-scores (n_params,)
        ci90: 90% coverage indicators (n_params,)
        param_names: Parameter names (optional)
        save_path: Path to save figure
        figsize: Figure size tuple

    Returns:
        Figure and axes objects
    """
    n_params = len(shrinkage)
    x = np.arange(n_params)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Shrinkage
    axes[0].bar(x, shrinkage)
    axes[0].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[0].set_xlabel('Parameter')
    axes[0].set_ylabel('Shrinkage')
    axes[0].set_title('Posterior Shrinkage\n(higher = more informative)')
    axes[0].grid(True, alpha=0.3)

    # Z-scores
    axes[1].bar(x, z_scores)
    axes[1].axhline(2, color='r', linestyle='--', alpha=0.5, label='±2σ')
    axes[1].axhline(-2, color='r', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Parameter')
    axes[1].set_ylabel('Z-score')
    axes[1].set_title('Posterior Z-scores\n(should be close to 0)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Coverage
    coverage_pct = ci90.mean() * 100
    axes[2].bar(x, ci90)
    axes[2].axhline(0.9, color='g', linestyle='--', alpha=0.5,
                   label=f'Expected 90%\n(actual: {coverage_pct:.1f}%)')
    axes[2].set_xlabel('Parameter')
    axes[2].set_ylabel('In 90% CI')
    axes[2].set_title('90% Credible Interval Coverage')
    axes[2].legend()
    axes[2].set_ylim([0, 1.1])
    axes[2].grid(True, alpha=0.3)

    if param_names is not None and n_params < 20:
        for ax in axes:
            ax.set_xticks(x)
            ax.set_xticklabels(param_names, rotation=45, ha='right', fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig, axes


def plot_training_trace(trace, save_path=None, figsize=(8, 4)):
    """Plot training loss curves.

    Args:
        trace: List of (train_loss, test_loss) tuples
        save_path: Path to save figure
        figsize: Figure size tuple

    Returns:
        Figure and axes objects
    """
    trace = np.array(trace)
    iterations = np.arange(len(trace))

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(iterations, trace[:, 0], label='Train Loss', alpha=0.7)
    ax.plot(iterations, trace[:, 1], label='Test Loss', alpha=0.7)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Log Loss')
    ax.set_title('Cross-Coder Training')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add final improvement text
    improvement = trace[0, 1] - trace[-1, 1]
    ax.text(0.98, 0.98, f'Δ Test Loss: {improvement:.3f}',
           transform=ax.transAxes, ha='right', va='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig, ax


def plot_posterior_samples(prior_samples, posterior_samples, true_values=None,
                          param_idx=0, param_name=None, save_path=None,
                          figsize=(8, 5)):
    """Plot prior vs posterior distributions for a parameter.

    Args:
        prior_samples: Prior samples (n_samples,)
        posterior_samples: Posterior samples (n_post_samples, n_obs)
        true_values: True parameter values (n_obs,) - optional
        param_idx: Index of observation to plot
        param_name: Parameter name for labeling
        save_path: Path to save figure
        figsize: Figure size tuple

    Returns:
        Figure and axes objects
    """
    fig, ax = plt.subplots(figsize=figsize)

    bins = np.linspace(
        min(prior_samples.min(), posterior_samples.min()),
        max(prior_samples.max(), posterior_samples.max()),
        50
    )

    ax.hist(prior_samples, bins=bins, alpha=0.3, density=True,
           label='Prior', color='gray')
    ax.hist(posterior_samples[:, param_idx], bins=bins, alpha=0.5,
           density=True, label='Posterior', color='blue')

    if true_values is not None:
        ax.axvline(true_values[param_idx], color='r', linestyle='--',
                  linewidth=2, label='True Value')

    ax.set_xlabel(param_name or f'Parameter {param_idx}')
    ax.set_ylabel('Density')
    ax.set_title('Prior vs Posterior')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig, ax
