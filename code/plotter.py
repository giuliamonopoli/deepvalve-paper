from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_all_splines(results, idx_plot, folder_name):
    plt.plot(
        results["ground_truth_septal"][idx_plot, :, 0].T,
        results["ground_truth_septal"][idx_plot, :, 1].T,
        "o-",
        label="GT septal",
    )
    plt.plot(
        results["ground_truth_lateral"][idx_plot, :, 0].T,
        results["ground_truth_lateral"][idx_plot, :, 1].T,
        "o-",
        label="GT lateral",
    )
    plt.plot(
        results["predicted_septal"][idx_plot, :, 0].T,
        results["predicted_septal"][idx_plot, :, 1].T,
        "o--",
        label="Predicted septal",
    )
    plt.plot(
        results["predicted_lateral"][idx_plot, :, 0].T,
        results["predicted_lateral"][idx_plot, :, 1].T,
        "o--",
        label="Predicted lateral",
    )
    plt.legend()

    plt.title(f"Comparison patient {idx_plot}")
    save_dir = Path(f"results/plots_splines_{folder_name}")
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
    plt.savefig(save_dir.joinpath(f"splines_patient_{idx_plot:02d}.png"))
    plt.close()


def plot_results_by_flag(results, flags, folder_name):
    n_patients = len(results)
    metrics = [
        "mae",
        "mse",
        "rmse",
        "mape",
        "cosine_similarity",
        "pro_dist",
        "pro_rotation_angle",
    ]
    flag_list = ["fp_1", "fp_2", "fp_3", "ff_1", "ff_2", "ff_3"]

    for metric in metrics:
        print(metric)
        fig, axs = plt.subplots(2, flags.shape[1] // 2, figsize=(10, 10))
        axs = axs.flatten()
        all_data = np.array([results[f"test_{i}"][metric] for i in range(n_patients)])
        for i_flag in range(len(flags[0, :])):
            sns.swarmplot(x=flags[:, i_flag], y=all_data, ax=axs[i_flag])
            axs[i_flag].set_title(flag_list[i_flag])
        axs[0].set_ylabel(metric.upper(), fontsize=16)
        plt.tight_layout()
        save_dir = Path(f"results/plots_flags_{folder_name}")
        if not save_dir.exists():
            save_dir.mkdir(parents=True)
        plt.savefig(
            save_dir.joinpath(f"flags_{metric}.png"),
            bbox_inches="tight",
            dpi=200,
        )
        plt.close()
