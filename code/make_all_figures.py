from simulate_bandwidth import run as run_bandwidth
from simulate_methods import plot_representative, run_replicates, plot_boundary_rmse
from simulate_noise import plot_error_boxplots, plot_representative as plot_noise_representative, run_replicates as run_noise_replicates
from simulate_boundary import run as run_boundary
from real_data_bike import run as run_bike


def main():
    run_bandwidth()
    _, method_summary = run_replicates()
    plot_representative()
    plot_boundary_rmse(method_summary)
    noise_results, _ = run_noise_replicates()
    plot_noise_representative()
    plot_error_boxplots(noise_results)
    run_boundary()
    run_bike()


if __name__ == "__main__":
    main()
