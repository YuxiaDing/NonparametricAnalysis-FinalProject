from simulate_methods import run_replicates
from simulate_noise import run_replicates as run_noise_replicates
from simulate_boundary import run as run_boundary
from real_data_bike import run as run_bike


def main():
    run_replicates()
    run_noise_replicates()
    run_boundary()
    run_bike()


if __name__ == "__main__":
    main()
