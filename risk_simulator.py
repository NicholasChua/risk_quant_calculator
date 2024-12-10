import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, poisson, gaussian_kde

MONTE_CARLO_SEED = 42
NUM_SIMULATIONS = 10000
KURTOSIS = 4  # Default value is 3


def get_beta_parameters_for_kurtosis(kurtosis):
    """
    Estimate parameters 'a' and 'b' for the beta distribution to achieve a desired kurtosis.
    This is an approximation since setting kurtosis directly is non-trivial.
    """
    # For demonstration purposes, we'll adjust 'a' and 'b' based on the desired kurtosis.
    if kurtosis <= 1.8:
        a = b = 0.5  # Higher kurtosis (leptokurtic)
    elif 1.8 < kurtosis <= 3:
        a = b = 1  # Standard uniform distribution
    elif 3 < kurtosis <= 9:
        a = b = 2  # Lower kurtosis (platykurtic)
    else:
        a = b = 5  # Even lower kurtosis

    return a, b


def find_first_non_zero_percentile(data):
    """Find the first non-zero value and its percentile in sorted data"""
    sorted_data = np.sort(data)
    non_zero_idx = np.argmax(sorted_data > 0)
    if non_zero_idx == 0 and sorted_data[0] == 0:
        return 0, 0
    percentile = (non_zero_idx / len(data)) * 100
    return percentile, sorted_data[non_zero_idx]


def main():
    np.random.seed(MONTE_CARLO_SEED)

    # Get user inputs
    AV = float(input("Enter the Asset Value (AV): "))
    EF = float(input("Enter the Exposure Factor (EF) between 0 and 1: "))
    ARO = float(input("Enter the Annual Rate of Occurrence (ARO): "))
    reduction_percentage = float(
        input("Enter the Percentage reduction after controls (%): ")
    )

    # Calculate SLE and ALE
    SLE = AV * EF
    ALE = ARO * SLE

    # Adjusted ARO after controls
    adjusted_ARO = ARO * (1 - reduction_percentage / 100)

    # Parameters for beta distribution to adjust kurtosis
    alpha, beta_param = get_beta_parameters_for_kurtosis(KURTOSIS)

    # Monte Carlo simulation for EF with adjusted kurtosis
    simulated_EF = beta(a=alpha, b=beta_param).rvs(NUM_SIMULATIONS)

    # Simulate ARO distributions
    simulated_ARO = poisson(mu=ARO).rvs(NUM_SIMULATIONS)
    simulated_adjusted_ARO = poisson(mu=adjusted_ARO).rvs(NUM_SIMULATIONS)

    # Calculate losses
    losses = AV * simulated_EF * simulated_ARO
    adjusted_losses = AV * simulated_EF * simulated_adjusted_ARO

    # Calculate statistics
    stats = {
        "Mean": np.mean(losses),
        "Std Dev": np.std(losses),
        "1st Percentile": np.percentile(losses, 1),
        "99th Percentile": np.percentile(losses, 99),
        "CI 95% Lower": np.percentile(losses, 2.5),
        "CI 95% Upper": np.percentile(losses, 97.5),
    }

    adjusted_stats = {
        "Mean": np.mean(adjusted_losses),
        "Std Dev": np.std(adjusted_losses),
        "1st Percentile": np.percentile(adjusted_losses, 1),
        "99th Percentile": np.percentile(adjusted_losses, 99),
        "CI 95% Lower": np.percentile(adjusted_losses, 2.5),
        "CI 95% Upper": np.percentile(adjusted_losses, 97.5),
    }

    # Add first non-zero percentile calculation
    first_nonzero_pct, first_nonzero_val = find_first_non_zero_percentile(
        adjusted_losses
    )
    adjusted_stats["First Non-Zero Percentile"] = first_nonzero_pct
    adjusted_stats["First Non-Zero Value"] = first_nonzero_val

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # Plot Risk Distributions (Histograms)
    bins = np.linspace(0, max(losses.max(), adjusted_losses.max()), 50)
    ax1.hist(losses, bins=bins, alpha=0.5, color="blue", label="Before Controls")
    ax1.hist(
        adjusted_losses, bins=bins, alpha=0.5, color="green", label="After Controls"
    )

    # Compute and plot KDE curves
    kde_losses = gaussian_kde(losses)
    kde_adjusted_losses = gaussian_kde(adjusted_losses)
    x_values = np.linspace(0, bins[-1], 1000)

    ax1.plot(
        x_values,
        kde_losses(x_values) * NUM_SIMULATIONS * np.diff(bins)[0],
        color="blue",
    )
    ax1.plot(
        x_values,
        kde_adjusted_losses(x_values) * NUM_SIMULATIONS * np.diff(bins)[0],
        color="green",
    )

    # Disable scientific notation for Risk Distribution
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ",")))
    ax1.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, p: f'${format(int(x), ",")}')
    )

    ax1.set_title("Risk Distribution")
    ax1.set_xlabel("Total Loss")
    ax1.set_ylabel("Frequency")
    ax1.legend()
    ax1.grid(True)

    # Plot Loss Exceedance Curves
    sorted_losses = np.sort(losses)
    exceedance_prob = 100 * (1.0 - np.arange(1, NUM_SIMULATIONS + 1) / NUM_SIMULATIONS)

    sorted_adjusted_losses = np.sort(adjusted_losses)
    adjusted_exceedance_prob = 100 * (
        1.0 - np.arange(1, NUM_SIMULATIONS + 1) / NUM_SIMULATIONS
    )

    ax2.plot(sorted_losses, exceedance_prob, color="blue", label="Before Controls")
    ax2.plot(
        sorted_adjusted_losses,
        adjusted_exceedance_prob,
        color="green",
        label="After Controls",
    )

    # Format Loss Exceedance Curve axes
    ax2.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, p: f'${format(int(x), ",")}')
    )
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: "{:.0f}%".format(y)))

    ax2.set_title("Loss Exceedance Curve")
    ax2.set_xlabel("Total Loss")
    ax2.set_ylabel("Probability of Exceedance (%)")
    ax2.legend()
    ax2.grid(True)

    # Format y-axis ticks as percentages
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: "{:.0f}%".format(y)))

    # Add statistics tables
    before_controls_text = "\n".join(
        [
            "Statistical Summary (Before Controls)",
            "-------------------------------------",
            f'Mean: ${stats["Mean"]:,.2f}',
            f'Std Dev: ${stats["Std Dev"]:,.2f}',
            f'1st Percentile: ${stats["1st Percentile"]:,.2f}',
            f'99th Percentile: ${stats["99th Percentile"]:,.2f}',
            f'95% CI: ${stats["CI 95% Lower"]:,.2f} - ${stats["CI 95% Upper"]:,.2f}',
        ]
    )

    after_controls_lines = [
        "Statistical Summary (After Controls)",
        "------------------------------------",
        f'Mean: ${adjusted_stats["Mean"]:,.2f}',
        f'Std Dev: ${adjusted_stats["Std Dev"]:,.2f}',
    ]

    # Conditionally add the 1st percentile if it is non-zero (otherwise use the first non-zero value)
    if adjusted_stats["1st Percentile"] > 0:
        after_controls_lines.append(f'1st Percentile: ${adjusted_stats["1st Percentile"]:,.2f}')
        after_controls_lines.append(f'CI 95%: ${adjusted_stats["CI 95% Lower"]:,.2f} - ${adjusted_stats["CI 95% Upper"]:,.2f}')
    else:
        after_controls_lines.append(f'{adjusted_stats["First Non-Zero Percentile"]:.1f}th Percentile: ${adjusted_stats["First Non-Zero Value"]:,.2f}')
        after_controls_lines.append(f'CI {adjusted_stats["First Non-Zero Percentile"]:.1f}%-95%: ${adjusted_stats["First Non-Zero Value"]:,.2f} - ${adjusted_stats["CI 95% Upper"]:,.2f}')

    after_controls_lines.extend([
        f'99th Percentile: ${adjusted_stats["99th Percentile"]:,.2f}',
    ])

    after_controls_text = "\n".join(after_controls_lines)

    # Display user input parameters
    input_text = "\n".join(
        [
            "Input Parameters",
            "---------------",
            f"Asset Value (AV): ${AV:,.2f}",
            f"Exposure Factor (EF): {EF:.2%}",
            f"Annualized Rate of Occurrence (ARO): {ARO:.2f}",
            f"Control Frequency Reduction: {reduction_percentage:.0f}%",
            f"Single Loss Expectancy (SLE): ${SLE:,.2f}",
            f"Annualized Loss Expectancy (ALE): ${ALE:,.2f}",
        ]
    )

    # Position the three tables
    plt.figtext(
        0.05,  # Left side
        0.02,
        before_controls_text,
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
    )

    plt.figtext(
        0.35,  # Center
        0.02,
        after_controls_text,
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
    )

    plt.figtext(
        0.65,  # Right side
        0.02,
        input_text,
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
    )

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Make room for the tables
    plt.show()


if __name__ == "__main__":
    main()
