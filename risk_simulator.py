import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import beta, poisson, gaussian_kde
import json

MONTE_CARLO_SEED = 42
NUM_SIMULATIONS = 10000
KURTOSIS = 1.7  # Default value is 3


def get_beta_parameters_for_kurtosis(kurtosis: int) -> tuple[float, float]:
    """Helper function to estimate parameters 'a' and 'b' for the beta distribution to achieve a desired kurtosis.

    This is a simplified heuristic mapping of kurtosis values to beta distribution parameters since the actual implementation involves math way beyond my pay grade/education level, and also runs too slowly for interactive use.

    Args:
        kurtosis: The desired kurtosis value for the distribution.

    Returns:
        tuple[float, float]: The parameters 'a' and 'b' for the beta distribution.
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


def find_first_non_zero_percentile(data: np.ndarray) -> tuple[float, float]:
    """Helper function to find the first non-zero value and its percentile in a data array. The data does not need to be sorted as the function will do it for you.

    Args:
        data: numpy array of numeric values to analyze

    Returns:
        tuple[float, float]: A tuple containing:
            - The percentile (0-100) at which first non-zero value occurs
            - The first non-zero value found

    Raises:
        TypeError: If input is not a numpy array
        ValueError: If array is empty
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if len(data) == 0:
        raise ValueError("Input array cannot be empty")

    sorted_data = np.sort(data)
    non_zero_idx = np.argmax(sorted_data > 0)
    if non_zero_idx == 0 and sorted_data[0] == 0:
        return 0.0, 0.0
    percentile = (non_zero_idx / len(data)) * 100
    return float(percentile), float(sorted_data[non_zero_idx])


def _validate_simulation_params(**kwargs) -> None:
    """Helper function to validate simulation parameters. Raises a ValueError if any parameter is invalid.

    Checks the following parameters:
        - exposure_factor: Must be between 0 and 1
        - reduction_percentage: Must be between 0 and 100 (exclusive)
        - annual_rate_of_occurrence: Must be positive
        - asset_value: Must be positive
        - kurtosis: Must be positive
        - num_simulations: Must be positive
        - monte_carlo_seed: Must be positive
        - cost_of_controls: Must be positive
        - plot: Must be boolean

    Args:
        **kwargs: Arbitrary keyword arguments to be validated.

    Raises:
        ValueError: If any parameter is invalid.
    """
    param_rules = {
        "exposure_factor": (lambda x: 0 <= x <= 1, "must be between 0 and 1"),
        "reduction_percentage": (
            lambda x: 0 < x < 100,
            "must be between 0 and 100 (exclusive)",
        ),
        "annual_rate_of_occurrence": (lambda x: x >= 0, "must be a positive value"),
        "asset_value": (lambda x: x >= 0, "must be a positive value"),
        "kurtosis": (lambda x: x >= 0, "must be a positive value"),
        "num_simulations": (lambda x: x > 0, "must be a positive value"),
        "monte_carlo_seed": (lambda x: x > 0, "must be a positive value"),
        "cost_of_controls": (lambda x: x >= 0, "must be a positive value"),
        "plot": (lambda x: isinstance(x, bool), "must be a boolean value"),
    }

    for param, (rule, message) in param_rules.items():
        if param in kwargs and not rule(kwargs[param]):
            raise ValueError(f"{param} {message}")


def plot_risk_calculation(
    asset_value: float,
    exposure_factor: float,
    annual_rate_of_occurrence: float,
    plot=True,
    monte_carlo_seed: int = MONTE_CARLO_SEED,
    num_simulations: int = NUM_SIMULATIONS,
    kurtosis: int = KURTOSIS,
) -> None | dict:
    """Estimate the mean loss and the effectiveness of risk controls using Monte Carlo simulations for one scenario. This function simulates the potential losses to an asset based on a given exposure factor (EF) and annual rate of occurrence (ARO) for a specific asset value (AV). It provides a risk distribution and a loss exceedance curve. The statistics and simulation results are either plotted or returned as a dictionary.

    This function is similar to plot_risk_calculation_with_controls() but only simulates one scenario where either no controls are applied or controls are applied but the scenario before application is not considered.

    Args:
        asset_value: The value of the asset at risk, expressed in monetary units.
        exposure_factor: The percentage of the asset value that is at risk during a risk event, expressed as a decimal.
        annual_rate_of_occurrence: The frequency of the risk event over a year, expressed as a decimal.
        plot: A boolean indicating whether to plot the results (default is True).
        monte_carlo_seed: The seed for the Monte Carlo simulation to ensure reproducibility (default is constant MONTE_CARLO_SEED).
        num_simulations: The number of simulations to run for the Monte Carlo analysis (default is constant NUM_SIMULATIONS).
        kurtosis: The kurtosis value to adjust the shape of the beta distribution for the EF (default is constant KURTOSIS).

    Returns:
        dict | None: A dictionary containing the statistics, input parameters, and simulation results if plot is False. If plot is True, the function displays the visualizations and tables without returning a dictionary.
    """
    # Input validation
    _validate_simulation_params(
        exposure_factor=exposure_factor,
        annual_rate_of_occurrence=annual_rate_of_occurrence,
        asset_value=asset_value,
        kurtosis=kurtosis,
        num_simulations=num_simulations,
        monte_carlo_seed=monte_carlo_seed,
        plot=plot,
    )

    # Set seed for reproducibility
    np.random.seed(monte_carlo_seed)

    # Calculate SLE and ALE
    single_loss_expectancy = asset_value * exposure_factor
    annualized_loss_expectancy = annual_rate_of_occurrence * single_loss_expectancy

    # Parameters for beta distribution
    alpha, beta_param = get_beta_parameters_for_kurtosis(kurtosis)

    # Monte Carlo simulation for EF with adjusted kurtosis
    simulated_EF = beta(a=alpha, b=beta_param).rvs(num_simulations)
    simulated_ARO = poisson(mu=annual_rate_of_occurrence).rvs(num_simulations)

    # Calculate losses
    losses = asset_value * simulated_EF * simulated_ARO

    # Calculate statistics
    calc_stats = {
        "Mean": np.mean(losses),
        "Median": np.median(losses),
        "Mode": float(stats.mode(np.round(losses))[0]),
        "Std Dev": np.std(losses),
        "1st Percentile": np.percentile(losses, 1),
        "2.5th Percentile": np.percentile(losses, 2.5),
        "5th Percentile": np.percentile(losses, 5),
        "10th Percentile": np.percentile(losses, 10),
        "25th Percentile": np.percentile(losses, 25),
        "75th Percentile": np.percentile(losses, 75),
        "90th Percentile": np.percentile(losses, 90),
        "95th Percentile": np.percentile(losses, 95),
        "97.5th Percentile": np.percentile(losses, 97.5),
        "99th Percentile": np.percentile(losses, 99),
    }

    # Calculate exceedance probabilities
    sorted_losses = np.sort(losses)
    exceedance_prob = 100 * (1.0 - np.arange(1, num_simulations + 1) / num_simulations)

    if plot:
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

        # Plot Risk Distribution
        bins = np.linspace(0, losses.max(), 50)
        ax1.hist(losses, bins=bins, alpha=0.5, color="blue", label="Base Case")

        # Compute and plot KDE curve
        kde_losses = gaussian_kde(losses)
        x_values = np.linspace(0, bins[-1], 1000)
        ax1.plot(
            x_values,
            kde_losses(x_values) * num_simulations * np.diff(bins)[0],
            color="blue",
        )

        # Format axes
        ax1.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: format(int(x), ","))
        )
        ax1.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: f'${format(int(x), ",")}')
        )

        ax1.set_title("Risk Distribution")
        ax1.set_xlabel("Total Loss")
        ax1.set_ylabel("Frequency")
        ax1.legend()
        ax1.grid(True)

        # Plot Loss Exceedance Curve
        ax2.plot(sorted_losses, exceedance_prob, color="blue", label="Base Case")

        # Format axes
        ax2.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: f'${format(int(x), ",")}')
        )
        ax2.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda y, _: "{:.0f}%".format(y))
        )

        ax2.set_title("Loss Exceedance Curve")
        ax2.set_xlabel("Total Loss")
        ax2.set_ylabel("Probability of Exceedance (%)")
        ax2.legend()
        ax2.grid(True)

        # Add statistics table
        statistics_text = "\n".join(
            [
                "Statistical Summary",
                "-------------------",
                f'Mean: ${calc_stats["Mean"]:,.2f}',
                f'Median: ${calc_stats["Median"]:,.2f}',
                f'Mode: ${calc_stats["Mode"]:,.2f}',
                f'Std Dev: ${calc_stats["Std Dev"]:,.2f}',
                f'1st Percentile: ${calc_stats["1st Percentile"]:,.2f}',
                f'95% CI: ${calc_stats["2.5th Percentile"]:,.2f} - ${calc_stats["97.5th Percentile"]:,.2f}',
                f'99th Percentile: ${calc_stats["99th Percentile"]:,.2f}',
            ]
        )

        # Add input parameters table
        parameters_text = "\n".join(
            [
                "Input Parameters",
                "----------------",
                f"Asset Value (AV): ${asset_value:,.2f}",
                f"Exposure Factor (EF): {exposure_factor:.2%}",
                f"Annualized Rate of Occurrence (ARO): {annual_rate_of_occurrence:.2f}",
                f"Single Loss Expectancy (SLE): ${single_loss_expectancy:,.2f}",
                f"Annualized Loss Expectancy (ALE): ${annualized_loss_expectancy:,.2f}",
            ]
        )

        # Position text boxes
        plt.figtext(
            0.15,
            0.02,
            statistics_text,
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
        )
        plt.figtext(
            0.65,
            0.02,
            parameters_text,
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
        )

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25, hspace=0.25)
        plt.show()

    else:
        return {
            "statistics": {
                "mean": float(calc_stats["Mean"]),
                "median": float(calc_stats["Median"]),
                "mode": float(calc_stats["Mode"]),
                "std_dev": float(calc_stats["Std Dev"]),
                "percentile_1": float(calc_stats["1st Percentile"]),
                "percentile_2.5": float(calc_stats["2.5th Percentile"]),
                "percentile_5": float(calc_stats["5th Percentile"]),
                "percentile_10": float(calc_stats["10th Percentile"]),
                "percentile_25": float(calc_stats["25th Percentile"]),
                "percentile_75": float(calc_stats["75th Percentile"]),
                "percentile_90": float(calc_stats["90th Percentile"]),
                "percentile_97.5": float(calc_stats["97.5th Percentile"]),
                "percentile_99": float(calc_stats["99th Percentile"]),
            },
            "input_parameters": {
                "asset_value": float(asset_value),
                "exposure_factor": float(exposure_factor),
                "annual_rate_of_occurrence": float(annual_rate_of_occurrence),
                "single_loss_expectancy": float(single_loss_expectancy),
                "annualized_loss_expectancy": float(annualized_loss_expectancy),
            },
            "simulation_results": {
                "losses": losses.tolist(),
                "exceedance_probabilities": exceedance_prob.tolist(),
            },
        }


def plot_risk_calculation_with_controls(
    asset_value: float,
    exposure_factor: float,
    annual_rate_of_occurrence: float,
    reduction_percentage: float,
    cost_of_controls: float,
    plot=True,
    monte_carlo_seed: int = MONTE_CARLO_SEED,
    num_simulations: int = NUM_SIMULATIONS,
    kurtosis: int = KURTOSIS,
) -> None | dict:
    """Estimate the mean loss and the effectiveness of risk controls using Monte Carlo simulations for a before-and-after scenario.

    This function simulates the potential losses to an asset based on a given exposure factor (EF) and annual rate of occurrence (ARO) for a specific asset value (AV). It then simulates the impact of risk reduction controls that lower the ARO and calculates how these controls reduce potential losses. It provides a risk distribution and a loss exceedance curve both before and after controls are applied. The statistics and simulation results are either plotted or returned as a dictionary.

    The simulation uses:
        - **Beta Distribution**: To simulate the Exposure Factor (EF), representing the percentage of asset value that is at risk during a risk event. The Beta distribution allows for variability in the EF, with the distribution's shape controlled by parameters that introduce kurtosis (peakedness).
        - **Poisson Distribution**: To simulate the Annual Rate of Occurrence (ARO), representing the frequency of the risk event over a year. It models the randomness of how often the event will occur. After applying controls, the ARO is reduced by a specified percentage and recalculated using the Poisson distribution.

    The plot shows two main visualizations and four statistical tables:
        - **Risk Distribution**: A histogram showing the distribution of potential losses before and after controls. It also includes Kernel Density Estimation (KDE) curves to visualize the probability density of the losses.
        - **Loss Exceedance Curve**: A line plot showing the cumulative probability of exceeding a given loss amount before and after controls. It helps visualize the likelihood of different loss scenarios.
        - **Statistical Summary Table**: Displays key statistics such as Mean, Median, Mode, Standard Deviation, Confidence Interval, and Percentiles (1%, 99%) of losses before and after controls. For the after control table, it dynamically uses *the first non-zero percentile and value* if the 2.5th percentile in the 95% CI is zero.
        - **Input Parameters Table**: Shows the user-provided input parameters such as Asset Value (AV), Exposure Factor (EF), Annual Rate of Occurrence (ARO), Single Loss Expectancy (SLE), and Annualized Loss Expectancy (ALE) before controls.
        - **Calculated Parameters Table**: Displays the Adjusted ARO after controls, Adjusted ALE, and the maximum acceptable cost of implementing controls based on the expected reduction in loss.

    The JSON output includes:
        - **Statistics**: Mean, Standard Deviation, 1st Percentile, 95% Confidence Interval (CI), and 99th Percentile of losses before and after controls. It also includes the first non-zero percentile and value after controls.
        - **Input Parameters**: Asset Value (AV), Exposure Factor (EF), Annual Rate of Occurrence (ARO), Single Loss Expectancy (SLE), and Annualized Loss Expectancy (ALE).
        - **Calculated Parameters**: Adjusted ARO after controls, Adjusted ALE, and the expected reduction in loss due to controls.
        - **Simulation Results**: Lists of simulated losses and exceedance probabilities before and after controls.

    Args:
        asset_value: The value of the asset at risk, expressed in monetary units.
        exposure_factor: The percentage of the asset value that is at risk during a risk event, expressed as a decimal.
        annual_rate_of_occurrence: The frequency of the risk event over a year, expressed as a decimal.
        reduction_percentage: The percentage reduction in the ARO after applying risk controls, expressed as a percentage.
        cost_of_controls: The cost of implementing the risk controls, expressed in monetary units.
        plot: A boolean indicating whether to plot the results (default is True).
        monte_carlo_seed: The seed for the Monte Carlo simulation to ensure reproducibility (default is constant MONTE_CARLO_SEED).
        num_simulations: The number of simulations to run for the Monte Carlo analysis (default is constant NUM_SIMULATIONS).
        kurtosis: The kurtosis value to adjust the shape of the beta distribution for the EF (default is constant KURTOSIS).

    Returns:
        dict | None: A dictionary containing the statistics, input parameters, calculated parameters, and simulation results if plot is False. If plot is True, the function displays the visualizations and tables without returning a dictionary.
    """
    # Input validation
    _validate_simulation_params(
        exposure_factor=exposure_factor,
        reduction_percentage=reduction_percentage,
        annual_rate_of_occurrence=annual_rate_of_occurrence,
        asset_value=asset_value,
        cost_of_controls=cost_of_controls,
        kurtosis=kurtosis,
        num_simulations=num_simulations,
        monte_carlo_seed=monte_carlo_seed,
        plot=plot,
    )

    # Set seed for reproducibility
    np.random.seed(monte_carlo_seed)

    # Calculate SLE and ALE
    single_loss_expectancy = asset_value * exposure_factor
    annualized_loss_expectancy = annual_rate_of_occurrence * single_loss_expectancy

    # Adjusted ARO after controls
    adjusted_ARO = annual_rate_of_occurrence * (1 - reduction_percentage / 100)

    # Parameters for beta distribution to adjust kurtosis
    alpha, beta_param = get_beta_parameters_for_kurtosis(kurtosis)

    # Monte Carlo simulation for EF with adjusted kurtosis
    simulated_EF = beta(a=alpha, b=beta_param).rvs(num_simulations)

    # Simulate ARO distributions
    simulated_ARO = poisson(mu=annual_rate_of_occurrence).rvs(num_simulations)
    simulated_adjusted_ARO = poisson(mu=adjusted_ARO).rvs(num_simulations)

    # Calculate losses
    losses = asset_value * simulated_EF * simulated_ARO
    adjusted_losses = asset_value * simulated_EF * simulated_adjusted_ARO

    # Calculate statistics before controls
    calc_stats = {
        "Mean": np.mean(losses),
        "Median": np.median(losses),  # Also 50th percentile
        "Mode": float(stats.mode(np.round(losses))[0]),
        "Std Dev": np.std(losses),
        "1st Percentile": np.percentile(losses, 1),
        "2.5th Percentile": np.percentile(losses, 2.5),
        "5th Percentile": np.percentile(losses, 5),
        "10th Percentile": np.percentile(losses, 10),
        "25th Percentile": np.percentile(losses, 25),
        "75th Percentile": np.percentile(losses, 75),
        "90th Percentile": np.percentile(losses, 90),
        "95th Percentile": np.percentile(losses, 95),
        "97.5th Percentile": np.percentile(losses, 97.5),
        "99th Percentile": np.percentile(losses, 99),
    }

    # Calculate statistics after controls
    calc_adjusted_stats = {
        "Mean": np.mean(adjusted_losses),
        "Median": np.median(adjusted_losses),
        "Mode": float(stats.mode(np.round(adjusted_losses))[0]),
        "Std Dev": np.std(adjusted_losses),
        "1st Percentile": np.percentile(adjusted_losses, 1),
        "2.5th Percentile": np.percentile(adjusted_losses, 2.5),
        "5th Percentile": np.percentile(adjusted_losses, 5),
        "10th Percentile": np.percentile(adjusted_losses, 10),
        "25th Percentile": np.percentile(adjusted_losses, 25),
        "75th Percentile": np.percentile(adjusted_losses, 75),
        "90th Percentile": np.percentile(adjusted_losses, 90),
        "95th Percentile": np.percentile(adjusted_losses, 95),
        "97.5th Percentile": np.percentile(adjusted_losses, 97.5),
        "99th Percentile": np.percentile(adjusted_losses, 99),
    }

    # Calculate expected reduction in loss
    expected_loss_before = np.mean(losses)
    expected_loss_after = np.mean(adjusted_losses)
    expected_reduction_in_loss = expected_loss_before - expected_loss_after

    # Calculate ROSI
    benefit = expected_reduction_in_loss
    if cost_of_controls > 0:
        rosi_percentage = (benefit - cost_of_controls) / cost_of_controls * 100
    else:
        rosi_percentage = float('inf')

    # For after controls, 2.5% percentile is likely to be zero due to the reduction
    # Calculate the first non-zero percentile and value to show in the table instead
    first_nonzero_pct, first_nonzero_val = find_first_non_zero_percentile(
        adjusted_losses
    )
    calc_adjusted_stats["First Non-Zero Percentile"] = first_nonzero_pct
    calc_adjusted_stats["First Non-Zero Value"] = first_nonzero_val

    # Calculate loss exceedance probabilities values
    sorted_losses = np.sort(losses)
    exceedance_prob = 100 * (1.0 - np.arange(1, num_simulations + 1) / num_simulations)

    # Filter out zeros for adjusted losses
    nonzero_adjusted_losses = adjusted_losses[adjusted_losses > 0]
    sorted_adjusted_losses = np.sort(nonzero_adjusted_losses)
    adjusted_exceedance_prob = 100 * (
        1.0
        - np.arange(1, len(nonzero_adjusted_losses) + 1) / len(nonzero_adjusted_losses)
    )

    # If plot is True, plot the risk distribution and loss exceedance curve
    if plot:
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

        # Plot Risk Distribution
        bins = np.linspace(0, max(losses.max(), adjusted_losses.max()), 50)
        ax1.hist(losses, bins=bins, alpha=0.5, color="blue", label="Before Controls")
        ax1.hist(
            adjusted_losses,
            bins=bins,
            alpha=0.5,
            color="green",
            label="After Controls",
        )

        # Compute and plot KDE curves
        kde_losses = gaussian_kde(losses)
        kde_adjusted_losses = gaussian_kde(adjusted_losses)
        x_values = np.linspace(0, bins[-1], 1000)
        ax1.plot(
            x_values,
            kde_losses(x_values) * num_simulations * np.diff(bins)[0],
            color="blue",
        )
        ax1.plot(
            x_values,
            kde_adjusted_losses(x_values) * num_simulations * np.diff(bins)[0],
            color="green",
        )

        # Disable scientific notation for Risk Distribution
        ax1.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: format(int(x), ","))
        )
        ax1.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: f'${format(int(x), ",")}')
        )

        ax1.set_title("Risk Distribution")
        ax1.set_xlabel("Total Loss")
        ax1.set_ylabel("Frequency")
        ax1.legend()
        ax1.grid(True)

        # Plot Loss Exceedance Curves
        ax2.plot(sorted_losses, exceedance_prob, color="blue", label="Before Controls")
        ax2.plot(
            sorted_adjusted_losses,
            adjusted_exceedance_prob,
            color="green",
            label="After Controls",
        )

        # Format Loss Exceedance Curve axes as non-scientific notation
        ax2.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: f'${format(int(x), ",")}')
        )
        ax2.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda y, _: "{:.0f}%".format(y))
        )

        ax2.set_title("Loss Exceedance Curve")
        ax2.set_xlabel("Total Loss")
        ax2.set_ylabel("Probability of Exceedance (%)")
        ax2.legend()
        ax2.grid(True)

        # Format y-axis ticks as percentages
        ax2.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda y, _: "{:.0f}%".format(y))
        )

        # Add statistics tables
        before_controls_text = "\n".join(
            [
                "Statistical Summary (Before Controls)",
                "-------------------------------------",
                f'Mean: ${calc_stats["Mean"]:,.2f}',
                f'Median: ${calc_stats["Median"]:,.2f}',
                f'Mode: ${calc_stats["Mode"]:,.2f}',
                f'Std Dev: ${calc_stats["Std Dev"]:,.2f}',
                f'1st Percentile: ${calc_stats["1st Percentile"]:,.2f}',
                f'95% CI: ${calc_stats["2.5th Percentile"]:,.2f} - ${calc_stats["97.5th Percentile"]:,.2f}',
                f'99th Percentile: ${calc_stats["99th Percentile"]:,.2f}',
            ]
        )

        after_controls_lines = [
            "Statistical Summary (After Controls)",
            "------------------------------------",
            f'Mean: ${calc_adjusted_stats["Mean"]:,.2f}',
            f'Median: ${calc_adjusted_stats["Median"]:,.2f}',
            f'Mode: ${calc_adjusted_stats["Mode"]:,.2f}',
            f'Std Dev: ${calc_adjusted_stats["Std Dev"]:,.2f}',
        ]

        # Conditionally add the 1st percentile if it is non-zero (otherwise use the first non-zero value)
        if calc_adjusted_stats["1st Percentile"] > 0:
            after_controls_lines.append(
                f'1st Percentile: ${calc_adjusted_stats["1st Percentile"]:,.2f}'
            )
            after_controls_lines.append(
                f'CI 95%: ${calc_adjusted_stats["2.5th Percentile"]:,.2f} - ${calc_adjusted_stats["97.5th Percentile"]:,.2f}'
            )
        else:
            after_controls_lines.append(
                f'{calc_adjusted_stats["First Non-Zero Percentile"]:.1f}th Percentile: ${calc_adjusted_stats["First Non-Zero Value"]:,.2f}'
            )
            after_controls_lines.append(
                f'CI {calc_adjusted_stats["First Non-Zero Percentile"]:.1f}%-95%: ${calc_adjusted_stats["First Non-Zero Value"]:,.2f} - ${calc_adjusted_stats["97.5th Percentile"]:,.2f}'
            )

        after_controls_lines.extend(
            [
                f'99th Percentile: ${calc_adjusted_stats["99th Percentile"]:,.2f}',
            ]
        )

        after_controls_text = "\n".join(after_controls_lines)

        # Display user input parameters before controls
        input_parameters = "\n".join(
            [
                "Input Parameters (Before Controls)",
                "--------------------------------",
                f"Asset Value (AV): ${asset_value:,.2f}",
                f"Exposure Factor (EF): {exposure_factor:.2%}",
                f"Annualized Rate of Occurrence (ARO): {annual_rate_of_occurrence:.2f}",
                f"Single Loss Expectancy (SLE): ${single_loss_expectancy:,.2f}",
                f"Annualized Loss Expectancy (ALE): ${annualized_loss_expectancy:,.2f}",
            ]
        )

        # Display parameters after controls
        calculated_parameters = "\n".join(
            [
                "Calculated Parameters (After Controls)",
                "-------------------------------",
                f"ARO after {reduction_percentage:.0f}% reduction: {adjusted_ARO:.2f}",
                f"Annualized Loss Expectancy (ALE): ${annualized_loss_expectancy * (1 - reduction_percentage/100):,.2f}",
                f"Expected Benefit: ${benefit:,.2f}",
                f"Cost of Controls: ${cost_of_controls:,.2f}",
                f"ROSI: {rosi_percentage:,.2f}%",
            ]
        )

        # Update the figtext positioning
        plt.figtext(
            0.05,  # Left side
            0.02,
            before_controls_text,
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
        )

        plt.figtext(
            0.30,  # Center-left
            0.02,
            after_controls_text,
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
        )

        plt.figtext(
            0.55,  # Center-right
            0.02,
            input_parameters,
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
        )

        plt.figtext(
            0.80,  # Right side
            0.02,
            calculated_parameters,
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
        )

        # Adjust layout to prevent overlap
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25, hspace=0.25)  # Make room for the tables
        plt.show()

    # If plot is False, return the statistics
    else:
        return {
            "before_controls_stats": {
                "mean": float(calc_stats["Mean"]),
                "median": float(calc_stats["Median"]),
                "mode": float(calc_stats["Mode"]),
                "std_dev": float(calc_stats["Std Dev"]),
                "percentile_1": float(calc_stats["1st Percentile"]),
                "percentile_2.5": float(calc_stats["2.5th Percentile"]),
                "percentile_5": float(calc_stats["5th Percentile"]),
                "percentile_10": float(calc_stats["10th Percentile"]),
                "percentile_25": float(calc_stats["25th Percentile"]),
                "percentile_75": float(calc_stats["75th Percentile"]),
                "percentile_90": float(calc_stats["90th Percentile"]),
                "percentile_97.5": float(calc_stats["97.5th Percentile"]),
                "percentile_99": float(calc_stats["99th Percentile"]),
            },
            "after_controls_stats": {
                "mean": float(calc_adjusted_stats["Mean"]),
                "median": float(calc_adjusted_stats["Median"]),
                "mode": float(calc_adjusted_stats["Mode"]),
                "std_dev": float(calc_adjusted_stats["Std Dev"]),
                "percentile_1": float(calc_adjusted_stats["1st Percentile"]),
                "percentile_2.5": float(calc_adjusted_stats["2.5th Percentile"]),
                "percentile_5": float(calc_adjusted_stats["5th Percentile"]),
                "percentile_10": float(calc_adjusted_stats["10th Percentile"]),
                "percentile_25": float(calc_adjusted_stats["25th Percentile"]),
                "percentile_75": float(calc_adjusted_stats["75th Percentile"]),
                "percentile_90": float(calc_adjusted_stats["90th Percentile"]),
                "percentile_97.5": float(calc_adjusted_stats["97.5th Percentile"]),
                "percentile_99": float(calc_adjusted_stats["99th Percentile"]),
                "first_nonzero_percentile": float(
                    calc_adjusted_stats["First Non-Zero Percentile"]
                ),
                "first_nonzero_value": float(
                    calc_adjusted_stats["First Non-Zero Value"]
                ),
            },
            "input_parameters": {
                "asset_value": float(asset_value),
                "exposure_factor": float(exposure_factor),
                "annual_rate_of_occurrence": float(annual_rate_of_occurrence),
                "single_loss_expectancy": float(single_loss_expectancy),
                "annualized_loss_expectancy": float(annualized_loss_expectancy),
            },
            "calculated_parameters": {
                "adjusted_aro": round(adjusted_ARO, 4),
                "adjusted_ale": round(
                    annualized_loss_expectancy * (1 - reduction_percentage / 100), 2
                ),
                "reduction_percentage": round(reduction_percentage, 2),
                "expected_benefit": round(benefit, 2),
                "cost_of_controls": round(cost_of_controls, 2),
                "rosi_percentage": round(rosi_percentage, 2),
            },
            "simulation_results": {
                "losses": losses.tolist(),
                "adjusted_losses": nonzero_adjusted_losses.tolist(),  # Filter out zero values
                "exceedance_probabilities": exceedance_prob.tolist(),
                "adjusted_exceedance_probabilities": adjusted_exceedance_prob.tolist(),
            },
        }


def main():
    # Get user inputs
    # AV = float(input("Enter the Asset Value (AV): "))
    # EF = float(input("Enter the Exposure Factor (EF) between 0 and 1: "))
    # ARO = float(input("Enter the Annual Rate of Occurrence (ARO): "))
    # reduction_percentage = float(
    #     input("Enter the Percentage reduction after controls (%): ")
    # )
    # control_cost = float(input("Enter the cost of implementing controls: "))

    # Hardcoded inputs for testing
    AV, EF, ARO, reduction_percentage, control_cost = 100000, 0.5, 5, 80, 10000

    # Plot the risk calculation with controls
    test = plot_risk_calculation_with_controls(
        AV, EF, ARO, reduction_percentage, control_cost, plot=True
    )
    # with open("risk_simulation_results.json", "w") as f:
    #     json.dump(test, f, indent=4)


if __name__ == "__main__":
    main()
