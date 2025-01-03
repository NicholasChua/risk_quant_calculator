import numpy as np
from scipy.stats import beta, poisson
from itertools import permutations
import matplotlib.pyplot as plt
import json
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze

from risk_simulator import (
    get_beta_parameters_for_kurtosis,
)

RANDOM_SEED = 42
ASSET_VALUE = 500000
EF_RANGE = [0.4, 0.6]  # Exposure factor range
ARO_RANGE = [1.5, 2.5]  # Annual rate of occurrence range
CONTROL_REDUCTIONS = [0.2, 0.35, 0.3, 0.45]  # Reduction percentages for controls
CONTROL_COSTS = [10000, 30000, 20000, 35000]  # Base annualized cost of controls
COST_ADJUSTMENT_RANGE = [0.0, 0.2]  # Cost adjustment per year range
NUM_SAMPLES = 16384  # Number of Sobol samples 2^14
NUM_YEARS = len(CONTROL_COSTS)  # Number of years to simulate
KURTOSIS = 1.7  # Results in alpha and beta of 0.5


# TODO: Move this to a common library file
def calculate_ale(
    asset_value: float, exposure_factor: float, annual_rate_of_occurrence: float
) -> float:
    """Calculate the annualized loss expectancy (ALE) given the asset value, exposure factor, and annual rate of occurrence.

    Args:
        asset_value: Asset value
        exposure_factor: Exposure factor
        annual_rate_of_occurrence: Annual rate of occurrence

    Returns:
        float: Annualized loss expectancy
    """
    return asset_value * exposure_factor * annual_rate_of_occurrence


def calculate_rosi(
    ale_before: float, ale_after: float, control_costs: list[float] | float
) -> float:
    """Calculate the return on security investment (ROSI) given the ALE before and after implementing controls and the control costs.

    Args:
        ale_before: Annualized loss expectancy before implementing controls
        ale_after: Annualized loss expectancy after implementing controls
        control_costs: Annualized cost of controls (either a single float or a list of floats)

    Returns:
        float: Return on security investment
    """
    if isinstance(control_costs, float):
        total_control_costs = control_costs
    else:
        total_control_costs = sum(control_costs)

    return (ale_before - ale_after - total_control_costs) / total_control_costs * 100


def simulate_exposure_factor_sobol(
    sobol_samples: np.ndarray, ef_range: list[float], kurtosis: int
) -> np.ndarray:
    """Simulate exposure factor using Sobol samples mapped to a Beta distribution with a specified kurtosis.

    Args:
        sobol_samples: Sobol samples
        ef_range: Exposure factor range
        kurtosis: Kurtosis of the distribution

    Returns:
        np.ndarray: Simulated exposure factors
    """
    # To avoid beta.ppf(0 or 1) = inf/NaN, clamp the samples:
    eps = 1e-9
    sobol_samples = np.clip(sobol_samples, eps, 1 - eps)

    # Calculate Beta parameters based on kurtosis
    alpha_param, beta_param = get_beta_parameters_for_kurtosis(kurtosis)

    # Map Sobol samples to Beta distribution using inverse CDF quantile function
    beta_samples = beta.ppf(sobol_samples, alpha_param, beta_param)

    # Scale Beta samples to the desired range
    exposure_factors = ef_range[0] + beta_samples * (ef_range[1] - ef_range[0])

    return exposure_factors


def simulate_annual_rate_of_occurrence_sobol(
    sobol_samples: np.ndarray, aro_range: list[float], decimal_places: int = 2
) -> np.ndarray:
    """Simulate annual rate of occurrence using Sobol samples mapped to a Poisson distribution with specified decimal precision. As Poisson distribution accepts only integer values, the Sobol samples are scaled up to the desired precision, mapped to the Poisson distribution, and then scaled back to the original scale, bypassing the decimal limitation.

    Args:
        sobol_samples: Sobol samples
        aro_range: Annual rate of occurrence range
        decimal_places: Number of decimal places for the generated values (default is 2)

    Returns:
        np.ndarray: Simulated annual rates of occurrence
    """
    # To avoid poisson.ppf(0 or 1) = inf/NaN, clamp the samples:
    eps = 1e-9
    sobol_samples = np.clip(sobol_samples, eps, 1 - eps)

    # Scale factor for desired decimal precision
    scale_factor = 10**decimal_places

    # Scale Sobol samples to integers
    scaled_sobol_samples = sobol_samples * scale_factor

    # Map Sobol samples to Poisson distribution using inverse CDF quantile function
    poisson_samples = poisson.ppf(
        scaled_sobol_samples / scale_factor, mu=np.mean(aro_range) * scale_factor
    )

    # Convert back to the original scale
    aro_samples = poisson_samples / scale_factor

    # Clip the values to the desired range
    aro_samples = np.clip(aro_samples, aro_range[0], aro_range[1])

    return aro_samples


# TODO: The compounding cost calculation and adjustment selection from sobol is broken. Fix this.
def calculate_compounding_costs(
    control_costs: list[float], cost_adjustment_range: list[float], years: int, num_samples: int = NUM_SAMPLES
) -> dict[str, dict[str, dict[str, np.float64]]]:
    """Given a list of control costs, a cost adjustment range, and a number of years, calculate the cost for each control for each year

    Args:
        control_costs: Base annualized cost of controls
        cost_adjustment_range: Cost adjustment per year range
        years: Number of years to simulate
        num_samples: Number of samples to simulate. Default is NUM_SAMPLES

    Returns:
        dict[str, dict[str, dict[str, np.float64]]]: Dictionary of costs and adjustments for each control, for each year
    """
    # Initialize costs dictionary
    costs = {}

    # Define the problem for cost adjustments using setup_sensitivity_problem
    problem = setup_sensitivity_problem(
        **{f"cost_adj_{i}": cost_adjustment_range for i in range(len(control_costs) * years)}
    )

    # Generate Sobol samples for cost adjustments
    sobol_samples = sobol_sample.sample(problem, num_samples, calc_second_order=False, seed=RANDOM_SEED)

    # Year 0 is the base cost and has no adjustment
    costs["year_0"] = {
        f"control_{index + 1}": {
            "cost": np.float64(control_cost),
            "adjustment": np.float64(0.0),
        }
        for index, control_cost in enumerate(control_costs)
    }

    # Calculate costs for each year on a compounding basis for each control
    for year in range(1, years + 1):
        costs[f"year_{year}"] = {}
        for index, control_cost in enumerate(control_costs):
            # Use Sobol samples to adjust costs. (number of years) * (number of controls) samples are generated
            sample_index = (year - 1) * len(control_costs) + index
            # Calculate adjustment based on Sobol sample
            adjustment = np.float64(
                sobol_samples[sample_index][0]
                * (cost_adjustment_range[1] - cost_adjustment_range[0])
                + cost_adjustment_range[0]
            )
            # Calculate new cost based on adjustment
            costs[f"year_{year}"][f"control_{index + 1}"] = {
                "cost": np.float64(control_cost + (control_cost * adjustment)),
                "adjustment": adjustment,
            }
    return costs


def calculate_statistics_for_permutation_per_year(
    costs: dict[str, dict[str, dict[str, np.float64]]],
    ef_samples: np.ndarray,
    aro_samples: np.ndarray,
    permutation: tuple[int],
    control_reductions: list = CONTROL_REDUCTIONS,
    num_of_simulations: int = NUM_SAMPLES,
) -> dict[str, float]:
    """Given a permutation of controls, calculate the ROSI for each year and the total ROSI for the permutation

    Args:
        costs: Dictionary of costs and adjustments for each control, for each year
        ef_samples: Simulated exposure factors
        aro_samples: Simulated annual rates of occurrence
        permutation: A single permutation of controls represented as a tuple of integers
        control_reductions: List of control reduction percentages. Default is CONTROL_REDUCTIONS
        num_of_simulations: Number of samples to simulate. Default is NUM_SAMPLES

    Returns:
        dict[str, float]: Dictionary of results for the permutation
    """
    # Initialize ROSI
    rosi_per_simulation = []

    for i in range(num_of_simulations):
        rosi_per_year = []
        results = {
            "permutation": permutation,
            "total_rosi": 0.0,
        }

        for year in range(len(permutation)):
            # Retrieve the appropriate control and its cost for the year
            control = permutation[year]
            control_cost = costs[f"year_{year + 1}"][f"control_{control}"]["cost"]

            # Calculate the total cost by summing the costs of all controls up to the current year, inclusive
            total_cost = sum(
                [
                    costs[f"year_{year + 1}"][f"control_{control}"]["cost"]
                    for control in permutation[: year + 1]
                ]
            )

            # Calculate the new ARO after applying the control
            aro_after = aro_samples[i][year] * (1 - control_reductions[control - 1])

            # Calculate the new ALE after applying the control
            ale_after = calculate_ale(ASSET_VALUE, ef_samples[i][year], aro_after)

            # Calculate the ALE before applying the control
            if year == 0:
                ale_before = calculate_ale(
                    ASSET_VALUE, ef_samples[i][year], aro_samples[i][year]
                )
            else:
                ale_before = results[f"year_{year}"]["ale_after"]

            # Calculate the ROSI after applying the control and save it
            rosi = calculate_rosi(ale_before, ale_after, total_cost)
            rosi_per_year.append(rosi)

            # Add year-by-year information to the results
            results[f"year_{year + 1}"] = {
                "ale_before": ale_before,
                "ale_after": ale_after,
                "control_cost": control_cost,
                "total_cost": total_cost,
                "rosi": rosi,
            }

        # Calculate the total ROSI for the permutation
        results["total_rosi"] = np.sum(rosi_per_year)
        rosi_per_simulation.append(results["total_rosi"])

    # Average the ROSI over all simulations
    average_rosi = np.mean(rosi_per_simulation)
    results["total_rosi"] = average_rosi

    return results


def calculate_statistics_for_permutation_aggregate(
    costs: dict[str, dict[str, dict[str, np.float64]]],
    ef_samples: np.ndarray,
    aro_samples: np.ndarray,
    permutation: tuple[int],
    control_reductions: list = CONTROL_REDUCTIONS,
    num_of_simulations: int = NUM_SAMPLES,
) -> dict[str, float]:
    """Given a permutation of controls, calculate the ROSI for the entire period

    Args:
        costs: Dictionary of costs and adjustments for each control, for each year
        ef_samples: Simulated exposure factors
        aro_samples: Simulated annual rates of occurrence
        permutation: A single permutation of controls represented as tuple of integers
        control_reductions: List of control reduction percentages
        num_of_simulations: Number of samples to simulate

    Returns:
        dict[str, float]: Dictionary with permutation and total ROSI
    """
    rosi_per_simulation = []

    for i in range(num_of_simulations):
        results = {
            "permutation": permutation,
            "total_rosi": 0.0,
        }

        # Calculate initial ALE and yearly data
        for year in range(len(permutation)):
            control = permutation[year]
            control_cost = costs[f"year_{year + 1}"][f"control_{control}"]["cost"]

            # Calculate total cost up to this year
            total_cost = sum(
                costs[f"year_{y + 1}"][f"control_{permutation[y]}"]["cost"]
                for y in range(year + 1)
            )

            # Calculate ALE before and after for this year
            ale_before = calculate_ale(
                ASSET_VALUE, ef_samples[i][year], aro_samples[i][year]
            )
            reduction = 1 - control_reductions[control - 1]
            ale_after = calculate_ale(
                ASSET_VALUE, ef_samples[i][year], aro_samples[i][year] * reduction
            )

            # Store year data
            results[f"year_{year + 1}"] = {
                "ale_before": ale_before,
                "ale_after": ale_after,
                "control_cost": control_cost,
                "total_cost": total_cost,
            }

        # Calculate aggregate ROSI
        total_ale_before = sum(
            results[f"year_{y + 1}"]["ale_before"] for y in range(len(permutation))
        )
        total_ale_after = sum(
            results[f"year_{y + 1}"]["ale_after"] for y in range(len(permutation))
        )
        total_costs = results[f"year_{len(permutation)}"]["total_cost"]

        rosi = calculate_rosi(total_ale_before, total_ale_after, total_costs)
        rosi_per_simulation.append(rosi)

    # Set final ROSI value
    results["total_rosi"] = float(np.mean(rosi_per_simulation))

    return results


def convert_to_serializable(obj: any) -> any:
    """Recursively converts numpy arrays, dictionaries, and lists into serializable formats.

    Args:
        obj: The object to be converted. It can be a numpy array, dictionary, list, or any other type.

    Returns:
        any: The converted object in a serializable format. Numpy arrays are converted to lists, dictionaries are recursively processed, and lists are recursively processed. Other types are returned as is.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    else:
        return obj


def setup_sensitivity_problem(**kwargs: dict[str, list[float]]) -> dict[str, list[float]]:
    """Define the model inputs and their bounds for sensitivity analysis

    Args:
        **kwargs: Dictionary of parameter names and their bounds in a list with two floats

    Returns:
        dict[str, list[float]]: Problem dictionary for sensitivity analysis

    Raises:
        ValueError: If the parameter values are not lists of two floats
    """
    # Verify kwargs is a dictionary with the correct type
    if not isinstance(kwargs, dict):
        raise ValueError("Input parameters must be provided as a dictionary")

    # Verify input parameters contain only two floats
    for key, value in kwargs.items():
        if not isinstance(key, str):
            raise ValueError(f"Parameter name {key} must be a string")
        if not isinstance(value, list) or len(value) != 2 or not all(isinstance(i, float) for i in value):
            raise ValueError(f"Parameter {key} must be a list of two floats")

    names = list(kwargs.keys())
    bounds = list(kwargs.values())

    problem = {
        "num_vars": len(names),
        "names": names,
        "bounds": bounds,
    }
    return problem


def evaluate_model(X: np.ndarray) -> np.array:
    """Evaluate model for sensitivity analysis samples. The model calculates the ROSI for each sample. The input samples are expected to be in the format of (EF, ARO, cost_adj).
    
    Args:
        X: Samples generated for sensitivity analysis

    Returns:
        np.array: ROSI values for each sample
    """
    Y = []
    for params in X:
        ef, aro, cost_adj = params

        # Calculate base ALE
        ale = calculate_ale(ASSET_VALUE, ef, aro)

        # Calculate costs with adjustment
        adjusted_costs = [cost * (1 + cost_adj) for cost in CONTROL_COSTS]

        # Calculate ROSI for first control
        ale_after = ale * (1 - CONTROL_REDUCTIONS[0])
        rosi = calculate_rosi(ale, ale_after, adjusted_costs[0])
        Y.append(rosi)

    return np.array(Y)


def perform_sensitivity_analysis(num_samples: int = NUM_SAMPLES, output_file: str = "sensitivity_analysis.png"):
    """Perform Sobol sensitivity analysis on the model using the specified number of samples and save the results to a plot, if specified.

    Args:
        num_samples: Number of samples to generate for sensitivity analysis. Default is NUM_SAMPLES
        output_file: Output file to save the sensitivity analysis plot. Default is "sensitivity_analysis.png"

    Returns:
        dict: Sensitivity analysis results
    """
    problem = setup_sensitivity_problem(EF_variance=EF_RANGE, ARO_variance=ARO_RANGE, cost_variance=COST_ADJUSTMENT_RANGE)

    # Generate samples using sobol sampler
    param_values = sobol_sample.sample(problem, num_samples, calc_second_order=False, seed=RANDOM_SEED)

    # Run model evaluations
    Y = evaluate_model(param_values)

    # Create a copy of the problem dictionary to convert to numpy arrays
    problem_array = problem.copy()
    for key in problem_array.keys():
        if key != "num_vars":
            problem_array[key] = np.array(problem_array[key])

    # Calculate sensitivity indices using sobol analyzer
    Si = sobol_analyze.analyze(problem_array, Y, calc_second_order=False, seed=RANDOM_SEED)

    # Visualize results
    plt.figure(figsize=(10, 6))
    plt.bar(problem["names"], Si["S1"], yerr=Si["S1_conf"], label="First Order", color="blue")
    plt.bar(
        problem["names"], Si["ST"], yerr=Si["ST_conf"], alpha=0.5, label="Total Order", color="orange"
    )
    plt.xlabel("Parameters")
    plt.ylabel("Sensitivity Index")
    plt.legend()
    plt.title("Parameter Sensitivity Analysis")
    plt.tight_layout()

    # Save the plot if output file is provided
    if output_file:
        plt.savefig(output_file)

    return Si


# Set random seed for reproducibility
np.random.seed(RANDOM_SEED)

# Define one EF_i and one ARO_i per year:
params = {}
for i in range(NUM_YEARS):
    params[f"EF_{i+1}"] = EF_RANGE
for i in range(NUM_YEARS):
    params[f"ARO_{i+1}"] = ARO_RANGE

problem_ef_aro = setup_sensitivity_problem(**params)

# Now sobol_samples_ef_aro.shape will be (NUM_SAMPLES, 2*NUM_YEARS)
sobol_samples_ef_aro = sobol_sample.sample(
    problem_ef_aro, NUM_SAMPLES, calc_second_order=False, seed=RANDOM_SEED
)

# Slice the first NUM_YEARS columns for EF, and the next NUM_YEARS columns for ARO:
ef_samples = simulate_exposure_factor_sobol(
    sobol_samples_ef_aro[:, :NUM_YEARS], EF_RANGE, KURTOSIS
)
aro_samples = simulate_annual_rate_of_occurrence_sobol(
    sobol_samples_ef_aro[:, NUM_YEARS:], ARO_RANGE
)

# Calculate compounding costs for each control
control_cost_values = calculate_compounding_costs(
    CONTROL_COSTS, COST_ADJUSTMENT_RANGE, NUM_YEARS
)

# Determine permutations of control orderings (starting from 1 instead of 0)
all_permutations = list(permutations(range(1, NUM_YEARS + 1)))

# List to store total ROSI values for each permutation after calculating all samples
simulate_all_permutations = [
    calculate_statistics_for_permutation_per_year(  # You can use either calculate_statistics_for_permutation_aggregate or calculate_statistics_for_permutation_per_year
        control_cost_values, ef_samples, aro_samples, permutation
    )
    for permutation in all_permutations
]

# Sort the permutations by total ROSI descending
sorted_permutations = sorted(
    simulate_all_permutations, key=lambda x: x["total_rosi"], reverse=True
)
best_permutation = sorted_permutations[0]["permutation"]
best_rosi = sorted_permutations[0]["total_rosi"]

# Run sensitivity analysis
sensitivity_results = perform_sensitivity_analysis()

# Initialize a results dictionary
results = {
    "simulation_parameters": {
        "asset_value": ASSET_VALUE,
        "ef_range": EF_RANGE,
        "aro_range": ARO_RANGE,
        "control_reductions": CONTROL_REDUCTIONS,
        "control_costs": CONTROL_COSTS,
        "cost_adjustment_range": COST_ADJUSTMENT_RANGE,
        "num_samples": NUM_SAMPLES,
        "num_years": NUM_YEARS,
        "kurtosis": KURTOSIS,
    },
    "results": {
        "best_permutation": best_permutation,
        "best_rosi": best_rosi,
        "control_cost_values": control_cost_values,
    },
    "ranked_permutations": sorted_permutations,
    "sensitivity_results": sensitivity_results,
    # "simulation_data": {
    #     "ef_samples": ef_samples,
    #     "aro_samples": aro_samples,
    # },
}

serializable_results = convert_to_serializable(results)

with open("test.json", "w") as f:
    json.dump(serializable_results, f, indent=4)
