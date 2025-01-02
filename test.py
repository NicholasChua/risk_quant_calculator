import numpy as np
from scipy.stats import beta, poisson, qmc
from itertools import permutations
import matplotlib.pyplot as plt
import json

from risk_simulator import (
    get_beta_parameters_for_kurtosis,
)

RANDOM_SEED = 42
ASSET_VALUE = 500000
EF_RANGE = (0.4, 0.6)  # Exposure factor range
ARO_RANGE = (1.5, 2.5)  # Annual rate of occurrence range
CONTROL_REDUCTIONS = [0.2, 0.35, 0.3, 0.45]  # Reduction percentages for controls
CONTROL_COSTS = [10000, 30000, 20000, 35000]  # Base annualized cost of controls
COST_ADJUSTMENT_RANGE = (0.0, 0.2)  # Cost adjustment per year range
NUM_SAMPLES = 16384  # Number of Sobol samples
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


def generate_sobol_samples(dimensions: int, samples: int) -> np.ndarray:
    """Generate Sobol samples using the Sobol sequence. The number of samples is rounded up to the nearest power of 2. The number of dimensions is the number of sequences to generate.

    Args:
        dimensions: Number of dimensions
        samples: Number of samples

    Returns:
        np.ndarray: Sobol samples
    """
    sampler = qmc.Sobol(d=dimensions, scramble=True, seed=RANDOM_SEED)
    # Generate Sobol samples rounded up to the nearest power of 2. Subtract 1 ensures m is the exponent for the largest power of 2 <= samples
    sobol_samples = sampler.random_base2(m=int(samples).bit_length() - 1)
    return sobol_samples


def simulate_exposure_factor_sobol(
    sobol_samples: np.ndarray, ef_range: tuple[float, float], kurtosis: int
) -> np.ndarray:
    """Simulate exposure factor using Sobol samples mapped to a Beta distribution with a specified kurtosis.

    Args:
        sobol_samples: Sobol samples
        ef_range: Exposure factor range
        kurtosis: Kurtosis of the distribution

    Returns:
        np.ndarray: Simulated exposure factors
    """
    # Calculate Beta parameters based on kurtosis
    alpha_param, beta_param = get_beta_parameters_for_kurtosis(kurtosis)
    # Map Sobol samples to Beta distribution using inverse CDF quantile function
    beta_samples = beta.ppf(sobol_samples, alpha_param, beta_param)

    # Scale Beta samples to the desired range
    exposure_factors = ef_range[0] + beta_samples * (ef_range[1] - ef_range[0])

    return exposure_factors


def simulate_annual_rate_of_occurrence_sobol(
    sobol_samples: np.ndarray, aro_range: tuple[float, float], decimal_places: int = 2
) -> np.ndarray:
    """Simulate annual rate of occurrence using Sobol samples mapped to a Poisson distribution with specified decimal precision. As Poisson distribution accepts only integer values, the Sobol samples are scaled up to the desired precision, mapped to the Poisson distribution, and then scaled back to the original scale, bypassing the decimal limitation.

    Args:
        sobol_samples: Sobol samples
        aro_range: Annual rate of occurrence range
        decimal_places: Number of decimal places for the generated values (default is 2)

    Returns:
        np.ndarray: Simulated annual rates of occurrence
    """
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


def calculate_compounding_costs(
    control_costs: list[float], cost_adjustment_range: tuple[float, float], years: int
) -> dict[str, dict[str, dict[str, np.float64]]]:
    """Given a list of control costs, a cost adjustment range, and a number of years, calculate the cost for each control for each year

    Args:
        control_costs: Base annualized cost of controls
        cost_adjustment_range: Cost adjustment per year range
        years: Number of years to simulate

    Returns:
        dict[str, dict[str, dict[str, np.float64]]]: Dictionary of costs and adjustments for each control, for each year
    """
    # Initialize costs dictionary
    costs = {}
    # Generate Sobol samples for cost adjustments
    sobol_samples = generate_sobol_samples(
        dimensions=len(control_costs) * years, samples=NUM_SAMPLES
    )

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


def calculate_statistics_for_permutation(costs: dict[str, dict[str, dict[str, np.float64]]], ef_samples: np.ndarray, aro_samples: np.ndarray, permutation: tuple[int], control_reductions: list = CONTROL_REDUCTIONS, num_of_simulations: int = NUM_SAMPLES) -> dict[str, float]:
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
                ale_before = calculate_ale(ASSET_VALUE, ef_samples[i][year], aro_samples[i][year])
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


# Convert NumPy arrays to lists for JSON serialization
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


# Set random seed for reproducibility
np.random.seed(RANDOM_SEED)

# Calculate the number of dimensions needed
num_dimensions = 2 * NUM_YEARS

# Generate Sobol samples for exposure factor and annual rate of occurrence
sobol_samples = generate_sobol_samples(dimensions=num_dimensions, samples=NUM_SAMPLES)

# Simulate EF and ARO using the appropriate slices of Sobol samples. All permutations use the same values for EF and ARO, thus all permutations are comparable
ef_samples = simulate_exposure_factor_sobol(
    sobol_samples[:, 0:NUM_YEARS], EF_RANGE, KURTOSIS
)
aro_samples = simulate_annual_rate_of_occurrence_sobol(
    sobol_samples[:, NUM_YEARS : 2 * NUM_YEARS], ARO_RANGE
)

# Calculate compounding costs for each control
control_cost_values = calculate_compounding_costs(
    CONTROL_COSTS, COST_ADJUSTMENT_RANGE, NUM_YEARS
)

# Determine permutations of control orderings (starting from 1 instead of 0)
all_permutations = list(permutations(range(1, NUM_YEARS + 1)))

# List to store total ROSI values for each permutation after calculating all samples
simulate_all_permutations = [
    calculate_statistics_for_permutation(
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
    "simulation_data": {
        "ef_samples": ef_samples,
        "aro_samples": aro_samples,
    },
}

serializable_results = convert_to_serializable(results)

with open("test.json", "w") as f:
    json.dump(serializable_results, f, indent=4)

# TODO: Add visualization of results
