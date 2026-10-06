import math

ACTIONS = {
    "KEEP",
    "LOCAL",
    "PREDICT",
    "DIVERSIFY"
}

def equal_budget(action, region_size, budget):

    if action not in ACTIONS:
        raise ValueError(
            f"Unknown action: {action}"
        )

    if region_size <= 0:
        return 0

    if action == "KEEP":
        return 0

    return int(budget)

def proportional_budget(action, region_size, eta_local, eta_predict, eta_diversify):
    
    if action not in ACTIONS:
        raise ValueError(f"Invalid action '{action}'. Must be one of {ACTIONS}.")

    if region_size <= 0:
        raise ValueError("Region size must be positive.")

    if action == "KEEP":
        return 0  # KEEP does not generate offspring
    
    if action == "LOCAL":
        eta = eta_local

    elif action == "PREDICT":
        eta = eta_predict

    elif action == "DIVERSIFY":
        eta = eta_diversify

    return math.ceil(eta * region_size)