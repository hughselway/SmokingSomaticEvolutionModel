import math


def convert_from_varying_scale(value: float, varying_scale: str) -> float:
    if varying_scale == "log":
        return math.exp(value)
    if varying_scale == "logit":
        return 1 / (1 + math.exp(-value))
    if varying_scale == "linear":
        return value
    raise ValueError(f"Unknown varying scale: {varying_scale}")


def convert_to_varying_scale(value: float, varying_scale: str) -> float:
    if varying_scale == "log":
        return math.log(value)
    if varying_scale == "logit":
        return math.log(value / (1 - value))
    if varying_scale == "linear":
        return value
    raise ValueError(f"Unknown varying scale: {varying_scale}")
