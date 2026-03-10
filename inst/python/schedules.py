import math


def _clamp_t(t: float):
    return min(1.0, max(0.0, float(t)))


def coslog_k(t: float, k: int = 4):
    # t in [0,1]
    t = _clamp_t(t)
    return 0.5 * (
        1.0 - math.cos(2.0 * math.pi * math.log2(1.0 + (2**k - 1) * t))
    )


def flat_cos(t: float):
    # t in [0,1]
    t = _clamp_t(t)
    u = max(1.0, 2.0 * t) - 1.0
    return 0.5 * (1.0 + math.cos(math.pi * u))


def get_schedule(name):
    if name is None:
        return lambda t: 1.0
    if name == "coslog4":
        return lambda t: coslog_k(t, k=4)
    if name == "flat_cos":
        return flat_cos
    raise ValueError(f"Unknown schedule: {name}")
