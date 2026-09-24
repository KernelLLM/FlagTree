"""Launch policy paired with the optimized single-wave layout."""


def select_grid(total_items):
    if total_items <= 516:
        return total_items
    if total_items <= 2193:
        return (3 * total_items + 3) // 4
    if total_items <= 8256:
        return (7 * total_items + 15) // 16
    return max(1, (total_items + 1) // 2 - 6144)


LAUNCH = {"num_warps": 1, "num_stages": 4, "waves_per_eu": 1}
