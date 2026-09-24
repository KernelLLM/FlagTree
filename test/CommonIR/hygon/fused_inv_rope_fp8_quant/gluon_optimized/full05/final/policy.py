"""Layout specialization selected from the number of launched CTAs."""


def use_contig4(num_tokens, num_heads):
    aligned_tokens = (num_tokens + 3) // 4 * 4
    return aligned_tokens * num_heads >= 256


LAUNCH = {"num_warps": 1, "num_stages": 1, "waves_per_eu": 1}
