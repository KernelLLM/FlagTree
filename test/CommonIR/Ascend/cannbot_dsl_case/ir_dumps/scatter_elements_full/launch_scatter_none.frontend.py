@host
def _launch_scatter_none(
    data: Tensor,
    indices: Tensor,
    updates: Tensor,
    output: Tensor,
    params: Tensor,
    core_num: cb.Constexpr[int],
) -> None:
    """Launch the selected device implementation with its compile-time geometry."""
    _scatter_none_simt[cb.dim3(core_num), cb.dim3(THREAD_COUNT)](
        data, indices, updates, output, params
    )
