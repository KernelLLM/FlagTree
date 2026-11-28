@host
def _launch_axis0_owner_add(
    data: Tensor,
    indices: Tensor,
    updates: Tensor,
    output: Tensor,
    core_num: cb.Constexpr[int],
    axis_inner_elements: cb.Constexpr[int],
    index_axis_size: cb.Constexpr[int],
) -> None:
    """Launch the selected device implementation with its compile-time geometry."""
    _scatter_axis0_owner_add_simt[cb.dim3(core_num), cb.dim3(NATIVE_THREAD_COUNT)](
        data, indices, updates, output, axis_inner_elements, index_axis_size
    )
