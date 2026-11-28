@host
def run(
    self,
    copy_data: Tensor,
    copy_output: Tensor,
    indices: Tensor,
    updates: Tensor,
    output: Tensor,
) -> None:
    """Launch the selected device implementation with its compile-time geometry."""
    self.operator[self.core_num](copy_data, copy_output, indices, updates, output)
