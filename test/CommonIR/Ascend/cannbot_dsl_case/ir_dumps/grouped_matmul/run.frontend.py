@host
def run(
    self,
    a_list: cannbotdsl.TensorList,
    b_list: cannbotdsl.TensorList,
    group_list_gm,
    c_list: cannbotdsl.TensorList,
    group_num,
):
    self.gmm_kernel[self.t.used_core_num](
        a_list, b_list, group_list_gm, c_list, group_num
    )
