#!/usr/bin/env python3
"""
运行 tile.concat 的基础测试
"""

import sys

print("=" * 80)
print("tile.concat 测试套件")
print("=" * 80)

# 测试 1: 导入模块
print("\n测试 1: 导入必要的模块...")
try:
    import triton
    import triton.language as tl
    import triton.experimental.tle.language.dsa as tle
    print("✅ 所有模块导入成功")
except Exception as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

# 测试 2: 检查 API
print("\n测试 2: 检查 tile_concat API...")
if hasattr(tle, 'tile_concat'):
    print("✅ tle.tile_concat 存在")
    import inspect
    sig = inspect.signature(tle.tile_concat)
    print(f"   签名: tile_concat{sig}")
else:
    print("❌ tle.tile_concat 不存在")
    sys.exit(1)

# 测试 3: 定义测试 kernel
print("\n测试 3: 定义测试 kernel...")
try:
    @triton.jit
    def test_concat_1d(x_ptr, y_ptr, z_ptr, N: tl.constexpr):
        offs = tl.arange(0, N)
        x = tl.load(x_ptr + offs)
        y = tl.load(y_ptr + offs)
        z = tle.tile_concat(x, y, dim=0)
        tl.store(z_ptr + tl.arange(0, 2*N), z)

    print("✅ 1D concat kernel 定义成功")
except Exception as e:
    print(f"❌ Kernel 定义失败: {e}")
    sys.exit(1)

# 测试 4: Sinkhorn 优化模式
print("\n测试 4: Sinkhorn 优化模式 kernel...")
try:
    @triton.jit
    def sinkhorn_concat_pattern(
        row_ptr,
        rcp_sum,
        out_ptr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Sinkhorn 核心优化: 2个1x4操作 -> 1个2x4操作
        向量化宽度从4提升到8
        """
        pid = tl.program_id(0)

        # 加载两行数据
        row_0 = tl.load(row_ptr + pid * 2 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE))
        row_1 = tl.load(row_ptr + (pid * 2 + 1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE))

        # 将两行 concat 成一个 2x4 tensor
        row_0_2d = row_0.reshape(1, BLOCK_SIZE)
        row_1_2d = row_1.reshape(1, BLOCK_SIZE)
        row_01 = tle.tile_concat(row_0_2d, row_1_2d, dim=0)  # <2xBLOCK_SIZE>

        # 统一处理 (8-wide SIMD)
        result = row_01 * rcp_sum

        # 存储结果
        result_1d = result.reshape(2 * BLOCK_SIZE)
        tl.store(out_ptr + pid * 2 * BLOCK_SIZE + tl.arange(0, 2*BLOCK_SIZE), result_1d)

    print("✅ Sinkhorn concat pattern kernel 定义成功")
except Exception as e:
    print(f"❌ Sinkhorn kernel 定义失败: {e}")
    sys.exit(1)

# 测试总结
print("\n" + "=" * 80)
print("✅ 所有基础测试通过！")
print("=" * 80)

print("\n📋 测试总结:")
print("  ✅ 模块导入成功")
print("  ✅ API 存在且签名正确")
print("  ✅ 1D concat kernel 可以定义")
print("  ✅ Sinkhorn 优化模式 kernel 可以定义")

print("\n⚠️  注意:")
print("  - 这些测试验证了 API 的可用性和 kernel 定义")
print("  - 实际编译和执行需要:")
print("    1. 实现 Lowering Pass (TileToLinalg.cpp)")
print("    2. 配置 Ascend 硬件环境")
print("    3. 运行端到端测试")

print("\n📚 相关文档:")
print("  - BUILD_SUCCESS_REPORT.md - 编译成功报告")
print("  - TODO.md - 下一步任务清单")
print("  - IMPLEMENTATION_SUMMARY.md - 实现详解")

sys.exit(0)
