// REQUIRES: flagtree-common-ir
// RUN: triton-opt %s --convert-common-ir-to-ttgir | FileCheck %s

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module {
  tt.func public @gpu_tileir_to_ttgir(%src: tensor<16x!tt.ptr<f32>>) {
    %buf = tile.alloc {layout = 0 : i64, space = 7 : i64, tle.gpu_layout = #shared}
      : <[2, 16], f32, shared>
    %desc = tile.get_memdesc %buf
      : !tile.buf<[2, 16], f32, shared> -> !ttg.memdesc<2x16xf32, #shared, #smem, mutable>
    tle.pipe.create %desc {capacity = 2 : i32, field_names = ["payload"], pipe_name = "p", scope = "cta"}
      : !ttg.memdesc<2x16xf32, #shared, #smem, mutable>
    %c0 = arith.constant 0 : index
    %slot = tile.subview %buf[%c0, %c0] [[16]] [[1]] {tle.gpu_layout = #shared1}
      : <[2, 16], f32, shared> -> <[16], f32, shared>
    tile.copy %src -> %slot {src_layout = 0 : i64}
      : tensor<16x!tt.ptr<f32>>, !tile.buf<[16], f32, shared>
    %ptr = "tile.local_ptr"(%slot)
      : (!tile.buf<[16], f32, shared>) -> tensor<16x!tt.ptr<f32, 3>>
    tt.return
  }
}

// CHECK: %[[BUF:.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x16xf32, #shared, #smem, mutable>
// CHECK: tle.pipe.create %[[BUF]]
// CHECK: %[[SLOT:.*]] = ttg.memdesc_index %[[BUF]][%{{.*}}]
// CHECK: tt.load %{{.*}} : tensor<16x!tt.ptr<f32>>
// CHECK: "tle.local_pointers"(%[[SLOT]],
// CHECK: "tle.local_pointers"(%[[SLOT]])
// CHECK-NOT: tile.
