// RUN: triton-opt %s -split-input-file -verify-diagnostics

// make_tensor_view must produce a base (encoding-free) view.
func.func @make_tensor_view_encoded(%p: !tv.ptr<f32>, %n: index) {
  // expected-error @+1 {{result must be a base view (no encoding)}}
  %v = tv.make_tensor_view %p, sizes = [%n], strides = [%n]
       : !tv.ptr<f32> -> !tv.tensor_view<?xf32, strides=[1], #tv.partition_view<tile = [128], dim_map = [0], padding = zero>>
  return
}

// -----

// view_load result shape must equal the tile shape.
func.func @view_load_bad_shape(%p: !tv.ptr<f32>, %n: index, %i: index) {
  %v = tv.make_tensor_view %p, sizes = [%n], strides = [%n]
       : !tv.ptr<f32> -> !tv.tensor_view<?xf32, strides=[1]>
  %pv = tv.make_partition_view %v
        : !tv.tensor_view<?xf32, strides=[1]>
       -> !tv.tensor_view<?xf32, strides=[1], #tv.partition_view<tile = [128], dim_map = [0], padding = zero>>
  // expected-error @+1 {{result shape must equal the view tile shape}}
  %t = tv.view_load %pv[%i]
       : !tv.tensor_view<?xf32, strides=[1], #tv.partition_view<tile = [128], dim_map = [0], padding = zero>>
      -> tensor<64xf32>
  return
}

// -----

// tensor_view strides rank must match shape rank.
func.func @tensor_view_bad_strides(%p: !tv.ptr<f32>, %n: index) {
  // expected-error @+1 {{expected strides rank (2) to match shape rank (1)}}
  %v = tv.make_tensor_view %p, sizes = [%n], strides = [%n]
       : !tv.ptr<f32> -> !tv.tensor_view<?xf32, strides=[1,1]>
  return
}
