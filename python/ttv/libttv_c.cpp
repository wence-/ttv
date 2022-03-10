#include <tlib/detail/layout.h>
#include <tlib/detail/shape.h>
#include <tlib/detail/strides.h>
#include <tlib/ttv.h>
#include <iostream>
typedef double value_t;
extern "C" {
void tensor_times_vector_double(
    std::size_t const q, std::size_t const p, value_t const *const a,
    std::size_t const *const na, std::size_t const *const wa,
    std::size_t const *const pia, value_t const *const b,
    std::size_t const *const nb, value_t *const c) {
  auto nc = std::vector<std::size_t>(p - 1);
  tlib::detail::compute_output_shape(na, na + p, nc.begin(), q);
  auto pic = std::vector<std::size_t>(p - 1);
  tlib::detail::compute_output_layout(pia, pia + p, pic.begin(), q);
  auto wc = tlib::detail::generate_strides(nc, pic);

  tlib::tensor_times_vector<value_t, std::size_t>(
      tlib::execution::seq, tlib::slicing::small, tlib::loop_fusion::none, q, p,
      a, na, wa, pia, b, nb, c, nc.data(), wc.data(), pic.data());
}
}
