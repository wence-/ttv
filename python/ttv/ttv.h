#include <stdlib.h>
typedef double value_t;
void tensor_times_vector_double(
    size_t const q, size_t const p, value_t const *const a,
    size_t const *const na, size_t const *const wa,
    size_t const *const pia, value_t const *const b,
    size_t const *const nb, value_t *const c);
