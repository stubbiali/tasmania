

#include <gridtools/common/defs.hpp>
#include <gridtools/stencil_composition/stencil_composition.hpp>

#include <boost/cstdfloat.hpp> 

#include <array>
#include <cstdint>
#include <vector>


using boost::float32_t; 
using boost::float64_t; 

using py_size_t = std::intptr_t;

struct BufferInfo {
    py_size_t ndim;
    std::vector<py_size_t> shape;
    std::vector<py_size_t> strides;
    void* ptr;
};


namespace gt = ::gridtools;

namespace _stencil_defs_gtmc {

// Backend
using backend_t = gt::backend::mc;

// These halo sizes are used to determine the sizes of the temporaries
static constexpr gt::uint_t halo_size_i = 0;
static constexpr gt::uint_t halo_size_j = 2;
static constexpr gt::uint_t halo_size_k = 0;

// Storage definitions
template <int Id>
using storage_info_t =
    gt::storage_traits<backend_t>::storage_info_t<
        Id, 3,
        gt::halo<halo_size_i, halo_size_j, 0 /* not used */>>;

template<typename T, int Id>
using data_store_t =
    gt::storage_traits<backend_t>::data_store_t<T, storage_info_t<Id>>;

template <typename T>
using global_parameter_t =
    decltype(gt::make_global_parameter<backend_t>(std::declval<T>()));

// Placeholder definitions
using p_in_phi = gt::arg<0, data_store_t<float64_t, 0>>;
using p_in_gamma = gt::arg<1, data_store_t<float64_t, 0>>;
using p_out_phi = gt::arg<2, data_store_t<float64_t, 0>>;

using p_dx = gt::arg<0, global_parameter_t<float64_t>>;
using p_dy = gt::arg<1, global_parameter_t<float64_t>>;

// Note: Creating the GTComputation has some cost (allocating temporaries) and
// should not be recreated over and over again. We should cache the most recent
// GTComputation objects on python side.
class GTComputation {
   public:
    GTComputation(std::array<gt::uint_t, 3> size);

    void run(
             const BufferInfo& b_in_phi, const std::array<gt::uint_t, 3>& in_phi_origin, 
             const BufferInfo& b_in_gamma, const std::array<gt::uint_t, 3>& in_gamma_origin, 
             const BufferInfo& b_out_phi, const std::array<gt::uint_t, 3>& out_phi_origin, 
             float64_t dx, 
             float64_t dy
            );

   private:
    const std::array<gt::uint_t, 3> size_;
    const global_parameter_t<float64_t> dx_param_;
    const global_parameter_t<float64_t> dy_param_;
    gt::computation<p_in_phi, p_in_gamma, p_out_phi, p_dx, p_dy> computation_;
};

}  // namespace _stencil_defs_gtmc