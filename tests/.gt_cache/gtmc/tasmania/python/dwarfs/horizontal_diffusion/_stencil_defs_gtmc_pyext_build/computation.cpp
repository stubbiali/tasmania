

#include "computation.hpp"
#include <gridtools/stencil_composition/stencil_composition.hpp>

#include <array>
#include <cassert>
#include <stdexcept>

namespace _stencil_defs_gtmc {

namespace {

// Axis
static constexpr gt::uint_t level_offset_limit = 2;

using axis_t =
    gridtools::axis<1, /* NIntervals */
                    gt::axis_config::offset_limit<level_offset_limit>>;

// Constants


// Functors

struct stage__6_func {
    using in_gamma = gt::in_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;
    using in_phi = gt::in_accessor<1, gt::extent<0, 0, -2, 2, 0, 0>>;
    using dy = gt::global_accessor<2>;
    using out_phi = gt::inout_accessor<3, gt::extent<0, 0, 0, 0, 0, 0>>;

    using param_list = gt::make_param_list<in_gamma, in_phi, dy, out_phi >;

    
    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval)
    {
        eval(out_phi()) = (eval(in_gamma()) * (((((-eval(in_phi(0, -2, 0))) + (float64_t{16.0} * eval(in_phi(0, -1, 0)))) - (float64_t{30.0} * eval(in_phi()))) + (float64_t{16.0} * eval(in_phi(0, 1, 0)))) - eval(in_phi(0, 2, 0)))) / ((float64_t{12.0} * eval(dy())) * eval(dy()));
    }
};


// Grids and halos
gt::halo_descriptor make_halo_descriptor(gt::uint_t compute_domain_shape) {
    return {0, 0, 0, compute_domain_shape - 1,
            compute_domain_shape};
}

auto make_grid(const std::array<gt::uint_t, 3>& compute_domain_shape)
{
    return gt::make_grid(make_halo_descriptor(compute_domain_shape[0]),
                         make_halo_descriptor(compute_domain_shape[1]),
                         axis_t(compute_domain_shape[2]));
}


// Placeholder definitions


template <typename Grid>
auto make_computation_helper(const Grid& grid) {
    return gt::make_computation<backend_t>(
        grid,
        gt::make_multistage(gt::execute::parallel(),
            gt::make_stage<stage__6_func>(
                p_in_gamma(), p_in_phi(), p_dy(), p_out_phi()
            )
        )
    );
}

template<typename T, int Id>
data_store_t<T, Id> make_data_store(const BufferInfo& bi,
                                const std::array<gt::uint_t, 3>& compute_domain_shape,
                                const std::array<gt::uint_t, 3>& origin) {
    if (bi.ndim != 3) {
        throw std::runtime_error("Wrong number of dimensions [" +
                                 std::to_string(bi.ndim) +
                                 " != " + std::to_string(3) + "]");
    }

    for (int i = 0; i < 3; ++i) {
        if (2*origin[i] + compute_domain_shape[i] > bi.shape[i])
            throw std::runtime_error(
                "Given shape and origin exceed buffer dimension");
    }

    // ptr, dims and strides are "outer domain" (i.e., compute domain + halo
    // region). The halo region is only defined through `make_grid` (and
    // currently, in the storage info)
    gt::array<gt::uint_t, 3> dims{};
    gt::array<gt::uint_t, 3> strides{};
    T* ptr = static_cast<T*>(bi.ptr);
    for (int i = 0; i < 3; ++i) {
        strides[i] = bi.strides[i] / sizeof(T);
        ptr += strides[i] * origin[i];
        dims[i] = compute_domain_shape[i]+2*origin[i];
    }
    return data_store_t<T, Id>{storage_info_t<Id>{dims, strides}, ptr,
                           gt::ownership::external_cpu};
}

}  // namespace

GTComputation::GTComputation(std::array<gt::uint_t, 3> size)  
    : size_(size),
    dx_param_(gt::make_global_parameter<backend_t>(float64_t{})),
    dy_param_(gt::make_global_parameter<backend_t>(float64_t{})),
    computation_(make_computation_helper(make_grid(size)))
{
}

void GTComputation::run(
                        const BufferInfo& bi_in_phi, const std::array<gt::uint_t, 3>& in_phi_origin, 
                        const BufferInfo& bi_in_gamma, const std::array<gt::uint_t, 3>& in_gamma_origin, 
                        const BufferInfo& bi_out_phi, const std::array<gt::uint_t, 3>& out_phi_origin, 
                        float64_t dx, 
                        float64_t dy
    )
{
    // Initialize data stores from input buffers
    auto ds_in_phi = make_data_store<float64_t, 0>(bi_in_phi, size_, in_phi_origin);
    auto ds_in_gamma = make_data_store<float64_t, 0>(bi_in_gamma, size_, in_gamma_origin);
    auto ds_out_phi = make_data_store<float64_t, 0>(bi_out_phi, size_, out_phi_origin);

    // Update global parameters
    gt::update_global_parameter(dx_param_, dx);
    gt::update_global_parameter(dy_param_, dy);

    // Run computation and wait for the synchronization of the output stores
    computation_.run(p_in_phi() = ds_in_phi, p_in_gamma() = ds_in_gamma, p_out_phi() = ds_out_phi, p_dx() = dx_param_, p_dy() = dy_param_);
        // computation_.sync_bound_data_stores();
}

}  // namespace _stencil_defs_gtmc