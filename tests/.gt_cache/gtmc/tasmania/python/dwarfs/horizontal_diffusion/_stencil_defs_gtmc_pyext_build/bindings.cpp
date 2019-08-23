

#include "computation.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>
#include <chrono>

namespace py = ::pybind11;


BufferInfo make_buffer_info(py::object& b) {
    auto buffer_info = static_cast<py::buffer&>(b).request();

    py_size_t ndim = static_cast<py_size_t>(buffer_info.ndim);
    std::vector<py_size_t>& shape = buffer_info.shape;
    std::vector<py_size_t>& strides = buffer_info.strides;
    void* ptr = static_cast<void*>(buffer_info.ptr);

    if (ndim != 3) {
        throw std::runtime_error("Wrong number of dimensions [" +
                                 std::to_string(ndim) +
                                 " != " + std::to_string(3) + "]");
    }

    return BufferInfo{ndim, shape, strides, ptr};
}

void run_computation(_stencil_defs_gtmc::GTComputation& computation,
             py::object in_phi, const std::array<gt::uint_t, 3>& in_phi_origin, 
             py::object in_gamma, const std::array<gt::uint_t, 3>& in_gamma_origin, 
             py::object out_phi, const std::array<gt::uint_t, 3>& out_phi_origin, 
             float64_t dx, 
             float64_t dy, py::object& exec_info)
{
    if (!exec_info.is(py::none()))
    {
        auto exec_info_dict = exec_info.cast<py::dict>();
        exec_info_dict["start_run_cpp_time"] = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count())/1e9;
    }
    auto bi_in_phi = make_buffer_info(in_phi);
    auto bi_in_gamma = make_buffer_info(in_gamma);
    auto bi_out_phi = make_buffer_info(out_phi);

    computation.run(
             bi_in_phi, in_phi_origin, 
             bi_in_gamma, in_gamma_origin, 
             bi_out_phi, out_phi_origin, 
             dx, 
             dy);
    if (!exec_info.is(py::none()))
    {
        auto exec_info_dict = exec_info.cast<py::dict>();
        exec_info_dict["end_run_cpp_time"] = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count()/1e9);
    }
}



PYBIND11_MODULE(_stencil_defs_gtmc_pyext, m) {
    m.def("run_computation", &run_computation, "Runs the given computation", py::arg("computation"),
          py::arg("in_phi"),  py::arg("in_phi_origin"), 
          py::arg("in_gamma"),  py::arg("in_gamma_origin"), 
          py::arg("out_phi"),  py::arg("out_phi_origin"), 
          py::arg("dx"), 
          py::arg("dy"), py::arg("exec_info"));

    py::class_<_stencil_defs_gtmc::GTComputation>(m, "GTComputation", py::module_local())
        .def(py::init<std::array<gt::uint_t, 3>>(),   
             py::arg("shape"))  
        .def("run", &_stencil_defs_gtmc::GTComputation::run,
             py::arg("in_phi"),  py::arg("in_phi_origin"), 
             py::arg("in_gamma"),  py::arg("in_gamma_origin"), 
             py::arg("out_phi"),  py::arg("out_phi_origin"), 
             py::arg("dx"), 
             py::arg("dy"));

}