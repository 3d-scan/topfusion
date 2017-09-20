#pragma once

#include <tfusion/types.hpp>
#include <tfusion/topfu.hpp>
// #include <tfusion/cuda/tsdf_volume.hpp>
#include <tfusion/cuda/imgproc.hpp>
#include <tfusion/cuda/projective_icp.hpp>
//#include <opencv2/calib3d/calib3d.hpp>
#include "internal.hpp"
#include <iostream>
#include "vector_functions.h"

namespace tfusion
{
    template<typename D, typename S>
    inline D device_cast(const S& source)
    {
        return *reinterpret_cast<const D*>(source.val);
    }

    template<>
    inline device::Aff3f device_cast<device::Aff3f, Affine3f>(const Affine3f& source)
    {
        device::Aff3f aff;
        Mat3f R = source.rotation();
        Vec3f t = source.translation();
        aff.R = device_cast<device::Mat3f>(R);
        aff.t = device_cast<device::Vec3f>(t);
        return aff;
    }
}
