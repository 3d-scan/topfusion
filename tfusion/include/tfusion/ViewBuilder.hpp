#pragma once

namespace tfusion
{
	class ViewBuilder
	{
	public:
		void ConvertDisparityToDepth(FloatImage *depth_out, const ShortImage *depth_in, const Intrinsics *depthIntrinsics, 
			Vector2f disparityCalibParams);
		void ConvertDepthAffineToFloat(FloatImage *depth_out, const ShortImage *depth_in, Vector2f depthCalibParams);

		void DepthFiltering(FloatImage *image_out, const FloatImage *image_in);
		void ComputeNormalAndWeights(Float4Image *normal_out, FloatImage *sigmaZ_out, const FloatImage *depth_in, Vector4f intrinsic);

		void UpdateView(View **view, UChar4Image *rgbImage, ShortImage *rawDepthImage, bool useBilateralFilter, bool modelSensorNoise = false, bool storePreviousImage = true);
		void UpdateView(View **view, UChar4Image *rgbImage, ShortImage *depthImage, bool useBilateralFilter, IMUMeasurement *imuMeasurement, bool modelSensorNoise = false, bool storePreviousImage = true);

		ViewBuilder(const RGBDCalib& calib_)
		:calib(calib_)
		{
			this->ShortImage = NULL;
			this->floatImage = NULL;
		}
		~ViewBuilder()
		{
			if(this->shortImage != NULL) delete this->shortImage;
			if(this->floatImage != NULL) delete this->floatImage;
		}
	};
}