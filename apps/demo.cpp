#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/viz/vizcore.hpp>
#include <io/capture.hpp>
#include <tfusion/topfu.hpp>
// #include <io/capture.hpp>

using namespace tfusion;

struct TopFuApp
{
    static void KeyboardCallback(const cv::viz::KeyboardEvent& event, void* pthis)
    {
        TopFuApp& topfu = *static_cast<TopFuApp*>(pthis);

        if(event.action != cv::viz::KeyboardEvent::KEY_DOWN)
            return;

        if(event.code == 't' || event.code == 'T')
            topfu.take_cloud(*topfu.topfu_);

        if(event.code == 'i' || event.code == 'I')
            topfu.iteractive_mode_ = !topfu.iteractive_mode_;
    }

    TopFuApp(OpenNISource& source) : exit_ (false),  iteractive_mode_(false), capture_ (source), pause_(false)
    {
        TopFuParams params = TopFuParams::default_params();
        topfu_ = TopFu::Ptr( new TopFu(params) );

        capture_.setRegistration(true);

        cv::viz::WCube cube(cv::Vec3d::all(0), cv::Vec3d(params.volume_size), true, cv::viz::Color::apricot());
        viz.showWidget("cube", cube, params.volume_pose);
        viz.showWidget("coor", cv::viz::WCoordinateSystem(0.1));
        viz.registerKeyboardCallback(KeyboardCallback, this);
    }

    void show_depth(const cv::Mat& depth)
    {
        cv::Mat display;
        //cv::normalize(depth, display, 0, 255, cv::NORM_MINMAX, CV_8U);
        depth.convertTo(display, CV_8U, 255.0/4000);
        cv::imshow("Depth", display);
    }

    void show_raycasted(TopFu& topfu)
    {
        const int mode = 3;
        // if (iteractive_mode_)
            topfu.renderImage(view_device_);//, viz.getViewerPose(), mode);
        // else
            // topfu.renderImage(view_device_, mode);

        view_host_.create(view_device_.rows(), view_device_.cols(), CV_8UC4);
        view_device_.download(view_host_.ptr<void>(), view_host_.step);
        // cv::Mat view(view_device_.rows(),view_device_.cols(),CV_8UC1);
        // for(int i=0;i<view_device_.rows();i++)
        // {
        //     for(int j=0;j<view_device_.cols();j++)
        //     {
        //         view.data[i * view_device_.cols() + j] = view_host_.data[i * view_device_.cols() * 4 + j * 4];
        //     }
        // }
        cv::imshow("Scene", view_host_);
        // cv::imshow("Scene",view);
    }

    void take_cloud(TopFu& topfu)
    {
        // cuda::DeviceArray<Point> cloud = topfu.tsdf().fetchCloud(cloud_buffer);
        // cv::Mat cloud_host(1, (int)cloud.size(), CV_32FC4);
        // cloud.download(cloud_host.ptr<Point>());
        // viz.showWidget("cloud", cv::viz::WCloud(cloud_host));
        //viz.showWidget("cloud", cv::viz::WPaintedCloud(cloud_host));
    }

    bool execute()
    {
        TopFu& topfu = *topfu_;
        cv::Mat depth, image;
        double time_ms = 0;
        bool has_image = false;

        for (int i = 0; !exit_ && !viz.wasStopped(); ++i)
        {
            bool has_frame = capture_.grab(depth, image);
            if (!has_frame)
                return std::cout << "Can't grab" << std::endl, false;

            depth_device_.upload(depth.data, depth.step, depth.rows, depth.cols);

            {
                SampledScopeTime fps(time_ms); (void)fps;
                has_image = topfu(depth_device_);
            }

            // std::cout<<"pose::"<<topfu<<std::endl;
            if (has_image)
                show_raycasted(topfu);

            show_depth(depth);
            //cv::imshow("Image", image);

            if (!iteractive_mode_)
                viz.setViewerPose(topfu.getCameraPose());

            int key = cv::waitKey(pause_ ? 0 : 3);

            switch(key)
            {
            case 't': case 'T' : take_cloud(topfu); break;
            case 'i': case 'I' : iteractive_mode_ = !iteractive_mode_; break;
            case 27: exit_ = true; break;
            case 32: pause_ = !pause_; break;
            }

            //exit_ = exit_ || i > 100;
            viz.spinOnce(3, true);
        }
        return true;
    }

    bool pause_ /*= false*/;
    bool exit_, iteractive_mode_;
    OpenNISource& capture_;
    TopFu::Ptr topfu_;
    cv::viz::Viz3d viz;

    cv::Mat view_host_;
    // cuda::Image view_device_;
    cuda::image4u view_device_;
    cuda::Depth depth_device_;
    cuda::DeviceArray<Point> cloud_buffer;
};


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main (int argc, char* argv[])
{
    int device = 0;
    cuda::setDevice (device);
    cuda::printShortCudaDeviceInfo (device);

    if(cuda::checkIfPreFermiGPU(device))
        return std::cout << std::endl << "Kinfu is not supported for pre-Fermi GPU architectures, and not built for them by default. Exiting..." << std::endl, 1;

    OpenNISource capture;
    capture.open (0);
    //capture.open("d:/onis/20111013-224932.oni");
    //capture.open("d:/onis/reg20111229-180846.oni");
    //capture.open("d:/onis/white1.oni");
    //capture.open("/media/Main/onis/20111013-224932.oni");
    //capture.open("20111013-225218.oni");
    //capture.open("d:/onis/20111013-224551.oni");
    //capture.open("d:/onis/20111013-224719.oni");

    TopFuApp app (capture);

    // executing
    try { app.execute (); }
    catch (const std::bad_alloc& /*e*/) { std::cout << "Bad alloc" << std::endl; }
    catch (const std::exception& /*e*/) { std::cout << "Exception" << std::endl; }

    return 0;
}
