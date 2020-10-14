#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_io.h>
#include <iostream>

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cmath>

using namespace cv;
using namespace dlib;
using namespace std;

const char *fmodel_name = "bestmodel.pt";
const char *fshapepredictor_name = "shape_predictor_68_face_landmarks.dat";
//---------------------------------------------------------------------------------------
struct Net : torch::nn::Module
{
    Net()
    {
        // Initialize CNN
        conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 10, 5)));
        conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(10, 20, 5)));
        conv2_drop = register_module("conv2_drop", torch::nn::Dropout2d());
        fc1 = register_module("fc1", torch::nn::Linear(3080, 100));
        fc2 = register_module("fc2", torch::nn::Linear(100, 13));
    }

    torch::Tensor forward(torch::Tensor x)
    {
        x = torch::relu(torch::max_pool2d(conv1->forward(x), 2));
        x = torch::relu(torch::max_pool2d(conv2_drop->forward(conv2->forward(x)), 2));
        x = x.view({-1, 3080});
        x = torch::relu(fc1->forward(x));
        x = torch::dropout(x, 0.5, is_training());
        x = fc2->forward(x);
        x = torch::log_softmax(x, 1);

        return x;
    }

    torch::nn::Conv2d conv1{nullptr};
    torch::nn::Conv2d conv2{nullptr};
    torch::nn::Dropout2d conv2_drop{nullptr};
    torch::nn::Linear fc1{nullptr};
    torch::nn::Linear fc2{nullptr};
};
//---------------------------------------------------------------------------------------
torch::Tensor matToTensor(cv::Mat const &src)
{
    torch::Tensor out;
    cv::Mat img_float;
    src.convertTo(img_float, CV_32F);

    out = torch::empty({1, 1, 40, 100}, torch::kFloat);
    memcpy(out.data_ptr(), img_float.data, out.numel() * sizeof(float));

    return out;
}
//---------------------------------------------------------------------------------------
bool is_file_exist(const char *fileName)
{
    std::ifstream infile(fileName);
    return infile.good();
}
//---------------------------------------------------------------------------------------
int main(int argc, char **argv)
{
    if (argc == 1)
    {
        std::cerr << "Usage: " << argv[0] << " [options]\n"
                  << "Options:\n"
                  << "\t-w <numdevice>\t\tfrom webcamera, where <numdevice> is number of device\n"
                  << "\t-v <filename>\t\tfrom videofile, where <filename> is file name of videofile\n"
                  << "\t-i <filename>\t\tfrom image file, where <filename> is file name of image"
                  << std::endl;
        return 1;
    }

    if (argc > 3)
    {
        std::cerr << "incorrect number of options" << std::endl;
        return 1;
    }

    if (!is_file_exist(fmodel_name))
    {
        std::cerr << "Can't find file with best model: " << fmodel_name << std::endl;
        return 1;
    }

    if (!is_file_exist(fshapepredictor_name))
    {
        std::cerr << "Can't find file for shape predictor: " << fshapepredictor_name << std::endl;
        return 1;
    }

    std::string p1 = argv[1];
    std::string p2 = argv[2];
    int inumdevice = 0;
    std::string sfilename_input = "";
    int imode = 0; //mode of work  0 - web, 1 - videofile, 2 - imagefile

    if (p1 == "-w")
    {
        inumdevice = std::stoi(p2);
        imode = 0;
    }
    else if (p1 == "-v")
    {
        sfilename_input = p2;
        imode = 1;
    }
    else if (p1 == "-i")
    {
        sfilename_input = p2;
        imode = 2;
    }
    else
    {
        std::cerr << "unknown option: " << p1.c_str() << std::endl;
        return 1;
    }

    cv::VideoCapture cap;

    bool bres = false;
    if (imode == 0)
        bres = cap.open(inumdevice);
    else
    {
        if (!is_file_exist(sfilename_input.c_str()))
        {
            std::cerr << "Can't find file: " << sfilename_input << std::endl;
            return 1;
        }
        bres = cap.open(sfilename_input);
    }

    if (!bres)
    {
        std::cerr << "Error opening webcamera, video stream or imagefile" << endl;
        return -1;
    }
   
    // Load face detection and pose estimation models.
    frontal_face_detector detector = get_frontal_face_detector();
    shape_predictor pose_model;
    deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;

    auto network = std::make_shared<Net>();
    torch::load(network, fmodel_name);
    network->to(torch::kCPU);
    network->eval();

    uint64_t iframecount = 0;
    while (true)
    {
        Mat srcmat;
        cap >> srcmat;

        // If the frame is empty, break immediately
        if (srcmat.empty())
            break;

        cv_image<bgr_pixel> cimg(srcmat);

        // Detect faces
        std::vector<dlib::rectangle> faces = detector(cimg);
        // Find the pose of each face.
        for (unsigned long i = 0; i < faces.size(); ++i)
        {
            full_object_detection fobj = pose_model(cimg, faces[i]);
            if (fobj.num_parts() == 68)
            {
                Mat greyMat, croppedImage;
                int ix = fobj.part(42).x();
                int iy = fobj.part(43).y() - 5;
                int iwidth = fobj.part(45).x() - fobj.part(42).x();
                int iheight = fobj.part(47).y() - fobj.part(43).y() + 10;

                if ((ix < 0) || ((ix + iwidth) > srcmat.size().width) || (iy < 0) || ((iy + iheight) > srcmat.size().height))
                    continue;

                cv::Rect myROI(ix, iy, iwidth, iheight);

                cvtColor(srcmat, greyMat, CV_BGR2GRAY);
                croppedImage = greyMat(myROI);
                resize(croppedImage, croppedImage, cv::Size(100, 40));

                torch::Tensor out = matToTensor(croppedImage);
                out = out.div_(255);                
                auto output = network->forward(out);
                int16_t ipred = output.argmax(1).item<int16_t>();

                if (imode == 1) //videofile
                {
                    if (ipred == 0)
                        std::cout << "#" << iframecount << "\tLOOKING AT CAM" << std::endl;
                    else
                        std::cout << "#" << iframecount << "\tNOT LOOKING AT CAM" << std::endl;
                }
                else
                    std::cout << ipred << std::endl;
            }           
        }

        if (imode == 2)
            break;
        iframecount++;
    }

    cap.release();    

    return 0;
}
