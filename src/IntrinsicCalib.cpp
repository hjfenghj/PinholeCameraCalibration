#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "Chessboard.h"
#include "CameraCalibration.h"

using namespace std;

int main(int argc, char** argv)
{
    cv::Size boardSize;
    float squareSize;    //棋盘格每个格子的大小
    std::string inputDir;
    std::string fileExtension;
    std::string cost_function_type;
    bool viewResults;
    bool verbose;

    float apriltag_size = 0.088;
    float apriltag_interval = 0.3;

    //========= Handling Program options =========
    boost::program_options::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("width,w", boost::program_options::value<int>(&boardSize.width)->default_value(7), "Number of inner corners on the chessboard pattern in x direction")
        ("height,h", boost::program_options::value<int>(&boardSize.height)->default_value(5), "Number of inner corners on the chessboard pattern in y direction")
        ("size,s", boost::program_options::value<float>(&squareSize)->default_value(48.f), "Size of one square in mm")
        ("input,i", boost::program_options::value<std::string>(&inputDir)->default_value("calibrationdata"), "Input directory containing chessboard images")
        ("cost,c", boost::program_options::value<std::string>(&cost_function_type)->default_value("auto"), "cost function type of camera, auto & manual")
        ("extension,e", boost::program_options::value<std::string>(&fileExtension)->default_value(".jpg"), "File extension of images")
        ("view-results", boost::program_options::bool_switch(&viewResults)->default_value(true), "View results")
        ("verbose,v", boost::program_options::bool_switch(&verbose)->default_value(true), "Verbose output");
    //w,h,s表示棋盘格格子的行列以及棋盘格的大小
    //i(input)表示输入文件所在的文件夹
    //c(cost)表示损失函数的类型
    //e(extension)输入文件的拓展名

    //verbose表示在代码运行的时候是否输出运行的详情
    //view_results表示标定结束以后是否显示

    //apriltag-size表示一种标定板的某个尺寸
    //apriltag_interval表示一种间隔
    //a(apriltag)表示是否使用apriltag标定板


    boost::program_options::positional_options_description pdesc;
    pdesc.add("input", 1);

    boost::program_options::variables_map vm;
    //store ()->将解析出的选项存储至variables_map中
    boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(desc).positional(pdesc).run(), vm);
    //更新decs对象的值
    boost::program_options::notify(vm);

    if (vm.count("help"))
    {
        std::cout << desc << std::endl;
        return 1;
    }

    if (!boost::filesystem::exists(inputDir) && !boost::filesystem::is_directory(inputDir))
    {
        std::cerr << "# ERROR: Cannot find input directory " << inputDir << "." << std::endl;
        return 1;
    }

    // look for images in input directory
    std::vector<std::string> imageFilenames; //图片文件的路径 
    boost::filesystem::directory_iterator itr;
    // directory_iterator(p)就是迭代器的起点,无参数的directory_iterator()就是迭代器的终点。
    for (boost::filesystem::directory_iterator itr(inputDir); itr != boost::filesystem::directory_iterator(); ++itr)
    {
        if (!boost::filesystem::is_regular_file(itr->status()))//status返回路径名对应的状态
        {
            continue;
        }

        std::string filename = itr->path().filename().string();
        // check if file extension matches
        if (filename.compare(filename.length() - fileExtension.length(), fileExtension.length(), fileExtension) != 0)
        {
            continue;
        }
        imageFilenames.push_back(itr->path().string());
    }

    if (imageFilenames.empty())
    {
        std::cerr << "# ERROR: No chessboard images found." << std::endl;
        return 1;
    }

    if (verbose)//verbose表示什么状态,输出详情
    {
        std::cerr << "# INFO: # images: " << imageFilenames.size() << std::endl;//表示有多少图片
    }

    std::sort(imageFilenames.begin(), imageFilenames.end());

    cv::Mat image = cv::imread(imageFilenames.front(), -1);
    const cv::Size frameSize = image.size();

    //在完成CameraCalibration类的初始化，调用了构造函数以后
    //类中的成员变量PinholeCameraPtr类型变量m_camera就完成了出初始化和赋值操作
    CameraCalibration calibration(frameSize, boardSize, squareSize);  //boardSize表示棋盘格子的数量,
                                                                      //frameSize表示每一帧图片的大小
                                                                      //squareSize每个格子的大小
    calibration.setVerbose(verbose);

    std::vector<bool> chessboardFound(imageFilenames.size(), false);  //所有的帧中,标定板是否出现

    // use chessboard target 
    for (size_t i = 0; i < imageFilenames.size(); ++i)
    {
        image = cv::imread(imageFilenames.at(i), -1);

        Chessboard chessboard(boardSize, image);//boardSize表示棋盘格子的行列形状
                                                //boardsize.width,boardsize.length

        chessboard.findCorners();
        if (chessboard.cornersFound())
        {
            if (verbose)
            {
                std::cerr << "# INFO: Detected chessboard in image " << i + 1 << ", " << imageFilenames.at(i) << std::endl;
            }
            //把棋盘格角点信息的2D坐标和3D坐标储存在CameraCalibreation类的成员变量
            //m_imagePoints和m_scenePoints中,2D坐标信息为相机视野中角点的像素坐标
            calibration.addChessboardData(chessboard.getCorners());//参数是角点的数组

            cv::Mat sketch;
            chessboard.getSketch().copyTo(sketch);

            cv::imshow("Image", sketch);
            cv::waitKey(50);
        }
        else if (verbose)
        {
            std::cerr << "# INFO: Did not detect chessboard in image " << i + 1 << std::endl;
        }
        chessboardFound.at(i) = chessboard.cornersFound();
    }
    cv::destroyWindow("Image");

    if (calibration.sampleCount() < 1)//sampleCount表示视野中棋盘格样例个数
    {
        std::cerr << "# ERROR: Insufficient number of detected chessboards." << std::endl;
        return 1;
    }

    if (verbose)
    {
        std::cerr << "# INFO: Calibrating..." << std::endl;
    }
    //进入计算内参的环节
    calibration.calibrate();

    if (viewResults)
    {
        std::vector<cv::Mat> cbImages;
        std::vector<std::string> cbImageFilenames;

        for (size_t i = 0; i < imageFilenames.size(); ++i)
        {
            if (!chessboardFound.at(i))
            {
                continue;
            }
            cbImages.push_back(cv::imread(imageFilenames.at(i), -1));
            cbImageFilenames.push_back(imageFilenames.at(i));
        }

        // visualize observed and reprojected points
        calibration.drawResults(cbImages);

        for (size_t i = 0; i < cbImages.size(); ++i)
        {
            cv::imshow("Image", cbImages.at(i));
            cv::waitKey(0);
        }
    }

    return 0;
}
