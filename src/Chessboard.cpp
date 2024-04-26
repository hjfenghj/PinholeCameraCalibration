#include "Chessboard.h"

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <opencv2/imgproc/types_c.h>
#include <opencv2/calib3d/calib3d_c.h>


Chessboard::Chessboard(cv::Size boardSize, cv::Mat& image)
 : mBoardSize(boardSize)
 , mCornersFound(false)
{
    if (image.channels() == 1)
    {
        cv::cvtColor(image, mSketch, CV_GRAY2BGR);
        image.copyTo(mImage);
    }
    else
    {
        image.copyTo(mSketch);
        cv::cvtColor(image, mImage, CV_BGR2GRAY);
    }
}

void
Chessboard::findCorners()
{
    mCornersFound = findChessboardCorners(mImage, mBoardSize, mCorners,
                                          CV_CALIB_CB_ADAPTIVE_THRESH +
                                          CV_CALIB_CB_NORMALIZE_IMAGE +
                                          CV_CALIB_CB_FILTER_QUADS +
                                          CV_CALIB_CB_FAST_CHECK);

    if (mCornersFound)
    {
        // draw chessboard corners
        cv::drawChessboardCorners(mSketch, mBoardSize, mCorners, mCornersFound);
        //msketch表示图片矩阵对象，,mBoardSize表示角点个数形状,mCorners表示,mCornersFound是否存在棋盘格
    }
}

const std::vector<cv::Point2f>&
Chessboard::getCorners(void) const
{
    return mCorners;
}

bool
Chessboard::cornersFound(void) const
{
    return mCornersFound;
}

const cv::Mat&
Chessboard::getImage(void) const
{
    return mImage;
}

const cv::Mat&
Chessboard::getSketch(void) const
{
    return mSketch;
}

bool
Chessboard::findChessboardCorners(const cv::Mat& image,
                                  const cv::Size& patternSize,
                                  std::vector<cv::Point2f>& corners,
                                  int flags)
{
    //确定输入图片是否有棋盘图案，并定位棋盘板上的内角点。如果所有的角点被找到且以一定的顺序排列
    //（一行接一行，从一行的左边到右边），该函数会返回一个非零值。
    //另外，如果该函数没有找到所有的角点或者重新排列他们，则返回0。
    //第三个参数表示找到的角点的输出储存数组
    if(cv::findChessboardCorners(image, patternSize, corners, flags)){
        //亚像素精度的优化,对上边找到的角点corners中的所有角点进行精细化的调整,
        //最后还是储存在corners中，即是输入也是输出
        cv::cornerSubPix(image, corners, cv::Size(11, 11), cv::Size(-1,-1),
                         cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
        return true;
    }

    return false;
}
