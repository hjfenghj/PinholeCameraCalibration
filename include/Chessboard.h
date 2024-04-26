#pragma once

#include <boost/shared_ptr.hpp>
#include <opencv2/core/core.hpp>

class Chessboard
{
public:
    Chessboard(cv::Size boardSize, cv::Mat& image);

    // 调用findChessbosrdCorners函数,如果视野中棋盘格角点个数满足的话,就将角点可视化出来
    void findCorners();
    // 此方法的应用需要在上边的函数使用以后，在求出角点以后用域返回角点数组
    const std::vector<cv::Point2f>& getCorners(void) const;
    // 同样,在使用findcorners方法以后，用于返回是否由棋盘格并且成功预期的角点序列,2D坐标信息为相机视野中的像素坐标
    bool cornersFound(void) const;

    //通过使用Chessboard对象，调用方法findChessboardCorners方法,将图片对象传入该类，赋予mImage值
    const cv::Mat& getImage(void) const;
    const cv::Mat& getSketch(void) const;

private:
    // 检查视野中是否存在棋盘格，如果存在棋盘格且角点个数满足patternSize的设置，
    // 就将所有的角点储存在corners数组中，并返回true；反之返回false
    bool findChessboardCorners(const cv::Mat& image,
                               const cv::Size& patternSize,
                               std::vector<cv::Point2f>& corners,
                               int flags);

    cv::Mat mImage;
    cv::Mat mSketch;
    std::vector<cv::Point2f> mCorners;//2维角点集合数组
    cv::Size mBoardSize;  //角点个数形状 
    bool mCornersFound;   //表示是否在图片里找到棋盘格
};

