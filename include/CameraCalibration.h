#pragma once

#include <opencv2/core/core.hpp>
#include <Eigen/Core>
#include "PinholeCamera.h"

class CameraCalibration
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    CameraCalibration();

    //构造函数
    CameraCalibration(const cv::Size& imageSize,
                      const cv::Size& boardSize,
                      float squareSize);


    void clear(void);
    //把所有帧中对应的棋盘格角点坐标在标定板坐标系中的3D位置坐标记录在m_scenePoints;
    void addChessboardData(const std::vector<cv::Point2f>& corners);
    //把所有帧中对应的apritag标定板角点坐标在标定板坐标系中的3D位置坐标记录在m_scenePoints;
    void addApriltagDate(const std::vector<cv::Point2f>& corners,
                        const std::vector<cv::Point3f>& tagpoints);

    bool calibrate(void);

    //所有图片中国,满足标定标准的样例个数
    int sampleCount(void) const;
    std::vector<std::vector<cv::Point2f> >& imagePoints(void);
    const std::vector<std::vector<cv::Point2f> >& imagePoints(void) const;
    std::vector<std::vector<cv::Point3f> >& scenePoints(void);
    const std::vector<std::vector<cv::Point3f> >& scenePoints(void) const;
    PinholeCameraPtr& camera(void);

    Eigen::Matrix2d& measurementCovariance(void);
    const Eigen::Matrix2d& measurementCovariance(void) const;

    cv::Mat& cameraPoses(void);
    const cv::Mat& cameraPoses(void) const;

    void drawResults(std::vector<cv::Mat>& images) const;

    void setVerbose(bool verbose);

private:
    bool calibrateHelper(PinholeCameraPtr& camera,
                         std::vector<cv::Mat>& rvecs, std::vector<cv::Mat>& tvecs) const;

    void optimize(PinholeCameraPtr& camera,
                  std::vector<cv::Mat>& rvecs, std::vector<cv::Mat>& tvecs) const;

    cv::Size m_boardSize;     //表示棋盘格的行和列数
    float m_squareSize;       //表示棋盘格中每个格子的大小

    PinholeCameraPtr m_camera;//
    cv::Mat m_cameraPoses;

    std::vector<std::vector<cv::Point2f> > m_imagePoints;  //image图片2D角点信息(具体代表四个角点还是所有角点不清楚)
    std::vector<std::vector<cv::Point3f> > m_scenePoints;  //每一帧数据对一个棋盘格角点信息

    Eigen::Matrix2d m_measurementCovariance;//重投影误差的协方差矩阵

    bool m_verbose;
};
