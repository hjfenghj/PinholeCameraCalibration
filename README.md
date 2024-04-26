# PinholeCameraCalibration
Pinhole Camera Internal Reference Calibration Toolbox
这是一个按照张正友标定法写的一个内参标定工具

0. 依赖安装
- eigen库安装
- opencv安装
- ceres非线性优化库安装

在Eigen(3.3.7),cmake(3.16.3),opencv(4.2.0),ceres(1.14.0)已经做了测试，正常运行出结果

1. 编译
cd PinholeCameraCalibration
cmake .
make
编译成功会生成一个可执行文件camera_calib

2. 运行
./camera_calib -i "数据路径(就是拍摄的棋盘格图片)"


- 参数介绍
工程主函数在文件IntrinsicCalib.cpp中，命令行参数介绍
w,h,s表示棋盘格格子的行列以及棋盘格的大小
i(input)表示输入文件所在的文件夹
c(cost)表示损失函数的类型
e(extension)输入文件的拓展名
verbose表示在代码运行的时候是否输出运行的详情
view_results表示标定结束以后是否显示

