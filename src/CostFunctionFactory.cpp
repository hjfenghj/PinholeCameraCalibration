#include "CostFunctionFactory.h"
#include "ceres/ceres.h"
#include "PinholeCamera.h"
#include <Eigen/Core>

class ReprojectionErrorAutoDiff
{
public:

    ReprojectionErrorAutoDiff(const Eigen::Vector3d& observed_P,
                       const Eigen::Vector2d& observed_p)
        : m_observed_P(observed_P), m_observed_p(observed_p) {}

    // variables: camera intrinsics and camera extrinsics
    template <typename T>
    bool operator()(const T* const params,
                    const T* const q,
                    const T* const t,
                    T* residuals) const
    {
        Eigen::Matrix<T, 3, 1> P = m_observed_P.cast<T>();
        Eigen::Matrix<T, 2, 1> predicted_p;

        Eigen::Matrix<T, 2, 1> e = Eigen::Matrix<T, 2, 1>::Zero();

        // TODO: homework2

        // 完成相机的投影过程，计算重投影误差

        ////////////////////////////////need to delete //////////////////////////////

        T P_w[3];
        P_w[0] = T(P(0));
        P_w[1] = T(P(1));
        P_w[2] = T(P(2));

        // Convert quaternion from Eigen convention (x, y, z, w)
        // to Ceres convention (w, x, y, z)
        T q_ceres[4] = {q[3], q[0], q[1], q[2]};

        T P_c[3];
        ceres::QuaternionRotatePoint(q_ceres, P_w, P_c);

        P_c[0] += t[0];
        P_c[1] += t[1];
        P_c[2] += t[2];

        // project 3D object point to the image plane
        T k1 = params[0];
        T k2 = params[1];
        T p1 = params[2];
        T p2 = params[3];
        T fx = params[4];
        T fy = params[5];
        T cx = params[6];
        T cy = params[7];

        // Transform to model plane
        T u = P_c[0] / P_c[2];
        T v = P_c[1] / P_c[2];

        T rho_sqr = u * u + v * v;
        T L = T(1.0) + k1 * rho_sqr + k2 * rho_sqr * rho_sqr;
        T du = T(2.0) * p1 * u * v + p2 * (rho_sqr + T(2.0) * u * u);
        T dv = p1 * (rho_sqr + T(2.0) * v * v) + T(2.0) * p2 * u * v;

        u = L * u + du;
        v = L * v + dv;
        predicted_p(0) = fx * u + cx;
        predicted_p(1) = fy * v + cy;

        e = predicted_p - m_observed_p.cast<T>();

        //////////////////////////////////need to delete//////////////////////////////////////////////

        residuals[0] = e(0);
        residuals[1] = e(1);

        return true;
    }

    // observed 3D point
    Eigen::Vector3d m_observed_P;

    // observed 2D point
    Eigen::Vector2d m_observed_p;
};


boost::shared_ptr<CostFunctionFactory> CostFunctionFactory::m_instance;

CostFunctionFactory::CostFunctionFactory()
{

}

boost::shared_ptr<CostFunctionFactory>
CostFunctionFactory::instance(void)
{
    if (m_instance.get() == 0)
    {
        m_instance.reset(new CostFunctionFactory);
    }

    return m_instance;
}

ceres::CostFunction*
CostFunctionFactory::generateCostFunction(const PinholeCameraConstPtr& camera,
        const Eigen::Vector3d& observed_P,
        const Eigen::Vector2d& observed_p) const
{

    std::vector<double> intrinsic_params;
    camera->writeParameters(intrinsic_params);
    ceres::CostFunction* costFunction = nullptr;
    //自动求导的优化函数
    costFunction = new ceres::AutoDiffCostFunction<ReprojectionErrorAutoDiff, 2, 8, 4, 3>(
                  new ReprojectionErrorAutoDiff(observed_P, observed_p));

    return costFunction;
}

