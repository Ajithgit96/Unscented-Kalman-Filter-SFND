#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1.0;
  //std_a_ = 2.0;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.5;
  //std_yawdd_ = 1.0;
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  
  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */
  is_initialized_ = false;
  n_x_ = 5;
  n_aug_ = 7;
  //lambda_ = 3 - n_aug_;
  lambda_ = 3.0 - n_x_;
  n_z_ = 3;
  //time_us_ = 0;     // helpful to compute the time difference - can be taken from first measurement

  //setting values in weights vector
  weights_ = VectorXd(2 * n_aug_ + 1);
  weights_.fill(0.0);
  weights_(0) = lambda_ / (lambda_ + n_aug_);
  for(int i = 1; i < 2 * n_aug_ + 1; i++){
    weights_(i) = 0.5 / (lambda_ + n_aug_);
  }
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */

  if(is_initialized_ == false){
    
    // This is first measurement
    is_initialized_ = true;     //setting to true after first call of process measurement
    time_us_ = meas_package.timestamp_;
    
    // Determining the sensor type to intialize 'initial state estimate' to first measurement
    if(meas_package.sensor_type_ == MeasurementPackage::SensorType::LASER){
        // creating variables to improve readability
        double p_x = meas_package.raw_measurements_(0);
        double p_y = meas_package.raw_measurements_(1);

        x_ << p_x, p_y, 0, 0, 0;

        //initializing with identity matrix
        /* P_ << 1, 0, 0, 0, 0,
              0, 1, 0, 0, 0,
              0, 0, 1, 0, 0,
              0, 0, 0, 1, 0,
              0, 0, 0, 0, 1; */

        P_ << std_laspx_ * std_laspx_, 0, 0, 0, 0,
              0, std_laspy_ * std_laspy_, 0, 0, 0,
              0, 0, 1, 0, 0,
              0, 0, 0, 1, 0,
              0, 0, 0, 0, 1; 
    }

    if(meas_package.sensor_type_ == MeasurementPackage::SensorType::RADAR){
        // creating variables to improve readability
        double rho = meas_package.raw_measurements_(0);
        double phi = meas_package.raw_measurements_(1);
        double rho_dot = meas_package.raw_measurements_(2);

        double p_x = rho * cos(phi);
        double p_y = rho * sin(phi);
        double v = rho_dot;

        //x_ << p_x, p_y, v, 0, 0;
        x_ << p_x, p_y, v, rho, rho_dot;

        //initializing with identity matrix
        /* P_ << 1, 0, 0, 0, 0,
              0, 1, 0, 0, 0,
              0, 0, 1, 0, 0,
              0, 0, 0, 1, 0,
              0, 0, 0, 0, 1; */

        P_ << std_radr_ * std_radr_, 0, 0, 0, 0,
              0, std_radr_ * std_radr_, 0, 0, 0,
              0, 0, std_radrd_ * std_radrd_, 0, 0,
              0, 0, 0, std_radphi_ * std_radphi_, 0,
              0, 0, 0, 0, std_radphi_ * std_radphi_; 
    }
    return;
  }

  //computing the time difference between successive measurements
  //Necessary for prediction step
  
  //double delta_t = (meas_package.timestamp_ - time_us_);
  double delta_t = (meas_package.timestamp_ - time_us_) / 1000000.0;  //converting from microseconds into seconds
  time_us_ = meas_package.timestamp_;

  //previous (posterior) state estimate is available by now
  //Next, calling Prediction, Measurement Update functions accordingly
  Prediction(delta_t);
  
  if(meas_package.sensor_type_ == MeasurementPackage::SensorType::LASER){
    UpdateLidar(meas_package);
  } else if (meas_package.sensor_type_ == MeasurementPackage::SensorType::RADAR)
  {
    UpdateRadar(meas_package);
  }
  
}

void UKF::Prediction(double delta_t) {
  /**
   * TODO: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */
  
  //Augmentation of previous (posterior) state estimate
  VectorXd x_aug(n_aug_);
  x_aug.fill(0.0);
  x_aug.head(n_x_) = x_;

  MatrixXd P_aug(n_aug_, n_aug_);
  P_aug.fill(0.0);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(n_x_,n_x_) = std_a_ * std_a_;
  P_aug(n_x_+1,n_x_+1) = std_yawdd_ * std_yawdd_;

  //Computing the augmented sigma points
  MatrixXd Xsig_aug(n_aug_, 2 * n_aug_ + 1);
  MatrixXd A = P_aug.llt().matrixL();

  Xsig_aug.fill(0.0);
  Xsig_aug.col(0) = x_aug;
  for(int i = 0; i < A.cols(); i++){
    Xsig_aug.col(i + 1) = x_aug + sqrt(lambda_ + n_aug_) * A.col(i);
    Xsig_aug.col(i + n_aug_ + 1) = x_aug - sqrt(lambda_ + n_aug_) * A.col(i);
  }

  //Predicting sigma points
  //MatrixXd Xsig_pred_(n_x_, 2 * n_aug_ + 1);   //Xsig_pred_ must be a global variable.
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
  Xsig_pred_.fill(0.0);
  for(int i = 0; i < Xsig_aug.cols(); i++){
    double p_x = Xsig_aug(0,i);
    double p_y = Xsig_aug(1,i);
    double v = Xsig_aug(2,i);
    double yaw = Xsig_aug(3,i);
    double yawd = Xsig_aug(4,i);
    double v_a = Xsig_aug(5,i);
    double v_yawdd = Xsig_aug(6,i);

    VectorXd temp(n_x_);
    temp.fill(0.0);
    if(std::abs(yawd) < 1e-10){
        temp << v * cos(yaw) * delta_t,
                v * sin(yaw) * delta_t,
                0,
                0,
                0;
    } else {
        temp << (v / yawd) * (sin(yaw + yawd * delta_t) - sin(yaw)),
                (v / yawd) * (-cos(yaw + yawd * delta_t) + cos(yaw)),
                0,
                yawd * delta_t,
                0;
    }
      
    VectorXd noise(n_x_);
    noise.fill(0.0);
    noise << 0.5 * delta_t * delta_t * cos(yaw) * v_a,
             0.5 * delta_t * delta_t * sin(yaw) * v_a,
             delta_t * v_a,
             0.5 * delta_t * delta_t * v_yawdd,
             delta_t * v_yawdd;

    Xsig_pred_.col(i) = Xsig_aug.col(i).head(n_x_) + temp + noise;
  }

  //Computing mean and covariance of predicted sigma points
  //weights vector was already initialized
  //updating x_ and P_ variables.
  x_.fill(0.0);
  for(int i = 0; i < 2 * n_aug_ + 1; i++){
    x_ += weights_(i) * Xsig_pred_.col(i);
  }

  P_.fill(0.0);
  for(int i = 0; i < 2 * n_aug_ + 1; i++){
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    
    // angle normalization
    //while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    //while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    // debugging
    std::cout << "4" << std::endl;

    P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
    // debugging
    std::cout << "5" << std::endl;
  }

  //Predcition is complete now
  //Next step is measurement update based on sensor measurement model
  
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */

  //LiDAR measurement model is linear and so we can use regular Kalman filter equations
  //Projecting predicted measurement into LiDAR measurement space
  //Measurement prediction
  MatrixXd H(2,n_x_);
  H << 1, 0, 0, 0, 0,
       0, 1, 0, 0, 0;

  MatrixXd R(2,2);    //Lidar measurement noise covariance matrix
  R.fill(0.0);
  R << std_laspx_ * std_laspx_, 0,
       0, std_laspy_ * std_laspy_;

  VectorXd z_pred = H * x_;
  MatrixXd S = H * P_ * H.transpose() + R;

  //computing the difference between measurement prediction and current measurement
  VectorXd y = meas_package.raw_measurements_ - z_pred;

  //Kalman gain
  MatrixXd K = P_ * H.transpose() * S.inverse();

  //Update equations
  x_ = x_ + K * y;
  MatrixXd I = MatrixXd::Identity(n_x_, n_x_);
  P_ = (I - K * H) * P_;

  // NIS for LiDAR
  NIS_lidar_ = y.transpose() * S.inverse() * y;
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */

  //Radar measurmeent model is non linear
  //Making use of predicted sigma points for measurement prediction
  //Measurement prediction
  MatrixXd Zsig(n_z_, 2 * n_aug_ + 1);
  VectorXd z_pred(n_z_);
  MatrixXd S(n_z_, n_z_);

  // transform sigma points into measurement space
  Zsig.fill(0.0);
  for(int i = 0; i < Xsig_pred_.cols(); i++){
      double p_x = Xsig_pred_.col(i)(0);
      double p_y = Xsig_pred_.col(i)(1);
      double v = Xsig_pred_.col(i)(2);
      double yaw = Xsig_pred_.col(i)(3);
      
      //Zsig.col(i)(0) = sqrt(p_x * p_x + p_y * p_y);
      //Zsig.col(i)(1) = atan2(p_y, p_x);
      //Zsig.col(i)(2) = (p_x * cos(yaw) * v + p_y * sin(yaw) * v) / sqrt(p_x * p_x + p_y * p_y);
      Zsig(0,i) = sqrt(p_x * p_x + p_y * p_y);
      Zsig(1,i) = atan2(p_y, p_x);
      Zsig(2,i) = (p_x * cos(yaw) * v + p_y * sin(yaw) * v) / sqrt(p_x * p_x + p_y * p_y);
  }
  
  // calculate mean predicted measurement
  z_pred.fill(0.0);
  for(int i = 0; i < Zsig.cols(); i++){
      z_pred += weights_(i) * Zsig.col(i);
  }
  // calculate innovation covariance matrix S
  S.fill(0.0);
  for(int i = 0; i < Zsig.cols(); i++){
      VectorXd z_diff = Zsig.col(i) - z_pred;
      
      //normalizing the yaw angle
      //while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
      //while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;
      
      S += weights_(i) * z_diff * z_diff.transpose();
  }
  
  MatrixXd meas_cov(n_z_, n_z_);
  meas_cov.fill(0.0);
  meas_cov  <<  std_radr_ * std_radr_, 0, 0,
                0, std_radphi_ * std_radphi_, 0,
                0, 0, std_radrd_ * std_radrd_;
                
  S += meas_cov;

  //Update equations
  //cross correlation matrix
  MatrixXd Tc(n_x_, n_z_);
  Tc.fill(0.0);
  for(int i = 0; i < Xsig_pred_.cols(); i++){
      VectorXd x_diff = Xsig_pred_.col(i) - x_;
      VectorXd z_diff = Zsig.col(i) - z_pred;
      
      // angle normalization
      //while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
      //while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;
      
      //while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
      //while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;
      
      Tc += weights_(i) * x_diff * z_diff.transpose();
  }
  // Kalman gain K;
  MatrixXd K = Tc * S.inverse();
  // updating state mean and covariance matrix
  x_ = x_ + K * (meas_package.raw_measurements_ - z_pred);
  P_ = P_ - K * S * K.transpose();

  // NIS for RADAR
  NIS_radar_ = (meas_package.raw_measurements_ - z_pred).transpose() * S.inverse() * (meas_package.raw_measurements_ - z_pred);
}