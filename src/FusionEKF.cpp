#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>
#include <math.h>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // measurement covariance matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);

  //laser measurement matrix
  H_laser_ = MatrixXd(2, 4);
  H_laser_ << 1, 0, 0, 0,
          0, 1, 0, 0;

  // radar Hj matrix
  Hj_ = MatrixXd(3, 4);

  Q_ = MatrixXd(4, 4);

  //state covariance matrix P
  P_ = MatrixXd(4, 4);
}

/**
 * Destructor.
 */
FusionEKF::~FusionEKF() {
}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    /**
     * Initialize the state ekf_.x_ with the first measurement.
     * Create the covariance matrix.
     * Remember: you'll need to convert radar from polar to cartesian coordinates.
     */
    // first measurement
    cout << "EKF initialise: " << endl;

    //create a 4D state vector, we don't know yet the values of the x state
    x_ = VectorXd(4);

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      /**
       Convert radar from polar to cartesian coordinates and initialize state.
       */
      float rho = measurement_pack.raw_measurements_[0]; // Range - radial distance from origin
      float phi = measurement_pack.raw_measurements_[1]; // Bearing - angle between rho and x
      float rho_dot = measurement_pack.raw_measurements_[2]; // Radial Velocity - change of p (range rate)
      float x = rho * cos(phi);
      float y = rho * sin(phi);
      float vx = rho_dot * cos(phi);
      float vy = rho_dot * sin(phi);

      x_ << x, y, vx , vy;
    } else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      /**
       Initialize state.
       */
      //set the state with the initial location and zero velocity
      x_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0, 0;
    }

    // initialise measurement covariances
//    R_radar_ << 0.01, 0, 0,
//          0,0.01, 0,
//          0, 0, 0.01;
    R_radar_ << 0.09, 0, 0,
        0, 0.0009, 0,
        0, 0, 0.09;

    R_laser_ << 0.0225, 0,
        0, 0.0225;

    // intial state and covariance matrix
    ekf_.x_ = x_;

    P_ << 1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1;
    ekf_.P_ = P_;

    cout << "init x_ = " << ekf_.x_ << endl;
    cout << "init P_ = " << ekf_.P_ << endl;

    previous_timestamp_ = measurement_pack.timestamp_;

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  /**
   * Update the state transition matrix F according to the new elapsed time.
   - Time is measured in seconds.
   * Update the process noise covariance matrix.
   */
  //compute the time elapsed between the current and previous measurements
  float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0; //dt - expressed in seconds
  previous_timestamp_ = measurement_pack.timestamp_;

  //the initial transition matrix F_
  F_ = MatrixXd(4, 4);
  F_ << 1, 0, dt, 0,
      0, 1, 0, dt,
      0, 0, 1, 0,
      0, 0, 0, 1;

  ekf_.F_ = F_;

  //set the acceleration noise components
  double noise_ax = 9.0;
  double noise_ay = 9.0;

  // Update the process noise covariance matrix.
  double dt_2 = dt * dt;
  double dt_3 = dt_2 * dt;
  double dt_4 = dt_3 * dt;
  double c1 = dt_4 / 4;
  double c2 = dt_3 / 2;
  Q_ = MatrixXd(4, 4);
  Q_ << c1 * noise_ax, 0, c2 * noise_ax, 0,
      0, c1 * noise_ay, 0, c2 * noise_ay,
      c2 * noise_ax, 0, dt_2 * noise_ax, 0,
      0, c2 * noise_ay, 0, dt_2 * noise_ay;

  ekf_.Q_ = Q_;
  ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  /**
   * Use the sensor type to perform the update step.
   * Update the state and covariance matrices.
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    VectorXd z = VectorXd(3);
    z << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1],
        measurement_pack.raw_measurements_[2];


    Hj_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.Init(ekf_.x_, ekf_.P_, F_, Hj_, R_radar_, Q_);
    cout << "Hj_ " << Hj_ << endl;

    // if we have an initialised Jacobian update
    if (!Hj_.isZero(0)){
      ekf_.UpdateEKF(z);
    }
  } else {
    // Laser updates
    VectorXd z = VectorXd(2);
    z << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1];
    ekf_.Init(ekf_.x_, ekf_.P_, F_, H_laser_, R_laser_, Q_);
    ekf_.Update(z);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}