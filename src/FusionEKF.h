#ifndef FusionEKF_H_
#define FusionEKF_H_

#include "measurement_package.h"
#include <vector>
#include <string>
#include <fstream>
#include "kalman_filter.h"
#include "tools.h"

class FusionEKF {
public:
  /**
  * Constructor.
  */
  FusionEKF();

  /**
  * Destructor.
  */
  virtual ~FusionEKF();

  /**
  * Run the whole flow of the Kalman Filter from here.
  */
  void ProcessMeasurement(const MeasurementPackage &measurement_pack);

  /**
  * Kalman Filter update and prediction math lives in here.
  */
  KalmanFilter ekf_;

private:
  // check whether the tracking toolbox was initialized or not (first measurement)
  bool is_initialized_;

  // previous timestamp
  long previous_timestamp_;

  ///* state vector
  VectorXd x_;

  ///* process covariance matrix
  MatrixXd Q_;

  ///* state transition matrix
  MatrixXd F_;

  ///* state covariance matrix
  MatrixXd P_;

  // tool object used to compute Jacobian and RMSE
  Tools tools;
  MatrixXd R_laser_;
  MatrixXd R_radar_;
  MatrixXd H_laser_;
  MatrixXd Hj_;


};

#endif /* FusionEKF_H_ */