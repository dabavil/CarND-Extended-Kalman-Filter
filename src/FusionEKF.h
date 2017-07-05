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

  ///* state, noisce, state transition, currrent covariance vector
  VectorXd x_;
  MatrixXd Q_;
  MatrixXd F_;
  MatrixXd P_;

  // For Jacobian mtx
  Tools tools;
  MatrixXd R_laser_;
  MatrixXd R_radar_;
  MatrixXd H_laser_;
  MatrixXd Hj_;


};

#endif /* FusionEKF_H_ */