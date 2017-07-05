#include "kalman_filter.h"
#include "tools.h"
#include <stdlib.h>
#include <iostream>
#include <math.h>

using namespace std;

KalmanFilter::KalmanFilter() {
}

KalmanFilter::~KalmanFilter() {
}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
    MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  VectorXd z_pred = H_ * x_;
  VectorXd y = z - z_pred;
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd K = P_ * Ht * Si;

  //new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();

  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
   TODO:
   * update the state by using Extended Kalman Filter equations
   */

  float rho = sqrt(x_(0)*x_(0) + x_(1)*x_(1));
  float phi = atan2(x_(1), x_(0));
  float rho_dot;

  //handle super small rhos / special case
  if (fabs(rho) < 0.0001) {
    rho_dot = 0;
  } else {
    rho_dot = (x_(0)*x_(2) + x_(1)*x_(3))/rho;
  }
  VectorXd z_pred(3);
  z_pred << rho, phi, rho_dot;
  VectorXd y = z - z_pred;
  
  y[1] = atan2(sin(y[1]), cos(y[1])); //normalize the angle!!!

  // H must be Jacobian at this stage!!
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd K = P_ * Ht * Si;

  //new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}

VectorXd KalmanFilter::h(const VectorXd &x) {
  
  float px = x(0);
  float py = x(1);
  float vx = x(2);
  float vy = x(3);

  // define the h function 
  float rho = sqrt(px*px + py*py);
  float phi = atan(py/px);
  phi = atan2(sin(phi),cos(phi)); //normalize the angle!!!

  float rho_dot = (px*vx + py*vy) / rho;

  VectorXd hx = VectorXd(3);
  hx << rho, phi, rho_dot;

  return hx;
}