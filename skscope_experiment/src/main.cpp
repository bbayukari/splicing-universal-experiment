#include <random>
#include <iostream>

#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>

#include <Eigen/Eigen>

using Eigen::MatrixXd;
using Eigen::MatrixXi;
using Eigen::VectorXd;
using Eigen::VectorXi;
using Eigen::ArrayXd;
using Eigen::Matrix;
namespace py = pybind11;



//**********************************************************************************
// Ising model data structure and generator
//**********************************************************************************
Eigen::MatrixXd comp_conf(int num_conf, int p){
  Eigen::MatrixXi conf = Eigen::MatrixXi::Zero(num_conf, p);
  Eigen::VectorXi num = Eigen::VectorXi::LinSpaced(num_conf, 0, num_conf - 1);

  for (int i = p - 1; i >= 0; i--){
    conf.col(i) = num - num / 2 * 2;
    num /= 2;
  }
  conf = conf.array() * 2 - 1;
  return conf.cast<double>();
}

// n is the number of samples
Eigen::MatrixXd sample_by_conf(int sample_size, Eigen::MatrixXd theta, int seed) {
  int p = theta.rows();
  int num_conf = pow(2, p);
  
  Eigen::MatrixXd table = comp_conf(num_conf, p);
  Eigen::VectorXd weight(num_conf);
  
  Eigen::VectorXd vec_diag = theta.diagonal();
  Eigen::MatrixXd theta_diag = vec_diag.asDiagonal();
  Eigen::MatrixXd theta_off = theta - theta_diag;
  
  for (int num = 0; num < num_conf; num++) {
    Eigen::VectorXd conf = table.row(num);
    weight(num) = 0.5 * (double) (conf.transpose() * theta_off * conf);
  }
  weight = weight.array().exp();
  
  std::vector<double> w;
  w.resize(weight.size());
  Eigen::VectorXd::Map(w.data(), weight.size()) = weight;
  
  // int sd = (((long long int)time(0)) * 2718) % 314159265;
  // Rcout << "Seed: "<< sd << endl;
  // std::default_random_engine generator(seed);  // implementation-defined
  // std::default_random_engine generator(1);

  std::mt19937_64 generator;                      // 64-bit Mersenne Twister by Matsumoto and Nishimura, 2000
  generator.seed(seed);
  std::discrete_distribution<int> distribution(std::begin(w), std::end(w));
  
  Eigen::VectorXd freq = Eigen::VectorXd::Zero(num_conf);
  
  for (int i = 0; i < sample_size; i++) {
    freq(distribution(generator))++;
  }
  
  Eigen::MatrixXd data(num_conf, p + 1);
  data.col(0) = freq;
  data.rightCols(p) = table;
  return data;
}




//**********************************************************************************
// pybind11 module
//**********************************************************************************
PYBIND11_MODULE(_skscope_experiment, m) {
    m.def("ising_generator", &sample_by_conf);
}
