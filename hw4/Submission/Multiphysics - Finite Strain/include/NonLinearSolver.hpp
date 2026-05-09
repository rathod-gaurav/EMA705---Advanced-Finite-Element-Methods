#pragma once //include this only once during compilation

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include "Assembler.hpp"
#include "BoundaryConditions.hpp"
#include "BoundaryConditionsScalar.hpp"

template <unsigned int Nsd, unsigned int Nne, unsigned int BfOrder>
class NonlinearSolver{
    public:
        NonlinearSolver(double tol, unsigned int maxIncr, unsigned int maxIter, unsigned int maxTimeSteps);

        void solve(
            Eigen::VectorXd& u, //displacement vector, modified in place
            Eigen::VectorXd& C, //chemical concentration vector, modified in place
            Eigen::VectorXd& T, //temperature vector, modified in place
            double dt, //time step size for the diffusion equations
            const Assembler<Nsd, Nne,BfOrder>& assembler, //provides Kglobal, Rglobal
            const BoundaryConditions<Nsd,Nne>& bcs, //provides dirischlet indexes and values for displacement
            const BoundaryConditionsScalar<Nsd,Nne>& bcs_C, //provides dirischlet indexes and values for chemical concentration
            const BoundaryConditionsScalar<Nsd,Nne>& bcs_T, //provides dirischlet indexes and values for temperature
            std::function<void(unsigned int, unsigned int, double)> iterCallback = nullptr //optional callback function for iteration progress (increment, iteration, residual norm)
        );

    private:
        double tol_; //tolerance for convergence
        unsigned int maxIncr_; //maximum number of increments (timesteps)
        unsigned int maxIter_; //maximum number of iterations per increment
        unsigned int maxTimeSteps_; //maximum number of time steps for the diffusion equations
};

#include "NonLinearSolver.tpp" //include the implementation of the NonlinearSolver class
