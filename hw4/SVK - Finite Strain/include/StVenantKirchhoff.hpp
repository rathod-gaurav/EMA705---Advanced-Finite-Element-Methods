#pragma once //include this only once during compilation

#include "MaterialModel.hpp"

template<unsigned int Nsd, unsigned int Nne>
class StVenantKirchhoff : public MaterialModel<Nsd,Nne>{ //derived from MaterialModel class
    public:
        using MatrixNsd = Eigen::Matrix<double, Nsd, Nsd>;
        StVenantKirchhoff(double lambda, double mu); //default constructor - takes Lame' parameters

        void compute(
            const MatrixNsd& F, //deformation gradient
            MatrixNsd& S, //second Piola-Kirchhoff stress tensor
            MatrixNsd& P, //first Piola-Kirchhoff stress tensor
            Eigen::MatrixXd& C_mat //material tangent stiffness matrix (4th order elasticity tensor in Voigt notation)
        ) const override; //override the pure virtual function from MaterialModel class

    private:
        double lambda_; //Lame' parameter lambda
        double mu_; //Lame' parameter mu
};

#include "StVenantKirchhoff.tpp"