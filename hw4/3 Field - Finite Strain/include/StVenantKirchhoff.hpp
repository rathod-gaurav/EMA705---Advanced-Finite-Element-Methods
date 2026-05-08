#pragma once //include this only once during compilation

#include "MaterialModel.hpp"

template<unsigned int Nsd, unsigned int Nne>
class StVenantKirchhoff : public MaterialModel<Nsd,Nne>{ //derived from MaterialModel class
    public:
        using MatrixNsd = Eigen::Matrix<double, Nsd, Nsd>;
        using VectorNsd = Eigen::Vector<double, Nsd>;
        StVenantKirchhoff(double lambda, double mu, const MatrixNsd& alpha_C, const MatrixNsd& alpha_T); //default constructor - takes Lame' parameters

        void compute(
            const MatrixNsd& F, //deformation gradient
            const double& C_val, //chemical concentration
            const double& T_val, //temperature
            MatrixNsd& S, //second Piola-Kirchhoff stress tensor
            MatrixNsd& P, //first Piola-Kirchhoff stress tensor
            Eigen::MatrixXd& C_mat //material tangent stiffness matrix (4th order elasticity tensor in Voigt notation)
        ) const override; //override the pure virtual function from MaterialModel class

    private:
        double lambda_; //Lame' parameter lambda
        double mu_; //Lame' parameter mu
        const MatrixNsd& alpha_C_; //coefficient for chemical expansion
        const MatrixNsd& alpha_T_; //coefficient for thermal expansion
};

#include "StVenantKirchhoff.tpp"