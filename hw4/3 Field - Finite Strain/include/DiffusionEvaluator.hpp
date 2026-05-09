#pragma once

#include <Eigen/Dense>
#include "Mesh.hpp"
#include "ShapeFunction.hpp"
#include "Quadrature.hpp"
#include "MaterialModel.hpp"

template <unsigned int Nsd, unsigned int Nne, unsigned int BfOrder>
class DiffusionEvaluator{
    public:
        using MatrixNsd = Eigen::Matrix<double, Nsd, Nsd>;
        using VectorNsd = Eigen::Vector<double, Nsd>;
        DiffusionEvaluator( //default constructor
            const Mesh<Nsd,Nne>& mesh,
            const QuadratureRule<Nsd,Nne>& quadRule,
            double DC,
            double DT,
            const MatrixNsd& alpha_C,
            const MatrixNsd& alpha_T,
            double lambda,
            double mu
        );

        void computeDiffusionMatrices(
            unsigned int e, //element index

            Eigen::MatrixXd& MCClocal, //element diffusion stiffness matrix for chemical concentration (Nne x Nne matrix)
            Eigen::MatrixXd& KCClocal,
            Eigen::MatrixXd& KCTlocal,

            Eigen::MatrixXd& MTTlocal, //element diffusion stiffness matrix for temperature (Nne x Nne matrix)
            Eigen::MatrixXd& KTTlocal,
            Eigen::MatrixXd& KTClocal,
            
            Eigen::MatrixXd& KuClocal, //terms from mechanics weak coupling (Nne*Nsd x Nne  matrix)
            Eigen::MatrixXd& KuTlocal //(Nne*Nsd x Nne  matrix)
        ) const;
    
    private:        
        MatrixNsd computeJacobian(unsigned int e, const VectorNsd& xi_vec) const; //function to compute the Jacobian matrix for the element at given quadrature point (xi1, xi2, xi3)

        const Mesh<Nsd,Nne>& mesh_; //reference to the mesh object
        const QuadratureRule<Nsd,Nne>& quadRule_; //reference to the quadrature rule object

        double DC_; //diffusion coefficient for chemical concentration
        double DT_; //diffusion coefficient for temperature
        MatrixNsd alpha_C_; //coefficient for chemical expansion
        MatrixNsd alpha_T_; //coefficient for thermal expansion
        double lambda_; //first Lamé parameter
        double mu_; //second Lamé parameter (shear modulus)
};

#include "DiffusionEvaluator.tpp" //include the implementation of the template class