#include "StVenantKirchhoff.hpp"

template<unsigned int Nsd, unsigned int Nne>
StVenantKirchhoff<Nsd,Nne>::StVenantKirchhoff( //assign the function parameters to the class member variables using an initializer list
    double lambda,
    double mu
): lambda_(lambda), mu_(mu)
{}

template<unsigned int Nsd, unsigned int Nne>
void StVenantKirchhoff<Nsd,Nne>::compute(
    const MatrixNsd& F, //deformation gradient
    MatrixNsd& S, //second Piola-Kirchhoff stress tensor
    MatrixNsd& P, //first Piola-Kirchhoff stress tensor
    Eigen::MatrixXd& C_mat
) const {

    MatrixNsd E = 0.5 * (F.transpose() * F - MatrixNsd::Identity()); //Green-Lagrange strain
    S = 2*mu_*E + lambda_*E.trace()*MatrixNsd::Identity(); //second Piola-Kirchhoff stress using St. Venant-Kirchhoff model
    P = F * S; //first Piola-Kirchhoff stress

    C_mat = Eigen::MatrixXd::Zero(9,9); //material tangent stiffness matrix (4th order elasticity tensor in Voigt notation)
    for(int P = 0; P < 3; P++){
        for(int Q = 0; Q < 3; Q++){
            for(int M = 0; M < 3; M++){
                for(int N = 0; N < 3; N++){
                    double C_PQMN = lambda_*(P==Q ? 1:0)*(M==N ? 1:0) + mu_*((P==M ? 1:0)*(Q==N ? 1:0) + (P==N ? 1:0)*(Q==M ? 1:0));
                    C_mat(P*3+Q, M*3+N) = C_PQMN;
                }
            }
        }
    }
}
