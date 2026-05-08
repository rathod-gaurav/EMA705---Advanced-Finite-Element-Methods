#include "StVenantKirchhoff.hpp"

template<unsigned int Nsd, unsigned int Nne>
StVenantKirchhoff<Nsd,Nne>::StVenantKirchhoff( //assign the function parameters to the class member variables using an initializer list
    double lambda,
    double mu,
    const MatrixNsd& alpha_C,
    const MatrixNsd& alpha_T
): lambda_(lambda), mu_(mu), alpha_C_(alpha_C), alpha_T_(alpha_T)
{}

template<unsigned int Nsd, unsigned int Nne>
void StVenantKirchhoff<Nsd,Nne>::compute(
    const MatrixNsd& F, //deformation gradient
    const double& C_val, //chemical concentration
    const double& T_val, //temperature
    MatrixNsd& S, //second Piola-Kirchhoff stress tensor
    MatrixNsd& P, //first Piola-Kirchhoff stress tensor
    Eigen::MatrixXd& C_mat
) const {

    MatrixNsd F_th = MatrixNsd::Identity() + alpha_T_*T_val; //thermal expansion deformation gradient
    MatrixNsd F_ch = MatrixNsd::Identity() + alpha_C_*C_val; //chemical expansion deformation gradient
    MatrixNsd F_el = F * F_th.inverse() * F_ch.inverse(); //elastic deformation gradient

    MatrixNsd E_el = 0.5*(F_el.transpose()*F_el - MatrixNsd::Identity()); //elastic Green-Lagrange strain

    MatrixNsd S_el = 2*mu_*E_el + lambda_*E_el.trace()*MatrixNsd::Identity(); //elastic second Piola-Kirchhoff stress

    S = F_th.inverse().transpose() * F_ch.inverse().transpose() * S_el * F_ch.inverse() * F_th.inverse(); //total second Piola-Kirchhoff stress

    P = F*S; //total first Piola-Kirchhoff stress

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
