#pragma once

#include "DiffusionEvaluator.hpp"

template <unsigned int Nsd, unsigned int Nne, unsigned int BfOrder>
DiffusionEvaluator<Nsd,Nne,BfOrder>::DiffusionEvaluator(
    const Mesh<Nsd,Nne>& mesh,
    const QuadratureRule<Nsd,Nne>& quadRule,
    double DC,
    double DT,
    const MatrixNsd& alpha_C,
    const MatrixNsd& alpha_T,
    double lambda,
    double mu
) : 
    mesh_(mesh),
    quadRule_(quadRule),
    DC_(DC),
    DT_(DT),
    alpha_C_(alpha_C),
    alpha_T_(alpha_T),
    lambda_(lambda),
    mu_(mu)
{}

template <unsigned int Nsd, unsigned int Nne, unsigned int BfOrder>
typename DiffusionEvaluator<Nsd,Nne,BfOrder>::MatrixNsd
DiffusionEvaluator<Nsd,Nne,BfOrder>::computeJacobian(unsigned int e, const VectorNsd& xi_vec) const{
    MatrixNsd J = MatrixNsd::Zero();
    
    if constexpr (Nsd == 2){
        for(int A = 0 ; A < Nne ; A++){
            VectorNsd basis_gradient_vec = ShapeFunction<Nsd,Nne,BfOrder>::basis_gradient(A, xi_vec);
            
            unsigned int Aglobal = mesh_.elements[e].node[A];
            J(0,0) += basis_gradient_vec(0)*mesh_.nodes[Aglobal].x1; //dx1/dxi1
            J(0,1) += basis_gradient_vec(1)*mesh_.nodes[Aglobal].x1; //dx1/dxi2
            J(1,0) += basis_gradient_vec(0)*mesh_.nodes[Aglobal].x2; //dx2/dxi1
            J(1,1) += basis_gradient_vec(1)*mesh_.nodes[Aglobal].x2; //dx2/dxi2
        }
    }
    else if constexpr (Nsd == 3){
        for(int A = 0 ; A < Nne ; A++){
            VectorNsd basis_gradient_vec = ShapeFunction<Nsd,Nne,BfOrder>::basis_gradient(A, xi_vec);
            
            unsigned int Aglobal = mesh_.elements[e].node[A];
            J(0,0) += basis_gradient_vec(0)*mesh_.nodes[Aglobal].x1; //dx1/dxi1
            J(0,1) += basis_gradient_vec(1)*mesh_.nodes[Aglobal].x1; //dx1/dxi2
            J(0,2) += basis_gradient_vec(2)*mesh_.nodes[Aglobal].x1; //dx1/dxi3
            J(1,0) += basis_gradient_vec(0)*mesh_.nodes[Aglobal].x2; //dx2/dxi1
            J(1,1) += basis_gradient_vec(1)*mesh_.nodes[Aglobal].x2; //dx2/dxi2
            J(1,2) += basis_gradient_vec(2)*mesh_.nodes[Aglobal].x2; //dx2/dxi3
            J(2,0) += basis_gradient_vec(0)*mesh_.nodes[Aglobal].x3; //dx3/dxi1
            J(2,1) += basis_gradient_vec(1)*mesh_.nodes[Aglobal].x3; //dx3/dxi2
            J(2,2) += basis_gradient_vec(2)*mesh_.nodes[Aglobal].x3; //dx3/dxi3
        }
    }

    return J;
}

template <unsigned int Nsd, unsigned int Nne, unsigned int BfOrder>
void DiffusionEvaluator<Nsd,Nne,BfOrder>::computeDiffusionMatrices(
    unsigned int e, //element index

    Eigen::MatrixXd& MCClocal, //element diffusion stiffness matrix for chemical concentration (Nne x Nne matrix)
    Eigen::MatrixXd& KCClocal,
    Eigen::MatrixXd& KCTlocal,

    Eigen::MatrixXd& MTTlocal, //element diffusion stiffness matrix for temperature (Nne x Nne matrix)
    Eigen::MatrixXd& KTTlocal,
    Eigen::MatrixXd& KTClocal,
    
    Eigen::MatrixXd& KuClocal, //terms from mechanics weak coupling (Nne*Nsd x Nne  matrix)
    Eigen::MatrixXd& KuTlocal //(Nne*Nsd x Nne  matrix)
) const {
    double alpha_tilde_CC = lambda_*alpha_C_.trace()*alpha_C_.trace() + 2*mu_*(alpha_C_*alpha_C_).trace();
    double alpha_tilde_CT = lambda_*alpha_C_.trace()*alpha_T_.trace() + 2*mu_*(alpha_C_*alpha_T_).trace();
    double alpha_tilde_TT = lambda_*alpha_T_.trace()*alpha_T_.trace() + 2*mu_*(alpha_T_*alpha_T_).trace();
    double alpha_tilde_TC = lambda_*alpha_T_.trace()*alpha_C_.trace() + 2*mu_*(alpha_T_*alpha_C_).trace();

    MatrixNsd Lambda_C = lambda_*alpha_C_.trace()*MatrixNsd::Identity() + 2*mu_*alpha_C_;
    MatrixNsd Lambda_T = lambda_*alpha_T_.trace()*MatrixNsd::Identity() + 2*mu_*alpha_T_;

    //Gaussian quadrature loop
    if constexpr (Nsd == 2){
        if constexpr (Nne == 3){
            const auto& quad_points_x1 = quadRule_.points_x1;
            const auto& quad_points_x2 = quadRule_.points_x2;
            const auto& quad_weights = quadRule_.weights;
            unsigned int quadOrder = quad_points_x1.size(); //number of quadrature points in each direction

            for (int I = 0 ; I < quadOrder ; I++){
                double xi1 = quad_points_x1[I];
                double xi2 = quad_points_x2[I];
                double weight = quad_weights[I];

                VectorNsd xi_vec(xi1,xi2);

                MatrixNsd Jac = computeJacobian(e, xi_vec); //compute the Jacobian matrix at the quadrature point
                double JacDet = Jac.determinant();
                MatrixNsd JacInv = Jac.inverse();

                for(int A = 0 ; A < Nne ; A++){
                    double N_A = ShapeFunction<Nsd,Nne,BfOrder>::basis_function(A, xi_vec);
                    VectorNsd basis_gradient_vecA = ShapeFunction<Nsd,Nne,BfOrder>::basis_gradient(A, xi_vec);
                    VectorNsd dNA_dx = JacInv.transpose()*basis_gradient_vecA;

                    for(int B = 0 ; B < Nne ; B++){
                        double N_B = ShapeFunction<Nsd,Nne,BfOrder>::basis_function(B, xi_vec);
                        VectorNsd basis_gradient_vecB = ShapeFunction<Nsd,Nne,BfOrder>::basis_gradient(B, xi_vec);
                        VectorNsd dNB_dx = JacInv.transpose()*basis_gradient_vecB;

                        MCClocal(A,B) += N_A * N_B * weight * JacDet;
                        MTTlocal(A,B) += N_A * N_B * weight * JacDet;

                        KCClocal(A,B) += dNA_dx.dot(dNB_dx) * DC_ * (1 + alpha_tilde_CC)* weight * JacDet;
                        KCTlocal(A,B) += dNA_dx.dot(dNB_dx) * DC_ * alpha_tilde_CT * weight * JacDet;
                        KTTlocal(A,B) += dNA_dx.dot(dNB_dx) * DT_ * (1 + alpha_tilde_TT)* weight * JacDet;
                        KTClocal(A,B) += dNA_dx.dot(dNB_dx) * DT_ * alpha_tilde_TC * weight * JacDet;

                        VectorNsd KuC_block = - Lambda_C * dNA_dx * N_B * weight * JacDet; //Nsd x 1 vector
                        VectorNsd KuT_block = - Lambda_T * dNA_dx * N_B * weight * JacDet; //Nsd x 1 vector

                        KuClocal.block<Nsd,1>(Nsd*A,B) += KuC_block;
                        KuTlocal.block<Nsd,1>(Nsd*A,B) += KuT_block;
                    }   
                }
            }
        }
        else if constexpr (Nne == 4 || Nne == 9){
            const auto& quad_points = quadRule_.points;
            const auto& quad_weights = quadRule_.weights;
            unsigned int quadOrder = quad_points.size(); //number of quadrature points in each direction

            for(int I = 0 ; I < quadOrder ; I++){
                for(int J = 0 ; J < quadOrder ; J++){
                    //Get the quadrature point coordinates and weights
                    double xi1 = quad_points[I]; 
                    double xi2 = quad_points[J];
                    
                    double weight = quad_weights[I] * quad_weights[J];

                    VectorNsd xi_vec(xi1,xi2);
                    
                    MatrixNsd Jac = computeJacobian(e, xi_vec); //compute the Jacobian matrix at the quadrature point
                    double JacDet = Jac.determinant();
                    MatrixNsd JacInv = Jac.inverse();

                    for(int A = 0 ; A < Nne ; A++){
                        double N_A = ShapeFunction<Nsd,Nne,BfOrder>::basis_function(A, xi_vec);
                        VectorNsd basis_gradient_vecA = ShapeFunction<Nsd,Nne,BfOrder>::basis_gradient(A, xi_vec);
                        VectorNsd dNA_dx = JacInv.transpose()*basis_gradient_vecA;

                        for(int B = 0 ; B < Nne ; B++){
                            double N_B = ShapeFunction<Nsd,Nne,BfOrder>::basis_function(B, xi_vec);
                            VectorNsd basis_gradient_vecB = ShapeFunction<Nsd,Nne,BfOrder>::basis_gradient(B, xi_vec);
                            VectorNsd dNB_dx = JacInv.transpose()*basis_gradient_vecB;

                            MCClocal(A,B) += N_A * N_B * weight * JacDet;
                            MTTlocal(A,B) += N_A * N_B * weight * JacDet;

                            KCClocal(A,B) += dNA_dx.dot(dNB_dx) * DC_ * (1 + alpha_tilde_CC)* weight * JacDet;
                            KCTlocal(A,B) += dNA_dx.dot(dNB_dx) * DC_ * alpha_tilde_CT * weight * JacDet;
                            KTTlocal(A,B) += dNA_dx.dot(dNB_dx) * DT_ * (1 + alpha_tilde_TT)* weight * JacDet;
                            KTClocal(A,B) += dNA_dx.dot(dNB_dx) * DT_ * alpha_tilde_TC * weight * JacDet;

                            VectorNsd KuC_block = - Lambda_C * dNA_dx * N_B * weight * JacDet; //Nsd x 1 vector
                            VectorNsd KuT_block = - Lambda_T * dNA_dx * N_B * weight * JacDet; //Nsd x 1 vector

                            KuClocal.block<Nsd,1>(Nsd*A,B) += KuC_block;
                            KuTlocal.block<Nsd,1>(Nsd*A,B) += KuT_block;
                        }   
                    }
                } 
            }
        }
    }
    else if constexpr (Nsd == 3){
        const auto& quad_points = quadRule_.points;
        const auto& quad_weights = quadRule_.weights;
        unsigned int quadOrder = quad_points.size(); //number of quadrature points in each direction

        for(int I = 0 ; I < quadOrder ; I++){
            for(int J = 0 ; J < quadOrder ; J++){
                for(int K = 0 ; K < quadOrder ; K++){
                    //Get the quadrature point coordinates and weights
                    double xi1 = quad_points[I]; 
                    double xi2 = quad_points[J];
                    double xi3 = quad_points[K];
                    double weight = quad_weights[I] * quad_weights[J] * quad_weights[K];

                    VectorNsd xi_vec(xi1,xi2,xi3);

                    MatrixNsd Jac = computeJacobian(e, xi_vec); //compute the Jacobian matrix at the quadrature point
                    double JacDet = Jac.determinant();
                    MatrixNsd JacInv = Jac.inverse();

                    for(int A = 0 ; A < Nne ; A++){
                        double N_A = ShapeFunction<Nsd,Nne,BfOrder>::basis_function(A, xi_vec);
                        VectorNsd basis_gradient_vecA = ShapeFunction<Nsd,Nne,BfOrder>::basis_gradient(A, xi_vec);
                        VectorNsd dNA_dx = JacInv.transpose()*basis_gradient_vecA;

                        for(int B = 0 ; B < Nne ; B++){
                            double N_B = ShapeFunction<Nsd,Nne,BfOrder>::basis_function(B, xi_vec);
                            VectorNsd basis_gradient_vecB = ShapeFunction<Nsd,Nne,BfOrder>::basis_gradient(B, xi_vec);
                            VectorNsd dNB_dx = JacInv.transpose()*basis_gradient_vecB;

                            MCClocal(A,B) += N_A * N_B * weight * JacDet;
                            MTTlocal(A,B) += N_A * N_B * weight * JacDet;

                            KCClocal(A,B) += dNA_dx.dot(dNB_dx) * DC_ * (1 + alpha_tilde_CC)* weight * JacDet;
                            KCTlocal(A,B) += dNA_dx.dot(dNB_dx) * DC_ * alpha_tilde_CT * weight * JacDet;
                            KTTlocal(A,B) += dNA_dx.dot(dNB_dx) * DT_ * (1 + alpha_tilde_TT)* weight * JacDet;
                            KTClocal(A,B) += dNA_dx.dot(dNB_dx) * DT_ * alpha_tilde_TC * weight * JacDet;

                            VectorNsd KuC_block = - Lambda_C * dNA_dx * N_B * weight * JacDet; //Nsd x 1 vector
                            VectorNsd KuT_block = - Lambda_T * dNA_dx * N_B * weight * JacDet; //Nsd x 1 vector

                            KuClocal.block<Nsd,1>(Nsd*A,B) += KuC_block;
                            KuTlocal.block<Nsd,1>(Nsd*A,B) += KuT_block;
                        }   
                    }
                }
            }
        }
    }
}