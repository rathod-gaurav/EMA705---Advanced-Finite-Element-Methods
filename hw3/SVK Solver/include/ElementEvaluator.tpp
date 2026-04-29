#pragma once

template <unsigned int Nsd, unsigned int Nne>
ElementEvaluator<Nsd, Nne>::ElementEvaluator(
    const Mesh<Nsd,Nne>& mesh,
    const MaterialModel<Nsd,Nne>& material,
    const QuadratureRule<Nsd,Nne>& quadRule
) : 
    mesh_(mesh),
    material_(material),
    quadRule_(quadRule)
{}

template <unsigned int Nsd, unsigned int Nne>
typename ElementEvaluator<Nsd,Nne>::MatrixNsd
ElementEvaluator<Nsd, Nne>::computeJacobian(unsigned int e, const VectorNsd& xi_vec) const{
    MatrixNsd J = MatrixNsd::Zero();
    
    if constexpr (Nsd == 2){
        for(int A = 0 ; A < Nne ; A++){
            VectorNsd basis_gradient_vec = ShapeFunction<Nsd,Nne>::basis_gradient(A, xi_vec);
            
            unsigned int Aglobal = mesh_.elements[e].node[A];
            J(0,0) += basis_gradient_vec(0)*mesh_.nodes[Aglobal].x1; //dx1/dxi1
            J(0,1) += basis_gradient_vec(1)*mesh_.nodes[Aglobal].x1; //dx1/dxi2
            J(1,0) += basis_gradient_vec(0)*mesh_.nodes[Aglobal].x2; //dx2/dxi1
            J(1,1) += basis_gradient_vec(1)*mesh_.nodes[Aglobal].x2; //dx2/dxi2
        }
    }
    else if constexpr (Nsd == 3){
        for(int A = 0 ; A < Nne ; A++){
            VectorNsd basis_gradient_vec = ShapeFunction<Nsd,Nne>::basis_gradient(A, xi_vec);
            
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

template <unsigned int Nsd, unsigned int Nne>
typename ElementEvaluator<Nsd,Nne>::MatrixNsd
ElementEvaluator<Nsd, Nne>::computeGradU(const Eigen::VectorXd& u_e, const VectorNsd& xi_vec, const MatrixNsd& JacInv) const {
    MatrixNsd grad_u = MatrixNsd::Zero();
    //compute the gradient of the displacement field at the quadrature point using the basis function gradients and the nodal displacements
    
    if constexpr (Nsd == 2){
        for(int A = 0 ; A < Nne ; A++){
            VectorNsd basis_gradient_vec = ShapeFunction<Nsd,Nne>::basis_gradient(A, xi_vec);
            VectorNsd dN_dx = JacInv.transpose()*basis_gradient_vec;
            
            grad_u(0,0) += dN_dx[0] * u_e(A*Nsd + 0); //du1/dx1
            grad_u(0,1) += dN_dx[1] * u_e(A*Nsd + 0); //du1/dx2

            grad_u(1,0) += dN_dx[0] * u_e(A*Nsd + 1); //du2/dx1
            grad_u(1,1) += dN_dx[1] * u_e(A*Nsd + 1); //du2/dx2
        }
    }
    else if constexpr (Nsd == 3){
        for(int A = 0 ; A < Nne ; A++){
            VectorNsd basis_gradient_vec = ShapeFunction<Nsd,Nne>::basis_gradient(A, xi_vec);
            VectorNsd dN_dx = JacInv.transpose()*basis_gradient_vec;
            
            grad_u(0,0) += dN_dx[0] * u_e(A*Nsd + 0); //du1/dx1
            grad_u(0,1) += dN_dx[1] * u_e(A*Nsd + 0); //du1/dx2
            grad_u(0,2) += dN_dx[2] * u_e(A*Nsd + 0); //du1/dx3

            grad_u(1,0) += dN_dx[0] * u_e(A*Nsd + 1); //du2/dx1
            grad_u(1,1) += dN_dx[1] * u_e(A*Nsd + 1); //du2/dx2
            grad_u(1,2) += dN_dx[2] * u_e(A*Nsd + 1); //du2/dx3

            grad_u(2,0) += dN_dx[0] * u_e(A*Nsd + 2); //du3/dx1
            grad_u(2,1) += dN_dx[1] * u_e(A*Nsd + 2); //du3/dx2
            grad_u(2,2) += dN_dx[2] * u_e(A*Nsd + 2); //du3/dx3
        }
    }
    return grad_u;
}

template <unsigned int Nsd, unsigned int Nne>
void ElementEvaluator<Nsd, Nne>::computeElement(
    unsigned int e, //element index
    const Eigen::VectorXd& u_e, //element nodal displacements (Nne*3 x 1 vector)
    Eigen::MatrixXd& Klocal, //element stiffness matrix (Nne*3 x Nne*3 matrix)
    Eigen::VectorXd& Rlocal //element internal force vector (Nne*3 x 1 vector)
) const {
    Rlocal = Eigen::VectorXd::Zero(Nne * Nsd); //local residual vector for the element
    Klocal = Eigen::MatrixXd::Zero(Nne * Nsd, Nne * Nsd); //local tangent stiffness matrix for the element

    MatrixNsd S = MatrixNsd::Zero(); //Second Piola-Kirchhoff stress tensor
    MatrixNsd P = MatrixNsd::Zero(); //First Piola-Kirchhoff stress tensor
    Eigen::MatrixXd C_mat = Eigen::MatrixXd::Zero(9,9); //material tangent stiffness matrix in Voigt notation (3x3 block for each pair of nodes)

    //Gaussian quadrature loop
    if constexpr (Nsd == 2){
        if constexpr (Nne == 3){
            const auto& quad_points_x1 = quadRule_.points_x1;
            const auto& quad_points_x2 = quadRule_.points_x2;
            const auto& quad_weights = quadRule_.weights;
            unsigned int quadOrder = quad_points_x1.size(); //number of quadrature points in each direction

            for(int I = 0 ; I < quadOrder ; I++){
                //Get the quadrature point coordinates and weights
                double xi1 = quad_points_x1[I]; 
                double xi2 = quad_points_x2[I];
                
                double weight = quad_weights[I];

                VectorNsd xi_vec(xi1,xi2);

                MatrixNsd Jac = computeJacobian(e, xi_vec); //compute the Jacobian matrix at the quadrature point
                double JacDet = Jac.determinant(); //compute the determinant of the Jacobian
                MatrixNsd JacInv = Jac.inverse(); //compute the inverse of the Jacobian

                MatrixNsd grad_u = computeGradU(u_e, xi_vec, JacInv); //compute the gradient of the displacement field at the quadrature point
                MatrixNsd F = MatrixNsd::Identity() + grad_u; //deformation gradient

                material_.compute(F, S, P, C_mat); //compute the stress tensors E,S,P and material tangent stiffness matrix at the quadrature point using the material model
                
                for(int B = 0 ; B < Nne ; B++){//Loop to calculate Residual
                    VectorNsd basis_gradient_vec = ShapeFunction<Nsd,Nne>::basis_gradient(B, xi_vec);
                    VectorNsd dN_dx = JacInv.transpose()*basis_gradient_vec;
                    
                    Rlocal.segment(B*Nsd, Nsd) += P * dN_dx * weight * JacDet; //contribution to the local residual vector
                }
                
                // cout << "Calculated Rlocal for element " << e+1 << "/" << Nel_t << "\r";
                
                for(int A = 0 ; A < Nne ; A++){//Loops to calculate tangent matrix
                    VectorNsd basis_gradient_vec = ShapeFunction<Nsd,Nne>::basis_gradient(A, xi_vec);
                    VectorNsd dNA_dx = JacInv.transpose()*basis_gradient_vec;
                    
                    
                    
                    for(int B = 0 ; B < Nne ; B++){
                        VectorNsd basis_gradient_vec = ShapeFunction<Nsd,Nne>::basis_gradient(B, xi_vec);
                        VectorNsd dNB_dx = JacInv.transpose()*basis_gradient_vec;
                        

                        //Kgeometric
                        double Kgeo_scalar = (dNA_dx.transpose() * S * dNB_dx)(0,0);
                        MatrixNsd KgeoAB =  Kgeo_scalar * JacDet * weight * MatrixNsd::Identity();
                        Klocal.block<Nsd,Nsd>(Nsd*A,Nsd*B) += KgeoAB;


                        // Correct KmatAB (3x3 block for nodes A,B)
                        MatrixNsd KmatAB = MatrixNsd::Zero();
                        
                        for(int i = 0; i < Nsd; i++){
                            for(int j = 0; j < Nsd; j++){
                                double val = 0.0;
                                for(int P = 0; P < Nsd; P++){
                                    for(int Q = 0; Q < Nsd; Q++){
                                        for(int M = 0; M < Nsd; M++){
                                            for(int N = 0; N < Nsd; N++){
                                                double C_PQMN = C_mat(3*P + Q, 3*M + N); //material tangent stiffness in Voigt notation
                                                val += F(i,P)*C_PQMN*F(j,M)*dNA_dx(Q)*dNB_dx(N);
                                            }
                                        }
                                    }
                                }
                                KmatAB(i,j) = val;
                            }
                        }
                        KmatAB *= JacDet * weight;
                        Klocal.block<Nsd,Nsd>(Nsd*A,Nsd*B) += KmatAB;
                    }
                }
                
                // cout << "Calculated Klocal for element " << e+1 << "/" << Nel_t << "\r";
                
            }
        }
        else if constexpr (Nne == 4){
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
                    double JacDet = Jac.determinant(); //compute the determinant of the Jacobian
                    MatrixNsd JacInv = Jac.inverse(); //compute the inverse of the Jacobian

                    MatrixNsd grad_u = computeGradU(u_e, xi_vec, JacInv); //compute the gradient of the displacement field at the quadrature point
                    MatrixNsd F = MatrixNsd::Identity() + grad_u; //deformation gradient

                    material_.compute(F, S, P, C_mat); //compute the stress tensors E,S,P and material tangent stiffness matrix at the quadrature point using the material model
                    
                    for(int B = 0 ; B < Nne ; B++){//Loop to calculate Residual
                        VectorNsd basis_gradient_vec = ShapeFunction<Nsd,Nne>::basis_gradient(B, xi_vec);
                        VectorNsd dN_dx = JacInv.transpose()*basis_gradient_vec;
                        
                        Rlocal.segment(B*Nsd, Nsd) += P * dN_dx * weight * JacDet; //contribution to the local residual vector
                    }
                    
                    // cout << "Calculated Rlocal for element " << e+1 << "/" << Nel_t << "\r";
                    
                    for(int A = 0 ; A < Nne ; A++){//Loops to calculate tangent matrix
                        VectorNsd basis_gradient_vec = ShapeFunction<Nsd,Nne>::basis_gradient(A, xi_vec);
                        VectorNsd dNA_dx = JacInv.transpose()*basis_gradient_vec;
                        
                        
                        
                        for(int B = 0 ; B < Nne ; B++){
                            VectorNsd basis_gradient_vec = ShapeFunction<Nsd,Nne>::basis_gradient(B, xi_vec);
                            VectorNsd dNB_dx = JacInv.transpose()*basis_gradient_vec;
                            

                            //Kgeometric
                            double Kgeo_scalar = (dNA_dx.transpose() * S * dNB_dx)(0,0);
                            MatrixNsd KgeoAB =  Kgeo_scalar * JacDet * weight * MatrixNsd::Identity();
                            Klocal.block<Nsd,Nsd>(Nsd*A,Nsd*B) += KgeoAB;


                            // Correct KmatAB (3x3 block for nodes A,B)
                            MatrixNsd KmatAB = MatrixNsd::Zero();
                            
                            for(int i = 0; i < Nsd; i++){
                                for(int j = 0; j < Nsd; j++){
                                    double val = 0.0;
                                    for(int P = 0; P < Nsd; P++){
                                        for(int Q = 0; Q < Nsd; Q++){
                                            for(int M = 0; M < Nsd; M++){
                                                for(int N = 0; N < Nsd; N++){
                                                    double C_PQMN = C_mat(3*P + Q, 3*M + N); //material tangent stiffness in Voigt notation
                                                    val += F(i,P)*C_PQMN*F(j,M)*dNA_dx(Q)*dNB_dx(N);
                                                }
                                            }
                                        }
                                    }
                                    KmatAB(i,j) = val;
                                }
                            }
                            KmatAB *= JacDet * weight;
                            Klocal.block<Nsd,Nsd>(Nsd*A,Nsd*B) += KmatAB;
                        }
                    }
                    
                    // cout << "Calculated Klocal for element " << e+1 << "/" << Nel_t << "\r";
                    
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
                    double JacDet = Jac.determinant(); //compute the determinant of the Jacobian
                    MatrixNsd JacInv = Jac.inverse(); //compute the inverse of the Jacobian

                    MatrixNsd grad_u = computeGradU(u_e, xi_vec, JacInv); //compute the gradient of the displacement field at the quadrature point
                    MatrixNsd F = MatrixNsd::Identity() + grad_u; //deformation gradient

                    material_.compute(F, S, P, C_mat); //compute the stress tensors E,S,P and material tangent stiffness matrix at the quadrature point using the material model
                    
                    
                    for(int B = 0 ; B < Nne ; B++){//Loop to calculate Residual
                        VectorNsd basis_gradient_vec = ShapeFunction<Nsd,Nne>::basis_gradient(B, xi_vec);
                        VectorNsd dN_dx = JacInv.transpose()*basis_gradient_vec;
                        
                        Rlocal.segment(B*Nsd, Nsd) += P * dN_dx * weight * JacDet; //contribution to the local residual vector
                    }
                    
                    // cout << "Calculated Rlocal for element " << e+1 << "/" << Nel_t << "\r";
                    
                    for(int A = 0 ; A < Nne ; A++){//Loops to calculate tangent matrix
                        VectorNsd basis_gradient_vec = ShapeFunction<Nsd,Nne>::basis_gradient(A, xi_vec);
                        VectorNsd dNA_dx = JacInv.transpose()*basis_gradient_vec;
                        
                        
                        
                        for(int B = 0 ; B < Nne ; B++){
                            VectorNsd basis_gradient_vec = ShapeFunction<Nsd,Nne>::basis_gradient(B, xi_vec);
                            VectorNsd dNB_dx = JacInv.transpose()*basis_gradient_vec;
                            

                            //Kgeometric
                            double Kgeo_scalar = (dNA_dx.transpose() * S * dNB_dx)(0,0);
                            MatrixNsd KgeoAB =  Kgeo_scalar * JacDet * weight * MatrixNsd::Identity();
                            Klocal.block<Nsd,Nsd>(Nsd*A,Nsd*B) += KgeoAB;


                            // Correct KmatAB (3x3 block for nodes A,B)
                            MatrixNsd KmatAB = MatrixNsd::Zero();
                            
                            for(int i = 0; i < Nsd; i++){
                                for(int j = 0; j < Nsd; j++){
                                    double val = 0.0;
                                    for(int P = 0; P < Nsd; P++){
                                        for(int Q = 0; Q < Nsd; Q++){
                                            for(int M = 0; M < Nsd; M++){
                                                for(int N = 0; N < Nsd; N++){
                                                    double C_PQMN = C_mat(3*P + Q, 3*M + N); //material tangent stiffness in Voigt notation
                                                    val += F(i,P)*C_PQMN*F(j,M)*dNA_dx(Q)*dNB_dx(N);
                                                }
                                            }
                                        }
                                    }
                                    KmatAB(i,j) = val;
                                }
                            }
                            KmatAB *= JacDet * weight;
                            Klocal.block<Nsd,Nsd>(Nsd*A,Nsd*B) += KmatAB;
                        }
                    }
                    
                    // cout << "Calculated Klocal for element " << e+1 << "/" << Nel_t << "\r";
                    
                }
            }
        }
    }
    
}