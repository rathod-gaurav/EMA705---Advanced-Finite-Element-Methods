// nonlinear FE Problem for finite strain mechanics using St. Venant-Kirchhoff strain energy density function

#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <Eigen/Dense>

using namespace std;

struct Node {
    float x1, x2, x3; //global coordinates of the node
};

template <unsigned int Nne>
struct Element {
    unsigned int node[Nne]; //IDs of the nodes that form the element
};

std::tuple<float,float,float> xi_at_node(unsigned int node){ //function to return xi1, xi2, and xi3 for given node A
        float xi1, xi2, xi3;
        switch(node){
            case 0:
                xi1 = -1.0;
                xi2 = -1.0;
                xi3 = -1.0;
                break;
            case 1:
                xi1 = 1.0;
                xi2 = -1.0;
                xi3 = -1.0;
                break;
            case 2:
                xi1 = 1.0;
                xi2 = 1.0;
                xi3 = -1.0;
                break;
            case 3:
                xi1 = -1.0;
                xi2 = 1.0;
                xi3 = -1.0;
                break;
            case 4:
                xi1 = -1.0;
                xi2 = -1.0;
                xi3 = 1.0;
                break;
            case 5:
                xi1 = 1.0;
                xi2 = -1.0;
                xi3 = 1.0;
                break;
            case 6:
                xi1 = 1.0;
                xi2 = 1.0;
                xi3 = 1.0;
                break;
            case 7:
                xi1 = -1.0;
                xi2 = 1.0;
                xi3 = 1.0;
                break;
            default:
                throw std::invalid_argument("xi_at_node mapping not implemented for this local node number");
        }
        return {xi1, xi2, xi3};
};

float basis_function(unsigned int node, float xi1, float xi2, float xi3){
        auto [xi1_node , xi2_node , xi3_node] = xi_at_node(node);
        float value = 0.125*(1 + xi1*xi1_node)*(1 + xi2*xi2_node)*(1 + xi3*xi3_node);
        return value;
};

std::tuple<float,float,float> basis_gradient(unsigned int node, float xi1, float xi2, float xi3){
    auto [xi1_node,xi2_node,xi3_node] = xi_at_node(node);
    float basis_gradient_xi1, basis_gradient_xi2, basis_gradient_xi3;
    basis_gradient_xi1 = 0.125*xi1_node*(1 + xi2*xi2_node)*(1 + xi3*xi3_node);
    basis_gradient_xi2 = 0.125*xi2_node*(1 + xi1*xi1_node)*(1 + xi3*xi3_node);
    basis_gradient_xi3 = 0.125*xi3_node*(1 + xi1*xi1_node)*(1 + xi2*xi2_node);
    return {basis_gradient_xi1, basis_gradient_xi2, basis_gradient_xi3};
}

struct QuadratureRule {
    std::vector<float> points;
    std::vector<float> weights;
};

QuadratureRule gauss_legendre(unsigned int n) {
    QuadratureRule rule;

    switch(n) {
        case 1:
            rule.points  = { 0.0 };
            rule.weights = { 2.0 };
            break;

        case 2:
            rule.points  = { -0.5773502691896257,  0.5773502691896257 };
            rule.weights = {  1.0,                 1.0 };
            break;

        case 3:
            rule.points  = { -0.7745966692414834, 0.0, 0.7745966692414834 };
            rule.weights = {  0.5555555555555556, 0.8888888888888888, 0.5555555555555556 };
            break;

        case 4:
            rule.points  = { -0.8611363115940526, -0.3399810435848563,
                              0.3399810435848563,  0.8611363115940526 };
            rule.weights = {  0.3478548451374539,  0.6521451548625461,
                              0.6521451548625461,  0.3478548451374539 };
            break;

        case 5:
            rule.points  = { -0.9061798459386640, -0.5384693101056831,
                              0.0,
                              0.5384693101056831,  0.9061798459386640 };
            rule.weights = {  0.2369268850561891,  0.4786286704993665,
                              0.5688888888888889,  0.4786286704993665,
                              0.2369268850561891 };
            break;

        case 6:
            rule.points  = { -0.9324695142031521, -0.6612093864662645,
                             -0.2386191860831969,  0.2386191860831969,
                              0.6612093864662645,  0.9324695142031521 };
            rule.weights = {  0.1713244923791704,  0.3607615730481386,
                              0.4679139345726910,  0.4679139345726910,
                              0.3607615730481386,  0.1713244923791704 };
            break;

        default:
            throw std::invalid_argument("Gauss-Legendre quadrature not implemented for this n");
    }

    return rule;
}

template <unsigned int Nne>
Eigen::MatrixXf compute_grad_u(Eigen::VectorXf u_e, float xi1, float xi2, float xi3, Eigen::MatrixXf JacInv){
    Eigen::MatrixXf grad_u = Eigen::MatrixXf::Zero(3,3);
    //compute the gradient of the displacement field at the quadrature point using the basis function gradients and the nodal displacements
    for(int A = 0 ; A < Nne ; A++){
        auto [dN_dxi1, dN_dxi2, dN_dxi3] = basis_gradient(A, xi1, xi2, xi3);
        Eigen::VectorXf dN_dx = JacInv.transpose()*Eigen::Vector3f(dN_dxi1, dN_dxi2, dN_dxi3);
        grad_u(0,0) += dN_dx[0] * u_e(A*3 + 0); //du1/dx1
        grad_u(0,1) += dN_dx[1] * u_e(A*3 + 0); //du1/dx2
        grad_u(0,2) += dN_dx[2] * u_e(A*3 + 0); //du1/dx3

        grad_u(1,0) += dN_dx[0] * u_e(A*3 + 1); //du2/dx1
        grad_u(1,1) += dN_dx[1] * u_e(A*3 + 1); //du2/dx2
        grad_u(1,2) += dN_dx[2] * u_e(A*3 + 1); //du2/dx3

        grad_u(2,0) += dN_dx[0] * u_e(A*3 + 2); //du3/dx1
        grad_u(2,1) += dN_dx[1] * u_e(A*3 + 2); //du3/dx2
        grad_u(2,2) += dN_dx[2] * u_e(A*3 + 2); //du3/dx3
    }
    return grad_u;
}


int main(){
    unsigned int Nsd = 3; //number of spatial dimensions - 3D problem
    constexpr int Nne = 8; //number of nodes per element - 8-node hexahedral element
    unsigned int quadRule = 2; //number of quadrature points in each direction for numerical integration
    double epsilon = 1e-8; //Newton Raphson solver tolerance

    //problem variables
    float lambda = 6*1e10; //first Lamé parameter
    float mu = 2*1e10; //second Lamé parameter (shear modulus)

    //domain
    float x1_ll = 0.0;
    float x1_ul = 10.0;
    float x2_ll = 0.0;
    float x2_ul = 3.0;
    float x3_ll = 0.0;
    float x3_ul = 3.0;

    //Mesh
    unsigned int Nel_x1 = 4; //number of elements in x1 direction
    unsigned int Nel_x2 = 2; //number of elements in x2 direction
    unsigned int Nel_x3 = 2; //number of elements in x3 direction

    unsigned int Nnodes_x1 = Nel_x1 + 1; //number of nodes in x1 direction
    unsigned int Nnodes_x2 = Nel_x2 + 1; //number of nodes in x2 direction
    unsigned int Nnodes_x3 = Nel_x3 + 1; //number of nodes in x3 direction

    float dx1 = (x1_ul - x1_ll) / Nel_x1; //element size in x1 direction
    float dx2 = (x2_ul - x2_ll) / Nel_x2; //element size in x2 direction
    float dx3 = (x3_ul - x3_ll) / Nel_x3; //element size in x3 direction

    unsigned int Nel_t = Nel_x1 * Nel_x2 * Nel_x3; //total number of elements
    unsigned int Nt = Nnodes_x1 * Nnodes_x2 * Nnodes_x3; //total number of nodes

    vector<Node> nodes;
    nodes.reserve(Nt);
    for(unsigned int k = 0 ; k < Nnodes_x3 ; k++){
        for(unsigned int j = 0 ; j < Nnodes_x2 ; j++){
            for(unsigned int i = 0 ; i < Nnodes_x1 ; i++){
                Node n;
                n.x1 = x1_ll + i*dx1;
                n.x2 = x2_ll + j*dx2;
                n.x3 = x3_ll + k*dx3;
                nodes.push_back(n);
            }
        }
    }

    //Local-Global node number mapping for every element
    using Element3D = Element<Nne>;
    vector<Element3D> elements;
    elements.reserve(Nel_t);
    for(unsigned int k = 0 ; k < Nel_x3 ; k++){
        for(unsigned int j = 0 ; j < Nel_x2 ; j++){
            for(unsigned int i = 0 ; i < Nel_x1 ; i++){
                Element3D elem;
                // int n0 = i + j*Nnodes_x1 + k*(Nnodes_x1*Nnodes_x2);
                // int n1 = n0 + 1;
                // int n2 = n1 + (Nnodes_x1*Nnodes_x2);
                // int n3 = n2 - 1;
                // int n4 = i + (j+1)*Nnodes_x1 + k*(Nnodes_x1*Nnodes_x2);
                // int n5 = n4 + 1;
                // int n6 = n5 + (Nnodes_x1*Nnodes_x2);
                // int n7 = n6 - 1;

                int base = i 
                     + j * Nnodes_x1 
                     + k * (Nnodes_x1 * Nnodes_x2);

                int n0 = base;
                int n1 = base + 1;
                int n3 = base + Nnodes_x1;
                int n2 = n3 + 1;

                int n4 = base + Nnodes_x1 * Nnodes_x2;
                int n5 = n4 + 1;
                int n7 = n4 + Nnodes_x1;
                int n6 = n7 + 1;

                elem.node[0] = n0;
                elem.node[1] = n1;
                elem.node[2] = n2;
                elem.node[3] = n3;
                elem.node[4] = n4;
                elem.node[5] = n5;
                elem.node[6] = n6;
                elem.node[7] = n7;

                elements.push_back(elem);
            }
        }
    }

    //store the mesh into points and hexa files
    std::ofstream points_file("mesh/points.txt");
    for(auto& node : nodes){
        points_file << node.x1 << " " << node.x2  << " " << node.x3 << "\n";
    }

    std::ofstream hexas_file("mesh/hexas.txt");
    for(auto& elem : elements){
        hexas_file << elem.node[0] << " " << elem.node[1] << " " << elem.node[2] << " " << elem.node[3] << " " << elem.node[4] << " " << elem.node[5] << " " << elem.node[6] << " " << elem.node[7] << "\n";
    }

    // Initialize the solution vector (displacements at each node)
    Eigen::VectorXf u = Eigen::VectorXf::Zero(Nt * Nsd); //displacement vector initialized to zero
    Eigen::VectorXf du = Eigen::VectorXf::Zero(Nt * Nsd); //incremental displacement vector initialized to zero

    //Setup Newton-Raphson increment and iteration parameters
    unsigned int maxIncrement = 10; //maximum number of load increments
    unsigned int maxIter = 20; //maximum number of iterations per increment

    auto calculate_Jacobian_3D = [Nsd, elements, nodes](int e, float xi1, float xi2, float xi3){//function to calculate jacobian
        Eigen::MatrixXf J = Eigen::MatrixXf::Zero(Nsd,Nsd);
        
        for(int A = 0 ; A < Nne ; A++){
            auto [basis_gradient_xi1, basis_gradient_xi2, basis_gradient_xi3] = basis_gradient(A, xi1, xi2, xi3);
            int Aglobal = elements[e].node[A];
            J(0,0) += basis_gradient_xi1*nodes[Aglobal].x1; //dx1/dxi1
            J(0,1) += basis_gradient_xi2*nodes[Aglobal].x1; //dx1/dxi2
            J(0,2) += basis_gradient_xi3*nodes[Aglobal].x1; //dx1/dxi3
            J(1,0) += basis_gradient_xi1*nodes[Aglobal].x2; //dx2/dxi1
            J(1,1) += basis_gradient_xi2*nodes[Aglobal].x2; //dx2/dxi2
            J(1,2) += basis_gradient_xi3*nodes[Aglobal].x2; //dx2/dxi3
            J(2,0) += basis_gradient_xi1*nodes[Aglobal].x3; //dx3/dxi1
            J(2,1) += basis_gradient_xi2*nodes[Aglobal].x3; //dx3/dxi2
            J(2,2) += basis_gradient_xi3*nodes[Aglobal].x3; //dx3/dxi3
        }
        return J;
    };

    //Quadrature points
    QuadratureRule q = gauss_legendre(quadRule);
    std::vector<float> points(quadRule), weights(quadRule);
    points = q.points;
    weights = q.weights;
    Eigen::VectorXf quad_points = Eigen::Map<Eigen::VectorXf>(points.data(), points.size());
    Eigen::VectorXf quad_weights = Eigen::Map<Eigen::VectorXf>(weights.data(), weights.size());

    //Boundary Conditions

    for(unsigned int increment = 0; increment < maxIncrement; increment++){
        //Apply dirischlet BCs


        for(unsigned int iter = 0; iter < maxIter; iter++){
            Eigen::VectorXf Rglobal = Eigen::VectorXf::Zero(Nt * Nsd); //residual vector initialized to zero
            Eigen::MatrixXf Kglobal = Eigen::MatrixXf::Zero(Nt * Nsd, Nt * Nsd); //tangent stiffness matrix initialized to zero
            //Loop over elements to compute element-level contributions to R and K
            for(unsigned int e = 0; e < Nel_t; e++){
                //Get the nodes of the current element`
                Eigen::VectorXf Rlocal = Eigen::VectorXf::Zero(Nne * Nsd); //local residual vector for the element
                Eigen::MatrixXf Klocal = Eigen::MatrixXf::Zero(Nne * Nsd, Nne * Nsd); //local tangent stiffness matrix for the element
                
                //Element nodal displacements
                Eigen::VectorXf u_e = Eigen::VectorXf::Zero(Nne * Nsd); //displacement vector for the current element
                for(unsigned int i = 0; i < Nne; i++){
                    unsigned int global_node_id = elements[e].node[i];
                    u_e.segment(i*Nsd, Nsd) = u.segment(global_node_id*Nsd, Nsd); //extract the displacements for the nodes of the current element 
                }

                //Gaussian quadrature loop
                for(int I = 0 ; I < quadRule ; I++){
                    for(int J = 0 ; J < quadRule ; J++){
                        for(int K = 0 ; K < quadRule ; K++){
                            //Get the quadrature point coordinates and weights
                            float xi1 = quad_points[I]; 
                            float xi2 = quad_points[J];
                            float xi3 = quad_points[K];
                            float weight = quad_weights[I] * quad_weights[J] * quad_weights[K];
                            Eigen::MatrixXf Jac = calculate_Jacobian_3D(e, xi1, xi2, xi3); //compute the Jacobian matrix at the quadrature point
                            float JacDet = Jac.determinant(); //compute the determinant of the Jacobian
                            Eigen::MatrixXf JacInv = Jac.inverse(); //compute the inverse of the Jacobian

                            Eigen::MatrixXf grad_u = compute_grad_u<Nne>(u_e, xi1, xi2, xi3, JacInv); //compute the gradient of the displacement field at the quadrature point
                            //Compute the deformation gradient F, Green-Lagrange strain E, and the second Piola-Kirchhoff stress S at the quadrature point
                            Eigen::MatrixXf F = Eigen::MatrixXf::Identity(3,3) + grad_u; //deformation gradient
                            Eigen::MatrixXf E = 0.5 * (F.transpose() * F - Eigen::MatrixXf::Identity(3,3)); //Green-Lagrange strain
                            Eigen::MatrixXf S = 2*mu*E + lambda*E.trace()*Eigen::MatrixXf::Identity(3,3); //second Piola-Kirchhoff stress using St. Venant-Kirchhoff model
                            Eigen::MatrixXf P = F * S; //first Piola-Kirchhoff stress

                            for(int B = 0 ; B < Nne ; B++){//Loop to calculate Residual
                                auto [dN_dxi1, dN_dxi2, dN_dxi3] = basis_gradient(B, xi1, xi2, xi3);
                                Eigen::VectorXf dN_dx = JacInv.transpose()*Eigen::Vector3f(dN_dxi1, dN_dxi2, dN_dxi3); //gradient of the basis function in global coordinates
                                Rlocal.segment(B*Nsd, Nsd) += P * dN_dx * weight * JacDet; //contribution to the local residual vector
                            }
                            
                            for(int A = 0 ; A < Nne ; A++){//Loops to calculate tangent matrix
                                for(int B = 0 ; B < Nne ; B++){

                                    auto [dNA_dxi1, dNA_dxi2, dNA_dxi3] = basis_gradient(A, xi1, xi2, xi3);
                                    auto [dNB_dxi1, dNB_dxi2, dNB_dxi3] = basis_gradient(B, xi1, xi2, xi3);

                                    Eigen::VectorXf dNA_dx = JacInv.transpose()*Eigen::Vector3f(dNA_dxi1, dNA_dxi2, dNA_dxi3);
                                    Eigen::VectorXf dNB_dx = JacInv.transpose()*Eigen::Vector3f(dNB_dxi1, dNB_dxi2, dNB_dxi3);

                                    //Kgeometric
                                    float Kgeo_scalar = (dNA_dx.transpose() * S * dNB_dx)(0,0);
                                    Eigen::MatrixXf KgeoAB =  Kgeo_scalar * JacDet * weight * Eigen::MatrixXf::Identity(3,3);
                                    Klocal.block<3,3>(3*A,3*B) += KgeoAB;

                                    //Kmaterial
                                    Eigen::MatrixXf FA = F.transpose()*dNA_dx;
                                    Eigen::MatrixXf FB = F.transpose()*dNB_dx;

                                    Eigen::MatrixXf KmatAB = (lambda + mu)*FA*FB.transpose() + mu*(F * dNA_dx.cwiseProduct(dNB_dx).asDiagonal() * F.transpose()) * JacDet * weight;
                                    Klocal.block<3,3>(3*A,3*B) += KmatAB;

                                }
                            }
                        }
                    }
                }

                //Assemble Rlocal and Klocal into Rglobal and Kglobal
                for(int A = 0; A < Nne; A++){
                    int Aglobal = elements[e].node[A];
                    for(int B = 0; B < Nne ; B++)
                    {
                        int Bglobal = elements[e].node[B];
                        Kglobal.block<3,3>(3*Aglobal,3*Bglobal) += Klocal.block<3,3>(3*A,3*B);
                    }
                    Rglobal.segment(3*Aglobal,3) += Rlocal.segment(3*A,3);
                }
            }

            Eigen::LDLT<Eigen::MatrixXf> solver(Kglobal);
            du = solver.solve(Rglobal);
            if(du.norm() < epsilon*u.norm()){ 
                break;
            }
            else{
                u += du;
            }
            
        }

    }
}   