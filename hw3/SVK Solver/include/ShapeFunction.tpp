#pragma once

#include <stdexcept> //for std::invalid_argument exception

template <unsigned int Nsd, unsigned int Nne>
typename ShapeFunction<Nsd,Nne>::VectorNsd
ShapeFunction<Nsd,Nne>::xi_at_node(unsigned int node){ //function to return xi1, xi2, and xi3 for given node A
        if constexpr (Nsd == 2){
            double xi1, xi2;
            if constexpr (Nne == 4){
                switch(node){
                    case 0:
                        xi1 = -1.0;
                        xi2 = -1.0;
                        break;
                    case 1:
                        xi1 = 1.0;
                        xi2 = -1.0;
                        break;
                    case 2:
                        xi1 = 1.0;
                        xi2 = 1.0;
                        break;
                    case 3:
                        xi1 = -1.0;
                        xi2 = 1.0;
                        break;
                    default:
                        throw std::invalid_argument("xi_at_node mapping not implemented for this local node number");
                }
                return VectorNsd(xi1, xi2);
            }
        }
        else if constexpr (Nsd == 3){
            double xi1, xi2, xi3;
            if constexpr (Nne == 8){
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
                return VectorNsd(xi1, xi2, xi3);
            }
        }
        
};

template <unsigned int Nsd, unsigned int Nne>
double ShapeFunction<Nsd,Nne>::basis_function(unsigned int node, const VectorNsd& xi_vec){
    if constexpr (Nsd == 2){
        VectorNsd xi_node_vec = xi_at_node(node);
        double xi1_node = xi_node_vec(0);
        double xi2_node = xi_node_vec(1);
        double value = 0.25*(1 + xi_vec(0)*xi1_node)*(1 + xi_vec(1)*xi2_node);
        return value;
    }
    else if constexpr(Nsd == 3){
        VectorNsd xi_node_vec = xi_at_node(node);
        double xi1_node = xi_node_vec(0);
        double xi2_node = xi_node_vec(1);
        double xi3_node = xi_node_vec(2);
        double value = 0.125*(1 + xi_vec(0)*xi1_node)*(1 + xi_vec(1)*xi2_node)*(1 + xi_vec(2)*xi3_node);
        return value;
    }
  
    
};

template <unsigned int Nsd, unsigned int Nne>
typename ShapeFunction<Nsd,Nne>::VectorNsd
ShapeFunction<Nsd,Nne>::basis_gradient(unsigned int node, const VectorNsd& xi_vec){
    if constexpr (Nsd == 2){
        VectorNsd xi_node_vec = xi_at_node(node);
        double xi1_node = xi_node_vec(0);
        double xi2_node = xi_node_vec(1);

        double basis_gradient_xi1, basis_gradient_xi2;
        basis_gradient_xi1 = 0.25*xi1_node*(1 + xi_vec(1)*xi2_node);
        basis_gradient_xi2 = 0.25*xi2_node*(1 + xi_vec(0)*xi1_node);
        VectorNsd basis_gradient_vec(basis_gradient_xi1, basis_gradient_xi2);
        return basis_gradient_vec;
    }
    else if constexpr(Nsd == 3){
        VectorNsd xi_node_vec = xi_at_node(node);
        double xi1_node = xi_node_vec(0);
        double xi2_node = xi_node_vec(1);
        double xi3_node = xi_node_vec(2);

        double basis_gradient_xi1, basis_gradient_xi2, basis_gradient_xi3;
        basis_gradient_xi1 = 0.125*xi1_node*(1 + xi_vec(1)*xi2_node)*(1 + xi_vec(2)*xi3_node);
        basis_gradient_xi2 = 0.125*xi2_node*(1 + xi_vec(0)*xi1_node)*(1 + xi_vec(2)*xi3_node);
        basis_gradient_xi3 = 0.125*xi3_node*(1 + xi_vec(0)*xi1_node)*(1 + xi_vec(1)*xi2_node);
        VectorNsd basis_gradient_vec(basis_gradient_xi1, basis_gradient_xi2, basis_gradient_xi3);
        return basis_gradient_vec;
    }
}

