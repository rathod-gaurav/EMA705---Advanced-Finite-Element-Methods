#pragma once

#include <stdexcept> //for std::invalid_argument exception

template <unsigned int Nsd, unsigned int Nne, unsigned int BfOrder>
typename ShapeFunction<Nsd,Nne,BfOrder>::VectorNsd
ShapeFunction<Nsd,Nne,BfOrder>::xi_at_node(unsigned int node){ //function to return xi1, xi2, and xi3 for given node A
    if constexpr (BfOrder == 1){
        if constexpr (Nsd == 2){
            double xi1, xi2;
            if constexpr (Nne == 3){
                switch(node){
                    case 0:
                        xi1 = 0.0;
                        xi2 = 0.0;
                        break;
                    case 1:
                        xi1 = 1.0;
                        xi2 = 0.0;
                        break;
                    case 2:
                        xi1 = 0.0;
                        xi2 = 1.0;
                        break;
                    default:
                        throw std::invalid_argument("xi_at_node mapping not implemented for this local node number");
                }
                return VectorNsd(xi1, xi2);
            }
            else if constexpr (Nne == 4){
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
            else{
                throw std::invalid_argument("Nne not acceptable for given Nsd and BfOrder");
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
            else{
                throw std::invalid_argument("Nne not acceptable for given Nsd and BfOrder");
            }
        }
    }
    else if constexpr (BfOrder == 2){
        if constexpr (Nsd == 2){
            double xi1, xi2;
            if constexpr (Nne == 9){
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
                    case 4:
                        xi1 = 0.5;
                        xi2 = 0.0;
                        break;
                    case 5:
                        xi1 = 1.0;
                        xi2 = 0.5;
                        break;
                    case 6:
                        xi1 = 0.5;
                        xi2 = 1.0;
                        break;
                    case 7:
                        xi1 = 0.0;
                        xi2 = 0.5;
                        break;
                    case 8:
                        xi1 = 0.0;
                        xi2 = 0.0;
                        break;
                    default:
                        throw std::invalid_argument("xi_at_node mapping not implemented for this local node number");
                }
                return VectorNsd(xi1, xi2);
            }
            else{
                throw std::invalid_argument("Nne not acceptable for given Nsd and BfOrder");
            }
        }
        else if constexpr (Nsd == 3){
            double xi1, xi2, xi3;
            if constexpr (Nne == 27){
                throw std::invalid_argument("You are yet to implement xi_at_node for 27 node quadratic hex element!");
            }
            else{
                throw std::invalid_argument("Nne not acceptable for given Nsd and BfOrder");
            }
        }
    }     
};

template <unsigned int Nsd, unsigned int Nne, unsigned int BfOrder>
double ShapeFunction<Nsd,Nne,BfOrder>::basis_function(unsigned int node, const VectorNsd& xi_vec){
    if constexpr (BfOrder == 1){
        if constexpr (Nsd == 2){
            if constexpr (Nne == 3){
                double value = 0.0;
                switch(node){
                    case 0:
                        value = 1 - xi_vec(0) - xi_vec(1);
                        break;
                    case 1:
                        value = xi_vec(0);
                        break;
                    case 2:
                        value = xi_vec(1);
                        break;
                    default:
                        throw std::invalid_argument("cannot evaluate basis function value for out of bound local node number");
                }
                return value;
            }
            else if constexpr (Nne == 4){
                VectorNsd xi_node_vec = xi_at_node(node);
                double xi1_node = xi_node_vec(0);
                double xi2_node = xi_node_vec(1);
                double value = 0.25*(1 + xi_vec(0)*xi1_node)*(1 + xi_vec(1)*xi2_node);
                return value;
            }
            else{
                throw std::invalid_argument("Nne not acceptable for given Nsd and BfOrder");
            }
        }
        else if constexpr(Nsd == 3){
            if constexpr (Nne == 8){
                VectorNsd xi_node_vec = xi_at_node(node);
                double xi1_node = xi_node_vec(0);
                double xi2_node = xi_node_vec(1);
                double xi3_node = xi_node_vec(2);
                double value = 0.125*(1 + xi_vec(0)*xi1_node)*(1 + xi_vec(1)*xi2_node)*(1 + xi_vec(2)*xi3_node);
                return value;
            }
            else {
                throw std::invalid_argument("Nne not acceptable for given Nsd and BfOrder");
            }
        }
    }
    else if constexpr (BfOrder == 2){
        if constexpr (Nsd == 2){
            if constexpr (Nne == 9){
                //edit me
                VectorNsd xi_node_vec = xi_at_node(node);
                double xi1_node = xi_node_vec(0);
                double xi2_node = xi_node_vec(1);
                double value = 0.0;
                switch(node){
                    case 0:
                    case 1:
                    case 2:
                    case 3:
                        value = 0.25*xi_vec(0)*(xi_vec(0) + xi1_node)*xi_vec(1)*(xi_vec(1) + xi2_node);
                        break;
                    case 4:
                        value = 0.5*(1 - xi_vec(0)*xi_vec(0))*xi_vec(1)*(xi_vec(1) - 1);
                        break;
                    case 5:
                        value = 0.5*(1 + xi_vec(0))*xi_vec(0)*(1 - xi_vec(0)*xi_vec(0));
                        break;
                    case 6:
                        value = 0.5*(1 - xi_vec(0)*xi_vec(0))*xi_vec(1)*(xi_vec(1) + 1);
                        break;
                    case 7:
                        value = 0.5*(xi_vec(0) - 1)*xi_vec(0)*(1 - xi_vec(0)*xi_vec(0));
                        break;
                    case 8:
                        value = (1 - xi_vec(0)*xi_vec(0))*(1 - xi_vec(1)*xi_vec(1));
                        break;
                    default:
                        throw std::invalid_argument("cannot evaluate basis function value for out of bound local node number");
                }
                return value;
            }
            else{
                throw std::invalid_argument("Nne not acceptable for given Nsd and BfOrder");
            }
        }   
        else if constexpr(Nsd == 3){
            if constexpr (Nne == 27){
                throw std::invalid_argument("You are yet to implement basis_function for 27 node quadratic hex element!");
            }
            else{
                throw std::invalid_argument("Nne not acceptable for given Nsd and BfOrder");
            }
        }
    }    
};

template <unsigned int Nsd, unsigned int Nne, unsigned int BfOrder>
typename ShapeFunction<Nsd,Nne,BfOrder>::VectorNsd
ShapeFunction<Nsd,Nne,BfOrder>::basis_gradient(unsigned int node, const VectorNsd& xi_vec){
    if constexpr (BfOrder == 1){
        if constexpr (Nsd == 2){
            if constexpr (Nne == 3){
                double basis_gradient_xi1, basis_gradient_xi2;
                switch(node){
                    case 0:
                        basis_gradient_xi1 = -1.0;
                        basis_gradient_xi2 = -1.0;
                        break;
                    case 1:
                        basis_gradient_xi1 = 1.0;
                        basis_gradient_xi2 = 0.0;
                        break;
                    case 2:
                        basis_gradient_xi1 = 0.0;
                        basis_gradient_xi2 = 1.0;
                        break;
                    default:
                        throw std::invalid_argument("cannot evaluate basis function gradient value for out of bound local node number");
                }
                VectorNsd basis_gradient_vec(basis_gradient_xi1, basis_gradient_xi2);
                return basis_gradient_vec;
            }
            else if constexpr (Nne == 4){
                VectorNsd xi_node_vec = xi_at_node(node);
                double xi1_node = xi_node_vec(0);
                double xi2_node = xi_node_vec(1);

                double basis_gradient_xi1, basis_gradient_xi2;
                basis_gradient_xi1 = 0.25*xi1_node*(1 + xi_vec(1)*xi2_node);
                basis_gradient_xi2 = 0.25*xi2_node*(1 + xi_vec(0)*xi1_node);
                VectorNsd basis_gradient_vec(basis_gradient_xi1, basis_gradient_xi2);
                return basis_gradient_vec;
            }
            else{
                throw std::invalid_argument("Nne not acceptable for given Nsd and BfOrder");
            }
        }
        else if constexpr(Nsd == 3){
            if constexpr (Nne == 8){
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
            else{
                throw std::invalid_argument("Nne not acceptable for given Nsd and BfOrder");
            }
        }
    }
    else if constexpr (BfOrder == 2){
        if constexpr (Nsd == 2){
            if constexpr (Nne == 9){
                VectorNsd xi_node_vec = xi_at_node(node);
                double xi1_node = xi_node_vec(0);
                double xi2_node = xi_node_vec(1);
                double basis_gradient_xi1, basis_gradient_xi2;
                switch(node){
                    case 0:
                    case 1:
                    case 2:
                    case 3:
                        basis_gradient_xi1 = 0.25*(2*xi_vec(0) + xi1_node)*xi_vec(1)*(xi_vec(1) + xi2_node);
                        basis_gradient_xi2 = 0.25*xi_vec(0)*(xi_vec(0) + xi1_node)*(2*xi_vec(1) + xi2_node);
                        break;
                    case 4:
                        basis_gradient_xi1 = -xi_vec(0)*xi_vec(1)*(xi_vec(1) - 1);
                        basis_gradient_xi2 = 0.5*(1 - xi_vec(0)*xi_vec(0))*(2*xi_vec(1) - 1);
                        break;
                    case 5:
                        basis_gradient_xi1 = 0.5*(2*xi_vec(0) + 1)*(1 - xi_vec(1)*xi_vec(1));
                        basis_gradient_xi2 = -xi_vec(0)*(xi_vec(0) + 1)*xi_vec(1);
                        break;
                    case 6:
                        basis_gradient_xi1 = -xi_vec(0)*xi_vec(1)*(xi_vec(1) + 1);
                        basis_gradient_xi2 = 0.5*(1 - xi_vec(0)*xi_vec(0))*(2*xi_vec(1) + 1);
                        break;
                    case 7:
                        basis_gradient_xi1 = 0.5*(2*xi_vec(0) - 1)*(1 - xi_vec(1)*xi_vec(1));
                        basis_gradient_xi2 = -xi_vec(0)*(xi_vec(0) - 1)*xi_vec(1);
                        break;
                    case 8:
                        basis_gradient_xi1 = -2*xi_vec(0)*(1 - xi_vec(1)*xi_vec(1));
                        basis_gradient_xi2 = -2*xi_vec(1)*(1 - xi_vec(0)*xi_vec(0));
                        break;
                    default:
                        throw std::invalid_argument("cannot evaluate basis function gradient value for out of bound local node number");
                }
                VectorNsd basis_gradient_vec(basis_gradient_xi1, basis_gradient_xi2);
                return basis_gradient_vec;
            }
            else{
                throw std::invalid_argument("Nne not acceptable for given Nsd and BfOrder");
            }
        }   
        else if constexpr(Nsd == 3){
            if constexpr (Nne == 27){
                throw std::invalid_argument("You are yet to implement basis_gradient for 27 node quadratic hex element!");
            }
            else{
                throw std::invalid_argument("Nne not acceptable for given Nsd and BfOrder");
            }
        }
    }
    
}

