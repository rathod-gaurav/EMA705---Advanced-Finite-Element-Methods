#pragma once //include this file only once during compilation

#include <tuple>

template <unsigned int Nsd, unsigned int Nne, unsigned int BfOrder>
class ShapeFunction{
    public:
        using VectorNsd = Eigen::Vector<double, Nsd>;
        //when "static" is used - it means that the function belongs to the class itself, not an instance. This means you can call the function without creating an object of the class. For example, you can call ShapeFunction::xi_at_node(node) directly without needing to instantiate a ShapeFunction object.
        static VectorNsd xi_at_node(unsigned int node); //function to return xi1, xi2, and xi3 for given node A

        static double basis_function(unsigned int node, const VectorNsd& xi_vec); //function to calculate basis function value for given node A and xi1, xi2, xi3

        static VectorNsd basis_gradient(unsigned int node, const VectorNsd& xi_vec); //function to calculate basis function gradient with respect to xi1, xi2, and xi3 for given node A and xi1, xi2, xi3
};

#include "ShapeFunction.tpp"