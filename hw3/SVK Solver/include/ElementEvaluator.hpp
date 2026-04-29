#pragma once //include this only once during compilation

#include <Eigen/Dense>
#include "Mesh.hpp"
#include "ShapeFunction.hpp"
#include "Quadrature.hpp"
#include "MaterialModel.hpp"

template <unsigned int Nsd, unsigned int Nne>
class ElementEvaluator{
    public:
        ElementEvaluator( //default constructor
            const Mesh<Nsd,Nne>& mesh,
            const MaterialModel<Nsd,Nne>& material,
            const QuadratureRule<Nsd,Nne>& quadRule
        );

        void computeElement(
            unsigned int e, //element index
            const Eigen::VectorXd& u_e, //element nodal displacements (Nne*3 x 1 vector)
            Eigen::MatrixXd& Klocal, //element stiffness matrix (Nne*3 x Nne*3 matrix)
            Eigen::VectorXd& Rlocal //element internal force vector (Nne*3 x 1 vector)
        ) const;
    
    private:
        using MatrixNsd = Eigen::Matrix<double, Nsd, Nsd>;
        using VectorNsd = Eigen::Vector<double, Nsd>;
        
        MatrixNsd computeJacobian(unsigned int e, const VectorNsd& xi_vec) const; //function to compute the Jacobian matrix for the element at given quadrature point (xi1, xi2, xi3)

        MatrixNsd computeGradU(const Eigen::VectorXd& u_e, const VectorNsd& xi_vec, const MatrixNsd& JacInv) const; //function to compute the gradient of the displacement field at the quadrature point using the basis function gradients and the nodal displacements

        const Mesh<Nsd,Nne>& mesh_; //reference to the mesh object
        const MaterialModel<Nsd,Nne>& material_; //reference to the material model object
        const QuadratureRule<Nsd,Nne>& quadRule_; //reference to the quadrature rule object
};

#include "ElementEvaluator.tpp" //include the implementation of the template class