#pragma once //include this only once during compilation

#include <Eigen/Dense>

//Any class that inherits from this MaterialModel class must provide these methods
template<unsigned int Nsd, unsigned int Nne>
class MaterialModel{
    public:
        using MatrixNsd = Eigen::Matrix<double, Nsd, Nsd>;
        virtual void compute(
            const MatrixNsd& F, //deformation gradient
            MatrixNsd& S, //second Piola-Kirchhoff stress tensor
            MatrixNsd& P, //first Piola-Kirchhoff stress tensor
            Eigen::MatrixXd& C_mat //material tangent stiffness matrix (4th order elasticity tensor in Voigt notation)
        ) const = 0; // = 0 means this is a pure virtual function. which meahc that MaterialModel itself cannot be instantiated, but any derived class from this must inplement this compute method

        virtual ~MaterialModel() = default; //virtual destructor to ensure proper cleanup of derived classes
        //this is required wherever polymorphism is used, i.e., when we have a base class pointer pointing to a derived class object. This ensures that when we delete the base class pointer, the destructor of the derived class is also called, preventing memory leaks and ensuring proper resource management.
};