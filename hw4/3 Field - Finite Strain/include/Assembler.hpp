#pragma once //include this only once during compilation

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "Mesh.hpp"
#include <ElementEvaluator.hpp>
#include <BoundaryConditions.hpp>
#include <unordered_set>

template <unsigned int Nsd, unsigned int Nne, unsigned int BfOrder>
class Assembler{
    public: 
        Assembler(const Mesh<Nsd,Nne>& mesh, const ElementEvaluator<Nsd,Nne,BfOrder>& elem_evaluator, const DiffusionEvaluator<Nsd,Nne,BfOrder>& diffusion_evaluator); //constructor to initialize the assembler with the mesh, element evaluator, and diffusion evaluator

        void assembleSystem(
            const Eigen::VectorXd& u, //global nodal displacement vector (Nnodes*Nsd x 1 vector)
            const Eigen::VectorXd& C, //global nodal chemical concentration vector (Nnodes x 1 vector)
            const Eigen::VectorXd& T, //global nodal temperature vector (Nnodes x 1 vector)
            Eigen::SparseMatrix<double>& Kglobal, //global stiffness matrix (Nnodes*Nsd x Nnodes*Nsd sparse matrix)
            Eigen::VectorXd& Rglobal //global internal force vector (Nnodes*Nsd x 1 vector)
        ) const;

        void assembleDiffusionSystem(
            Eigen::SparseMatrix<double>& MCCglobal, //global chemical mass matrix (Nnodes x Nnodes sparse matrix) 
            Eigen::SparseMatrix<double>& KCCglobal, //global chemical stiffness matrix (Nnodes x Nnodes sparse matrix)
            Eigen::SparseMatrix<double>& KCTglobal, //global coupling stiffness matrix between chemical concentration and temperature (Nnodes x Nnodes sparse matrix)

            Eigen::SparseMatrix<double>& MTTglobal, //global thermal mass matrix (Nnodes x Nnodes sparse matrix)
            Eigen::SparseMatrix<double>& KTTglobal, //global thermal stiffness matrix (Nnodes x Nnodes sparse matrix)
            Eigen::SparseMatrix<double>& KTCglobal, //global coupling stiffness matrix between temperature and chemical concentration (Nnodes x Nnodes sparse matrix)

            Eigen::SparseMatrix<double>& KuCglobal, //global coupling stiffness matrix between displacement and chemical concentration (Nnodes*Nsd x Nnodes sparse matrix)
            Eigen::SparseMatrix<double>& KuTglobal //global coupling stiffness matrix between displacement and temperature (Nnodes*Nsd x Nnodes sparse matrix)
        ) const;

        void partition(
            const Eigen::SparseMatrix<double>& Kglobal, //global stiffness matrix (Nnodes*Nsd x Nnodes*Nsd sparse matrix)
            Eigen::VectorXd& Rglobal, //global internal force vector (Nnodes*Nsd x 1 vector)
            const BoundaryConditions<Nsd,Nne>& bcs, //boundary conditions object containing the indexes of the dirischlet DOFs

            Eigen::SparseMatrix<double>& KUU, //extract the submatrix of K corresponding to the unknown degrees of freedom
            Eigen::SparseMatrix<double>& KUD, //extract the submatrix of K corresponding to the coupling between unknown and dirischlet degrees of freedom
            Eigen::VectorXd& RU //extract the subvector of R corresponding to the unknown degrees of freedom
        ) const;
    
    private:
        Eigen::SparseMatrix<double> extractSparseSubmatrix(
            const Eigen::SparseMatrix<double>& K,
            const std::vector<unsigned int>& rows,
            const std::vector<unsigned int>& cols) const; //function to extract a sparse submatrix from the global stiffness matrix given row and column indexes
    
        const Mesh<Nsd,Nne>& mesh_; //reference to the mesh object
        const ElementEvaluator<Nsd,Nne,BfOrder>& elem_evaluator_; //reference to the element evaluator object
        const DiffusionEvaluator<Nsd,Nne,BfOrder>& diffusion_evaluator_; //reference to the diffusion evaluator object


};

#include "Assembler.tpp" //include the implementation of the Assembler class