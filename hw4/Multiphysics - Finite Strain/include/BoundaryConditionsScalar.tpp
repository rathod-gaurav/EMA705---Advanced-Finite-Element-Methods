#pragma once //include this file only once during compilation
#include "BoundaryConditionsScalar.hpp"

template <unsigned int Nsd, unsigned int Nne>
BoundaryConditionsScalar<Nsd,Nne>::BoundaryConditionsScalar(
    const Mesh<Nsd,Nne>& mesh
):
    mesh_(mesh),
    totalDOFs_(mesh.Nnodes()) //calculate total degrees of freedom based on number of nodes and spatial dimensions
{}

template <unsigned int Nsd, unsigned int Nne>
void BoundaryConditionsScalar<Nsd,Nne>::addDirischlet(unsigned int node, int dof, double value){
    unsigned int globalNodeIndex = node + dof; //calculate the global node index based on the node number and degree of freedom
    dirischletVals_[globalNodeIndex] = value; //store the dirischlet value in the map with the global node index as the key
}

template <unsigned int Nsd, unsigned int Nne>
void BoundaryConditionsScalar<Nsd,Nne>::buildBCs(){
    isDirischlet_.assign(totalDOFs_, false); //resize the isDirichlet vector to the total number of degrees of freedom and initialize all values to false
    dirischletIndexes_.clear(); //clear the dirischletIndexes vector to prepare for building the boundary conditions
    unknownIndexes_.clear(); //clear the unknownIndexes vector to prepare for building the boundary conditions

    //mark dirischlet indexes
    for(const auto& [dof,val] : dirischletVals_){
        isDirischlet_[dof] = true; //mark the degrees of freedom that are subject to dirischlet boundary conditions as true in the isDirichlet vector
        dirischletIndexes_.push_back(dof); //add the degree of freedom to the dirischletIndexes vector
    }
    //everything else is free
    for(unsigned int i = 0 ; i < totalDOFs_ ; i++){
        if(!isDirischlet_[i]){
            unknownIndexes_.push_back(i); //add the degree of freedom to the unknownIndexes vector if it is not subject to dirischlet boundary conditions
        }
    }
}

template <unsigned int Nsd, unsigned int Nne>
void BoundaryConditionsScalar<Nsd,Nne>::applyToSolution(Eigen::VectorXd& u, double incrementFraction) const{
    for(const auto& [dof,val]: dirischletVals_){
        u[dof] = val * incrementFraction; //apply the dirischlet boundary conditions to the solution vector u based on the current increment fraction
    }
}

template <unsigned int Nsd, unsigned int Nne>
void BoundaryConditionsScalar<Nsd,Nne>::printSummary() const{
    std::cout << "Boundary Conditions Summary:" << std::endl;
    std::cout << "Total DOFs: " << totalDOFs_ << std::endl;
    std::cout << "Dirischlet DOFs: " << dirischletIndexes_.size() << std::endl;
    std::cout << "Unknown DOFs: " << unknownIndexes_.size() << std::endl;
}
