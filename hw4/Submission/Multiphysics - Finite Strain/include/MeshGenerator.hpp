#pragma once //include this file only once during compilation
#include "Mesh.hpp"

template <unsigned int Nsd, unsigned int Nne, unsigned int BfOrder>
class MeshGenerator;

//MeshGenerator for 2D
template <unsigned int Nne, unsigned int BfOrder>
class MeshGenerator<2,Nne,BfOrder>{
    public:
        static constexpr unsigned int Nsd = 2; //2D problem

        MeshGenerator(
            //domain parameters
            double x1_ll, double x1_ul,
            double x2_ll, double x2_ul,
            //mesh parameters
            unsigned int Nel_x1, unsigned int Nel_x2
        );

        Mesh<Nsd,Nne> buildMesh() const; //function to build the mesh and return a Mesh object | this is a const member function and does not modify the state of the MeshGenerator object

    private:
        //domain parameters
        double x1_ll_, x1_ul_;
        double x2_ll_, x2_ul_;
        //mesh parameters
        unsigned int Nel_x1_, Nel_x2_;
};

//MeshGenerator for 3D
template <unsigned int Nne, unsigned int BfOrder>
class MeshGenerator<3,Nne,BfOrder>{
    public:
        static constexpr unsigned int Nsd = 3; //3D problem

        MeshGenerator(
            //domain parameters
            double x1_ll, double x1_ul,
            double x2_ll, double x2_ul,
            double x3_ll, double x3_ul,
            //mesh parameters
            unsigned int Nel_x1, unsigned int Nel_x2, unsigned int Nel_x3
        );

        Mesh<Nsd,Nne> buildMesh() const; //function to build the mesh and return a Mesh object | this is a const member function and does not modify the state of the MeshGenerator object

    private:
        //domain parameters
        double x1_ll_, x1_ul_;
        double x2_ll_, x2_ul_;
        double x3_ll_, x3_ul_;
        //mesh parameters
        unsigned int Nel_x1_, Nel_x2_, Nel_x3_;
};


#include "MeshGenerator.tpp" //include the implementation of the member functions of the MeshGenerator class