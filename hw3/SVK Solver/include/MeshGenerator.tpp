#pragma once //include this file only once during compilation

// 2D //
template <unsigned int Nne, unsigned int BfOrder>
MeshGenerator<2,Nne,BfOrder>::MeshGenerator( //assign the function parameters to the class member variables using an initializer list
    //domain parameters
    double x1_ll, double x1_ul,
    double x2_ll, double x2_ul,
    //mesh parameters
    unsigned int Nel_x1, unsigned int Nel_x2
):
    x1_ll_(x1_ll), x1_ul_(x1_ul),
    x2_ll_(x2_ll), x2_ul_(x2_ul),
    Nel_x1_(Nel_x1), Nel_x2_(Nel_x2)
{}

// 2D //
template <unsigned int Nne, unsigned int BfOrder>
Mesh<2,Nne> MeshGenerator<2,Nne,BfOrder>::buildMesh() const{
    Mesh<Nsd,Nne> mesh; //create an empty mesh object

    if constexpr (BfOrder == 1){
        unsigned int Nnodes_x1 = Nel_x1_ + 1; //number of nodes in x1 direction
        unsigned int Nnodes_x2 = Nel_x2_ + 1; //number of nodes in x2 direction
        
        unsigned int Nt = Nnodes_x1 * Nnodes_x2; //total number of nodes

        double dx1 = (x1_ul_ - x1_ll_) / Nel_x1_; //spacing between nodes in x1 direction
        double dx2 = (x2_ul_ - x2_ll_) / Nel_x2_; //spacing between nodes in x2 direction

        //Build the nodes list of the mesh
        mesh.nodes.reserve(Nt);
        
        for(unsigned int j = 0 ; j < Nnodes_x2 ; j++){
            for(unsigned int i = 0 ; i < Nnodes_x1 ; i++){
                Node<Nsd> n;
                n.x1 = x1_ll_ + i*dx1;
                n.x2 = x2_ll_ + j*dx2;
                mesh.nodes.push_back(n);
            }
        }

        //variables required for element connectivity
        unsigned int Nel_t = Nel_x1_ * Nel_x2_; //total number of elements
        mesh.elements.reserve(Nel_t);

        if constexpr (Nne == 3){//linear triangle elements
            for(unsigned int j = 0 ; j < Nel_x2_ ; j++){
                for(unsigned int i = 0 ; i < Nel_x1_ ; i++){
                    Element<Nne> elem1;
                    Element<Nne> elem2;
                    
                    int n0 = i + j*Nnodes_x1;
                    int n1 = n0 + 1;
                    int n2 = Nnodes_x1 + i + j*Nnodes_x1 + 1;
                    int n3 = n2 - 1;

                    elem1.node[0] = n0;
                    elem1.node[1] = n1;
                    elem1.node[2] = n3;
                    
                    elem2.node[0] = n1;
                    elem2.node[1] = n2;
                    elem2.node[2] = n3;

                    mesh.elements.push_back(elem1);
                    mesh.elements.push_back(elem2);
                }
            }
        }
        else if constexpr (Nne == 4){//linear quadrilateral elements
            
            for(unsigned int j = 0 ; j < Nel_x2_ ; j++){
                for(unsigned int i = 0 ; i < Nel_x1_ ; i++){
                    Element<Nne> elem;
                    
                    int n0 = i + j*Nnodes_x1;
                    int n1 = n0 + 1;
                    int n2 = Nnodes_x1 + i + j*Nnodes_x1 + 1;
                    int n3 = n2 - 1;

                    elem.node[0] = n0;
                    elem.node[1] = n1;
                    elem.node[2] = n2;
                    elem.node[3] = n3;

                    mesh.elements.push_back(elem);
                }
            }
        }    
        else{
            throw std::invalid_argument("Nne not acceptable for given Nsd and BfOrder");
        }
        //Build the elements list of the mesh
    }
    else if constexpr (BfOrder == 2){
        unsigned int Nnodes_x1 = 2*Nel_x1_ + 1; //number of nodes in x1 direction
        unsigned int Nnodes_x2 = 2*Nel_x2_ + 1; //number of nodes in x2 direction
        
        unsigned int Nt = Nnodes_x1 * Nnodes_x2; //total number of nodes

        double dx1 = (x1_ul_ - x1_ll_) / (2*Nel_x1_); //spacing between nodes in x1 direction
        double dx2 = (x2_ul_ - x2_ll_) / (2*Nel_x2_); //spacing between nodes in x2 direction

        //Build the nodes list of the mesh
        mesh.nodes.reserve(Nt);
        
        for(unsigned int j = 0 ; j < Nnodes_x2 ; j++){
            for(unsigned int i = 0 ; i < Nnodes_x1 ; i++){
                Node<Nsd> n;
                n.x1 = x1_ll_ + i*dx1;
                n.x2 = x2_ll_ + j*dx2;
                mesh.nodes.push_back(n);
            }
        }

        //variables required for element connectivity
        unsigned int Nel_t = Nel_x1_ * Nel_x2_; //total number of elements
        mesh.elements.reserve(Nel_t);

        if constexpr (Nne == 9){//linear quadrilateral elements
            
            for(unsigned int j = 0 ; j < Nel_x2_ ; j++){
                for(unsigned int i = 0 ; i < Nel_x1_ ; i++){
                    Element<Nne> elem;

                    int n0 = 2*(i + j*Nnodes_x1);
                    int n1 = n0 + 2;
                    int n2 = 2*(Nnodes_x1 + i + j*Nnodes_x1 + 1);
                    int n3 = n2 - 2;
                    int n4 = n1 - 1;
                    int n5 = n1 + Nnodes_x1;
                    int n6 = n2 - 1;
                    int n7 = n5 - 2;
                    int n8 = n5 - 1;

                    elem.node[0] = n0;
                    elem.node[1] = n1;
                    elem.node[2] = n2;
                    elem.node[3] = n3;
                    elem.node[4] = n4;
                    elem.node[5] = n5;
                    elem.node[6] = n6;
                    elem.node[7] = n7;
                    elem.node[8] = n8;

                    mesh.elements.push_back(elem);
                }
            }
        }    
        else{
            throw std::invalid_argument("Nne not acceptable for given Nsd and BfOrder");
        }
    }
    else{
        throw std::invalid_argument("Invalid basis function order given, only supported BfOrder are 1 and 2");
    }

    
    

    return mesh; //return the built mesh object

}

// 3D //
template <unsigned int Nne, unsigned int BfOrder>
MeshGenerator<3,Nne,BfOrder>::MeshGenerator( //assign the function parameters to the class member variables using an initializer list
    //domain parameters
    double x1_ll, double x1_ul,
    double x2_ll, double x2_ul,
    double x3_ll, double x3_ul,
    //mesh parameters
    unsigned int Nel_x1, unsigned int Nel_x2, unsigned int Nel_x3
):
    x1_ll_(x1_ll), x1_ul_(x1_ul),
    x2_ll_(x2_ll), x2_ul_(x2_ul),
    x3_ll_(x3_ll), x3_ul_(x3_ul),
    Nel_x1_(Nel_x1), Nel_x2_(Nel_x2), Nel_x3_(Nel_x3)
{}

// 3D //
template <unsigned int Nne, unsigned int BfOrder>
Mesh<3,Nne> MeshGenerator<3,Nne,BfOrder>::buildMesh() const{
    Mesh<Nsd,Nne> mesh; //create an empty mesh object
    if constexpr (BfOrder == 1){
        unsigned int Nnodes_x1 = Nel_x1_ + 1; //number of nodes in x1 direction
        unsigned int Nnodes_x2 = Nel_x2_ + 1; //number of nodes in x2 direction
        unsigned int Nnodes_x3 = Nel_x3_ + 1; //number of nodes in x3 direction
        unsigned int Nt = Nnodes_x1 * Nnodes_x2 * Nnodes_x3; //total number of nodes

        double dx1 = (x1_ul_ - x1_ll_) / Nel_x1_; //spacing between nodes in x1 direction
        double dx2 = (x2_ul_ - x2_ll_) / Nel_x2_; //spacing between nodes in x2 direction
        double dx3 = (x3_ul_ - x3_ll_) / Nel_x3_; //spacing between nodes in x3 direction

        //Build the nodes list of the mesh
        mesh.nodes.reserve(Nt);
        for(unsigned int k = 0 ; k < Nnodes_x3 ; k++){
            for(unsigned int j = 0 ; j < Nnodes_x2 ; j++){
                for(unsigned int i = 0 ; i < Nnodes_x1 ; i++){
                    Node<Nsd> n;
                    n.x1 = x1_ll_ + i*dx1;
                    n.x2 = x2_ll_ + j*dx2;
                    n.x3 = x3_ll_ + k*dx3;
                    mesh.nodes.push_back(n);
                }
            }
        }

        //variables required for element connectivity
        unsigned int Nel_t = Nel_x1_ * Nel_x2_ * Nel_x3_; //total number of elements

        //Build the elements list of the mesh
        mesh.elements.reserve(Nel_t);

        if constexpr (Nne == 8){//hexahedral elements
            for(unsigned int k = 0 ; k < Nel_x3_ ; k++){
                for(unsigned int j = 0 ; j < Nel_x2_ ; j++){
                    for(unsigned int i = 0 ; i < Nel_x1_ ; i++){
                        Element<Nne> elem;
                        unsigned int base = i 
                            + j * Nnodes_x1 
                            + k * (Nnodes_x1 * Nnodes_x2);

                        unsigned int n0 = base;
                        unsigned int n1 = base + 1;
                        unsigned int n3 = base + Nnodes_x1;
                        unsigned int n2 = n3 + 1;

                        unsigned int n4 = base + Nnodes_x1 * Nnodes_x2;
                        unsigned int n5 = n4 + 1;
                        unsigned int n7 = n4 + Nnodes_x1;
                        unsigned int n6 = n7 + 1;

                        elem.node[0] = n0;
                        elem.node[1] = n1;
                        elem.node[2] = n2;
                        elem.node[3] = n3;
                        elem.node[4] = n4;
                        elem.node[5] = n5;
                        elem.node[6] = n6;
                        elem.node[7] = n7;

                        mesh.elements.push_back(elem);
                    }
                }
            }
        }
        else{
            throw std::runtime_error("Element triangulation not implemented for given Nne");
        }
    }
    else if constexpr (BfOrder == 2){
        throw std::invalid_argument("You are yet to implement buildMesh for higher order 3D hexahedral elements!");
    }
    else{
        throw std::invalid_argument("Invalid basis function order given, only supported BfOrder are 1 and 2");
    }

    

    

    return mesh; //return the built mesh object

}