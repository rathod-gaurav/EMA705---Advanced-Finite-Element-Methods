// Linear Elliptic PDE with vector variable - 3D elasto statics
// no body forces, no neumann conditions, only dirischlet boundary conditions at top and bottom faces of the domain

#include <iostream>
using namespace std;
#include <vector>
#include <fstream>
#include <Eigen/Dense>
#include <map>

struct Node{
    float x1, x2;
};

template <unsigned int Nne>
struct Element{
    int node[Nne];
};

template <unsigned int Nne, unsigned int BfOrder>
std::tuple<float,float> xi_at_node(unsigned int node){ //function to return xi1 and xi2 for given node A
    float xi1, xi2;
    
    if constexpr (BfOrder == 1){
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
        }
        else{
            throw std::invalid_argument("Nne not acceptable for given Nsd and BfOrder");
        }
        
    }
    else if constexpr (BfOrder == 2){
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
        }
        else{
            throw std::invalid_argument("Nne not acceptable for given Nsd and BfOrder");
        }
    }

    return {xi1, xi2};
};

template <unsigned int Nne, unsigned int BfOrder>
float basis_function(unsigned int node, float xi1, float xi2){
    Eigen::Vector2f xi_vec(xi1, xi2);
    auto [xi1_node,xi2_node] = xi_at_node<Nne,BfOrder>(node);
    float value = 0.0f;
    if constexpr (BfOrder == 1){
        if constexpr (Nne == 3){
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
            value = 0.25*(1 + xi_vec(0)*xi1_node)*(1 + xi_vec(1)*xi2_node);
        }
        else{
            throw std::invalid_argument("Nne not acceptable for given Nsd and BfOrder");
        }
    }
    else if constexpr (BfOrder == 2){
        if constexpr (Nne == 9){            
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
    return value;
};

template <unsigned int Nne, unsigned int BfOrder>
std::tuple<float,float> basis_gradient(unsigned int node, float xi1, float xi2){
    Eigen::Vector2f xi_vec(xi1, xi2);
    auto [xi1_node,xi2_node] = xi_at_node<Nne,BfOrder>(node);
    float basis_gradient_xi1, basis_gradient_xi2;
    
    if constexpr (BfOrder == 1){
        if constexpr (Nne == 3){
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
            
            
        }
        else if constexpr (Nne == 4){                         
            basis_gradient_xi1 = 0.25*xi1_node*(1 + xi_vec(1)*xi2_node);
            basis_gradient_xi2 = 0.25*xi2_node*(1 + xi_vec(0)*xi1_node);
        }
        else{
            throw std::invalid_argument("Nne not acceptable for given Nsd and BfOrder");
        }
    }
    else if constexpr (BfOrder == 2){
        if constexpr (Nne == 9){                   
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
        }
        else{
            throw std::invalid_argument("Nne not acceptable for given Nsd and BfOrder");
        }
    }

    return {basis_gradient_xi1, basis_gradient_xi2};
}

template <unsigned int Nne>
struct QuadratureRule {
    std::vector<float> points;
    std::vector<float> points2;
    std::vector<float> weights;
};

template <unsigned int Nne>
QuadratureRule<Nne> gauss_legendre(unsigned int n) {
    QuadratureRule<Nne> rule;

    if constexpr (Nne == 3){
        switch(n){
            case 1:
                rule.points = { 0.333 };
                rule.points2 = { 0.333 };
                rule.weights = { 1.0 };
                break;
            case 3:
                rule.points = { 0.167, 0.667, 0.167};
                rule.points2 = { 0.167, 0.167, 0.667 };
                rule.weights = { 0.3333, 0.3333, 0.3333 };
                break;
            case 4:
                rule.points = { 0.333, 0.600, 0.200, 0.200 };
                rule.points2 = { 0.333, 0.200, 0.600, 0.200 };
                rule.weights = { 0.5625, 0.5208, 0.5208, 0.5208 };
                break;
            default:
                throw std::invalid_argument("Gauss-Legendre quadrature not implemented for this n");
        }
    }
    else if constexpr (Nne == 4 || Nne == 9){
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
    }

    return rule;
}


Eigen::MatrixXf extractSubmatrix(const Eigen::MatrixXf& OriginalMatrix , const vector<int> rows , const vector<int> cols){//extract submatrix from original matrix given row and column indexes
    Eigen::MatrixXf subMatrix(rows.size(), cols.size());

    for(int i = 0 ; i < rows.size() ; i++){
        for(int j = 0 ; j < cols.size() ; j++){
            subMatrix(i,j) = OriginalMatrix(rows[i],cols[j]);
        }
    }
    return subMatrix;
}

template <unsigned int Nne>
void writeVTK(
    const std::string& filename,
    int Nt,
    int Nel_t,
    const std::vector<Node>& nodes,
    const std::vector<Element<Nne>>& elements,
    const Eigen::VectorXf& D_full
)
{
    std::ofstream vtk(filename);

    if (!vtk.is_open()) {
        std::cerr << "Error opening VTK file.\n";
        return;
    }

    // Determine VTK Cell Type based on Nne
    int vtk_type = 0;
    if constexpr (Nne == 3)      vtk_type = 5;  // VTK_TRIANGLE
    else if constexpr (Nne == 4) vtk_type = 9;  // VTK_QUAD
    else if constexpr (Nne == 9) vtk_type = 28; // VTK_BIQUADRATIC_QUADRATIC_QUAD

    vtk << "# vtk DataFile Version 3.0\n";
    vtk << "2D Elasticity Solution\n";
    vtk << "ASCII\n";
    vtk << "DATASET UNSTRUCTURED_GRID\n\n";

    // -------------------------
    // POINTS (VTK needs 3D, we provide Z=0)
    // -------------------------
    vtk << "POINTS " << Nt << " float\n";
    for (int i = 0; i < Nt; i++) {
        vtk << nodes[i].x1 << " " << nodes[i].x2 << " 0.0\n";
    }

    // -------------------------
    // CELLS
    // -------------------------
    // Total entries = Nel_t * (nodes_per_elem + 1 header)
    vtk << "\nCELLS " << Nel_t << " " << Nel_t * (Nne + 1) << "\n";
    for (int e = 0; e < Nel_t; e++) {
        vtk << Nne << " ";
        for (int A = 0; A < Nne; A++) {
            vtk << elements[e].node[A] << " ";
        }
        vtk << "\n";
    }

    // -------------------------
    // CELL TYPES
    // -------------------------
    vtk << "\nCELL_TYPES " << Nel_t << "\n";
    for (int e = 0; e < Nel_t; e++) {
        vtk << vtk_type << "\n";
    }

    // -------------------------
    // DISPLACEMENT FIELD (POINT_DATA)
    // -------------------------
    vtk << "\nPOINT_DATA " << Nt << "\n";
    vtk << "VECTORS displacement float\n";

    for (int i = 0; i < Nt; i++) {
        // Since Nsd=2, displacements are stored as [u1, v1, u2, v2, ...]
        vtk << D_full(2 * i)     << " "
            << D_full(2 * i + 1) << " "
            << "0.0\n"; // No Z displacement
    }

    vtk.close();
    std::cout << "VTK file written to: " << filename << std::endl;
}

int main(){
    unsigned int Nsd = 2; //number of spatial dimensions - 3D problem
    constexpr unsigned int BfOrder = 2; //Order of basis functions 
    constexpr unsigned int Nne = 9; //number of nodes in an element - 8 for hexahedral element
    unsigned int quadRule = 2; //quadrature rule for numerical integration

    //problem variables
    float E = 1e3;
    float nu = 0.3;
    float mu = (E*nu)/((1 + nu)*(1 - 2*nu));
    float lambda = E/(2*(1 + nu));

    //domain
    float x1_ll = 0.0;
    float x1_ul = 0.01;
    float x2_ll = 0.0;
    float x2_ul = 0.01;

    //Mesh
    unsigned int Nel_x1 = 4; //number of elements in x1 direction
    unsigned int Nel_x2 = 4; //number of elements in x2 direction
    unsigned int Nt, Nel_t;

    // Mesh Generator
    std::vector<Node> nodes;
    std::vector<Element<Nne>> elements;
    if constexpr (BfOrder == 1){
        unsigned int Nnodes_x1 = Nel_x1 + 1; //number of nodes in x1 direction
        unsigned int Nnodes_x2 = Nel_x2 + 1; //number of nodes in x2 direction
        
        Nt = Nnodes_x1 * Nnodes_x2; //total number of nodes

        double dx1 = (x1_ul - x1_ll) / Nel_x1; //spacing between nodes in x1 direction
        double dx2 = (x2_ul - x2_ll) / Nel_x2; //spacing between nodes in x2 direction

        //Build the nodes list of the mesh
        nodes.reserve(Nt);
        
        for(unsigned int j = 0 ; j < Nnodes_x2 ; j++){
            for(unsigned int i = 0 ; i < Nnodes_x1 ; i++){
                Node n;
                n.x1 = x1_ll + i*dx1;
                n.x2 = x2_ll + j*dx2;
                nodes.push_back(n);
            }
        }

        if constexpr (Nne == 3){//linear triangle elements
            //variables required for element connectivity
            Nel_t = 2*Nel_x1 * Nel_x2; //total number of elements
            elements.reserve(Nel_t);
            for(unsigned int j = 0 ; j < Nel_x2 ; j++){
                for(unsigned int i = 0 ; i < Nel_x1 ; i++){
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

                    elements.push_back(elem1);
                    elements.push_back(elem2);
                }
            }
        }
        else if constexpr (Nne == 4){//linear quadrilateral elements
            //variables required for element connectivity
            Nel_t = Nel_x1 * Nel_x2; //total number of elements
            elements.reserve(Nel_t);
            for(unsigned int j = 0 ; j < Nel_x2 ; j++){
                for(unsigned int i = 0 ; i < Nel_x1 ; i++){
                    Element<Nne> elem;
                    
                    int n0 = i + j*Nnodes_x1;
                    int n1 = n0 + 1;
                    int n2 = Nnodes_x1 + i + j*Nnodes_x1 + 1;
                    int n3 = n2 - 1;

                    elem.node[0] = n0;
                    elem.node[1] = n1;
                    elem.node[2] = n2;
                    elem.node[3] = n3;

                    elements.push_back(elem);
                }
            }
        }    
        else{
            throw std::invalid_argument("Nne not acceptable for given Nsd and BfOrder");
        }
        //Build the elements list of the mesh
    }
    else if constexpr (BfOrder == 2){
        unsigned int Nnodes_x1 = 2*Nel_x1 + 1; //number of nodes in x1 direction
        unsigned int Nnodes_x2 = 2*Nel_x2 + 1; //number of nodes in x2 direction
        
        Nt = Nnodes_x1 * Nnodes_x2; //total number of nodes

        double dx1 = (x1_ul - x1_ll) / (2*Nel_x1); //spacing between nodes in x1 direction
        double dx2 = (x2_ul - x2_ll) / (2*Nel_x2); //spacing between nodes in x2 direction

        //Build the nodes list of the mesh
        nodes.reserve(Nt);
        
        for(unsigned int j = 0 ; j < Nnodes_x2 ; j++){
            for(unsigned int i = 0 ; i < Nnodes_x1 ; i++){
                Node n;
                n.x1 = x1_ll + i*dx1;
                n.x2 = x2_ll + j*dx2;
                nodes.push_back(n);
            }
        }

        //variables required for element connectivity
        Nel_t = Nel_x1 * Nel_x2; //total number of elements
        elements.reserve(Nel_t);

        if constexpr (Nne == 9){//linear quadrilateral elements
            
            for(unsigned int j = 0 ; j < Nel_x2 ; j++){
                for(unsigned int i = 0 ; i < Nel_x1 ; i++){
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

                    elements.push_back(elem);
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


    //store the mesh into points and hexa files
    std::ofstream points_file("mesh/points.txt");
    for(auto& node : nodes){
        points_file << node.x1 << " " << node.x2 << "\n";
    }

    std::ofstream elems_file("mesh/elems.txt");
    for(auto& elem : elements){ 
        for(unsigned int i = 0 ; i < Nne ; i++){
            elems_file << elem.node[i] << " ";
        }
        elems_file << "\n";
    }

    // Isoparametric Mapping - Calculate 3D jacobian
    auto calculate_Jacobian_2D = [Nsd, elements, nodes](int e, float xi1, float xi2){//function to calculate jacobian
        Eigen::Matrix2f J = Eigen::Matrix2f::Zero();
        
        for(int A = 0 ; A < Nne ; A++){
            auto [basis_gradient_xi1, basis_gradient_xi2] = basis_gradient<Nne,BfOrder>(A, xi1, xi2);
            int Aglobal = elements[e].node[A];
            J(0,0) += basis_gradient_xi1*nodes[Aglobal].x1; //dx1/dxi1
            J(0,1) += basis_gradient_xi2*nodes[Aglobal].x1; //dx1/dxi2
            
            J(1,0) += basis_gradient_xi1*nodes[Aglobal].x2; //dx2/dxi1
            J(1,1) += basis_gradient_xi2*nodes[Aglobal].x2; //dx2/dxi2            
        }
        return J;
    };


    Eigen::MatrixXf Kglobal = Eigen::MatrixXf::Zero(Nt*Nsd,Nt*Nsd);
    Eigen::VectorXf Fglobal = Eigen::VectorXf::Zero(Nt*Nsd);

    for(int e = 0; e < Nel_t ; e++){
        Eigen::MatrixXf Klocal = Eigen::MatrixXf::Zero(Nne*Nsd,Nne*Nsd);
        for(int A = 0 ; A < Nne ; A++){
            for(int B = 0 ; B < Nne ; B++){
                if constexpr (Nne == 3){//specialization for triangle elements
                    auto q = gauss_legendre<Nne>(quadRule);
                    std::vector<float> points_x1(quadRule), points_x2(quadRule), weights(quadRule);
                    points_x1 = q.points;
                    points_x2 = q.points2;
                    weights = q.weights;
                    Eigen::VectorXf quad_points_x1 = Eigen::Map<Eigen::VectorXf>(points_x1.data(), points_x1.size());
                    Eigen::VectorXf quad_points_x2 = Eigen::Map<Eigen::VectorXf>(points_x2.data(), points_x2.size());
                    Eigen::VectorXf quad_weights = Eigen::Map<Eigen::VectorXf>(weights.data(), weights.size());

                    for(int I = 0 ; I < quadRule ; I++){
                        float xi1 = quad_points_x1(I);
                        float xi2 = quad_points_x2(I);
                        float weights = quad_weights(I)*quad_weights(I);

                        Eigen::Matrix2f Jac = calculate_Jacobian_2D(e, xi1, xi2);
                        if(Jac.determinant() < 0){ 
                            throw std::runtime_error("Negative Jacobian detected");
                            break;
                        }
                        Eigen::MatrixXf Jac_inv = Jac.inverse();

                        auto [bfgradientA_xi1 , bfgradientA_xi2] = basis_gradient<Nne,BfOrder>(A, xi1, xi2);
                        auto [bfgradientB_xi1 , bfgradientB_xi2] = basis_gradient<Nne,BfOrder>(B, xi1, xi2);

                        Eigen::Vector2f bfgradientA(bfgradientA_xi1 , bfgradientA_xi2);
                        Eigen::Vector2f bfgradientB(bfgradientB_xi1 , bfgradientB_xi2);

                        Eigen::VectorXf term1 = Jac_inv.transpose()*bfgradientA;
                        Eigen::VectorXf term2 = Jac_inv.transpose()*bfgradientB;
                        Eigen::Matrix2f II = Eigen::Matrix2f::Identity();

                        Eigen::Matrix2f Klocal_mu = 2*mu*(term1.dot(term2))*II*Jac.determinant()*weights;
                        Eigen::Matrix2f Klocal_lambda = lambda*(term1*term2.transpose())*Jac.determinant()*weights;

                        Eigen::MatrixXf Kblock = Klocal_mu + Klocal_lambda;

                        Klocal.block<2,2>(2*A, 2*B) += Kblock;
                    }
                }
                else if constexpr (Nne == 4 || Nne == 9){
                    auto q = gauss_legendre<Nne>(quadRule);
                    std::vector<float> points(quadRule), weights(quadRule);

                    points = q.points;
                    weights = q.weights;
                    Eigen::VectorXf quad_points = Eigen::Map<Eigen::VectorXf>(points.data(), points.size());
                    Eigen::VectorXf quad_weights = Eigen::Map<Eigen::VectorXf>(weights.data(), weights.size());

                    for(int I = 0 ; I < quadRule ; I++){
                        for(int J = 0 ; J < quadRule ; J++){
                            float xi1 = quad_points(I);
                            float xi2 = quad_points(J);
                            float weights = quad_weights(I)*quad_weights(J);

                            Eigen::Matrix2f Jac = calculate_Jacobian_2D(e, xi1, xi2);
                            if(Jac.determinant() < 0){ 
                                throw std::runtime_error("Negative Jacobian detected");
                                break;
                            }
                            Eigen::MatrixXf Jac_inv = Jac.inverse();

                            auto [bfgradientA_xi1 , bfgradientA_xi2] = basis_gradient<Nne,BfOrder>(A, xi1, xi2);
                            auto [bfgradientB_xi1 , bfgradientB_xi2] = basis_gradient<Nne,BfOrder>(B, xi1, xi2);

                            Eigen::Vector2f bfgradientA(bfgradientA_xi1 , bfgradientA_xi2);
                            Eigen::Vector2f bfgradientB(bfgradientB_xi1 , bfgradientB_xi2);

                            Eigen::VectorXf term1 = Jac_inv.transpose()*bfgradientA;
                            Eigen::VectorXf term2 = Jac_inv.transpose()*bfgradientB;
                            Eigen::Matrix2f II = Eigen::Matrix2f::Identity();

                            Eigen::Matrix2f Klocal_mu = 2*mu*(term1.dot(term2))*II*Jac.determinant()*weights;
                            Eigen::Matrix2f Klocal_lambda = lambda*(term1*term2.transpose())*Jac.determinant()*weights;

                            Eigen::MatrixXf Kblock = Klocal_mu + Klocal_lambda;

                            Klocal.block<2,2>(2*A, 2*B) += Kblock;
                        }
                    }
                }
            }
        }

        //Assembly
        for(int A = 0; A < Nne; A++){
            int Aglobal = elements[e].node[A];
            for(int B = 0; B < Nne ; B++)
            {
                int Bglobal = elements[e].node[B];
                Kglobal.block<2,2>(2*Aglobal,2*Bglobal) += Klocal.block<2,2>(2*A,2*B);
            }
        }
        // cout << "Assembled into Kglobal for element : " << e << endl;
    }

    // std::ofstream Kglobal_file("Kglobal.txt");
    // Kglobal_file << Kglobal;
    
    //Boundary Conditions
    //global nodelocations where dirischlet boundary conditions are specified
    std::map<int, std::vector<int>> nodeLocationsD_map;
    for(int i = 0 ; i < Nt ; i++){
        //Fig1(a)
        // if(nodes[i].x2 == x2_ll){
        //     nodeLocationsD_map[i].push_back(0); // 0 => X1 displacement specified on this node 
        //     nodeLocationsD_map[i].push_back(1); // 1 => X2 displacement specified on this node
        // }
        // else if(nodes[i].x2 == x2_ul){
        //     if(nodes[i].x1 == x1_ul){
        //         nodeLocationsD_map[i].push_back(0); // 0=> only X1 displacement specified on this node
        //     }
        // }

        //Fig1(b)
        // if(nodes[i].x2 == x2_ll){
        //     nodeLocationsD_map[i].push_back(0); // 0 => X1 displacement specified on this node 
        //     nodeLocationsD_map[i].push_back(1); // 1 => X2 displacement specified on this node
        // }
        // else if(nodes[i].x2 == x2_ul){
        //     if(nodes[i].x1 == x1_ul){
        //         nodeLocationsD_map[i].push_back(0); // 0=> only X1 displacement specified on this node
        //         nodeLocationsD_map[i].push_back(1); // 1=> only X2 displacement specified on this node
        //     }
        // }

        //Fig1(c)
        if(nodes[i].x2 == x2_ll){
            nodeLocationsD_map[i].push_back(0); // 0 => X1 displacement specified on this node 
            nodeLocationsD_map[i].push_back(1); // 1 => X2 displacement specified on this node
        }
        else if(nodes[i].x1 == x1_ul){
            nodeLocationsD_map[i].push_back(0); // 0=> only X1 displacement specified on this node
            nodeLocationsD_map[i].push_back(1); // 1=> only X2 displacement specified on this node
        }
        else if(nodes[i].x2 == x2_ul){
            if(nodes[i].x1 == x1_ul){
                nodeLocationsD_map[i].push_back(0); // 0=> only X1 displacement specified on this node
                nodeLocationsD_map[i].push_back(1); // 1=> only X2 displacement specified on this node
            }
        }



    }
    vector<bool> isDirischlet(Nt,false);
    for(const auto& [key,vec] : nodeLocationsD_map){
        isDirischlet[key] = true;
    }
    //print this DirischletMap
    // for (const auto& [key, vec] : nodeLocationsD_map) {
    //     std::cout << key << " -> [ ";
    //     for (int v : vec) std::cout << v << " ";
    //     std::cout << "]\n";
    // }
    

    //indexes to remove from the solution array corresponding to dirischlet boundary conditions
    vector<int> dirischletIndexes;
    for(int i = 0 ; i < Nt ; i++){
        if(isDirischlet[i]){
            for(int dof : nodeLocationsD_map[i]){
                dirischletIndexes.push_back(2*i + dof);
            }
        }
    }
    vector<int> unknownIndexes;
    for(int i = 0 ; i < Nt ; i++){
        if(!isDirischlet[i]){
            unknownIndexes.push_back(2*i);
            unknownIndexes.push_back(2*i + 1);
        }
    }


    //given values of displacement field at dirischlet boundary
    Eigen::VectorXf dirischletVal(dirischletIndexes.size());
    for(int i = 0 ; i < dirischletIndexes.size() ; i++){
        int nodeD = dirischletIndexes[i]/2;
        int dof = dirischletIndexes[i]%2;

        //Fig1(a)
        // if(nodes[nodeD].x2 == x2_ll){
        //     dirischletVal(i) = 0.0; //all displacements are 0 at bottom face
        // }
        // else if(nodes[nodeD].x2 == x2_ul){
        //     if(nodes[nodeD].x1 == x1_ul){
        //         if(dof == 0){ //only x1 displacement is specified at top face
        //             dirischletVal(i) = 0.001; 
        //         }
        //     }
        // }

        //Fig1(b)
        // if(nodes[nodeD].x2 == x2_ll){
        //     dirischletVal(i) = 0.0; //all displacements are 0 at bottom face
        // }
        // else if(nodes[nodeD].x2 == x2_ul){
        //     if(nodes[nodeD].x1 == x1_ul){
        //         if(dof == 0){ //only x1 displacement is specified at top face
        //             dirischletVal(i) = 0.0007071068; 
        //         }
        //         if(dof == 1){
        //             dirischletVal(i) = 0.0007071068;
        //         }
        //     }
        // }

        //Fig1(c)
        if(nodes[nodeD].x2 == x2_ll){
            dirischletVal(i) = 0.0;
        }
        else if(nodes[nodeD].x1 == x1_ll){
            dirischletVal(i) = 0.0;
        }
        else if(nodes[nodeD].x2 == x2_ul){
            if(nodes[nodeD].x1 == x1_ul){
                if(dof == 0){ //only x1 displacement is specified at top face
                    dirischletVal(i) = 0.0007071068; 
                }
                if(dof == 1){
                    dirischletVal(i) = 0.0007071068;
                }
            }
        }

    } 


    Eigen::MatrixXf KUU = extractSubmatrix(Kglobal, unknownIndexes, unknownIndexes); //extract from Kglobal - only rows and columns pertaining to unknown node locations
    Eigen::MatrixXf KUD = extractSubmatrix(Kglobal, unknownIndexes, dirischletIndexes); //extract from Kglobal - only columns corresponding to Dirischlet node locations, for rows corresponding to unknown node locations
    
    Eigen::VectorXf FU(unknownIndexes.size()); //extract from Fglobal - only rows corresponding to unknown node locations
    for(int i = 0; i < unknownIndexes.size(); i++){
        FU(i) = Fglobal(unknownIndexes[i]);
    }

    Eigen::VectorXf F(FU.size()); //final forcing function vector after applying dirischlet boundary conditions
    F = FU - KUD*dirischletVal;

    //Solve for unknown displacements
    Eigen::LDLT<Eigen::MatrixXf> solver(KUU);
    Eigen::VectorXf DU = solver.solve(F);

    //construct final solution vector including known values at dirischlet boundary
    Eigen::VectorXf D_full = Eigen::VectorXf::Zero(Nt*3);
    for(int i = 0 ; i < unknownIndexes.size() ; i++){
        D_full(unknownIndexes[i]) = DU(i);
    }
    for(int i = 0 ; i < dirischletIndexes.size() ; i++){
        D_full(dirischletIndexes[i]) = dirischletVal(i);
    }

    //write solution to file
    std::ofstream D_file("solution.txt");
    for(int i = 0 ; i < Nt ; i++){
        D_file << D_full(2*i) << " " << D_full(2*i + 1) << "\n";
    }

    writeVTK("solution.vtk", Nt, Nel_t, nodes, elements, D_full);

}