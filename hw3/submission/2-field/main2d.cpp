// Linear Elliptic PDE with vector variable - 2D elasto statics
// no body forces, no neumann conditions, only dirischlet boundary conditions at top and bottom faces of the domain

#include <iostream>
using namespace std;
#include <vector>
#include <fstream>
#include <Eigen/Dense>
#include <map>
#include <numeric>

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

template <unsigned int Nne, unsigned int Npe, unsigned int BfOrderP>
std::tuple<float,float> xi_at_nodeP(bool bubble, unsigned int node){ //function to return xi1 and xi2 for given node A
    float xi1, xi2;   
    if constexpr (BfOrderP == 0){
        if constexpr (Nne == 3){
            if constexpr (Npe == 1){
                switch(node){
                    case 0:
                        xi1 = 1.0/3.0;
                        xi2 = 1.0/3.0;
                        break;
                    default:
                        throw std::invalid_argument("xi_at_node mapping not implemented for this local node number");
                }
            }
            else{
                throw std::invalid_argument("Npe not acceptable for given Nne and BfOrderP");
            }
        }
        else if constexpr (Nne == 4 || Nne == 9){
            if constexpr (Npe == 1){
                switch(node){
                    case 0:
                        xi1 = 0.0;
                        xi2 = 0.0;
                        break;
                    default:
                        throw std::invalid_argument("xi_at_node mapping not implemented for this local node number");
                }
            }
            else{
                throw std::invalid_argument("Npe not acceptable for given Nne and BfOrderP");
            }
        }
        else{
            throw std::invalid_argument("Nne not acceptable for given Nsd and BfOrder");
        }
    }
    else if constexpr (BfOrderP == 1){
        if constexpr (Nne == 4 || Nne == 9){
            if constexpr (Npe == 4){
                if(!bubble){//P nodes at the vertices
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
                    switch(node){
                        case 0:
                            xi1 = -0.5773502691896257;
                            xi2 = -0.5773502691896257;
                            break;
                        case 1:
                            xi1 = 0.5773502691896257;
                            xi2 = -0.5773502691896257;
                            break;
                        case 2:
                            xi1 = 0.5773502691896257;
                            xi2 = 0.5773502691896257;
                            break;
                        case 3:
                            xi1 = -0.5773502691896257;
                            xi2 = 0.5773502691896257;
                            break;
                        default:
                            throw std::invalid_argument("xi_at_node mapping not implemented for this local node number");
                    }
                }
            }
            else{
                throw std::invalid_argument("Npe not acceptable for given Nne and BfOrderP");
            }
        }
        else{
            throw std::invalid_argument("Nne not acceptable for given Nsd and BfOrder");
        }
    }

    return {xi1, xi2};
};

template <unsigned int Nne, unsigned int Npe, unsigned int BfOrderP>
float basis_functionP(bool bubble, unsigned int node, float xi1, float xi2){
    Eigen::Vector2f xi_vec(xi1, xi2);
    auto [xi1_node,xi2_node] = xi_at_nodeP<Nne,Npe,BfOrderP>(bubble, node);
    float value = 0.0f;
    if constexpr (BfOrderP == 0){
        if constexpr (Nne == 3 || Nne == 4 || Nne == 9){
            if constexpr (Npe == 1){
                switch(node){
                    case 0:
                        value = 1.0;
                        break;
                    default:
                        throw std::invalid_argument("cannot evaluate basis function value for out of bound local node number");
                }
            }
            else{
                throw std::invalid_argument("Npe not acceptable for given Nne and BfOrderP");
            }
        }
        else{
            throw std::invalid_argument("Nne not acceptable for given Nsd and BfOrder");
        }
    }
    else if constexpr (BfOrderP == 1){
        if constexpr (Nne == 4 || Nne == 9){
            if constexpr (Npe == 4){
                if(!bubble){//P dofs at vertices
                    value = 0.25*(1 + xi_vec(0)*xi1_node)*(1 + xi_vec(1)*xi2_node);
                }
                else{//P dofs inside the element
                    const double g = 1.0 / std::sqrt(3.0);
                    const double b = (1.0 - xi_vec(0)*xi_vec(0)) * (1.0 - xi_vec(1)*xi_vec(1));

                    const double lm_xi  = (xi_vec(0) - g) / (-2.0 * g);
                    const double lp_xi  = (xi_vec(0) + g) / ( 2.0 * g);
                    const double lm_eta = (xi_vec(1) - g) / (-2.0 * g);
                    const double lp_eta = (xi_vec(1) + g) / ( 2.0 * g);

                    switch (node) {
                        case 0: 
                            value = b * lm_xi  * lm_eta;   
                            break;
                        case 1: 
                            value = b * lp_xi  * lm_eta;  
                            break;
                        case 2: 
                            value = b * lp_xi  * lp_eta;   
                            break;
                        case 3: 
                            value = b * lm_xi  * lp_eta;  
                            break;
                        default: 
                            throw std::invalid_argument("cannot evaluate basis function value for out of bound local node number");
                    }
                }
            }
            else{
                throw std::invalid_argument("Npe not acceptable for given Nne and BfOrderP");
            }
        }
        else{
            throw std::invalid_argument("Nne not acceptable for given Nsd and BfOrder");
        }
    }
    return value;
};

template <unsigned int Nne, unsigned int Npe, unsigned int BfOrderP>
std::tuple<float,float> basis_gradientP(bool bubble, unsigned int node, float xi1, float xi2){
    Eigen::Vector2f xi_vec(xi1, xi2);
    auto [xi1_node,xi2_node] = xi_at_nodeP<Nne,Npe,BfOrderP>(node);
    float basis_gradient_xi1, basis_gradient_xi2;
    if constexpr (BfOrderP == 0){
        if constexpr (Nne == 3 || Nne == 4 || Nne == 9){
            if constexpr (Npe == 1){
                switch(node){
                    case 0:
                        basis_gradient_xi1 = 0.0;
                        basis_gradient_xi2 = 0.0;
                        break;
                    default:
                        throw std::invalid_argument("cannot evaluate basis function value for out of bound local node number");
                }
            }
            else{
                throw std::invalid_argument("Npe not acceptable for given Nne and BfOrderP");
            }
        }
        else{
            throw std::invalid_argument("Nne not acceptable for given Nsd and BfOrder");
        }
    }
    else if constexpr (BfOrderP == 1){
        if constexpr (Nne == 4 || Nne == 9){
            if constexpr (Npe == 4){
                if(!bubble){//P dofs at vertices
                    basis_gradient_xi1 = 0.25*xi1_node*(1 + xi_vec(1)*xi2_node);
                    basis_gradient_xi2 = 0.25*xi2_node*(1 + xi_vec(0)*xi1_node);
                }
                else { // P dofs inside the element
                    const float g = 1.0f / std::sqrt(3.0f);

                    // bubble and its derivatives
                    const float b       = (1.0f - xi1*xi1) * (1.0f - xi2*xi2);
                    const float db_dxi1 = -2.0f*xi1 * (1.0f - xi2*xi2);
                    const float db_dxi2 = -2.0f*xi2 * (1.0f - xi1*xi1);

                    // 1D Lagrange values on {-g, +g}
                    const float lm_xi1 = (xi1 - g) / (-2.0f*g);
                    const float lp_xi1 = (xi1 + g) / ( 2.0f*g);
                    const float lm_xi2 = (xi2 - g) / (-2.0f*g);
                    const float lp_xi2 = (xi2 + g) / ( 2.0f*g);

                    // 1D Lagrange derivatives (constants)
                    const float dlm_dxi1 = 1.0f / (-2.0f*g);
                    const float dlp_dxi1 = 1.0f / ( 2.0f*g);
                    const float dlm_dxi2 = 1.0f / (-2.0f*g);
                    const float dlp_dxi2 = 1.0f / ( 2.0f*g);

                    float L, dL_dxi1, dL_dxi2;

                    switch (node) {
                        case 0:                              
                            L        = lm_xi1 * lm_xi2;
                            dL_dxi1  = dlm_dxi1 * lm_xi2;
                            dL_dxi2  = lm_xi1  * dlm_dxi2;
                            break;
                        case 1:                              
                            L        = lp_xi1 * lm_xi2;
                            dL_dxi1  = dlp_dxi1 * lm_xi2;
                            dL_dxi2  = lp_xi1  * dlm_dxi2;
                            break;
                        case 2:                              
                            L        = lp_xi1 * lp_xi2;
                            dL_dxi1  = dlp_dxi1 * lp_xi2;
                            dL_dxi2  = lp_xi1  * dlp_dxi2;
                            break;
                        case 3:                              
                            L        = lm_xi1 * lp_xi2;
                            dL_dxi1  = dlm_dxi1 * lp_xi2;
                            dL_dxi2  = lm_xi1  * dlp_dxi2;
                            break;
                        default:
                            throw std::invalid_argument("cannot evaluate basis function gradient for out of bound local node number");
                    }

                    basis_gradient_xi1 = db_dxi1 * L + b * dL_dxi1;
                    basis_gradient_xi2 = db_dxi2 * L + b * dL_dxi2;
                }
            }
            else{
                throw std::invalid_argument("Npe not acceptable for given Nne and BfOrderP");
            }
        }
        else{
            throw std::invalid_argument("Nne not acceptable for given Nsd and BfOrder");
        }
    }
    return {basis_gradient_xi1, basis_gradient_xi2};
};

template <unsigned int Nn>
struct QuadratureRule {
    std::vector<float> points;
    std::vector<float> points2;
    std::vector<float> weights;
};

template <unsigned int Nn>
QuadratureRule<Nn> gauss_legendre(unsigned int n) {
    QuadratureRule<Nn> rule;

    if constexpr (Nn == 3){
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
    else if constexpr (Nn == 4 || Nn == 9){
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

    constexpr unsigned int BfOrderP = 1;
    constexpr unsigned int Npe = 4;
    unsigned int quadRuleP = 2;
    bool bubble = true; //false = P dofs at the u dofs (vertices)

    //problem variables
    float lambda = 9.15e12;
    float mu = 1.83e10;
    // std::cout << lambda << std::endl;
    // std::cout << mu << std::endl;

    //domain
    float x1_ll = 0.0;
    float x1_ul = 0.03;
    float x2_ll = 0.0;
    float x2_ul = 0.01;

    //Mesh
    unsigned int Nel_x1 = 6; //number of elements in x1 direction
    unsigned int Nel_x2 = 4; //number of elements in x2 direction
    unsigned int Nt, Nel_t;
    double dx1, dx2;

    // Mesh Generator
    std::vector<Node> nodes;
    std::vector<Element<Nne>> elements;
    if constexpr (BfOrder == 1){
        unsigned int Nnodes_x1 = Nel_x1 + 1; //number of nodes in x1 direction
        unsigned int Nnodes_x2 = Nel_x2 + 1; //number of nodes in x2 direction
        
        Nt = Nnodes_x1 * Nnodes_x2; //total number of nodes

        dx1 = (x1_ul - x1_ll) / Nel_x1; //spacing between nodes in x1 direction
        dx2 = (x2_ul - x2_ll) / Nel_x2; //spacing between nodes in x2 direction

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

        dx1 = (x1_ul - x1_ll) / (2*Nel_x1); //spacing between nodes in x1 direction
        dx2 = (x2_ul - x2_ll) / (2*Nel_x2); //spacing between nodes in x2 direction

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

    //for Pressure DOFs
    std::vector<Node> nodesP;
    std::vector<Element<Npe>> elementsP;
    elementsP.reserve(Nel_t);
    if constexpr (BfOrderP == 0){
        if constexpr (Nne == 3 || Nne == 4 || Nne == 9){
            if constexpr (Npe == 1){
                for(int e = 0 ; e < Nel_t ; e++){
                    //global coordinates for P
                    Node n;
                    std::vector<float> x1_arr(Nne); //global x1 coordinates of all nodes for element e
                    std::vector<float> x2_arr(Nne); //global x2 coordinates of all nodes for element e
                    unsigned int A_lim = Nne;
                    if constexpr (Nne == 9){
                        A_lim = 4;
                    }
                    for(int A = 0 ; A < A_lim ; A++){
                        int Aglobal = elements[e].node[A];
                        x1_arr[A] = nodes[Aglobal].x1;
                        x2_arr[A] = nodes[Aglobal].x2;
                    }
                    float sum_x1 = std::accumulate(x1_arr.begin(), x1_arr.end(), 0.0f);
                    float sum_x2 = std::accumulate(x2_arr.begin(), x2_arr.end(), 0.0f);
                    n.x1 = sum_x1/A_lim;
                    n.x2 = sum_x2/A_lim;
                    nodesP.push_back(n);

                    //triangulation for P
                    Element<Npe> elem;
                    elem.node[0] = e;
                    elementsP.push_back(elem);
                }
            }
            else{
                throw std::invalid_argument("Npe not acceptable for given Nne and BfOrderP");
            }
        }
        else{
            throw std::invalid_argument("Nne not acceptable for given Nsd and BfOrderP");
        }
    }
    else if constexpr (BfOrderP == 1){
        if constexpr (Nne == 4 || Nne == 9){
            if constexpr (Npe == 4){
                if(!bubble){//if P dofs are on the vertex of element
                    nodesP = nodes; //global coordinates of P will be same as that of u
                    elementsP.clear();
                    elementsP.reserve(elements.size());
                    for (const auto& e : elements) {
                        Element<Npe> newElem;
                        // Copy only the nodes that fit in Npe (4 nodes)
                        for (unsigned int i = 0; i < Npe; ++i) {
                            newElem.node[i] = e.node[i];
                        }
                        elementsP.push_back(newElem);
                    }
                }
                else{//if P dofs are inside the element
                    auto qp = gauss_legendre<Nne>(2);
                    std::vector<float> points(2);

                    points = qp.points;
                    Eigen::VectorXf p_points = Eigen::Map<Eigen::VectorXf>(points.data(), points.size());
                    
                    for(int e = 0 ; e < Nel_t ; e++){
                        //global coordinates for P
                        std::vector<float> x1_arr(Nne); //global x1 coordinates of all nodes for element e
                        std::vector<float> x2_arr(Nne); //global x2 coordinates of all nodes for element e
                        unsigned int A_lim = Nne;
                        if constexpr (Nne == 9){
                            A_lim = 4;
                        }
                        for(int A = 0 ; A < A_lim ; A++){
                            int Aglobal = elements[e].node[A];
                            x1_arr[A] = nodes[Aglobal].x1;
                            x2_arr[A] = nodes[Aglobal].x2;
                        }
                        float sum_x1 = std::accumulate(x1_arr.begin(), x1_arr.end(), 0.0f);
                        float sum_x2 = std::accumulate(x2_arr.begin(), x2_arr.end(), 0.0f);
                        float c_x1 = sum_x1/A_lim;
                        float c_x2 = sum_x2/A_lim;

                        for(int j = 0 ; j < 2 ; j++){
                            for(int i = 0 ; i < 2 ; i++){
                                float sub_x1 = p_points(i)*(dx1/2);
                                float sub_x2 = p_points(j)*(dx2/2);

                                Node n;
                                n.x1 = c_x1 + sub_x1;
                                n.x2 = c_x2 + sub_x2;

                                nodesP.push_back(n);
                            }
                        }

                        //triangulation for P
                        // Element<Npe> elem;

                        // int n0 = e*Nne + 0;
                        // int n1 = n0 + 1;
                        // int n2 = n0 + 3;
                        // int n3 = n0 + 2;

                        // elem.node[0] = n0;
                        // elem.node[1] = n1;
                        // elem.node[2] = n2;
                        // elem.node[3] = n3;

                        // elementsP.push_back(elem);


                        int np_x = 2 * Nel_x1;  // pressure nodes per row

                        int ex = e % Nel_x1;
                        int ey = e / Nel_x2;

                        int base_row = ey * 2;       // which pair of pressure rows this element starts at
                        int base_col = ex * 2;       // which pair of pressure cols

                        int n0 = (base_row    ) * np_x + base_col;      // bottom-left
                        int n1 = (base_row    ) * np_x + base_col + 1;  // bottom-right
                        int n2 = (base_row + 1) * np_x + base_col;      // top-left
                        int n3 = (base_row + 1) * np_x + base_col + 1;  // top-right
                        
                        Element<Npe> elem;
                        elem.node[0] = n0;
                        elem.node[1] = n1;
                        elem.node[2] = n3;  // winding: bottom-left, bottom-right, top-right, top-left
                        elem.node[3] = n2;
                        elementsP.push_back(elem);
                    }

                }
            }
            else{
                throw std::invalid_argument("Npe not acceptable for given Nne and BfOrderP");
            }
        }
        else{
            throw std::invalid_argument("Nne not acceptable for given Nsd and BfOrderP");
        }
    }
    else{
        throw std::invalid_argument("Invalid basis function order given for pressure, only supported BfOrderP are 0 and 1");
    }

    //store the mesh into points and hexa files
    std::ofstream pointsP_file("mesh/pointsP.txt");
    for(auto& node : nodesP){
        pointsP_file << node.x1 << " " << node.x2 << "\n";
    }

    std::ofstream elemsP_file("mesh/elemsP.txt");
    for(auto& elem : elementsP){ 
        for(unsigned int i = 0 ; i < Npe ; i++){
            elemsP_file << elem.node[i] << " ";
        }
        elemsP_file << "\n";
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

    auto global_x_from_xi = [Nsd, Nne, elements, nodes, BfOrder](int e, float xi1, float xi2){//function to calculate global x coordinates for given quadrature points
        Eigen::VectorXf xglobal = Eigen::VectorXf::Zero(Nsd);
        for(int A = 0 ; A < Nne ; A++){
            int Aglobal = elements[e].node[A];
            xglobal(0) += basis_function<Nne,BfOrder>(A, xi1, xi2)*nodes[Aglobal].x1;
            xglobal(1) += basis_function<Nne,BfOrder>(A, xi1, xi2)*nodes[Aglobal].x2;
        }
        return xglobal;
    };

    // Eigen::VectorXf vtry;
    // vtry = global_x_from_xi(0, -0.5773502691896257, -0.5773502691896257);
    // std::cout << vtry(0) << " ; " << vtry(1) << std::endl;
    // vtry = global_x_from_xi(0, 0.5773502691896257, -0.5773502691896257);
    // std::cout << vtry(0) << " ; " << vtry(1) << std::endl;
    // vtry = global_x_from_xi(0, -0.5773502691896257, 0.5773502691896257);
    // std::cout << vtry(0) << " ; " << vtry(1) << std::endl;
    // vtry = global_x_from_xi(0, 0.5773502691896257, 0.5773502691896257);
    // std::cout << vtry(0) << " ; " << vtry(1) << std::endl;

    // Nnp = total number of pressure nodes
    unsigned int Nnp = nodesP.size();

    Eigen::MatrixXf Kglobal  = Eigen::MatrixXf::Zero(Nt*Nsd,Nt*Nsd);  // [2*Nnu x 2*Nnu]
    Eigen::MatrixXf GglobalT = Eigen::MatrixXf::Zero(Nnp,Nt*Nsd);  // [Nnp x 2*Nnu] 
    Eigen::MatrixXf Mglobal  = Eigen::MatrixXf::Zero(Nnp,Nnp);     // [Nnp x Nnp]
    Eigen::VectorXf Fglobal  = Eigen::VectorXf::Zero(Nt*Nsd);

    for(int e = 0; e < Nel_t ; e++){

        Eigen::MatrixXf Klocal  = Eigen::MatrixXf::Zero(Nne*Nsd, Nne*Nsd);
        Eigen::MatrixXf GlocalT = Eigen::MatrixXf::Zero(Npe,Nne*Nsd);
        Eigen::MatrixXf Mlocal  = Eigen::MatrixXf::Zero(Npe,Npe);

        if constexpr(Nne == 3){
            // 1-point quadrature for triangle
            auto qp = gauss_legendre<Nne>(quadRule);
            for(int q = 0 ; q < qp.points.size() ; q++){
                float xi1_q = qp.points[q];
                float xi2_q = qp.points2[q];
                float w_q   = qp.weights[q];  

                // Jacobian and its determinant
                Eigen::Matrix2f J = calculate_Jacobian_2D(e, xi1_q, xi2_q);
                float detJ = J.determinant();
                Eigen::Matrix2f Jinv = J.inverse();

                // K_local
                for(int A = 0 ; A < Nne ; A++){
                    auto [dNA_dxi1, dNA_dxi2] = basis_gradient<Nne,BfOrder>(A, xi1_q, xi2_q);
                    float dNA_dx1 = Jinv(0,0)*dNA_dxi1 + Jinv(1,0)*dNA_dxi2;
                    float dNA_dx2 = Jinv(0,1)*dNA_dxi1 + Jinv(1,1)*dNA_dxi2;

                    for(int B = 0 ; B < Nne ; B++){
                        auto [dNB_dxi1, dNB_dxi2] = basis_gradient<Nne,BfOrder>(B, xi1_q, xi2_q);
                        float dNB_dx1 = Jinv(0,0)*dNB_dxi1 + Jinv(1,0)*dNB_dxi2;
                        float dNB_dx2 = Jinv(0,1)*dNB_dxi1 + Jinv(1,1)*dNB_dxi2;

                        float grad_dot = dNA_dx1*dNB_dx1 + dNA_dx2*dNB_dx2;

                        float K_AB = 2.0f * mu * grad_dot * w_q * detJ;
                        Klocal(2*A,   2*B)   += K_AB;  // i=1,j=1
                        Klocal(2*A+1, 2*B+1) += K_AB;  // i=2,j=2
                    }
                }

                // GlocalT
                for(int A = 0 ; A < Npe ; A++){
                    float NA_p = basis_functionP<Nne,Npe,BfOrderP>(bubble, A, xi1_q, xi2_q);

                    for(int B = 0 ; B < Nne ; B++){
                        auto [dNB_dxi1, dNB_dxi2] = basis_gradient<Nne,BfOrder>(B, xi1_q, xi2_q);
                        float dNB_dx1 = Jinv(0,0)*dNB_dxi1 + Jinv(1,0)*dNB_dxi2;  // dNB/dx1
                        float dNB_dx2 = Jinv(0,1)*dNB_dxi1 + Jinv(1,1)*dNB_dxi2;  // dNB/dx2

                        // column 2*B   → u1-dof of node B → dNB/dx1 (divergence contribution)
                        // column 2*B+1 → u2-dof of node B → dNB/dx2
                        GlocalT(A, 2*B)   += NA_p * dNB_dx1 * w_q * detJ;
                        GlocalT(A, 2*B+1) += NA_p * dNB_dx2 * w_q * detJ;
                    }
                }

                // Mlocal
                for(int A = 0 ; A < Npe ; A++){
                    float NA_p = basis_functionP<Nne,Npe,BfOrderP>(bubble, A, xi1_q, xi2_q);
                    for(int B = 0 ; B < Npe ; B++){
                        float NB_p = basis_functionP<Nne,Npe,BfOrderP>(bubble, B, xi1_q, xi2_q);
                        Mlocal(A,B) += (NA_p * NB_p / lambda) * w_q * detJ;
                    }
                }
            }
        }

        if constexpr(Nne == 4 || Nne == 9){
            
            auto qp = gauss_legendre<Nne>(quadRule);
            for(int qi = 0 ; qi < qp.points.size() ; qi++){
                for(int qj = 0 ; qj < qp.points.size() ; qj++){
                    float xi1_q = qp.points[qi];
                    float xi2_q = qp.points[qj];
                    float w_q   = qp.weights[qi] * qp.weights[qj];

                    // Jacobian
                    Eigen::Matrix2f J = calculate_Jacobian_2D(e, xi1_q, xi2_q);
                    float detJ = J.determinant();
                    Eigen::Matrix2f Jinv = J.inverse();

                    //K_local
                    for(int A = 0 ; A < Nne ; A++){
                        auto [dNA_dxi1, dNA_dxi2] = basis_gradient<Nne,BfOrder>(A, xi1_q, xi2_q);
                        float dNA_dx1 = Jinv(0,0)*dNA_dxi1 + Jinv(1,0)*dNA_dxi2;
                        float dNA_dx2 = Jinv(0,1)*dNA_dxi1 + Jinv(1,1)*dNA_dxi2;

                        for(int B = 0 ; B < Nne ; B++){
                            auto [dNB_dxi1, dNB_dxi2] = basis_gradient<Nne,BfOrder>(B, xi1_q, xi2_q);
                            float dNB_dx1 = Jinv(0,0)*dNB_dxi1 + Jinv(1,0)*dNB_dxi2;
                            float dNB_dx2 = Jinv(0,1)*dNB_dxi1 + Jinv(1,1)*dNB_dxi2;

                            float grad_dot = dNA_dx1*dNB_dx1 + dNA_dx2*dNB_dx2;

                            float K_AB = 2.0f * mu * grad_dot * w_q * detJ;
                            Klocal(2*A,   2*B)   += K_AB;
                            Klocal(2*A+1, 2*B+1) += K_AB;
                        }
                    }

                    //GlocalT
                    for(int A = 0 ; A < Npe ; A++){
                        float NA_p = basis_functionP<Nne,Npe,BfOrderP>(bubble, A, xi1_q, xi2_q);

                        for(int B = 0 ; B < Nne ; B++){
                            auto [dNB_dxi1, dNB_dxi2] = basis_gradient<Nne,BfOrder>(B, xi1_q, xi2_q);
                            float dNB_dx1 = Jinv(0,0)*dNB_dxi1 + Jinv(1,0)*dNB_dxi2;
                            float dNB_dx2 = Jinv(0,1)*dNB_dxi1 + Jinv(1,1)*dNB_dxi2;

                            GlocalT(A, 2*B)   += NA_p * dNB_dx1 * w_q * detJ;
                            GlocalT(A, 2*B+1) += NA_p * dNB_dx2 * w_q * detJ;
                        }
                    }

                    //Mlocal
                    for(int A = 0 ; A < Npe ; A++){
                        float NA_p = basis_functionP<Nne,Npe,BfOrderP>(bubble, A, xi1_q, xi2_q);
                        for(int B = 0 ; B < Npe ; B++){
                            float NB_p = basis_functionP<Nne,Npe,BfOrderP>(bubble, B, xi1_q, xi2_q);
                            Mlocal(A,B) += (NA_p * NB_p / lambda) * w_q * detJ;
                        }
                    }
                }
            }
        }


        // K assembly
        for(int A = 0; A < Nne; A++){
            int Aglobal = elements[e].node[A];
            for(int B = 0; B < Nne ; B++)
            {
                int Bglobal = elements[e].node[B];
                Kglobal.block<2,2>(2*Aglobal,2*Bglobal) += Klocal.block<2,2>(2*A,2*B);
            }
        }
        
        // GglobalT assembly: GglobalT(p-global, u-global dof)
        for(int A = 0; A < Npe; A++){
            int Aglobal = elementsP[e].node[A];   // global pressure node index
            for(int B = 0; B < Nne ; B++)
            {
                int Bglobal = elements[e].node[B]; // global u node index
                // each u node contributes 2 dofs: 2*Bglobal and 2*Bglobal+1
                GglobalT(Aglobal, 2*Bglobal)   += GlocalT(A, 2*B);
                GglobalT(Aglobal, 2*Bglobal+1) += GlocalT(A, 2*B+1);
            }
        }

        // Mglobal assembly
        for(int A = 0; A < Npe; A++){
            int Aglobal = elementsP[e].node[A];
            for(int B = 0; B < Npe ; B++)
            {
                int Bglobal = elementsP[e].node[B];
                Mglobal(Aglobal, Bglobal) += Mlocal(A, B);
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
        // if(nodes[i].x2 == x2_ll){
        //     nodeLocationsD_map[i].push_back(0); // 0 => X1 displacement specified on this node 
        //     nodeLocationsD_map[i].push_back(1); // 1 => X2 displacement specified on this node
        // }
        // else if(nodes[i].x1 == x1_ul){
        //     nodeLocationsD_map[i].push_back(0); // 0=> only X1 displacement specified on this node
        //     nodeLocationsD_map[i].push_back(1); // 1=> only X2 displacement specified on this node
        // }
        // else if(nodes[i].x2 == x2_ul){
        //     if(nodes[i].x1 == x1_ul){
        //         nodeLocationsD_map[i].push_back(0); // 0=> only X1 displacement specified on this node
        //         nodeLocationsD_map[i].push_back(1); // 1=> only X2 displacement specified on this node
        //     }
        // }

        //Fig2
        if(nodes[i].x1 == x1_ll){
            nodeLocationsD_map[i].push_back(0); // 0 => X1 displacement specified on this node 
            nodeLocationsD_map[i].push_back(1); // 1 => X2 displacement specified on this node
        }
        else if(nodes[i].x1 == x1_ul){
            nodeLocationsD_map[i].push_back(0); // 0=> only X1 displacement specified on this node
            nodeLocationsD_map[i].push_back(1); // 1=> only X2 displacement specified on this node
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
        // if(nodes[nodeD].x2 == x2_ll){
        //     dirischletVal(i) = 0.0;
        // }
        // else if(nodes[nodeD].x1 == x1_ll){
        //     dirischletVal(i) = 0.0;
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

        //Fig2
        if(nodes[nodeD].x1 == x1_ll){
            dirischletVal(i) = 0.0;
        }
        else if(nodes[nodeD].x1 == x1_ul){
            dirischletVal(i) = 0.0;
            if(dof == 1){ //y displacement on entire x = 0.3 face
                dirischletVal(i) = -0.002;
            }
        }
    } 


    Eigen::MatrixXf KUU = extractSubmatrix(Kglobal, unknownIndexes, unknownIndexes);
    Eigen::MatrixXf KUD = extractSubmatrix(Kglobal, unknownIndexes, dirischletIndexes);
    
    std::vector<int> allPressureIndexes(Nnp);
    std::iota(allPressureIndexes.begin(), allPressureIndexes.end(), 0);

    Eigen::MatrixXf GU = extractSubmatrix(GglobalT, allPressureIndexes, unknownIndexes);   // [Nnp x Nu_free]
    Eigen::MatrixXf GD = extractSubmatrix(GglobalT, allPressureIndexes, dirischletIndexes); // [Nnp x Nu_fixed]

    Eigen::VectorXf FU(unknownIndexes.size());
    for(int i = 0; i < (int)unknownIndexes.size(); i++){
        FU(i) = Fglobal(unknownIndexes[i]);
    }

    Eigen::VectorXf F1 = FU  - KUD * dirischletVal;
    Eigen::VectorXf F2 =     - GD  * dirischletVal;

    Eigen::LLT<Eigen::MatrixXf> llt_KUU(KUU);
    if(llt_KUU.info() != Eigen::Success){
        std::cerr << "WARNING: KUU is not SPD — Cholesky failed. Falling back to LU.\n";
    }

    Eigen::MatrixXf GUT = GU.transpose();                                
    Eigen::MatrixXf KinvGUT = llt_KUU.solve(GUT);                      
    
    Eigen::VectorXf KinvF1 = llt_KUU.solve(F1);                        
    
    Eigen::MatrixXf S = GU * KinvGUT - Mglobal;                        

    Eigen::VectorXf S_rhs = GU * KinvF1 - F2;

    Eigen::VectorXf alpha = S.lu().solve(S_rhs);


    Eigen::VectorXf DU = KinvF1 - KinvGUT * alpha;                   

    //construct final solution vector including known values at dirischlet boundary
    Eigen::VectorXf D_full = Eigen::VectorXf::Zero(Nt*Nsd);
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