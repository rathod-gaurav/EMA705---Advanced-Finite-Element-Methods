#pragma once //include this only once during compilation

#include <vector>

template <unsigned int Nsd, unsigned int Nne>
struct QuadratureRule {
    std::vector<double> points;
    std::vector<double> weights;
};

template <> //specialization for 2D triangle elements
struct QuadratureRule<2,3>{
    std::vector<double> points_x1;
    std::vector<double> points_x2;
    std::vector<double> weights;
};

template <unsigned int Nsd, unsigned int Nne>
class Quadrature{
    public:
        static QuadratureRule<Nsd,Nne> gauss_legendre(unsigned int n);
};

#include "Quadrature.tpp"
