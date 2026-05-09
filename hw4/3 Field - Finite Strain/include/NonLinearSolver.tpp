#pragma once //include this only once during compilation

template <unsigned int Nsd, unsigned int Nne, unsigned int BfOrder>
NonlinearSolver<Nsd,Nne,BfOrder>::NonlinearSolver(double tol, unsigned int maxIncr, unsigned int maxIter, unsigned int maxTimeSteps)
    : tol_(tol), maxIncr_(maxIncr), maxIter_(maxIter), maxTimeSteps_(maxTimeSteps)
{}

template <unsigned int Nsd, unsigned int Nne, unsigned int BfOrder>
void NonlinearSolver<Nsd,Nne,BfOrder>::solve(
    Eigen::VectorXd& u, //displacement vector, modified in place
    Eigen::VectorXd& C, //chemical concentration vector, modified in place
    Eigen::VectorXd& T, //temperature vector, modified in place
    double dt, //time step size for the diffusion equations
    const Assembler<Nsd,Nne,BfOrder>& assembler, //provides Kglobal, Rglobal
    const BoundaryConditions<Nsd,Nne>& bcs, //provides dirischlet indexes and values for displacement
    const BoundaryConditionsScalar<Nsd,Nne>& bcs_C, //provides dirischlet indexes and values for chemical concentration
    const BoundaryConditionsScalar<Nsd,Nne>& bcs_T, //provides dirischlet indexes and values for temperature
    std::function<void(unsigned int, unsigned int, double)> iterCallback
){
    Eigen::VectorXd Rglobal, RU; //global residual vector
    Eigen::SparseMatrix<double> Kglobal, KUU, KUD; //global stiffness matrix

    Eigen::SparseMatrix<double> MCCglobal, KCCglobal, KCTglobal; //global matrices for the chemical diffusion equation

    Eigen::SparseMatrix<double> MTTglobal, KTTglobal, KTCglobal; //global matrices for the thermal diffusion equation

    Eigen::SparseMatrix<double> KuCglobal, KuTglobal; //global matrices for the eigenstrain loads due to chemical and thermal expansion
    
    assembler.assembleDiffusionSystem(MCCglobal, KCCglobal, KCTglobal, MTTglobal, KTTglobal, KTCglobal, KuCglobal, KuTglobal); //assemble the global matrices for the diffusion equations //done only once since they are constant for this problem
    
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solverMCC; //create a sparse LU solver for the chemica diffusion equation
    solverMCC.analyzePattern(MCCglobal);
    solverMCC.compute(MCCglobal);

    Eigen::SparseLU<Eigen::SparseMatrix<double>> solverMTT; //create a sparse LU solver for the thermal diffusion equations
    solverMTT.analyzePattern(MTTglobal);
    solverMTT.compute(MTTglobal);

    for(unsigned int incr = 0; incr < maxIncr_; incr++){
        double incrFraction = (incr+1)/static_cast<double>(maxIncr_); //factor to scale dirischlet values for current incr
        bcs.applyToSolution(u, incrFraction); //apply dirischlet boundary conditions to the solution vector for the current incr
        
        for(unsigned int timestep = 0 ; timestep < maxTimeSteps_ ; timestep++){
            bcs_C.applyToSolution(C, 1.0);
            bcs_T.applyToSolution(T, 1.0);
            
            //update C and T for the current time step using forward Euler method
            Eigen::VectorXd RC = -KCCglobal*C - KCTglobal*T;
            Eigen::VectorXd RT = -KTTglobal*T - KTCglobal*C;

            for(auto dof : bcs_C.getDirischletIndexes()) RC(dof) = 0.0;
            for(auto dof : bcs_T.getDirischletIndexes()) RT(dof) = 0.0;

            Eigen::VectorXd MCC_aux = solverMCC.solve(RC); //solve for the change in chemical concentration due to diffusion
            Eigen::VectorXd MTT_aux = solverMTT.solve(RT); //solve for the change in temperature due to diffusion

            C += dt*MCC_aux;
            T += dt*MTT_aux;

            for(unsigned int iter = 0; iter < maxIter_; iter++){
                
                assembler.assembleSystem(u, C, T, Kglobal, Rglobal); //assemble the global stiffness matrix and residual vector based on the current solution vector
                
                //add eigenstrain loads (effect of C and T on u) to the residual vector
                Rglobal -= KuCglobal * C + KuTglobal * T; //subtract the contribution of chemical and thermal expansion to the residual

                assembler.partition(Kglobal, Rglobal, bcs, KUU, KUD, RU); //partition the global stiffness matrix and residual vector into submatrices/vectors corresponding to unknown and dirischlet degrees of freedom

                // solve the linear system
                // std::cout << "Initilising solver for incr " << incr+1 << ", iteration " << iter+1 << "\n";

                // Eigen::FullPivLU<Eigen::MatrixXd> solver(KUU);
                Eigen::SparseLU<Eigen::SparseMatrix<double>> linear_solver;
                linear_solver.analyzePattern(KUU);
                linear_solver.factorize(KUU);
                if(linear_solver.info() != Eigen::Success) {
                    std::cout << "Decomposition failed for incr " << incr+1 << ", iteration " << iter+1 << "\n";
                    
                    throw std::runtime_error("KUU Matrix decomposition failed");
                }

                // std::cout << "Initilised solver for incr " << incr+1 << ", iteration " << iter+1 << "\n";

                Eigen::VectorXd duU = linear_solver.solve(-RU); //solve for the incral displacements at the unknown degrees of freedom
                
                // std::cout << "Solved for incr " << incr+1 << ", iteration " << iter+1 << "\n";

                //construct full du vector including known values at dirischlet boundary
                const auto& unknownIndexes = bcs.getUnknownIndexes();
                for(int i = 0 ; i < unknownIndexes.size() ; i++){
                    u(unknownIndexes[i]) += duU(i);
                }

                //check residual norm for convergence
                double residualNorm = RU.norm();
                std::cout << "timeStep: " << timestep << ", incr: " << incr+1 << ", Iteration: " << iter+1 << "\n";
                std::cout << "Modified residual norm: " << residualNorm << "\n"; //print the norm of the modified residual to monitor convergence of the unknown degrees of freedom
                std::cout << "-----------------------------------" << "\n";
                
                if(iterCallback){
                    iterCallback(timestep, iter, residualNorm); //call the iteration callback function if provided
                }

                if(residualNorm < tol_){
                    std::cout << "Convergence achieved for incr " << incr+1 << " in " << iter+1 << " iterations." << "\n";
                    break; 
                }
            }   
        }        
    }
}