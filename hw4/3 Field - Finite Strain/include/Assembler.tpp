#pragma once //include this only once during compilation

template <unsigned int Nsd, unsigned int Nne, unsigned int BfOrder>
Assembler<Nsd,Nne,BfOrder>::Assembler(
    const Mesh<Nsd,Nne>& mesh, const ElementEvaluator<Nsd,Nne,BfOrder>& elem_evaluator, const DiffusionEvaluator<Nsd,Nne,BfOrder>& diffusion_evaluator
) : mesh_(mesh), elem_evaluator_(elem_evaluator), diffusion_evaluator_(diffusion_evaluator)
{}

template <unsigned int Nsd, unsigned int Nne, unsigned int BfOrder>
Eigen::SparseMatrix<double> Assembler<Nsd,Nne,BfOrder>::extractSparseSubmatrix(
    const Eigen::SparseMatrix<double>& K,
    const std::vector<unsigned int>& rows,
    const std::vector<unsigned int>& cols) const {
        
    // Build a lookup set for fast membership testing
    std::unordered_set<unsigned int> rowSet(rows.begin(), rows.end());
    std::unordered_set<unsigned int> colSet(cols.begin(), cols.end());

    // Build index remapping: global index → local index in submatrix
    std::unordered_map<unsigned int,unsigned int> rowMap, colMap;
    for(unsigned int i = 0; i < rows.size(); i++) rowMap[rows[i]] = i;
    for(unsigned int j = 0; j < cols.size(); j++) colMap[cols[j]] = j;

    std::vector<Eigen::Triplet<double>> triplets;

    // Iterate over non-zeros of K
    for(int col = 0; col < K.outerSize(); col++){
        if(colSet.count(col) == 0) continue; // skip columns not in subset
        for(Eigen::SparseMatrix<double>::InnerIterator it(K, col); it; ++it){
            if(rowSet.count(it.row()) == 0) continue; // skip rows not in subset
            triplets.emplace_back(rowMap[it.row()], colMap[col], it.value());
        }
    }

    Eigen::SparseMatrix<double> sub(rows.size(), cols.size());
    sub.setFromTriplets(triplets.begin(), triplets.end());
    return sub;
}

template <unsigned int Nsd, unsigned int Nne, unsigned int BfOrder>
void Assembler<Nsd,Nne,BfOrder>::assembleSystem(
    const Eigen::VectorXd& u, //global nodal displacement vector (Nnodes*Nsd x 1 vector)
    const Eigen::VectorXd& C, //global nodal chemical concentration vector (Nnodes x 1 vector)
    const Eigen::VectorXd& T, //global nodal temperature vector (Nnodes x 1 vector)
    Eigen::SparseMatrix<double>& Kglobal, //global stiffness matrix (Nnodes*Nsd x Nnodes*Nsd sparse matrix)
    Eigen::VectorXd& Rglobal //global internal force vector (Nnodes*Nsd x 1 vector)
) const {
    unsigned int Nt = mesh_.Nnodes(); //total number of nodes in the mesh
    unsigned int Nel_t = mesh_.Nelements(); //total number of elements in the mesh

    Rglobal = Eigen::VectorXd::Zero(Nt * Nsd); //residual vector initialized to zero
    Kglobal = Eigen::SparseMatrix<double>(Nt * Nsd, Nt * Nsd); //sparse version of the tangent stiffness matrix for solving linear systems
    std::vector<Eigen::Triplet<double>> Kglobal_triplets; //triplet format for constructing the sparse tangent stiffness matrix
    // Kglobal_triplets.reserve(Nel_t*Nne*Nne*9); //reserve space for triplets to avoid dynamic resizing during assembly

    Eigen::MatrixXd Klocal = Eigen::MatrixXd::Zero(Nne * Nsd, Nne * Nsd); //local tangent stiffness matrix for the element
    Eigen::VectorXd Rlocal = Eigen::VectorXd::Zero(Nne * Nsd); //local residual vector for the element

    //Loop over elements and assemble the global stiffness matrix and residual vector
    for(unsigned int e = 0 ; e < Nel_t ; e++){
        //extract the chemical concentrations and temperatures for the nodes of the current element
        Eigen::VectorXd C_e = Eigen::VectorXd::Zero(Nne); //chemical concentration vector for the current element
        Eigen::VectorXd T_e = Eigen::VectorXd::Zero(Nne); //temperature vector for the current element
        Eigen::VectorXd u_e = Eigen::VectorXd::Zero(Nne * Nsd); //displacement vector for the current element
        for(unsigned int i = 0; i < Nne; i++){
            unsigned int global_node_id = mesh_.elements[e].node[i];
            C_e(i) = C(global_node_id); //extract chemical concentration for the node
            T_e(i) = T(global_node_id); //extract temperature for the node

            u_e.segment(i*Nsd, Nsd) = u.segment(global_node_id*Nsd, Nsd); //extract the displacements for the nodes of the current element 
        }
        
        elem_evaluator_.computeElement(
            e, //element index
            u_e, //element nodal displacements (Nne*Nsd x 1 vector)
            C_e, //element nodal chemical concentrations (Nne x 1 vector)
            T_e, //element nodal temperatures (Nne x 1 vector)
            Klocal, //element stiffness matrix (Nne*Nsd x Nne*Nsd matrix)
            Rlocal //element internal force vector (Nne*Nsd x 1 vector)
        );

        //Assemble Rlocal and Klocal into Rglobal and Kglobal
        for(int A = 0; A < Nne; A++){
            int Aglobal = mesh_.elements[e].node[A];
            for(int B = 0; B < Nne ; B++)
            {
                int Bglobal = mesh_.elements[e].node[B];
                for(int i = 0 ; i < Nsd ; i++){
                    for(int j = 0 ; j < Nsd ; j++){
                        Kglobal_triplets.emplace_back(Nsd*Aglobal + i, Nsd*Bglobal + j, Klocal(Nsd*A + i, Nsd*B + j));
                    }
                }
            }
            Rglobal.segment(Nsd*Aglobal,Nsd) += Rlocal.segment(Nsd*A,Nsd);
            // std::cout << "Assembled element " << e+1 << "/" << Nel_t << std::endl;
        }
    }
    Kglobal.setFromTriplets(Kglobal_triplets.begin(), Kglobal_triplets.end()); //construct the sparse global tangent stiffness matrix from the triplets
    Kglobal.makeCompressed(); //compress the sparse matrix for efficient arithmetic and solving
}

template <unsigned int Nsd, unsigned int Nne, unsigned int BfOrder>
void Assembler<Nsd,Nne,BfOrder>::assembleDiffusionSystem(
    Eigen::SparseMatrix<double>& MCCglobal, //global chemical mass matrix (Nnodes x Nnodes sparse matrix) 
    Eigen::SparseMatrix<double>& KCCglobal, //global chemical stiffness matrix (Nnodes x Nnodes sparse matrix)
    Eigen::SparseMatrix<double>& KCTglobal, //global coupling stiffness matrix between chemical concentration and temperature (Nnodes x Nnodes sparse matrix)

    Eigen::SparseMatrix<double>& MTTglobal, //global thermal mass matrix (Nnodes x Nnodes sparse matrix)
    Eigen::SparseMatrix<double>& KTTglobal, //global thermal stiffness matrix (Nnodes x Nnodes sparse matrix)
    Eigen::SparseMatrix<double>& KTCglobal, //global coupling stiffness matrix between temperature and chemical concentration (Nnodes x Nnodes sparse matrix)

    Eigen::SparseMatrix<double>& KuCglobal, //global coupling stiffness matrix between displacement and chemical concentration (Nnodes*Nsd x Nnodes sparse matrix)
    Eigen::SparseMatrix<double>& KuTglobal //global coupling stiffness matrix between displacement and temperature (Nnodes*Nsd x Nnodes sparse matrix)
) const {
    unsigned int Nt = mesh_.Nnodes(); //total number of nodes in the mesh
    unsigned int Nel_t = mesh_.Nelements(); //total number of elements in the mesh

    MCCglobal = Eigen::SparseMatrix<double>(Nt, Nt);
    KCCglobal = Eigen::SparseMatrix<double>(Nt, Nt);
    KCTglobal = Eigen::SparseMatrix<double>(Nt, Nt);

    MTTglobal = Eigen::SparseMatrix<double>(Nt, Nt);
    KTTglobal = Eigen::SparseMatrix<double>(Nt, Nt);
    KTCglobal = Eigen::SparseMatrix<double>(Nt, Nt);

    KuCglobal = Eigen::SparseMatrix<double>(Nt * Nsd, Nt);
    KuTglobal = Eigen::SparseMatrix<double>(Nt * Nsd, Nt);

    std::vector<Eigen::Triplet<double>> MCC_triplets, KCC_triplets, KCT_triplets;
    std::vector<Eigen::Triplet<double>> MTT_triplets, KTT_triplets, KTC_triplets;
    std::vector<Eigen::Triplet<double>> KuC_triplets, KuT_triplets;

    for(unsigned int e = 0 ; e < Nel_t ; e++){
        Eigen::MatrixXd MCClocal = Eigen::MatrixXd::Zero(Nne,Nne);
        Eigen::MatrixXd KCClocal = Eigen::MatrixXd::Zero(Nne,Nne);
        Eigen::MatrixXd KCTlocal = Eigen::MatrixXd::Zero(Nne,Nne);

        Eigen::MatrixXd MTTlocal = Eigen::MatrixXd::Zero(Nne,Nne);
        Eigen::MatrixXd KTTlocal = Eigen::MatrixXd::Zero(Nne,Nne);
        Eigen::MatrixXd KTClocal = Eigen::MatrixXd::Zero(Nne,Nne);

        Eigen::MatrixXd KuClocal = Eigen::MatrixXd::Zero(Nne*Nsd,Nne);
        Eigen::MatrixXd KuTlocal = Eigen::MatrixXd::Zero(Nne*Nsd,Nne);
        
        diffusion_evaluator_.computeDiffusionMatrices(
            e,
            MCClocal, KCClocal, KCTlocal,
            MTTlocal, KTTlocal, KTClocal,
            KuClocal, KuTlocal
        );
        
        for(int A = 0; A < Nne; A++){
            int Aglobal = mesh_.elements[e].node[A];
            for(int B = 0; B < Nne ; B++)
            {
                int Bglobal = mesh_.elements[e].node[B];
                for(int i = 0 ; i < Nsd ; i++){
                    for(int j = 0 ; j < Nsd ; j++){
                        MCC_triplets.emplace_back(Aglobal + i, Bglobal + j, MCClocal(A + i, B + j));
                        KCC_triplets.emplace_back(Aglobal + i, Bglobal + j, KCClocal(A + i, B + j));
                        KCT_triplets.emplace_back(Aglobal + i, Bglobal + j, KCTlocal(A + i, B + j));

                        MTT_triplets.emplace_back(Aglobal + i, Bglobal + j, MTTlocal(A + i, B + j));
                        KTT_triplets.emplace_back(Aglobal + i, Bglobal + j, KTTlocal(A + i, B + j));
                        KTC_triplets.emplace_back(Aglobal + i, Bglobal + j, KTClocal(A + i, B + j));

                        KuC_triplets.emplace_back(Nsd*Aglobal + i, Bglobal + j, KuClocal(Nsd*A + i, B + j));
                        KuT_triplets.emplace_back(Nsd*Aglobal + i, Bglobal + j, KuTlocal(Nsd*A + i, B + j));
                    }
                }
            }
            // std::cout << "Assembled diffusion for element " << e+1 << "/" << Nel_t << std::endl;
        }
    }
    MCCglobal.setFromTriplets(MCC_triplets.begin(), MCC_triplets.end());
    KCCglobal.setFromTriplets(KCC_triplets.begin(), KCC_triplets.end());
    KCTglobal.setFromTriplets(KCT_triplets.begin(), KCT_triplets.end());
    MCCglobal.makeCompressed();
    KCCglobal.makeCompressed();
    KCTglobal.makeCompressed();

    MTTglobal.setFromTriplets(MTT_triplets.begin(), MTT_triplets.end());
    KTTglobal.setFromTriplets(KTT_triplets.begin(), KTT_triplets.end());
    KTCglobal.setFromTriplets(KTC_triplets.begin(), KTC_triplets.end());
    MTTglobal.makeCompressed();
    KTTglobal.makeCompressed();
    KTCglobal.makeCompressed();

    KuCglobal.setFromTriplets(KuC_triplets.begin(), KuC_triplets.end());
    KuTglobal.setFromTriplets(KuT_triplets.begin(), KuT_triplets.end());
    KuCglobal.makeCompressed();
    KuTglobal.makeCompressed();
}


template <unsigned int Nsd, unsigned int Nne, unsigned int BfOrder>
void Assembler<Nsd,Nne,BfOrder>::partition(
    const Eigen::SparseMatrix<double>& Kglobal, //global stiffness matrix (Nnodes*Nsd x Nnodes*Nsd sparse matrix)
    Eigen::VectorXd& Rglobal, //global internal force vector (Nnodes*Nsd x 1 vector)
    const BoundaryConditions<Nsd,Nne>& bcs, //boundary conditions object containing the indexes of the dirischlet DOFs

    Eigen::SparseMatrix<double>& KUU, //extract the submatrix of K corresponding to the unknown degrees of freedom
    Eigen::SparseMatrix<double>& KUD, //extract the submatrix of K corresponding to the coupling between unknown and dirischlet degrees of freedom
    Eigen::VectorXd& RU //extract the subvector of R corresponding to the unknown degrees of freedom
) const {
    const auto& dirischletIndexes = bcs.getDirischletIndexes(); //get the indexes of the dirischlet degrees of freedom from the boundary conditions object
    const auto& unknownIndexes = bcs.getUnknownIndexes(); //get the indexes of the unknown degrees of freedom from the boundary conditions object

    KUU = extractSparseSubmatrix(Kglobal, unknownIndexes, unknownIndexes); //extract the KUU submatrix corresponding to the unknown degrees of freedom
    KUD = extractSparseSubmatrix(Kglobal, unknownIndexes, dirischletIndexes); //extract the KUD submatrix corresponding to the coupling between unknown and dirischlet degrees of freedom
    
    RU.resize(unknownIndexes.size()); //extract the subvector of R corresponding to the unknown degrees of freedom
    for(int i = 0; i < unknownIndexes.size() ; i++){
        RU(i) = Rglobal(unknownIndexes[i]);
    }
}