#pragma once //include this only once during compilation


template<unsigned int Nsd, unsigned int Nne>
OutputWriter<Nsd, Nne>::OutputWriter(const std::string& outputDir)
: outputDir_(outputDir) //initialize the output directory member variable
{
    // Create the output directory if it doesn't exist
    std::filesystem::create_directories(outputDir_);
}

// template<unsigned int Nsd, unsigned int Nne>
// std::string OutputWriter<Nsd, Nne>::writeVTU(
//     const Mesh<Nsd,Nne>& mesh,
//     const Eigen::VectorXd& u,
//     unsigned int incr
// ){
//     std::string filename = outputDir_ + "/solution_"
//                          + std::to_string(incr + 1) + ".vtu";

//     std::ofstream file(filename);
//     if (!file.is_open()) {
//         std::cerr << "OutputWriter: could not open " << filename << "\n";
//         return filename;
//     }

//     unsigned int Nn = mesh.Nnodes();
//     unsigned int Ne = mesh.Nelements();

//     file << "<?xml version=\"1.0\"?>\n"
//          << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
//          << "<UnstructuredGrid>\n"
//          << "<Piece NumberOfPoints=\"" << Nn << "\" NumberOfCells=\"" << Ne << "\">\n";

//     // --- Points ---
//     if constexpr (Nsd == 2){
//         file << "<Points>\n"
//             << "<DataArray type=\"Float32\" NumberOfComponents=\"2\" format=\"ascii\">\n";
//         for (unsigned int i = 0; i < Nn; i++)
//             file << mesh.nodes[i].x1 << " "
//                 << mesh.nodes[i].x2 << "\n";
//         file << "</DataArray>\n</Points>\n";
//     }
//     else if constexpr (Nsd == 3){
//         file << "<Points>\n"
//             << "<DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n";
//         for (unsigned int i = 0; i < Nn; i++)
//             file << mesh.nodes[i].x1 << " "
//                 << mesh.nodes[i].x2 << " "
//                 << mesh.nodes[i].x3 << "\n";
//         file << "</DataArray>\n</Points>\n";
//     }


//     // --- Cells ---
//     file << "<Cells>\n"
//          << "<DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
//     for (unsigned int e = 0; e < Ne; e++) {
//         for (unsigned int A = 0; A < Nne; A++)
//             file << mesh.elements[e].node[A] << " ";
//         file << "\n";
//     }
//     file << "</DataArray>\n"
//          << "<DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
//     for (unsigned int e = 0; e < Ne; e++) file << (e+1)*8 << " ";
//     file << "\n</DataArray>\n"
//          << "<DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
//     for (unsigned int e = 0; e < Ne; e++) file << "12 "; // 12 = VTK hexahedron
//     file << "\n</DataArray>\n</Cells>\n";

//     // --- Point data: displacement ---
//     if constexpr (Nsd == 2){
//         file << "<PointData Vectors=\"Displacement\">\n"
//             << "<DataArray type=\"Float32\" Name=\"Displacement\" "
//             << "NumberOfComponents=\"3\" format=\"ascii\">\n";
//         for (unsigned int i = 0; i < Nn; i++)
//             file << u(Nsd*i) << " " << u(Nsd*i+1) << "\n";
//         file << "</DataArray>\n</PointData>\n";
//     }
//     else if constexpr (Nsd == 3){
//         file << "<PointData Vectors=\"Displacement\">\n"
//             << "<DataArray type=\"Float32\" Name=\"Displacement\" "
//             << "NumberOfComponents=\"3\" format=\"ascii\">\n";
//         for (unsigned int i = 0; i < Nn; i++)
//             file << u(Nsd*i) << " " << u(Nsd*i+1) << " " << u(Nsd*i+2) << "\n";
//         file << "</DataArray>\n</PointData>\n";
//     }
    

//     file << "</Piece>\n</UnstructuredGrid>\n</VTKFile>\n";

//     vtuFiles_.push_back(filename);
//     timestamps_.push_back(incr + 1);
//     return filename;
// }

template<unsigned int Nsd, unsigned int Nne>
std::string OutputWriter<Nsd, Nne>::writeVTU(
    const Mesh<Nsd, Nne>& mesh,
    const Eigen::VectorXd& u,
    unsigned int incr
){
    std::string filename = outputDir_ + "/solution_" + std::to_string(incr + 1) + ".vtu";
    std::ofstream file(filename);
    if (!file.is_open()) return filename;

    unsigned int Nn = mesh.Nnodes();
    unsigned int Ne = mesh.Nelements();

    file << "<?xml version=\"1.0\"?>\n"
         << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
         << "<UnstructuredGrid>\n"
         << "<Piece NumberOfPoints=\"" << Nn << "\" NumberOfCells=\"" << Ne << "\">\n";

    // --- Points ---
    // ParaView requires 3 components for coordinates, even in 2D
    file << "<Points>\n"
         << "<DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (unsigned int i = 0; i < Nn; ++i) {
        file << mesh.nodes[i].x1 << " " << mesh.nodes[i].x2 << " ";
        if constexpr (Nsd == 3) file << mesh.nodes[i].x3 << "\n";
        else                    file << "0.0\n";
    }
    file << "</DataArray>\n</Points>\n";

    // --- Cells ---
    file << "<Cells>\n"
         << "<DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
    for (unsigned int e = 0; e < Ne; ++e) {
        for (unsigned int A = 0; A < Nne; ++A) {
            file << mesh.elements[e].node[A] << " ";
        }
        file << "\n";
    }
    file << "</DataArray>\n"
         << "<DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
    for (unsigned int e = 0; e < Ne; ++e) {
        file << (e + 1) * Nne << " ";
    }
    file << "\n</DataArray>\n"
         << "<DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
    
    // Inline constexpr lambda to determine VTK Type ID based on Nne
    constexpr int vtkType = []() constexpr {
        if constexpr (Nne == 3)  return 5;  // VTK_TRIANGLE
        if constexpr (Nne == 4)  return 9;  // VTK_QUAD
        if constexpr (Nne == 8)  return 12; // VTK_HEXAHEDRON
        if constexpr (Nne == 27) return 29; // VTK_BIQUADRATIC_HEXAHEDRON
        return 1; // Default
    }();

    for (unsigned int e = 0; e < Ne; ++e) {
        file << vtkType << " ";
    }
    file << "\n</DataArray>\n</Cells>\n";

    // --- Point Data (Displacement) ---
    // ParaView requires 3 components for vectors, even in 2D
    file << "<PointData Vectors=\"Displacement\">\n"
         << "<DataArray type=\"Float32\" Name=\"Displacement\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (unsigned int i = 0; i < Nn; ++i) {
        file << u(Nsd * i) << " " << u(Nsd * i + 1) << " ";
        if constexpr (Nsd == 3) file << u(Nsd * i + 2) << "\n";
        else                    file << "0.0\n";
    }
    file << "</DataArray>\n</PointData>\n";

    file << "</Piece>\n</UnstructuredGrid>\n</VTKFile>\n";

    vtuFiles_.push_back(filename);
    timestamps_.push_back(incr + 1);
    return filename;
}

template<unsigned int Nsd, unsigned int Nne>
void OutputWriter<Nsd, Nne>::writePVD(
    const std::string& filename
) const {
    std::string path = outputDir_ + "/" + filename;
    std::ofstream file(path);
    if (!file.is_open()) {
        std::cerr << "OutputWriter: could not open " << path << "\n";
        return;
    }

    file << "<?xml version=\"1.0\"?>\n"
         << "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
         << "<Collection>\n";
    for (size_t i = 0; i < vtuFiles_.size(); i++)
        file << "  <DataSet timestep=\"" << timestamps_[i]
             << "\" group=\"\" part=\"0\" file=\"" << vtuFiles_[i] << "\"/>\n";
    file << "</Collection>\n</VTKFile>\n";

    std::cout << "PVD written: " << path << "\n";
}

template<unsigned int Nsd, unsigned int Nne>
void OutputWriter<Nsd, Nne>::sendResidual(
    unsigned int incr,
    unsigned int iter,
    double residualNorm
) const {

    CURL* curl = curl_easy_init();
    if (!curl) return;

    std::ostringstream json;
    json << "{\"increment\":" << incr
        << ",\"iteration\":" << iter
        << ",\"residual\":"  << residualNorm << "}";
    std::string body = json.str();

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");

    curl_easy_setopt(curl, CURLOPT_URL,       "http://127.0.0.1:8000/residual");
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT,    2L); // don't block the solver

    curl_easy_perform(curl);
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
}