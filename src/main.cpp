#include <iostream>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <vector>
#include <string>
#include <iomanip>
#include <eigen-3.4.0/Eigen/Dense>
#include <argparse/argparse.hpp>
#include <bitset>
#include <queue>
#include <algorithm>
#include <set>
#include <unordered_map>
#include <time.h>
#include <cmath>
#include <chrono>
#include <myproj_bigint.hpp>

using xyz = Eigen::Vector3d;

// Read PDB file and return a vector of xyz coordinates
void ReadPDB(std::string filename, std::vector<xyz>& coords) {
    std::ifstream pdbFile(filename);
    std::string line;
    int num_atoms = 0;
    while (std::getline(pdbFile, line)) {

        // read line and split by whitespace
        std::istringstream iss(line);
        std::vector<std::string> tokens;
        for (std::string s; iss >> s;) tokens.push_back(s);

        if (tokens[0] == "ATOM") { // only process ATOM lines
            // if there are multiple entities, read the first one
            int now_num_atoms = std::stoi(tokens[1]);
            if (now_num_atoms > num_atoms){
                num_atoms = now_num_atoms;
            }else{
                break;
            }
            if (tokens[2] == "CA") { // only process CA atoms

                double x = std::stof(tokens[6]);
                double y = std::stof(tokens[7]);
                double z = std::stof(tokens[8]);
                coords.push_back(xyz(x, y, z));
            }
        }
    }

    pdbFile.close();
}

double Distance(xyz a, xyz b){
    return xyz(a - b).norm();
}

bool which_distance_is_smaller(xyz a, xyz b){
    return xyz(a).norm() < xyz(b).norm();
}

// the function to compute the rotation matrix that aligns two vectors p1 and p2
Eigen::Matrix3d calculateRotationMatrix(const Eigen::Vector3d& p1, const Eigen::Vector3d& p2) {

    // Normalize to get unit vectors
    Eigen::Vector3d u1 = p1.normalized();
    Eigen::Vector3d u2 = p2.normalized();

    // Compute cross product
    Eigen::Vector3d v = u1.cross(u2);
    // Compute dot product
    double c = u1.dot(u2);

    // If vectors are parallel
    if ( c > 1.0 - 1e-6) {
        return Eigen::Matrix3d::Identity();
    }

    // If vectors are in opposite directions
    if ( c < -1.0 + 1e-6) {
        // Choose any orthogonal vector as axis
        Eigen::Vector3d axis = Eigen::Vector3d::UnitX(); // Default is x-axis
        if (std::abs(u1.x()) > 0.9) { // If u1 is close to x-axis, use z-axis
            axis = Eigen::Vector3d::UnitZ();
        }
        axis = axis.cross(u1).normalized(); // Find axis orthogonal to u1

        // Construct rotation matrix
        Eigen::AngleAxisd angleAxis(M_PI, axis);
        return angleAxis.toRotationMatrix();
    }

    // Magnitude of cross product
    double s = v.norm();

    // Generate rotation matrix between unit vectors
    Eigen::Matrix3d vx;
    vx << 0, -v.z(), v.y(),
          v.z(), 0, -v.x(),
         -v.y(), v.x(), 0;

    Eigen::Matrix3d R = Eigen::Matrix3d::Identity() + vx + (vx * vx) * ((1 - c) / (s * s));
    return R;
}


// Function: Calculate the transformation matrix that superimposes (p1, p2) onto (q1, q2)
Eigen::Matrix4d calculateTransformationMatrix(
    const Eigen::Vector3d& p1, const Eigen::Vector3d& p2,
    const Eigen::Vector3d& q1, const Eigen::Vector3d& q2) {
    
    // Calculate vectors p2-p1 and q2-q1
    Eigen::Vector3d v1 = p2 - p1;
    Eigen::Vector3d v2 = q2 - q1;

    if (v1.norm() < 1e-6 || v2.norm() < 1e-6) {
        Eigen::Matrix4d translation = Eigen::Matrix4d::Identity();
        translation.block<3, 1>(0, 3) = q1 - p1; // Only translation needed
        return translation;
    }

    v1 = v1.normalized();
    v2 = v2.normalized();

    // Calculate rotation matrix
    Eigen::Matrix3d rotationMatrix = calculateRotationMatrix(v1, v2);

    // Combine translation and rotation
    Eigen::Matrix4d transformationMatrix = Eigen::Matrix4d::Identity();

    // 1. Translate p1 to the origin
    Eigen::Vector3d translationToOrigin = -p1;
    Eigen::Matrix4d translateToOrigin = Eigen::Matrix4d::Identity();
    translateToOrigin.block<3, 1>(0, 3) = translationToOrigin;

    // 2. Apply rotation
    Eigen::Matrix4d rotation = Eigen::Matrix4d::Identity();
    rotation.block<3, 3>(0, 0) = rotationMatrix;

    // 3. Translate to q1
    Eigen::Vector3d translationToQ1 = q1;
    Eigen::Matrix4d translateToQ1 = Eigen::Matrix4d::Identity();
    translateToQ1.block<3, 1>(0, 3) = translationToQ1;

    // Final transformation matrix
    transformationMatrix = translateToQ1 * rotation * translateToOrigin;

    // Debug
    if (transformationMatrix.hasNaN()){
        std::cerr << "transformationMatrix has nan" << std::endl;
        std::cerr << "p1" << std::endl;
        std::cerr << p1 << std::endl;
        std::cerr << "p2" << std::endl;
        std::cerr << p2 << std::endl;
        std::cerr << "q1" << std::endl;
        std::cerr << q1 << std::endl;
        std::cerr << "q2" << std::endl;
        std::cerr << q2 << std::endl;
        std::cerr << "v1" << std::endl;
        std::cerr << v1 << std::endl;
        std::cerr << "v2" << std::endl;
        std::cerr << v2 << std::endl;
        std::cerr << "rotationMatrix" << std::endl;
        std::cerr << rotationMatrix << std::endl;
        std::cerr << "translateToOrigin" << std::endl;
        std::cerr << translateToOrigin << std::endl;
        std::cerr << "rotation" << std::endl;
        std::cerr << rotation << std::endl;
        std::cerr << "translateToQ1" << std::endl;
        std::cerr << translateToQ1 << std::endl;
        exit(0);
    }

    return transformationMatrix;
}

// Function: Compute transformation matrix that moves p to the origin and aligns q with the z-axis
Eigen::Matrix4d computeTransformationMatrixToZero(const Eigen::Vector3d& p, const Eigen::Vector3d& q) {

    // Translation: move p to the origin
    Eigen::Matrix4d translation = Eigen::Matrix4d::Identity();
    translation.block<3, 1>(0, 3) = -p;

    // Coordinates of q after translation
    Eigen::Vector3d q_translated = q - p;

    // If zero vector, no rotation needed
    if (q_translated.norm() < 1e-6) {
        return translation; // Only translation needed
    }

    // Compute rotation to align q_translated with the z-axis
    Eigen::Vector3d z_axis(0, 0, 1);
    Eigen::Vector3d rotation_axis = {q_translated[1], -q_translated[0], 0};
    double rotation_angle = std::acos(q_translated.normalized().dot(z_axis));

    // If rotation axis is zero vector, no rotation needed
    Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
    if (rotation_axis.norm() > 1e-6) {
        rotation_axis.normalize();
        Eigen::AngleAxisd angleAxis(rotation_angle, rotation_axis);
        rotation = angleAxis.toRotationMatrix();
    }

    // Expand rotation matrix to 4x4
    Eigen::Matrix4d rotation_4x4 = Eigen::Matrix4d::Identity();
    rotation_4x4.block<3, 3>(0, 0) = rotation;

    // Final transformation matrix
    Eigen::Matrix4d transformation = rotation_4x4 * translation;
    return transformation;
}

void push_back_angle_segment_with_index(std::vector<std::pair<double, int>>& angles, double min_theta, double max_theta, int index) {
    if (min_theta < -M_PI){
        angles.push_back(std::make_pair(min_theta + 2*M_PI + 1e-6, +index));
        angles.push_back(std::make_pair(M_PI, -index));
        angles.push_back(std::make_pair(-M_PI, +index));
        angles.push_back(std::make_pair(max_theta - 1e-6, -index));
    } else if (max_theta > M_PI){
        angles.push_back(std::make_pair(min_theta + 1e-6, +index));
        angles.push_back(std::make_pair(M_PI, -index));
        angles.push_back(std::make_pair(-M_PI, +index));
        angles.push_back(std::make_pair(max_theta - 2*M_PI - 1e-6, -index));
    } else {
        angles.push_back(std::make_pair(min_theta + 1e-6, +index));
        angles.push_back(std::make_pair(max_theta - 1e-6, -index));
    }

    return;
} 

// Function: Simplified version of BasicAlign (A slight modification is made to facilitate implementation.)
int BasicAlign_simplified(const Eigen::Matrix3Xd& P, const Eigen::Matrix3Xd& Q, double epsilon, Eigen::Matrix3Xd& res_P, Eigen::Matrix3Xd& res_Q){

    int N = P.cols();
    int M = Q.cols();
    int max_size_of_solution = 0;
    int cnt = 0;

    for (int i1 = 0; i1 < N; i1++) for (int i2 = 0; i2 < N; i2++) for (int i3 = 0; i3 < N; i3++) {
        for (int j1 = 0; j1 < M; j1++) for (int j2 = 0; j2 < M ; j2++) for (int j3 = 0; j3 < M ; j3++) {

            if (i1 == i2 || i2 == i3 || i3 == i1) continue;
            if (j1 == j2 || j2 == j3 || j3 == j1) continue;

            // Calculate the transformation matrix that moves A[i1] to the origin and aligns A[i2] with the z-axis
            Eigen::Matrix4d transformP = computeTransformationMatrixToZero(P.col(i1), P.col(i2));
            Eigen::Matrix3Xd P_tmp = transformP.block<3, 3>(0, 0) * P + transformP.block<3, 1>(0, 3).replicate(1, N);
            // Calculate the rotation angle of P_tmp[i3] from the x-axis
            double theta_A = 0;
            if (P_tmp(1, i3)*P_tmp(1, i3) + P_tmp(0, i3)*P_tmp(0, i3) > 1e-6) theta_A = atan2(P_tmp(1, i3), P_tmp(0, i3));
            // Rotate A_tmp around the z-axis by that angle
            Eigen::Matrix3d R_A = Eigen::AngleAxisd(-theta_A, Eigen::Vector3d::UnitZ()).toRotationMatrix();
            P_tmp = R_A * P_tmp;
            // Calculate the transformation matrix that moves B[j1] to the origin and aligns B[j2] with the z-axis
            Eigen::Matrix4d transformQ = computeTransformationMatrixToZero(Q.col(j1), Q.col(j2));
            Eigen::Matrix3Xd Q_tmp = transformQ.block<3, 3>(0, 0) * Q + transformQ.block<3, 1>(0, 3).replicate(1, M);
            // Calculate the rotation angle of Q_tmp[j3] from the x-axis
            double theta_B;
            if (Q_tmp(1, j3)*Q_tmp(1, j3) + Q_tmp(0, j3)*Q_tmp(0, j3) > 1e-6) theta_B = atan2(Q_tmp(1, j3), Q_tmp(0, j3));
            // Rotate B_tmp around the z-axis by that angle
            Eigen::Matrix3d R_B = Eigen::AngleAxisd(-theta_B, Eigen::Vector3d::UnitZ()).toRotationMatrix();
            Q_tmp = R_B * Q_tmp;

            // If the maximum distance between corresponding points is too large, skip
            double d_Tppqq = std::max({Distance(P_tmp.col(i1), Q_tmp.col(j1)), Distance(P_tmp.col(i2), Q_tmp.col(j2)), Distance(P_tmp.col(i3), Q_tmp.col(j3))});
            if (d_Tppqq > (epsilon)*8.0) continue;

            // Compute the maximum (ordered) matching between A_tmp and B_tmp
            std::vector<std::vector<bool>> connected(N, std::vector<bool>(M, 0));
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < M; j++) {
                    double dist = Distance(P_tmp.col(i), Q_tmp.col(j));
                    if (dist < (epsilon)*8.0) connected[i][j] = 1;
                }
            }

            std::vector<std::vector<int>> A(N+1, std::vector<int>(M+1, 0));
            for (int i = 1; i <= N; i++) {
                for (int j = 1; j <= M; j++) {
                    A[i][j] = std::max({A[i-1][j], A[i][j-1], A[i-1][j-1]+connected[i-1][j-1]});
                }
            }
            if (A[N][M] > max_size_of_solution) {
                max_size_of_solution = A[N][M];
                res_P = P_tmp;
                res_Q = Q_tmp;
            }
        }
    }

    return max_size_of_solution;
}


// Function: Calculate the maximum size of solution and the corresponding angle using the original LiNgAlign DP algorithm
void calculate_with_original_LiNgAlign_DP(std::vector<std::pair<double,int>> &angles, int N, int M, int &max_size_of_solution, double &max_theta) {

    int NM_angle_segments = angles.size();
    int cnt = 0;

    std::vector<std::vector<bool>> connected(N+2, std::vector<bool>(M+2, false));

    for (int a = 0; a < NM_angle_segments-1; a++){
        int index = angles[a].second;
        if (index >= 0) {
            index--;
            int x = index / M;
            int y = index % M;
            connected[x+1][y+1] = true;
        } else {
            index = -index - 1;
            int x = index / M;
            int y = index % M;
            connected[x+1][y+1] = false;
        }

        if (angles[a].first == angles[a+1].first) continue;

        // Note: While it is possible to implement f using an unordered_map, this approach incurs significant overhead due to constant factors,
        //       resulting in considerably slower performance. Therefore, an implementation using a vector is adopted in this work.

        std::vector<std::vector<int>> f(N+2,std::vector<int>(M+2,0));

        for(int i=N;i>=1;i--) for(int j=M;j>=1;j--) {
            if (connected[i+1][j]) {
                f[i][j] = std::max(f[i][j], f[i+1][j]);
            }
            if (connected[i][j+1]) {
                f[i][j] = std::max(f[i][j], f[i][j+1]);
            }
            for (int x = 0; j+x <= M; x++) {
                if (connected[i][j+x]) {
                    f[i][j] = std::max(f[i][j], f[i+1][j+x+1]+1);
                }
            }
            for (int x = 0; i+x <= N; x++) {
                if (connected[i+x][j]) {
                    f[i][j] = std::max(f[i][j], f[i+x+1][j+1]+1);
                }
            }
    
            cnt = std::max(cnt, f[i][j]);
        }

        if (cnt > max_size_of_solution){
            max_size_of_solution = cnt;
            max_theta = (angles[a].first + angles[a+1].first) / 2.0;
        }
    }

    return;
}

void calculate_with_LIS_based_algorithm(std::vector<std::pair<double,int>> &angles, int N, int M, int &max_size_of_solution, double &max_theta);
void calculate_with_LCS_based_algorithm(std::vector<std::pair<double,int>> &angles, int N, int M, int &max_size_of_solution, double &max_theta);


// To simplify the problem, the l-axis is assumed to be aligned with the z-axis in this function.
// threshold is the distance threshold of (1 + delta) * epsilon, where epsilon is the distance threshold of the original problem.
int ComputeMax(const Eigen::Matrix3Xd& P, const Eigen::Matrix3Xd& Q, double threshold, Eigen::Matrix3Xd& res_R, std::string DP_type = "LiNgAlign") {

    int N = P.cols();
    int M = Q.cols();

    std::vector<std::pair<double, int>> angles;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            double dist_z = abs(P(2,i) - Q(2,j));
            if (dist_z > threshold) continue;
            double radius = sqrtl(threshold*threshold - dist_z*dist_z);
            double Pdist_from_Zaxis = sqrtl(P(0,i)*P(0,i) + P(1,i)*P(1,i));
            double Qdist_from_Zaxis = sqrtl(Q(0,j)*Q(0,j) + Q(1,j)*Q(1,j));
            if (abs(Pdist_from_Zaxis - Qdist_from_Zaxis) > radius) continue;

            if (Pdist_from_Zaxis + Qdist_from_Zaxis < radius) {
                angles.push_back(std::make_pair(-M_PI, i*M+j+1));
                angles.push_back(std::make_pair(M_PI, -i*M-j-1));
            }else{
                
                double theta = atan2(P(1,i), P(0,i))-atan2(Q(1,j), Q(0,j));
                if (theta < -M_PI) theta += 2*M_PI;
                if (theta > M_PI) theta -= 2*M_PI;
                
                double angle = acos((Pdist_from_Zaxis*Pdist_from_Zaxis + Qdist_from_Zaxis*Qdist_from_Zaxis - radius*radius) / (2*Pdist_from_Zaxis*Qdist_from_Zaxis));
                double min_theta = theta - angle;
                double max_theta = theta + angle;
                push_back_angle_segment_with_index(angles, min_theta, max_theta, i*M+j+1);
            }
        }
    }

    int max_size_of_solution = 0;
    int NM_angle_segments = angles.size();

    std::sort(angles.begin(), angles.end());
    double max_theta = -M_PI;

    if (DP_type == "LiNgAlign"){
        calculate_with_original_LiNgAlign_DP(angles, N, M, max_size_of_solution, max_theta);
    }else if (DP_type == "AlignFastLIS"){
        calculate_with_LIS_based_algorithm(angles, N, M, max_size_of_solution, max_theta);
    }else if (DP_type == "AlignFastLCS"){
        calculate_with_LCS_based_algorithm(angles, N, M, max_size_of_solution, max_theta);
    }else{
        std::cerr << "Unknown DP type: " << DP_type << std::endl;
        exit(1);
    }
    
    Eigen::Matrix3d R = Eigen::AngleAxisd(max_theta, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    res_R = R;

    return max_size_of_solution;
}

// Function: Simplified version of LiNgAlign (A slight modification is made to facilitate implementation.)
int LiNgAlign_simplified(const Eigen::Matrix3Xd& P, const Eigen::Matrix3Xd& Q, double epsilon, double delta, Eigen::Matrix3Xd& res_P, Eigen::Matrix3Xd& res_Q) {

    int N = P.cols();
    int M = Q.cols();
    
    const double grid_length = epsilon*delta/3.0;
    int grid_num = floor(3.0/delta-1e-6);

    long long sum = (long long)N*N*M*M*(2*grid_num+1)*(2*grid_num+1)*(2*grid_num+1);

    int cnt = 0;

    int max_size_of_solution = 0;

    Eigen::Matrix3Xd R;

    // The grid configuration employed in this function represents only one possible approach; various other constructions are also feasible.
    for (int i = 0; i < N; i++){
        for (int j = 0; j < M; j++){
            for (int k = 0; k < N; k++){
                for (int l = 0; l < M; l++){

                    if (i == k || j == l) continue;

                    double Pi_Pk_distance = Distance(P.col(i), P.col(k));
                    double Qj_Ql_distance = Distance(Q.col(j), Q.col(l));

                    if (abs(Pi_Pk_distance - Qj_Ql_distance) > 2.0*epsilon*(1.0+delta)) continue;

                    if (Pi_Pk_distance < 1e-6 || Qj_Ql_distance < 1e-6) continue;

                    Eigen::Matrix4d transformP = computeTransformationMatrixToZero(P.col(i), P.col(k));
                    Eigen::Matrix3Xd P_tmp = transformP.block<3, 3>(0, 0) * P + transformP.block<3, 1>(0, 3).replicate(1, N);

                    Eigen::Matrix4d transformQ = computeTransformationMatrixToZero(Q.col(j), Q.col(l));
                    Eigen::Matrix3Xd Q_tmp = transformQ.block<3, 3>(0, 0) * Q + transformQ.block<3, 1>(0, 3).replicate(1, M);
                    
                    for (int xi = -grid_num; xi <= grid_num; xi++){
                        for (int yi = -grid_num; yi <= grid_num; yi++){
                            for (int zi = -grid_num; zi <= grid_num; zi++){
                                
                                if ((xi ^ yi ^ zi) & 1) continue;

                                double dx1 = xi*grid_length;
                                double dy1 = yi*grid_length;
                                double dz1 = zi*grid_length;

                                if (dx1*dx1 + dy1*dy1 + dz1*dz1 > epsilon*epsilon) continue;

                                xyz pi = xyz(dx1,dy1,dz1);

                                for (int xj = -grid_num; xj <= grid_num; xj++){
                                    for (int yj = -grid_num; yj <= grid_num; yj++){
                                        for (int zj = -grid_num; zj <= grid_num; zj++){

                                            if ((xj ^ yj ^ zj) & 1) continue;

                                            double dx2 = xj*grid_length;
                                            double dy2 = yj*grid_length;
                                            double dz2 = zj*grid_length;

                                            if (dx2*dx2 + dy2*dy2 + dz2*dz2 > epsilon*epsilon) continue;

                                            xyz pk = xyz(dx2, dy2, P_tmp.col(k)[2] + dz2);

                                            double pi_pk_distance = Distance(pi, pk);

                                            if (abs(pi_pk_distance - Pi_Pk_distance) > grid_length/2.0) continue; 

                                            Eigen::Matrix4d transform = calculateTransformationMatrix(P_tmp.col(i), P_tmp.col(k), pi, pk);

                                            Eigen::Matrix3Xd P_transformed = transform.block<3, 3>(0, 0) * P_tmp + transform.block<3, 1>(0, 3).replicate(1, N);

                                            Eigen::Matrix3Xd R_tmp;

                                            // The following function, ComputeMax, is implemented such that the l-axis is aligned with the z-axis to simplify the implementation.
                                            int size_of_solution = ComputeMax(P_transformed, Q_tmp, (1.0+delta)*epsilon, R_tmp);
                                            if (size_of_solution > max_size_of_solution){
                                                max_size_of_solution = size_of_solution;
                                                res_P = P_transformed;
                                                res_Q = R_tmp*Q_tmp;
                                                R = R_tmp;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return max_size_of_solution;
}


struct Calculate_dp_with_binarysearch{
    std::vector<std::vector<int>> S;
    int N, M;

    void init(int n, int m){
        N = n;
        M = m;
        S = std::vector<std::vector<int>>(N);
    }

    void add_edge(int i, int j){
        S[i].push_back(j);
    }

    void delete_edge(int i, int j){
        for (int k=0; k<S[i].size(); k++){
            if (S[i][k] == j){
                S[i].erase(S[i].begin() + k);
                break;
            }
        }
    }

    int calculate(){
        std::vector<int> D(M+1, 1e9);
        D[0] = -1;
        for (int i=0; i<N; i++){
            std::vector<std::array<int, 2>> indexes;
            for (auto j : S[i]){
                int idx = std::lower_bound(D.begin(), D.end(), j) - D.begin();
                indexes.push_back({idx, j});
            }
            for (auto idx : indexes){
                D[idx[0]] = std::min(D[idx[0]], idx[1]);
            }
        }
        return std::lower_bound(D.begin(), D.end(), 1e9) - D.begin() - 1;
    }
};

void calculate_with_LIS_based_algorithm(std::vector<std::pair<double,int>> &angles, int N, int M, int &max_size_of_solution, double &max_theta) {

    int NM_angle_segments = angles.size();
    int cnt = 0;

    Calculate_dp_with_binarysearch calculate_dp_with_binary_search;
    calculate_dp_with_binary_search.init(N, M);

    for (int a = 0; a < NM_angle_segments-1; a++){
        int index = angles[a].second;
        if (index >= 0) {
            index--;
            int x = index / M;
            int y = index % M;
            calculate_dp_with_binary_search.add_edge(x, y);
        } else {
            index = -index - 1;
            int x = index / M;
            int y = index % M;
            calculate_dp_with_binary_search.delete_edge(x, y);
        }

        if (angles[a].first == angles[a+1].first) continue;

        cnt = calculate_dp_with_binary_search.calculate();

        if (cnt > max_size_of_solution){
            max_size_of_solution = cnt;
            max_theta = (angles[a].first + angles[a+1].first) / 2.0;
        }
    }

    return;
}

template <typename T>
void calculate_cnt(std::vector<std::pair<double,int>> &angles, int N, int M, int &max_size_of_solution, double &max_theta){
    
    int NM_angle_segments = angles.size();
    int cnt = 0;

    std::vector<T> connected(N, 0);

    for (int a = 0; a < NM_angle_segments-1; a++){
        int index = angles[a].second;
        if (index >= 0) {
            index--;
            int x = index / M;
            int y = index % M;
            connected[x] = connected[x] | T(1) << y;
        } else {
            index = -index - 1;
            int x = index / M;
            int y = index % M;
            connected[x] = connected[x] ^ T(1) << y;
        }

        if (angles[a].first == angles[a+1].first) continue;

        std::vector<T> D(2, 0);

        for (int i=0; i<N; i++){
            T x = D[0] | connected[i];
            D[1] = x & (((x-((D[0] << 1) | T(1))) ^ x));
            D[0] = D[1];
        }

        cnt = D[0].popcount();

        if (cnt > max_size_of_solution){
            max_size_of_solution = cnt;
            max_theta = (angles[a].first + angles[a+1].first) / 2.0;
        }
    }
    return;
}

void calculate_with_LCS_based_algorithm(std::vector<std::pair<double,int>> &angles, int N, int M, int &max_size_of_solution, double &max_theta) {

    int NM_angle_segments = angles.size();
    int cnt = 0;

    if (M <= 64){
        std::vector<unsigned long long> connected(N, 0);

        for (int a = 0; a < NM_angle_segments-1; a++){
            int index = angles[a].second;
            if (index >= 0) {
                index--;
                int x = index / M;
                int y = index % M;
                connected[x] |= 1ull << y;
            } else {
                index = -index - 1;
                int x = index / M;
                int y = index % M;
                connected[x] ^= 1ull << y;
            }

            if (angles[a].first == angles[a+1].first) continue;

            std::vector<unsigned long long> D(2, 0);

            for (int i=0; i<N; i++){
                unsigned long long x = D[0] | connected[i];
                D[1] = x & (((x-((D[0] << 1) | 1)) ^ x));
                D[0] = D[1];
            }

            cnt = __builtin_popcountll(D[0]);

            if (cnt > max_size_of_solution){
                max_size_of_solution = cnt;
                max_theta = (angles[a].first + angles[a+1].first) / 2.0;
            }
        }

        return;

    }else if (M <= 128){
        std::vector<__uint128_t> connected(N, 0);

        for (int a = 0; a < NM_angle_segments-1; a++){
            int index = angles[a].second;
            if (index >= 0) {
                index--;
                int x = index / M;
                int y = index % M;
                connected[x] |= __uint128_t(1) << y;
            } else {
                index = -index - 1;
                int x = index / M;
                int y = index % M;
                connected[x] ^= __uint128_t(1) << y;
            }

            if (angles[a].first == angles[a+1].first) continue;

            std::vector<__uint128_t> D(2, 0);

            for (int i=0; i<N; i++){
                __uint128_t x = D[0] | connected[i];
                D[1] = x & (((x-((D[0] << 1) | 1)) ^ x));
                D[0] = D[1];
            }

            cnt = __builtin_popcountll(D[0]) + __builtin_popcountll(D[0] >> 64);

            if (cnt > max_size_of_solution){
                max_size_of_solution = cnt;
                max_theta = (angles[a].first + angles[a+1].first) / 2.0;
            }
        }

        return;
    }else if (M <= 1024){
        if (M <= 192){
            calculate_cnt<int192t>(angles, N, M, max_size_of_solution, max_theta);
        }else if (M <= 256){
            calculate_cnt<int256t>(angles, N, M, max_size_of_solution, max_theta);
        }else if (M <= 320){
            calculate_cnt<int320t>(angles, N, M, max_size_of_solution, max_theta);
        }else if (M <= 384){
            calculate_cnt<int384t>(angles, N, M, max_size_of_solution, max_theta);
        }else if (M <= 448){
            calculate_cnt<int448t>(angles, N, M, max_size_of_solution, max_theta);
        }else if (M <= 512){
            calculate_cnt<int512t>(angles, N, M, max_size_of_solution, max_theta);
        }else if (M <= 576){
            calculate_cnt<int576t>(angles, N, M, max_size_of_solution, max_theta);
        }else if (M <= 640){
            calculate_cnt<int640t>(angles, N, M, max_size_of_solution, max_theta);
        }else if (M <= 704){
            calculate_cnt<int704t>(angles, N, M, max_size_of_solution, max_theta);
        }else if (M <= 768){
            calculate_cnt<int768t>(angles, N, M, max_size_of_solution, max_theta);
        }else if (M <= 832){
            calculate_cnt<int832t>(angles, N, M, max_size_of_solution, max_theta);
        }else if (M <= 896){
            calculate_cnt<int896t>(angles, N, M, max_size_of_solution, max_theta);
        }        
        else if (M <= 960){
            calculate_cnt<int960t>(angles, N, M, max_size_of_solution, max_theta);
        }        
        else if (M <= 1024){
            calculate_cnt<int1024t>(angles, N, M, max_size_of_solution, max_theta);
        }
        return;
    }else{
        std::cerr << "M is too large: " << M << std::endl;
        std::cerr << "This implementation does not support M larger than 1024." << std::endl;
        std::cerr << "Please use a different implementation." << std::endl;
        exit(0);
    }
}

// Function: ALignFastLIS or AlignFastLCS
// This function is almost the same as LiNgAlign_simplified, but it uses a different DP algorithm to calculate the maximum size of solution.
int AlignFast(std::string Algo, Eigen::Matrix3Xd& P, Eigen::Matrix3Xd& Q, double epsilon, double delta, Eigen::Matrix3Xd& res_P, Eigen::Matrix3Xd& res_Q) {

    int N = P.cols();
    int M = Q.cols();

    bool swapped = false;
    if (N > M){
        swapped = true;
        std::swap(N, M);
        Eigen::Matrix3Xd P_tmp = P;
        P = Q;
        Q = P_tmp;
    }
    
    const double grid_length = epsilon*delta/3.0;
    int grid_num = floor(3.0/delta-1e-6);

    long long sum = (long long)N*N*M*M*(2*grid_num+1)*(2*grid_num+1)*(2*grid_num+1);

    int cnt = 0;

    int max_size_of_solution = 0;

    Eigen::Matrix3Xd R;

    // The grid configuration employed in this function represents only one possible approach; various other constructions are also feasible.
    for (int i = 0; i < N; i++){
        for (int j = 0; j < M; j++){
            for (int k = 0; k < N; k++){
                for (int l = 0; l < M; l++){

                    if (i == k || j == l) continue;

                    double Pi_Pk_distance = Distance(P.col(i), P.col(k));
                    double Qj_Ql_distance = Distance(Q.col(j), Q.col(l));

                    if (abs(Pi_Pk_distance - Qj_Ql_distance) > 2.0*epsilon*(1.0+delta)) continue;

                    if (Pi_Pk_distance < 1e-6 || Qj_Ql_distance < 1e-6) continue;

                    Eigen::Matrix4d transformP = computeTransformationMatrixToZero(P.col(i), P.col(k));
                    Eigen::Matrix3Xd P_tmp = transformP.block<3, 3>(0, 0) * P + transformP.block<3, 1>(0, 3).replicate(1, N);

                    Eigen::Matrix4d transformQ = computeTransformationMatrixToZero(Q.col(j), Q.col(l));
                    Eigen::Matrix3Xd Q_tmp = transformQ.block<3, 3>(0, 0) * Q + transformQ.block<3, 1>(0, 3).replicate(1, M);
                    
                    for (int xi = -grid_num; xi <= grid_num; xi++){
                        for (int yi = -grid_num; yi <= grid_num; yi++){
                            for (int zi = -grid_num; zi <= grid_num; zi++){
                                
                                if ((xi ^ yi ^ zi) & 1) continue;

                                double dx1 = xi*grid_length;
                                double dy1 = yi*grid_length;
                                double dz1 = zi*grid_length;

                                if (dx1*dx1 + dy1*dy1 + dz1*dz1 > epsilon*epsilon) continue;

                                xyz pi = xyz(dx1,dy1,dz1);

                                for (int xj = -grid_num; xj <= grid_num; xj++){
                                    for (int yj = -grid_num; yj <= grid_num; yj++){
                                        for (int zj = -grid_num; zj <= grid_num; zj++){

                                            if ((xj ^ yj ^ zj) & 1) continue;

                                            double dx2 = xj*grid_length;
                                            double dy2 = yj*grid_length;
                                            double dz2 = zj*grid_length;

                                            if (dx2*dx2 + dy2*dy2 + dz2*dz2 > epsilon*epsilon) continue;

                                            xyz pk = xyz(dx2, dy2, P_tmp.col(k)[2] + dz2);

                                            double pi_pk_distance = Distance(pi, pk);

                                            if (abs(pi_pk_distance - Pi_Pk_distance) > grid_length/2.0) continue; 

                                            Eigen::Matrix4d transform = calculateTransformationMatrix(P_tmp.col(i), P_tmp.col(k), pi, pk);

                                            Eigen::Matrix3Xd P_transformed = transform.block<3, 3>(0, 0) * P_tmp + transform.block<3, 1>(0, 3).replicate(1, N);

                                            Eigen::Matrix3Xd R_tmp;

                                            // The following function, ComputeMax, is implemented such that the l-axis is aligned with the z-axis to simplify the implementation.
                                            int size_of_solution = ComputeMax(P_transformed, Q_tmp, (1.0+delta)*epsilon, R_tmp, Algo);
                                            if (size_of_solution > max_size_of_solution){
                                                max_size_of_solution = size_of_solution;
                                                res_P = P_transformed;
                                                res_Q = R_tmp*Q_tmp;
                                                R = R_tmp;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if (swapped) {
        Eigen::Matrix3Xd temp = res_P;
        res_P = res_Q;
        res_Q = temp;
    }

    return max_size_of_solution;
}


int solve_with_BasicAlign(Eigen::Matrix3Xd& P, Eigen::Matrix3Xd& Q, double epsilon, Eigen::Matrix3Xd& res_P, Eigen::Matrix3Xd& res_Q) {
    return BasicAlign_simplified(P, Q, epsilon/8.00000, res_P, res_Q);
}

int solve_with_LiNgAlign(Eigen::Matrix3Xd& P, Eigen::Matrix3Xd& Q, double epsilon, double delta, Eigen::Matrix3Xd& res_P, Eigen::Matrix3Xd& res_Q) {
    return LiNgAlign_simplified(P, Q, epsilon/(1.00000+delta), delta, res_P, res_Q);
}

int solve_with_AlignFastLIS(Eigen::Matrix3Xd& P, Eigen::Matrix3Xd& Q, double epsilon, double delta, Eigen::Matrix3Xd& res_P, Eigen::Matrix3Xd& res_Q) {
    return AlignFast("AlignFastLIS", P, Q, epsilon/(1.00000+delta), delta, res_P, res_Q);
}
int solve_with_AlignFastLCS(Eigen::Matrix3Xd& P, Eigen::Matrix3Xd& Q, double epsilon, double delta, Eigen::Matrix3Xd& res_P, Eigen::Matrix3Xd& res_Q) {
    return AlignFast("AlignFastLCS", P, Q, epsilon/(1.00000+delta), delta, res_P, res_Q);
}

int main(int argc, char* argv[]) {

    std::cerr << std::fixed << std::setprecision(6);
    std::cout << std::fixed << std::setprecision(6);

    argparse::ArgumentParser program("aligner");

    program.add_argument("path1")
        .help("pdb file path for the first structure");

    program.add_argument("path2")
        .help("pdb file path for the second structure");

    program.add_argument("--epsilon")
        .help("Epsilon value for the algorithm (more than 0.0)")
        .scan<'f', double>()
        .default_value(0.5);

    program.add_argument("--delta")
        .help("Delta value for the algorithm (more than 0.0 and at most 3.0)")
        .scan<'f', double>()
        .default_value(1.0);

    program.add_argument("--Algorithm")
        .help("Algorithm to use for alignment")
        .default_value(std::string("AlignFastLIS"))
        .action([](const std::string& value) {
            if (value == "LiNgAlign" || value == "BasicAlign" || value == "AlignFastLIS" || value == "AlignFastLCS") {
                return value;
            } else {
                throw std::runtime_error("Invalid algorithm specified. Use 'LiNgAlign' or 'BasicAlign' or 'AlignFastLIS' or 'AlignFastLCS'.");
            }
        });


    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    std::vector<xyz> coords1, coords2;

    double epsilon = program.get<double>("--epsilon");
    double delta = program.get<double>("--delta");

    std::string filename1 = program.get<std::string>("path1");
    std::string filename2 = program.get<std::string>("path2");

    ReadPDB(filename1, coords1);
    ReadPDB(filename2, coords2);

    std::cerr << "Number of CA atoms in " << filename1 << ": " << coords1.size() << std::endl;
    std::cerr << "Number of CA atoms in " << filename2 << ": " << coords2.size() << std::endl;
    std::cerr << "Epsilon value: " << epsilon << std::endl;
    std::cerr << "Delta value: " << delta << std::endl;

    int N = coords1.size();
    int M = coords2.size();

    Eigen::Matrix3Xd P(3, N);
    Eigen::Matrix3Xd Q(3, M);

    for (int i = 0; i < N; i++){
        P.col(i) = coords1[i];
    }

    for (int i = 0; i < M; i++){
        Q.col(i) = coords2[i];
    }

    Eigen::Matrix3Xd res_P = P, res_Q = Q;

    if (program.get<std::string>("--Algorithm") == "BasicAlign") {
        std::cerr << "Using BasicAlign algorithm." << std::endl;
        int solution_size = solve_with_BasicAlign(P, Q, epsilon, res_P, res_Q);
        std::cout << "Solution size with BasicAlign: " << solution_size << std::endl;
        std::cerr << "BasicAlign completed." << std::endl;
    }else if (program.get<std::string>("--Algorithm") == "LiNgAlign") {
        std::cerr << "Using LiNgAlign algorithm." << std::endl;
        int solution_size = solve_with_LiNgAlign(P, Q, epsilon, delta, res_P, res_Q);
        std::cout << "Solution size with LiNgAlign: " << solution_size << std::endl;
        std::cerr << "LiNgAlign completed." << std::endl;
    }else if (program.get<std::string>("--Algorithm") == "AlignFastLIS") {
        std::cerr << "Using AlignFastLIS algorithm." << std::endl;
        int solution_size = solve_with_AlignFastLIS(P, Q, epsilon, delta, res_P, res_Q);
        std::cout << "Solution size with AlignFastLIS: " << solution_size << std::endl;
        std::cerr << "AlignFastLIS completed." << std::endl;
    }else if (program.get<std::string>("--Algorithm") == "AlignFastLCS") {
        std::cerr << "Using AlignFastLCS algorithm." << std::endl;
        int solution_size = solve_with_AlignFastLCS(P, Q, epsilon, delta, res_P, res_Q);
        std::cout << "Solution size with AlignFastLCS: " << solution_size << std::endl;
        std::cerr << "AlignFastLCS completed." << std::endl;
    }else{
        std::cerr << "Unknown algorithm specified: " << program.get<std::string>("--Algorithm") << std::endl;
        return 1;
    }

    // Output the results to files
    std::filesystem::path exec_path = std::filesystem::canonical(argv[0]);

    std::ofstream output1(exec_path.parent_path() / "results" / "outputP.txt");
    std::ofstream output2(exec_path.parent_path() / "results" / "outputQ.txt");
    output1 << std::fixed << std::setprecision(6);
    output2 << std::fixed << std::setprecision(6);

    for (int i = 0; i < res_P.cols(); i++) {
        output1 << res_P(0, i) << " " << res_P(1, i) << " " << res_P(2, i) << std::endl;
    }

    for (int i = 0; i < res_Q.cols(); i++) {
        output2 << res_Q(0, i) << " " << res_Q(1, i) << " " << res_Q(2, i) << std::endl;
    }

    output1.close();
    output2.close();

    std::cerr << "Results are outputted to outputP.txt and outputQ.txt in the results directory." << std::endl;

    return 0;
}
