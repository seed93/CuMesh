#include "cumesh.h"
#include "dtypes.cuh"
#include <cub/cub.cuh>


namespace cumesh {


/**
 * Pack edge index and cost into a single uint64_t for atomic operations
 * Lower cost = higher priority (lower packed value)
 */
__device__ inline uint64_t pack_skinny_priority(int edge_idx, float cost) {
    unsigned int v = __float_as_uint(cost);
    return (static_cast<uint64_t>(v) << 32) |
           static_cast<unsigned int>(edge_idx);
}


/**
 * Compute quality metrics for each face
 * - min_angle: minimum angle in the triangle (radians)
 * - aspect_ratio: longest_edge / shortest_altitude
 * - shape_quality: 4*sqrt(3)*area / sum_of_edge_lengths_squared (1.0 for equilateral)
 * - shortest_edge_idx: index (0,1,2) of the shortest edge in the face
 */
static __global__ void compute_face_metrics_kernel(
    const float3* vertices,
    const int3* faces,
    const int F,
    float* min_angles,
    float* aspect_ratios,
    float* shape_qualities,
    int* shortest_edge_indices
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= F) return;
    
    const float EPS = 1e-12f;
    
    int3 f_vids = faces[tid];
    Vec3f a(vertices[f_vids.x]);
    Vec3f b(vertices[f_vids.y]);
    Vec3f c(vertices[f_vids.z]);
    
    // Edge vectors
    Vec3f e0 = b - a;  // edge opposite to vertex c (index 2)
    Vec3f e1 = c - b;  // edge opposite to vertex a (index 0)
    Vec3f e2 = a - c;  // edge opposite to vertex b (index 1)
    
    // Edge lengths
    float l0 = e0.norm();
    float l1 = e1.norm();
    float l2 = e2.norm();
    
    // Find shortest edge
    int shortest_idx = 0;
    float shortest_len = l0;
    if (l1 < shortest_len) { shortest_idx = 1; shortest_len = l1; }
    if (l2 < shortest_len) { shortest_idx = 2; shortest_len = l2; }
    shortest_edge_indices[tid] = shortest_idx;
    
    // Compute area using cross product
    Vec3f cross = e0.cross(e1 * -1.0f);
    float area2 = cross.norm();  // 2 * area
    float area = 0.5f * area2;
    
    // Shape quality: 4*sqrt(3)*area / (l0^2 + l1^2 + l2^2)
    float denom = l0*l0 + l1*l1 + l2*l2;
    float shape_quality = (denom < EPS) ? 0.0f : (4.0f * sqrtf(3.0f) * area / denom);
    shape_qualities[tid] = fminf(fmaxf(shape_quality, 0.0f), 1.0f);
    
    // Compute angles using law of cosines
    // cos(A) = (b^2 + c^2 - a^2) / (2*b*c) where A is angle at vertex opposite to edge a
    // Angle at vertex a (between edges e0 and -e2)
    float cos_a = (l0 > EPS && l2 > EPS) ? (-e0.dot(e2)) / (l0 * l2) : 1.0f;
    // Angle at vertex b (between edges -e0 and e1)
    float cos_b = (l0 > EPS && l1 > EPS) ? (-e0.dot(e1 * -1.0f)) / (l0 * l1) : 1.0f;
    // Angle at vertex c (between edges -e1 and e2)
    float cos_c = (l1 > EPS && l2 > EPS) ? (e1.dot(e2)) / (l1 * l2) : 1.0f;
    
    // Clamp cosines to valid range
    cos_a = fminf(fmaxf(cos_a, -1.0f), 1.0f);
    cos_b = fminf(fmaxf(cos_b, -1.0f), 1.0f);
    cos_c = fminf(fmaxf(cos_c, -1.0f), 1.0f);
    
    float angle_a = acosf(cos_a);
    float angle_b = acosf(cos_b);
    float angle_c = acosf(cos_c);
    
    min_angles[tid] = fminf(fminf(angle_a, angle_b), angle_c);
    
    // Aspect ratio: longest_edge / shortest_altitude
    // altitude = 2 * area / base_length
    float longest_edge = fmaxf(fmaxf(l0, l1), l2);
    float shortest_altitude = (longest_edge < EPS) ? 0.0f : (2.0f * area / longest_edge);
    aspect_ratios[tid] = (shortest_altitude < EPS) ? 1e6f : (longest_edge / shortest_altitude);
}


/**
 * Mark skinny faces based on combined criteria
 */
static __global__ void mark_skinny_faces_kernel(
    const float* min_angles,
    const float* aspect_ratios,
    const float* shape_qualities,
    const int F,
    const float min_angle_thresh,      // in radians
    const float aspect_ratio_thresh,
    const float shape_quality_thresh,
    uint8_t* is_skinny
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= F) return;
    
    bool skinny = (min_angles[tid] < min_angle_thresh) ||
                  (aspect_ratios[tid] > aspect_ratio_thresh) ||
                  (shape_qualities[tid] < shape_quality_thresh);
    
    is_skinny[tid] = skinny ? 1 : 0;
}


/**
 * Count skinny faces (reduction kernel)
 */
static __global__ void count_skinny_faces_kernel(
    const uint8_t* is_skinny,
    const int F,
    int* count
) {
    __shared__ int sdata[256];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (gid < F && is_skinny[gid]) ? 1 : 0;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(count, sdata[0]);
    }
}


/**
 * Check if collapsing an edge would flip any incident triangle normal
 */
inline __device__ bool check_collapse_valid(
    int tri_idx,
    int collapse_keep_vert,
    int collapse_other_vert,
    const float3* vertices,
    const int3* faces,
    const Vec3f& v_new
) {
    int3 f_vids = faces[tri_idx];
    
    // If this triangle contains the other vertex (shares the edge), it will be removed
    if (f_vids.x == collapse_other_vert || f_vids.y == collapse_other_vert || f_vids.z == collapse_other_vert)
        return true;  // Will be removed, no flip check needed
    
    Vec3f a(vertices[f_vids.x]);
    Vec3f b(vertices[f_vids.y]);
    Vec3f c(vertices[f_vids.z]);
    
    // Build new positions
    Vec3f na = (f_vids.x == collapse_keep_vert) ? v_new : a;
    Vec3f nb = (f_vids.y == collapse_keep_vert) ? v_new : b;
    Vec3f nc = (f_vids.z == collapse_keep_vert) ? v_new : c;
    
    // Compute old and new normals
    Vec3f old_e1 = b - a;
    Vec3f old_e2 = c - a;
    Vec3f old_normal = old_e1.cross(old_e2);
    
    Vec3f new_e1 = nb - na;
    Vec3f new_e2 = nc - na;
    Vec3f new_normal = new_e1.cross(new_e2);
    
    // Check for flip (normals pointing in opposite directions)
    return old_normal.dot(new_normal) >= 0.0f;
}


/**
 * Compute collapse cost for edges incident to skinny faces
 * Skinny-face edges get lower cost (higher priority)
 * Invalid collapses (would flip normals) get INFINITY cost
 */
static __global__ void get_skinny_collapse_cost_kernel(
    const float3* vertices,
    const int3* faces,
    const int* vert2face,
    const int* vert2face_offset,
    const uint64_t* edges,
    const uint8_t* is_skinny,
    const uint8_t* vert_is_boundary,
    const int* face_shortest_edge_idx,
    const int3* face2edge,
    const int V,
    const int F,
    const int E,
    float* edge_collapse_costs
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= E) return;
    
    // Get edge endpoints
    uint64_t e = edges[tid];
    int e0 = int(e >> 32);
    int e1 = int(e & 0xFFFFFFFF);
    
    // Compute new vertex position (midpoint, or boundary-aware)
    Vec3f v0(vertices[e0]);
    Vec3f v1(vertices[e1]);
    uint8_t v0_is_bound = vert_is_boundary[e0];
    uint8_t v1_is_bound = vert_is_boundary[e1];
    float w0 = 0.5f;
    if (v0_is_bound && !v1_is_bound) w0 = 1.0f;
    else if (!v0_is_bound && v1_is_bound) w0 = 0.0f;
    Vec3f v_new = v0 * w0 + v1 * (1.0f - w0);
    
    // Check if collapse is valid (no normal flips)
    for (int f = vert2face_offset[e0]; f < vert2face_offset[e0+1]; f++) {
        int tri_idx = vert2face[f];
        if (!check_collapse_valid(tri_idx, e0, e1, vertices, faces, v_new)) {
            edge_collapse_costs[tid] = INFINITY;
            return;
        }
    }
    for (int f = vert2face_offset[e1]; f < vert2face_offset[e1+1]; f++) {
        int tri_idx = vert2face[f];
        if (!check_collapse_valid(tri_idx, e1, e0, vertices, faces, v_new)) {
            edge_collapse_costs[tid] = INFINITY;
            return;
        }
    }
    
    // Check if this edge is incident to any skinny face AND is the shortest edge of that face
    bool is_skinny_edge = false;
    for (int f = vert2face_offset[e0]; f < vert2face_offset[e0+1]; f++) {
        int fid = vert2face[f];
        int3 f_vids = faces[fid];
        // Check if this face contains both endpoints (i.e., contains this edge)
        bool has_e1 = (f_vids.x == e1 || f_vids.y == e1 || f_vids.z == e1);
        if (has_e1 && is_skinny[fid]) {
            // Check if this is the shortest edge of this skinny face
            int3 f_edges = face2edge[fid];
            int shortest_local_idx = face_shortest_edge_idx[fid];
            int shortest_edge_id = (shortest_local_idx == 0) ? f_edges.x :
                                   (shortest_local_idx == 1) ? f_edges.y : f_edges.z;
            if (shortest_edge_id == tid) {
                is_skinny_edge = true;
                break;
            }
        }
    }
    
    if (is_skinny_edge) {
        // Prioritize by edge length (shorter = lower cost = higher priority)
        float edge_length = (v1 - v0).norm();
        edge_collapse_costs[tid] = edge_length;
    } else {
        // Non-skinny edges get high cost (won't be collapsed)
        edge_collapse_costs[tid] = INFINITY;
    }
}


/**
 * Propagate collapse priority to neighboring faces
 */
static __global__ void propagate_skinny_cost_kernel(
    const uint64_t* edges,
    const int* vert2face,
    const int* vert2face_offset,
    const float* edge_collapse_costs,
    const int V,
    const int F,
    const int E,
    uint64_t* propagated_costs
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= E) return;
    
    float cost = edge_collapse_costs[tid];
    if (isinf(cost)) return;  // Skip invalid edges
    
    uint64_t e = edges[tid];
    int e0 = int(e >> 32);
    int e1 = int(e & 0xFFFFFFFF);
    
    uint64_t pack = pack_skinny_priority(tid, cost);
    
    // Propagate to all neighboring faces of both endpoints
    for (int f = vert2face_offset[e0]; f < vert2face_offset[e0+1]; f++) {
        atomicMin(reinterpret_cast<unsigned long long*>(&propagated_costs[vert2face[f]]), 
                  static_cast<unsigned long long>(pack));
    }
    for (int f = vert2face_offset[e1]; f < vert2face_offset[e1+1]; f++) {
        atomicMin(reinterpret_cast<unsigned long long*>(&propagated_costs[vert2face[f]]), 
                  static_cast<unsigned long long>(pack));
    }
}


/**
 * Collapse skinny edges in parallel
 */
static __global__ void collapse_skinny_edges_kernel(
    float3* vertices,
    int3* faces,
    uint64_t* edges,
    const int* vert2face,
    const int* vert2face_offset,
    const float* edge_collapse_costs,
    const uint64_t* propagated_costs,
    const uint8_t* vert_is_boundary,
    const int V,
    const int F,
    const int E,
    int* vertices_kept,
    int* faces_kept
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= E) return;
    
    float cost = edge_collapse_costs[tid];
    if (isinf(cost)) return;  // Skip invalid edges
    
    uint64_t e = edges[tid];
    int e0 = int(e >> 32);
    int e1 = int(e & 0xFFFFFFFF);
    uint64_t pack = pack_skinny_priority(tid, cost);
    
    // Check if all neighboring faces agree this edge should be collapsed
    for (int f = vert2face_offset[e0]; f < vert2face_offset[e0+1]; f++) {
        if (propagated_costs[vert2face[f]] != pack) return;
    }
    for (int f = vert2face_offset[e1]; f < vert2face_offset[e1+1]; f++) {
        if (propagated_costs[vert2face[f]] != pack) return;
    }
    
    // Perform collapse
    Vec3f v0(vertices[e0]);
    Vec3f v1(vertices[e1]);
    uint8_t v0_is_bound = vert_is_boundary[e0];
    uint8_t v1_is_bound = vert_is_boundary[e1];
    float w0 = 0.5f;
    if (v0_is_bound && !v1_is_bound) w0 = 1.0f;
    else if (!v0_is_bound && v1_is_bound) w0 = 0.0f;
    Vec3f v_new = v0 * w0 + v1 * (1.0f - w0);
    
    vertices[e0] = { v_new.x, v_new.y, v_new.z };
    vertices_kept[e1] = 0;
    
    // Delete shared faces (those containing both e0 and e1)
    for (int f = vert2face_offset[e0]; f < vert2face_offset[e0+1]; f++) {
        int fid = vert2face[f];
        int3 f_vids = faces[fid];
        if (f_vids.x == e1 || f_vids.y == e1 || f_vids.z == e1) {
            faces_kept[fid] = 0;
        }
    }
    
    // Update faces to reference e0 instead of e1
    for (int f = vert2face_offset[e1]; f < vert2face_offset[e1+1]; f++) {
        int fid = vert2face[f];
        int3 f_vids = faces[fid];
        if (f_vids.x == e1) f_vids.x = e0;
        else if (f_vids.y == e1) f_vids.y = e0;
        else if (f_vids.z == e1) f_vids.z = e0;
        faces[fid] = f_vids;
    }
}


/**
 * Compress vertices after collapse
 */
static __global__ void compress_vertices_skinny_kernel(
    const int* vertices_map,
    const float3* old_vertices,
    const int V,
    float3* new_vertices
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= V) return;
    int new_id = vertices_map[tid];
    int is_kept = vertices_map[tid + 1] == new_id + 1;
    if (is_kept) {
        new_vertices[new_id] = old_vertices[tid];
    }
}


/**
 * Compress faces after collapse
 */
static __global__ void compress_faces_skinny_kernel(
    const int* faces_map,
    const int* vertices_map,
    const int3* old_faces,
    const int F,
    int3* new_faces
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= F) return;
    int new_id = faces_map[tid];
    int is_kept = faces_map[tid + 1] == new_id + 1;
    if (is_kept) {
        new_faces[new_id].x = vertices_map[old_faces[tid].x];
        new_faces[new_id].y = vertices_map[old_faces[tid].y];
        new_faces[new_id].z = vertices_map[old_faces[tid].z];
    }
}


/**
 * Main function to collapse skinny faces
 */
std::tuple<int, int> CuMesh::collapse_skinny_faces(float min_angle_deg, int max_iterations) {
    const float PI = 3.14159265358979323846f;
    float min_angle_rad = min_angle_deg * PI / 180.0f;
    float aspect_ratio_thresh = 100.0f;
    float shape_quality_thresh = 0.001f;
    
    size_t V = vertices.size;
    size_t F = faces.size;
    
    // Allocate metric buffers
    Buffer<float> face_min_angles;
    Buffer<float> face_aspect_ratios;
    Buffer<float> face_shape_qualities;
    Buffer<int> face_shortest_edge_idx;
    Buffer<uint8_t> face_is_skinny;
    Buffer<int> skinny_count;
    
    face_min_angles.resize(F);
    face_aspect_ratios.resize(F);
    face_shape_qualities.resize(F);
    face_shortest_edge_idx.resize(F);
    face_is_skinny.resize(F);
    skinny_count.resize(1);
    
    for (int iter = 0; iter < max_iterations; iter++) {
        V = vertices.size;
        F = faces.size;
        
        if (F == 0) break;
        
        // 1. Compute face metrics
        compute_face_metrics_kernel<<<(F+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
            vertices.ptr,
            faces.ptr,
            F,
            face_min_angles.ptr,
            face_aspect_ratios.ptr,
            face_shape_qualities.ptr,
            face_shortest_edge_idx.ptr
        );
        CUDA_CHECK(cudaGetLastError());
        
        // 2. Mark skinny faces
        mark_skinny_faces_kernel<<<(F+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
            face_min_angles.ptr,
            face_aspect_ratios.ptr,
            face_shape_qualities.ptr,
            F,
            min_angle_rad,
            aspect_ratio_thresh,
            shape_quality_thresh,
            face_is_skinny.ptr
        );
        CUDA_CHECK(cudaGetLastError());
        
        // 3. Count skinny faces
        skinny_count.zero();
        count_skinny_faces_kernel<<<(F+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
            face_is_skinny.ptr,
            F,
            skinny_count.ptr
        );
        CUDA_CHECK(cudaGetLastError());
        
        int num_skinny;
        CUDA_CHECK(cudaMemcpy(&num_skinny, skinny_count.ptr, sizeof(int), cudaMemcpyDeviceToHost));
        
        if (num_skinny == 0) break;
        
        // 4. Build connectivity if needed
        this->get_vertex_face_adjacency();
        this->get_edges();
        this->get_edge_face_adjacency();
        this->get_boundary_info();
        
        size_t E = edges.size;
        
        // 5. Compute collapse costs
        edge_collapse_costs.resize(E);
        get_skinny_collapse_cost_kernel<<<(E+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
            vertices.ptr,
            faces.ptr,
            vert2face.ptr,
            vert2face_offset.ptr,
            edges.ptr,
            face_is_skinny.ptr,
            vert_is_boundary.ptr,
            face_shortest_edge_idx.ptr,
            face2edge.ptr,
            V, F, E,
            edge_collapse_costs.ptr
        );
        CUDA_CHECK(cudaGetLastError());
        
        // 6. Propagate costs
        propagated_costs.resize(F);
        propagated_costs.fill(std::numeric_limits<uint64_t>::max());
        propagate_skinny_cost_kernel<<<(E+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
            edges.ptr,
            vert2face.ptr,
            vert2face_offset.ptr,
            edge_collapse_costs.ptr,
            V, F, E,
            propagated_costs.ptr
        );
        CUDA_CHECK(cudaGetLastError());
        
        // 7. Collapse edges
        vertices_map.resize(V + 1);
        faces_map.resize(F + 1);
        vertices_map.fill(1);
        faces_map.fill(1);
        
        collapse_skinny_edges_kernel<<<(E+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
            vertices.ptr,
            faces.ptr,
            edges.ptr,
            vert2face.ptr,
            vert2face_offset.ptr,
            edge_collapse_costs.ptr,
            propagated_costs.ptr,
            vert_is_boundary.ptr,
            V, F, E,
            vertices_map.ptr,
            faces_map.ptr
        );
        CUDA_CHECK(cudaGetLastError());
        
        // 8. Compress vertices
        size_t temp_storage_bytes = 0;
        CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
            nullptr, temp_storage_bytes,
            vertices_map.ptr, V+1
        ));
        cub_temp_storage.resize(temp_storage_bytes);
        CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
            cub_temp_storage.ptr, temp_storage_bytes,
            vertices_map.ptr, V+1
        ));
        
        int new_num_vertices;
        CUDA_CHECK(cudaMemcpy(&new_num_vertices, vertices_map.ptr + V, sizeof(int), cudaMemcpyDeviceToHost));
        
        temp_storage.resize(new_num_vertices * sizeof(float3));
        compress_vertices_skinny_kernel<<<(V+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
            vertices_map.ptr,
            vertices.ptr,
            V,
            reinterpret_cast<float3*>(temp_storage.ptr)
        );
        CUDA_CHECK(cudaGetLastError());
        swap_buffers(temp_storage, vertices);
        temp_storage.free();
        
        // 9. Compress faces
        CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
            nullptr, temp_storage_bytes,
            faces_map.ptr, F+1
        ));
        cub_temp_storage.resize(temp_storage_bytes);
        CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
            cub_temp_storage.ptr, temp_storage_bytes,
            faces_map.ptr, F+1
        ));
        
        int new_num_faces;
        CUDA_CHECK(cudaMemcpy(&new_num_faces, faces_map.ptr + F, sizeof(int), cudaMemcpyDeviceToHost));
        
        temp_storage.resize(new_num_faces * sizeof(int3));
        compress_faces_skinny_kernel<<<(F+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
            faces_map.ptr,
            vertices_map.ptr,
            faces.ptr,
            F,
            reinterpret_cast<int3*>(temp_storage.ptr)
        );
        CUDA_CHECK(cudaGetLastError());
        swap_buffers(temp_storage, faces);
        
        // Clear cache since mesh changed
        this->clear_cache();
        
        // Check if any collapses happened
        if (new_num_faces == (int)F) break;
        
        // Resize metric buffers for next iteration
        F = new_num_faces;
        face_min_angles.resize(F);
        face_aspect_ratios.resize(F);
        face_shape_qualities.resize(F);
        face_shortest_edge_idx.resize(F);
        face_is_skinny.resize(F);
    }
    
    // Clean up
    face_min_angles.free();
    face_aspect_ratios.free();
    face_shape_qualities.free();
    face_shortest_edge_idx.free();
    face_is_skinny.free();
    skinny_count.free();
    
    return std::make_tuple(vertices.size, faces.size);
}


} // namespace cumesh

