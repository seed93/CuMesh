#include "cumesh.h"
#include "dtypes.cuh"
#include <cub/cub.cuh>
#include <cmath>


namespace cumesh {


// ============================================================================
// Helper: pack/unpack key-value for atomicMin conflict resolution
// ============================================================================

__device__ inline uint64_t sq_pack_key_value(int key, float value) {
    unsigned int v = __float_as_uint(value);
    return (static_cast<uint64_t>(v) << 32) |
           static_cast<unsigned int>(key);
}


__device__ inline void sq_unpack_key_value(uint64_t key_value, int& key, float& value) {
    key = static_cast<int>(key_value & 0xffffffffu);
    value = __uint_as_float(static_cast<unsigned int>(key_value >> 32));
}


// ============================================================================
// Kernel 1: Initialize per-vertex QEM (MeshLab-quality)
//
// Ports InitQuadric from tri_edge_collapse_quadric.h:
// - Face plane quadrics (optionally area-weighted)
// - Boundary edge quadrics (weighted by BoundaryQuadricWeight)
// - Quality quadrics for all edges (if QualityQuadric enabled)
// ============================================================================

static __global__ void init_quadric_kernel(
    const float3* vertices,
    const int3* faces,
    const int* vert2face,
    const int* vert2face_offset,
    const uint64_t* edges,
    const int* edge2face_cnt,
    const int* vert2edge,
    const int* vert2edge_offset,
    const int V,
    const int F,
    const int E,
    const bool UseArea,
    const bool QualityQuadric,
    const float BoundaryQuadricWeight,
    const float QualityQuadricWeight,
    QEM* qems
) {
    const int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= V) return;

    QEM v_qem;

    // --- Accumulate face plane quadrics ---
    for (int fi = vert2face_offset[vid]; fi < vert2face_offset[vid + 1]; fi++) {
        int fid = vert2face[fi];
        int3 f_vids = faces[fid];
        Vec3f v0(vertices[f_vids.x]);
        Vec3f v1(vertices[f_vids.y]);
        Vec3f v2(vertices[f_vids.z]);

        Vec3f e1 = v1 - v0;
        Vec3f e2 = v2 - v0;
        Vec3f dirArea = e1.cross(e2);  // unnormalized normal, magnitude = 2*area
        float area = dirArea.norm();

        if (area < 1e-20f) continue;

        // Normalized normal
        Vec3f n = dirArea / area;
        float d = -(n.dot(v0));  // plane offset: n . x + d = 0

        // Build face plane quadric
        QEM face_qem;
        float4 plane = { n.x, n.y, n.z, d };

        // If UseArea, weight by area (area here = |cross product| = 2 * triangle area)
        if (UseArea) {
            // The VCG code weights by |dirArea| which is 2*area
            face_qem.add_plane(plane);
            // Scale by area: multiply each element
            #pragma unroll
            for (int k = 0; k < 10; k++) face_qem.e[k] *= area;
        } else {
            face_qem.add_plane(plane);
        }

        v_qem += face_qem;

        // --- Boundary / Quality quadrics for edges of this face incident to vid ---
        // For each edge of this face that includes vid, check if it's boundary or quality
        int local_idx = -1;
        if (f_vids.x == vid) local_idx = 0;
        else if (f_vids.y == vid) local_idx = 1;
        else if (f_vids.z == vid) local_idx = 2;

        // We process edges (vid, next_vertex) for edges starting from vid in this face
        // Edge (vid, V_next) and edge (vid, V_prev)
        // To avoid double-counting: only process edge j if the face's vertex at j == vid
        // Then edge j -> (V(j), V((j+1)%3))
        if (local_idx >= 0) {
            // Process the two edges of this face that include vid
            for (int ei = 0; ei < 2; ei++) {
                int other_local;
                if (ei == 0) other_local = (local_idx + 1) % 3;
                else other_local = (local_idx + 2) % 3;

                int other_vid;
                if (other_local == 0) other_vid = f_vids.x;
                else if (other_local == 1) other_vid = f_vids.y;
                else other_vid = f_vids.z;

                // Check if this edge is a boundary edge
                // Find the edge in the edge list via vert2edge
                bool is_boundary_edge = false;
                for (int ve = vert2edge_offset[vid]; ve < vert2edge_offset[vid + 1]; ve++) {
                    int eid = vert2edge[ve];
                    uint64_t edge = edges[eid];
                    int ea = int(edge >> 32);
                    int eb = int(edge & 0xFFFFFFFF);
                    if ((ea == vid && eb == other_vid) || (ea == other_vid && eb == vid)) {
                        if (edge2face_cnt[eid] == 1) {
                            is_boundary_edge = true;
                        }
                        break;
                    }
                }

                if (is_boundary_edge || QualityQuadric) {
                    // Build border plane: orthogonal to face, passing through edge
                    Vec3f edge_dir;
                    Vec3f v_this(vertices[vid]);
                    Vec3f v_other(vertices[other_vid]);
                    edge_dir = (v_other - v_this).normalized();

                    // borderPlane direction = faceNormal x edgeDirection
                    Vec3f border_dir = n.cross(edge_dir);
                    float border_dir_len = border_dir.norm();
                    if (border_dir_len < 1e-20f) continue;
                    border_dir /= border_dir_len;

                    float weight;
                    if (is_boundary_edge)
                        weight = BoundaryQuadricWeight;
                    else
                        weight = QualityQuadricWeight;

                    border_dir *= weight;
                    float border_d = -(border_dir.dot(v_this));

                    QEM border_qem;
                    border_qem.add_plane({ border_dir.x, border_dir.y, border_dir.z, border_d });
                    v_qem += border_qem;
                }
            }
        }
    }

    qems[vid] = v_qem;
}


// ============================================================================
// Host wrapper: Initialize vertex QEMs
// ============================================================================

static void init_quadric(CuMesh& ctx) {
    size_t V = ctx.vertices.size;
    size_t F = ctx.faces.size;
    size_t E = ctx.edges.size;
    const auto& params = ctx.simplify_quadric_params;

    // Allocate QEM buffer (initialized once, then accumulated across collapses)
    ctx.vertex_qems.resize(V * sizeof(QEM));

    init_quadric_kernel<<<(V + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        ctx.vertices.ptr,
        ctx.faces.ptr,
        ctx.vert2face.ptr,
        ctx.vert2face_offset.ptr,
        ctx.edges.ptr,
        ctx.edge2face_cnt.ptr,
        ctx.vert2edge.ptr,
        ctx.vert2edge_offset.ptr,
        V, F, E,
        params.UseArea,
        params.QualityQuadric,
        params.BoundaryQuadricWeight,
        params.QualityQuadricWeight,
        reinterpret_cast<QEM*>(ctx.vertex_qems.ptr)
    );
    CUDA_CHECK(cudaGetLastError());
}


// ============================================================================
// Helper device function: compute face quality (shape metric)
// quality = 4 * sqrt(3) * area / (a^2 + b^2 + c^2)
// where area is the triangle area, a, b, c are edge lengths
// Returns 1.0 for equilateral, 0.0 for degenerate
// ============================================================================

__device__ inline float compute_face_quality(
    const Vec3f& p0, const Vec3f& p1, const Vec3f& p2
) {
    Vec3f e0 = p1 - p0;
    Vec3f e1 = p2 - p1;
    Vec3f e2 = p0 - p2;
    Vec3f cross = e0.cross(Vec3f(p2.x - p0.x, p2.y - p0.y, p2.z - p0.z));
    float area2 = cross.norm();  // 2 * area
    float denom = e0.norm2() + e1.norm2() + e2.norm2();
    if (denom < 1e-20f) return 0.0f;
    // 4 * sqrt(3) * (area) / (sum of edge^2) = 2*sqrt(3) * area2 / denom
    return 2.0f * 1.7320508f * area2 / denom;
}


// ============================================================================
// Link condition check for topology preservation (GPU port of VCG LinkConditions)
//
// For edge (e0, e1), checks:
// 1. No shared edges between the 1-rings of e0 and e1 (would create non-manifold)
// 2. Number of shared 1-ring vertices equals the link of the edge
// Returns true if collapse is topologically safe.
// ============================================================================

__device__ inline bool check_link_conditions_gpu(
    int e0, int e1,
    const int3* faces,
    const int* vert2face,
    const int* vert2face_offset
) {
    const int MAX_RING = 48;

    // --- Step 1: Build unique 1-ring vertex sets and count link vertices ---
    int ring0[MAX_RING], ring0_n = 0;
    int ring1[MAX_RING], ring1_n = 0;
    int link_cnt = 0;       // |Lk(e0,e1)| = opposite vertices in shared faces

    // Build ring0: unique 1-ring vertices of e0
    for (int fi = vert2face_offset[e0]; fi < vert2face_offset[e0 + 1]; fi++) {
        int fid = vert2face[fi];
        int3 f = faces[fid];
        // Find the two non-e0 vertices
        int a, b;
        if      (f.x == e0) { a = f.y; b = f.z; }
        else if (f.y == e0) { a = f.x; b = f.z; }
        else                { a = f.x; b = f.y; }

        // Check if this face is shared (contains e1)
        if (a == e1 || b == e1) {
            link_cnt++;
            // (don't skip -- still add vertices to ring)
        }

        // Add a to ring0 (unique)
        {
            bool found = false;
            for (int k = 0; k < ring0_n; k++) { if (ring0[k] == a) { found = true; break; } }
            if (!found && ring0_n < MAX_RING) ring0[ring0_n++] = a;
        }
        // Add b to ring0 (unique)
        {
            bool found = false;
            for (int k = 0; k < ring0_n; k++) { if (ring0[k] == b) { found = true; break; } }
            if (!found && ring0_n < MAX_RING) ring0[ring0_n++] = b;
        }
    }

    // Build ring1: unique 1-ring vertices of e1
    for (int fi = vert2face_offset[e1]; fi < vert2face_offset[e1 + 1]; fi++) {
        int fid = vert2face[fi];
        int3 f = faces[fid];
        int a, b;
        if      (f.x == e1) { a = f.y; b = f.z; }
        else if (f.y == e1) { a = f.x; b = f.z; }
        else                { a = f.x; b = f.y; }

        {
            bool found = false;
            for (int k = 0; k < ring1_n; k++) { if (ring1[k] == a) { found = true; break; } }
            if (!found && ring1_n < MAX_RING) ring1[ring1_n++] = a;
        }
        {
            bool found = false;
            for (int k = 0; k < ring1_n; k++) { if (ring1[k] == b) { found = true; break; } }
            if (!found && ring1_n < MAX_RING) ring1[ring1_n++] = b;
        }
    }

    // Overflow guard: if valence exceeds MAX_RING, conservatively reject
    if (ring0_n >= MAX_RING || ring1_n >= MAX_RING) return false;

    // --- Step 2: Boundary handling (VCG dummy vertex trick) ---
    // A vertex v is on boundary if one of its 1-ring vertices appears only once
    // (i.e., the fan around v is open). For the link condition, a boundary edge
    // gets an extra virtual link vertex.
    // Simple check: if link_cnt == 1, the edge is on the boundary → add 1
    if (link_cnt == 1) {
        link_cnt++;  // virtual boundary closure vertex
    }

    // --- Step 3: Check shared edges (would create non-manifold after collapse) ---
    // An edge (a,b) is shared if there exist faces (e0,a,b) and (e1,a,b).
    for (int fi0 = vert2face_offset[e0]; fi0 < vert2face_offset[e0 + 1]; fi0++) {
        int fid0 = vert2face[fi0];
        int3 f0 = faces[fid0];
        int a0, b0;
        if      (f0.x == e0) { a0 = f0.y; b0 = f0.z; }
        else if (f0.y == e0) { a0 = f0.x; b0 = f0.z; }
        else                 { a0 = f0.x; b0 = f0.y; }

        // Skip shared faces (containing e1)
        if (a0 == e1 || b0 == e1) continue;

        // Check if edge (a0, b0) also appears as an opposite edge in e1's ring
        for (int fi1 = vert2face_offset[e1]; fi1 < vert2face_offset[e1 + 1]; fi1++) {
            int fid1 = vert2face[fi1];
            int3 f1 = faces[fid1];
            int a1, b1;
            if      (f1.x == e1) { a1 = f1.y; b1 = f1.z; }
            else if (f1.y == e1) { a1 = f1.x; b1 = f1.z; }
            else                 { a1 = f1.x; b1 = f1.y; }

            if (a1 == e0 || b1 == e0) continue;

            // Check if edge (a0,b0) == edge (a1,b1)
            if ((a0 == a1 && b0 == b1) || (a0 == b1 && b0 == a1)) {
                return false; // shared edge found → topologically unsafe
            }
        }
    }

    // --- Step 4: Count shared vertices (Lk(e0) ∩ Lk(e1), excluding e0 and e1) ---
    int shared_cnt = 0;
    for (int i = 0; i < ring0_n; i++) {
        if (ring0[i] == e0 || ring0[i] == e1) continue;
        for (int j = 0; j < ring1_n; j++) {
            if (ring0[i] == ring1[j]) {
                shared_cnt++;
                break;
            }
        }
    }

    // --- Step 5: Link condition: |shared vertices| must equal |link| ---
    return shared_cnt == link_cnt;
}


// ============================================================================
// Kernel 2: Compute edge collapse cost (MeshLab-quality)
//
// For each edge, computes:
// - Optimal position via quadric minimization
// - QEM error at optimal position
// - Quality check, normal check, area check, hard checks
// - Link condition check (PreserveTopology)
// - Stores cost and optimal position
// ============================================================================

static __global__ void compute_edge_cost_quadric_kernel(
    const float3* vertices,
    const int3* faces,
    const int* vert2face,
    const int* vert2face_offset,
    const uint64_t* edges,
    const uint8_t* vert_is_boundary,
    const QEM* qems,
    const int V,
    const int F,
    const int E,
    // Parameters
    const bool OptimalPlacement,
    const float QuadricEpsilon,
    const float ScaleFactor,
    const bool QualityCheck,
    const float QualityThr,
    const bool NormalCheck,
    const float CosineThr,
    const bool AreaCheck,
    const bool HardQualityCheck,
    const float HardQualityThr,
    const bool HardNormalCheck,
    const bool FastPreserveBoundary,
    const bool PreserveBoundary,
    const bool PreserveTopology,
    // Outputs
    float* edge_collapse_costs,
    float3* optimal_positions
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= E) return;

    // Get edge vertices
    uint64_t e = edges[tid];
    int e0 = int(e >> 32);
    int e1 = int(e & 0xFFFFFFFF);

    Vec3f v0(vertices[e0]);
    Vec3f v1(vertices[e1]);
    uint8_t v0_is_bound = vert_is_boundary[e0];
    uint8_t v1_is_bound = vert_is_boundary[e1];

    // Boundary preservation: reject collapse if endpoint is boundary
    if (FastPreserveBoundary || PreserveBoundary) {
        if (v0_is_bound || v1_is_bound) {
            edge_collapse_costs[tid] = INFINITY;
            optimal_positions[tid] = { (v0.x + v1.x) * 0.5f, (v0.y + v1.y) * 0.5f, (v0.z + v1.z) * 0.5f };
            return;
        }
    }

    // Topology preservation: check link conditions
    if (PreserveTopology) {
        if (!check_link_conditions_gpu(e0, e1, faces, vert2face, vert2face_offset)) {
            edge_collapse_costs[tid] = INFINITY;
            optimal_positions[tid] = { (v0.x + v1.x) * 0.5f, (v0.y + v1.y) * 0.5f, (v0.z + v1.z) * 0.5f };
            return;
        }
    }

    // ---- Compute optimal position ----
    Vec3f newPos;
    Vec3f midpoint = (v0 + v1) * 0.5f;
    newPos = midpoint;

    if (!OptimalPlacement) {
        // No optimal placement: use v1 (the surviving vertex convention)
        newPos = v1;
    } else {
        // Check if midpoint quadric error is significant
        QEM q = qems[e0] + qems[e1];
        double midErr = q.evaluate(midpoint);

        if (midErr > 2.0 * (double)QuadricEpsilon) {
            // Try to solve for optimal
            float3 opt;
            float optErr;
            bool solved = q.solve_optimal(opt, optErr);
            if (solved) {
                newPos = Vec3f(opt.x, opt.y, opt.z);
            }
            // If not solved, keep midpoint
        }
    }

    // Boundary-aware position: if one vertex is boundary, bias towards it
    if (v0_is_bound && !v1_is_bound) {
        newPos = v0;
    } else if (!v0_is_bound && v1_is_bound) {
        newPos = v1;
    }

    // Store optimal position
    optimal_positions[tid] = { newPos.x, newPos.y, newPos.z };

    // ---- Collect original normals (for NormalCheck) ----
    // ---- Collect original area (for AreaCheck) ----
    // ---- Collect original quality (for HardQualityCheck) ----

    float origArea = 0.0f;
    float origQual = 1e30f;

    // Simulate collapse: compute metrics with both v0, v1 moved to newPos
    float newMinCos = 1e30f;   // minimum cosine of normal change
    float newMinQual = 1e30f;  // minimum quality after collapse
    float newArea = 0.0f;

    // Process faces around v0
    for (int fi = vert2face_offset[e0]; fi < vert2face_offset[e0 + 1]; fi++) {
        int fid = vert2face[fi];
        int3 f_vids = faces[fid];

        Vec3f fa(vertices[f_vids.x]);
        Vec3f fb(vertices[f_vids.y]);
        Vec3f fc(vertices[f_vids.z]);

        // Collect original metrics
        if (AreaCheck) {
            Vec3f oe1 = fb - fa;
            Vec3f oe2 = fc - fa;
            origArea += oe1.cross(oe2).norm();  // 2 * area
        }
        if (HardQualityCheck) {
            origQual = fminf(origQual, compute_face_quality(fa, fb, fc));
        }

        // Skip degenerate faces (shared by both v0 and v1 -- will be deleted)
        bool has_e1 = (f_vids.x == e1 || f_vids.y == e1 || f_vids.z == e1);
        if (has_e1) continue;

        // Compute old normal
        Vec3f old_e1 = fb - fa;
        Vec3f old_e2 = fc - fa;
        Vec3f old_normal = old_e1.cross(old_e2);
        float old_len = old_normal.norm();

        // Build new positions: replace v0 with newPos
        Vec3f na = (f_vids.x == e0) ? newPos : fa;
        Vec3f nb = (f_vids.y == e0) ? newPos : fb;
        Vec3f nc = (f_vids.z == e0) ? newPos : fc;

        Vec3f new_e1 = nb - na;
        Vec3f new_e2 = nc - na;
        Vec3f new_normal = new_e1.cross(new_e2);
        float new_len = new_normal.norm();

        // Quality check
        if (QualityCheck || HardQualityCheck) {
            float qual = compute_face_quality(na, nb, nc);
            newMinQual = fminf(newMinQual, qual);
        }

        // Normal check
        if (NormalCheck || HardNormalCheck) {
            if (old_len > 1e-20f && new_len > 1e-20f) {
                float cos_angle = old_normal.dot(new_normal) / (old_len * new_len);
                newMinCos = fminf(newMinCos, cos_angle);
            }
        }

        // Area
        if (AreaCheck) {
            newArea += new_len;  // 2 * area
        }
    }

    // Process faces around v1
    for (int fi = vert2face_offset[e1]; fi < vert2face_offset[e1 + 1]; fi++) {
        int fid = vert2face[fi];
        int3 f_vids = faces[fid];

        Vec3f fa(vertices[f_vids.x]);
        Vec3f fb(vertices[f_vids.y]);
        Vec3f fc(vertices[f_vids.z]);

        // Skip faces shared with v0 (already counted in origArea, and will be deleted)
        bool has_e0 = (f_vids.x == e0 || f_vids.y == e0 || f_vids.z == e0);

        if (!has_e0) {
            // Collect original metrics
            if (AreaCheck) {
                Vec3f oe1 = fb - fa;
                Vec3f oe2 = fc - fa;
                origArea += oe1.cross(oe2).norm();
            }
            if (HardQualityCheck) {
                origQual = fminf(origQual, compute_face_quality(fa, fb, fc));
            }
        }

        if (has_e0) continue;  // shared face, will be deleted

        // Compute old normal
        Vec3f old_e1 = fb - fa;
        Vec3f old_e2 = fc - fa;
        Vec3f old_normal = old_e1.cross(old_e2);
        float old_len = old_normal.norm();

        // Build new positions: replace v1 with newPos
        Vec3f na = (f_vids.x == e1) ? newPos : fa;
        Vec3f nb = (f_vids.y == e1) ? newPos : fb;
        Vec3f nc = (f_vids.z == e1) ? newPos : fc;

        Vec3f new_e1 = nb - na;
        Vec3f new_e2 = nc - na;
        Vec3f new_normal = new_e1.cross(new_e2);
        float new_len = new_normal.norm();

        if (QualityCheck || HardQualityCheck) {
            float qual = compute_face_quality(na, nb, nc);
            newMinQual = fminf(newMinQual, qual);
        }

        if (NormalCheck || HardNormalCheck) {
            if (old_len > 1e-20f && new_len > 1e-20f) {
                float cos_angle = old_normal.dot(new_normal) / (old_len * new_len);
                newMinCos = fminf(newMinCos, cos_angle);
            }
        }

        if (AreaCheck) {
            newArea += new_len;
        }
    }

    // ---- Compute QEM error (double precision, matching MeshLab) ----
    QEM qq = qems[e0] + qems[e1];
    double QuadErr_d = (double)ScaleFactor * qq.evaluate(newPos);

    // Clamp quality
    if (newMinQual > QualityThr) newMinQual = QualityThr;

    // Normal: transform MinCos to 0..1 range (0 = very bad, 1 = perfect)
    float MinCosNorm = 1.0f;
    if (NormalCheck) {
        if (newMinCos > CosineThr) newMinCos = CosineThr;
        MinCosNorm = fabsf((newMinCos + 1.0f) / 2.0f);
        if (MinCosNorm < 1e-10f) MinCosNorm = 1e-10f;
    }

    // Ensure QuadErr >= epsilon (double precision preserves the tiebreaker)
    // With double QEMs, flat-area error is ~1e-30 (well below eps=1e-15),
    // so the edge-length tiebreaker activates properly (matching MeshLab)
    QuadErr_d = fmax(QuadErr_d, (double)QuadricEpsilon);
    if (QuadErr_d <= (double)QuadricEpsilon) {
        // Tie-break by edge length: short edges in flat areas collapse first
        QuadErr_d *= (double)(v1 - v0).norm();
    }
    float QuadErr = (float)QuadErr_d;

    // ---- Combine cost ----
    float error;
    if (!QualityCheck && !NormalCheck) error = QuadErr;
    else if (QualityCheck && !NormalCheck) error = QuadErr / newMinQual;
    else if (!QualityCheck && NormalCheck) error = QuadErr / MinCosNorm;
    else error = QuadErr / (newMinQual * MinCosNorm);

    // ---- Hard checks: reject by setting cost = INF ----
    // Area check
    if (AreaCheck) {
        float totalArea = origArea + newArea;
        if (totalArea > 1e-20f && fabsf(origArea - newArea) / totalArea > 0.01f) {
            error = INFINITY;
        }
    }

    // Hard quality check
    if (HardQualityCheck) {
        if (newMinQual < HardQualityThr && newMinQual < origQual * 0.9f) {
            error = INFINITY;
        }
    }

    // Hard normal check (flip detection): reject if any normal flips
    if (HardNormalCheck) {
        if (newMinCos < 0.0f) {
            error = INFINITY;
        }
    }

    edge_collapse_costs[tid] = error;
}


// ============================================================================
// Host wrapper: Compute edge collapse costs
// ============================================================================

static void compute_edge_cost_quadric(CuMesh& ctx) {
    size_t V = ctx.vertices.size;
    size_t F = ctx.faces.size;
    size_t E = ctx.edges.size;
    const auto& params = ctx.simplify_quadric_params;

    ctx.edge_collapse_costs.resize(E);
    ctx.quadric_optimal_positions.resize(E);

    compute_edge_cost_quadric_kernel<<<(E + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        ctx.vertices.ptr,
        ctx.faces.ptr,
        ctx.vert2face.ptr,
        ctx.vert2face_offset.ptr,
        ctx.edges.ptr,
        ctx.vert_is_boundary.ptr,
        reinterpret_cast<const QEM*>(ctx.vertex_qems.ptr),
        V, F, E,
        // Parameters
        params.OptimalPlacement,
        params.QuadricEpsilon,
        params.ScaleFactor,
        params.QualityCheck,
        params.QualityThr,
        params.NormalCheck,
        params.CosineThr,
        params.AreaCheck,
        params.HardQualityCheck,
        params.HardQualityThr,
        params.HardNormalCheck,
        params.FastPreserveBoundary,
        params.PreserveBoundary,
        params.PreserveTopology,
        // Outputs
        ctx.edge_collapse_costs.ptr,
        ctx.quadric_optimal_positions.ptr
    );
    CUDA_CHECK(cudaGetLastError());
}


// ============================================================================
// Kernel 3: Propagate cost to neighboring faces (reuse pattern from simplify.cu)
// ============================================================================

static __global__ void propagate_cost_quadric_kernel(
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

    uint64_t e = edges[tid];
    int e0 = int(e >> 32);
    int e1 = int(e & 0xFFFFFFFF);

    uint64_t cost = sq_pack_key_value(tid, edge_collapse_costs[tid]);

    for (int f = vert2face_offset[e0]; f < vert2face_offset[e0 + 1]; f++) {
        atomicMin(reinterpret_cast<unsigned long long*>(&propagated_costs[vert2face[f]]),
                  static_cast<unsigned long long>(cost));
    }
    for (int f = vert2face_offset[e1]; f < vert2face_offset[e1 + 1]; f++) {
        atomicMin(reinterpret_cast<unsigned long long*>(&propagated_costs[vert2face[f]]),
                  static_cast<unsigned long long>(cost));
    }
}


static void propagate_cost_quadric(CuMesh& ctx) {
    size_t V = ctx.vertices.size;
    size_t F = ctx.faces.size;
    size_t E = ctx.edges.size;
    ctx.propagated_costs.resize(F);
    ctx.propagated_costs.fill(std::numeric_limits<uint64_t>::max());
    propagate_cost_quadric_kernel<<<(E + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        ctx.edges.ptr,
        ctx.vert2face.ptr,
        ctx.vert2face_offset.ptr,
        ctx.edge_collapse_costs.ptr,
        V, F, E,
        ctx.propagated_costs.ptr
    );
    CUDA_CHECK(cudaGetLastError());
}


// ============================================================================
// Kernel 4: Collapse edges (with optimal position and QEM merge)
// ============================================================================

static __global__ void collapse_edges_quadric_kernel(
    float3* vertices,
    int3* faces,
    uint64_t* edges,
    const int* vert2face,
    const int* vert2face_offset,
    const float* edge_collapse_costs,
    const uint64_t* propagated_costs,
    const float3* optimal_positions,
    QEM* qems,
    const int V,
    const int F,
    const int E,
    const float collapse_thresh,
    int* vertices_kept,
    int* faces_kept
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= E) return;

    float cost = edge_collapse_costs[tid];
    if (cost > collapse_thresh) return;

    // Get edge
    uint64_t e = edges[tid];
    int e0 = int(e >> 32);
    int e1 = int(e & 0xFFFFFFFF);
    uint64_t pack = sq_pack_key_value(tid, cost);

    // Check if this edge won all its incident faces
    for (int f = vert2face_offset[e0]; f < vert2face_offset[e0 + 1]; f++) {
        if (propagated_costs[vert2face[f]] != pack) return;
    }
    for (int f = vert2face_offset[e1]; f < vert2face_offset[e1 + 1]; f++) {
        if (propagated_costs[vert2face[f]] != pack) return;
    }

    // ---- Execute collapse ----
    // e0 survives (gets new position), e1 is deleted

    // Set surviving vertex to optimal position
    float3 opt = optimal_positions[tid];
    vertices[e0] = opt;

    // Merge QEMs: Q[e0] += Q[e1]
    qems[e0] += qems[e1];

    // Mark e1 as deleted
    vertices_kept[e1] = 0;

    // Delete shared faces (faces containing both e0 and e1)
    for (int f = vert2face_offset[e0]; f < vert2face_offset[e0 + 1]; f++) {
        int fid = vert2face[f];
        int3 f_vids = faces[fid];
        if (f_vids.x == e1 || f_vids.y == e1 || f_vids.z == e1) {
            faces_kept[fid] = 0;
        }
    }

    // Update faces: remap e1 -> e0
    for (int f = vert2face_offset[e1]; f < vert2face_offset[e1 + 1]; f++) {
        int fid = vert2face[f];
        int3 f_vids = faces[fid];
        if (f_vids.x == e1) f_vids.x = e0;
        else if (f_vids.y == e1) f_vids.y = e0;
        else if (f_vids.z == e1) f_vids.z = e0;
        faces[fid] = f_vids;
    }
}


// ============================================================================
// Compress kernels (same as simplify.cu)
// ============================================================================

static __global__ void sq_compress_vertices_kernel(
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


static __global__ void sq_compress_faces_kernel(
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


static __global__ void sq_compress_qems_kernel(
    const int* vertices_map,
    const QEM* old_qems,
    const int V,
    QEM* new_qems
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= V) return;
    int new_id = vertices_map[tid];
    int is_kept = vertices_map[tid + 1] == new_id + 1;
    if (is_kept) {
        new_qems[new_id] = old_qems[tid];
    }
}


// ============================================================================
// Host wrapper: Collapse edges and compress
// ============================================================================

static void collapse_edges_quadric(CuMesh& ctx, float collapse_thresh) {
    size_t V = ctx.vertices.size;
    size_t F = ctx.faces.size;
    size_t E = ctx.edges.size;

    ctx.vertices_map.resize(V + 1);
    ctx.faces_map.resize(F + 1);
    ctx.vertices_map.fill(1);
    ctx.faces_map.fill(1);

    collapse_edges_quadric_kernel<<<(E + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        ctx.vertices.ptr,
        ctx.faces.ptr,
        ctx.edges.ptr,
        ctx.vert2face.ptr,
        ctx.vert2face_offset.ptr,
        ctx.edge_collapse_costs.ptr,
        ctx.propagated_costs.ptr,
        ctx.quadric_optimal_positions.ptr,
        reinterpret_cast<QEM*>(ctx.vertex_qems.ptr),
        V, F, E,
        collapse_thresh,
        ctx.vertices_map.ptr,
        ctx.faces_map.ptr
    );
    CUDA_CHECK(cudaGetLastError());

    // --- Compress vertices ---
    size_t temp_storage_bytes = 0;
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        nullptr, temp_storage_bytes,
        ctx.vertices_map.ptr, V + 1
    ));
    ctx.cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        ctx.cub_temp_storage.ptr, temp_storage_bytes,
        ctx.vertices_map.ptr, V + 1
    ));
    int new_num_vertices;
    CUDA_CHECK(cudaMemcpy(&new_num_vertices, ctx.vertices_map.ptr + V, sizeof(int), cudaMemcpyDeviceToHost));

    Buffer<float3> new_verts_buf;
    new_verts_buf.resize(new_num_vertices);
    sq_compress_vertices_kernel<<<(V + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        ctx.vertices_map.ptr,
        ctx.vertices.ptr,
        V,
        new_verts_buf.ptr
    );
    CUDA_CHECK(cudaGetLastError());

    // Swap in new vertices
    ctx.vertices.free();
    ctx.vertices.ptr = new_verts_buf.ptr;
    ctx.vertices.size = new_verts_buf.size;
    ctx.vertices.capacity = new_verts_buf.capacity;
    new_verts_buf.ptr = nullptr;
    new_verts_buf.size = 0;
    new_verts_buf.capacity = 0;

    // --- Compress QEMs (preserve accumulated QEMs across vertex compression) ---
    Buffer<char> new_qems_buf;
    new_qems_buf.resize(new_num_vertices * sizeof(QEM));
    sq_compress_qems_kernel<<<(V + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        ctx.vertices_map.ptr,
        reinterpret_cast<const QEM*>(ctx.vertex_qems.ptr),
        V,
        reinterpret_cast<QEM*>(new_qems_buf.ptr)
    );
    CUDA_CHECK(cudaGetLastError());

    // Swap in new QEMs
    ctx.vertex_qems.free();
    ctx.vertex_qems.ptr = new_qems_buf.ptr;
    ctx.vertex_qems.size = new_qems_buf.size;
    ctx.vertex_qems.capacity = new_qems_buf.capacity;
    new_qems_buf.ptr = nullptr;
    new_qems_buf.size = 0;
    new_qems_buf.capacity = 0;

    // --- Compress faces ---
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        nullptr, temp_storage_bytes,
        ctx.faces_map.ptr, F + 1
    ));
    ctx.cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        ctx.cub_temp_storage.ptr, temp_storage_bytes,
        ctx.faces_map.ptr, F + 1
    ));
    int new_num_faces;
    CUDA_CHECK(cudaMemcpy(&new_num_faces, ctx.faces_map.ptr + F, sizeof(int), cudaMemcpyDeviceToHost));

    Buffer<int3> new_faces_buf;
    new_faces_buf.resize(new_num_faces);
    sq_compress_faces_kernel<<<(F + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        ctx.faces_map.ptr,
        ctx.vertices_map.ptr,
        ctx.faces.ptr,
        F,
        new_faces_buf.ptr
    );
    CUDA_CHECK(cudaGetLastError());

    // Swap in new faces
    ctx.faces.free();
    ctx.faces.ptr = new_faces_buf.ptr;
    ctx.faces.size = new_faces_buf.size;
    ctx.faces.capacity = new_faces_buf.capacity;
    new_faces_buf.ptr = nullptr;
    new_faces_buf.size = 0;
    new_faces_buf.capacity = 0;
}


// ============================================================================
// set_simplify_quadric_params: set parameters from Python
// ============================================================================

void CuMesh::set_simplify_quadric_params(
    float BoundaryQuadricWeight,
    bool  FastPreserveBoundary,
    bool  PreserveBoundary,
    bool  OptimalPlacement,
    float QuadricEpsilon,
    bool  UseArea,
    bool  ScaleIndependent,
    bool  QualityCheck,
    float QualityThr,
    bool  HardQualityCheck,
    float HardQualityThr,
    bool  QualityQuadric,
    float QualityQuadricWeight,
    bool  NormalCheck,
    float NormalThrRad,
    bool  HardNormalCheck,
    bool  AreaCheck,
    bool  PreserveTopology,
    float Aggressiveness
) {
    simplify_quadric_params.BoundaryQuadricWeight = BoundaryQuadricWeight;
    simplify_quadric_params.FastPreserveBoundary = FastPreserveBoundary;
    simplify_quadric_params.PreserveBoundary = PreserveBoundary;
    simplify_quadric_params.OptimalPlacement = OptimalPlacement;
    simplify_quadric_params.QuadricEpsilon = QuadricEpsilon;
    simplify_quadric_params.UseArea = UseArea;
    simplify_quadric_params.ScaleIndependent = ScaleIndependent;
    simplify_quadric_params.QualityCheck = QualityCheck;
    simplify_quadric_params.QualityThr = QualityThr;
    simplify_quadric_params.HardQualityCheck = HardQualityCheck;
    simplify_quadric_params.HardQualityThr = HardQualityThr;
    simplify_quadric_params.QualityQuadric = QualityQuadric;
    simplify_quadric_params.QualityQuadricWeight = QualityQuadricWeight;
    simplify_quadric_params.NormalCheck = NormalCheck;
    simplify_quadric_params.NormalThrRad = NormalThrRad;
    simplify_quadric_params.CosineThr = cosf(NormalThrRad);
    simplify_quadric_params.HardNormalCheck = HardNormalCheck;
    simplify_quadric_params.AreaCheck = AreaCheck;
    simplify_quadric_params.PreserveTopology = PreserveTopology;
    simplify_quadric_params.Aggressiveness = std::max(0.001f, std::min(1.0f, Aggressiveness));

    // Reset QEM accumulation for new simplification session
    qems_initialized = false;
    vertex_qems.free();
}


// ============================================================================
// simplify_quadric_step: orchestrate one round of MeshLab-quality simplification
// ============================================================================

std::tuple<int, int> CuMesh::simplify_quadric_step(int target_num_faces, float threshold, bool timing) {
    std::chrono::high_resolution_clock::time_point start, end;

    bool use_adaptive = (target_num_faces > 0);

    // 1. Build all adjacency (needed every step for cost computation and collapse)
    if (timing) start = std::chrono::high_resolution_clock::now();
    this->get_vertex_face_adjacency();
    if (timing) {
        CUDA_CHECK(cudaDeviceSynchronize());
        end = std::chrono::high_resolution_clock::now();
        std::cout << "  [quadric] get_vertex_face_adjacency: "
                  << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " us" << std::endl;
    }

    if (timing) start = std::chrono::high_resolution_clock::now();
    this->get_edges();
    this->get_boundary_info();
    if (timing) {
        CUDA_CHECK(cudaDeviceSynchronize());
        end = std::chrono::high_resolution_clock::now();
        std::cout << "  [quadric] get_edges + get_boundary_info: "
                  << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " us" << std::endl;
    }

    if (timing) start = std::chrono::high_resolution_clock::now();
    this->get_edge_face_adjacency();
    this->get_vertex_edge_adjacency();
    if (timing) {
        CUDA_CHECK(cudaDeviceSynchronize());
        end = std::chrono::high_resolution_clock::now();
        std::cout << "  [quadric] get_edge_face_adjacency + get_vertex_edge_adjacency: "
                  << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " us" << std::endl;
    }

    // 2+3. First step: compute scale factor and initialize QEMs.
    //       Subsequent steps: reuse accumulated QEMs (matching MeshLab behavior).
    //       QEM accumulation (Q[surviving] += Q[deleted]) preserves the original
    //       surface information, making high-curvature regions more expensive to
    //       simplify and creating proper adaptive density.
    //       Double-precision QEMs ensure the edge-length tiebreaker works in flat
    //       areas (float noise ~1e-7 overwhelms eps=1e-15, double noise ~1e-30 does not).
    if (!this->qems_initialized) {
        // Compute ScaleFactor if ScaleIndependent (one-time)
        if (simplify_quadric_params.ScaleIndependent) {
            size_t V = this->vertices.size;
            std::vector<float3> h_verts(V);
            CUDA_CHECK(cudaMemcpy(h_verts.data(), this->vertices.ptr, V * sizeof(float3), cudaMemcpyDeviceToHost));

            float3 h_min, h_max;
            h_min = h_max = h_verts[0];
            for (size_t i = 1; i < V; i++) {
                h_min.x = std::min(h_min.x, h_verts[i].x);
                h_min.y = std::min(h_min.y, h_verts[i].y);
                h_min.z = std::min(h_min.z, h_verts[i].z);
                h_max.x = std::max(h_max.x, h_verts[i].x);
                h_max.y = std::max(h_max.y, h_verts[i].y);
                h_max.z = std::max(h_max.z, h_verts[i].z);
            }

            float dx = h_max.x - h_min.x;
            float dy = h_max.y - h_min.y;
            float dz = h_max.z - h_min.z;
            float diag = sqrtf(dx * dx + dy * dy + dz * dz);
            if (diag > 1e-20f) {
                double inv_diag = 1.0 / static_cast<double>(diag);
                simplify_quadric_params.ScaleFactor = static_cast<float>(1e8 * pow(inv_diag, 6.0));
            } else {
                simplify_quadric_params.ScaleFactor = 1.0f;
            }
        }

        // Initialize QEMs from current geometry (first step only)
        if (timing) start = std::chrono::high_resolution_clock::now();
        init_quadric(*this);
        if (timing) {
            CUDA_CHECK(cudaDeviceSynchronize());
            end = std::chrono::high_resolution_clock::now();
            std::cout << "  [quadric] init_quadric (first step): "
                      << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " us" << std::endl;
        }

        this->qems_initialized = true;
    }
    // On subsequent steps: vertex_qems already contains accumulated QEMs
    // from previous collapses (compressed alongside vertices).

    // 4. Compute edge collapse costs
    if (timing) start = std::chrono::high_resolution_clock::now();
    compute_edge_cost_quadric(*this);
    if (timing) {
        CUDA_CHECK(cudaDeviceSynchronize());
        end = std::chrono::high_resolution_clock::now();
        std::cout << "  [quadric] compute_edge_cost_quadric: "
                  << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " us" << std::endl;
    }

    // 5. Compute effective collapse threshold
    size_t E = this->edges.size;
    size_t F = this->faces.size;
    float effective_thresh = threshold;

    if (use_adaptive) {
        // Sort edge costs to find the adaptive threshold at a controlled percentile.
        // This limits how many edges can collapse per step, mimicking MeshLab's
        // sequential "always collapse the globally cheapest edge" behavior.
        if (timing) start = std::chrono::high_resolution_clock::now();

        float aggressiveness = simplify_quadric_params.Aggressiveness;

        // Compute how many edges we want eligible this step:
        // - Each edge collapse removes ~2 faces, conflict resolution allows ~50%
        //   of eligible edges, so N eligible → ~N faces removed
        // - Limit by aggressiveness fraction of total edges
        // - Also limit by remaining faces to remove
        int faces_to_remove = std::max(0, (int)F - target_num_faces);
        int max_eligible = std::max(1, std::min(
            (int)(E * aggressiveness),
            faces_to_remove
        ));
        max_eligible = std::min(max_eligible, (int)E);

        // Sort a copy of edge costs using CUB radix sort
        Buffer<float> costs_in, costs_out;
        costs_in.resize(E);
        costs_out.resize(E);
        CUDA_CHECK(cudaMemcpy(costs_in.ptr, this->edge_collapse_costs.ptr,
                              E * sizeof(float), cudaMemcpyDeviceToDevice));

        size_t sort_temp_bytes = 0;
        CUDA_CHECK(cub::DeviceRadixSort::SortKeys(
            nullptr, sort_temp_bytes, costs_in.ptr, costs_out.ptr, (int)E));
        this->cub_temp_storage.resize(sort_temp_bytes);
        CUDA_CHECK(cub::DeviceRadixSort::SortKeys(
            this->cub_temp_storage.ptr, sort_temp_bytes,
            costs_in.ptr, costs_out.ptr, (int)E));

        // Read the cost at the target percentile index
        float adaptive_thresh;
        int thresh_idx = std::min(max_eligible - 1, (int)E - 1);
        CUDA_CHECK(cudaMemcpy(&adaptive_thresh, costs_out.ptr + thresh_idx,
                              sizeof(float), cudaMemcpyDeviceToHost));

        costs_in.free();
        costs_out.free();

        // Ensure we never use INFINITY as threshold (would allow invalid collapses)
        if (!std::isfinite(adaptive_thresh)) {
            adaptive_thresh = 1e30f;
        }

        // Effective threshold is the minimum of adaptive and external
        effective_thresh = std::min(adaptive_thresh, threshold);

        if (timing) {
            CUDA_CHECK(cudaDeviceSynchronize());
            end = std::chrono::high_resolution_clock::now();
            std::cout << "  [quadric] adaptive_threshold (sort): "
                      << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
                      << " us, thresh=" << effective_thresh
                      << ", max_eligible=" << max_eligible << std::endl;
        }
    }

    // 6. Propagate costs to faces
    if (timing) start = std::chrono::high_resolution_clock::now();
    propagate_cost_quadric(*this);
    if (timing) {
        CUDA_CHECK(cudaDeviceSynchronize());
        end = std::chrono::high_resolution_clock::now();
        std::cout << "  [quadric] propagate_cost_quadric: "
                  << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " us" << std::endl;
    }

    // 7. Collapse edges and compress
    if (timing) start = std::chrono::high_resolution_clock::now();
    collapse_edges_quadric(*this, effective_thresh);
    if (timing) {
        CUDA_CHECK(cudaDeviceSynchronize());
        end = std::chrono::high_resolution_clock::now();
        std::cout << "  [quadric] collapse_edges_quadric: "
                  << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " us" << std::endl;
    }

    // Delete all cached info since mesh has changed
    this->clear_cache();

    return std::make_tuple(static_cast<int>(this->vertices.size), static_cast<int>(this->faces.size));
}


} // namespace cumesh
