#pragma once

#include <cuda.h>
#include <cuda_runtime.h>


namespace cumesh {


/**
 * A 3D vector class with overloaded operators and methods.
 */
struct __align__(16) Vec3f {
    float x, y, z;

    __device__ __forceinline__ Vec3f();
    __device__ __forceinline__ Vec3f(float x, float y, float z);
    __device__ __forceinline__ Vec3f(float3 v);
    __device__ __forceinline__ Vec3f operator+(const Vec3f& o) const;
    __device__ __forceinline__ Vec3f& operator+=(const Vec3f& o);
    __device__ __forceinline__ Vec3f operator-(const Vec3f& o) const;
    __device__ __forceinline__ Vec3f& operator-=(const Vec3f& o);
    __device__ __forceinline__ Vec3f operator*(float s) const;
    __device__ __forceinline__ Vec3f& operator*=(float s);
    __device__ __forceinline__ Vec3f operator/(float s) const;
    __device__ __forceinline__ Vec3f& operator/=(float s);
    __device__ __forceinline__ float dot(const Vec3f& o) const;
    __device__ __forceinline__ float norm() const;
    __device__ __forceinline__ float norm2() const;
    __device__ __forceinline__ Vec3f normalized() const;
    __device__ __forceinline__ void normalize();
    __device__ __forceinline__ Vec3f cross(const Vec3f& o) const;
    __device__ __forceinline__ Vec3f slerp(const Vec3f& o, float t) const;
};


/**
 * QEM (Quadric Error Metric) class for mesh simplification.
 * Uses double precision to match MeshLab's accuracy.
 * Double precision is critical for the edge-length tiebreaker in flat areas:
 * with float, QEM noise (~1e-7) overwhelms QuadricEpsilon (1e-15), preventing
 * the tiebreaker from activating. With double, noise is ~1e-15, allowing
 * proper edge-length ordering that creates adaptive density.
 */
struct __align__(16) QEM
{
    // store upper triangle of symmetric 4x4 matrix:
    // e = [ 00, 01, 02, 03, 11, 12, 13, 22, 23, 33 ]
    double e[10];

    __device__ __forceinline__ QEM();
    __device__ __forceinline__ QEM operator+(const QEM& o) const;
    __device__ __forceinline__ QEM& operator+=(const QEM& o);
    __device__ __forceinline__ QEM operator-(const QEM& o) const;
    __device__ __forceinline__ QEM& operator-=(const QEM& o);
    __device__ __forceinline__ void zero();
    __device__ __forceinline__ void add_plane(float4 p);
    __device__ __forceinline__ double evaluate(const Vec3f& p) const;
    __device__ __forceinline__ bool solve_optimal(float3 &out, float &err) const;
};


__device__ __forceinline__ Vec3f::Vec3f() {
    x = 0.0f;
    y = 0.0f;
    z = 0.0f;
}

__device__ __forceinline__ Vec3f::Vec3f(float x, float y, float z) {
    this->x = x;
    this->y = y;
    this->z = z;
}

__device__ __forceinline__ Vec3f::Vec3f(float3 v) {
    x = v.x;
    y = v.y;
    z = v.z;
}


__device__ __forceinline__ Vec3f Vec3f::operator+(const Vec3f& o) const {
    return Vec3f(x + o.x, y + o.y, z + o.z);
}


__device__ __forceinline__ Vec3f& Vec3f::operator+=(const Vec3f& o) {
    x += o.x;
    y += o.y;
    z += o.z;
    return *this;
}


__device__ __forceinline__ Vec3f Vec3f::operator-(const Vec3f& o) const {
    return Vec3f(x - o.x, y - o.y, z - o.z);
}


__device__ __forceinline__ Vec3f& Vec3f::operator-=(const Vec3f& o) {
    x -= o.x;
    y -= o.y;
    z -= o.z;
    return *this;
}


__device__ __forceinline__ Vec3f Vec3f::operator*(float s) const {
    return Vec3f(x * s, y * s, z * s);
}


__device__ __forceinline__ Vec3f& Vec3f::operator*=(float s) {
    x *= s;
    y *= s;
    z *= s;
    return *this;
}


__device__ __forceinline__ Vec3f Vec3f::operator/(float s) const {
    return Vec3f(x / s, y / s, z / s);
}


__device__ __forceinline__ Vec3f& Vec3f::operator/=(float s) {
    x /= s;
    y /= s;
    z /= s;
    return *this;
}


__device__ __forceinline__ float Vec3f::dot(const Vec3f& o) const {
    return x * o.x + y * o.y + z * o.z;
}


__device__ __forceinline__ float Vec3f::norm() const {
    return sqrtf(x * x + y * y + z * z);
}


__device__ __forceinline__ float Vec3f::norm2() const {
    return x * x + y * y + z * z;
}


__device__ __forceinline__ Vec3f Vec3f::normalized() const {
    float inv_norm = rsqrtf(x * x + y * y + z * z);
    return Vec3f(x * inv_norm, y * inv_norm, z * inv_norm);
}


__device__ __forceinline__ void Vec3f::normalize() {
    float inv_norm = rsqrtf(x * x + y * y + z * z);
    x *= inv_norm;
    y *= inv_norm;
    z *= inv_norm;
}


__device__ __forceinline__ Vec3f Vec3f::cross(const Vec3f& o) const {
    return Vec3f(y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x);
}


__device__ __forceinline__ Vec3f Vec3f::slerp(const Vec3f& o, float t) const {
    float dot_prod = this->dot(o);
    dot_prod = fmaxf(fminf(dot_prod, 1.0f), -1.0f); // Clamp to [-1, 1]
    float theta = acosf(dot_prod) * t;
    Vec3f relative_vec = (o - (*this) * dot_prod).normalized();
    return (*this) * cosf(theta) + relative_vec * sinf(theta);
}


__device__ __forceinline__ QEM::QEM() {
    zero();
}


__device__ __forceinline__ QEM QEM::operator+(const QEM& o) const {
    QEM res;
    #pragma unroll
    for (int i = 0; i < 10; ++i) res.e[i] = e[i] + o.e[i];
    return res;
}


__device__ __forceinline__ QEM& QEM::operator+=(const QEM& o) {
    #pragma unroll
    for (int i = 0; i < 10; ++i) e[i] += o.e[i];
    return *this;
}


__device__ __forceinline__ QEM QEM::operator-(const QEM& o) const {
    QEM res;
    #pragma unroll
    for (int i = 0; i < 10; ++i) res.e[i] = e[i] - o.e[i];
    return res;
}


__device__ __forceinline__ QEM& QEM::operator-=(const QEM& o) {
    #pragma unroll
    for (int i = 0; i < 10; ++i) e[i] -= o.e[i];
    return *this;
}

__device__ __forceinline__ void QEM::zero() {
    #pragma unroll
    for (int i = 0; i < 10; ++i) e[i] = 0.0;
}


// Add plane p = (a,b,c,d) as outer product p * p^T
// Float plane equation is promoted to double for accumulation precision
__device__ __forceinline__ void QEM::add_plane(float4 p) {
    double a = (double)p.x, b = (double)p.y, c = (double)p.z, d = (double)p.w;
    e[0] += a * a;
    e[1] += a * b;
    e[2] += a * c;
    e[3] += a * d;
    e[4] += b * b;
    e[5] += b * c;
    e[6] += b * d;
    e[7] += c * c;
    e[8] += c * d;
    e[9] += d * d;
}


// Evaluate v^T * Q * v for v = (x,y,z,1)
// Returns double for full precision (critical for edge-length tiebreaker in flat areas)
__device__ __forceinline__ double QEM::evaluate(const Vec3f& p) const {
    double x = (double)p.x, y = (double)p.y, z = (double)p.z, w = 1.0;
    double res = 0.0;
    res += e[0] * x * x;
    res += 2.0 * e[1] * x * y;
    res += 2.0 * e[2] * x * z;
    res += 2.0 * e[3] * x * w;
    res += e[4] * y * y;
    res += 2.0 * e[5] * y * z;
    res += 2.0 * e[6] * y * w;
    res += e[7] * z * z;
    res += 2.0 * e[8] * z * w;
    res += e[9] * w * w;
    return res;
}


// Try to solve for optimal point minimizing v^T Q v with constraint v = (x,y,z,1)
// Uses double precision internally for numerical stability
__device__ __forceinline__ bool QEM::solve_optimal(float3 &out, float &err) const {
    // Build A (symmetric) in double
    double A00 = e[0], A01 = e[1], A02 = e[2];
    double A11 = e[4], A12 = e[5], A22 = e[7];
    double b0 = e[3], b1 = e[6], b2 = e[8];

    // Compute determinant
    double det =
        A00 * (A11 * A22 - A12 * A12) -
        A01 * (A01 * A22 - A12 * A02) +
        A02 * (A01 * A12 - A11 * A02);

    if (fabs(det) < 1e-20) {
        out = make_float3(0.0f, 0.0f, 0.0f);
        err = (float)evaluate(Vec3f(0.0f, 0.0f, 0.0f));
        return false;
    }

    double invDet = 1.0 / det;

    // Compute inverse(A) via adjugate
    double inv00 =  (A11 * A22 - A12 * A12) * invDet;
    double inv01 = -(A01 * A22 - A12 * A02) * invDet;
    double inv02 =  (A01 * A12 - A11 * A02) * invDet;
    double inv11 =  (A00 * A22 - A02 * A02) * invDet;
    double inv12 = -(A00 * A12 - A01 * A02) * invDet;
    double inv22 =  (A00 * A11 - A01 * A01) * invDet;

    // x = -inv(A) * b
    double x = -(inv00 * b0 + inv01 * b1 + inv02 * b2);
    double y = -(inv01 * b0 + inv11 * b1 + inv12 * b2);
    double z = -(inv02 * b0 + inv12 * b1 + inv22 * b2);

    out = make_float3((float)x, (float)y, (float)z);
    err = (float)evaluate(Vec3f(out));
    return true;
}


} // namespace cumesh
