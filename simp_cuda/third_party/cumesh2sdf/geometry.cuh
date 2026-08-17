#pragma once
#include "commons.cuh"

constexpr const float EPS = 1e-12f;


__forceinline__ __device__ float lensqr(float3 v)
{
    return dot(v, v);
}

__forceinline__ __device__ float point_to_segment_dist_sqr(float3 v, float3 w, float3 p)
{
  // Return minimum distance between line segment vw and point p
  w -= v;
  p -= v;
  const float l2 = lensqr(w);  // i.e. |w-v|^2 -  avoid a sqrt
  if (l2 < EPS) return lensqr(p);   // v == w case
  // Consider the line extending the segment, parameterized as v + t (w - v).
  // We find projection of point p onto the line. 
  // It falls where t = [(p-v) . (w-v)] / |w-v|^2
  // We clamp t from [0,1] to handle points outside the segment vw.
  const float t = clamp(dot(p, w) / l2, 0.0f, 1.0f);
  const float3 projection = t * w;  // Projection falls on the segment
  return lensqr(p - projection);
}

__forceinline__ __device__ float origin_to_segment_dist_sqr(const float3 v, const float3 w)
{
  const float l2 = lensqr(v - w);
  if (l2 < EPS) return lensqr(v);
  const float t = clamp(dot(v, v - w) / l2, 0.0f, 1.0f);
  const float3 projection = lerp(v, w, t);
  return lensqr(projection);
}

__forceinline__ __device__ float point_to_plane_dist(float3 v1, float3 v2, float3 v3, float3 p)
{
    const float3 nor = normalize(cross(v2 - v1, v3 - v1));
    return abs(dot(nor, p - v1));
}

__forceinline__ __device__ float point_to_tri_dist_sqr(float3 v1, float3 v2, float3 v3, float3 p)
{
    v1 -= p;
    v2 -= p;
    v3 -= p;
    const float d1 = origin_to_segment_dist_sqr(v1, v2);
    const float d2 = origin_to_segment_dist_sqr(v2, v3);
    const float d3 = origin_to_segment_dist_sqr(v3, v1);

    const float min_edge = fminf(fminf(d1, d2), d3);

    const float3 e0 = v2 - v1;
    const float3 e1 = v3 - v1;

    const float3 noru = cross(e0, e1);
    const float scl = lensqr(noru);
    if (scl < EPS) return min_edge;  // 0-area tri

    const float3 proj = dot(v1, noru) / scl * noru;
    const float3 e2 = proj - v1;

    const float dot00 = dot(e0, e0);
    const float dot01 = dot(e0, e1);
    const float dot11 = dot(e1, e1);
    const float dot02 = dot(e0, e2);
    const float dot12 = dot(e1, e2);

    const float denom = dot00 * dot11 - dot01 * dot01;
    // TODO: denom ~ 0 when cosc ~ 1. do we need another degenerate check?
    if (denom < EPS) return min_edge;
    const float invDenom = 1.0 / denom;
    const float u = (dot11 * dot02 - dot01 * dot12) * invDenom;
    const float v = (dot00 * dot12 - dot01 * dot02) * invDenom;
    const float uc = clamp(u, 0.0f, 1.0f);
    const float vc = clamp(v, 0.0f, 1.0f - uc);

    const float3 prc = v1 + uc * e0 + vc * e1;
    // printf("%.3f %.3f; %.3f %.3f %.3f; %.3f %.3f %.3f\n", uc, vc, proj.x, proj.y, proj.z, prc.x, prc.y, prc.z);
    // printf("%.3f %.3f\n", lensqr(p - prc), min_edge);
    return fminf(lensqr(prc), min_edge);
}

__forceinline__ __device__ bool is_approx_equal(const float a, const float b)
{
  return abs(a - b) < FLT_EPSILON;
}

__forceinline__ __device__ bool is_approx_equal(const float3 a, const float3 b)
{
  return is_approx_equal(a.x, b.x) && is_approx_equal(a.y, b.y) && is_approx_equal(a.z, b.z);
}

__forceinline__ __device__ float3 closest_point_on_segment_to_point(const float3& a, const float3& b, const float3& p, float& t)
{
    float3 ab = b - a;
    t = dot(p - a, ab);

    if (t <= 0.0f) {
        // c projects outside the [a,b] interval, on the a side.
        t = 0.0f;
        return a;
    } else {

        // always nonnegative since denom = ||ab||^2
        float denom = dot(ab, ab);

        if (t >= denom) {
            // c projects outside the [a,b] interval, on the b side.
            t = 1.0f;
            return b;
        } else {
            // c projects inside the [a,b] interval.
            t = t / denom;
            return a + (ab * t);
        }
    }
}

__forceinline__ __device__ float3 closest_point_on_triangle_to_point(
    const float3& a, const float3& b, const float3& c, const float3& p)
{
    float uvw[3] = {0, 0, 0};
    // degenerate triangle, singular
    if ((is_approx_equal(a, b) && is_approx_equal(a, c))) {
        uvw[0] = 1.0f;
        return a;
    }

    float3 ab = b - a, ac = c - a, ap = p - a;
    float d1 = dot(ab, ap), d2 = dot(ac, ap);

    // degenerate triangle edges
    if (is_approx_equal(a, b)) {

        float t = 0.0f;
        float3 cp = closest_point_on_segment_to_point(a, c, p, t);

        uvw[0] = 1.0f - t;
        uvw[2] = t;

        return cp;

    } else if (is_approx_equal(a, c) || is_approx_equal(b, c)) {

        float t = 0.0f;
        float3 cp = closest_point_on_segment_to_point(a, b, p, t);
        uvw[0] = 1.0f - t;
        uvw[1] = t;
        return cp;
    }

    if (d1 <= 0.0f && d2 <= 0.0f) {
        uvw[0] = 1.0f;
        return a; // barycentric coordinates (1,0,0)
    }

    // Check if P in vertex region outside B
    float3 bp = p - b;
    float d3 = dot(ab, bp), d4 = dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) {
        uvw[1] = 1.0f;
        return b; // barycentric coordinates (0,1,0)
    }

    // Check if P in edge region of AB, if so return projection of P onto AB
    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        uvw[1] = d1 / (d1 - d3);
        uvw[0] = 1.0f - uvw[1];
        return a + uvw[1] * ab; // barycentric coordinates (1-v,v,0)
    }

    // Check if P in vertex region outside C
    float3 cp = p - c;
    float d5 = dot(ab, cp), d6 = dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) {
        uvw[2] = 1.0f;
        return c; // barycentric coordinates (0,0,1)
    }

    // Check if P in edge region of AC, if so return projection of P onto AC
    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        uvw[2] = d2 / (d2 - d6);
        uvw[0] = 1.0f - uvw[2];
        return a + uvw[2] * ac; // barycentric coordinates (1-w,0,w)
    }

    // Check if P in edge region of BC, if so return projection of P onto BC
    float va = d3*d6 - d5*d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
        uvw[2] = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        uvw[1] = 1.0f - uvw[2];
        return b + uvw[2] * (c - b); // barycentric coordinates (0,1-w,w)
    }

    // P inside face region. Compute Q through its barycentric coordinates (u,v,w)
    float denom = 1.0f / (va + vb + vc);
    uvw[2] = vc * denom;
    uvw[1] = vb * denom;
    uvw[0] = 1.0f - uvw[1] - uvw[2];

    return a + ab*uvw[1] + ac*uvw[2]; // = u*a + v*b + w*c , u= va*denom = 1.0-v-w
}

constexpr const float EPS_PLANE_CLOSE = 1e-5f;

__forceinline__ __device__ float ray_triangle_hit_dist(float3 v1, float3 v2, float3 v3, float3 ro, float3 rd, float pdist)
{
    v2 -= v1;
    v3 -= v1;
    ro -= v1;
    const float3 cr = cross(rd, v3);
    const float det = dot(cr, v2);
    
    const float u = dot(ro, cr) / det;
    const float3 scr = cross(ro, v2);
    const float v = dot(rd, scr) / det;
    const float t = dot(v3, scr) / det;
    const bool di = abs(det) > EPS;
    const bool hit = di && t >= 0 && u >= -FLT_EPSILON && v >= -FLT_EPSILON && u + v <= 1 + FLT_EPSILON;
    const bool planeclose = !di && abs(dot(ro, normalize(cross(v2, v3)))) < EPS_PLANE_CLOSE;
    return hit ? t : planeclose ? pdist : FLT_MAX;
}

__forceinline__ __device__ float ray_triangle_hit_dist(float3 v1, float3 v2, float3 v3, float3 ro, float3 rd)
{
    float pd = sqrt(point_to_tri_dist_sqr(make_float3(0, 0, 0), v2, v3, ro));
    return ray_triangle_hit_dist(v1, v2, v3, ro, rd, pd);
}
