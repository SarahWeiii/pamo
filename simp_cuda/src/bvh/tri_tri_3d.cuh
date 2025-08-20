#include <cmath>
#include <cassert>

#define SWAP(type, a, b) do { type temp = a; a = b; b = temp; } while(0)

__device__ __host__
inline void copy_v3_v3_float_float3(float r[3], float3 a){
    r[0] = a.x;
    r[1] = a.y;
    r[2] = a.z;
}

__device__ __host__
inline void copy_v3_v3_float3_float(float3 &r, float a[3]){
    r.x = a[0];
    r.y = a[1];
    r.z = a[2];
}

__device__ __host__
inline void sub_v3db_v3fl_v3fl(double r[3], const float a[3], const float b[3])
{
  r[0] = (double)a[0] - (double)b[0];
  r[1] = (double)a[1] - (double)b[1];
  r[2] = (double)a[2] - (double)b[2];
}

__device__ __host__
inline void cross_v3_v3v3_db(double r[3], const double a[3], const double b[3])
{
  assert(r != a && r != b);
  r[0] = a[1] * b[2] - a[2] * b[1];
  r[1] = a[2] * b[0] - a[0] * b[2];
  r[2] = a[0] * b[1] - a[1] * b[0];
}

__device__ __host__
inline double dot_v3db_v3fl(const double a[3], const float b[3])
{
  return a[0] * (double)b[0] + a[1] * (double)b[1] + a[2] * (double)b[2];
}

__device__ __host__
inline void interp_v3_v3v3(float r[3], const float a[3], const float b[3], const float t)
{
  const float s = 1.0f - t;

  r[0] = s * a[0] + t * b[0];
  r[1] = s * a[1] + t * b[1];
  r[2] = s * a[2] + t * b[2];
}

__device__ __host__
inline void copy_v3_v3(float r[3], const float a[3])
{
  r[0] = a[0];
  r[1] = a[1];
  r[2] = a[2];
}
__device__ __host__
float compute_average_length(const float tri_a[3][3], const float tri_b[3][3]) {
    float total_length = 0.0;
    int count = 6; // 총 꼭짓점 수
    for (int i = 0; i < 3; i++) {
        total_length += sqrt(tri_a[i][0] * tri_a[i][0] + tri_a[i][1] * tri_a[i][1] + tri_a[i][2] * tri_a[i][2]);
        total_length += sqrt(tri_b[i][0] * tri_b[i][0] + tri_b[i][1] * tri_b[i][1] + tri_b[i][2] * tri_b[i][2]);
    }
    return total_length / count;
}

__device__ __host__
float find_max_coordinate(const float tri_a[3][3], const float tri_b[3][3]) {
    float max_coordinate = 0.0;

    
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            max_coordinate = fmax(max_coordinate, fabs(tri_a[i][j]));
        }
    }


    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            max_coordinate = fmax(max_coordinate, fabs(tri_b[i][j]));
        }
    }

    return max_coordinate;
}

__device__ __host__
void cross_product(const float u[3], const float v[3], float result[3]) {
    result[0] = u[1] * v[2] - u[2] * v[1];
    result[1] = u[2] * v[0] - u[0] * v[2];
    result[2] = u[0] * v[1] - u[1] * v[0];
}

__device__ __host__
bool is_coplanar(const float tri_a[3][3],
                         const float tri_b[3][3])
{
  struct {
    /* Factor that indicates the position of the intersection point on the line
     * that intersects the planes of the triangles. */
    float min, max;
    /* Intersection point location. */
    float loc[2][3];
  } range[2];

  double side[2][3];
  double ba[3], bc[3], plane_a[4], plane_b[4];
  float scale = find_max_coordinate(tri_a, tri_b);

  sub_v3db_v3fl_v3fl(ba, tri_a[0], tri_a[1]);
  sub_v3db_v3fl_v3fl(bc, tri_a[2], tri_a[1]);
  cross_v3_v3v3_db(plane_a, ba, bc);
  plane_a[3] = -dot_v3db_v3fl(plane_a, tri_a[1]);
  side[1][0] = (dot_v3db_v3fl(plane_a, tri_b[0]) + plane_a[3]);
  side[1][1] = (dot_v3db_v3fl(plane_a, tri_b[1]) + plane_a[3]);
  side[1][2] = (dot_v3db_v3fl(plane_a, tri_b[2]) + plane_a[3]);

  double eps = (double)scale * 1e-6;
  if (fabs(side[1][0]) <= eps && fabs(side[1][1]) <= eps && fabs(side[1][2]) <= eps) {
    return true;
  }

  return false;
}
__device__ __host__
bool isect_tri_tri_v3_ex(const float tri_a[3][3],
                         const float tri_b[3][3],
                         float r_i1[3],
                         float r_i2[3],
                         int *r_tri_a_edge_isect_count)
{
  struct {
    /* Factor that indicates the position of the intersection point on the line
     * that intersects the planes of the triangles. */
    float min, max;
    /* Intersection point location. */
    float loc[2][3];
  } range[2];

  float side[2][3];
  double ba[3], bc[3], plane_a[4], plane_b[4];
  *r_tri_a_edge_isect_count = 0;

  sub_v3db_v3fl_v3fl(ba, tri_a[0], tri_a[1]);
  sub_v3db_v3fl_v3fl(bc, tri_a[2], tri_a[1]);
  cross_v3_v3v3_db(plane_a, ba, bc);
  plane_a[3] = -dot_v3db_v3fl(plane_a, tri_a[1]);
  side[1][0] = (float)(dot_v3db_v3fl(plane_a, tri_b[0]) + plane_a[3]);
  side[1][1] = (float)(dot_v3db_v3fl(plane_a, tri_b[1]) + plane_a[3]);
  side[1][2] = (float)(dot_v3db_v3fl(plane_a, tri_b[2]) + plane_a[3]);

  if (!side[1][0] && !side[1][1] && !side[1][2]) {
    /* Coplanar case is not supported. */
    return false;
  }

  if ((side[1][0] && side[1][1] && side[1][2]) && (side[1][0] < 0.0f) == (side[1][1] < 0.0f) &&
      (side[1][0] < 0.0f) == (side[1][2] < 0.0f))
  {
    /* All vertices of the 2nd triangle are positioned on the same side to the
     * plane defined by the 1st triangle. */
    return false;
  }

  sub_v3db_v3fl_v3fl(ba, tri_b[0], tri_b[1]);
  sub_v3db_v3fl_v3fl(bc, tri_b[2], tri_b[1]);
  cross_v3_v3v3_db(plane_b, ba, bc);
  plane_b[3] = -dot_v3db_v3fl(plane_b, tri_b[1]);
  side[0][0] = (float)(dot_v3db_v3fl(plane_b, tri_a[0]) + plane_b[3]);
  side[0][1] = (float)(dot_v3db_v3fl(plane_b, tri_a[1]) + plane_b[3]);
  side[0][2] = (float)(dot_v3db_v3fl(plane_b, tri_a[2]) + plane_b[3]);

  if ((side[0][0] && side[0][1] && side[0][2]) && (side[0][0] < 0.0f) == (side[0][1] < 0.0f) &&
      (side[0][0] < 0.0f) == (side[0][2] < 0.0f))
  {
    /* All vertices of the 1st triangle are positioned on the same side to the
     * plane defined by the 2nd triangle. */
    return false;
  }

  /* Direction of the line that intersects the planes of the triangles. */
  double isect_dir[3];
  cross_v3_v3v3_db(isect_dir, plane_a, plane_b);
  for (int i = 0; i < 2; i++) {
    const float(*tri)[3] = i == 0 ? tri_a : tri_b;
    /* Rearrange the triangle so that the vertex that is alone on one side
     * of the plane is located at index 1. */
    int tri_i[3];
    if ((side[i][0] && side[i][1]) && (side[i][0] < 0.0f) == (side[i][1] < 0.0f)) {
      tri_i[0] = 1;
      tri_i[1] = 2;
      tri_i[2] = 0;
    }
    else if ((side[i][1] && side[i][2]) && (side[i][1] < 0.0f) == (side[i][2] < 0.0f)) {
      tri_i[0] = 2;
      tri_i[1] = 0;
      tri_i[2] = 1;
    }
    // counterexample case-------------
    // side 1 is zero, 0, 2 is nonzero ------------- cannot divide into 2 and 1, pick anything which is nonzero
    else if (!side[i][1] && (side[i][2] < 0.0f) != (side[i][0] < 0.0f)){
      tri_i[0] = 1;
      tri_i[1] = 2;
      tri_i[2] = 0;
      //printf("for i : %d case add\n", i);
    }
    else {
      tri_i[0] = 0;
      tri_i[1] = 1;
      tri_i[2] = 2;
    }

    double dot_b = dot_v3db_v3fl(isect_dir, tri[tri_i[1]]);
    float sidec = side[i][tri_i[1]];
    if (sidec) {
      double dot_a = dot_v3db_v3fl(isect_dir, tri[tri_i[0]]);
      double dot_c = dot_v3db_v3fl(isect_dir, tri[tri_i[2]]);
      float fac0 = sidec / (sidec - side[i][tri_i[0]]);
      float fac1 = sidec / (sidec - side[i][tri_i[2]]);
      double offset0 = fac0 * (dot_a - dot_b);
      double offset1 = fac1 * (dot_c - dot_b);
      if (offset0 > offset1) {
        /* Sort min max. */
        SWAP(double, offset0, offset1);
        SWAP(float, fac0, fac1);
        SWAP(int, tri_i[0], tri_i[2]);
      }

      range[i].min = (float)(dot_b + offset0);
      range[i].max = (float)(dot_b + offset1);
      interp_v3_v3v3(range[i].loc[0], tri[tri_i[1]], tri[tri_i[0]], fac0);
      interp_v3_v3v3(range[i].loc[1], tri[tri_i[1]], tri[tri_i[2]], fac1);
    }
    else {
      range[i].min = range[i].max = (float)dot_b;
      copy_v3_v3(range[i].loc[0], tri[tri_i[1]]);
      copy_v3_v3(range[i].loc[1], tri[tri_i[1]]);
    }
  }

  if ((range[0].max > range[1].min) && (range[0].min < range[1].max)) {
    /* The triangles intersect because they overlap on the intersection line.
     * Now identify the two points of intersection that are in the middle to get the actual
     * intersection between the triangles. (B--C from A--B--C--D) */
    if (range[0].min >= range[1].min) {
      copy_v3_v3(r_i1, range[0].loc[0]);
      if (range[0].max <= range[1].max) {
        copy_v3_v3(r_i2, range[0].loc[1]);
        *r_tri_a_edge_isect_count = 2;
      }
      else {
        copy_v3_v3(r_i2, range[1].loc[1]);
        *r_tri_a_edge_isect_count = 1;
      }
    }
    else {
      if (range[0].max <= range[1].max) {
        copy_v3_v3(r_i1, range[0].loc[1]);
        copy_v3_v3(r_i2, range[1].loc[0]);
        *r_tri_a_edge_isect_count = 1;
      }
      else {
        copy_v3_v3(r_i1, range[1].loc[0]);
        copy_v3_v3(r_i2, range[1].loc[1]);
      }
    }
    return true;
  }

  return false;
}

__device__ __host__
bool isect_tri_tri_v3(const float3 p1,
                      const float3 q1,
                      const float3 r1,
                      const float3 p2,
                      const float3 q2,
                      const float3 r2,
                      float r_i1[3],
                      float r_i2[3])
{
  float tri_a[3][3], tri_b[3][3];
  int dummy;
  copy_v3_v3_float_float3(tri_a[0], p1);
  copy_v3_v3_float_float3(tri_a[1], q1);
  copy_v3_v3_float_float3(tri_a[2], r1);
  copy_v3_v3_float_float3(tri_b[0], p2);
  copy_v3_v3_float_float3(tri_b[1], q2);
  copy_v3_v3_float_float3(tri_b[2], r2);
  return isect_tri_tri_v3_ex(tri_a, tri_b, r_i1, r_i2, &dummy);
}