#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Polygon_mesh_processing/self_intersections.h>
#include <CGAL/Polygon_mesh_processing/IO/polygon_mesh_io.h>
#include <CGAL/Real_timer.h>
#include <iostream>
#include <iterator>
#include <fstream>
#include <string>
#include <vector>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Surface_mesh<K::Point_3> Mesh;
typedef boost::graph_traits<Mesh>::face_descriptor face_descriptor;
typedef boost::graph_traits<Mesh>::vertex_descriptor vertex_descriptor;
typedef boost::graph_traits<Mesh>::halfedge_descriptor halfedge_descriptor;
namespace PMP = CGAL::Polygon_mesh_processing;

int main(int argc, char *argv[])
{
  const std::string filename = (argc > 1) ? argv[1] : CGAL::data_file_path("your_default_mesh_path.obj");
  const std::string output_filename = (argc > 2) ? argv[2] : "log.txt";
  Mesh mesh;
  std::ofstream log_file(output_filename, std::ios::app); // Open log file to append

  // if (!PMP::IO::read_polygon_mesh(filename, mesh) || !CGAL::is_triangle_mesh(mesh))
  // {
  //   std::cerr << "Invalid input." << std::endl;
  //   return 1;
  // }

  // Read the mesh from file, enable verbose mode
  if (!PMP::IO::read_polygon_mesh(filename, mesh, CGAL::parameters::repair_polygon_soup(true).verbose(true)))
  {
    std::cerr << "Error: cannot read file " << filename << std::endl;
    return EXIT_FAILURE;
  }

  // Print mesh information
  std::cout << "Mesh info: " << std::endl;
  std::cout << "Vertices: " << num_vertices(mesh) << std::endl;
  std::cout << "Faces: " << num_faces(mesh) << std::endl;
  std::cout << "Edges: " << num_edges(mesh) << std::endl;
  
  // std::cout << "First 10 vertices: " << std::endl;
  // int i = 0;
  // for (vertex_descriptor v : vertices(mesh))
  // {
  //   if (i++ > 10)
  //     break;
  //   K::Point_3 p = mesh.point(v);
  //   std::cout << "v " << p.x() << " " << p.y() << " " << p.z() << std::endl;
  // }



  // bool intersecting = PMP::does_self_intersect(mesh, CGAL::parameters::vertex_point_map(get(CGAL::vertex_point, mesh)));
  // std::cout << (intersecting ? "There are self-intersections." : "There is no self-intersection.") << std::endl;
  // // std::cout << "Elapsed time (does self intersect): " << timer.time() << std::endl;

  std::cout << "Using parallel mode? " << std::is_same<CGAL::Parallel_if_available_tag, CGAL::Parallel_tag>::value << std::endl;

  CGAL::Real_timer timer;
  timer.start();
  std::vector<std::pair<face_descriptor, face_descriptor>> intersected_tris;
  PMP::self_intersections(faces(mesh), mesh, std::back_inserter(intersected_tris));

  std::cout << intersected_tris.size() << " pairs of triangles intersect." << std::endl;
  if (intersected_tris.size())
  {
    log_file << filename << std::endl; // Write filename to log if intersections are found
    std::cout << "intersecting pairs" << std::endl;
  }
  std::cout << "Elapsed time (self intersections): " << timer.time() << " seconds." << std::endl;

  // Print the coordinates of the vertices of the intersecting triangles
  for (auto &pair : intersected_tris)
  {
    auto f1 = pair.first;
    auto f2 = pair.second;

    // std::cout << " # Intersecting Triangles: " << f1 << ", " << f2 << std::endl;

    halfedge_descriptor h1 = halfedge(f1, mesh), h2 = halfedge(f2, mesh);

    std::cout << "Intersecting Triangles: " << std::endl;

    // std::cout << "Triangle 1: ";
    for (halfedge_descriptor h : halfedges_around_face(h1, mesh))
    {
      K::Point_3 p = mesh.point(target(h, mesh));
      // std::cout << "# Vertex: " << target(h, mesh) << std::endl;
      std::cout << "v " << p.x() << " " << p.y() << " " << p.z() << "\n";
    }
    // std::cout << std::endl;

    // std::cout << "Triangle 2: ";
    for (halfedge_descriptor h : halfedges_around_face(h2, mesh))
    {
      // std::cout << "# Vertex id: " << target(h, mesh) << std::endl;
      K::Point_3 p = mesh.point(target(h, mesh));
      std::cout << "v " << p.x() << " " << p.y() << " " << p.z() << "\n";
    }
    // std::cout << std::endl;
    std::cout << "f 1 2 3" << std::endl;
    std::cout << "f 4 5 6" << std::endl;
  }

  log_file.close(); // Close the log file
  return EXIT_SUCCESS;
}