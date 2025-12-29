"""
Example script demonstrating how to collapse skinny/sliver faces.

This script shows how to use the collapse_skinny_faces() function to
improve mesh quality by removing skinny triangles that can cause
numerical issues or visual artifacts (like staircase patterns).
"""
import torch
import trimesh
import cumesh
import pymeshlab


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Collapse skinny faces in a mesh')
    parser.add_argument('input', type=str, help='Input mesh file')
    parser.add_argument('--output', type=str, default='collapsed_skinny.ply', 
                        help='Output mesh file')
    parser.add_argument('--min-angle', type=float, default=5.0, # or 1.0
                        help='Minimum angle threshold in degrees (default: 1)')
    parser.add_argument('--max-iterations', type=int, default=100,
                        help='Maximum number of collapse iterations (default: 100)')
    parser.add_argument('--verbose', action='store_true', help='Print progress')
    parser.add_argument('--repair-non-manifold', action='store_true', help='Repair non-manifold geometry')
    args = parser.parse_args()
    
    # Load mesh
    print(f"Loading mesh from {args.input}...")
    mesh = trimesh.load(args.input, process=False)
    mesh.merge_vertices()
    
    vertices = torch.from_numpy(mesh.vertices).float().cuda()
    faces = torch.from_numpy(mesh.faces).int().cuda()
    
    print(f"\n=== Original Mesh ===")
    print(f"Vertices: {vertices.shape[0]}, Faces: {faces.shape[0]}")
        
    # Initialize CuMesh and collapse skinny faces
    print(f"\n=== Collapsing skinny faces ===")
    print(f"Min angle threshold: {args.min_angle} deg")
    print(f"Max iterations: {args.max_iterations}")
    
    cu_mesh = cumesh.CuMesh()
    cu_mesh.init(vertices, faces)
    
    new_num_vert, new_num_face = cu_mesh.collapse_skinny_faces(
        min_angle_deg=args.min_angle,
        max_iterations=args.max_iterations,
        verbose=args.verbose
    )
    
    new_vertices, new_faces = cu_mesh.read()
    
    print(f"\n=== Result ===")
    print(f"Vertices: {new_vertices.shape[0]}, Faces: {new_faces.shape[0]}")
    print(f"Removed {vertices.shape[0] - new_vertices.shape[0]} vertices")
    print(f"Removed {faces.shape[0] - new_faces.shape[0]} faces")

    cu_mesh.remove_degenerate_faces()
    new_vertices, new_faces = cu_mesh.read()

    # Repair non-manifold faces and vertices using pymeshlab
    print("\n=== Repairing non-manifold geometry ===")
    ms = pymeshlab.MeshSet()
    m = pymeshlab.Mesh(
        vertex_matrix=new_vertices.cpu().numpy(),
        face_matrix=new_faces.cpu().numpy()
    )
    ms.add_mesh(m)
    
    if args.repair_non_manifold:
        for i in range(2):
            # Repair non-manifold edges by splitting vertices
            ms.meshing_repair_non_manifold_edges(method='Remove Faces')
            
            # Repair non-manifold vertices by splitting
            ms.meshing_repair_non_manifold_vertices(vertdispratio=0.0)
            
            # Remove unreferenced vertices
            ms.meshing_remove_unreferenced_vertices()
    
    repaired_mesh = ms.current_mesh()
    repaired_vertices = repaired_mesh.vertex_matrix()
    repaired_faces = repaired_mesh.face_matrix()
    
    print(f"After repair: Vertices: {repaired_vertices.shape[0]}, Faces: {repaired_faces.shape[0]}")
        
    # Save result
    print(f"\nSaving result to {args.output}...")
    new_mesh = trimesh.Trimesh(
        vertices=repaired_vertices,
        faces=repaired_faces,
        process=False
    )
    new_mesh.export(args.output)
    print("Done!")

