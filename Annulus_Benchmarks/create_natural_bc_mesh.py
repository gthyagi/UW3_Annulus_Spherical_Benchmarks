def main():
    if simplex:
        # Initialize the Gmsh API
        gmsh.initialize()
    
        gmsh.option.setNumber("General.Verbosity", 0)
        
        # Create a new model
        gmsh.model.add("2D Unstructured Box with Internal Boundary")
        
        # Add points (corners of the unit box) with the specified cell size
        p1 = gmsh.model.geo.addPoint(xmin, ymin, 0, cellSize)  # Bottom-left corner
        p2 = gmsh.model.geo.addPoint(xmax, ymin, 0, cellSize)  # Bottom-right corner
        p3 = gmsh.model.geo.addPoint(xmax, ymax, 0, cellSize)  # Top-right corner
        p4 = gmsh.model.geo.addPoint(xmin, ymax, 0, cellSize)  # Top-left corner
        
        # Add points for the internal boundary at y = yint
        p5 = gmsh.model.geo.addPoint(xmin, yint, 0, cellSize)  # Left point on internal boundary
        p6 = gmsh.model.geo.addPoint(xmax, yint, 0, cellSize)  # Right point on internal boundary
        
        # Add lines (edges of the unit box and internal boundary)
        l1 = gmsh.model.geo.addLine(p1, p2)  # Bottom edge
        l2_bottom = gmsh.model.geo.addLine(p2, p6)  # Right bottom part
        l2_top = gmsh.model.geo.addLine(p6, p3)  # Right top part
        l3 = gmsh.model.geo.addLine(p3, p4)  # Top edge
        l4_bottom = gmsh.model.geo.addLine(p1, p5)  # Left bottom part
        l4_top = gmsh.model.geo.addLine(p5, p4)  # Left top part
        l_internal = gmsh.model.geo.addLine(p5, p6)  # Internal boundary line at yint
        
        # Create curve loops for the two regions (bottom and top parts of the box)
        cl1 = gmsh.model.geo.addCurveLoop([l1, l2_bottom, -l_internal, -l4_bottom])  # Bottom region
        cl2 = gmsh.model.geo.addCurveLoop([l_internal, l2_top, l3, -l4_top])  # Top region
        
        # Create plane surfaces for both regions
        surface_bottom = gmsh.model.geo.addPlaneSurface([cl1])
        surface_top = gmsh.model.geo.addPlaneSurface([cl2])
        
        # Add physical groups (labels) for the edges using the enum class
        gmsh.model.geo.addPhysicalGroup(1, [l1], tag=boundaries.Bottom.value)  # Bottom
        gmsh.model.setPhysicalName(1, boundaries.Bottom.value, boundaries.Bottom.name)
        
        gmsh.model.geo.addPhysicalGroup(1, [l2_bottom, l2_top], tag=boundaries.Right.value)  # Right
        gmsh.model.setPhysicalName(1, boundaries.Right.value, boundaries.Right.name)
        
        gmsh.model.geo.addPhysicalGroup(1, [l3], tag=boundaries.Top.value)  # Top
        gmsh.model.setPhysicalName(1, boundaries.Top.value, boundaries.Top.name)
        
        gmsh.model.geo.addPhysicalGroup(1, [l4_bottom, l4_top], tag=boundaries.Left.value)  # Left
        gmsh.model.setPhysicalName(1, boundaries.Left.value, boundaries.Left.name)
        
        # Add physical group for the internal boundary at y = yint
        gmsh.model.geo.addPhysicalGroup(1, [l_internal], tag=boundaries.Internal.value)
        gmsh.model.setPhysicalName(1, boundaries.Internal.value, boundaries.Internal.name)
        
        # Add physical groups for the surface (optional)
        gmsh.model.geo.addPhysicalGroup(2, [surface_bottom], tag=6)  # Bottom region surface
        gmsh.model.setPhysicalName(2, 6, "Surface Bottom")
        
        gmsh.model.geo.addPhysicalGroup(2, [surface_top], tag=7)  # Top region surface
        gmsh.model.setPhysicalName(2, 7, "Surface Top")
        
        # Synchronize the internal CAD representation with the Gmsh model
        gmsh.model.geo.synchronize()
        
        # Mesh the surface using unstructured triangles
        gmsh.model.mesh.generate(2)  # 2D mesh
        
        # Optionally save the mesh to a file
        gmsh.write(f"{output_dir}/mesh_ib_simp_res_{int(1/cellSize)}.msh")
        
        # # Launch Gmsh GUI to visualize (optional)
        # gmsh.fltk.run()
        
        # Finalize the Gmsh API
        gmsh.finalize()
    
    else:
        # Initialize the Gmsh API
        gmsh.initialize()
        
        gmsh.option.setNumber("General.Verbosity", 0)
        
        # Create a new model
        gmsh.model.add("2D Structured Box with Internal Boundary")
        
        # Calculate cell size for y axis
        cell_size_y = (ymax - ymin) / ny_total
        
        # Calculate the number of divisions for each region based on cell size
        ny_bottom = int((yint - ymin) / cell_size_y)  # Number of divisions for the bottom region
        ny_top = int((ymax - yint) / cell_size_y)  # Number of divisions for the top region
        
        # Add points (corners of the unit box)
        p1 = gmsh.model.geo.addPoint(xmin, ymin, 0)  # Bottom-left corner
        p2 = gmsh.model.geo.addPoint(xmax, ymin, 0)  # Bottom-right corner
        p3 = gmsh.model.geo.addPoint(xmax, ymax, 0)  # Top-right corner
        p4 = gmsh.model.geo.addPoint(xmin, ymax, 0)  # Top-left corner
        
        # Add points for the internal boundary at y = yint
        p5 = gmsh.model.geo.addPoint(xmin, yint, 0)  # Left point on internal boundary
        p6 = gmsh.model.geo.addPoint(xmax, yint, 0)  # Right point on internal boundary
        
        # Add lines (edges of the unit box and internal boundary)
        l1 = gmsh.model.geo.addLine(p1, p2)  # Bottom edge
        l2_bottom = gmsh.model.geo.addLine(p2, p6)  # Right bottom part
        l2_top = gmsh.model.geo.addLine(p6, p3)  # Right top part
        l3 = gmsh.model.geo.addLine(p3, p4)  # Top edge
        l4_bottom = gmsh.model.geo.addLine(p1, p5)  # Left bottom part
        l4_top = gmsh.model.geo.addLine(p5, p4)  # Left top part
        l_internal = gmsh.model.geo.addLine(p5, p6)  # Internal boundary line at yint
        
        # Set transfinite lines for structured mesh generation (subdividing the edges)
        gmsh.model.geo.mesh.setTransfiniteCurve(l1, nx + 1)  # Bottom edge
        gmsh.model.geo.mesh.setTransfiniteCurve(l2_bottom, ny_bottom + 1)  # Right bottom
        gmsh.model.geo.mesh.setTransfiniteCurve(l2_top, ny_top + 1)  # Right top
        gmsh.model.geo.mesh.setTransfiniteCurve(l3, nx + 1)  # Top edge
        gmsh.model.geo.mesh.setTransfiniteCurve(l4_bottom, ny_bottom + 1)  # Left bottom
        gmsh.model.geo.mesh.setTransfiniteCurve(l4_top, ny_top + 1)  # Left top
        gmsh.model.geo.mesh.setTransfiniteCurve(l_internal, nx + 1)  # Internal boundary
        
        # Create curve loops for the two regions (bottom and top parts of the box)
        cl1 = gmsh.model.geo.addCurveLoop([l1, l2_bottom, -l_internal, -l4_bottom])  # Bottom region
        cl2 = gmsh.model.geo.addCurveLoop([l_internal, l2_top, l3, -l4_top])  # Top region
        
        # Create plane surfaces for both regions
        surface_bottom = gmsh.model.geo.addPlaneSurface([cl1])
        surface_top = gmsh.model.geo.addPlaneSurface([cl2])
        
        # Apply transfinite surface to both regions to generate a structured mesh
        gmsh.model.geo.mesh.setTransfiniteSurface(surface_bottom)
        gmsh.model.geo.mesh.setTransfiniteSurface(surface_top)
        
        # Add physical groups (labels) for the edges using the enum class
        gmsh.model.geo.addPhysicalGroup(1, [l1], tag=boundaries.Bottom.value)  # Bottom
        gmsh.model.setPhysicalName(1, boundaries.Bottom.value, boundaries.Bottom.name)
        
        gmsh.model.geo.addPhysicalGroup(1, [l2_bottom, l2_top], tag=boundaries.Right.value)  # Right
        gmsh.model.setPhysicalName(1, boundaries.Right.value, boundaries.Right.name)
        
        gmsh.model.geo.addPhysicalGroup(1, [l3], tag=boundaries.Top.value)  # Top
        gmsh.model.setPhysicalName(1, boundaries.Top.value, boundaries.Top.name)
        
        gmsh.model.geo.addPhysicalGroup(1, [l4_bottom, l4_top], tag=boundaries.Left.value)  # Left
        gmsh.model.setPhysicalName(1, boundaries.Left.value, boundaries.Left.name)
        
        # Add physical group for the internal boundary at y = yint
        gmsh.model.geo.addPhysicalGroup(1, [l_internal], tag=boundaries.Internal.value)
        gmsh.model.setPhysicalName(1, boundaries.Internal.value, boundaries.Internal.name)
        
        # Add physical groups for the surface (optional)
        gmsh.model.geo.addPhysicalGroup(2, [surface_bottom], tag=6)  # Bottom region surface
        gmsh.model.setPhysicalName(2, 6, "Surface Bottom")
        
        gmsh.model.geo.addPhysicalGroup(2, [surface_top], tag=7)  # Top region surface
        gmsh.model.setPhysicalName(2, 7, "Surface Top")
        
        # Synchronize the internal CAD representation with the Gmsh model
        gmsh.model.geo.synchronize()
        
        # Mesh the surface using structured quadrilaterals
        gmsh.option.setNumber("Mesh.RecombineAll", 1)  # Ensures quadrilateral elements
        gmsh.model.mesh.generate(2)  # 2D structured mesh
        
        # Optionally save the mesh to a file
        gmsh.write(f"{output_dir}/mesh_ib_quad_res_{nx}_{ny_total}.msh")
        
        # # Launch Gmsh GUI to visualize (optional)
        # gmsh.fltk.run()
        
        # Finalize the Gmsh API
        gmsh.finalize()


if __name__ == "__main__":
    main()


