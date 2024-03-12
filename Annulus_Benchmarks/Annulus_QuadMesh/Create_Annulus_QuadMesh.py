import underworld3 as uw
import gmsh
from enum import Enum

if uw.mpi.size == 1:
    # to fix trame issue
    import nest_asyncio
    nest_asyncio.apply()
    
    import pyvista as pv
    import underworld3.visualisation as vis
    import matplotlib.pyplot as plt
    import cmcrameri.cm as cmc

radiusOuter = 1.0
radiusInner = 0.547
cellSizeOuter = 1/128
cellSizeInner = 1/128


class boundaries(Enum):
    Lower = 1
    Upper = 2
    Centre = 10


# +
gmsh.initialize()
gmsh.option.setNumber("General.Verbosity", 0)
gmsh.option.setNumber("Mesh.Algorithm", 9) #9
gmsh.model.add("Annulus")

p1 = gmsh.model.geo.add_point(0.00, 0.00, 0.00, meshSize=cellSizeInner)

loops = []

if radiusInner > 0.0:
    p2 = gmsh.model.geo.add_point(radiusInner, 0.0, 0.0, meshSize=cellSizeInner)
    p3 = gmsh.model.geo.add_point(-radiusInner, 0.0, 0.0, meshSize=cellSizeInner)

    c1 = gmsh.model.geo.add_circle_arc(p2, p1, p3)
    c2 = gmsh.model.geo.add_circle_arc(p3, p1, p2)

    cl1 = gmsh.model.geo.add_curve_loop([c1, c2], tag=boundaries.Lower.value)

    loops = [cl1] + loops

p4 = gmsh.model.geo.add_point(radiusOuter, 0.0, 0.0, meshSize=cellSizeOuter)
p5 = gmsh.model.geo.add_point(-radiusOuter, 0.0, 0.0, meshSize=cellSizeOuter)

c3 = gmsh.model.geo.add_circle_arc(p4, p1, p5)
c4 = gmsh.model.geo.add_circle_arc(p5, p1, p4)

# l1 = gmsh.model.geo.add_line(p5, p4)

cl2 = gmsh.model.geo.add_curve_loop([c3, c4], tag=boundaries.Upper.value)

loops = [cl2] + loops

s = gmsh.model.geo.add_plane_surface(loops)

gmsh.model.geo.synchronize()
gmsh.model.mesh.embed(0, [p1], 2, s)

if radiusInner > 0.0:
    gmsh.model.addPhysicalGroup(1, [c1, c2], boundaries.Lower.value, name=boundaries.Lower.name,)
else:
    gmsh.model.addPhysicalGroup(0, [p1], tag=boundaries.Centre.value, name=boundaries.Centre.name)

gmsh.model.addPhysicalGroup(1, [c3, c4], boundaries.Upper.value, name=boundaries.Upper.name)
gmsh.model.addPhysicalGroup(2, [s], 666666, "Elements")

gmsh.model.geo.synchronize()

gmsh.model.mesh.set_recombine(2, s)

gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)
gmsh.option.setNumber("Mesh.FlexibleTransfinite", 0.5)

gmsh.model.mesh.generate(2)

gmsh.write('./annulus_quad.vtk')
gmsh.finalize()
# -


