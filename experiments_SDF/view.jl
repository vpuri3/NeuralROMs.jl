#
using NeuralROMs, NPZ

let
    pkgpath = dirname(dirname(pathof(NeuralROMs)))
    sdfpath = joinpath(pkgpath, "experiments_SDF")
    !(sdfpath in LOAD_PATH) && push!(LOAD_PATH, sdfpath)
end

#======================================================#

basedir  = joinpath(pkgdir(NeuralROMs), "experiments_SDF", "dataset_netfabb")
datafile = joinpath(basedir, "56250_3b6024e3_54.npz")

data = npzread(datafile)
verts = data["verts"]' # [3, V]
elems = data["elems"]' # [8, F] (hex mesh)
displ = data["disp"]'  # [3, V]

import Meshes
using GeometryBasics
using GeometryBasics: Mesh

verts = [Point3f(verts[:, i]...) for i in axes(verts, 2)]

elems = vcat(
    elems[1, :]',
    elems[2, :]',
    elems[3, :]',
    elems[4, :]',
    #
    elems[7, :]',
    elems[8, :]',
    elems[5, :]',
    elems[6, :]',
)

faces1 = vcat(elems[1, :]', elems[2, :]', elems[3, :]', elems[4, :]') # bot
faces2 = vcat(elems[1, :]', elems[2, :]', elems[6, :]', elems[5, :]')
faces3 = vcat(elems[2, :]', elems[3, :]', elems[7, :]', elems[6, :]')
faces4 = vcat(elems[3, :]', elems[4, :]', elems[8, :]', elems[7, :]')
faces5 = vcat(elems[4, :]', elems[1, :]', elems[5, :]', elems[8, :]')
faces6 = vcat(elems[5, :]', elems[6, :]', elems[7, :]', elems[8, :]') # top

faces = hcat(faces1, faces2, faces3, faces4, faces5, faces6)
mesh  = Mesh(verts, vec(faces), QuadFace)

using MeshCat
isdefined(Main, :vis) && MeshCat.close_server!(vis.core)
vis = Visualizer()
setobject!(vis, mesh)
open(vis; start_browser = false)

# using WriteVTK
# cell_type = VTKCellTypes.VTK_HEXAHEDRON
# cells = [MeshCell(cell_type, elems[:, i]) for i in axes(elems, 2)]
#
# outfile = "outfile"
# vtk_out = vtk_grid(outfile, verts, cells)
# # vtk_out["displacement", VTKPointData()] =  displ
# close(vtk_out)

#======================================================#
nothing
