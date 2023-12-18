

cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9



def add_parameters(p):
    p.add('--prop_dist', type=float, default=4.4*mm, help='propagation distance to reference(intermediate) plane')
    p.add('--prop_dists_from_wrp', type=list, default=[-4.4*mm, -3.2*mm, -2.4*mm, -1.0*mm, 0.0*mm, 1.3*mm, 2.8*mm, 3.8*mm], help='propagation distance between the reference(intermediate) plane and all the 8 target planes')
    p.add()







########################
# 1. Optics Parameters #
########################

# distance between the reference(middle) plane and slm
prop_dist = 4.4 * mm
# distance between the reference(middle) plane and all the 8 target planes 
prop_dists_from_wrp = [-4.4, -3.2, -2.4, -1.0, 0.0, 1.3, 2.8, 3.8] * mm
# depth in diopter space (m^-1) to compute the masks for rgbd input
virtual_depth_planes = [0.0, 0.08417508417508479, 0.14124293785310726, 0.24299599771297942, 0.3171856978085348, 0.4155730533683304, 0.5319148936170226, 0.6112104949314254]
# specify how many target planes used to compute loss here
plane_idx = [0, 1, 2, 3, 4, 5, 6, 7]
# plane_idx = [4]
prop_dists_from_wrp = [prop_dists_from_wrp[idx] for idx in plane_idx]
virtual_depth_planes = [virtual_depth_planes[idx] for idx in plane_idx]
wavelength = 523.0 * nm # 5.177e-07
feature_size = (8.0, 8.0) * um # (6.4e-06, 6.4e-06)
F_aperture = 0.5