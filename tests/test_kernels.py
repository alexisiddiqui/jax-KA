from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from jaxpropka.geometry import PolarGeometry,empty_polar,frame,Candidate,CandidateLibrary
from jaxpropka.precompute import radial_summary,hbond_strength,radius_graph,build_cache
from jaxpropka.cache import ResidueKey


def test_radial_volume_and_burial_cutoffs():
    center=np.array([[0.,0.,0.]])
    xyz=np.array([[1.,0.,0.],[3.,0.,0.],[15.,0.,0.],[20.,0.,0.]])
    v,m=radial_summary(center,xyz,np.array([1.,2.,3.,4.]))
    np.testing.assert_allclose(v,1/2.75**4+2/3**4+3/15**4)
    assert m[0]==2  # strict <15-A burial; strict <20-A volume
    v,m=radial_summary(center,np.zeros((0,3)),np.zeros(0))
    np.testing.assert_array_equal(v,0);np.testing.assert_array_equal(m,0)


def test_hbond_angle_and_distance_are_geometric_constants():
    d=PolarGeometry(np.array([[0.,0.,0.]]),np.array([[1.,0.,0.]]),np.array([[1.,0.,0.]]),np.array([True]),np.zeros((0,3)))
    a=replace(empty_polar(),acceptor=np.array([[3.5,0.,0.]]))
    np.testing.assert_allclose(hbond_strength(d,a,(2,3)),.425)
    a=replace(a,acceptor=np.array([[-1.,0.,0.]]))
    assert hbond_strength(d,a,(2,3))==0
    assert hbond_strength(empty_polar(),a,(2,3))==0


def test_radius_graph_union_and_no_silent_overflow():
    anchors=np.array([[0.,0.,0.],[12.,0.,0.],[100.,0.,0.]])
    idx,mask=radius_graph(anchors,np.array([4.,4.,4.]),5)
    assert idx[0,mask[0]].tolist()==[1]
    assert idx[1,mask[1]].tolist()==[0]
    assert not mask[2].any()
    with pytest.raises(ValueError,match="overflow"):
        radius_graph(anchors,np.array([4.,4.,4.]),200,max_neighbors=1)


def test_rotation_frames_are_proper_and_degenerate_rejected():
    f=frame(np.array([1.,0.,0.]),np.zeros(3),np.array([0.,1.,0.]))
    np.testing.assert_allclose(f.T@f,np.eye(3),atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(f),1)
    with pytest.raises(ValueError):frame(np.zeros(3),np.zeros(3),np.ones(3))


def geometric_fixture():
    bb=np.array([[[0,0,0],[1,0,0],[1,1,0],[1,2,0]],
                 [[7,0,0],[8,0,0],[8,1,0],[8,2,0]]],dtype=float)
    top=SimpleNamespace(n_residues=2,keys=(ResidueKey("A",1),ResidueKey("B",1)),
                        chain_ids=("A","B"),native_index=np.array([2,8],np.int32),
                        chain_index=np.array([0,1],np.int32),disulfide=np.zeros(2,bool),
                        backbone=bb,metadata={"fixture":"pure numerical atom-cloud fixture"})
    candidates=[]
    for i in range(2):
        row=[]
        for a in range(20):
            xyz=np.array([[2+7*i,0,1+a/100]])
            row.append(Candidate({}, {},xyz,np.array([2.64]),empty_polar()))
        candidates.append(tuple(row))
    centers=np.repeat(bb[:,1:2],9,axis=1)
    lib=CandidateLibrary(tuple(candidates),centers,tuple(tuple(empty_polar() for _ in range(9)) for i in range(2)),
                         bb[:,1],tuple(bb),tuple(np.array([1.06,1.4,1.4,1.]) for _ in range(2)),
                         tuple(empty_polar() for _ in range(2)),tuple(empty_polar() for _ in range(2)),
                         np.ones((2,9),bool),{"candidate_geometry":"synthetic"})
    return top,lib


def test_full_kernel_builder_uses_other_chain_environment():
    top,lib=geometric_fixture();cache=build_cache(top,lib)
    cache.validate()
    for i in range(2):
        j=1-i;k=int(np.flatnonzero(cache.env_mask[i])[0])
        v,m=radial_summary(lib.centers[i],lib.backbone_xyz[j],lib.backbone_volume[j])
        np.testing.assert_allclose(cache.bb_volume[i],v,rtol=1e-6)
        np.testing.assert_allclose(cache.bb_mass[i],m)
        assert cache.env_neighbors[i,k]==j
    assert any(cache.pair_mask[i,k].any() for i in range(2) for k,j in enumerate(cache.neighbors[i]) if i!=j)


def test_precompute_fingerprint_is_not_affected_by_timing():
    top,lib=geometric_fixture();first=build_cache(top,lib);second=build_cache(top,lib)
    assert first.fingerprint()==second.fingerprint()
