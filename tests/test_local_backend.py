"""Local advancing-front candidates on CPU and the actual CUDA kernel."""

import importlib.util

import numpy as np
import pytest

from saft3d.backend import CPUBackend, create_backend


@pytest.fixture(scope="module")
def cuda_backend():
    if importlib.util.find_spec("cupy") is None:
        pytest.skip("CUDA unavailable: CuPy is not installed")
    import cupy as cp

    try:
        count = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError as exc:
        if exc.status in (3, 35, 100):
            pytest.skip(f"CUDA unavailable: {exc}")
        raise
    if count == 0:
        pytest.skip("CUDA unavailable: no device detected")
    # Compilation or kernel failures after device discovery must fail, not skip.
    return create_backend("cuda")


def _triangle():
    return np.array([[[0., 0., 0.], [1., 0., 0.], [.5, np.sqrt(3.) / 2., 0.]]])


def test_local_shapes_heights_and_positive_volume():
    cpu = CPUBackend()
    triangle = _triangle()
    factors = np.array([1., .8, .6, .4, .2, .1])
    for target, effective_size in ((.25, .25), (3., 1.)):
        proposals, valid = cpu.propose_local(triangle, target)
        assert proposals.shape == (1, 6, 4, 3)
        assert proposals.dtype == np.float64
        assert valid.shape == (1, 6) and valid.dtype == bool
        assert valid.all()
        expected_base = np.broadcast_to(triangle[:, None, [0, 2, 1]], (1, 6, 3, 3))
        np.testing.assert_array_equal(proposals[:, :, :3], expected_base)
        np.testing.assert_allclose(proposals[0, :, 3, 2],
                                   -np.sqrt(2. / 3.) * effective_size * factors)
        volumes, _ = cpu.evaluate(proposals.reshape(-1, 4, 3))
        assert np.all(volumes > 0.)
        np.testing.assert_allclose(volumes / volumes[0], factors)
    assert cpu.calls["propose_local"] == 2


def test_local_rotation_and_translation_equivariance():
    cpu = CPUBackend()
    triangle = _triangle()
    rng = np.random.default_rng(515)
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1.
    translation = np.array([.25, -.5, .125])
    reference, valid = cpu.propose_local(triangle, .3)
    actual, rotated_valid = cpu.propose_local(triangle @ rotation.T + translation, .3)
    np.testing.assert_allclose(actual, reference @ rotation.T + translation, atol=1e-14)
    np.testing.assert_array_equal(rotated_valid, valid)


def test_local_degeneracy_tolerance_empty_and_invalid_inputs():
    cpu = CPUBackend()
    flat = np.array([[[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]],
                     [[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]]])
    proposals, valid = cpu.propose_local(flat, .2)
    assert np.isfinite(proposals).all() and not valid.any()
    tiny = _triangle() * 1e-6
    proposals, valid = cpu.propose_local(tiny, 1e-6, tol=1e-10)
    volumes, _ = cpu.evaluate(proposals.reshape(-1, 4, 3))
    np.testing.assert_array_equal(valid.ravel(), volumes > 1e-20)
    assert cpu.propose_local(tiny, 1e-6, tol=1e-8)[1].sum() == 0
    proposals, valid = cpu.propose_local(np.empty((0, 3, 3)), .1)
    assert proposals.shape == (0, 6, 4, 3) and valid.shape == (0, 6)
    with pytest.raises(ValueError):
        cpu.propose_local(np.zeros((3, 3)), .1)
    with pytest.raises(ValueError):
        cpu.propose_local(np.full((1, 3, 3), np.nan), .1)
    with pytest.raises(ValueError):
        cpu.propose_local(_triangle(), 0.)
    with pytest.raises(ValueError):
        cpu.propose_local(_triangle(), .1, tol=-1.)


def test_local_cuda_matches_cpu_random_rotated_and_degenerate_faces(cuda_backend):
    cpu, gpu = CPUBackend(), cuda_backend
    rng = np.random.default_rng(999)
    triangles = []
    for _ in range(24):
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(rotation) < 0:
            rotation[:, 0] *= -1.
        scale = rng.uniform(.01, .25)
        triangles.append((_triangle()[0] * scale) @ rotation.T + rng.uniform(-.3, .3, 3))
    triangles.extend([np.zeros((3, 3)), np.array([[0., 0., 0.], [.1, 0., 0.], [.2, 0., 0.]])])
    triangles = np.stack(triangles)
    for target in (.001, .3):
        expected, expected_valid = cpu.propose_local(triangles, target)
        actual, actual_valid = gpu.propose_local(triangles, target)
        np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-13)
        np.testing.assert_array_equal(actual_valid, expected_valid)
        ev, eq = cpu.evaluate(expected.reshape(-1, 4, 3))
        av, aq = gpu.evaluate(actual.reshape(-1, 4, 3))
        np.testing.assert_allclose(av, ev, rtol=1e-9, atol=1e-15)
        np.testing.assert_allclose(aq, eq, rtol=1e-9, atol=1e-12)
        assert actual_valid[:-2].all() and not actual_valid[-2:].any()
    tiny = _triangle() * 1e-6
    for tol in (1e-10, 1e-8):
        np.testing.assert_array_equal(gpu.propose_local(tiny, 1e-6, tol)[1],
                                      cpu.propose_local(tiny, 1e-6, tol)[1])
    assert gpu.propose_local(np.empty((0, 3, 3)), .1)[0].shape == (0, 6, 4, 3)
    assert gpu.kernel_calls["propose_local_tets"] >= 4
    assert gpu.timings["propose_local"] > 0


def test_spatial_targets_are_per_face_on_cpu_and_cuda(cuda_backend):
    # Identical faces isolate the effect of local size from their geometry.
    faces = np.repeat(_triangle(), 3, axis=0)
    sizes = np.array([.16, .32, .64])
    for engine in (CPUBackend(), cuda_backend):
        tets, valid = engine.propose_local(faces, sizes)
        assert valid.all()
        np.testing.assert_allclose(-tets[:, 0, 3, 2], np.sqrt(2./3.) * sizes)
        with pytest.raises(ValueError, match="per front face"):
            engine.propose_local(faces, sizes[:-1])
        with pytest.raises(ValueError, match="finite and positive"):
            engine.propose_local(faces, [1., .1, np.nan])
