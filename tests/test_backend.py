"""Small CPU/CUDA regressions for the four geometry kernels.

CUDA tests skip only if CuPy is absent or CUDA reports no usable device/driver.
Unexpected import errors, NVRTC compilation errors and kernel failures fail the
tests; they are deliberately not disguised as an unavailable GPU.
"""

import importlib.util

import numpy as np
import pytest

import saft3d.backend as backend_module
from saft3d.backend import CPUBackend, create_backend


@pytest.fixture(scope="module")
def cuda_backend():
    if importlib.util.find_spec("cupy") is None:
        pytest.skip("CUDA unavailable: CuPy is not installed")
    import cupy as cp

    try:
        count = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError as exc:
        # CUDA's initialization error, insufficient driver and no-device codes.
        if exc.status in (3, 35, 100):
            pytest.skip(f"CUDA unavailable: {exc}")
        raise
    if count == 0:
        pytest.skip("CUDA unavailable: no device detected")
    return create_backend("cuda")


def _regular_tet():
    return np.array([[0., 0., 0.], [1., 0., 0.],
                     [.5, np.sqrt(3.) / 2., 0.],
                     [.5, np.sqrt(3.) / 6., np.sqrt(2. / 3.)]])


def _front_batch(count=24):
    rng = np.random.default_rng(20261029)
    normals = rng.normal(size=(count, 3))
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    u = np.cross(normals, [0., 0., 1.])
    u /= np.linalg.norm(u, axis=1)[:, None]
    v = np.cross(normals, u)
    center = normals * .4
    triangles = np.stack([center + .12 * u, center + .08 * v,
                          center - .12 * u - .08 * v], axis=1)
    return triangles, rng.random(count) < .3


def test_cpu_signed_volume_regular_quality_and_degeneracy():
    regular = _regular_tet()
    inverted = regular[[0, 2, 1, 3]]
    flat = regular.copy()
    flat[3, 2] = 0.
    volumes, quality = CPUBackend().evaluate(np.stack([regular, inverted, flat]))
    np.testing.assert_allclose(volumes, [np.sqrt(2.) / 12., -np.sqrt(2.) / 12., 0.],
                               rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(quality, [1., 0., 0.], atol=1e-13)


def test_cpu_empty_inputs_and_validation():
    cpu = CPUBackend()
    empty = np.empty((0, 4, 3))
    assert cpu.evaluate(empty)[0].shape == (0,)
    assert cpu.conflicts(empty).shape == (0, 0)
    assert cpu.overlaps(empty, _regular_tet()[None]).shape == (0,)
    np.testing.assert_array_equal(cpu.overlaps(_regular_tet()[None], empty), [False])
    proposals, valid = cpu.propose(np.empty((0, 3, 3)), np.zeros(3), .1, [])
    assert proposals.shape == (0, 6, 4, 3)
    assert valid.shape == (0, 6)
    for bad in (np.zeros((4, 3)), np.full((1, 4, 3), np.nan)):
        with pytest.raises(ValueError):
            cpu.evaluate(bad)
    with pytest.raises(ValueError, match="nonnegative"):
        cpu.conflicts(_regular_tet()[None], padding=-1.)
    with pytest.raises(ValueError, match="positive"):
        cpu.propose(np.zeros((1, 3, 3)), np.zeros(3), 0., [False])


@pytest.mark.parametrize("name", ["cuda:no-such-device", "cuda999", "unobtainium"])
def test_unknown_backend_name_fails(name):
    with pytest.raises(ValueError, match="Unknown backend"):
        create_backend(name)


def test_auto_records_fallback_but_explicit_cuda_raises(monkeypatch):
    def unavailable():
        raise RuntimeError("CUDA backend unavailable: simulated missing CUDA driver")

    monkeypatch.setattr(backend_module, "CUDABackend", unavailable)
    auto = create_backend("auto")
    assert auto.name == "cpu"
    assert "simulated missing CUDA driver" in auto.fallback_reason
    assert auto.stats["fallback_reason"] == auto.fallback_reason
    with pytest.raises(RuntimeError, match="simulated missing CUDA driver"):
        create_backend("cuda")
    assert create_backend("cpu").fallback_reason is None


def test_cuda_all_four_kernels_match_cpu(cuda_backend):
    cpu, gpu = CPUBackend(), cuda_backend
    triangles, close = _front_batch()
    for size in (.03, .4):
        expected, expected_valid = cpu.propose(triangles, np.zeros(3), size, close)
        actual, actual_valid = gpu.propose(triangles, np.zeros(3), size, close)
        np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-13)
        np.testing.assert_array_equal(actual_valid, expected_valid)
        # Closed faces use just the shared anchor slot, never growth slots.
        assert actual_valid[close, 5].all()
        assert not actual_valid[close, :5].any()
        ev, eq = cpu.evaluate(expected.reshape(-1, 4, 3))
        av, aq = gpu.evaluate(actual.reshape(-1, 4, 3))
        np.testing.assert_allclose(av, ev, rtol=1e-10, atol=1e-14)
        np.testing.assert_allclose(aq, eq, rtol=1e-10, atol=1e-13)

    rng = np.random.default_rng(619)
    centers = np.array([[i % 4, i // 4, 0] for i in range(16)]) * .2
    candidates = centers[:, None, :] + .025 * rng.normal(size=(16, 4, 3))
    existing = centers[:, None, :] + .025 * rng.normal(size=(16, 4, 3))
    for padding in (0., .01):
        expected = cpu.conflicts(candidates, padding)
        actual = gpu.conflicts(candidates, padding)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual, actual.T)
        assert not np.diag(actual).any()
    for tolerance in (0., 1e-10, 1e-5):
        np.testing.assert_array_equal(gpu.overlaps(candidates, existing, tolerance),
                                      cpu.overlaps(candidates, existing, tolerance))
    for kernel in ("propose_tets", "evaluate_tets", "aabb_conflicts", "tet_overlaps"):
        assert gpu.kernel_calls[kernel] > 0
    assert gpu.name == "cuda" and gpu.fallback_reason is None


def test_cuda_near_contact_distinguishes_tolerance_band(cuda_backend):
    cpu, gpu = CPUBackend(), cuda_backend
    tet = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    tolerance = 1e-9
    neighbor = tet.copy()
    neighbor[3, 2] = -1.
    candidates = np.stack([neighbor + [0., 0., delta]
                           for delta in (0., -8 * tolerance, .25 * tolerance, 8 * tolerance)])
    expected = np.array([False, False, False, True])
    np.testing.assert_array_equal(cpu.overlaps(candidates, tet[None], tolerance), expected)
    np.testing.assert_array_equal(gpu.overlaps(candidates, tet[None], tolerance), expected)
    # AABB broad phase includes the exactly touching pair even though SAT does not.
    assert gpu.conflicts(np.stack([tet, neighbor]))[0, 1]


def test_cuda_degenerate_and_near_parallel_sat(cuda_backend):
    cpu, gpu = CPUBackend(), cuda_backend
    tet = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    tolerance = 1e-9
    for height, expected in ((0., False), (.25 * tolerance, False), (8 * tolerance, True)):
        thin = tet.copy()
        thin[3, 2] = height
        for engine in (cpu, gpu):
            actual = engine.overlaps(thin[None], thin[None], tolerance)
            np.testing.assert_array_equal(actual, [expected])
    variants = []
    for angle in (0., 1e-16, 1e-14, 1e-10):
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
        variants.append(tet @ rotation.T + .1)
    variants = np.stack(variants)
    expected = np.ones(len(variants), dtype=bool)
    np.testing.assert_array_equal(cpu.overlaps(variants, tet[None], tolerance), expected)
    np.testing.assert_array_equal(gpu.overlaps(variants, tet[None], tolerance), expected)
    inverted = tet[[0, 2, 1, 3]][None]
    np.testing.assert_array_equal(gpu.overlaps(inverted, tet[None]), [True])
    np.testing.assert_array_equal(gpu.evaluate(inverted)[1], [0.])


def test_cuda_proposal_degeneracy_and_forced_closure(cuda_backend):
    cpu, gpu = CPUBackend(), cuda_backend
    outward = np.array([[[0., 0., 0.], [0., 1., 0.], [1., 0., 0.]]])
    # Closure volume = 1.2*tol^2; every partial growth volume is below tol^2.
    tolerance = 1e-3
    anchor = np.array([.2, .2, 7.2 * tolerance**2])
    for engine in (cpu, gpu):
        proposals, valid = engine.propose(outward, anchor, 1., [False], tolerance)
        np.testing.assert_array_equal(valid, [[False, False, False, False, False, True]])
        np.testing.assert_array_equal(proposals[0, 5, 3], anchor)
        _, flat_valid = engine.propose(np.zeros((1, 3, 3)), anchor, .1, [False], tolerance)
        assert not flat_valid.any()
        assert engine.propose(np.empty((0, 3, 3)), anchor, .1, [])[0].shape == (0, 6, 4, 3)
        assert engine.evaluate(np.empty((0, 4, 3)))[0].shape == (0,)
        assert engine.conflicts(np.empty((0, 4, 3))).shape == (0, 0)
