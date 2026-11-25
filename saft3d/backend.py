"""CPU and CUDA geometry batches for the experimental SAFT-inspired mesher.

Proposal construction, quality evaluation and conflict/intersection checks are
accelerated here. Front management and acceptance remain on the CPU.
CUDA methods return NumPy arrays and include transfer/synchronization time in
``timings``. ``create_backend('cuda')`` always raises if CUDA cannot be used.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import Iterator

import numpy as np


def _tet_array(tets: np.ndarray) -> np.ndarray:
    """Validate and normalize the host-side batch representation."""
    result = np.ascontiguousarray(tets, dtype=np.float64)
    if result.ndim != 3 or result.shape[1:] != (4, 3):
        raise ValueError(f"Expected tetrahedra shaped (N, 4, 3), got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError("Tetrahedron coordinates must be finite")
    return result


def _nonnegative(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def _proposal_inputs(triangles, anchor, target_size, close_flags, tol):
    triangles = np.ascontiguousarray(triangles, dtype=np.float64)
    anchor = np.ascontiguousarray(anchor, dtype=np.float64)
    close_flags = np.ascontiguousarray(close_flags, dtype=np.uint8)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
        raise ValueError("Expected front triangles shaped (N, 3, 3)")
    if anchor.shape != (3,) or not np.isfinite(anchor).all() or not np.isfinite(triangles).all():
        raise ValueError("Front triangles and anchor must contain finite 3D points")
    if close_flags.shape != (len(triangles),):
        raise ValueError("close_flags must have shape (N,)")
    target_size = _nonnegative(target_size, "target_size")
    if target_size == 0.0:
        raise ValueError("target_size must be positive")
    return triangles, anchor, target_size, close_flags, _nonnegative(tol, "tol")


def _local_proposal_inputs(triangles, target_size, tol):
    triangles = np.ascontiguousarray(triangles, dtype=np.float64)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
        raise ValueError("Expected front triangles shaped (N, 3, 3)")
    if not np.isfinite(triangles).all():
        raise ValueError("Front triangles must contain finite 3D points")
    sizes = np.asarray(target_size, dtype=np.float64)
    if not np.isfinite(sizes).all() or np.any(sizes <= 0):
        raise ValueError("target_size must be finite and positive")
    if sizes.ndim == 0:
        sizes = np.full(len(triangles), sizes.item(), dtype=np.float64)
    if sizes.shape != (len(triangles),):
        raise ValueError("target_size must be a scalar or one value per front face")
    return triangles, np.ascontiguousarray(sizes), _nonnegative(tol, "tol")


class CPUBackend:
    """NumPy batches with a floating-point SAT overlap test (not exact arithmetic)."""

    name = "cpu"

    def __init__(self, fallback_reason: str | None = None) -> None:
        self.device_info = {"backend": "numpy", "device": "CPU"}
        self.fallback_reason = fallback_reason
        self.timings: dict[str, float] = {}
        self.calls: dict[str, int] = {}
        self.kernel_calls: dict[str, int] = {}

    @contextmanager
    def _timed(self, operation: str) -> Iterator[None]:
        start = perf_counter()
        try:
            yield
        finally:
            self.timings[operation] = self.timings.get(operation, 0.0) + perf_counter() - start
            self.calls[operation] = self.calls.get(operation, 0) + 1

    @property
    def stats(self) -> dict:
        """JSON-serializable execution metadata; seconds are host wall time."""
        return {
            "backend": self.name,
            "device_info": dict(self.device_info),
            "fallback_reason": self.fallback_reason,
            "timings_seconds": dict(self.timings),
            "calls": dict(self.calls),
            "kernel_calls": dict(self.kernel_calls),
        }

    def propose(self, triangles: np.ndarray, anchor: np.ndarray, target_size: float,
                close_flags: np.ndarray, tol: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
        """Construct six candidate slots per outward front face.

        Slots 0..2 move along the inward normal; slots 3..4 move toward the
        shared visibility-kernel anchor; slot 5 closes directly to that anchor.
        Growth preserves the anchor outside/on each of the three filled-tet
        side planes. Closure is selected only when requested or no growth is
        valid. Return coordinates (N,6,4,3) and a validity mask (N,6).
        """
        with self._timed("propose"):
            triangles, anchor, target_size, close_flags, tol = _proposal_inputs(
                triangles, anchor, target_size, close_flags, tol)
            n = len(triangles)
            result = np.empty((n, 6, 4, 3), dtype=np.float64)
            valid = np.zeros((n, 6), dtype=bool)
            if not n:
                return result, valid
            center = triangles.mean(axis=1)
            normal = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
            lengths = np.linalg.norm(normal, axis=1)
            normal = np.divide(normal, lengths[:, None], out=np.zeros_like(normal), where=lengths[:, None] > 0)
            depth = -np.einsum("ij,ij->i", normal, anchor - triangles[:, 0])
            ideal_h = np.minimum(np.sqrt(2.0 / 3.0) * target_size, 0.65 * depth)
            for slot, factor in enumerate((1.0, 0.75, 0.5)):
                result[:, slot, 3] = center - normal * (ideal_h * factor)[:, None]
            frac = np.minimum(np.divide(ideal_h, depth, out=np.zeros_like(depth), where=depth > 0), 0.65)
            result[:, 3, 3] = center + (anchor - center) * frac[:, None]
            result[:, 4, 3] = center + (anchor - center) * 0.35
            result[:, 5, 3] = anchor
            result[:, :, :3] = triangles[:, None, [0, 2, 1], :]
            u, v, w = (result[:, :, j] - result[:, :, 0] for j in (1, 2, 3))
            volumes = np.einsum("nsi,nsi->ns", np.cross(u, v), w) / 6.0
            valid[:, :5] = (volumes[:, :5] > tol * tol) & (~close_flags.astype(bool))[:, None]
            valid[:, :5] &= ((lengths > 0) & (depth > 0))[:, None]
            growth = result[:, :5]
            for face, other in (((0, 1, 3), 2), ((0, 3, 2), 1), ((1, 2, 3), 0)):
                x, y, z = (growth[:, :, j] for j in face)
                side = np.cross(y - x, z - x)
                inward = np.einsum("nsi,nsi->ns", side, growth[:, :, other] - x) > 0
                side = np.where(inward[:, :, None], -side, side)
                side_len = np.linalg.norm(side, axis=2)
                numer = np.einsum("nsi,nsi->ns", side, anchor - x)
                dist = np.divide(numer, side_len, out=np.zeros_like(numer), where=side_len > 0)
                valid[:, :5] &= (side_len > 0) & (dist >= -tol)
            valid[:, 5] = (close_flags.astype(bool) | ~valid[:, :5].any(axis=1)) & (volumes[:, 5] > tol * tol)
            return result, valid

    def propose_local(self, triangles: np.ndarray, target_size: float,
                      tol: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
        """Propose six inward-normal apexes without a shared kernel anchor.

        Heights use sqrt(2/3)*min(target_size, mean face edge length) times
        [1, .8, .6, .4, .2, .1]. Validity checks only volume > tol**2; the caller
        must check intersections, domain containment, quality and front topology.
        """
        with self._timed("propose_local"):
            triangles, target_size, tol = _local_proposal_inputs(triangles, target_size, tol)
            n = len(triangles)
            result = np.empty((n, 6, 4, 3), dtype=np.float64)
            if not n:
                return result, np.zeros((0, 6), dtype=bool)
            edge_a = triangles[:, 1] - triangles[:, 0]
            edge_b = triangles[:, 2] - triangles[:, 0]
            edge_c = triangles[:, 2] - triangles[:, 1]
            normal = np.cross(edge_a, edge_b)
            lengths = np.linalg.norm(normal, axis=1)
            normal = np.divide(normal, lengths[:, None], out=np.zeros_like(normal), where=lengths[:, None] > 0)
            mean_edge = (np.linalg.norm(edge_a, axis=1) + np.linalg.norm(edge_b, axis=1)
                         + np.linalg.norm(edge_c, axis=1)) / 3.0
            heights = np.sqrt(2.0 / 3.0) * np.minimum(target_size, mean_edge)
            factors = np.array([1., .8, .6, .4, .2, .1])
            result[:, :, :3] = triangles[:, None, [0, 2, 1], :]
            result[:, :, 3] = (triangles.mean(axis=1)[:, None, :]
                - normal[:, None, :] * heights[:, None, None] * factors[None, :, None])
            a, b, c = (result[:, :, j] - result[:, :, 0] for j in (1, 2, 3))
            volumes = np.einsum("nsi,nsi->ns", np.cross(a, b), c) / 6.0
            return result, volumes > tol * tol

    def evaluate(self, tets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return signed volumes and mean ratio (zero for nonpositive volume).

        Mean ratio is ``12 * (3 * V)**(2/3) / sum(squared edge lengths)``.
        A positively oriented regular tetrahedron has mean ratio one.
        """
        with self._timed("evaluate"):
            tets = _tet_array(tets)
            a = tets[:, 1] - tets[:, 0]
            b = tets[:, 2] - tets[:, 0]
            c = tets[:, 3] - tets[:, 0]
            volumes = np.einsum("ij,ij->i", np.cross(a, b), c) / 6.0
            edge_sum = np.zeros(len(tets), dtype=np.float64)
            for i in range(4):
                for j in range(i + 1, 4):
                    edge = tets[:, i] - tets[:, j]
                    edge_sum += np.einsum("ij,ij->i", edge, edge)
            quality = np.zeros(len(tets), dtype=np.float64)
            valid = (volumes > 0.0) & (edge_sum > 0.0)
            quality[valid] = 12.0 * np.power(3.0 * volumes[valid], 2.0 / 3.0) / edge_sum[valid]
            return volumes, quality

    def conflicts(self, tets: np.ndarray, padding: float = 0.0) -> np.ndarray:
        """Conservative AABB conflict matrix, including touching boxes.

        ``padding`` expands each box by this distance on each side. The diagonal
        is always false. True entries are broad-phase conflicts, not proof of
        tetrahedron overlap.
        """
        with self._timed("conflicts"):
            tets = _tet_array(tets)
            padding = _nonnegative(padding, "padding")
            lo = tets.min(axis=1) - padding
            hi = tets.max(axis=1) + padding
            result = np.all((lo[:, None] <= hi[None, :]) & (lo[None, :] <= hi[:, None]), axis=2)
            np.fill_diagonal(result, False)
            return result

    def overlaps(self, candidates: np.ndarray, existing: np.ndarray, tol: float = 1e-10) -> np.ndarray:
        """Flag candidates with positive-volume intersection with any existing tet.

        Touching on a face, edge, or vertex is allowed. ``tol`` is an absolute
        projected length tolerance in the caller's normalized coordinates.
        """
        from .geometry import tet_overlap

        with self._timed("overlaps"):
            candidates, existing = _tet_array(candidates), _tet_array(existing)
            tol = _nonnegative(tol, "tol")
            result = np.zeros(len(candidates), dtype=bool)
            if not len(candidates) or not len(existing):
                return result
            old_lo, old_hi = existing.min(axis=1), existing.max(axis=1)
            for i, candidate in enumerate(candidates):
                lo, hi = candidate.min(axis=0), candidate.max(axis=0)
                broad = np.all(np.minimum(hi, old_hi) - np.maximum(lo, old_lo) > tol, axis=1)
                for j in np.flatnonzero(broad):
                    if tet_overlap(candidate, existing[j], tol=tol):
                        result[i] = True
                        break
            return result


class CUDABackend(CPUBackend):
    """CuPy RawModule kernels using float64, with explicit initialization failure."""

    name = "cuda"

    def __init__(self) -> None:
        super().__init__()
        try:
            import cupy as cp

            self.cp = cp
            if cp.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError("No CUDA devices detected")
            self.device = cp.cuda.Device()
            self.device.use()
            props = cp.cuda.runtime.getDeviceProperties(self.device.id)
            device_name = props["name"]
            if isinstance(device_name, bytes):
                device_name = device_name.decode("utf-8", errors="replace")
            free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
            self.device_info = {
                "backend": "cupy RawModule / NVRTC",
                "device": device_name,
                "device_id": self.device.id,
                "compute_capability": f"{props['major']}.{props['minor']}",
                "total_memory_bytes": int(total_bytes),
                "free_memory_bytes_at_init": int(free_bytes),
                "cupy_version": cp.__version__,
                "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
                "cuda_driver_version": int(cp.cuda.runtime.driverGetVersion()),
            }
            source = Path(__file__).with_name("kernels.cu").read_text(encoding="utf-8")
            self.module = cp.RawModule(code=source, options=("--std=c++11",), backend="nvrtc")
            self.module.compile()
            self._kernels = {
                name: self.module.get_function(name)
                for name in ("propose_tets", "propose_local_tets", "evaluate_tets", "aabb_conflicts", "tet_overlaps")
            }
            # Launch and synchronize once so auto mode also catches unusable
            # device binaries, rather than failing halfway through generation.
            self.evaluate(np.array([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]]))
            self.timings.clear()
            self.calls.clear()
            self.kernel_calls.clear()
        except Exception as exc:
            raise RuntimeError(f"CUDA backend unavailable: {type(exc).__name__}: {exc}") from exc

    def _launch(self, kernel: str, count: int, args: tuple) -> None:
        if not count:
            return
        threads = 128
        self._kernels[kernel](((count + threads - 1) // threads,), (threads,), args)
        self.kernel_calls[kernel] = self.kernel_calls.get(kernel, 0) + 1

    def propose(self, triangles: np.ndarray, anchor: np.ndarray, target_size: float,
                close_flags: np.ndarray, tol: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
        with self._timed("propose"), self.device:
            triangles, anchor, target_size, close_flags, tol = _proposal_inputs(
                triangles, anchor, target_size, close_flags, tol)
            cp = self.cp
            n = len(triangles)
            if not n:
                return np.empty((0, 6, 4, 3), dtype=np.float64), np.zeros((0, 6), dtype=bool)
            gpu_triangles, gpu_anchor, gpu_flags = cp.asarray(triangles), cp.asarray(anchor), cp.asarray(close_flags)
            result = cp.empty((n, 6, 4, 3), dtype=cp.float64)
            valid = cp.empty((n, 6), dtype=cp.uint8)
            self._launch("propose_tets", n, (gpu_triangles, np.int64(n), gpu_anchor,
                np.float64(target_size), gpu_flags, np.float64(tol), result, valid))
            return cp.asnumpy(result), cp.asnumpy(valid).astype(bool)

    def propose_local(self, triangles: np.ndarray, target_size: float,
                      tol: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
        with self._timed("propose_local"), self.device:
            triangles, target_size, tol = _local_proposal_inputs(triangles, target_size, tol)
            cp = self.cp
            n = len(triangles)
            if not n:
                return np.empty((0, 6, 4, 3), dtype=np.float64), np.zeros((0, 6), dtype=bool)
            gpu_triangles = cp.asarray(triangles)
            result = cp.empty((n, 6, 4, 3), dtype=cp.float64)
            valid = cp.empty((n, 6), dtype=cp.uint8)
            self._launch("propose_local_tets", n, (gpu_triangles, np.int64(n),
                cp.asarray(target_size), np.float64(tol), result, valid))
            return cp.asnumpy(result), cp.asnumpy(valid).astype(bool)

    def evaluate(self, tets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        with self._timed("evaluate"), self.device:
            tets = _tet_array(tets)
            cp = self.cp
            n = len(tets)
            if not n:
                return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
            gpu_tets = cp.asarray(tets)
            volumes, quality = cp.empty(n, dtype=cp.float64), cp.empty(n, dtype=cp.float64)
            self._launch("evaluate_tets", n, (gpu_tets, np.int64(n), volumes, quality))
            return cp.asnumpy(volumes), cp.asnumpy(quality)

    def conflicts(self, tets: np.ndarray, padding: float = 0.0) -> np.ndarray:
        with self._timed("conflicts"), self.device:
            tets = _tet_array(tets)
            padding = _nonnegative(padding, "padding")
            cp = self.cp
            n = len(tets)
            if not n:
                return np.zeros((0, 0), dtype=bool)
            gpu_tets = cp.asarray(tets)
            result = cp.empty((n, n), dtype=cp.uint8)
            self._launch("aabb_conflicts", n * n, (gpu_tets, np.int64(n), np.float64(padding), result))
            return cp.asnumpy(result).astype(bool)

    def overlaps(self, candidates: np.ndarray, existing: np.ndarray, tol: float = 1e-10) -> np.ndarray:
        with self._timed("overlaps"), self.device:
            candidates, existing = _tet_array(candidates), _tet_array(existing)
            tol = _nonnegative(tol, "tol")
            cp = self.cp
            n, m = len(candidates), len(existing)
            if not n or not m:
                return np.zeros(n, dtype=bool)
            gpu_candidates, gpu_existing = cp.asarray(candidates), cp.asarray(existing)
            result = cp.empty(n, dtype=cp.uint8)
            self._launch("tet_overlaps", n, (gpu_candidates, np.int64(n), gpu_existing, np.int64(m), np.float64(tol), result))
            return cp.asnumpy(result).astype(bool)


def create_backend(name: str = "auto") -> CPUBackend:
    """Select cpu/cuda/auto; only auto may fall back, recording its reason."""
    name = name.lower().strip()
    if name == "cpu":
        return CPUBackend()
    if name == "cuda":
        return CUDABackend()
    if name == "auto":
        try:
            return CUDABackend()
        except RuntimeError as exc:
            return CPUBackend(fallback_reason=str(exc))
    raise ValueError(f"Unknown backend {name!r}; choose cpu, cuda, or auto")


def cuda_smoke_test() -> dict:
    """Exercise all CUDA kernels; raise on errors and return execution metadata.

    Invoke with ``python -m saft3d.backend``. This checks a regular tetrahedron,
    a contained tetrahedron, a shared-face neighbor, and a separated tetrahedron.
    """
    regular = np.array([[0., 0., 0.], [1., 0., 0.], [.5, np.sqrt(3.) / 2., 0.],
                        [.5, np.sqrt(3.) / 6., np.sqrt(2. / 3.)]])
    neighbor = regular.copy()
    neighbor[3, 2] *= -1.
    inner = regular.mean(axis=0) + .2 * (regular - regular.mean(axis=0))
    batch = np.stack([regular, inner, neighbor, regular + 3.])
    cpu, gpu = CPUBackend(), CUDABackend()
    cpu_v, cpu_q = cpu.evaluate(batch)
    gpu_v, gpu_q = gpu.evaluate(batch)
    np.testing.assert_allclose(gpu_v, cpu_v, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(gpu_q, cpu_q, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(gpu_q[0], 1., rtol=1e-12)
    np.testing.assert_array_equal(gpu.conflicts(batch), cpu.conflicts(batch))
    expected = np.array([True, True, False, False])
    np.testing.assert_array_equal(gpu.overlaps(batch, regular[None]), expected)
    outward = regular[[0, 2, 1]][None]
    anchor = regular.mean(axis=0)
    cpu_p, cpu_valid = cpu.propose(np.repeat(outward, 2, axis=0), anchor, 0.1, [False, True])
    gpu_p, gpu_valid = gpu.propose(np.repeat(outward, 2, axis=0), anchor, 0.1, [False, True])
    np.testing.assert_allclose(gpu_p, cpu_p, rtol=1e-12, atol=1e-14)
    np.testing.assert_array_equal(gpu_valid, cpu_valid)
    assert gpu_valid[0, :5].any() and not gpu_valid[0, 5]
    assert not gpu_valid[1, :5].any() and gpu_valid[1, 5]
    return gpu.stats


if __name__ == "__main__":
    import json

    print(json.dumps(cuda_smoke_test(), indent=2))
