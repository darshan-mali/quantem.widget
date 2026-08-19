"""choose_lattice: pick an ordered origin + two lattice-vector points on an image
and fit a quantem.imaging.Lattice from them.

A focused sibling of Show2D: displays a single 2D image, lets the user
wheel-zoom/drag-pan to inspect a region, and lets them click 3 ordered points
(origin, a1, a2 by default) whose pixel coordinates (in the ORIGINAL,
un-zoomed image) are exposed for downstream lattice calculations, along with
the derived lattice vectors ``u = a1 - origin`` and ``v = a2 - origin``.

When ``quantem.imaging`` is importable, the picked points can be handed
straight to a ``Lattice``: ``fit_lattice()`` refines the vectors,
``detect_atoms()`` tiles the image with lattice sites, and ``refine_atoms()``
fits each atom centre. Results are synced back to the front end as overlays.

Fitting also builds an averaged unit cell: every complete cell in the image
sampled onto one small tile. Because that tile has far better contrast than
any single cell, it is what makes the unit-cell sites pickable by eye - a
click anywhere in the image (or in the tile itself) is converted to a
fractional (a, b) coordinate and snapped to the nearest column in the tile,
which is far more precise than the raw click.
"""

import base64
import pathlib
from typing import Any, Sequence

import anywidget
import matplotlib
import numpy as np
import traitlets
from scipy import ndimage
from quantem.widget.utils.array import to_numpy
from quantem.widget.utils.static_fallback import StaticFallbackMixin

_CORE_IMAGE_DATASET_IMPORT_ATTEMPTED = False
_CORE_IMAGE_DATASET_TYPES: tuple[type[Any], ...] = ()

_IMAGING_LATTICE_IMPORT_ATTEMPTED = False
_IMAGING_LATTICE_TYPE: type[Any] | None = None


def _core_image_dataset_types() -> tuple[type[Any], ...]:
    """Return core image dataset classes when quantem.core is importable."""
    global _CORE_IMAGE_DATASET_IMPORT_ATTEMPTED, _CORE_IMAGE_DATASET_TYPES
    if not _CORE_IMAGE_DATASET_IMPORT_ATTEMPTED:
        _CORE_IMAGE_DATASET_IMPORT_ATTEMPTED = True
        try:
            from quantem.core.datastructures import Dataset2d
        except Exception:
            _CORE_IMAGE_DATASET_TYPES = ()
        else:
            _CORE_IMAGE_DATASET_TYPES = (Dataset2d,)
    return _CORE_IMAGE_DATASET_TYPES


def _imaging_lattice_type() -> type[Any] | None:
    """Return ``quantem.imaging.Lattice`` when importable, else None."""
    global _IMAGING_LATTICE_IMPORT_ATTEMPTED, _IMAGING_LATTICE_TYPE
    if not _IMAGING_LATTICE_IMPORT_ATTEMPTED:
        _IMAGING_LATTICE_IMPORT_ATTEMPTED = True
        try:
            from quantem.imaging import Lattice
        except Exception:
            _IMAGING_LATTICE_TYPE = None
        else:
            _IMAGING_LATTICE_TYPE = Lattice
    return _IMAGING_LATTICE_TYPE


def _require_lattice_type() -> type[Any]:
    """Return the Lattice class or raise a user-facing ImportError."""
    lattice_type = _imaging_lattice_type()
    if lattice_type is None:
        raise ImportError(
            "Lattice fitting requires `quantem.imaging`, which is not importable "
            "in this environment. Install quantem to use fit_lattice()/detect_atoms()."
        )
    return lattice_type


def _reject_unknown_kwargs(cls, kwargs: dict) -> None:
    """Raise TypeError for any kwarg that isn't a declared trait (catches typos)."""
    traits = set(cls.class_trait_names())
    unknown = [k for k in kwargs if k not in traits]
    if unknown:
        key = sorted(unknown)[0]
        raise TypeError(f"{cls.__name__}() got unexpected keyword argument {key!r}.")


def _frame_to_rgb(
    frame: np.ndarray,
    *,
    cmap: str,
    vmin: float | None,
    vmax: float | None,
    log_scale: bool,
) -> np.ndarray:
    """Colormap a 2D float frame into a uint8 (H, W, 3) RGB array."""
    values = frame.astype(np.float64, copy=False)
    if log_scale:
        values = np.log1p(np.clip(values - np.nanmin(values), 0, None))
    lo = float(np.nanpercentile(values, 1)) if vmin is None else float(vmin)
    hi = float(np.nanpercentile(values, 99)) if vmax is None else float(vmax)
    if hi <= lo:
        hi = lo + 1.0
    normalized = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
    colormap = matplotlib.colormaps[cmap]
    rgba = colormap(normalized)
    return (rgba[..., :3] * 255).astype(np.uint8)


def _rgb_to_png_bytes(rgb: np.ndarray) -> bytes:
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def _lattice_index_bounds(lat: np.ndarray, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    """Integer (n, m) range covering the image, solving corner = r0 + n*u + m*v."""
    r0, u, v = lat
    height, width = shape
    corners = np.array(
        [[0, 0], [0, width - 1], [height - 1, 0], [height - 1, width - 1]], dtype=float
    )
    det = u[0] * v[1] - u[1] * v[0]
    offsets = corners - r0
    n_idx = (offsets[:, 0] * v[1] - offsets[:, 1] * v[0]) / det
    m_idx = (offsets[:, 1] * u[0] - offsets[:, 0] * u[1]) / det
    return (
        int(np.floor(n_idx.min())),
        int(np.ceil(n_idx.max())),
        int(np.floor(m_idx.min())),
        int(np.ceil(m_idx.max())),
    )


def average_unit_cell(
    image: np.ndarray,
    lat: np.ndarray,
    samples: int = 64,
    max_cells: int = 2000,
    seed: int = 0,
) -> tuple[np.ndarray, int]:
    """Average every fully-contained unit cell onto one ``(samples, samples)`` tile.

    Tile rows run along ``u`` and columns along ``v``, so tile index
    ``(i, j)`` is the fractional position
    ``((i + 0.5) / samples, (j + 0.5) / samples)``.

    Cells beyond ``max_cells`` are subsampled at random: tile contrast
    saturates after a couple of thousand cells while cost keeps growing
    linearly with them, so a 2048-pixel image averages in about two seconds
    instead of seven with no measurable loss.

    Returns the tile and the number of cells actually averaged.
    """
    empty = np.zeros((samples, samples), dtype=np.float32)
    r0, u, v = np.asarray(lat, dtype=float)
    height, width = image.shape

    n_lo, n_hi, m_lo, m_hi = _lattice_index_bounds(np.array([r0, u, v]), (height, width))
    n_grid, m_grid = np.meshgrid(
        np.arange(n_lo, n_hi + 1), np.arange(m_lo, m_hi + 1), indexing="ij"
    )
    cells = np.stack([n_grid.ravel(), m_grid.ravel()], axis=1).astype(float)
    if cells.shape[0] == 0:
        return empty, 0

    # Containment is tested on the four cell corners only, so the full sample
    # grid is built once for the cells that survive rather than for every
    # candidate cell in the bounding box.
    cell_corners = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
    corner_pos = (
        r0
        + (cells[:, None, 0] + cell_corners[None, :, 0])[..., None] * u
        + (cells[:, None, 1] + cell_corners[None, :, 1])[..., None] * v
    )
    inside = (
        (corner_pos[..., 0] >= 0)
        & (corner_pos[..., 0] <= height - 1)
        & (corner_pos[..., 1] >= 0)
        & (corner_pos[..., 1] <= width - 1)
    ).all(axis=1)
    cells = cells[inside]
    if cells.shape[0] == 0:
        return empty, 0
    if max_cells and cells.shape[0] > max_cells:
        chosen = np.random.default_rng(seed).choice(cells.shape[0], max_cells, replace=False)
        cells = cells[chosen]

    axis = (np.arange(samples) + 0.5) / samples
    frac_a, frac_b = np.meshgrid(axis, axis, indexing="ij")
    n_col = cells[:, 0][:, None, None]
    m_col = cells[:, 1][:, None, None]
    rows = r0[0] + (n_col + frac_a) * u[0] + (m_col + frac_b) * v[0]
    cols = r0[1] + (n_col + frac_a) * u[1] + (m_col + frac_b) * v[1]
    stack = ndimage.map_coordinates(
        image.astype(float), [rows.ravel(), cols.ravel()], order=1, mode="nearest"
    ).reshape(rows.shape)
    return stack.mean(axis=0).astype(np.float32), int(cells.shape[0])


# Fractional (a, b) positions common enough in real lattices that a snapped
# click landing near one is almost certainly meant to be exactly there
# (corner, edge, and body-diagonal sites of a simple/centered/hexagonal cell).
COMMON_SITES = np.array(
    [
        [0.0, 0.0],
        [0.5, 0.0],
        [0.0, 0.5],
        [0.5, 0.5],
        [0.333, 0.333],
        [0.667, 0.667],
    ]
)
COMMON_SITE_TOL = 0.05


def snap_to_common_site(frac: Sequence[float], tol: float = COMMON_SITE_TOL) -> np.ndarray:
    """Snap to the nearest ``COMMON_SITES`` entry if within ``tol`` on both axes.

    ``frac`` is clamped into ``[0, 1]`` rather than wrapped, so a click
    outside the cell (e.g. in the 3x3 inset's surrounding context) snaps
    toward the nearest edge/corner of the cell instead of jumping to the
    opposite side. Once inside the cell, distances to ``COMMON_SITES`` still
    wrap across the cell boundary (e.g. 0.98 is 0.02 from 0.0), so a click
    near the edge of the tile still snaps to a corner site.
    """
    frac = np.clip(np.asarray(frac, dtype=float), 0.0, 1.0)
    delta = ((frac - COMMON_SITES + 0.5) % 1.0) - 0.5
    within = (np.abs(delta) < tol).all(axis=1)
    if not within.any():
        return frac
    candidates = np.where(within)[0]
    nearest = candidates[np.argmin((delta[candidates] ** 2).sum(axis=1))]
    return COMMON_SITES[nearest].copy()


def snap_frac(
    tile: np.ndarray,
    frac: Sequence[float],
    search_frac: float = 0.15,
    centroid_frac: float = 0.06,
) -> np.ndarray:
    """Snap a fractional position to the nearest column in an averaged cell.

    ``frac`` is clamped into ``[0, 1]`` rather than wrapped, so a click
    outside the cell (e.g. in the 3x3 inset's surrounding context) snaps
    toward the nearest edge/corner of the cell instead of jumping to the
    opposite side. From there it takes the brightest sample within
    ``search_frac`` of the click, then an intensity centroid in a tighter
    window around it; those searches still wrap across the cell boundary, so
    a click near an edge snaps to a column on the far side instead of being
    dragged inward - the cell itself is genuinely periodic, unlike the space
    outside the panel's box.
    """
    frac = np.clip(np.asarray(frac, dtype=float), 0.0, 1.0)
    if tile.size == 0:
        return frac
    samples = tile.shape[0]
    axis = (np.arange(samples) + 0.5) / samples
    grid_a, grid_b = np.meshgrid(axis, axis, indexing="ij")

    delta_a = (grid_a - frac[0] + 0.5) % 1.0 - 0.5
    delta_b = (grid_b - frac[1] + 0.5) % 1.0 - 0.5
    search = (delta_a**2 + delta_b**2) <= search_frac**2
    if not search.any():
        return frac
    peak = int(np.argmax(np.where(search, tile, -np.inf)))
    peak_a, peak_b = float(delta_a.ravel()[peak]), float(delta_b.ravel()[peak])

    local_a = (grid_a - (frac[0] + peak_a) + 0.5) % 1.0 - 0.5
    local_b = (grid_b - (frac[1] + peak_b) + 0.5) % 1.0 - 0.5
    window = (local_a**2 + local_b**2) <= centroid_frac**2
    weights = np.clip(tile - tile[window].min(), 0, None) * window
    total = float(weights.sum())
    if total <= 0:
        return np.array([(frac[0] + peak_a) % 1.0, (frac[1] + peak_b) % 1.0])
    return np.array(
        [
            (frac[0] + peak_a + float((weights * local_a).sum()) / total) % 1.0,
            (frac[1] + peak_b + float((weights * local_b).sum()) / total) % 1.0,
        ]
    )


def propose_sites(
    tile: np.ndarray,
    min_sep_frac: float = 0.12,
    rel_threshold: float = 0.05,
    max_sites: int = 8,
) -> np.ndarray:
    """Candidate site fractions from wrap-aware local maxima of an averaged cell.

    ``rel_threshold`` is a fraction of the tile's own dynamic range rather
    than an absolute intensity, so weak columns are still proposed alongside
    the brightest site instead of being scaled out by it.
    """
    if tile.size == 0:
        return np.zeros((0, 2))
    samples = tile.shape[0]
    size = max(3, int(round(min_sep_frac * samples)) | 1)
    maxima = ndimage.maximum_filter(tile, size=size, mode="wrap")
    low, high = float(tile.min()), float(tile.max())
    peaks = (tile >= maxima) & (tile >= low + rel_threshold * (high - low))
    coords = np.argwhere(peaks)
    if coords.size == 0:
        return np.zeros((0, 2))
    coords = coords[np.argsort(-tile[peaks])]

    kept: list[np.ndarray] = []
    for index in coords:
        candidate = (index + 0.5) / samples
        if any(
            np.hypot(*(((candidate - other + 0.5) % 1.0) - 0.5)) < min_sep_frac
            for other in kept
        ):
            continue
        kept.append(candidate)
        if len(kept) >= max_sites:
            break
    return np.array(kept) if kept else np.zeros((0, 2))


class ChooseLattice(StaticFallbackMixin, anywidget.AnyWidget):
    """Interactive picker for an ordered origin + two lattice-vector points.

    Parameters
    ----------
    data : array_like, quantem Dataset2d, or quantem.imaging.Lattice
        A single 2D image. NumPy, PyTorch, CuPy arrays, a quantem
        ``Dataset2d`` (its ``.array``/``.name`` are auto-extracted), or an
        existing ``Lattice`` (its image is displayed and the object is
        reused, so an already-fitted lattice is shown immediately).
    cmap : str, default "gray"
        Matplotlib colormap name used to render the image.
    vmin, vmax : float, optional
        Explicit display range. Defaults to a robust 1st/99th percentile
        auto-contrast when not given.
    log_scale : bool, default False
        Apply a log1p display stretch before contrast scaling.
    title : str, default ""
        Title shown above the image. Defaults to a quantem Dataset2d's
        ``.name`` when not given explicitly.
    point_labels : sequence of str, default ("Origin", "u", "v")
        Labels for the 3 points, in placement order. The first label is
        shown next to the raw origin point; the second and third are shown
        next to the derived lattice vectors (see ``u`` and ``v`` below),
        not the raw pixel positions of the 2nd/3rd clicks.
    points : sequence of (row, col), optional
        Initial picked points, in placement order.
    positions_frac : array_like, shape (S, 2), default ((0.0, 0.0),)
        Fractional (a, b) positions of the S sites in the unit cell, passed
        to ``Lattice.add_atoms``.
    refine_lattice : bool, default True
        Whether ``fit_lattice()`` refines origin/u/v by maximizing the
        bilinear intensity sum.
    block_size : int, optional
        Block size for staged lattice refinement, default 5. Exposed in the
        widget as a slider from 1 to 10 with a "None" position past the top
        end. Pass None (or 0) to fit the whole image at once.
    cell_samples : int, default 64
        Edge length of the averaged unit-cell tile built after each fit.
        32 is faster and coarser, 128 costs about five times as much for no
        useful gain in snapping precision.
    cell_max_cells : int, default 2000
        Cap on how many unit cells are averaged into that tile.
    edge_min_dist_px : float, default 6.0
        Minimum distance from the image border for a detected atom, passed
        to ``Lattice.add_atoms``. Keeps atoms whose fit patch would be
        clipped or empty out of ``refine_atoms``, whose amplitude bounds
        collapse on a flat patch and raise from ``least_squares``.
    max_sites : int, optional
        Cap on how many sites ``propose_sites_from_cell()`` (the "Auto
        Sites" button) proposes from the averaged cell. Exposed in the
        widget as a slider from "Auto" (the default, ``None``, which lets
        the algorithm decide) to 10. When a site count is set (not "Auto")
        and exactly that many sites are placed, the front end lets sites be
        dragged around the averaged-cell inset to refine their position -
        see ``snap_to_common_sites`` below for how that interacts with
        snapping.
    snap_to_common_sites : bool, default True
        Whether ``snap_site()`` (and ``propose_sites_from_cell()``) pulls a
        snapped position onto the nearest ``COMMON_SITES`` fraction (corner/
        edge/diagonal, e.g. 0.5 or 0.333) when it lands close to one.
        Exposed in the widget as a toggle. Disabling it still snaps to the
        nearest bright column in the averaged cell, just without the extra
        pull toward a "clean" fraction. Dragging a site in the inset (see
        ``max_sites``) always places it exactly under the cursor regardless
        of this setting - snapping of either kind only applies to clicks.
    true_cell_geometry : bool, default False
        Whether the averaged-cell inset renders as a true parallelogram
        matching the fitted ``u``/``v`` angle, instead of a plain square
        that silently assumes ``u`` and ``v`` are orthogonal. The square
        rendering is a close approximation for a near-square lattice but a
        poor one for e.g. a hexagonal lattice (~60-120 degrees between
        ``u`` and ``v``), where it visibly distorts the true cell shape.
        Purely a front-end display setting - has no effect on
        ``positions_frac``, snapping, or detection. Exposed in the widget
        as a toggle, off by default to preserve the existing look unless
        opted into.
    save_state : bool, default False
        Embed full interactive state in the notebook so a cold reopen
        restores the picked points. See ``StaticFallbackMixin`` for the
        image-only fallback used otherwise.
    notebook_preview_format : {"jpeg", "webp", "png"} or None, default None
        Static preview format used when ``save_state=False``. Defaults to
        ``None`` (no fallback image): unlike Show2D/Show3D, ChooseLattice's
        live widget does not reliably hide the saved-notebook fallback
        sibling while interactive, so enabling it shows a redundant image
        alongside the live widget. Opt in explicitly if a cold-reopen
        preview is worth that tradeoff.
    notebook_preview_quality : int, default 88
        Lossy preview quality for JPEG/WebP, from 1 to 100. Ignored for PNG.
    notebook_preview_max_px : int, default 512
        Longest image side for the saved-notebook preview.

    Notes
    -----
    Click on the image to place points in order; once 3 are placed, click
    near an existing point and drag to adjust it. Use ``clear_points()`` (or
    the "Clear Points" button) to start over. Pixel coordinates are always
    reported in the ORIGINAL image's ``(row, col)`` space, regardless of the
    current zoom/pan. The ``u`` and ``v`` properties expose the lattice
    vectors ``a1 - origin`` and ``a2 - origin``.

    ``fit_lattice()`` hands those vectors to ``quantem.imaging.Lattice``;
    the refined ``r0, u, v`` are exposed as ``lattice_vectors`` and drawn as
    a grid, and ``detect_atoms()``/``refine_atoms()`` overlay atom centres.
    The underlying object is available as ``.lattice`` for any further
    analysis. Editing the picked points invalidates the fit.

    A successful fit also builds an averaged unit-cell tile, shown as an
    inset tiled 3x3 so atoms near a cell edge aren't cut off. Sites for
    ``detect_atoms`` can then be chosen by eye rather than typed:
    ``propose_sites_from_cell()`` fills ``positions_frac`` from the tile's
    own peaks, ``add_site()`` appends one snapped position, and in the
    front end a click in the inset does the same. Because a raw click
    carries roughly a tenth of a cell of error, every picked position is
    snapped to the nearest column in the tile before being stored.
    """

    _esm = pathlib.Path(__file__).parent / "static" / "chooselattice.js"

    # Two sites closer than this are the same column as far as detection is
    # concerned, and closer still they break refinement outright.
    MIN_SITE_SEPARATION_PX = 3.0

    widget_version = traitlets.Unicode("unknown").tag(sync=True)
    height = traitlets.Int(1).tag(sync=True)
    width = traitlets.Int(1).tag(sync=True)
    frame_bytes = traitlets.Bytes(b"").tag(sync=True)
    title = traitlets.Unicode("").tag(sync=True)
    point_labels = traitlets.List(
        traitlets.Unicode(), default_value=["Origin", "u", "v"]
    ).tag(sync=True)
    points = traitlets.List(
        trait=traitlets.List(traitlets.Float()), default_value=[]
    ).tag(sync=True)

    has_imaging = traitlets.Bool(False).tag(sync=True)
    lattice_vectors = traitlets.List(
        trait=traitlets.List(traitlets.Float()), default_value=[]
    ).tag(sync=True)
    positions_frac = traitlets.List(
        trait=traitlets.List(traitlets.Float()), default_value=[[0.0, 0.0]]
    ).tag(sync=True)
    refine_lattice = traitlets.Bool(True).tag(sync=True)
    block_size = traitlets.Int(5, allow_none=True).tag(sync=True)
    atom_bytes = traitlets.Bytes(b"").tag(sync=True)
    num_atoms = traitlets.Int(0).tag(sync=True)
    show_grid = traitlets.Bool(True).tag(sync=True)
    show_atoms = traitlets.Bool(True).tag(sync=True)
    busy = traitlets.Bool(False).tag(sync=True)
    status = traitlets.Unicode("").tag(sync=True)

    cell_tile_bytes = traitlets.Bytes(b"").tag(sync=True)
    cell_samples = traitlets.Int(64).tag(sync=True)
    cell_max_cells = traitlets.Int(2000).tag(sync=True)
    cell_count = traitlets.Int(0).tag(sync=True)
    edge_min_dist_px = traitlets.Float(6.0).tag(sync=True)
    max_sites = traitlets.Int(None, allow_none=True).tag(sync=True)
    snap_to_common_sites = traitlets.Bool(True).tag(sync=True)
    true_cell_geometry = traitlets.Bool(False).tag(sync=True)

    def __init__(
        self,
        data,
        *,
        cmap: str = "gray",
        vmin: float | None = None,
        vmax: float | None = None,
        log_scale: bool = False,
        title: str = "",
        point_labels: Sequence[str] = ("Origin", "u", "v"),
        points: Sequence[Sequence[float]] | None = None,
        positions_frac: Sequence[Sequence[float]] = ((0.0, 0.0),),
        refine_lattice: bool = True,
        block_size: int | None = 5,
        cell_samples: int = 64,
        cell_max_cells: int = 2000,
        edge_min_dist_px: float = 6.0,
        max_sites: int | None = None,
        snap_to_common_sites: bool = True,
        true_cell_geometry: bool = False,
        save_state: bool = False,
        notebook_preview_format: str | None = None,
        notebook_preview_quality: int = 88,
        notebook_preview_max_px: int = 512,
        **kwargs,
    ) -> None:
        _reject_unknown_kwargs(type(self), kwargs)
        super().__init__(**kwargs)

        self._lattice: Any = None
        self._dataset: Any = None
        self._syncing_points = False
        self._syncing_sites = False
        self._cell_tile: np.ndarray | None = None

        lattice_type = _imaging_lattice_type()
        if lattice_type is not None and isinstance(data, lattice_type):
            self._lattice = data
            data = data.image

        core_image_dataset_types = _core_image_dataset_types()
        if bool(core_image_dataset_types) and isinstance(data, core_image_dataset_types):
            if not title and getattr(data, "name", None):
                title = data.name
            self._dataset = data
            data = data.array
        elif hasattr(data, "array") and hasattr(data, "name"):
            # Duck-typed Dataset2d fallback (quantem.core not importable here).
            if not title and getattr(data, "name", None):
                title = data.name
            self._dataset = data
            data = data.array

        frame = to_numpy(data, dtype=np.float32)
        if frame.ndim != 2:
            raise ValueError(
                f"ChooseLattice expects a single 2D image, got array with shape {frame.shape!r}."
            )
        self._data = frame

        rgb = _frame_to_rgb(frame, cmap=cmap, vmin=vmin, vmax=vmax, log_scale=log_scale)
        png_bytes = _rgb_to_png_bytes(rgb)

        self._configure_static_fallback(
            notebook_preview_format=notebook_preview_format,
            notebook_preview_quality=notebook_preview_quality,
            notebook_preview_max_px=notebook_preview_max_px,
        )
        self._save_state = bool(save_state)

        with self.hold_sync():
            self.height = int(frame.shape[0])
            self.width = int(frame.shape[1])
            self.frame_bytes = png_bytes
            self.title = str(title)
            self.point_labels = list(point_labels)
            self.has_imaging = _imaging_lattice_type() is not None
            self.positions_frac = self._validate_frac_value(positions_frac)
            self.refine_lattice = bool(refine_lattice)
            self.block_size = block_size if block_size is None else int(block_size)
            self.cell_samples = int(cell_samples)
            self.cell_max_cells = int(cell_max_cells)
            self.edge_min_dist_px = float(edge_min_dist_px)
            self.max_sites = max_sites if max_sites is None else int(max_sites)
            self.snap_to_common_sites = bool(snap_to_common_sites)
            self.true_cell_geometry = bool(true_cell_geometry)
            if points is not None:
                self.points = self._validate_points_value(points)

        if self._lattice is not None:
            self._sync_lattice_vectors()
            self._sync_atoms()
            if not self.points and self.lattice_vectors:
                r0, u, v = (np.asarray(x, dtype=float) for x in self.lattice_vectors)
                self._set_points_silently([r0, r0 + u, r0 + v])

        self.on_msg(self._on_client_action)

        try:
            from importlib.metadata import version

            self.widget_version = version("quantem-widget")
        except Exception:
            pass

    ### --- Trait validation and invalidation ---
    @traitlets.validate("block_size")
    def _validate_block_size(self, proposal):
        """Treat 0 as an alias for None: both mean "fit the whole image at once"."""
        value = proposal["value"]
        return None if value == 0 else value

    @traitlets.validate("max_sites")
    def _validate_max_sites(self, proposal):
        """Treat 0 as an alias for None: both mean "let the algorithm decide"."""
        value = proposal["value"]
        return None if value == 0 else value

    @traitlets.validate("points")
    def _validate_points(self, proposal):
        return self._validate_points_value(proposal["value"])

    def _validate_points_value(self, value) -> list[list[float]]:
        points = list(value)
        if len(points) > 3:
            raise traitlets.TraitError(
                f"ChooseLattice supports at most 3 points, got {len(points)}."
            )
        validated = []
        for point in points:
            row, col = point
            row = float(np.clip(row, 0, max(0, self.height - 1)))
            col = float(np.clip(col, 0, max(0, self.width - 1)))
            validated.append([row, col])
        return validated

    @traitlets.validate("positions_frac")
    def _validate_positions_frac(self, proposal):
        return self._validate_frac_value(proposal["value"])

    def _validate_frac_value(self, value) -> list[list[float]]:
        frac = np.asarray(value, dtype=float).reshape(-1, 2)
        if frac.size == 0:
            raise traitlets.TraitError("positions_frac must contain at least one site.")
        return [[float(a), float(b)] for a, b in frac]

    @traitlets.observe("points")
    def _invalidate_on_point_change(self, change):
        """Any edit to the picked points makes an existing fit stale."""
        if getattr(self, "_syncing_points", False):
            return
        if self.lattice_vectors or self.num_atoms:
            self.reset_lattice(
                status="Points changed - refit the lattice." if self.points else ""
            )

    @traitlets.observe("positions_frac")
    def _invalidate_on_site_change(self, change):
        """Changing the site list makes previously detected atoms stale."""
        if getattr(self, "_syncing_sites", False) or not self.num_atoms:
            return
        with self.hold_sync():
            self.atom_bytes = b""
            self.num_atoms = 0
            self.status = "Sites changed - detect atoms again."

    def _set_points_silently(self, points: Sequence[Sequence[float]]) -> None:
        """Set points without invalidating a fit that produced them."""
        self._syncing_points = True
        try:
            self.points = [list(map(float, p)) for p in points]
        finally:
            self._syncing_points = False

    ### --- Point picking API ---
    def set_points(self, points: Sequence[Sequence[float]]) -> None:
        """Programmatically set the picked points (up to 3, in order)."""
        self.points = [list(p) for p in points]

    def clear_points(self) -> None:
        """Remove all picked points and discard any fitted lattice."""
        self.points = []
        self.reset_lattice()

    @property
    def points_array(self) -> np.ndarray:
        """Picked points as an ``(n, 2)`` array of ``(row, col)`` pixel coordinates."""
        return np.array(self.points, dtype=np.float64).reshape(-1, 2)

    def _point_at(self, index: int) -> tuple[float, float] | None:
        pts = self.points
        if index >= len(pts):
            return None
        return (pts[index][0], pts[index][1])

    @property
    def origin(self) -> tuple[float, float] | None:
        """First picked point ``(row, col)``, or None if not yet placed."""
        return self._point_at(0)

    @property
    def a1(self) -> tuple[float, float] | None:
        """Second picked point ``(row, col)``, or None if not yet placed."""
        return self._point_at(1)

    @property
    def a2(self) -> tuple[float, float] | None:
        """Third picked point ``(row, col)``, or None if not yet placed."""
        return self._point_at(2)

    @property
    def u(self) -> tuple[float, float] | None:
        """Lattice vector ``a1 - origin``, or None until both are placed."""
        origin, a1 = self.origin, self.a1
        if origin is None or a1 is None:
            return None
        return (a1[0] - origin[0], a1[1] - origin[1])

    @property
    def v(self) -> tuple[float, float] | None:
        """Lattice vector ``a2 - origin``, or None until both are placed."""
        origin, a2 = self.origin, self.a2
        if origin is None or a2 is None:
            return None
        return (a2[0] - origin[0], a2[1] - origin[1])

    ### --- quantem.imaging.Lattice integration ---
    def _lattice_source(self):
        """Return an image payload safe to hand to ``Lattice.from_data``.

        ``from_data`` normalizes its input in place, so a caller-owned
        Dataset2d is copied first. When ``copy`` is unavailable the dataset
        is rebuilt from its calibration metadata, and only a metadata-free
        float array is used as a last resort.
        """
        dataset = self._dataset
        if dataset is None:
            return np.array(self._data, dtype=float)

        copier = getattr(dataset, "copy", None)
        if callable(copier):
            try:
                return copier()
            except Exception:
                pass

        rebuilder = getattr(type(dataset), "from_array", None)
        if callable(rebuilder):
            try:
                return rebuilder(
                    array=np.array(dataset.array, dtype=float),
                    name=getattr(dataset, "name", None),
                    origin=getattr(dataset, "origin", None),
                    sampling=getattr(dataset, "sampling", None),
                    units=getattr(dataset, "units", None),
                )
            except Exception:
                pass

        return np.array(self._data, dtype=float)

    @property
    def lattice(self):
        """The backing ``quantem.imaging.Lattice``, created on first access."""
        if self._lattice is None:
            lattice_type = _require_lattice_type()
            self._lattice = lattice_type.from_data(self._lattice_source())
        return self._lattice

    def _fitted_lat(self) -> np.ndarray | None:
        """Return the backing Lattice's fitted ``(3, 2)`` r0/u/v, or None."""
        lat = getattr(self._lattice, "_lat", None)
        if lat is None:
            return None
        try:
            array = np.asarray(lat, dtype=float).reshape(3, 2)
        except (TypeError, ValueError):
            return None
        return array if np.all(np.isfinite(array)) else None

    @property
    def is_fitted(self) -> bool:
        """True once ``define_lattice_vectors`` has run on the backing Lattice."""
        return self._fitted_lat() is not None

    def fit_lattice(
        self,
        *,
        refine: bool | None = None,
        block_size: int | None = None,
        refine_maxiter: int = 200,
    ):
        """Fit the picked origin/u/v with ``Lattice.define_lattice_vectors``.

        Returns the backing ``Lattice`` so calls can be chained. On success,
        ``points`` (and therefore ``origin``/``a1``/``a2``/``u``/``v``) are
        snapped to the fitted ``r0``/``u``/``v`` (rounded to 2 decimals) so
        they reflect the fit rather than the original clicks.

        Defaults to a staged fit (``block_size=5``, from the constructor's
        ``block_size``). Refining over the whole image at once
        (``block_size=None`` or ``0``) can converge to an aliased
        sub-lattice on real images: a much smaller, near-parallel
        ``u``/``v`` that also maximizes the bilinear intensity sum but no
        longer matches the periodicity the user clicked. Left unchecked
        this silently "succeeds" and then blows up ``detect_atoms()``,
        which tiles the *entire* image with that tiny basis. So a refined
        fit is sanity-checked against the picked vectors and rejected
        (with a message suggesting a staged ``block_size``) rather than
        accepted as-is.
        """
        origin, u, v = self.origin, self.u, self.v
        if origin is None or u is None or v is None:
            raise ValueError(
                "Place all three points (origin, u, v) before fitting the lattice."
            )
        if block_size is None:
            block_size = self.block_size
        if refine is None:
            refine = bool(self.refine_lattice)

        lattice = self.lattice
        lattice.define_lattice_vectors(
            origin,
            u,
            v,
            refine_lattice=bool(refine),
            block_size=block_size,
            refine_maxiter=int(refine_maxiter),
        )
        if refine:
            self._check_fit_not_degenerate(u, v, block_size)
        self._sync_lattice_vectors()
        self._sync_points_to_fit()
        self.build_cell_tile()
        return lattice

    def _sync_points_to_fit(self) -> None:
        """Snap the displayed origin/u/v points to the fitted r0/u/v.

        Uses ``Lattice._lat`` (via ``_fitted_lat``) rather than the picked
        points, rounded to 2 decimals since the fit is only meaningful to
        sub-pixel precision. Silent so this doesn't itself invalidate the
        fit it's reflecting.
        """
        fitted = self._fitted_lat()
        if fitted is None:
            return
        r0, u, v = fitted
        self._set_points_silently(
            [[round(float(c), 2) for c in p] for p in (r0, r0 + u, r0 + v)]
        )

    def _check_fit_not_degenerate(
        self, picked_u: tuple[float, float], picked_v: tuple[float, float], block_size: int | None
    ) -> None:
        """Reject a refined fit that collapsed onto an aliased sub-lattice.

        Compares the refined ``u``/``v`` against the picked ones: a healthy
        refinement nudges the vectors slightly, it doesn't shrink the cell
        area by a large factor or make the two vectors nearly parallel.
        """
        fitted = self._fitted_lat()
        if fitted is None:
            return
        _, fit_u, fit_v = fitted
        picked_area = abs(picked_u[0] * picked_v[1] - picked_u[1] * picked_v[0])
        fit_area = abs(fit_u[0] * fit_v[1] - fit_u[1] * fit_v[0])
        fit_norms = np.hypot(fit_u[0], fit_u[1]) * np.hypot(fit_v[0], fit_v[1])
        sin_angle = fit_area / fit_norms if fit_norms > 0 else 0.0
        degenerate = fit_area < 0.25 * picked_area or sin_angle < 0.1
        if not degenerate:
            return
        self.reset_lattice()
        hint = (
            "try a smaller block_size for staged refinement"
            if block_size is None
            else "try a different block_size"
        )
        raise ValueError(
            "Lattice refinement converged to a degenerate lattice (u, v became "
            f"near-parallel or the unit cell collapsed) instead of the picked "
            f"periodicity; {hint}, or pass refine=False to use the picked "
            "points as-is."
        )

    ### --- Averaged unit cell and site picking ---
    @property
    def cell_tile(self) -> np.ndarray | None:
        """The averaged unit-cell tile from the last fit, or None."""
        return self._cell_tile

    def build_cell_tile(
        self, samples: int | None = None, max_cells: int | None = None
    ) -> np.ndarray | None:
        """Rebuild the averaged unit-cell tile from the current fit."""
        lat = self._fitted_lat()
        if lat is None:
            self._cell_tile = None
            self._sync_cell_tile()
            return None
        tile, count = average_unit_cell(
            self._data,
            lat,
            samples=int(samples if samples is not None else self.cell_samples),
            max_cells=int(max_cells if max_cells is not None else self.cell_max_cells),
        )
        self._cell_tile = tile
        self._sync_cell_tile(count)
        return tile

    def pixel_to_frac(self, point: Sequence[float]) -> np.ndarray:
        """Convert an image ``(row, col)`` to a fractional ``(a, b)`` in the cell.

        Wrapping is modulo one cell, so the same column type anywhere in the
        image maps to the same pair - as long as the fit is good. A poor fit
        drifts by roughly its per-step error times the cell index, which is
        why picked positions are snapped to the averaged tile rather than
        trusted as clicked.
        """
        lat = self._fitted_lat()
        if lat is None:
            raise ValueError("Fit the lattice before converting pixels to fractions.")
        r0, u, v = lat
        det = u[0] * v[1] - u[1] * v[0]
        d_row, d_col = np.asarray(point, dtype=float) - r0
        return np.array(
            [
                ((d_row * v[1] - d_col * v[0]) / det) % 1.0,
                ((d_col * u[0] - d_row * u[1]) / det) % 1.0,
            ]
        )

    def snap_site(self, frac: Sequence[float]) -> np.ndarray:
        """Snap a fractional position to the nearest column of the averaged cell.

        A final snap to ``COMMON_SITES`` catches the case where the column
        snap lands close to (but not exactly on) a corner/edge/diagonal site
        - e.g. a slightly off fit or a noisy peak - so the reported position
        is the clean fraction rather than 0.498 or 0.334. Skipped when
        ``self.snap_to_common_sites`` is off; the column snap itself always
        runs.
        """
        if self._cell_tile is None:
            self.build_cell_tile()
        if self._cell_tile is None:
            clamped = np.clip(np.asarray(frac, dtype=float), 0.0, 1.0)
            return snap_to_common_site(clamped) if self.snap_to_common_sites else clamped
        snapped = snap_frac(self._cell_tile, frac)
        return snap_to_common_site(snapped) if self.snap_to_common_sites else snapped

    def site_separation_px(self, sites: Sequence[Sequence[float]] | None = None) -> float:
        """Smallest distance in pixels between any two sites, cell images included.

        This is the quantity ``refine_atoms`` halves to pick its automatic
        fit radius, so it doubles as the guard on whether refinement can
        work at all: near-duplicate sites shrink the radius until the fit
        patch holds a couple of pixels and the Gaussian fit degenerates.
        """
        lat = self._fitted_lat()
        frac = np.asarray(
            self.positions_frac if sites is None else sites, dtype=float
        ).reshape(-1, 2)
        if lat is None or frac.shape[0] == 0:
            return float("inf")
        _, u, v = lat
        pos = frac[:, 0:1] * u + frac[:, 1:2] * v
        best = float("inf")
        for d_n in (-1, 0, 1):
            for d_m in (-1, 0, 1):
                delta = pos[:, None, :] - pos[None, :, :] - (d_n * u + d_m * v)
                dist = np.hypot(delta[..., 0], delta[..., 1])
                if d_n == 0 and d_m == 0:
                    np.fill_diagonal(dist, np.inf)
                best = min(best, float(dist.min()))
        return best

    def add_site(self, frac: Sequence[float], *, snap: bool = True) -> list[list[float]]:
        """Append one site to ``positions_frac``, snapped by default.

        A pick landing within ``MIN_SITE_SEPARATION_PX`` of an existing site
        is treated as a repeat click on the same column and replaces it
        instead of creating a near-duplicate, which would collapse the
        automatic fit radius in ``refine_atoms``.
        """
        position = self.snap_site(frac) if snap else np.asarray(frac, dtype=float) % 1.0
        sites = [list(map(float, s)) for s in self.positions_frac]
        for index, existing in enumerate(sites):
            if self.site_separation_px([existing, list(position)]) < self.MIN_SITE_SEPARATION_PX:
                sites[index] = [float(position[0]), float(position[1])]
                self.positions_frac = sites
                return self.positions_frac
        sites.append([float(position[0]), float(position[1])])
        self.positions_frac = sites
        return self.positions_frac

    def add_site_from_pixel(self, point: Sequence[float]) -> list[list[float]]:
        """Append the site corresponding to an image ``(row, col)`` click."""
        return self.add_site(self.pixel_to_frac(point))

    def remove_site(self, index: int) -> list[list[float]]:
        """Drop one site by index, keeping at least one behind."""
        sites = [list(map(float, s)) for s in self.positions_frac]
        if 0 <= index < len(sites) and len(sites) > 1:
            sites.pop(index)
            self.positions_frac = sites
        return self.positions_frac

    def clear_sites(self) -> list[list[float]]:
        """Reset the site list to the single cell origin."""
        self.positions_frac = [[0.0, 0.0]]
        return self.positions_frac

    def propose_sites_from_cell(self, **kwargs) -> list[list[float]]:
        """Replace ``positions_frac`` with peaks found in the averaged cell.

        Keyword arguments are forwarded to ``propose_sites``
        (``min_sep_frac``, ``rel_threshold``, ``max_sites``). ``max_sites``
        defaults to ``self.max_sites`` (the widget's "number of sites"
        slider; ``None`` lets the algorithm decide, up to ``propose_sites``'s
        own default cap) when not passed explicitly.
        """
        if self._cell_tile is None:
            self.build_cell_tile()
        if self._cell_tile is None:
            raise ValueError("Fit the lattice before proposing sites.")
        if self.max_sites is not None:
            kwargs.setdefault("max_sites", self.max_sites)
        found = propose_sites(self._cell_tile, **kwargs)
        if found.shape[0] == 0:
            self.status = "No sites found in the averaged cell."
            return self.positions_frac

        # propose_sites separates peaks in cell fractions, which on a small
        # or smeared cell can still land two peaks a pixel or two apart.
        # Enforce the pixel-space minimum here, where the fitted vectors are
        # known, keeping the brightest of any crowded pair. Snap each peak to
        # a common site first (same rule as manually placed sites, and
        # likewise skipped when snap_to_common_sites is off) so two peaks
        # that both round to e.g. (0.5, 0.5) get deduplicated too.
        kept: list[list[float]] = []
        for site in found:
            snapped = snap_to_common_site(site) if self.snap_to_common_sites else site
            candidate = [float(snapped[0]), float(snapped[1])]
            if any(
                self.site_separation_px([existing, candidate]) < self.MIN_SITE_SEPARATION_PX
                for existing in kept
            ):
                continue
            kept.append(candidate)
        self.positions_frac = kept
        self.status = f"Proposed {len(kept)} site(s) from the averaged cell."
        return self.positions_frac

    def detect_atoms(
        self,
        positions_frac: Sequence[Sequence[float]] | None = None,
        *,
        refine: bool = False,
        **kwargs,
    ):
        """Populate lattice sites with ``Lattice.add_atoms``.

        Fits the lattice first when that has not happened yet. Extra keyword
        arguments (``intensity_min``, ``intensity_radius``, ``mask``,
        ``contrast_min``, ``radius_units``, ...) are forwarded unchanged;
        ``edge_min_dist_px`` defaults to the trait of the same name.
        """
        if positions_frac is not None:
            self.positions_frac = self._validate_frac_value(positions_frac)
        if not self.is_fitted:
            self.fit_lattice()
        kwargs.setdefault("edge_min_dist_px", float(self.edge_min_dist_px))

        lattice = self.lattice
        lattice.add_atoms(
            np.asarray(self.positions_frac, dtype=float),
            refine_atoms=bool(refine),
            **kwargs,
        )
        self._sync_atoms()
        return lattice

    def refine_atoms(self, **kwargs):
        """Refine atom centres with ``Lattice.refine_atoms``.

        Refuses to run when two sites sit closer than
        ``MIN_SITE_SEPARATION_PX``, unless an explicit ``fit_radius`` is
        given. The automatic radius is half the site spacing, so crowded
        sites leave a patch of a few pixels whose intensity range collapses;
        the solver then raises about its own bounds rather than about the
        sites, which is not a useful thing to hand back to the user.
        """
        lattice = self.lattice
        if getattr(lattice, "atoms", None) is None:
            raise ValueError("Detect atoms before refining. Call detect_atoms() first.")
        separation = self.site_separation_px()
        if "fit_radius" not in kwargs and separation < self.MIN_SITE_SEPARATION_PX:
            raise ValueError(
                f"Sites are only {separation:.2f} px apart, too close to refine "
                "(the automatic fit radius would be half that). Remove the "
                "near-duplicate sites, or pass an explicit fit_radius."
            )
        lattice.refine_atoms(**kwargs)
        self._sync_atoms()
        return lattice

    def reset_lattice(self, *, status: str = "") -> None:
        """Drop the backing Lattice and clear every derived overlay."""
        self._lattice = None
        self._cell_tile = None
        with self.hold_sync():
            self.lattice_vectors = []
            self.atom_bytes = b""
            self.num_atoms = 0
            self.cell_tile_bytes = b""
            self.cell_count = 0
            self.status = status

    @property
    def lattice_vectors_array(self) -> np.ndarray:
        """Refined ``r0, u, v`` as a ``(3, 2)`` array, empty until fitted."""
        return np.array(self.lattice_vectors, dtype=np.float64).reshape(-1, 2)

    @property
    def atoms_array(self) -> np.ndarray:
        """Detected atoms as an ``(n, 3)`` array of ``(row, col, site_index)``."""
        return self._pack_atoms()[0].reshape(-1, 3)

    def _pack_atoms(self) -> tuple[np.ndarray, int]:
        """Flatten per-site atom tables into a float32 ``(n, 3)`` overlay array.

        Column order is resolved from the atom container's declared fields so
        a reordering upstream cannot silently transpose row and column, and
        non-finite centres from failed fits are dropped.
        """
        empty = np.zeros((0, 3), dtype=np.float32)
        atoms = getattr(self._lattice, "atoms", None)
        if atoms is None:
            return empty, 0

        fields = list(getattr(atoms, "fields", []))
        if "x" not in fields or "y" not in fields:
            self.status = (
                "Detected atoms are missing the expected 'x'/'y' fields; "
                "overlay unavailable."
            )
            return empty, 0
        row_index, col_index = fields.index("x"), fields.index("y")

        blocks = []
        for site in range(len(self.positions_frac)):
            try:
                table = np.asarray(atoms[site].array, dtype=np.float64)
            except (IndexError, KeyError, AttributeError):
                continue
            table = table.reshape(-1, len(fields))
            if table.shape[0] == 0:
                continue
            block = np.empty((table.shape[0], 3), dtype=np.float32)
            block[:, 0] = table[:, row_index]
            block[:, 1] = table[:, col_index]
            block[:, 2] = float(site)
            blocks.append(block)

        if not blocks:
            return empty, 0
        stacked = np.concatenate(blocks, axis=0)
        stacked = stacked[np.all(np.isfinite(stacked), axis=1)]
        return stacked, int(stacked.shape[0])

    def _sync_lattice_vectors(self) -> None:
        """Push the fitted ``r0, u, v`` to the front end."""
        lat = self._fitted_lat()
        with self.hold_sync():
            if lat is None:
                self.lattice_vectors = []
                self.status = (
                    "Lattice fitted, but the fitted vectors could not be read "
                    "back for display."
                )
                return
            self.lattice_vectors = lat.tolist()
            r0, u, v = lat
            self.status = (
                f"Lattice fitted: origin ({r0[0]:.1f}, {r0[1]:.1f}), "
                f"u ({u[0]:.2f}, {u[1]:.2f}), v ({v[0]:.2f}, {v[1]:.2f})"
            )

    def _sync_atoms(self) -> None:
        """Push detected atom centres to the front end as a float32 buffer."""
        packed, count = self._pack_atoms()
        with self.hold_sync():
            self.atom_bytes = packed.tobytes() if count else b""
            self.num_atoms = count
            self.status = f"{count} atoms across {len(self.positions_frac)} site(s)"

    def _sync_cell_tile(self, count: int | None = None) -> None:
        """Push the averaged unit cell to the front end as a colormapped PNG.

        A PNG rather than raw floats: the tile is only ever displayed and
        clicked, and snapping happens here in the kernel, so the front end
        needs pixels rather than intensities.

        Tiled 3x3 before colormapping so every atom near a cell edge shows
        full, un-chopped context instead of being cut off at the panel
        border - the tile is already treated as exactly periodic elsewhere
        (wrap-mode peak finding in ``propose_sites``, wrap-aware distances
        inside ``snap_frac``), so repeating it is equivalent to re-averaging
        a 3x3 cell neighborhood without the 9x cost. Site fractions
        themselves (``self._cell_tile``, ``positions_frac``) still refer to
        the single canonical cell; only the display changes. A click outside
        that canonical cell (i.e. in the tiled panel's surrounding context)
        is clamped to the nearest point in it before snapping, not wrapped -
        see ``snap_frac``.
        """
        tile = self._cell_tile
        with self.hold_sync():
            if tile is None or tile.size == 0:
                self.cell_tile_bytes = b""
                self.cell_count = 0
                return
            rgb = _frame_to_rgb(
                np.tile(tile, (3, 3)), cmap="gray", vmin=None, vmax=None, log_scale=False
            )
            self.cell_tile_bytes = _rgb_to_png_bytes(rgb)
            if count is not None:
                self.cell_count = int(count)

    def _on_client_action(self, _widget, content, _buffers) -> None:
        """Dispatch front-end button actions onto the backing Lattice.

        Named to avoid shadowing ``ipywidgets.Widget._handle_custom_msg``,
        which is what actually loops over ``on_msg`` callbacks; overriding
        it here previously broke that dispatch (called with the wrong arity).
        """
        if not isinstance(content, dict):
            return
        action = content.get("action")
        known = {
            "fit", "detect", "refine", "reset",
            "pick_site_frac", "remove_site",
            "clear_sites", "propose_sites",
        }
        if action not in known:
            return

        self.busy = True
        try:
            if action == "fit":
                self.fit_lattice()
            elif action == "detect":
                self.detect_atoms()
            elif action == "refine":
                self.refine_atoms()
            elif action == "pick_site_frac":
                self.add_site(content.get("frac", (0.0, 0.0)))
            elif action == "remove_site":
                self.remove_site(int(content.get("index", -1)))
            elif action == "clear_sites":
                self.clear_sites()
            elif action == "propose_sites":
                self.propose_sites_from_cell()
            else:
                self.reset_lattice()
        except Exception as exc:
            self.status = f"{type(exc).__name__}: {exc}"
        finally:
            self.busy = False

    def _static_png_b64(self, max_px: int = 512) -> str | None:
        if not self.frame_bytes:
            return None
        return base64.b64encode(bytes(self.frame_bytes)).decode("ascii")