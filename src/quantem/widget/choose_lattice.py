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
"""

import base64
import pathlib
from typing import Any, Sequence

import anywidget
import matplotlib
import numpy as np
import traitlets
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
        widget as a slider from 1 to 10 with a "None" position at each end.
        Pass None (or 0) to fit the whole image at once.
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
    """

    _esm = pathlib.Path(__file__).parent / "static" / "chooselattice.js"

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

        Returns the backing ``Lattice`` so calls can be chained.

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
        return lattice

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
        ``contrast_min``, ``radius_units``, ...) are forwarded unchanged.
        """
        if positions_frac is not None:
            self.positions_frac = self._validate_frac_value(positions_frac)
        if not self.is_fitted:
            self.fit_lattice()

        lattice = self.lattice
        lattice.add_atoms(
            np.asarray(self.positions_frac, dtype=float),
            refine_atoms=bool(refine),
            **kwargs,
        )
        self._sync_atoms()
        return lattice

    def refine_atoms(self, **kwargs):
        """Refine detected atom centres with ``Lattice.refine_atoms``."""
        lattice = self.lattice
        if getattr(lattice, "atoms", None) is None:
            raise ValueError("Detect atoms before refining. Call detect_atoms() first.")
        lattice.refine_atoms(**kwargs)
        self._sync_atoms()
        return lattice

    def reset_lattice(self, *, status: str = "") -> None:
        """Drop the backing Lattice and clear every derived overlay."""
        self._lattice = None
        with self.hold_sync():
            self.lattice_vectors = []
            self.atom_bytes = b""
            self.num_atoms = 0
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

    def _on_client_action(self, _widget, content, _buffers) -> None:
        """Dispatch front-end button actions onto the backing Lattice.

        Named to avoid shadowing ``ipywidgets.Widget._handle_custom_msg``,
        which is what actually loops over ``on_msg`` callbacks; overriding
        it here previously broke that dispatch (called with the wrong arity).
        """
        if not isinstance(content, dict):
            return
        action = content.get("action")
        if action not in {"fit", "detect", "refine", "reset"}:
            return

        self.busy = True
        try:
            if action == "fit":
                self.fit_lattice()
            elif action == "detect":
                self.detect_atoms()
            elif action == "refine":
                self.refine_atoms()
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