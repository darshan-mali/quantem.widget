/**
 * ChooseLattice — pick an ordered origin + two lattice-vector points on an image,
 * then fit a quantem.imaging.Lattice from them.
 *
 * Lean single-panel viewer: one canvas showing a pre-rendered (server-side
 * colormapped) image. Scroll to zoom, drag to pan, click to place up to 3
 * ordered points, drag an existing point to adjust it. Points are reported
 * in ORIGINAL image pixel coordinates regardless of the current zoom/pan.
 *
 * Once the kernel has fitted a lattice, the refined r0/u/v arrive as
 * `lattice_vectors` and detected atom centres as a packed float32
 * `atom_bytes` buffer; both are drawn on the overlay canvas in image
 * coordinates so they track zoom and pan.
 *
 * A fit also delivers `cell_tile_bytes`, a PNG of the averaged unit cell
 * pre-tiled 3x3 by the kernel, shown as an inset panel so atoms near a cell
 * edge aren't cut off. Clicking it asks the kernel for a site at that
 * position; the kernel snaps the pick to the nearest column (and, unless
 * `snap_to_common_sites` is off, on to a nearby corner/edge/diagonal
 * fraction) and appends it to `positions_frac`, which `detect_atoms` then
 * uses. Once `max_sites` names an exact count and that many sites are
 * placed, an existing site can instead be dragged in the inset to refine
 * its position - always exactly under the cursor, with no snapping of
 * either kind.
 *
 * The inset renders the tile as a plain square by default, which silently
 * assumes the fitted u/v lattice vectors are orthogonal - a small lie for a
 * near-square lattice, a bad one for e.g. hexagonal. `true_cell_geometry`
 * (off by default) switches it to a true parallelogram matching the real
 * u/v angle instead; purely a display setting, `positions_frac` is
 * unaffected either way.
 */

import * as React from "react";
import { createRender, useModel, useModelState } from "@anywidget/react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Stack from "@mui/material/Stack";
import Button from "@mui/material/Button";
import Slider from "@mui/material/Slider";
import IconButton from "@mui/material/IconButton";
import KeyboardArrowUpIcon from "@mui/icons-material/KeyboardArrowUp";
import KeyboardArrowDownIcon from "@mui/icons-material/KeyboardArrowDown";
import { useTheme } from "../theme";
import { extractBytes, extractFloat32, preserveRestoredWidgetModelsOnSave } from "../format";
import { useHideStaticFallback } from "../staticFallback";

const MIN_ZOOM = 0.5;
const MAX_ZOOM = 20;
const CANVAS_SIZE = 512;
const CANVAS_BORDER_PX = 1;
const HIT_PX = 10;
const CLICK_MOVE_THRESHOLD_PX = 4;
const POINT_COLORS = ["#00ffff", "#ff00ff", "#ffff00"]; // Origin, u, v (cyan, magenta, yellow)

// Matches quantem.imaging.lattice_visualization.site_colors(), whose palette
// is indexed by site (cycling via `% len(palette)`) the same way `atoms[i+2]`
// (the site index) is used below.
const ATOM_COLORS = [
  "#ff0000", // 0: red
  "#00b3ff", // 1: lighter blue
  "#00b300", // 2: green (lower perceptual brightness)
  "#ff00ff", // 3: magenta
  "#ffb300", // 4: orange
  "#0000ff", // 5: full blue
  "#9933cc",
  "#4dbfbf",
  "#cc6600",
  "#339933",
  "#b3b300",
  "#ffffff",
  "#000000", // always last, per upstream palette's ordering convention
];
const ATOM_RADIUS_PX = 2.5;
const GRID_COLOR = "rgba(64,169,255,0.45)";
const MAX_GRID_LINES = 400;
const EDGE_MARGIN_COLOR = "rgba(255,90,90,0.7)";

// Averaged unit-cell inset: the kernel sends the tile pre-tiled 3x3 (see
// _sync_cell_tile) so atoms near a cell edge aren't cut off, so the panel
// spans fractional [-1, 2) on both axes instead of [0, 1). Tile row index
// maps to fractional a, column index to fractional b, so a click at (x, y)
// in the panel is (a, b) = CELL_TILE_REPEAT * (y, x) / size - 1.
// 2x the pre-3x3-tiling size: showing 3x the fractional extent in the same
// footprint would otherwise shrink each unit cell to a third of its former
// on-screen size.
const CELL_PANEL_PX = 352;
const CELL_TILE_REPEAT = 3;
const CELL_SITE_RADIUS_PX = 5;
const CELL_SITE_HIT_PX = 10;

// Block-size slider: indices 1-10 are staged block sizes of that size;
// index 11 (one past the largest real size) means "None" (fit the whole
// image at once).
const BLOCK_SIZE_MIN_IDX = 1;
const BLOCK_SIZE_MAX_IDX = 11;
const blockSizeToIdx = (value: number | null): number =>
  value == null ? BLOCK_SIZE_MAX_IDX : Math.max(BLOCK_SIZE_MIN_IDX, Math.min(10, Math.round(value)));
const idxToBlockSize = (idx: number): number | null =>
  idx >= BLOCK_SIZE_MAX_IDX ? null : idx;

// Number-of-sites slider: 0 means "Auto" (let propose_sites decide), 1-10
// suggest an exact site count as the max_sites cap.
const MAX_SITES_MIN = 0;
const MAX_SITES_MAX = 10;
const RIGHT_PANEL_PX = 368;

const SPACING = { XS: 4, SM: 8, MD: 12, LG: 16 } as const;
const compactButton = {
  fontSize: 10,
  py: 0.25,
  px: 1,
  minWidth: 0,
  textTransform: "none" as const,
};

type Point = [number, number]; // [row, col] in original image pixels
type Vec2 = [number, number];
type DragMode = "none" | "pan" | "point";

function clamp(value: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, value));
}

/**
 * Integer lattice index range (n, m) whose sites can reach the image corners,
 * solving corner = r0 + n*u + m*v. Returns null for a degenerate basis.
 */
function latticeIndexBounds(
  r0: Vec2,
  u: Vec2,
  v: Vec2,
  height: number,
  width: number,
): { nMin: number; nMax: number; mMin: number; mMax: number } | null {
  const det = u[0] * v[1] - u[1] * v[0];
  if (!isFinite(det) || Math.abs(det) < 1e-9) return null;
  const corners: Point[] = [
    [0, 0],
    [0, width - 1],
    [height - 1, 0],
    [height - 1, width - 1],
  ];
  let nMin = Infinity;
  let nMax = -Infinity;
  let mMin = Infinity;
  let mMax = -Infinity;
  for (const [row, col] of corners) {
    const dr = row - r0[0];
    const dc = col - r0[1];
    const n = (dr * v[1] - dc * v[0]) / det;
    const m = (dc * u[0] - dr * u[1]) / det;
    if (!isFinite(n) || !isFinite(m)) return null;
    nMin = Math.min(nMin, n);
    nMax = Math.max(nMax, n);
    mMin = Math.min(mMin, m);
    mMax = Math.max(mMax, m);
  }
  return {
    nMin: Math.floor(nMin),
    nMax: Math.ceil(nMax),
    mMin: Math.floor(mMin),
    mMax: Math.ceil(mMax),
  };
}

/**
 * Distance from each image edge excluded by `edge_min_dist_px`, clamped so
 * the remaining region never inverts on a tiny image or a huge margin.
 */
function edgeInsetPx(marginPx: number, height: number, width: number): number {
  return Math.max(0, Math.min(marginPx, height / 2 - 0.5, width / 2 - 0.5));
}

/**
 * Draw the fitted lattice as two families of lines spanning the image,
 * clipped to the `edge_min_dist_px` margin so the grid doesn't imply
 * periodicity out to the very edge, where `detect_atoms()` won't place
 * sites anyway.
 */
function drawLatticeGrid(
  ctx: CanvasRenderingContext2D,
  lat: number[][],
  height: number,
  width: number,
  marginPx: number,
  imgToScreen: (row: number, col: number) => [number, number],
): void {
  const r0: Vec2 = [lat[0][0], lat[0][1]];
  const u: Vec2 = [lat[1][0], lat[1][1]];
  const v: Vec2 = [lat[2][0], lat[2][1]];
  const bounds = latticeIndexBounds(r0, u, v, height, width);
  if (!bounds) return;
  const { nMin, nMax, mMin, mMax } = bounds;
  if (nMax - nMin + (mMax - mMin) > MAX_GRID_LINES) return;

  const at = (n: number, m: number): [number, number] =>
    imgToScreen(r0[0] + n * u[0] + m * v[0], r0[1] + n * u[1] + m * v[1]);

  // latticeIndexBounds gives the (n, m) range that COVERS the image corners,
  // but individual lines within that range still overshoot past the image
  // edges (it's a bounding box in lattice-index space, not image space).
  // Clip to the image rectangle inset by the edge margin, in screen space,
  // so nothing is ever drawn beyond the excluded border.
  const inset = edgeInsetPx(marginPx, height, width);
  const [ix0, iy0] = imgToScreen(inset, inset);
  const [ix1, iy1] = imgToScreen(height - inset, width - inset);

  ctx.save();
  ctx.beginPath();
  ctx.rect(Math.min(ix0, ix1), Math.min(iy0, iy1), Math.abs(ix1 - ix0), Math.abs(iy1 - iy0));
  ctx.clip();
  ctx.strokeStyle = GRID_COLOR;
  ctx.lineWidth = 1;
  for (let n = nMin; n <= nMax; n++) {
    const [x0, y0] = at(n, mMin);
    const [x1, y1] = at(n, mMax);
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(x1, y1);
    ctx.stroke();
  }
  for (let m = mMin; m <= mMax; m++) {
    const [x0, y0] = at(nMin, m);
    const [x1, y1] = at(nMax, m);
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(x1, y1);
    ctx.stroke();
  }
  ctx.restore();
}

/**
 * Draw detected atoms from a packed (n, 3) float32 buffer of
 * [row, col, site_index], colour-coded per site and culled to the viewport.
 */
function drawAtoms(
  ctx: CanvasRenderingContext2D,
  atoms: Float32Array,
  canvasW: number,
  canvasH: number,
  imgToScreen: (row: number, col: number) => [number, number],
): void {
  ctx.save();
  const margin = ATOM_RADIUS_PX + 2;
  for (let i = 0; i + 2 < atoms.length; i += 3) {
    const [x, y] = imgToScreen(atoms[i], atoms[i + 1]);
    if (x < -margin || y < -margin || x > canvasW + margin || y > canvasH + margin) continue;
    const site = Math.max(0, Math.round(atoms[i + 2]));
    ctx.beginPath();
    ctx.arc(x, y, ATOM_RADIUS_PX, 0, 2 * Math.PI);
    ctx.fillStyle = ATOM_COLORS[site % ATOM_COLORS.length];
    ctx.fill();
  }
  ctx.restore();
}

/**
 * Draw the excluded border as a dashed rectangle inset from the image edges
 * by `marginPx` on every side, so the region `detect_atoms()` won't place
 * sites in (via `edge_min_dist_px`) is visible before detecting.
 */
function drawEdgeMargin(
  ctx: CanvasRenderingContext2D,
  marginPx: number,
  height: number,
  width: number,
  imgToScreen: (row: number, col: number) => [number, number],
): void {
  if (marginPx <= 0 || height <= 0 || width <= 0) return;
  const inset = edgeInsetPx(marginPx, height, width);
  if (inset <= 0) return;
  const corners: Point[] = [
    [inset, inset],
    [inset, width - inset],
    [height - inset, width - inset],
    [height - inset, inset],
  ];
  ctx.save();
  ctx.strokeStyle = EDGE_MARGIN_COLOR;
  ctx.lineWidth = 1;
  ctx.setLineDash([5, 4]);
  ctx.beginPath();
  corners.forEach(([row, col], i) => {
    const [x, y] = imgToScreen(row, col);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.closePath();
  ctx.stroke();
  ctx.restore();
}

function ChooseLattice() {
  const model = useModel();
  const rootRef = React.useRef<HTMLDivElement>(null);
  const { colors: themeColors } = useTheme();

  React.useEffect(() => preserveRestoredWidgetModelsOnSave(model), [model]);
  useHideStaticFallback(model, rootRef);

  const [height] = useModelState<number>("height");
  const [width] = useModelState<number>("width");
  const [frameBytes] = useModelState<DataView>("frame_bytes");
  const [title] = useModelState<string>("title");
  const [pointLabels] = useModelState<string[]>("point_labels");
  const [points, setPoints] = useModelState<Point[]>("points");

  const [latticeVectors] = useModelState<number[][]>("lattice_vectors");
  const [atomBytes] = useModelState<DataView>("atom_bytes");
  const [numAtoms] = useModelState<number>("num_atoms");
  const [hasImaging] = useModelState<boolean>("has_imaging");
  const [blockSize, setBlockSize] = useModelState<number | null>("block_size");
  // Local slider position: mirrors `blockSize` but also remembers WHICH end
  // ("None" is reachable from either extreme) the user last dragged to,
  // since the model value alone can't distinguish idx 0 from idx 11.
  const [blockSizeIdx, setBlockSizeIdx] = React.useState<number>(() => blockSizeToIdx(blockSize));
  React.useEffect(() => {
    if (blockSize != null) setBlockSizeIdx(blockSize);
  }, [blockSize]);
  const handleBlockSizeChange = React.useCallback(
    (idx: number) => {
      setBlockSizeIdx(idx);
      setBlockSize(idxToBlockSize(idx));
    },
    [setBlockSize],
  );
  const [showGrid, setShowGrid] = useModelState<boolean>("show_grid");
  const [showAtoms, setShowAtoms] = useModelState<boolean>("show_atoms");
  const [busy] = useModelState<boolean>("busy");
  const [status] = useModelState<string>("status");
  const [cellTileBytes] = useModelState<DataView>("cell_tile_bytes");
  const [cellCount] = useModelState<number>("cell_count");
  const [positionsFrac, setPositionsFrac] = useModelState<number[][]>("positions_frac");
  const [maxSites, setMaxSites] = useModelState<number | null>("max_sites");
  const [snapToCommonSites, setSnapToCommonSites] = useModelState<boolean>("snap_to_common_sites");
  const [trueCellGeometry, setTrueCellGeometry] = useModelState<boolean>("true_cell_geometry");
  const [edgeMinDistPx, setEdgeMinDistPx] = useModelState<number>("edge_min_dist_px");
  const handleMaxSitesChange = React.useCallback(
    (value: number) => setMaxSites(value <= MAX_SITES_MIN ? null : value),
    [setMaxSites],
  );

  // Decode the PNG payload once per change into a drawable bitmap.
  const [image, setImage] = React.useState<ImageBitmap | HTMLImageElement | null>(null);
  React.useEffect(() => {
    const bytes = extractBytes(frameBytes);
    if (bytes.length === 0) {
      setImage(null);
      return;
    }
    let cancelled = false;
    const blob = new Blob([bytes as unknown as BlobPart], { type: "image/png" });
    if (typeof createImageBitmap === "function") {
      createImageBitmap(blob).then((bmp) => { if (!cancelled) setImage(bmp); });
    } else {
      const url = URL.createObjectURL(blob);
      const img = new Image();
      img.onload = () => { if (!cancelled) setImage(img); URL.revokeObjectURL(url); };
      img.src = url;
    }
    return () => { cancelled = true; };
  }, [frameBytes]);

  // Atom centres arrive as a packed float32 buffer rather than JSON: a 10k-atom
  // field is ~120 KB here versus several MB as a nested list trait.
  const atoms = React.useMemo(
    () => extractFloat32(atomBytes, numAtoms * 3),
    [atomBytes, numAtoms],
  );

  const [cellTile, setCellTile] = React.useState<ImageBitmap | HTMLImageElement | null>(null);
  React.useEffect(() => {
    const bytes = extractBytes(cellTileBytes);
    if (bytes.length === 0) {
      setCellTile(null);
      return;
    }
    let cancelled = false;
    const blob = new Blob([bytes as unknown as BlobPart], { type: "image/png" });
    if (typeof createImageBitmap === "function") {
      createImageBitmap(blob).then((bmp) => { if (!cancelled) setCellTile(bmp); });
    } else {
      const url = URL.createObjectURL(blob);
      const img = new Image();
      img.onload = () => { if (!cancelled) setCellTile(img); URL.revokeObjectURL(url); };
      img.src = url;
    }
    return () => { cancelled = true; };
  }, [cellTileBytes]);

  // View state: zoom + pan (CSS px, canvas-centered).
  const [zoom, setZoom] = React.useState(1);
  const [panX, setPanX] = React.useState(0);
  const [panY, setPanY] = React.useState(0);

  // displayScale maps original image pixels -> CSS px at zoom=1.
  const displayScale = height > 0 && width > 0
    ? CANVAS_SIZE / Math.max(height, width)
    : 1;
  const canvasW = CANVAS_SIZE;
  const canvasH = CANVAS_SIZE;

  // Edge-margin slider range scales with the image so a tiny crop and a
  // huge scan both get a usable range of excludable border widths.
  const edgeMarginMax = React.useMemo(
    () => Math.max(20, Math.floor(Math.min(width || 0, height || 0) / 4)),
    [width, height],
  );

  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const uiRef = React.useRef<HTMLCanvasElement>(null);
  const containerRef = React.useRef<HTMLDivElement>(null);

  // Block page scroll on wheel-zoom with a native non-passive listener;
  // React's synthetic onWheel is passive and would only warn if it called
  // preventDefault itself.
  React.useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const prevent = (e: WheelEvent) => e.preventDefault();
    el.addEventListener("wheel", prevent, { passive: false });
    return () => el.removeEventListener("wheel", prevent);
  }, []);

  // Edge margin: scroll-adjusted or typed directly. Same non-passive-listener
  // trick as above for the wheel part, since a synthetic onWheel can't
  // actually block page scroll.
  const edgeMarginRef = React.useRef<HTMLInputElement>(null);
  const edgeMinDistPxRef = React.useRef(edgeMinDistPx);
  React.useEffect(() => { edgeMinDistPxRef.current = edgeMinDistPx; }, [edgeMinDistPx]);
  const stepEdgeMargin = React.useCallback((delta: number) => {
    const current = edgeMinDistPxRef.current ?? 0;
    const next = clamp(current + delta, 0, edgeMarginMax);
    if (next !== current) setEdgeMinDistPx(next);
  }, [edgeMarginMax, setEdgeMinDistPx]);

  // Press-and-hold repeat for the edge-margin stepper buttons: step once
  // immediately, pause, then repeat on an interval until released.
  const holdTimerRef = React.useRef<number | null>(null);
  const clearHoldTimer = React.useCallback(() => {
    if (holdTimerRef.current != null) {
      window.clearTimeout(holdTimerRef.current);
      holdTimerRef.current = null;
    }
  }, []);
  const startEdgeMarginHold = React.useCallback((delta: number) => {
    clearHoldTimer();
    stepEdgeMargin(delta);
    holdTimerRef.current = window.setTimeout(function repeat() {
      stepEdgeMargin(delta);
      holdTimerRef.current = window.setTimeout(repeat, 60);
    }, 350);
  }, [clearHoldTimer, stepEdgeMargin]);
  React.useEffect(() => {
    window.addEventListener("pointerup", clearHoldTimer);
    return () => {
      window.removeEventListener("pointerup", clearHoldTimer);
      clearHoldTimer();
    };
  }, [clearHoldTimer]);
  React.useEffect(() => {
    const el = edgeMarginRef.current;
    if (!el || busy) return;
    const handleScroll = (e: WheelEvent) => {
      e.preventDefault();
      const step = e.shiftKey ? 5 : 1;
      stepEdgeMargin(e.deltaY < 0 ? step : -step);
    };
    el.addEventListener("wheel", handleScroll, { passive: false });
    return () => el.removeEventListener("wheel", handleScroll);
  }, [busy, stepEdgeMargin]);

  // Typed-input buffer: a free-form string while the field has focus (so an
  // in-progress edit like "1" isn't clobbered by the synced trait value),
  // reset from the trait whenever the field isn't focused (scroll-wheel
  // edits included).
  const [edgeMarginText, setEdgeMarginText] = React.useState(
    () => String(Math.round(edgeMinDistPx ?? 0)),
  );
  const edgeMarginEditingRef = React.useRef(false);
  React.useEffect(() => {
    if (!edgeMarginEditingRef.current) setEdgeMarginText(String(Math.round(edgeMinDistPx ?? 0)));
  }, [edgeMinDistPx]);
  const commitEdgeMarginText = React.useCallback((raw: string) => {
    const parsed = Number(raw);
    const next = Number.isFinite(parsed) && raw.trim() !== ""
      ? clamp(Math.round(parsed), 0, edgeMarginMax)
      : Math.round(edgeMinDistPxRef.current ?? 0);
    setEdgeMinDistPx(next);
    setEdgeMarginText(String(next));
  }, [edgeMarginMax, setEdgeMinDistPx]);

  // Draw the base image with pan/zoom applied.
  React.useLayoutEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    canvas.width = canvasW;
    canvas.height = canvasH;
    ctx.imageSmoothingEnabled = false;
    ctx.fillStyle = themeColors.bg;
    ctx.fillRect(0, 0, canvasW, canvasH);
    if (!image || !width || !height) return;
    const cx = canvasW / 2;
    const cy = canvasH / 2;
    const drawW = width * displayScale * zoom;
    const drawH = height * displayScale * zoom;
    const x = cx - drawW / 2 + panX;
    const y = cy - drawH / 2 + panY;
    ctx.drawImage(image, x, y, drawW, drawH);
  }, [image, width, height, displayScale, zoom, panX, panY, canvasW, canvasH, themeColors.bg]);

  // Convert a mouse event to original-image (row, col) coordinates.
  const screenToImg = React.useCallback((e: { clientX: number; clientY: number }): Point => {
    const canvas = canvasRef.current;
    if (!canvas) return [0, 0];
    const rect = canvas.getBoundingClientRect();
    const mouseCanvasX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const mouseCanvasY = (e.clientY - rect.top) * (canvas.height / rect.height);
    const cx = canvasW / 2;
    const cy = canvasH / 2;
    const col = (mouseCanvasX - cx - panX) / (displayScale * zoom) + width / 2;
    const row = (mouseCanvasY - cy - panY) / (displayScale * zoom) + height / 2;
    return [row, col];
  }, [canvasW, canvasH, panX, panY, displayScale, zoom, width, height]);

  const imgToScreen = React.useCallback((row: number, col: number): [number, number] => {
    const cx = canvasW / 2;
    const cy = canvasH / 2;
    const x = cx + (col - width / 2) * displayScale * zoom + panX;
    const y = cy + (row - height / 2) * displayScale * zoom + panY;
    return [x, y];
  }, [canvasW, canvasH, panX, panY, displayScale, zoom, width, height]);

  const hitTestPoint = React.useCallback((row: number, col: number): number => {
    const hitArea = HIT_PX / (displayScale * zoom);
    const list = points || [];
    for (let i = list.length - 1; i >= 0; i--) {
      const [pr, pc] = list[i];
      if (Math.hypot(row - pr, col - pc) <= hitArea) return i;
    }
    return -1;
  }, [points, displayScale, zoom]);

  // Wheel: cursor-anchored zoom. Page-scroll prevention is handled by a
  // native non-passive listener below (React's synthetic onWheel is passive,
  // so calling preventDefault directly here would only log a console warning).
  const handleWheel = (e: React.WheelEvent) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mouseCanvasX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const mouseCanvasY = (e.clientY - rect.top) * (canvas.height / rect.height);
    const cx = canvasW / 2;
    const cy = canvasH / 2;
    const mouseImageX = (mouseCanvasX - cx - panX) / zoom + cx;
    const mouseImageY = (mouseCanvasY - cy - panY) / zoom + cy;
    const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
    const newZoom = clamp(zoom * zoomFactor, MIN_ZOOM, MAX_ZOOM);
    setPanX(mouseCanvasX - (mouseImageX - cx) * newZoom - cx);
    setPanY(mouseCanvasY - (mouseImageY - cy) * newZoom - cy);
    setZoom(newZoom);
  };

  const resetView = React.useCallback(() => {
    setZoom(1);
    setPanX(0);
    setPanY(0);
  }, []);

  // A click's mousedown/mouseup fire BEFORE the browser knows whether a
  // second click will follow (making it a double-click). Placing a point
  // immediately on every plain click would mean double-clicking to reset the
  // view always drops a spurious point first. So a plain click's point is
  // held for this long — if a double-click follows, it is cancelled instead.
  const DOUBLE_CLICK_GRACE_MS = 300;
  const pointsRef = React.useRef(points);
  React.useEffect(() => { pointsRef.current = points; }, [points]);
  const pendingPointTimeoutRef = React.useRef<number | null>(null);
  const cancelPendingPoint = React.useCallback(() => {
    if (pendingPointTimeoutRef.current !== null) {
      window.clearTimeout(pendingPointTimeoutRef.current);
      pendingPointTimeoutRef.current = null;
    }
  }, []);
  React.useEffect(() => cancelPendingPoint, [cancelPendingPoint]);

  // Mouse drag: pan the view, drag an existing point, or (on a plain click)
  // place the next point in order.
  const dragRef = React.useRef<{
    mode: DragMode;
    startClientX: number;
    startClientY: number;
    startPanX: number;
    startPanY: number;
    pointIndex: number;
  } | null>(null);

  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.detail >= 2) {
      // Second (and later) click of a double-click: cancel any point the
      // first click was about to place, and let onDoubleClick reset the view.
      cancelPendingPoint();
      dragRef.current = null;
      return;
    }
    const [row, col] = screenToImg(e);
    const hitIdx = hitTestPoint(row, col);
    if (hitIdx !== -1) {
      dragRef.current = {
        mode: "point", startClientX: e.clientX, startClientY: e.clientY,
        startPanX: panX, startPanY: panY, pointIndex: hitIdx,
      };
      return;
    }
    dragRef.current = {
      mode: "none", startClientX: e.clientX, startClientY: e.clientY,
      startPanX: panX, startPanY: panY, pointIndex: -1,
    };
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    const drag = dragRef.current;
    if (!drag) return;
    if (drag.mode === "point") {
      const [row, col] = screenToImg(e);
      const next = (points || []).slice();
      next[drag.pointIndex] = [clamp(row, 0, Math.max(0, height - 1)), clamp(col, 0, Math.max(0, width - 1))];
      setPoints(next);
      return;
    }
    const moved = Math.hypot(e.clientX - drag.startClientX, e.clientY - drag.startClientY);
    if (drag.mode === "none" && moved > CLICK_MOVE_THRESHOLD_PX) {
      dragRef.current = { ...drag, mode: "pan" };
    }
    if (dragRef.current?.mode === "pan") {
      setPanX(drag.startPanX + (e.clientX - drag.startClientX));
      setPanY(drag.startPanY + (e.clientY - drag.startClientY));
    }
  };

  const handleMouseUp = (e: React.MouseEvent) => {
    const drag = dragRef.current;
    dragRef.current = null;
    if (!drag) return;
    if (drag.mode === "none") {
      // Plain click (no drag past the threshold): place the next point,
      // after a short grace period a following double-click can cancel.
      const list = pointsRef.current || [];
      if (list.length < 3) {
        const [row, col] = screenToImg(e);
        const clamped: Point = [
          clamp(row, 0, Math.max(0, height - 1)),
          clamp(col, 0, Math.max(0, width - 1)),
        ];
        cancelPendingPoint();
        pendingPointTimeoutRef.current = window.setTimeout(() => {
          pendingPointTimeoutRef.current = null;
          setPoints([...(pointsRef.current || []), clamped]);
        }, DOUBLE_CLICK_GRACE_MS);
      }
    }
  };

  const handleDoubleClick = (e: React.MouseEvent) => {
    e.preventDefault();
    cancelPendingPoint();
    resetView();
  };

  // Cursor readout while hovering (not dragging).
  const [cursorPos, setCursorPos] = React.useState<Point | null>(null);
  const handleMouseMoveReadout = (e: React.MouseEvent) => {
    handleMouseMove(e);
    if (!dragRef.current || dragRef.current.mode === "none") {
      setCursorPos(screenToImg(e));
    }
  };

  // Overlay, painted back to front: fitted lattice grid, detected atoms,
  // guide lines from Origin -> u and Origin -> v, then the point markers.
  React.useLayoutEffect(() => {
    const canvas = uiRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    canvas.width = canvasW;
    canvas.height = canvasH;
    ctx.clearRect(0, 0, canvasW, canvasH);

    if (showGrid && latticeVectors && latticeVectors.length === 3) {
      drawLatticeGrid(ctx, latticeVectors, height, width, edgeMinDistPx ?? 0, imgToScreen);
    }
    if (showAtoms && atoms && atoms.length) {
      drawAtoms(ctx, atoms, canvasW, canvasH, imgToScreen);
    }
    if (hasImaging && edgeMinDistPx) {
      drawEdgeMargin(ctx, edgeMinDistPx, height, width, imgToScreen);
    }

    const list = points || [];
    if (list.length > 1) {
      const [ox, oy] = imgToScreen(list[0][0], list[0][1]);
      ctx.strokeStyle = "rgba(255,255,255,0.6)";
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      for (let i = 1; i < list.length; i++) {
        const [x, y] = imgToScreen(list[i][0], list[i][1]);
        ctx.beginPath();
        ctx.moveTo(ox, oy);
        ctx.lineTo(x, y);
        ctx.stroke();
      }
      ctx.setLineDash([]);
    }
    list.forEach(([row, col], i) => {
      const [x, y] = imgToScreen(row, col);
      const color = POINT_COLORS[i % POINT_COLORS.length];
      ctx.beginPath();
      ctx.arc(x, y, 6, 0, 2 * Math.PI);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.strokeStyle = "#000";
      ctx.lineWidth = 1;
      ctx.stroke();
      const label = pointLabels && pointLabels[i] ? pointLabels[i] : String(i + 1);
      ctx.font = "bold 11px -apple-system, sans-serif";
      ctx.fillStyle = "#fff";
      ctx.strokeStyle = "rgba(0,0,0,0.85)";
      ctx.lineWidth = 3;
      ctx.textAlign = "left";
      ctx.textBaseline = "bottom";
      ctx.strokeText(label, x + 9, y - 6);
      ctx.fillText(label, x + 9, y - 6);
    });
  }, [
    points, pointLabels, imgToScreen, canvasW, canvasH,
    latticeVectors, atoms, showGrid, showAtoms, height, width,
    hasImaging, edgeMinDistPx,
  ]);

  // Averaged unit cell: the panel shows a 3x3 tiling of the averaged cell
  // (fractional [-1, 2) on both axes) so atoms near a cell edge aren't cut
  // off. Sites are always canonical [0,1) fractions.
  //
  // The raster is sampled along the real (possibly non-orthogonal) fitted
  // u/v lattice vectors, so rendering it as a plain axis-aligned square
  // silently assumes u perp v - a small lie for a near-square lattice, a
  // bad one for e.g. hexagonal (~60-120 degrees between u and v). When
  // trueCellGeometry is on, cellTransform instead derives a linear map
  // from the real u/v (scaled/centered to fit the fixed panel size) so the
  // panel renders a true parallelogram; when off (the default) or the
  // fitted vectors are missing/degenerate, it falls back to exactly the
  // original per-axis formulas.
  type CellTransform = {
    toScreen: (a: number, b: number) => [number, number];
    toFrac: (x: number, y: number) => [number, number];
    drawRaster: (ctx: CanvasRenderingContext2D, img: CanvasImageSource & { width: number }) => void;
  };
  const cellTransform = React.useMemo<CellTransform>(() => {
    const identity: CellTransform = {
      toScreen: (a, b) => [
        ((b + 1) / CELL_TILE_REPEAT) * CELL_PANEL_PX,
        ((a + 1) / CELL_TILE_REPEAT) * CELL_PANEL_PX,
      ],
      toFrac: (x, y) => [
        (CELL_TILE_REPEAT * y) / CELL_PANEL_PX - 1,
        (CELL_TILE_REPEAT * x) / CELL_PANEL_PX - 1,
      ],
      drawRaster: (ctx, img) => ctx.drawImage(img, 0, 0, CELL_PANEL_PX, CELL_PANEL_PX),
    };
    if (!trueCellGeometry || !latticeVectors || latticeVectors.length !== 3) return identity;

    const u: [number, number] = [latticeVectors[1][0], latticeVectors[1][1]];
    const v: [number, number] = [latticeVectors[2][0], latticeVectors[2][1]];
    const det = u[0] * v[1] - u[1] * v[0];
    if (!Number.isFinite(det) || Math.abs(det) < 1e-6) return identity;

    // Bounding box (in u/v pixel space) of the 3x3-tiled parallelogram
    // (a, b in [-1, 2]), centered on the panel and scaled to fit with a
    // small margin.
    const extents = [-1, 2].flatMap((a) =>
      [-1, 2].map((b): [number, number] => {
        const da = a - 0.5;
        const db = b - 0.5;
        return [da * u[0] + db * v[0], da * u[1] + db * v[1]];
      }),
    );
    const maxRow = Math.max(...extents.map(([row]) => Math.abs(row)));
    const maxCol = Math.max(...extents.map(([, col]) => Math.abs(col)));
    const span = 2 * Math.max(maxRow, maxCol);
    if (!Number.isFinite(span) || span <= 0) return identity;
    const scale = (CELL_PANEL_PX * 0.92) / span;
    const cx = CELL_PANEL_PX / 2;
    const cy = CELL_PANEL_PX / 2;

    const toScreen = (a: number, b: number): [number, number] => {
      const da = a - 0.5;
      const db = b - 0.5;
      return [cx + scale * (da * u[1] + db * v[1]), cy + scale * (da * u[0] + db * v[0])];
    };
    const toFrac = (x: number, y: number): [number, number] => {
      const col = (x - cx) / scale;
      const row = (y - cy) / scale;
      const da = (v[1] * row - v[0] * col) / det;
      const db = (u[0] * col - u[1] * row) / det;
      return [da + 0.5, db + 0.5];
    };
    // img is the 3x3-tiled raster: pixel (xTile, yTile) is fractional
    // b = xTile / samples - 1, a = yTile / samples - 1. Express that as
    // the linear map above, composed into canvas transform coefficients.
    const drawRaster = (ctx: CanvasRenderingContext2D, img: CanvasImageSource & { width: number }) => {
      const samples = img.width / CELL_TILE_REPEAT;
      ctx.save();
      ctx.transform(
        (scale * v[1]) / samples, (scale * v[0]) / samples,
        (scale * u[1]) / samples, (scale * u[0]) / samples,
        cx - 1.5 * scale * (u[1] + v[1]), cy - 1.5 * scale * (u[0] + v[0]),
      );
      ctx.drawImage(img, 0, 0);
      ctx.restore();
    };
    return { toScreen, toFrac, drawRaster };
  }, [trueCellGeometry, latticeVectors]);

  const cellRef = React.useRef<HTMLCanvasElement>(null);
  React.useLayoutEffect(() => {
    const canvas = cellRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    canvas.width = CELL_PANEL_PX;
    canvas.height = CELL_PANEL_PX;
    ctx.imageSmoothingEnabled = false;
    ctx.fillStyle = themeColors.bg;
    ctx.fillRect(0, 0, CELL_PANEL_PX, CELL_PANEL_PX);
    if (!cellTile) return;
    cellTransform.drawRaster(ctx, cellTile);

    ctx.save();
    ctx.setLineDash([4, 3]);
    ctx.strokeStyle = themeColors.border;
    ctx.lineWidth = 1;
    ctx.beginPath();
    const corners: [number, number][] = [[0, 0], [0, 1], [1, 1], [1, 0]];
    corners.forEach(([a, b], i) => {
      const [x, y] = cellTransform.toScreen(a, b);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.closePath();
    ctx.stroke();
    ctx.restore();

    (positionsFrac || []).forEach(([a, b], i) => {
      const [x, y] = cellTransform.toScreen(a, b);
      ctx.beginPath();
      ctx.arc(x, y, CELL_SITE_RADIUS_PX, 0, 2 * Math.PI);
      ctx.strokeStyle = ATOM_COLORS[i % ATOM_COLORS.length];
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(x, y, 1.5, 0, 2 * Math.PI);
      ctx.fillStyle = ATOM_COLORS[i % ATOM_COLORS.length];
      ctx.fill();
    });
  }, [cellTile, positionsFrac, themeColors.bg, themeColors.border, cellTransform]);

  // Once a specific site count (not "Auto") has been reached, existing
  // sites can be dragged in the inset to refine their position - always at
  // the exact dragged fraction, no column/common-site snapping either way,
  // regardless of the snapToCommonSites toggle (that only applies to
  // clicks, which place/replace a site via the kernel's snap).
  const canDragSites = maxSites != null && (positionsFrac || []).length === maxSites;
  const cellDragRef = React.useRef<{ index: number } | null>(null);

  const cellFracFromEvent = React.useCallback(
    (e: { clientX: number; clientY: number; currentTarget: HTMLCanvasElement }): [number, number] => {
      const rect = e.currentTarget.getBoundingClientRect();
      const x = ((e.clientX - rect.left) / rect.width) * CELL_PANEL_PX;
      const y = ((e.clientY - rect.top) / rect.height) * CELL_PANEL_PX;
      return cellTransform.toFrac(x, y);
    },
    [cellTransform],
  );

  const hitTestSite = React.useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>): number => {
      const rect = e.currentTarget.getBoundingClientRect();
      const x = ((e.clientX - rect.left) / rect.width) * CELL_PANEL_PX;
      const y = ((e.clientY - rect.top) / rect.height) * CELL_PANEL_PX;
      const list = positionsFrac || [];
      for (let i = list.length - 1; i >= 0; i--) {
        const [a, b] = list[i];
        const [mx, my] = cellTransform.toScreen(a, b);
        if (Math.hypot(x - mx, y - my) <= CELL_SITE_HIT_PX) return i;
      }
      return -1;
    },
    [positionsFrac, cellTransform],
  );

  const handleCellMouseDown = React.useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (busy) return;
      if (canDragSites) {
        const hit = hitTestSite(e);
        if (hit !== -1) {
          cellDragRef.current = { index: hit };
        }
      }
    },
    [busy, canDragSites, hitTestSite],
  );

  const handleCellMouseMove = React.useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const drag = cellDragRef.current;
      if (!drag) return;
      const [a, b] = cellFracFromEvent(e);
      const next = (positionsFrac || []).map((site, i) =>
        i === drag.index ? [clamp(a, 0, 1), clamp(b, 0, 1)] : site,
      );
      setPositionsFrac(next);
    },
    [cellFracFromEvent, positionsFrac, setPositionsFrac],
  );

  const handleCellMouseUp = React.useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const wasDragging = cellDragRef.current !== null;
      cellDragRef.current = null;
      if (busy || wasDragging) return;
      const [a, b] = cellFracFromEvent(e);
      model.send({ action: "pick_site_frac", frac: [a, b] });
    },
    [busy, cellFracFromEvent, model],
  );

  const pointCount = (points || []).length;
  const hasFit = Boolean(latticeVectors && latticeVectors.length === 3);
  const siteCount = (positionsFrac || []).length;
  const sendAction = React.useCallback(
    (action: string) => model.send({ action }),
    [model],
  );

  const canvasBox = {
    position: "relative" as const,
    border: `${CANVAS_BORDER_PX}px solid ${themeColors.border}`,
    overflow: "hidden",
    width: canvasW,
    height: canvasH,
  };
  // The bordered canvas is a fixed pixel width, so the CONTENT column (title
  // row, canvas, footer, readout) is capped to that width and everything
  // aligns to the canvas's own edges. The background is a separate, always
  // full-width wrapper: capping IT to the content width would leave the
  // notebook/page background exposed beside the widget whenever the output
  // cell is wider than the canvas.
  const contentMaxWidth = canvasW + 2 * CANVAS_BORDER_PX;

  // MUI's built-in .Mui-disabled color (a low-opacity black/white) is nearly
  // invisible against this widget's own dark background, so every button
  // overrides it explicitly to the theme's muted text color instead.
  const buttonSx = (color: string) => ({
    ...compactButton,
    color,
    "&.Mui-disabled": { color: themeColors.textMuted },
  });

  // MUI defaults mark-label color to the light-mode text-secondary, which is
  // invisible on this widget's dark background, so recolor + resize them to
  // match the compact layout. Each label stays centered on its value's exact
  // position (MUI's default), including at the slider's own two ends.
  const sliderMarkSx = () => ({
    "& .MuiSlider-markLabel": {
      top: 22,
      fontSize: 10,
      color: themeColors.textMuted,
    },
  });

  return (
    <Box
      ref={rootRef}
      sx={{
        p: `${SPACING.LG}px`,
        bgcolor: themeColors.bg,
        color: themeColors.text,
        width: "100%",
        boxSizing: "border-box",
      }}
    >
      <Box sx={{ maxWidth: contentMaxWidth }}>
        <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: `${SPACING.SM}px` }}>
          <Typography sx={{ fontSize: 13, fontWeight: 600 }}>{title || "Choose Lattice"}</Typography>
          <Stack direction="row" spacing={1}>
            <Button
              size="small"
              sx={buttonSx(themeColors.accent)}
              disabled={busy || !hasImaging || pointCount < 3}
              onClick={() => sendAction("fit")}
            >
              Fit Lattice
            </Button>
            <Button
              size="small"
              sx={buttonSx(themeColors.accent)}
              disabled={busy || !hasImaging || !hasFit}
              onClick={() => sendAction("detect")}
            >
              Detect Atoms
            </Button>
            <Button
              size="small"
              sx={buttonSx(themeColors.accent)}
              disabled={busy || !numAtoms}
              onClick={() => sendAction("refine")}
            >
              Refine Atoms
            </Button>
            <Button
              size="small"
              sx={buttonSx(themeColors.accent)}
              disabled={busy || pointCount === 0}
              onClick={() => { setPoints([]); sendAction("reset"); }}
            >
              Clear Points
            </Button>
          </Stack>
        </Stack>

        {hasImaging && (
          <Stack direction="row" spacing={4} alignItems="center" sx={{ mb: "26px" }}>
            <Stack direction="row" spacing={1} alignItems="center">
              <Typography sx={{ fontSize: 10, color: themeColors.textMuted, whiteSpace: "nowrap" }}>
                Block size
              </Typography>
              <Slider
                size="small"
                min={BLOCK_SIZE_MIN_IDX}
                max={BLOCK_SIZE_MAX_IDX}
                step={1}
                value={blockSizeIdx}
                onChange={(_, v) => handleBlockSizeChange(v as number)}
                disabled={busy}
                valueLabelDisplay="auto"
                valueLabelFormat={(v) => (v >= BLOCK_SIZE_MAX_IDX ? "None" : String(v))}
                marks={[
                  { value: BLOCK_SIZE_MIN_IDX, label: "1" },
                  { value: BLOCK_SIZE_MAX_IDX, label: "None" },
                ]}
                sx={{ width: 150, mx: 1, color: themeColors.accent, ...sliderMarkSx() }}
                aria-label="Block size"
              />
              <Typography sx={{ fontSize: 10, color: themeColors.textMuted, minWidth: 28 }}>
                {blockSize == null ? "None" : blockSize}
              </Typography>
            </Stack>

            <Stack direction="row" spacing={1} alignItems="center">
              <Typography sx={{ fontSize: 10, color: themeColors.textMuted, whiteSpace: "nowrap" }}>
                Edge margin
              </Typography>
              <Box
                component="input"
                ref={edgeMarginRef}
                type="text"
                inputMode="numeric"
                value={edgeMarginText}
                disabled={busy}
                onFocus={() => { edgeMarginEditingRef.current = true; }}
                onChange={(e) => setEdgeMarginText(e.target.value.replace(/[^0-9]/g, ""))}
                onBlur={(e) => {
                  edgeMarginEditingRef.current = false;
                  commitEdgeMarginText(e.target.value);
                }}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    (e.target as HTMLInputElement).blur();
                  } else if (e.key === "Escape") {
                    edgeMarginEditingRef.current = false;
                    setEdgeMarginText(String(Math.round(edgeMinDistPxRef.current ?? 0)));
                    (e.target as HTMLInputElement).blur();
                  }
                }}
                sx={{
                  font: "inherit",
                  fontSize: 11,
                  fontFamily: "monospace",
                  color: EDGE_MARGIN_COLOR,
                  bgcolor: "transparent",
                  border: `1px solid ${themeColors.border}`,
                  borderRadius: "4px",
                  px: 1,
                  py: 0.25,
                  width: 40,
                  textAlign: "center",
                  cursor: busy ? "wait" : "text",
                  opacity: busy ? 0.5 : 1,
                  outline: "none",
                  "&:focus": { borderColor: EDGE_MARGIN_COLOR },
                }}
                aria-label="Edge margin"
              />
              <Stack direction="column" sx={{ lineHeight: 0 }}>
                <IconButton
                  size="small"
                  disabled={busy || (edgeMinDistPx ?? 0) >= edgeMarginMax}
                  onMouseDown={() => startEdgeMarginHold(1)}
                  onMouseUp={clearHoldTimer}
                  onMouseLeave={clearHoldTimer}
                  onTouchStart={() => startEdgeMarginHold(1)}
                  onTouchEnd={clearHoldTimer}
                  aria-label="Increase edge margin"
                  sx={{ p: 0, color: themeColors.textMuted }}
                >
                  <KeyboardArrowUpIcon sx={{ fontSize: 14 }} />
                </IconButton>
                <IconButton
                  size="small"
                  disabled={busy || (edgeMinDistPx ?? 0) <= 0}
                  onMouseDown={() => startEdgeMarginHold(-1)}
                  onMouseUp={clearHoldTimer}
                  onMouseLeave={clearHoldTimer}
                  onTouchStart={() => startEdgeMarginHold(-1)}
                  onTouchEnd={clearHoldTimer}
                  aria-label="Decrease edge margin"
                  sx={{ p: 0, color: themeColors.textMuted }}
                >
                  <KeyboardArrowDownIcon sx={{ fontSize: 14 }} />
                </IconButton>
              </Stack>
              <Typography sx={{ fontSize: 10, color: themeColors.textMuted }}>
                px — scroll or type
              </Typography>
            </Stack>
          </Stack>
        )}
      </Box>

      <Stack direction="row" spacing={`${SPACING.LG}px`} alignItems="flex-start">
        <Box sx={{ maxWidth: contentMaxWidth }}>
          <Box ref={containerRef} sx={canvasBox}>
            <canvas
              ref={canvasRef}
              style={{ position: "absolute", top: 0, left: 0, width: canvasW, height: canvasH, imageRendering: "pixelated" }}
            />
            <canvas
              ref={uiRef}
              style={{ position: "absolute", top: 0, left: 0, width: canvasW, height: canvasH, pointerEvents: "none" }}
            />
            <canvas
              width={canvasW}
              height={canvasH}
              style={{ position: "absolute", top: 0, left: 0, width: canvasW, height: canvasH, cursor: "crosshair", opacity: 0 }}
              onWheel={handleWheel}
              onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMoveReadout}
              onMouseUp={handleMouseUp}
              onMouseLeave={() => { dragRef.current = null; setCursorPos(null); }}
              onDoubleClick={handleDoubleClick}
            />
          </Box>

          <Typography sx={{ fontSize: 10, color: themeColors.textMuted, mt: `${SPACING.XS}px` }}>
            {pointCount < 3
              ? "Click to place the next point. Scroll to zoom, drag to pan."
              : "Drag a point to adjust it. Scroll to zoom, drag to pan."}
            {cursorPos && (
              <span style={{ marginLeft: 8, color: themeColors.accent }}>
                ({cursorPos[0].toFixed(1)}, {cursorPos[1].toFixed(1)})
              </span>
            )}
          </Typography>

          <Stack direction="row" spacing={1} alignItems="center" sx={{ mt: `${SPACING.XS}px` }}>
            <Button
              size="small"
              sx={buttonSx(showGrid ? themeColors.accent : themeColors.textMuted)}
              disabled={!hasFit}
              onClick={() => setShowGrid(!showGrid)}
            >
              {showGrid ? "Hide Grid" : "Show Grid"}
            </Button>
            <Button
              size="small"
              sx={buttonSx(showAtoms ? themeColors.accent : themeColors.textMuted)}
              disabled={!numAtoms}
              onClick={() => setShowAtoms(!showAtoms)}
            >
              {showAtoms ? "Hide Atoms" : "Show Atoms"}
            </Button>
            <Typography sx={{ fontSize: 10, color: themeColors.textMuted }}>
              {busy ? "Working…" : status}
            </Typography>
          </Stack>
        </Box>

        {hasFit && cellTile && (
          <Box sx={{ width: RIGHT_PANEL_PX, flexShrink: 0 }}>
            <canvas
              ref={cellRef}
              onMouseDown={handleCellMouseDown}
              onMouseMove={handleCellMouseMove}
              onMouseUp={handleCellMouseUp}
              onMouseLeave={() => { cellDragRef.current = null; }}
              style={{
                width: CELL_PANEL_PX,
                height: CELL_PANEL_PX,
                border: `${CANVAS_BORDER_PX}px solid ${themeColors.border}`,
                cursor: busy ? "wait" : canDragSites ? "grab" : "crosshair",
                imageRendering: "pixelated",
              }}
            />
            <Typography sx={{ fontSize: 10, color: themeColors.textMuted, mt: `${SPACING.XS}px` }}>
              Averaged cell ({cellCount} cells) —{" "}
              {canDragSites ? "drag a site to refine it" : "click a column to add a site"}
            </Typography>

            <Stack direction="row" spacing={1} sx={{ mt: `${SPACING.SM}px`, mb: `${SPACING.XS}px`, flexWrap: "wrap", gap: `${SPACING.XS}px` }}>
              <Button
                size="small"
                sx={buttonSx(themeColors.accent)}
                disabled={busy}
                onClick={() => sendAction("propose_sites")}
              >
                Auto Sites
              </Button>
              <Button
                size="small"
                sx={buttonSx(themeColors.accent)}
                disabled={busy || siteCount <= 1}
                onClick={() => sendAction("clear_sites")}
              >
                Clear Sites
              </Button>
              <Button
                size="small"
                sx={buttonSx(snapToCommonSites ? themeColors.accent : themeColors.textMuted)}
                disabled={busy}
                onClick={() => setSnapToCommonSites(!snapToCommonSites)}
              >
                {snapToCommonSites ? "Snap: On" : "Snap: Off"}
              </Button>
              <Button
                size="small"
                sx={buttonSx(trueCellGeometry ? themeColors.accent : themeColors.textMuted)}
                disabled={busy}
                onClick={() => setTrueCellGeometry(!trueCellGeometry)}
              >
                {trueCellGeometry ? "True Geometry: On" : "True Geometry: Off"}
              </Button>
            </Stack>

            <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: "26px" }}>
              <Typography sx={{ fontSize: 10, color: themeColors.textMuted, whiteSpace: "nowrap" }}>
                Sites
              </Typography>
              <Slider
                size="small"
                min={MAX_SITES_MIN}
                max={MAX_SITES_MAX}
                step={1}
                value={maxSites == null ? MAX_SITES_MIN : maxSites}
                onChange={(_, v) => handleMaxSitesChange(v as number)}
                disabled={busy}
                valueLabelDisplay="auto"
                valueLabelFormat={(v) => (v <= MAX_SITES_MIN ? "Auto" : String(v))}
                marks={[
                  { value: MAX_SITES_MIN, label: "Auto" },
                  { value: MAX_SITES_MAX, label: "10" },
                ]}
                sx={{ width: 120, mx: 1, color: themeColors.accent, ...sliderMarkSx() }}
                aria-label="Number of sites"
              />
              <Typography sx={{ fontSize: 10, color: themeColors.textMuted, minWidth: 28 }}>
                {maxSites == null ? "Auto" : maxSites}
              </Typography>
            </Stack>

            {(positionsFrac || []).map(([a, b], i) => (
              <Stack key={`site-${i}`} direction="row" spacing={1} alignItems="center">
                <Typography
                  sx={{
                    fontSize: 11,
                    fontFamily: "monospace",
                    color: ATOM_COLORS[i % ATOM_COLORS.length],
                  }}
                >
                  site {i}: ({a.toFixed(3)}, {b.toFixed(3)})
                </Typography>
                <Button
                  size="small"
                  sx={buttonSx(themeColors.textMuted)}
                  disabled={busy || siteCount <= 1}
                  onClick={() => model.send({ action: "remove_site", index: i })}
                >
                  ×
                </Button>
              </Stack>
            ))}
          </Box>
        )}
      </Stack>

      <Box sx={{ maxWidth: contentMaxWidth, mt: `${SPACING.SM}px` }}>
        {(pointLabels || []).map((label, i) => {
          const p = (points || [])[i];
          const origin = (points || [])[0];
          // Origin is reported as its raw pixel position; the other two
          // points are reported as lattice vectors relative to the origin
          // (u = a1 - origin, v = a2 - origin), not raw pixel positions.
          const isVector = i > 0;
          const value = isVector && p && origin
            ? [p[0] - origin[0], p[1] - origin[1]]
            : (!isVector ? p : null);
          return (
            <Typography key={label + i} sx={{ fontSize: 11, fontFamily: "monospace", color: value ? POINT_COLORS[i % POINT_COLORS.length] : themeColors.textMuted }}>
              {label}: {value ? `(${value[0].toFixed(1)}, ${value[1].toFixed(1)})` : "not placed"}
            </Typography>
          );
        })}
        {hasFit && (
          <Typography sx={{ fontSize: 11, fontFamily: "monospace", color: themeColors.textMuted, mt: `${SPACING.XS}px` }}>
            fitted: r0 ({latticeVectors[0][0].toFixed(1)}, {latticeVectors[0][1].toFixed(1)})
            {" "}u ({latticeVectors[1][0].toFixed(2)}, {latticeVectors[1][1].toFixed(2)})
            {" "}v ({latticeVectors[2][0].toFixed(2)}, {latticeVectors[2][1].toFixed(2)})
          </Typography>
        )}
      </Box>
    </Box>
  );
}

export const render = createRender(ChooseLattice);