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
 */

import * as React from "react";
import { createRender, useModel, useModelState } from "@anywidget/react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Stack from "@mui/material/Stack";
import Button from "@mui/material/Button";
import Slider from "@mui/material/Slider";
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

// Block-size slider: index 0 and index 11 both mean "None" (fit the whole
// image at once); indices 1-10 are staged block sizes of that size.
const BLOCK_SIZE_MIN_IDX = 0;
const BLOCK_SIZE_MAX_IDX = 11;
const blockSizeToIdx = (value: number | null): number =>
  value == null ? BLOCK_SIZE_MIN_IDX : Math.max(1, Math.min(10, Math.round(value)));
const idxToBlockSize = (idx: number): number | null =>
  idx <= BLOCK_SIZE_MIN_IDX || idx >= BLOCK_SIZE_MAX_IDX ? null : idx;

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

/** Draw the fitted lattice as two families of lines spanning the image. */
function drawLatticeGrid(
  ctx: CanvasRenderingContext2D,
  lat: number[][],
  height: number,
  width: number,
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
  // Clip to the actual image rectangle in screen space so nothing is ever
  // drawn beyond the original image's dimensions.
  const [ix0, iy0] = imgToScreen(0, 0);
  const [ix1, iy1] = imgToScreen(height, width);

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
      drawLatticeGrid(ctx, latticeVectors, height, width, imgToScreen);
    }
    if (showAtoms && atoms && atoms.length) {
      drawAtoms(ctx, atoms, canvasW, canvasH, imgToScreen);
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
  ]);

  const pointCount = (points || []).length;
  const hasFit = Boolean(latticeVectors && latticeVectors.length === 3);
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
              sx={{ ...compactButton, color: themeColors.accent }}
              disabled={busy || !hasImaging || pointCount < 3}
              onClick={() => sendAction("fit")}
            >
              Fit Lattice
            </Button>
            <Button
              size="small"
              sx={{ ...compactButton, color: themeColors.accent }}
              disabled={busy || !hasImaging || !hasFit}
              onClick={() => sendAction("detect")}
            >
              Detect Atoms
            </Button>
            <Button
              size="small"
              sx={{ ...compactButton, color: themeColors.accent }}
              disabled={busy || !numAtoms}
              onClick={() => sendAction("refine")}
            >
              Refine Atoms
            </Button>
            <Button
              size="small"
              sx={{ ...compactButton, color: themeColors.accent }}
              disabled={busy || pointCount === 0}
              onClick={() => { setPoints([]); sendAction("reset"); }}
            >
              Clear Points
            </Button>
          </Stack>
        </Stack>

        {hasImaging && (
          <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: `${SPACING.SM}px` }}>
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
              valueLabelFormat={(v) => (v <= BLOCK_SIZE_MIN_IDX || v >= BLOCK_SIZE_MAX_IDX ? "None" : String(v))}
              marks={[
                { value: BLOCK_SIZE_MIN_IDX, label: "None" },
                { value: 10, label: "10" },
                { value: BLOCK_SIZE_MAX_IDX, label: "None" },
              ]}
              sx={{ width: 180, mx: 1, color: themeColors.accent }}
              aria-label="Block size"
            />
            <Typography sx={{ fontSize: 10, color: themeColors.textMuted, minWidth: 28 }}>
              {blockSize == null ? "None" : blockSize}
            </Typography>
          </Stack>
        )}

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
            sx={{ ...compactButton, color: showGrid ? themeColors.accent : themeColors.textMuted }}
            disabled={!hasFit}
            onClick={() => setShowGrid(!showGrid)}
          >
            {showGrid ? "Hide Grid" : "Show Grid"}
          </Button>
          <Button
            size="small"
            sx={{ ...compactButton, color: showAtoms ? themeColors.accent : themeColors.textMuted }}
            disabled={!numAtoms}
            onClick={() => setShowAtoms(!showAtoms)}
          >
            {showAtoms ? "Hide Atoms" : "Show Atoms"}
          </Button>
          <Typography sx={{ fontSize: 10, color: themeColors.textMuted }}>
            {busy ? "Working…" : status}
          </Typography>
        </Stack>

        <Box sx={{ mt: `${SPACING.SM}px` }}>
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
    </Box>
  );
}

export const render = createRender(ChooseLattice);