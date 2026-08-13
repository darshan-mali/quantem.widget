# ChooseLattice

Pick an ordered origin and two lattice-vector points on a 2D image. Displays
a single image, lets you wheel-zoom and drag-pan to inspect a region, and
lets you click 3 ordered points whose pixel coordinates (in the ORIGINAL,
un-zoomed image) are exposed for downstream lattice-vector calculations.

```python
import numpy as np
from quantem.widget import ChooseLattice

widget = ChooseLattice(image, cmap="gray")
```

After clicking the origin, then `a1`, then `a2` on the image:

```python
widget.origin   # (row, col) or None
widget.a1       # (row, col) or None
widget.a2       # (row, col) or None
widget.u        # a1 - origin, or None until both are placed
widget.v        # a2 - origin, or None until both are placed
widget.points_array  # (n, 2) array of the picked (row, col) pairs so far
```

Use `set_points(...)` / `clear_points()` to set or reset the picks
programmatically.

## Fitting a lattice and detecting atoms

When `quantem.imaging` is importable (`widget.has_imaging`), the picked
points can be handed straight to a `quantem.imaging.Lattice`:

```python
widget.fit_lattice()      # refines origin/u/v; also runs from the "Fit Lattice" button
widget.detect_atoms()     # tiles the lattice with atom sites; "Detect Atoms" button
widget.refine_atoms()     # fits each atom center; "Refine Atoms" button

widget.lattice_vectors    # fitted [r0, u, v], each (row, col)
widget.is_fitted          # True once fit_lattice() has run
widget.atoms_array        # (n, 3) array of (row, col, site_index)
widget.lattice            # the backing quantem.imaging.Lattice, for further analysis
```

`fit_lattice()` refines in stages controlled by `block_size` (default `5`):
the fit is done in growing blocks of `block_size` unit cells at a time,
which avoids converging onto an aliased sub-lattice on real images. Pass
`block_size=None` (equivalent to `0`) to fit the whole image in one shot
instead — the widget also exposes this as a slider, with a "None" position
at each end and `1`–`10` in between.

On success, `fit_lattice()` snaps `points` (and therefore `origin`/`a1`/
`a2`/`u`/`v`) to the fitted `r0`/`u`/`v` from `lattice._lat`, rounded to 2
decimals, so the point markers move to reflect the fit rather than the
original clicks.

`reset_lattice()` drops the fitted lattice and detected atoms (picked
points are kept); editing the points does this automatically.

## Reference

```{eval-rst}
.. autoclass:: quantem.widget.choose_lattice.ChooseLattice
   :members:
   :show-inheritance:
```

## Interactive controls

| Control | Trait | Expected effect |
|---|---|---|
| Click on the image (fewer than 3 points placed) | `points` | Appends the next ordered point |
| Drag an existing point | `points` | Adjusts that point's pixel coordinates in place |
| Clear Points button | `points` | Resets to no points and any fitted lattice |
| Fit Lattice button | `lattice_vectors` | Runs `fit_lattice()`; enabled once 3 points are placed |
| Detect Atoms button | `atom_bytes`, `num_atoms` | Runs `detect_atoms()`; enabled once fitted |
| Refine Atoms button | `atom_bytes`, `num_atoms` | Runs `refine_atoms()`; enabled once atoms exist |
| Block size slider | `block_size` | `1`-`10` in the middle; both ends are "None" (whole-image fit) |
| Show/Hide Grid button | `show_grid` | Toggles the fitted-lattice grid overlay |
| Show/Hide Atoms button | `show_atoms` | Toggles the detected-atom overlay |
| Pan (drag) / zoom (wheel) | view transform | Image translates / zooms about the cursor |
| Double-click | view transform | Resets zoom/pan |

Picked points are colored cyan (Origin), magenta (`u`), and yellow (`v`).
Detected atoms are colored per site, cycling through the same palette as
`quantem.imaging.lattice_visualization.site_colors()` so overlay colors
match `widget.lattice.plot()`'s own atom-site coloring.
