# TODO: SORA 2.5 Ground Risk Tool

Open issues in `sora_app_v25_stable.py`. Examples use the built-in "Default: Unmanned Valley" flight
plan and the default drone (3 m, 25 m/s).

## Correctness (blocks use for flight preparation)

- [ ] **iGRC lookup does not follow SORA 2.5 Table 2** (`sora_app_v25_stable.py`, "Calculate Initial GRC").
  - The UA characteristic dimension and max speed are ignored: only the 1 m / 25 m/s column is used.
    The column must follow the worse of dimension and speed (1 m/25 m/s, 3 m/35 m/s, 8 m/75 m/s,
    20 m/120 m/s, 40 m/200 m/s).
  - Density band edges are 1/50/500/5000/50000 ppl/km²; Table 2 uses <5, <50, <500, <5000, <50000,
    >50000. Densities 5-49.9 now give iGRC 2 instead of 3.
  - Density 0 gives iGRC 1, the "controlled ground area" value. Zero population is not a controlled
    ground area; it is the <5 row unless a controlled ground area is declared explicitly.
  - Unmanned Valley example: the tool gives iGRC 1; Table 2 gives at least 3 for a 3 m UA.
- [ ] **Final GRC floor**: `max(1, initial_grc + mitigations)` must be floored at the controlled-ground-area
  value of the chosen column, not at 1.
- [ ] **M2 not implemented**: the app shows its own TODO; the GRB is not recalculated (step-back,
  Annex B/F) when M2 is applied.

## Population data

- [ ] **CBS and GHSL disagree completely** for the same GRB (Unmanned Valley example): CBS 0 ppl/km², GHSL 2504 ppl/km².
  Decide which source is authoritative and which "max density" rule is used.
- [ ] **CBS: suppressed cells counted as 0.** CBS 100 m only publishes cells with >= 5 inhabitants;
  negative "secret" values are set to 0. This underestimates risk. Treat suppressed/missing cells
  conservatively (1-4 inhabitants in a 100 m cell = 100-400 ppl/km²) or flag them.
  (The RD → WGS84 conversion is correct; in the example the nearest populated CBS cell is ~20 m outside the GRB.)
- [ ] **GHSL: max cell drives the result regardless of overlap.** In the example, 2504 ppl/km² comes from one cell
  overlapping the GRB by 3%. Consider an averaging window or overlap threshold as allowed by
  SORA 2.5 Annex F, and document the choice.
- [ ] GHSL source is labelled "GHSL GHS-POP 2025 (100m, stub)" but the file is `ghs_pop2023_100m.tif`.

## Setup

- [ ] GeoTIFF path: code expects `data/ghs_pop2023_100m.tif` (relative to the working directory);
  the file is in the repo root. Move it to `data/` or resolve the path relative to the script.
- [ ] `rasterio` missing from `requirements.txt`; without it GHSL silently reports "Unavailable".
  `requests` is also used but not listed.
- [x] CartoDB basemap tiles now require an API key: replaced by keyless PDOK BRT grijs (default),
  PDOK Luchtfoto and OpenStreetMap base layers (switch in the layer control).
- [ ] `shapely.ops.transform` is deprecated (warnings on every run).

## Next

- [ ] Run a real MAVLab operation (flight area KML + real drone dimension/speed) and compare with the
  approved SORA.
