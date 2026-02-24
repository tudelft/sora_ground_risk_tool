import streamlit as st
import folium
from streamlit_folium import st_folium
from folium.plugins import Draw
import math
import requests
import json
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple, Protocol
from enum import Enum
from shapely.geometry import Polygon, LineString, Point, shape, mapping
from shapely.ops import transform
from shapely import wkt
import pyproj
import xml.etree.ElementTree as ET
import os

# Session-state keys for constraint layer toggles (section 4)
CONSTRAINT_LAYER_KEYS = [
    "show_major_roads", "show_local_roads", "show_hospitals", "show_airports",
    "show_protected_areas", "show_power_lines", "show_vital_infrastructure",
]

# --- 1. CORE DATA MODELS ---
@dataclass
class MissionSpec:
    """All user / aircraft inputs needed for Step-2 buffer maths."""
    v0: float
    dimension: float
    t_rz: float
    s_gps: float
    s_pos: float
    s_map: float
    h_gps: float
    h_fg: float
    v_wind: float
    v_climb: float
    maneuver_type: str
    pitch_angle: float
    bank_angle: float
    emergency_type: str
    t_open: float
    v_z: float
    glide_ratio: float
    geometry_type: str


@dataclass
class BufferMetrics:
    """Traceable output from SORA buffer engine."""
    w_cv: float
    w_grb: float
    h_cv: float


@dataclass
class BufferGeometries:
    """Geometries returned in WGS84 + associated metrics."""
    FG: Polygon
    CV: Polygon
    GRB: Polygon
    metrics: BufferMetrics


# --- 1.1 MITIGATION ENUMS ---
class MitigationType(Enum):
    """SORA v2.5 mitigation types."""
    M1A_SHELTERING = "M1(A) - Strategic mitigations - Sheltering"
    M1B_OPERATIONAL_RESTRICTIONS = "M1(B) - Strategic mitigations - Operational restrictions"
    M1C_GROUND_OBSERVATION = "M1(C) - Tactical mitigations - Ground observation"
    M2_IMPACT_DYNAMICS = "M2 - Effects of UA impact dynamics are reduced"


class RobustnessLevel(Enum):
    """Robustness levels for mitigations."""
    NONE = "None"
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


# --- 1.2 MITIGATION DATA MODELS ---
@dataclass
class MitigationConfig:
    """Configuration for a single mitigation application."""
    mitigation_type: MitigationType
    level: RobustnessLevel
    corrective_factor: float
    time_constraint_applied: bool = False
    description: str = ""  # User's description of how they meet requirements
    integrity_description: str = ""  # User's description of integrity evidence
    assurance_description: str = ""  # User's description of assurance evidence
    reference_document_name: str = ""  # Compliance matrix: Document name
    reference_chapter_page: str = ""  # Compliance matrix: Chapter/Page


@dataclass
class MitigationSet:
    """Complete set of applied mitigations."""
    m1a: Optional[MitigationConfig] = None
    m1b: Optional[MitigationConfig] = None
    m1c: Optional[MitigationConfig] = None
    m2: Optional[MitigationConfig] = None
    
    def validate(self) -> Tuple[bool, Optional[str]]:
        """Validate mitigation combination rules."""
        # Rule: M1(A) Medium cannot be combined with M1(B) at any level
        # M1(A) Medium includes time-based restrictions which conflict with M1(B)
        # M1(A) Low + M1(B) is allowed because M1(A) Low has no time restrictions
        if self.m1a and self.m1b:
            if self.m1a.level == RobustnessLevel.MEDIUM:
                return False, "M1(A) Medium robustness cannot be combined with M1(B) mitigations. M1(A) Medium includes time-based restrictions which conflict with M1(B). See Annex B assurance requirements."
        return True, None
    
    def calculate_total_reduction(self) -> float:
        """Calculate total GRC reduction (additive)."""
        total = 0.0
        if self.m1a:
            total += self.m1a.corrective_factor
        if self.m1b:
            total += self.m1b.corrective_factor
        if self.m1c:
            total += self.m1c.corrective_factor
        if self.m2:
            total += self.m2.corrective_factor
        return total
    
    def requires_grb_recalculation(self) -> bool:
        """Check if M2 is applied, requiring GRB step-back procedure."""
        return self.m2 is not None
    
    def to_dict(self) -> Dict:
        """Serialize for session persistence."""
        result = {}
        if self.m1a:
            m1a_dict = asdict(self.m1a)
            m1a_dict["mitigation_type"] = self.m1a.mitigation_type.value
            m1a_dict["level"] = self.m1a.level.value
            result["m1a"] = m1a_dict
        if self.m1b:
            m1b_dict = asdict(self.m1b)
            m1b_dict["mitigation_type"] = self.m1b.mitigation_type.value
            m1b_dict["level"] = self.m1b.level.value
            result["m1b"] = m1b_dict
        if self.m1c:
            m1c_dict = asdict(self.m1c)
            m1c_dict["mitigation_type"] = self.m1c.mitigation_type.value
            m1c_dict["level"] = self.m1c.level.value
            result["m1c"] = m1c_dict
        if self.m2:
            m2_dict = asdict(self.m2)
            m2_dict["mitigation_type"] = self.m2.mitigation_type.value
            m2_dict["level"] = self.m2.level.value
            result["m2"] = m2_dict
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> "MitigationSet":
        """Deserialize from session persistence."""
        result = cls()
        if data.get("m1a"):
            m1a_data = data["m1a"].copy()
            m1a_data["mitigation_type"] = MitigationType(m1a_data["mitigation_type"])
            m1a_data["level"] = RobustnessLevel(m1a_data["level"])
            result.m1a = MitigationConfig(**m1a_data)
        if data.get("m1b"):
            m1b_data = data["m1b"].copy()
            m1b_data["mitigation_type"] = MitigationType(m1b_data["mitigation_type"])
            m1b_data["level"] = RobustnessLevel(m1b_data["level"])
            result.m1b = MitigationConfig(**m1b_data)
        if data.get("m1c"):
            m1c_data = data["m1c"].copy()
            m1c_data["mitigation_type"] = MitigationType(m1c_data["mitigation_type"])
            m1c_data["level"] = RobustnessLevel(m1c_data["level"])
            result.m1c = MitigationConfig(**m1c_data)
        if data.get("m2"):
            m2_data = data["m2"].copy()
            m2_data["mitigation_type"] = MitigationType(m2_data["mitigation_type"])
            m2_data["level"] = RobustnessLevel(m2_data["level"])
            result.m2 = MitigationConfig(**m2_data)
        return result


class PopulationSource(Protocol):
    """Interface for any population dataset (CBS, GHSL, WorldPop, etc.)."""

    name: str
    resolution_m: int
    vintage: str  # e.g. "2023"

    def query(
        self, polygon: Polygon
    ) -> Tuple[int, int, str, Optional[Dict], List[Dict], Dict]:
        """
        Return:
          - max_density: int (people / km^2)
          - cell_count: int (number of grid cells analysed in bbox)
          - method: str (description of intersection method)
          - hotspot_geom: GeoJSON-like dict or None
          - grid_features: list of GeoJSON Feature dicts (for map overlay)
          - debug_info: dict with extra diagnostics (cell table, attempts, etc.)
        """
        ...


# --- 2. SORA MATH ENGINE ---
class SORACalculator:
    def __init__(self, specs: MissionSpec):
        self.specs = specs
        self.g = 9.81

    def calculate_cv_lateral(self):
        v0 = self.specs.v0
        t_rz = self.specs.t_rz
        errors = self.specs.s_gps + self.specs.s_pos + self.specs.s_map
        s_rz = v0 * t_rz
        s_cm = 0
        if self.specs.maneuver_type == "Multirotor (Stop)":
            theta = math.radians(self.specs.pitch_angle)
            s_cm = (v0 ** 2) / (2 * self.g * math.tan(theta))
        else:
            theta = math.radians(self.specs.bank_angle)
            s_cm = (v0 ** 2) / (self.g * math.tan(theta))
        return s_rz + s_cm + errors

    def calculate_cv_height(self):
        h_fg = self.specs.h_fg
        h_rz = self.specs.v_climb * self.specs.t_rz
        h_cm = (self.specs.v0 ** 2) / (2 * self.g)
        return h_fg + h_rz + h_cm + self.specs.h_gps

    def calculate_grb_width(self, h_cv):
        v0 = self.specs.v0
        v_wind = self.specs.v_wind
        if self.specs.emergency_type == "Parachute (Ballistic)":
            dist_deploy = v0 * self.specs.t_open
            t_descent = h_cv / self.specs.v_z
            return dist_deploy + (v_wind * t_descent)
        else:
            dist_glide = h_cv * self.specs.glide_ratio
            return dist_glide + (v_wind * (dist_glide / v0))

    def get_buffers(self, geometry_coords: List[List[float]]) -> BufferGeometries:
        project_to_m = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32631", always_xy=True).transform
        project_to_deg = pyproj.Transformer.from_crs("EPSG:32631", "EPSG:4326", always_xy=True).transform
        points_m = [transform(project_to_m, Point(p[1], p[0])) for p in geometry_coords]
        
        if self.specs.geometry_type == "Polygon (Area)":
            base = Polygon([(p.x, p.y) for p in points_m])
            if not base.is_valid:
                base = LineString([(p.x, p.y) for p in points_m]).buffer(1)
        else:
            # Corridor: start from centre line and add position + map error as initial width
            base = LineString([(p.x, p.y) for p in points_m]).buffer(
                self.specs.s_pos + self.specs.s_map
            )

        w_cv = self.calculate_cv_lateral()
        h_cv = self.calculate_cv_height()
        w_grb = self.calculate_grb_width(h_cv)
        
        fg = base
        cv = fg.buffer(w_cv)
        grb = cv.buffer(w_grb)
        
        metrics = BufferMetrics(w_cv=w_cv, w_grb=w_grb, h_cv=h_cv)

        return BufferGeometries(
            FG=transform(project_to_deg, fg),
            CV=transform(project_to_deg, cv),
            GRB=transform(project_to_deg, grb),
            metrics=metrics,
        )


# --- 2.1 MITIGATION CORRECTIVE FACTORS ---
def get_corrective_factor(mitigation_type: MitigationType, level: RobustnessLevel) -> float:
    """
    Get corrective factor from Table 11 (Annex B).
    
    Returns:
        Corrective factor (negative value for GRC reduction)
    """
    # Table 11: Mitigations effect for final GRC determination
    factors = {
        (MitigationType.M1A_SHELTERING, RobustnessLevel.LOW): -1.0,
        (MitigationType.M1A_SHELTERING, RobustnessLevel.MEDIUM): -2.0,
        (MitigationType.M1B_OPERATIONAL_RESTRICTIONS, RobustnessLevel.MEDIUM): -1.0,
        (MitigationType.M1B_OPERATIONAL_RESTRICTIONS, RobustnessLevel.HIGH): -2.0,
        (MitigationType.M1C_GROUND_OBSERVATION, RobustnessLevel.LOW): -1.0,
        (MitigationType.M2_IMPACT_DYNAMICS, RobustnessLevel.MEDIUM): -1.0,
        (MitigationType.M2_IMPACT_DYNAMICS, RobustnessLevel.HIGH): -2.0,
    }
    return factors.get((mitigation_type, level), 0.0)


# --- 2.2 MITIGATION REQUIREMENTS (Annex B Tables 2-10) ---
# Structure for integrity and assurance requirements from Annex B Tables 2-10
# All mitigation requirements populated from SORA v2.5 Annex B (Tables 2-10)
MITIGATION_REQUIREMENTS = {
    MitigationType.M1A_SHELTERING: {
        RobustnessLevel.LOW: {
            "integrity": [
                "Criterion #1 (Evaluation of people at risk): If the applicant claims a reduction due to a sheltered operational environment, the applicant: flies over operational environments generally consisting of structures providing shelter, and it is reasonable to expect that on average a vast majority of the uninvolved people will be located under a structure. This mitigation cannot work when only overflying open-air assemblies of people or areas with no shelter.",
                "Criterion #2 (Evaluation of Penetration Hazard): The applicant uses a drone that is not expected to penetrate structures and fatally injure people under the shelter.",
            ],
            "assurance": [
                "Criterion #1 (Evaluation of people at risk): The applicant declares that the operation is in an environment that has structures providing shelter where people are generally expected to be, and the applicant does not fly over large open-air assemblies of people.",
                "Criterion #2 (Evaluation of Penetration Hazard): The applicant declares that the UA used is under 25 kg MTOM. OR For UA with MTOM higher than 25 kg, the applicant has supporting evidence that the required level of integrity is achieved. This is typically done by means of testing, analysis, simulation, inspection, design review or through operational experience.",
            ],
            "operator_declaration_required": True,
        },
        RobustnessLevel.MEDIUM: {
            "integrity": [
                "Criterion #1 (Evaluation of people at risk): Same as Low. In addition, the applicant restricts operating times and demonstrates that an even greater proportion of uninvolved people are sheltered.",
                "Criterion #2 (Evaluation of Penetration Hazard): The applicant uses a drone that is not expected to penetrate structures and fatally injure people under the shelter.",
            ],
            "assurance": [
                "Criterion #1 (Evaluation of people at risk): Same as Low. In addition, the applicant has time-based restrictions in place and evidence to support that a higher proportion of people are sheltered. Medium robustness M1(A) mitigation cannot be combined with M1(B) mitigations.",
                "Criterion #2 (Evaluation of Penetration Hazard): The applicant declares that the UA used is under 25 kg MTOM. OR For UA with MTOM higher than 25 kg, the applicant has supporting evidence that the required level of integrity is achieved. This is typically done by means of testing, analysis, simulation, inspection, design review or through operational experience.",
            ],
            "operator_declaration_required": True,
        },
    },
    MitigationType.M1B_OPERATIONAL_RESTRICTIONS: {
        RobustnessLevel.MEDIUM: {
            "integrity": [
                "Criterion #1 (Evaluation of people at risk): The applicant provides spacetime-based restrictions (e.g., flying over a market square when it is not crowded) to substantiate that the actual density of people during the operation is lower than in Step #2. This can be done by means of: An analysis or appraisal of characteristics of the location and time of operation, AND/OR Use of temporal density data (e.g., data from a supplemental data service provider) relevant for the proposed area. This can incorporate real time or historical data.",
                "Criterion #2 (Impact on Risk Population): The at-risk population is lowered by at least 1 iGRC population band (~90%) using one or more methods described in the Level of Integrity for Criterion #1 above.",
            ],
            "assurance": [
                "Criterion #1 (Evaluation of people at risk): All mapping products, data sources and processes used to claim lowering the density of population at risk are accepted by the competent authority.",
                "Criterion #2 (Impact on Risk Population): The applicant has supporting evidence that the required level of integrity is achieved. This is typically done by means of analysis, simulation, surveys or through operational experience.",
            ],
            "operator_declaration_required": False,
        },
        RobustnessLevel.HIGH: {
            "integrity": [
                "Criterion #1 (Evaluation of people at risk): The applicant provides spacetime-based restrictions (e.g., flying over a market square when it is not crowded) to substantiate that the actual density of people during the operation is lower than in Step #2. This can be done by means of: An analysis or appraisal of characteristics of the location and time of operation, AND/OR Use of temporal density data (e.g., data from a supplemental data service provider) relevant for the proposed area. This can incorporate real time or historical data.",
                "Criterion #2 (Impact on Risk Population): The at-risk population is lowered by at least 2 iGRC population bands (~99%) using one or more methods described in the Level of Integrity for Criterion #1 above.",
            ],
            "assurance": [
                "Criterion #1 (Evaluation of people at risk): All mapping products, data sources and processes used to claim lowering the density of population at risk are accepted by the competent authority.",
                "Criterion #2 (Impact on Risk Population): The claimed level of integrity is validated by a competent third party against a standard considered adequate by the competent authority and/or in accordance with means of compliance acceptable to that authority.",
            ],
            "operator_declaration_required": False,
        },
    },
    MitigationType.M1C_GROUND_OBSERVATION: {
        RobustnessLevel.LOW: {
            "integrity": [
                "Criterion #1 (Procedures): To achieve a reduction of people at risk: The remote crew members observe the vast majority of the overflown areas during the operation, and identify area(s) of less risk on the ground; The remote pilot will reduce the number of people at risk by adjusting the flight path while the operation is ongoing (e.g., flying away from the area with a higher risk on the ground or overflying only the identified area(s) of less risk on the ground).",
                "Criterion #2 (Technical Means): If the mitigation is achieved through the use of technical means (e.g., camera(s) mounted on the UA or visual ground observers with radios/phones), these should provide data of sufficient quality allowing reliable detection of uninvolved people on the ground.",
            ],
            "assurance": [
                "Criterion #1 (Procedures): The operational procedures for the mitigation are documented. The applicant declares that the required level of integrity has been achieved.",
                "Criterion #2 (Technical Means): Authorities may allow the use of technical means for ground observation with assurance criteria acceptable to them.",
            ],
            "operator_declaration_required": True,
        },
    },
    MitigationType.M2_IMPACT_DYNAMICS: {
        RobustnessLevel.MEDIUM: {
            "integrity": [
                "Criterion #1 (Technical design): Effects of impact dynamics and immediate post impact hazards, critical area or the combination of these results are reduced such that the risk to population is reduced by an approximate 1 order of magnitude (90%). When applicable, in case of malfunctions, failures or any combinations thereof that may lead to a crash, the UAS contains all elements required for the activation of the mitigation. When applicable, any failure or malfunction of the proposed mitigation itself (e.g., inadvertent activation) does not adversely affect the safety of the operation.",
                "Criterion #2 (Procedures): Any equipment used to reduce the effect of the UA impact dynamics are installed and maintained in accordance with UAS/Mitigation designer instructions.",
                "Criterion #3 (Training): When use of the mitigation requires action from the remote crew, then appropriate training must be provided for the remote crew by the operator. The operator must ensure that the personnel responsible (internal or external) for the installation and maintenance of the mitigation measures are qualified for the task.",
            ],
            "assurance": [
                "Criterion #1 (Technical design): The applicant has supporting evidence to claim the required level of integrity and reliability is achieved. This is typically done by means of testing, analysis, simulation, inspection, design review or through operational experience. A UAS with MTOM lower or equal to 900g and a maximum speed of 19 m/s fulfils this assurance Criteria 1.",
                "Criterion #2 (Procedures): Procedures are validated against standards considered adequate by the competent authority and/or in accordance with means of compliance acceptable to that authority. The adequacy of the procedures is proved through: Dedicated flight tests, or Simulation, provided that the representativeness of the simulation means is proven for the intended purpose with positive results.",
                "Criterion #3 (Training): Training syllabus is available. The operator provides theoretical and practical training for the remote crew. Personnel responsible for installation and maintenance of the mitigation measures have completed relevant training.",
            ],
            "operator_declaration_required": False,
        },
        RobustnessLevel.HIGH: {
            "integrity": [
                "Criterion #1 (Technical design): Same as Medium. In addition: When applicable, the activation of the mitigation is automated. The effects of impact dynamics and immediate post impact hazards, critical area or the combination of them are reduced such that the risk to the population is reduced by an approximate 2 orders of magnitude (99%).",
                "Criterion #2 (Procedures): [Same as Medium]",
                "Criterion #3 (Training): [Same as Medium]",
            ],
            "assurance": [
                "Criterion #1 (Technical design): The claimed level of integrity is validated by a competent third party against a standard considered adequate by the competent authority and/or in accordance with means of compliance acceptable to that authority (when applicable).",
                "Criterion #2 (Procedures): Same as Medium. In addition: Flight tests performed to validate the procedures cover the complete flight envelope or are proven to be conservative. The procedures, flight tests and simulations are validated by a competent third party.",
                "Criterion #3 (Training): Same as Medium. In addition, a competent third party: validates the training syllabus. Verifies the remote crew competencies",
            ],
            "operator_declaration_required": False,
        },
    },
}

# --- 3. DATA FETCHER (STRICT LOGIC, CBS IMPLEMENTATION) ---
def sanitize_geometry(geojson):
    if not geojson: return None
    new_geo = json.loads(json.dumps(geojson))
    
    rd_to_wgs = pyproj.Transformer.from_crs("EPSG:28992", "EPSG:4326", always_xy=True)

    def recursive_fix(coords):
        if isinstance(coords[0], (list, tuple)): return [recursive_fix(c) for c in coords]
        x, y = coords[0], coords[1]
        if x > 1000 or y > 1000:
            lon, lat = rd_to_wgs.transform(x, y)
            return [lon, lat]
        if x > 40 and y < 15: return [y, x]
        return [x, y]

    new_geo['coordinates'] = recursive_fix(new_geo['coordinates'])
    return new_geo

class CbsPopulationSource:
    """Current CBS 100m grid implementation as a PopulationSource."""

    name: str = "CBS 100m"
    resolution_m: int = 100
    vintage: str = "2023"

    def query(
        self, polygon: Polygon
    ) -> Tuple[int, int, str, Optional[Dict], List[Dict], Dict]:
        return _get_population_density_cbs(polygon)


class MockScaledPopulationSource:
    """
    Simple second population source for comparison.
    It reuses CBS data but scales the resulting density to simulate a different dataset.
    """

    name: str = "Mock: CBS x1.2"
    resolution_m: int = 100
    vintage: str = "2023"

    def query(
        self, polygon: Polygon
    ) -> Tuple[int, int, str, Optional[Dict], List[Dict], Dict]:
        base_density, count, method, hotspot_geom, grid_features, debug_info = _get_population_density_cbs(polygon)
        scaled_density = int(base_density * 1.2)
        debug_info = dict(debug_info)
        debug_info["source"] = {
            "name": self.name,
            "resolution_m": self.resolution_m,
            "vintage": self.vintage,
            "provider": "Mock derived from CBS 100m",
        }
        return scaled_density, count, method, hotspot_geom, grid_features, debug_info


class GhslPopulationSource:
    """
    Skeleton adapter for GHSL GHS-POP population grid.

    NOTE: This currently validates environment and reports status, but does not yet
    compute real GHSL-based densities until a GeoTIFF path is configured and
    raster dependencies are installed.
    """

    name: str = "GHSL GHS-POP 2025 (100m, stub)"
    resolution_m: int = 100
    vintage: str = "2025"

    # Default relative path placeholder; update to your actual file
    ghsl_tiff_path: str = os.path.join("data", "ghs_pop2023_100m.tif")

    def query(
        self, polygon: Polygon
    ) -> Tuple[int, int, str, Optional[Dict], List[Dict], Dict]:
        debug_info: Dict = {
            "source": {
                "name": self.name,
                "resolution_m": self.resolution_m,
                "vintage": self.vintage,
                "provider": "GHSL GHS-POP (Copernicus)",
            },
            "notes": [],
            "cell_analysis": [],
        }

        # Check for rasterio availability
        try:
            import rasterio  # type: ignore
            from rasterio import windows  # type: ignore
        except ImportError:
            debug_info["notes"].append(
                "rasterio not installed: GHSL integration is unavailable."
            )
            return 0, 0, "Unavailable (rasterio missing)", None, [], debug_info

        # Check that a GHSL GeoTIFF is present
        if not os.path.isfile(self.ghsl_tiff_path):
            debug_info["notes"].append(
                f"GHSL GeoTIFF not found at '{self.ghsl_tiff_path}'. "
                "Place a GHS-POP 100m GeoTIFF there or update GhslPopulationSource.ghsl_tiff_path."
            )
            return 0, 0, "Unavailable (file missing)", None, [], debug_info

        try:
            with rasterio.open(self.ghsl_tiff_path) as src:
                raster_crs = src.crs
                # Transformers between WGS84 and raster CRS
                to_raster = pyproj.Transformer.from_crs(
                    "EPSG:4326", raster_crs, always_xy=True
                ).transform
                to_wgs84 = pyproj.Transformer.from_crs(
                    raster_crs, "EPSG:4326", always_xy=True
                ).transform

                # Project GRB polygon into raster CRS
                poly_raster = transform(to_raster, polygon)

                # Use a context buffer (~1 km) around the GRB to inspect nearby cells
                context_buffer_m = 1000.0
                context_poly = poly_raster.buffer(context_buffer_m)

                # Compute raster window covering buffered polygon bounds
                minx, miny, maxx, maxy = context_poly.bounds
                win = windows.from_bounds(minx, miny, maxx, maxy, src.transform)

                data = src.read(1, window=win, masked=True)

                # Approximate cell area (m^2) and factor to km^2
                pixel_width = src.transform.a
                pixel_height = -src.transform.e
                cell_area_m2 = abs(pixel_width * pixel_height)
                cell_area_km2 = cell_area_m2 / 1_000_000.0 if cell_area_m2 > 0 else 0

                strict_max_pop = 0.0
                strict_count = 0
                adjacent_max_pop = 0.0
                hotspot_geom_wgs: Optional[Dict] = None
                grid_features: List[Dict] = []
                cell_rows, cell_cols = data.shape

                for r in range(cell_rows):
                    for c in range(cell_cols):
                        val = float(data[r, c])
                        if getattr(data, "mask", None) is not None and data.mask[r, c]:
                            continue
                        if val <= 0:
                            continue

                        # Pixel indices in full raster
                        row = win.row_off + r
                        col = win.col_off + c

                        # Cell corners in raster CRS
                        x_left, y_top = src.transform * (col, row)
                        x_right, y_bottom = src.transform * (col + 1, row + 1)
                        cell_poly_raster = Polygon(
                            [
                                (x_left, y_top),
                                (x_right, y_top),
                                (x_right, y_bottom),
                                (x_left, y_bottom),
                                (x_left, y_top),
                            ]
                        )

                        # Track adjacent max regardless of overlap
                        if val > adjacent_max_pop:
                            adjacent_max_pop = val

                        # Only consider cells within the 1km context buffer
                        if not context_poly.intersects(cell_poly_raster):
                            continue

                        overlap_pct = 0.0
                        # Strict intersection: only cells touching the GRB itself
                        if poly_raster.intersects(cell_poly_raster):
                            strict_count += 1
                            try:
                                overlap_area = poly_raster.intersection(
                                    cell_poly_raster
                                ).area
                                total_area = cell_poly_raster.area
                                if total_area > 0:
                                    overlap_pct = (overlap_area / total_area) * 100.0
                            except Exception:
                                pass

                            if val >= strict_max_pop:
                                strict_max_pop = val
                                # Track hotspot geometry in WGS84
                                hotspot_geom_wgs = mapping(
                                    transform(to_wgs84, cell_poly_raster)
                                )

                            # Basic cell record for the debug table (strict only)
                            centroid_raster = cell_poly_raster.centroid
                            centroid_wgs = transform(to_wgs84, centroid_raster)
                            debug_info["cell_analysis"].append(
                                {
                                    "Population": int(val),
                                    "Overlap %": round(overlap_pct, 2),
                                    "Lat": round(centroid_wgs.y, 5),
                                    "Lon": round(centroid_wgs.x, 5),
                                }
                            )

                        # Build a GeoJSON feature for map overlay (context + strict)
                        geom_wgs = mapping(transform(to_wgs84, cell_poly_raster))
                        color = "#008080"  # teal
                        if val > 200:
                            color = "#000000"
                        elif val >= 100:
                            color = "#800080"
                        feature = {
                            "type": "Feature",
                            "geometry": geom_wgs,
                            "properties": {
                                "viz_color": color,
                                "viz_pop": int(val),
                                "overlap_pct": round(overlap_pct, 1),
                                "is_hotspot": False,
                            },
                        }
                        grid_features.append(feature)

                # Convert max population per cell to ppl/km² using cell area
                if strict_count > 0 and cell_area_km2 > 0:
                    max_density = int(strict_max_pop / cell_area_km2)
                    method = "GHSL raster intersection"
                else:
                    max_density = 0
                    method = "None (Safe)"

                debug_info["adjacent_max"] = adjacent_max_pop

                # Mark hotspot feature if available
                if hotspot_geom_wgs is not None:
                    for f in grid_features:
                        if f["geometry"] == hotspot_geom_wgs:
                            f["properties"]["is_hotspot"] = True
                            f["properties"]["viz_color"] = "#0000FF"
                            break

                return max_density, strict_count, method, hotspot_geom_wgs, grid_features, debug_info

        except Exception as e:  # pragma: no cover - defensive
            debug_info.setdefault("attempts", [])
            debug_info["attempts"].append(str(e))
            return 0, 0, "Error", None, [], debug_info


# --- 4. CONSTRAINT LAYERS (ROADS) ---
def get_major_roads(grb_polygon_wgs: Polygon):
    """
    Fetch Rijksweg lane segments (Weggegevens max speeds) around the GRB and
    compute how much of the GRB overlaps with high‑speed national roads.

    Major roads/highways are defined here as segments with a maximum speed
    of at least 80 km/h (omschr >= 80).
    """
    rd_crs = "EPSG:28992"
    wgs_crs = "EPSG:4326"
    to_rd = pyproj.Transformer.from_crs(wgs_crs, rd_crs, always_xy=True).transform
    to_wgs = pyproj.Transformer.from_crs(rd_crs, wgs_crs, always_xy=True).transform

    grb_rd = transform(to_rd, grb_polygon_wgs)
    minx, miny, maxx, maxy = grb_rd.bounds
    buffer_m = 1000.0

    bbox = f"{minx-buffer_m},{miny-buffer_m},{maxx+buffer_m},{maxy+buffer_m},EPSG:28992"
    url = "https://service.pdok.nl/rws/weggeg/wfs/v1_0"
    params = {
        "service": "WFS",
        "request": "GetFeature",
        "version": "2.0.0",
        "typeNames": "weggeg:wegvak_max_snelheden",
        "outputFormat": "application/json",
        "bbox": bbox,
    }

    road_features_wgs: List[Dict] = []
    total_overlap_m = 0.0

    try:
        r = requests.get(url, params=params, timeout=8)
        if r.status_code != 200:
            return road_features_wgs, total_overlap_m

        data = r.json()
        for f in data.get("features", []):
            geom_rd = shape(f["geometry"])
            props = f.get("properties", {})

            # omschr contains the max speed as a string, e.g. "120", "100", "80"
            try:
                vmax = float(str(props.get("omschr", "0")).strip() or 0.0)
            except ValueError:
                vmax = 0.0

            is_major = vmax >= 80.0

            if geom_rd.intersects(grb_rd) and is_major:
                try:
                    total_overlap_m += geom_rd.intersection(grb_rd).length
                except Exception:
                    pass

            # Convert geometry to WGS84 for display
            geom_wgs = transform(to_wgs, geom_rd)
            road_features_wgs.append(
                {
                    "type": "Feature",
                    "geometry": mapping(geom_wgs),
                    "properties": {**props, "is_major": is_major, "vmax": vmax},
                }
            )

    except Exception:
        # Fail quietly – constraint layer is helpful but not critical
        return road_features_wgs, total_overlap_m

    return road_features_wgs, total_overlap_m


def get_local_roads(grb_polygon_wgs: Polygon):
    """
    Fetch general road segments from PDOK NWB Wegen around the GRB for
    visual context (local / municipal roads etc.). No overlap metrics
    are computed here; this is purely for display.
    """
    rd_crs = "EPSG:28992"
    wgs_crs = "EPSG:4326"
    to_rd = pyproj.Transformer.from_crs(wgs_crs, rd_crs, always_xy=True).transform
    to_wgs = pyproj.Transformer.from_crs(rd_crs, wgs_crs, always_xy=True).transform

    grb_rd = transform(to_rd, grb_polygon_wgs)
    minx, miny, maxx, maxy = grb_rd.bounds
    buffer_m = 1000.0

    bbox = f"{minx-buffer_m},{miny-buffer_m},{maxx+buffer_m},{maxy+buffer_m},EPSG:28992"
    url = "https://service.pdok.nl/rws/nwbwegen/wfs/v1_0"
    params = {
        "service": "WFS",
        "request": "GetFeature",
        "version": "2.0.0",
        "typeNames": "nwbwegen:wegvakken",
        "outputFormat": "application/json",
        "bbox": bbox,
    }

    road_features_wgs: List[Dict] = []

    try:
        r = requests.get(url, params=params, timeout=8)
        if r.status_code != 200:
            return road_features_wgs, 0

        data = r.json()
        for f in data.get("features", []):
            geom_rd = shape(f["geometry"])
            geom_wgs = transform(to_wgs, geom_rd)
            road_features_wgs.append(
                {
                    "type": "Feature",
                    "geometry": mapping(geom_wgs),
                    "properties": f.get("properties", {}),
                }
            )

    except Exception:
        return road_features_wgs, 0

    return road_features_wgs, len(road_features_wgs)


def get_hospitals(grb_polygon_wgs: Polygon):
    """
    Fetch hospitals / ER facilities around the GRB using OpenStreetMap
    (Overpass API) and compute simple overlap / distance metrics.

    This uses amenity=hospital as a proxy; in NL OSM coverage is good.
    """
    # Bbox in WGS84 with context buffer (approx ~2km)
    minx, miny, maxx, maxy = grb_polygon_wgs.bounds
    buffer_deg = 0.02  # rough buffer in degrees
    south = miny - buffer_deg
    west = minx - buffer_deg
    north = maxy + buffer_deg
    east = maxx + buffer_deg

    overpass_url = "https://overpass-api.de/api/interpreter"
    query = f"""
    [out:json][timeout:10];
    (
      node["amenity"="hospital"]({south},{west},{north},{east});
      way["amenity"="hospital"]({south},{west},{north},{east});
      relation["amenity"="hospital"]({south},{west},{north},{east});
    );
    out center;
    """

    hospital_features: List[Dict] = []
    intersecting_count = 0
    nearest_dist_m = None
    nearest_name = None

    # Project to RD for distance in meters
    wgs_crs = "EPSG:4326"
    rd_crs = "EPSG:28992"
    to_rd = pyproj.Transformer.from_crs(wgs_crs, rd_crs, always_xy=True).transform

    grb_rd = transform(to_rd, grb_polygon_wgs)

    try:
        r = requests.post(overpass_url, data={"data": query}, timeout=12)
        if r.status_code != 200:
            return hospital_features, intersecting_count, nearest_name, nearest_dist_m

        data = r.json()
        for el in data.get("elements", []):
            tags = el.get("tags", {})
            name = tags.get("name", "Unnamed hospital")

            if el.get("type") == "node":
                lon = el.get("lon")
                lat = el.get("lat")
            else:
                center = el.get("center", {})
                lon = center.get("lon")
                lat = center.get("lat")

            if lon is None or lat is None:
                continue

            pt_wgs = Point(lon, lat)
            pt_rd = transform(to_rd, pt_wgs)

            # Distance to GRB in meters
            dist_m = grb_rd.distance(pt_rd)

            # Check if inside GRB
            inside = grb_polygon_wgs.contains(pt_wgs)
            if inside:
                intersecting_count += 1

            if nearest_dist_m is None or dist_m < nearest_dist_m:
                nearest_dist_m = dist_m
                nearest_name = name

            hospital_features.append(
                {
                    "type": "Feature",
                    "geometry": mapping(pt_wgs),
                    "properties": {
                        "name": name,
                        "inside_grb": inside,
                        "distance_m": dist_m,
                    },
                }
            )

    except Exception:
        return hospital_features, intersecting_count, nearest_name, nearest_dist_m

    return hospital_features, intersecting_count, nearest_name, nearest_dist_m


def get_airports(grb_polygon_wgs: Polygon):
    """
    Fetch airports and airfields around the GRB using OpenStreetMap Overpass API.
    Computes distance metrics and identifies airports within or near the GRB.
    """
    minx, miny, maxx, maxy = grb_polygon_wgs.bounds
    buffer_deg = 0.05  # ~5km buffer
    south = miny - buffer_deg
    west = minx - buffer_deg
    north = maxy + buffer_deg
    east = maxx + buffer_deg

    overpass_url = "https://overpass-api.de/api/interpreter"
    query = f"""
    [out:json][timeout:15];
    (
      node["aeroway"~"^(aerodrome|airport|airfield|heliport)$"]({south},{west},{north},{east});
      way["aeroway"~"^(aerodrome|airport|airfield|heliport)$"]({south},{west},{north},{east});
      relation["aeroway"~"^(aerodrome|airport|airfield|heliport)$"]({south},{west},{north},{east});
    );
    out center;
    """

    airport_features: List[Dict] = []
    intersecting_count = 0
    nearest_dist_m = None
    nearest_name = None

    wgs_crs = "EPSG:4326"
    rd_crs = "EPSG:28992"
    to_rd = pyproj.Transformer.from_crs(wgs_crs, rd_crs, always_xy=True).transform
    grb_rd = transform(to_rd, grb_polygon_wgs)

    try:
        r = requests.post(overpass_url, data={"data": query}, timeout=15)
        if r.status_code != 200:
            return airport_features, intersecting_count, nearest_name, nearest_dist_m, {}

        data = r.json()
        source_metadata = {
            "name": "OpenStreetMap Aeroway",
            "provider": "OSM Overpass API",
            "vintage": "Live",
            "coverage": "Global",
        }

        for el in data.get("elements", []):
            tags = el.get("tags", {})
            name = tags.get("name", tags.get("ref", "Unnamed airport"))
            aeroway_type = tags.get("aeroway", "airport")

            if el.get("type") == "node":
                lon = el.get("lon")
                lat = el.get("lat")
            else:
                center = el.get("center", {})
                lon = center.get("lon")
                lat = center.get("lat")

            if lon is None or lat is None:
                continue

            pt_wgs = Point(lon, lat)
            pt_rd = transform(to_rd, pt_wgs)
            dist_m = grb_rd.distance(pt_rd)
            inside = grb_polygon_wgs.contains(pt_wgs)

            if inside:
                intersecting_count += 1

            if nearest_dist_m is None or dist_m < nearest_dist_m:
                nearest_dist_m = dist_m
                nearest_name = name

            airport_features.append(
                {
                    "type": "Feature",
                    "geometry": mapping(pt_wgs),
                    "properties": {
                        "name": name,
                        "aeroway_type": aeroway_type,
                        "inside_grb": inside,
                        "distance_m": dist_m,
                    },
                }
            )

    except Exception:
        return airport_features, intersecting_count, nearest_name, nearest_dist_m, {}

    return airport_features, intersecting_count, nearest_name, nearest_dist_m, source_metadata


def get_protected_areas(grb_polygon_wgs: Polygon):
    """
    Fetch protected areas (Natura 2000, national parks, etc.) from PDOK.
    Uses PDOK N2000 service for nature reserves and protected zones.
    """
    rd_crs = "EPSG:28992"
    wgs_crs = "EPSG:4326"
    to_rd = pyproj.Transformer.from_crs(wgs_crs, rd_crs, always_xy=True).transform
    to_wgs = pyproj.Transformer.from_crs(rd_crs, wgs_crs, always_xy=True).transform

    grb_rd = transform(to_rd, grb_polygon_wgs)
    minx, miny, maxx, maxy = grb_rd.bounds
    buffer_m = 2000.0  # 2km buffer

    bbox = f"{minx-buffer_m},{miny-buffer_m},{maxx+buffer_m},{maxy+buffer_m},EPSG:28992"
    
    # Try Natura 2000 areas first (most relevant for NL)
    url = "https://service.pdok.nl/kadaster/n2000/wfs/v1_0"
    params = {
        "service": "WFS",
        "request": "GetFeature",
        "version": "2.0.0",
        "typeNames": "n2000:n2000",
        "outputFormat": "application/json",
        "bbox": bbox,
    }

    protected_features_wgs: List[Dict] = []
    intersecting_count = 0
    total_overlap_km2 = 0.0
    source_metadata = {
        "name": "Natura 2000 Protected Areas",
        "provider": "PDOK Kadaster",
        "vintage": "2024",
        "coverage": "Netherlands",
    }

    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 200:
            data = r.json()
            for f in data.get("features", []):
                try:
                    geom_rd = shape(f["geometry"])
                    props = f.get("properties", {})
                    name = props.get("naam", props.get("name", "Protected Area"))

                    # Check intersection with GRB
                    if geom_rd.intersects(grb_rd):
                        intersecting_count += 1
                        try:
                            overlap_area_m2 = geom_rd.intersection(grb_rd).area
                            total_overlap_km2 += overlap_area_m2 / 1_000_000.0
                        except Exception:
                            pass

                    # Convert to WGS84 for display
                    geom_wgs = transform(to_wgs, geom_rd)
                    protected_features_wgs.append(
                        {
                            "type": "Feature",
                            "geometry": mapping(geom_wgs),
                            "properties": {
                                "name": name,
                                **props,
                            },
                        }
                    )
                except Exception:
                    continue

    except Exception:
        # Fail quietly - protected areas are informative but not critical
        pass

    return protected_features_wgs, intersecting_count, total_overlap_km2, source_metadata


def get_power_lines(grb_polygon_wgs: Polygon):
    """
    Fetch high-voltage power lines around the GRB using OpenStreetMap.
    These are important infrastructure constraints for drone operations.
    """
    minx, miny, maxx, maxy = grb_polygon_wgs.bounds
    buffer_deg = 0.03  # ~3km buffer
    south = miny - buffer_deg
    west = minx - buffer_deg
    north = maxy + buffer_deg
    east = maxx + buffer_deg

    overpass_url = "https://overpass-api.de/api/interpreter"
    query = f"""
    [out:json][timeout:12];
    (
      way["power"~"^(line|minor_line)$"]({south},{west},{north},{east});
      relation["power"~"^(line|minor_line)$"]({south},{west},{north},{east});
    );
    (._;>;);
    out geom;
    """

    powerline_features: List[Dict] = []
    intersecting_count = 0
    total_overlap_m = 0.0

    wgs_crs = "EPSG:4326"
    rd_crs = "EPSG:28992"
    to_rd = pyproj.Transformer.from_crs(wgs_crs, rd_crs, always_xy=True).transform
    grb_rd = transform(to_rd, grb_polygon_wgs)

    try:
        r = requests.post(overpass_url, data={"data": query}, timeout=12)
        if r.status_code != 200:
            return powerline_features, intersecting_count, total_overlap_m, {}

        data = r.json()
        source_metadata = {
            "name": "OpenStreetMap Power Lines",
            "provider": "OSM Overpass API",
            "vintage": "Live",
            "coverage": "Global",
        }

        for el in data.get("elements", []):
            if el.get("type") != "way":
                continue

            tags = el.get("tags", {})
            voltage = tags.get("voltage", "Unknown")
            power_type = tags.get("power", "line")

            # Reconstruct LineString from nodes
            nodes = el.get("geometry", [])
            if len(nodes) < 2:
                continue

            coords = [[n.get("lon"), n.get("lat")] for n in nodes if "lon" in n and "lat" in n]
            if len(coords) < 2:
                continue

            try:
                line_wgs = LineString(coords)
                line_rd = transform(to_rd, line_wgs)

                if line_rd.intersects(grb_rd):
                    intersecting_count += 1
                    try:
                        overlap_length = line_rd.intersection(grb_rd).length
                        total_overlap_m += overlap_length
                    except Exception:
                        pass

                powerline_features.append(
                    {
                        "type": "Feature",
                        "geometry": mapping(line_wgs),
                        "properties": {
                            "voltage": voltage,
                            "power_type": power_type,
                        },
                    }
                )
            except Exception:
                continue

    except Exception:
        return powerline_features, intersecting_count, total_overlap_m, {}

    return powerline_features, intersecting_count, total_overlap_m, source_metadata


def get_vital_infrastructure(grb_polygon_wgs: Polygon):
    """
    Fetch vital infrastructure (vitale infrastructuur) using PDOK WFS services AND OpenStreetMap.
    Organized by NCTV Category A (Critical - Red) and Category B (Severe - Blue).
    
    Category A: High Voltage Grid, Primary Dikes, Pumping Stations, Drinking Water, Power Plants, Water Works
    Category B: Fire Departments, Police Stations, Railways, Main Roads, Low Voltage Grid, Hospitals, 
                Government Facilities, Military Facilities, Telecommunications
    """
    rd_crs = "EPSG:28992"
    wgs_crs = "EPSG:4326"
    to_rd = pyproj.Transformer.from_crs(wgs_crs, rd_crs, always_xy=True).transform
    to_wgs = pyproj.Transformer.from_crs(rd_crs, wgs_crs, always_xy=True).transform

    grb_rd = transform(to_rd, grb_polygon_wgs)
    minx, miny, maxx, maxy = grb_rd.bounds
    buffer_m = 2000.0  # 2km buffer
    bbox_rd = f"{minx-buffer_m},{miny-buffer_m},{maxx+buffer_m},{maxy+buffer_m},EPSG:28992"

    # Also prepare WGS84 bbox for OpenStreetMap queries
    minx_wgs, miny_wgs, maxx_wgs, maxy_wgs = grb_polygon_wgs.bounds
    buffer_deg = 0.05  # ~5km buffer
    south = miny_wgs - buffer_deg
    west = minx_wgs - buffer_deg
    north = maxy_wgs + buffer_deg
    east = maxx_wgs + buffer_deg

    infrastructure_features: List[Dict] = []
    intersecting_count = 0
    nearest_dist_m = None
    nearest_name = None
    infrastructure_types = {}
    category_a_count = 0
    category_b_count = 0

    source_metadata = {
        "name": "PDOK + OSM Vitale Infrastructuur (NCTV Categories)",
        "provider": "PDOK (Liander, Waterschappen, BRT, ProRail, NWB) + OpenStreetMap",
        "vintage": "2024 (PDOK) / Live (OSM)",
        "coverage": "Netherlands",
        "categories": {
            "A": "Critical (National Stability) - Red",
            "B": "Severe (Economic/Social) - Blue"
        }
    }

    # === CATEGORY A: CRITICAL INFRASTRUCTURE (RED) ===
    
    # 1. High Voltage Grid (Liander)
    try:
        url = "https://service.pdok.nl/liander/elektriciteitsnetten/wfs/v1_0"
        params = {
            "service": "WFS",
            "request": "GetFeature",
            "version": "2.0.0",
            "typeNames": "elektriciteitsnetten:elektriciteitskabels",
            "outputFormat": "application/json",
            "bbox": bbox_rd,
        }
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 200:
            data = r.json()
            for f in data.get("features", []):
                try:
                    props = f.get("properties", {})
                    spanning = props.get("spanning", 0)
                    # Filter for high voltage (> 50kV or type HS)
                    if spanning and isinstance(spanning, (int, float)) and spanning > 50000:
                        geom_rd = shape(f["geometry"])
                        if geom_rd.intersects(grb_rd):
                            intersecting_count += 1
                            category_a_count += 1
                        
                        geom_wgs = transform(to_wgs, geom_rd)
                        infrastructure_types["High Voltage Grid"] = infrastructure_types.get("High Voltage Grid", 0) + 1
                        
                        infrastructure_features.append({
                            "type": "Feature",
                            "geometry": mapping(geom_wgs),
                            "properties": {
                                "name": f"High Voltage Line ({spanning/1000:.0f}kV)",
                                "infrastructure_type": "High Voltage Grid",
                                "category": "A",
                                "spanning": spanning,
                                "inside_grb": geom_rd.intersects(grb_rd),
                            },
                        })
                except Exception:
                    continue
    except Exception:
        pass

    # 2. Primary Dikes (Waterschappen - Legger Waterkeringen)
    try:
        url = "https://service.pdok.nl/hwh/leggerwaterkeringen/wfs/v1_0"
        params = {
            "service": "WFS",
            "request": "GetFeature",
            "version": "2.0.0",
            "typeNames": "leggerwaterkeringen:waterkering",
            "outputFormat": "application/json",
            "bbox": bbox_rd,
        }
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 200:
            data = r.json()
            for f in data.get("features", []):
                try:
                    props = f.get("properties", {})
                    categorie = props.get("categorie", "").lower()
                    # Filter for primary dikes
                    if "primair" in categorie:
                        geom_rd = shape(f["geometry"])
                        if geom_rd.intersects(grb_rd):
                            intersecting_count += 1
                            category_a_count += 1
                        
                        geom_wgs = transform(to_wgs, geom_rd)
                        name = props.get("naam", props.get("name", "Primary Dike"))
                        infrastructure_types["Primary Dike"] = infrastructure_types.get("Primary Dike", 0) + 1
                        
                        infrastructure_features.append({
                            "type": "Feature",
                            "geometry": mapping(geom_wgs),
                            "properties": {
                                "name": name,
                                "infrastructure_type": "Primary Dike",
                                "category": "A",
                                "categorie": categorie,
                                "inside_grb": geom_rd.intersects(grb_rd),
                            },
                        })
                except Exception:
                    continue
    except Exception:
        pass

    # 3. Pumping Stations (Waterschappen - Kunstwerken)
    try:
        url = "https://service.pdok.nl/hwh/kunstwerken/wfs/v1_0"
        params = {
            "service": "WFS",
            "request": "GetFeature",
            "version": "2.0.0",
            "typeNames": "kunstwerken:gemaal",
            "outputFormat": "application/json",
            "bbox": bbox_rd,
        }
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 200:
            data = r.json()
            for f in data.get("features", []):
                try:
                    geom_rd = shape(f["geometry"])
                    if geom_rd.intersects(grb_rd):
                        intersecting_count += 1
                        category_a_count += 1
                    
                    geom_wgs = transform(to_wgs, geom_rd)
                    props = f.get("properties", {})
                    name = props.get("naam", props.get("name", "Pumping Station"))
                    
                    # Calculate distance
                    if geom_rd.geom_type == "Point":
                        pt_rd = geom_rd
                    else:
                        pt_rd = geom_rd.centroid
                    dist_m = grb_rd.distance(pt_rd)
                    if nearest_dist_m is None or dist_m < nearest_dist_m:
                        nearest_dist_m = dist_m
                        nearest_name = f"{name} (Pumping Station)"
                    
                    infrastructure_types["Pumping Station"] = infrastructure_types.get("Pumping Station", 0) + 1
                    
                    infrastructure_features.append({
                        "type": "Feature",
                        "geometry": mapping(geom_wgs),
                        "properties": {
                            "name": name,
                            "infrastructure_type": "Pumping Station",
                            "category": "A",
                            "inside_grb": geom_rd.intersects(grb_rd),
                            "distance_m": dist_m,
                        },
                    })
                except Exception:
                    continue
    except Exception:
        pass

    # === CATEGORY B: SEVERE INFRASTRUCTURE (BLUE) ===
    
    # 4. Fire Departments (BRT TOP10NL)
    try:
        url = "https://service.pdok.nl/brt/top10nl/wfs/v1_0"
        params = {
            "service": "WFS",
            "request": "GetFeature",
            "version": "2.0.0",
            "typeNames": "top10nl:gebouw",
            "outputFormat": "application/json",
            "bbox": bbox_rd,
        }
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 200:
            data = r.json()
            for f in data.get("features", []):
                try:
                    props = f.get("properties", {})
                    type_gebouw = props.get("typeGebouw", "").lower()
                    if "brandweerkazerne" in type_gebouw or "fire" in type_gebouw:
                        geom_rd = shape(f["geometry"])
                        if geom_rd.intersects(grb_rd):
                            intersecting_count += 1
                            category_b_count += 1
                        
                        geom_wgs = transform(to_wgs, geom_rd)
                        name = props.get("naam", "Fire Station")
                        
                        if geom_rd.geom_type == "Point":
                            pt_rd = geom_rd
                        else:
                            pt_rd = geom_rd.centroid
                        dist_m = grb_rd.distance(pt_rd)
                        if nearest_dist_m is None or dist_m < nearest_dist_m:
                            nearest_dist_m = dist_m
                            nearest_name = f"{name} (Fire Station)"
                        
                        infrastructure_types["Fire Station"] = infrastructure_types.get("Fire Station", 0) + 1
                        
                        infrastructure_features.append({
                            "type": "Feature",
                            "geometry": mapping(geom_wgs),
                            "properties": {
                                "name": name,
                                "infrastructure_type": "Fire Station",
                                "category": "B",
                                "inside_grb": geom_rd.intersects(grb_rd),
                                "distance_m": dist_m,
                            },
                        })
                except Exception:
                    continue
    except Exception:
        pass

    # 5. Police Stations (BRT TOP10NL)
    try:
        url = "https://service.pdok.nl/brt/top10nl/wfs/v1_0"
        params = {
            "service": "WFS",
            "request": "GetFeature",
            "version": "2.0.0",
            "typeNames": "top10nl:gebouw",
            "outputFormat": "application/json",
            "bbox": bbox_rd,
        }
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 200:
            data = r.json()
            for f in data.get("features", []):
                try:
                    props = f.get("properties", {})
                    type_gebouw = props.get("typeGebouw", "").lower()
                    if "politiebureau" in type_gebouw or "police" in type_gebouw:
                        geom_rd = shape(f["geometry"])
                        if geom_rd.intersects(grb_rd):
                            intersecting_count += 1
                            category_b_count += 1
                        
                        geom_wgs = transform(to_wgs, geom_rd)
                        name = props.get("naam", "Police Station")
                        
                        if geom_rd.geom_type == "Point":
                            pt_rd = geom_rd
                        else:
                            pt_rd = geom_rd.centroid
                        dist_m = grb_rd.distance(pt_rd)
                        if nearest_dist_m is None or dist_m < nearest_dist_m:
                            nearest_dist_m = dist_m
                            nearest_name = f"{name} (Police Station)"
                        
                        infrastructure_types["Police Station"] = infrastructure_types.get("Police Station", 0) + 1
                        
                        infrastructure_features.append({
                            "type": "Feature",
                            "geometry": mapping(geom_wgs),
                            "properties": {
                                "name": name,
                                "infrastructure_type": "Police Station",
                                "category": "B",
                                "inside_grb": geom_rd.intersects(grb_rd),
                                "distance_m": dist_m,
                            },
                        })
                except Exception:
                    continue
    except Exception:
        pass

    # 6. Railways (ProRail)
    try:
        url = "https://service.pdok.nl/prorail/spoorwegen/wfs/v1_0"
        params = {
            "service": "WFS",
            "request": "GetFeature",
            "version": "2.0.0",
            "typeNames": "spoorwegen:spoor",
            "outputFormat": "application/json",
            "bbox": bbox_rd,
        }
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 200:
            data = r.json()
            for f in data.get("features", []):
                try:
                    props = f.get("properties", {})
                    spoortype = props.get("spoortype", "").lower()
                    # Filter for main tracks (exclude service tracks)
                    if "spoor" in spoortype and "dienst" not in spoortype:
                        geom_rd = shape(f["geometry"])
                        if geom_rd.intersects(grb_rd):
                            intersecting_count += 1
                            category_b_count += 1
                        
                        geom_wgs = transform(to_wgs, geom_rd)
                        infrastructure_types["Railway"] = infrastructure_types.get("Railway", 0) + 1
                        
                        infrastructure_features.append({
                            "type": "Feature",
                            "geometry": mapping(geom_wgs),
                            "properties": {
                                "name": "Railway Track",
                                "infrastructure_type": "Railway",
                                "category": "B",
                                "inside_grb": geom_rd.intersects(grb_rd),
                            },
                        })
                except Exception:
                    continue
    except Exception:
        pass

    # 7. Main Roads (NWB - State managed)
    try:
        url = "https://service.pdok.nl/rws/nwbwegen/wfs/v1_0"
        params = {
            "service": "WFS",
            "request": "GetFeature",
            "version": "2.0.0",
            "typeNames": "nwbwegen:wegvakken",
            "outputFormat": "application/json",
            "bbox": bbox_rd,
        }
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 200:
            data = r.json()
            for f in data.get("features", []):
                try:
                    props = f.get("properties", {})
                    wegbeheerder = props.get("wegbeheerder", "").lower()
                    # Filter for state-managed roads
                    if "rijk" in wegbeheerder or "state" in wegbeheerder:
                        geom_rd = shape(f["geometry"])
                        if geom_rd.intersects(grb_rd):
                            intersecting_count += 1
                            category_b_count += 1
                        
                        geom_wgs = transform(to_wgs, geom_rd)
                        infrastructure_types["Main Road (State)"] = infrastructure_types.get("Main Road (State)", 0) + 1
                        
                        infrastructure_features.append({
                            "type": "Feature",
                            "geometry": mapping(geom_wgs),
                            "properties": {
                                "name": "State Highway",
                                "infrastructure_type": "Main Road (State)",
                                "category": "B",
                                "inside_grb": geom_rd.intersects(grb_rd),
                            },
                        })
                except Exception:
                    continue
    except Exception:
        pass

    # 8. Low Voltage Grid (Liander)
    try:
        url = "https://service.pdok.nl/liander/elektriciteitsnetten/wfs/v1_0"
        params = {
            "service": "WFS",
            "request": "GetFeature",
            "version": "2.0.0",
            "typeNames": "elektriciteitsnetten:laagspanningskabels",
            "outputFormat": "application/json",
            "bbox": bbox_rd,
        }
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 200:
            data = r.json()
            for f in data.get("features", []):
                try:
                    props = f.get("properties", {})
                    kabeltype = props.get("type", "").upper()
                    # Filter for low voltage (LS)
                    if "LS" in kabeltype or "LAAG" in kabeltype:
                        geom_rd = shape(f["geometry"])
                        if geom_rd.intersects(grb_rd):
                            intersecting_count += 1
                            category_b_count += 1
                        
                        geom_wgs = transform(to_wgs, geom_rd)
                        infrastructure_types["Low Voltage Grid"] = infrastructure_types.get("Low Voltage Grid", 0) + 1
                        
                        infrastructure_features.append({
                            "type": "Feature",
                            "geometry": mapping(geom_wgs),
                            "properties": {
                                "name": "Low Voltage Cable",
                                "infrastructure_type": "Low Voltage Grid",
                                "category": "B",
                                "inside_grb": geom_rd.intersects(grb_rd),
                            },
                        })
                except Exception:
                    continue
    except Exception:
        pass

    # === SUPPLEMENT WITH OPENSTREETMAP DATA ===
    # This adds additional infrastructure types that may not be in PDOK or provides backup coverage
    
    overpass_url = "https://overpass-api.de/api/interpreter"
    
    # Category A: Power plants, water works, wastewater plants (OSM)
    try:
        query_a = f"""
        [out:json][timeout:20];
        (
          // Power plants and substations (Category A)
          node["power"~"^(plant|substation)$"]({south},{west},{north},{east});
          way["power"~"^(plant|substation)$"]({south},{west},{north},{east});
          relation["power"~"^(plant|substation)$"]({south},{west},{north},{east});
          
          // Water treatment and supply (Category A)
          node["man_made"~"^(water_works|wastewater_plant|pumping_station)$"]({south},{west},{north},{east});
          way["man_made"~"^(water_works|wastewater_plant|pumping_station)$"]({south},{west},{north},{east});
          relation["man_made"~"^(water_works|wastewater_plant|pumping_station)$"]({south},{west},{north},{east});
        );
        out center;
        """
        r = requests.post(overpass_url, data={"data": query_a}, timeout=20)
        if r.status_code == 200:
            data = r.json()
            for el in data.get("elements", []):
                try:
                    tags = el.get("tags", {})
                    
                    # Determine infrastructure type
                    infra_type = None
                    category = "A"
                    
                    if "power" in tags:
                        if tags.get("power") == "plant":
                            infra_type = "Power Plant"
                        elif tags.get("power") == "substation":
                            infra_type = "Substation"
                    elif "man_made" in tags:
                        man_made = tags.get("man_made", "")
                        if "water_works" in man_made:
                            infra_type = "Water Works"
                        elif "wastewater_plant" in man_made:
                            infra_type = "Wastewater Plant"
                        elif "pumping_station" in man_made:
                            infra_type = "Pumping Station (OSM)"
                    
                    if not infra_type:
                        continue
                    
                    name = tags.get("name", tags.get("operator", f"Unnamed {infra_type}"))
                    
                    if el.get("type") == "node":
                        lon = el.get("lon")
                        lat = el.get("lat")
                    else:
                        center = el.get("center", {})
                        lon = center.get("lon")
                        lat = center.get("lat")
                    
                    if lon is None or lat is None:
                        continue
                    
                    pt_wgs = Point(lon, lat)
                    pt_rd = transform(to_rd, pt_wgs)
                    dist_m = grb_rd.distance(pt_rd)
                    inside = grb_polygon_wgs.contains(pt_wgs)
                    
                    if inside:
                        intersecting_count += 1
                        category_a_count += 1
                    
                    if nearest_dist_m is None or dist_m < nearest_dist_m:
                        nearest_dist_m = dist_m
                        nearest_name = f"{name} ({infra_type})"
                    
                    infrastructure_types[infra_type] = infrastructure_types.get(infra_type, 0) + 1
                    
                    infrastructure_features.append({
                        "type": "Feature",
                        "geometry": mapping(pt_wgs),
                        "properties": {
                            "name": name,
                            "infrastructure_type": infra_type,
                            "category": category,
                            "inside_grb": inside,
                            "distance_m": dist_m,
                            "source": "OSM",
                        },
                    })
                except Exception:
                    continue
    except Exception:
        pass
    
    # Category B: Government, military, emergency services, telecommunications (OSM)
    try:
        query_b = f"""
        [out:json][timeout:20];
        (
          // Government and military facilities (Category B)
          node["office"="government"]({south},{west},{north},{east});
          way["office"="government"]({south},{west},{north},{east});
          relation["office"="government"]({south},{west},{north},{east});
          node["military"="*"]({south},{west},{north},{east});
          way["military"="*"]({south},{west},{north},{east});
          relation["military"="*"]({south},{west},{north},{east});
          
          // Emergency services (Category B) - supplement PDOK
          node["amenity"~"^(fire_station|police)$"]({south},{west},{north},{east});
          way["amenity"~"^(fire_station|police)$"]({south},{west},{north},{east});
          relation["amenity"~"^(fire_station|police)$"]({south},{west},{north},{east});
          
          // Telecommunications infrastructure (Category B)
          node["tower:type"="communication"]({south},{west},{north},{east});
          way["tower:type"="communication"]({south},{west},{north},{east});
          relation["tower:type"="communication"]({south},{west},{north},{east});
        );
        out center;
        """
        r = requests.post(overpass_url, data={"data": query_b}, timeout=20)
        if r.status_code == 200:
            data = r.json()
            for el in data.get("elements", []):
                try:
                    tags = el.get("tags", {})
                    
                    infra_type = None
                    category = "B"
                    
                    if tags.get("office") == "government":
                        infra_type = "Government Facility"
                    elif "military" in tags:
                        infra_type = "Military Facility"
                    elif tags.get("amenity") == "fire_station":
                        infra_type = "Fire Station (OSM)"
                    elif tags.get("amenity") == "police":
                        infra_type = "Police Station (OSM)"
                    elif tags.get("tower:type") == "communication":
                        infra_type = "Telecommunications Tower"
                    
                    if not infra_type:
                        continue
                    
                    name = tags.get("name", tags.get("operator", f"Unnamed {infra_type}"))
                    
                    if el.get("type") == "node":
                        lon = el.get("lon")
                        lat = el.get("lat")
                    else:
                        center = el.get("center", {})
                        lon = center.get("lon")
                        lat = center.get("lat")
                    
                    if lon is None or lat is None:
                        continue
                    
                    pt_wgs = Point(lon, lat)
                    pt_rd = transform(to_rd, pt_wgs)
                    dist_m = grb_rd.distance(pt_rd)
                    inside = grb_polygon_wgs.contains(pt_wgs)
                    
                    if inside:
                        intersecting_count += 1
                        category_b_count += 1
                    
                    if nearest_dist_m is None or dist_m < nearest_dist_m:
                        nearest_dist_m = dist_m
                        nearest_name = f"{name} ({infra_type})"
                    
                    infrastructure_types[infra_type] = infrastructure_types.get(infra_type, 0) + 1
                    
                    infrastructure_features.append({
                        "type": "Feature",
                        "geometry": mapping(pt_wgs),
                        "properties": {
                            "name": name,
                            "infrastructure_type": infra_type,
                            "category": category,
                            "inside_grb": inside,
                            "distance_m": dist_m,
                            "source": "OSM",
                        },
                    })
                except Exception:
                    continue
    except Exception:
        pass

    # Add category counts to metadata
    source_metadata["category_counts"] = {
        "A": category_a_count,
        "B": category_b_count,
    }

    return infrastructure_features, intersecting_count, nearest_name, nearest_dist_m, infrastructure_types, source_metadata


# --- 4.1 CACHED WRAPPERS FOR CONSTRAINT LAYERS ---
@st.cache_data(ttl=3600)
def get_major_roads_cached(grb_wkt: str):
    grb = wkt.loads(grb_wkt)
    return get_major_roads(grb)


@st.cache_data(ttl=3600)
def get_local_roads_cached(grb_wkt: str):
    grb = wkt.loads(grb_wkt)
    return get_local_roads(grb)


@st.cache_data(ttl=3600)
def get_hospitals_cached(grb_wkt: str):
    grb = wkt.loads(grb_wkt)
    return get_hospitals(grb)


@st.cache_data(ttl=3600)
def get_airports_cached(grb_wkt: str):
    grb = wkt.loads(grb_wkt)
    return get_airports(grb)


@st.cache_data(ttl=3600)
def get_protected_areas_cached(grb_wkt: str):
    grb = wkt.loads(grb_wkt)
    return get_protected_areas(grb)


@st.cache_data(ttl=3600)
def get_power_lines_cached(grb_wkt: str):
    grb = wkt.loads(grb_wkt)
    return get_power_lines(grb)


@st.cache_data(ttl=3600)
def get_vital_infrastructure_cached(grb_wkt: str):
    grb = wkt.loads(grb_wkt)
    return get_vital_infrastructure(grb)


def _get_population_density_cbs(polygon: Polygon):
    minx, miny, maxx, maxy = polygon.bounds
    buffer = 0.005 
    url = "https://service.pdok.nl/cbs/vierkantstatistieken100m/2023/wfs/v1_0"
    
    strategies = [{"name": "Strategy A (Lat, Lon)", "bbox": f"{miny-buffer},{minx-buffer},{maxy+buffer},{maxx+buffer},EPSG:4326"}]
    debug_info = {"url": url, "attempts": [], "cell_analysis": []}
    
    for strategy in strategies:
        params = {
            "request": "GetFeature", "service": "WFS", "version": "2.0.0",
            "typeNames": "vierkantstatistieken100m:vierkant_100m", 
            "outputFormat": "application/json", "bbox": strategy["bbox"]
        }
        try:
            r = requests.get(url, params=params, timeout=8)
            if r.status_code == 200:
                data = r.json()
                features = data.get('features', [])
                debug_info["total_features_in_box"] = len(features)
                
                strict_max = 0; adjacent_max = 0; strict_count = 0
                strict_hotspot = None; 
                grid_features = [] 
                cell_data_list = []
                
                for f in features:
                    props = f['properties']
                    pop = props.get('aantalInwoners', props.get('inwoners', -9999))
                    if pop < 0: pop = 0
                    
                    # Sanitize
                    raw_geom = f['geometry']
                    clean_geom = sanitize_geometry(raw_geom)
                    f['geometry'] = clean_geom 
                    
                    # Track Adjacent Max (For Context only)
                    if pop > adjacent_max: adjacent_max = pop
                    
                    # STRICT INTERSECTION CHECK
                    cell_shape = shape(clean_geom)
                    overlap_pct = 0.0
                    is_intersecting = False
                    
                    if polygon.intersects(cell_shape):
                        is_intersecting = True
                        try:
                            overlap_area = polygon.intersection(cell_shape).area
                            total_area = cell_shape.area
                            if total_area > 0: overlap_pct = (overlap_area / total_area) * 100
                        except: pass

                        if pop >= strict_max: 
                            strict_max = pop
                            strict_hotspot = clean_geom
                        strict_count += 1
                        
                        if pop > 0:
                            cell_data_list.append({
                                "Population": pop, "Overlap %": round(overlap_pct, 2),
                                "Lat": round(cell_shape.centroid.y, 5), "Lon": round(cell_shape.centroid.x, 5)
                            })
                    
                    # Visual Properties
                    color = "#008080" # Teal
                    if pop > 200: color = "#000000" 
                    elif pop >= 100: color = "#800080" 
                    
                    f['properties']['viz_color'] = color
                    f['properties']['viz_pop'] = pop
                    f['properties']['overlap_pct'] = round(overlap_pct, 1)
                    f['properties']['is_hotspot'] = False
                    
                    if pop > 0: grid_features.append(f)
                
                # --- THE FIX: NO FORCED FALLBACK ---
                # If strict_count is 0, final_pop is 0.
                final_pop = strict_max
                final_geom = strict_hotspot
                method = "Strict Intersection" if strict_count > 0 else "None (Safe)"
                
                # Mark hotspot only if it exists inside buffer
                if final_geom:
                    for g in grid_features:
                        if str(g['geometry']) == str(final_geom):
                            g['properties']['is_hotspot'] = True
                            g['properties']['viz_color'] = "#0000FF" # Blue
                
                debug_info["cell_analysis"] = sorted(cell_data_list, key=lambda x: x['Population'], reverse=True)
                debug_info["adjacent_max"] = adjacent_max # Store this for context display
                debug_info["source"] = {
                    "name": CbsPopulationSource.name,
                    "resolution_m": CbsPopulationSource.resolution_m,
                    "vintage": CbsPopulationSource.vintage,
                    "provider": "CBS via PDOK WFS",
                }
                
                return final_pop * 100, len(features), method, final_geom, grid_features, debug_info
        except Exception as e:
            debug_info["attempts"].append(str(e))

    return 0, 0, "No Data", None, [], debug_info

# --- 3. PARSERS ---
def parse_flight_plan(file_content):
    try:
        root = ET.fromstring(file_content)
        coords = []
        found_kml = False
        for elem in root.iter():
            if 'coordinates' in elem.tag:
                text = elem.text.strip()
                for pair in text.split():
                    parts = pair.split(',')
                    if len(parts) >= 2:
                        lon = float(parts[0]); lat = float(parts[1])
                        coords.append([lat, lon])
                found_kml = True
        if found_kml and coords: return coords
        points = []
        for elem in root.iter():
            lat = None; lon = None
            for child in elem:
                if 'lat' in child.tag.lower(): lat = float(child.text)
                if 'lon' in child.tag.lower(): lon = float(child.text)
            if lat and lon: points.append([lat, lon])
        return points
    except: return []

# --- 4. SESSION PERSISTENCE ---
def save_mission_config(specs: MissionSpec, population_source: str, layer_settings: Dict, mission_name: str = "default", mitigation_set: Optional[MitigationSet] = None):
    """Save mission configuration to session state for persistence."""
    config = {
        "mission_name": mission_name,
        "specs": asdict(specs),
        "population_source": population_source,
        "layer_settings": layer_settings,
        "timestamp": pd.Timestamp.now().isoformat(),
        "mitigation_set": mitigation_set.to_dict() if mitigation_set else None,
    }
    if "saved_configs" not in st.session_state:
        st.session_state["saved_configs"] = {}
    st.session_state["saved_configs"][mission_name] = config
    return config

def load_mission_config(mission_name: str) -> Optional[Dict]:
    """Load mission configuration from session state."""
    if "saved_configs" not in st.session_state:
        return None
    config = st.session_state["saved_configs"].get(mission_name)
    if config and "mitigation_set" in config and config["mitigation_set"]:
        # Restore mitigation set to session state
        try:
            mitigation_set = MitigationSet.from_dict(config["mitigation_set"])
            st.session_state["mitigation_set"] = mitigation_set
        except Exception as e:
            st.warning(f"Could not restore mitigation set: {e}")
    return config

def get_saved_config_names() -> List[str]:
    """Get list of saved configuration names."""
    if "saved_configs" not in st.session_state:
        return []
    return list(st.session_state["saved_configs"].keys())

# --- 5. EXPORT KML ---
def create_kml_download(geometries, heights, hotspot_geom=None):
    kml = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2"><Document><name>SORA Buffers</name>
<Style id="grb"><LineStyle><color>ff0000ff</color><width>2</width></LineStyle><PolyStyle><color>400000ff</color></PolyStyle></Style>
<Style id="cv"><LineStyle><color>ff00ffff</color><width>2</width></LineStyle><PolyStyle><color>4000ffff</color></PolyStyle></Style>
<Style id="fg"><LineStyle><color>ff00ff00</color><width>2</width></LineStyle><PolyStyle><color>4000ff00</color></PolyStyle></Style>
<Style id="hotspot"><LineStyle><color>ffff0000</color><width>4</width></LineStyle><PolyStyle><color>00000000</color></PolyStyle></Style>"""
    
    def geom_to_kml(geom, height, name, style):
        placemark = f'<Placemark><name>{name}</name><styleUrl>#{style}</styleUrl>'
        def c_str(ex): return " ".join([f"{x},{y},{height}" for x, y in ex.coords])
        if geom.geom_type == 'Polygon':
            placemark += f'<Polygon><tessellate>1</tessellate><outerBoundaryIs><LinearRing><coordinates>{c_str(geom.exterior)}</coordinates></LinearRing></outerBoundaryIs></Polygon>'
        elif geom.geom_type == 'MultiPolygon':
            placemark += '<MultiGeometry>'
            for poly in geom.geoms:
                placemark += f'<Polygon><tessellate>1</tessellate><outerBoundaryIs><LinearRing><coordinates>{c_str(poly.exterior)}</coordinates></LinearRing></outerBoundaryIs></Polygon>'
            placemark += '</MultiGeometry>'
        placemark += '</Placemark>'
        return placemark

    kml += geom_to_kml(geometries['GRB'], 0, "GRB", "grb")
    kml += geom_to_kml(geometries['CV'], heights['h_cv'], "CV", "cv")
    kml += geom_to_kml(geometries['FG'], heights['h_fg'], "FG", "fg")
    if hotspot_geom:
        hs_poly = shape(hotspot_geom)
        kml += geom_to_kml(hs_poly, 0, "Hotspot", "hotspot")
    kml += "</Document></kml>"
    return kml

# --- 5.1 MITIGATION UI COMPONENTS ---
def render_mitigation_section(
    mitigation_type: MitigationType,
    available_levels: List[RobustnessLevel],
    current_config: Optional[MitigationConfig],
    key_prefix: str,
) -> Optional[MitigationConfig]:
    """
    Render UI for a single mitigation type.
    
    Returns:
        MitigationConfig if enabled, None otherwise
    """
    st.subheader(mitigation_type.value)
    
    # Enable checkbox
    enabled = st.checkbox(
        f"Enable {mitigation_type.value}",
        value=current_config is not None,
        key=f"{key_prefix}_enabled"
    )
    
    if not enabled:
        return None
    
    # Level selector
    level_options = [level.value for level in available_levels]
    current_level_index = 0
    if current_config:
        try:
            current_level_index = level_options.index(current_config.level.value)
        except ValueError:
            current_level_index = 0
    
    selected_level_str = st.selectbox(
        "Robustness Level",
        level_options,
        index=current_level_index,
        key=f"{key_prefix}_level"
    )
    selected_level = RobustnessLevel(selected_level_str)
    
    # Get corrective factor
    corrective_factor = get_corrective_factor(mitigation_type, selected_level)
    
    # Time constraint (only for M1(A))
    time_constraint = False
    if mitigation_type == MitigationType.M1A_SHELTERING:
        time_constraint = st.checkbox(
            "Apply time constraint (moves from Low to Medium if Low selected)",
            value=current_config.time_constraint_applied if current_config else False,
            key=f"{key_prefix}_time_constraint"
        )
        # Auto-upgrade to Medium if time constraint applied and Low selected
        if time_constraint and selected_level == RobustnessLevel.LOW:
            selected_level = RobustnessLevel.MEDIUM
            corrective_factor = get_corrective_factor(mitigation_type, selected_level)
            st.info("⚠️ Time constraint applied: Level upgraded to Medium (corrective factor: -2)")
    
    # Display requirements
    if selected_level in MITIGATION_REQUIREMENTS.get(mitigation_type, {}):
        reqs = MITIGATION_REQUIREMENTS[mitigation_type][selected_level]
        
        with st.expander("📋 Integrity & Assurance Requirements", expanded=False):
            st.markdown("**Integrity Requirements:**")
            for req in reqs["integrity"]:
                st.markdown(f"• {req}")
            
            st.markdown("**Assurance Requirements:**")
            for req in reqs["assurance"]:
                st.markdown(f"• {req}")
            
            if reqs.get("operator_declaration_required"):
                st.warning("⚠️ Operator declaration required for this level")
    
    # Description fields
    description = st.text_area(
        "Description: How you meet these requirements",
        value=current_config.description if current_config else "",
        key=f"{key_prefix}_description",
        placeholder="Describe how your mitigation meets the integrity and assurance requirements..."
    )
    
    integrity_desc = st.text_area(
        "Integrity Evidence Description",
        value=current_config.integrity_description if current_config else "",
        key=f"{key_prefix}_integrity_desc",
        placeholder="Describe your evidence for integrity requirements..."
    )
    
    assurance_desc = st.text_area(
        "Assurance Evidence Description",
        value=current_config.assurance_description if current_config else "",
        key=f"{key_prefix}_assurance_desc",
        placeholder="Describe your evidence for assurance requirements..."
    )
    
    # Documentation references (compliance matrix)
    st.markdown("**📄 Compliance Matrix References:**")
    ref_doc = st.text_input(
        "Reference Document Name",
        value=current_config.reference_document_name if current_config else "",
        key=f"{key_prefix}_ref_doc",
        placeholder="e.g., Operations Manual, Safety Case Document"
    )
    
    ref_page = st.text_input(
        "Chapter or Page Number",
        value=current_config.reference_chapter_page if current_config else "",
        key=f"{key_prefix}_ref_page",
        placeholder="e.g., Chapter 5.2 or Page 42"
    )
    
    # Display corrective factor
    st.caption(f"Corrective Factor: {corrective_factor}")
    
    return MitigationConfig(
        mitigation_type=mitigation_type,
        level=selected_level,
        corrective_factor=corrective_factor,
        time_constraint_applied=time_constraint,
        description=description,
        integrity_description=integrity_desc,
        assurance_description=assurance_desc,
        reference_document_name=ref_doc,
        reference_chapter_page=ref_page,
    )

# --- 6. APP LAYOUT ---
st.set_page_config(page_title="SORA Tool", layout="wide")
st.title("🚁 SORA Automated Flight Planner")

if 'map_center' not in st.session_state: st.session_state['map_center'] = [52.0, 5.0]
if 'map_zoom' not in st.session_state: st.session_state['map_zoom'] = 10
if 'drawn_coords' not in st.session_state: st.session_state['drawn_coords'] = []

with st.sidebar:
    with st.form("config"):
        st.header("1. Configuration")
        v0 = st.number_input("Max Speed (v0)", 25.0)
        dimension = st.number_input("Dimension [m]", 3.0)
        geo_type = st.radio("Geometry", ["Corridor (Line)", "Polygon (Area)"])
        
        with st.expander("Advanced"):
            t_rz = st.number_input("Reaction [s]", 1.0)
            errors = st.number_input("Errors [m]", 9.0)
            pitch = st.number_input("Pitch [deg]", 45.0)
            t_open = st.number_input("Chute Time [s]", 2.0)
            v_z = st.number_input("Descent [m/s]", 4.0)
            v_wind = st.number_input("Wind [m/s]", 3.0)
            
        st.header("2. Mitigations")
        
        # Initialize mitigation set in session state if not present
        if "mitigation_set" not in st.session_state:
            st.session_state["mitigation_set"] = MitigationSet()
        
        # Load existing mitigations from session state
        current_mitigations = st.session_state.get("mitigation_set", MitigationSet())
        
        # M1(A) - Sheltering
        m1a_config = render_mitigation_section(
            MitigationType.M1A_SHELTERING,
            [RobustnessLevel.LOW, RobustnessLevel.MEDIUM],
            current_mitigations.m1a,
            "m1a"
        )
        
        # M1(B) - Operational Restrictions
        m1b_config = render_mitigation_section(
            MitigationType.M1B_OPERATIONAL_RESTRICTIONS,
            [RobustnessLevel.MEDIUM, RobustnessLevel.HIGH],
            current_mitigations.m1b,
            "m1b"
        )
        
        # M1(C) - Ground Observation
        m1c_config = render_mitigation_section(
            MitigationType.M1C_GROUND_OBSERVATION,
            [RobustnessLevel.LOW],
            current_mitigations.m1c,
            "m1c"
        )
        
        # M2 - Impact Dynamics
        m2_config = render_mitigation_section(
            MitigationType.M2_IMPACT_DYNAMICS,
            [RobustnessLevel.MEDIUM, RobustnessLevel.HIGH],
            current_mitigations.m2,
            "m2"
        )
        
        applied = st.form_submit_button("Apply")
        
        # Build mitigation set
        mitigation_set = MitigationSet(
            m1a=m1a_config,
            m1b=m1b_config,
            m1c=m1c_config,
            m2=m2_config,
        )
        
        # Validate
        is_valid, error_msg = mitigation_set.validate()
        if not is_valid:
            st.error(f"❌ {error_msg}")
        
        # Store in session state
        st.session_state["mitigation_set"] = mitigation_set
        
        # Calculate total reduction
        total_mitigation = mitigation_set.calculate_total_reduction()
    
    # Build a structured MissionSpec so all inputs are traceable and serialisable
    specs = MissionSpec(
        v0=v0,
        dimension=dimension,
        t_rz=t_rz,
        s_gps=errors / 3,
        s_pos=errors / 3,
        s_map=errors / 3,
        h_gps=4.0,
        h_fg=120.0,
        v_wind=v_wind,
        v_climb=2.0,
        maneuver_type="Multirotor (Stop)",
        pitch_angle=pitch,
        bank_angle=30.0,
        emergency_type="Parachute (Ballistic)",
        t_open=t_open,
        v_z=v_z,
        glide_ratio=10.0,
        geometry_type=geo_type,
    )

    st.header("3. Population Data")
    # For now we only expose CBS, but this is wired for future multi-source support
    available_sources = {
        "CBS 100m (2023)": CbsPopulationSource,
        "Mock: CBS x1.2": MockScaledPopulationSource,
        "GHSL GHS-POP (stub)": GhslPopulationSource,
    }
    selected_source_name = st.selectbox("Population dataset", list(available_sources.keys()))
    st.session_state["population_source_name"] = selected_source_name

    st.header("4. Constraint Layers")
    for key in CONSTRAINT_LAYER_KEYS:
        if key not in st.session_state:
            st.session_state[key] = False
    # Apply "Select all" / "Deselect all" before widgets are created (Streamlit forbids mutating widget key state after widget creation)
    apply = st.session_state.pop("_constraint_apply", None)
    if apply == "select":
        for key in CONSTRAINT_LAYER_KEYS:
            st.session_state[key] = True
    elif apply == "deselect":
        for key in CONSTRAINT_LAYER_KEYS:
            st.session_state[key] = False

    show_major_roads = st.checkbox("Show major roads / highways (≥ 80 km/h)", value=st.session_state["show_major_roads"], key="show_major_roads")
    show_local_roads = st.checkbox("Show local / all roads (context)", value=st.session_state["show_local_roads"], key="show_local_roads")
    show_hospitals = st.checkbox("Hospitals / ER facilities", value=st.session_state["show_hospitals"], key="show_hospitals")
    show_airports = st.checkbox("Airports / Airfields", value=st.session_state["show_airports"], key="show_airports")
    show_protected_areas = st.checkbox("Protected Areas (Natura 2000)", value=st.session_state["show_protected_areas"], key="show_protected_areas")
    show_power_lines = st.checkbox("Power Lines / Infrastructure", value=st.session_state["show_power_lines"], key="show_power_lines")
    show_vital_infrastructure = st.checkbox("Vitale Infrastructuur (Critical Infrastructure)", value=st.session_state["show_vital_infrastructure"], key="show_vital_infrastructure")

    col_sel, col_desel = st.columns(2)
    with col_sel:
        if st.button("Select all", use_container_width=True, key="constraint_select_all"):
            st.session_state["_constraint_apply"] = "select"
            st.rerun()
    with col_desel:
        if st.button("Deselect all", use_container_width=True, key="constraint_deselect_all"):
            st.session_state["_constraint_apply"] = "deselect"
            st.rerun()

    show_major_roads = st.session_state["show_major_roads"]
    show_local_roads = st.session_state["show_local_roads"]
    show_hospitals = st.session_state["show_hospitals"]
    show_airports = st.session_state["show_airports"]
    show_protected_areas = st.session_state["show_protected_areas"]
    show_power_lines = st.session_state["show_power_lines"]
    show_vital_infrastructure = st.session_state["show_vital_infrastructure"]

    st.header("5. Session Management")
    with st.expander("💾 Save / Load Configuration", expanded=False):
        config_name = st.text_input("Configuration name", value="mission_1", key="config_name_input")
        
        col_save, col_load = st.columns(2)
        with col_save:
            if st.button("💾 Save", use_container_width=True):
                layer_settings = {
                    "show_major_roads": show_major_roads,
                    "show_local_roads": show_local_roads,
                    "show_hospitals": show_hospitals,
                    "show_airports": show_airports,
                    "show_protected_areas": show_protected_areas,
                    "show_power_lines": show_power_lines,
                    "show_vital_infrastructure": show_vital_infrastructure,
                }
                # Get current mitigation set from session state
                mitigation_set_to_save = st.session_state.get("mitigation_set", MitigationSet())
                save_mission_config(specs, selected_source_name, layer_settings, config_name, mitigation_set_to_save)
                st.success(f"Saved: {config_name}")
        
        with col_load:
            saved_configs = get_saved_config_names()
            if saved_configs:
                selected_config = st.selectbox("Load config", [""] + saved_configs, key="load_config_select")
                if selected_config and st.button("📂 Load", use_container_width=True):
                    loaded = load_mission_config(selected_config)
                    if loaded:
                        # Restore to session state for next rerun
                        st.session_state["load_config"] = loaded
                        st.session_state["load_config_name"] = selected_config
                        st.rerun()
        
        # Display saved configurations
        if saved_configs:
            st.caption(f"Saved: {len(saved_configs)} configuration(s)")
            for cfg_name in saved_configs[:5]:  # Show first 5
                cfg = load_mission_config(cfg_name)
                if cfg:
                    timestamp = cfg.get("timestamp", "Unknown")
                    st.caption(f"• {cfg_name} ({timestamp[:10] if len(timestamp) > 10 else timestamp})")

# Handle loaded configuration on rerun
if "load_config" in st.session_state:
    loaded = st.session_state["load_config"]
    layer_settings = loaded.get("layer_settings", {})
    for key in CONSTRAINT_LAYER_KEYS:
        st.session_state[key] = layer_settings.get(key, False)
    del st.session_state["load_config"]

# Get mitigation set from session state for sidebar display
mitigation_set_sidebar = st.session_state.get("mitigation_set", MitigationSet())
total_mitigation_sidebar = mitigation_set_sidebar.calculate_total_reduction()
st.sidebar.metric("Total GRC Reduction", f"{int(total_mitigation_sidebar)}", delta=int(total_mitigation_sidebar), delta_color="inverse")

col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Flight Plan Editor")
    fp_source = st.radio("Source:", ["Manual Draw", "Default: Rotterdam", "Default: Amsterdam", "Default: Unmanned Valley", "Upload File"], horizontal=True)
    
    active_coords = []
    
    if fp_source == "Upload File":
        up = st.file_uploader("KML/XML", type=['kml','xml'])
        if up: 
            parsed = parse_flight_plan(up.getvalue().decode("utf-8"))
            if parsed:
                active_coords = parsed
                if st.session_state.get('last_upload') != up.name:
                    st.session_state['map_center'] = active_coords[0]; st.session_state['map_zoom'] = 15; st.session_state['last_upload'] = up.name
                    st.session_state['drawn_coords'] = [] 

    elif fp_source == "Default: Rotterdam":
        active_coords = [[51.950, 4.050], [51.952, 4.055], [51.955, 4.052]]
        if st.session_state.get('last_source') != "Rotterdam":
            st.session_state['map_center'] = [51.952, 4.052]; st.session_state['map_zoom'] = 14; st.session_state['last_source'] = "Rotterdam"
            st.session_state['drawn_coords'] = []

    elif fp_source == "Default: Amsterdam":
        active_coords = [[52.300, 4.850], [52.300, 4.860], [52.305, 4.860], [52.305, 4.850]] 
        if st.session_state.get('last_source') != "Amsterdam":
            st.session_state['map_center'] = [52.302, 4.855]; st.session_state['map_zoom'] = 14; st.session_state['last_source'] = "Amsterdam"
            st.session_state['drawn_coords'] = []
            
    elif fp_source == "Default: Unmanned Valley":
        # Precise field coordinates
        active_coords = [[52.169189, 4.410820], [52.168049, 4.406923], [52.166515, 4.408235], [52.163255, 4.407668], [52.161908, 4.410082], [52.162624, 4.416992], [52.164861, 4.427268], [52.170422, 4.427511], [52.172276, 4.424011]]
        if st.session_state.get('last_source') != "Unmanned Valley":
             st.session_state['map_center'] = [52.166, 4.415]; st.session_state['map_zoom'] = 14; st.session_state['last_source'] = "Unmanned Valley"
             st.session_state['drawn_coords'] = []

    if st.session_state['drawn_coords']: active_coords = st.session_state['drawn_coords']

    geoms: Optional[BufferGeometries] = None
    density, count, hotspot, grid_features, debug_info = 0, 0, None, [], {}
    roads_overlap_m = 0.0
    local_roads_count = 0
    hospitals_intersecting = 0
    nearest_hospital_name = None
    nearest_hospital_dist_m = None
    # New layer variables
    airports_intersecting = 0
    nearest_airport_name = None
    nearest_airport_dist_m = None
    airport_metadata = {}
    protected_areas_intersecting = 0
    protected_areas_overlap_km2 = 0.0
    protected_areas_metadata = {}
    powerlines_intersecting = 0
    powerlines_overlap_m = 0.0
    powerlines_metadata = {}
    vital_infrastructure_intersecting = 0
    nearest_vital_name = None
    nearest_vital_dist_m = None
    vital_infrastructure_types = {}
    vital_infrastructure_metadata = {}

    if active_coords:
        engine = SORACalculator(specs)
        geoms = engine.get_buffers(active_coords)
        
        # Check if M2 requires GRB recalculation (step-back procedure)
        mitigation_set = st.session_state.get("mitigation_set", MitigationSet())
        if mitigation_set.requires_grb_recalculation():
            st.warning("⚠️ M2 mitigation applied: GRB step-back procedure required")
            st.info("""
            **TODO: Implement M2 GRB Recalculation Logic**
            
            When M2 mitigation is applied, the GRB width may need to be recalculated 
            using the step-back procedure per SORA v2.5 Annex B and Annex F.
            
            This may require:
            - Recalculating GRB width based on reduced critical area
            - Using modified parameters in buffer calculation
            - Referencing Annex F for calculation details
            
            For now, the standard GRB calculation is used.
            """)
        
        if geoms:
            with st.spinner("Calculating..."):
                # Resolve selected population source
                src_name = st.session_state.get("population_source_name", "CBS 100m (2023)")
                if src_name == "CBS 100m (2023)":
                    source_impl: PopulationSource = CbsPopulationSource()
                elif src_name == "Mock: CBS x1.2":
                    source_impl = MockScaledPopulationSource()
                elif src_name == "GHSL GHS-POP (stub)":
                    source_impl = GhslPopulationSource()
                else:
                    # Fallback to CBS if something unexpected is stored
                    source_impl = CbsPopulationSource()
                density, count, mode, hotspot, grid_features, debug_info = source_impl.query(geoms.GRB)

    m = folium.Map(location=st.session_state['map_center'], zoom_start=st.session_state['map_zoom'], tiles="CartoDB positron")
    Draw(export=True, draw_options={'polyline': True, 'polygon': True, 'rectangle': True, 'circle': False, 'marker': False}).add_to(m)

    if geoms:
        # Initialize and fetch constraint layers in parallel (cached)
        major_features, roads_overlap_m = [], 0.0
        local_features, local_roads_count = [], 0
        hospital_features, hospitals_intersecting, nearest_hospital_name, nearest_hospital_dist_m = [], 0, None, None
        airport_features, airports_intersecting, nearest_airport_name, nearest_airport_dist_m, airport_metadata = [], 0, None, None, {}
        protected_features, protected_areas_intersecting, protected_areas_overlap_km2, protected_areas_metadata = [], 0, 0.0, {}
        powerline_features, powerlines_intersecting, powerlines_overlap_m, powerlines_metadata = [], 0, 0.0, {}
        vital_features, vital_infrastructure_intersecting, nearest_vital_name, nearest_vital_dist_m, vital_infrastructure_types, vital_infrastructure_metadata = [], 0, None, None, {}, {}
        grb_wkt = geoms.GRB.wkt
        jobs = []
        if show_major_roads:
            jobs.append(("major_roads", lambda w=grb_wkt: get_major_roads_cached(w)))
        if show_local_roads:
            jobs.append(("local_roads", lambda w=grb_wkt: get_local_roads_cached(w)))
        if show_hospitals:
            jobs.append(("hospitals", lambda w=grb_wkt: get_hospitals_cached(w)))
        if show_airports:
            jobs.append(("airports", lambda w=grb_wkt: get_airports_cached(w)))
        if show_protected_areas:
            jobs.append(("protected_areas", lambda w=grb_wkt: get_protected_areas_cached(w)))
        if show_power_lines:
            jobs.append(("power_lines", lambda w=grb_wkt: get_power_lines_cached(w)))
        if show_vital_infrastructure:
            jobs.append(("vital_infrastructure", lambda w=grb_wkt: get_vital_infrastructure_cached(w)))
        if jobs:
            with st.spinner("Loading constraint layers..."):
                with ThreadPoolExecutor(max_workers=7) as executor:
                    future_to_layer = {}
                    for layer_id, fn in jobs:
                        future_to_layer[executor.submit(fn)] = layer_id
                    _layer_display_names = {
                        "major_roads": "Major roads / highways",
                        "local_roads": "Local / all roads",
                        "hospitals": "Hospitals / ER",
                        "airports": "Airports / Airfields",
                        "protected_areas": "Protected Areas (Natura 2000)",
                        "power_lines": "Power Lines",
                        "vital_infrastructure": "Vitale Infrastructuur (Critical Infrastructure)",
                    }
                    for future in as_completed(future_to_layer):
                        layer_id = future_to_layer[future]
                        try:
                            result = future.result()
                        except Exception:
                            display_name = _layer_display_names.get(layer_id, layer_id)
                            st.warning(f"Constraint layer could not be loaded: **{display_name}**.")
                            continue
                        if layer_id == "major_roads":
                            major_features, roads_overlap_m = result
                        elif layer_id == "local_roads":
                            local_features, local_roads_count = result
                        elif layer_id == "hospitals":
                            hospital_features, hospitals_intersecting, nearest_hospital_name, nearest_hospital_dist_m = result
                        elif layer_id == "airports":
                            airport_features, airports_intersecting, nearest_airport_name, nearest_airport_dist_m, airport_metadata = result
                        elif layer_id == "protected_areas":
                            protected_features, protected_areas_intersecting, protected_areas_overlap_km2, protected_areas_metadata = result
                        elif layer_id == "power_lines":
                            powerline_features, powerlines_intersecting, powerlines_overlap_m, powerlines_metadata = result
                        elif layer_id == "vital_infrastructure":
                            vital_features, vital_infrastructure_intersecting, nearest_vital_name, nearest_vital_dist_m, vital_infrastructure_types, vital_infrastructure_metadata = result
        # Constraint layers: major roads (Rijkswegen, ≥80 km/h)
        if show_major_roads:
            if major_features:
                major_group = folium.FeatureGroup(name="Major roads / highways (≥80 km/h)")
                for rf in major_features:
                    is_major = rf["properties"].get("is_major", False)
                    vmax = rf["properties"].get("vmax", 0)
                    folium.GeoJson(
                        rf["geometry"],
                        style_function=lambda feature, major=is_major: {
                            "color": "#ff4500" if major else "#ffcc80",
                            "weight": 3 if major else 1,
                            "dashArray": "4, 4" if major else "2, 6",
                            "opacity": 0.9 if major else 0.4,
                        },
                        tooltip=f"{'Autosnelweg/autoweg' if is_major else 'Other Rijksweg'} – {vmax} km/h",
                    ).add_to(major_group)
                major_group.add_to(m)

        # Constraint layers: local / full road network (context)
        if show_local_roads:
            if local_features:
                local_group = folium.FeatureGroup(name="Local / all roads")
                for rf in local_features:
                    folium.GeoJson(
                        rf["geometry"],
                        style_function=lambda x: {
                            "color": "#00bcd4",  # light blue for local roads
                            "weight": 1,
                            "dashArray": "2, 6",
                            "opacity": 0.5,
                        },
                        tooltip="Road",
                    ).add_to(local_group)
                local_group.add_to(m)

        # Constraint layers: hospitals / ER
        if show_hospitals:
            if hospital_features:
                hosp_group = folium.FeatureGroup(name="Hospitals / ER")
                for hf in hospital_features:
                    geom = hf["geometry"]
                    props = hf["properties"]
                    inside = props.get("inside_grb", False)
                    name = props.get("name", "Hospital")
                    lon, lat = geom["coordinates"]

                    icon_html = f"""
                    <div style="
                        display:flex;
                        align-items:center;
                        justify-content:center;
                        width:20px;
                        height:20px;
                        background-color:{'#ff0000' if inside else '#ff4d4d'};
                        color:white;
                        font-weight:bold;
                        border-radius:3px;
                        border:2px solid white;
                        box-shadow:0 0 3px rgba(0,0,0,0.4);
                        ">
                        H
                    </div>
                    """
                    icon = folium.DivIcon(html=icon_html)
                    folium.Marker(
                        location=[lat, lon],
                        icon=icon,
                        tooltip=name,
                    ).add_to(hosp_group)
                hosp_group.add_to(m)

        # Constraint layers: airports / airfields
        if show_airports:
            if airport_features:
                airport_group = folium.FeatureGroup(name="Airports / Airfields")
                for af in airport_features:
                    geom = af["geometry"]
                    props = af["properties"]
                    inside = props.get("inside_grb", False)
                    name = props.get("name", "Airport")
                    aeroway_type = props.get("aeroway_type", "airport")
                    lon, lat = geom["coordinates"]

                    icon_html = f"""
                    <div style="
                        display:flex;
                        align-items:center;
                        justify-content:center;
                        width:24px;
                        height:24px;
                        background-color:{'#ff6600' if inside else '#ff8800'};
                        color:white;
                        font-weight:bold;
                        border-radius:50%;
                        border:2px solid white;
                        box-shadow:0 0 4px rgba(0,0,0,0.5);
                        font-size:12px;
                        ">
                        ✈
                    </div>
                    """
                    icon = folium.DivIcon(html=icon_html)
                    folium.Marker(
                        location=[lat, lon],
                        icon=icon,
                        tooltip=f"{name} ({aeroway_type})",
                    ).add_to(airport_group)
                airport_group.add_to(m)

        # Constraint layers: protected areas (Natura 2000)
        if show_protected_areas:
            if protected_features:
                protected_group = folium.FeatureGroup(name="Protected Areas (Natura 2000)")
                for pf in protected_features:
                    folium.GeoJson(
                        pf["geometry"],
                        style_function=lambda x: {
                            "color": "#228B22",  # forest green
                            "weight": 2,
                            "fillOpacity": 0.3,
                            "dashArray": "8, 4",
                        },
                        tooltip=pf["properties"].get("name", "Protected Area"),
                    ).add_to(protected_group)
                protected_group.add_to(m)

        # Constraint layers: power lines
        if show_power_lines:
            if powerline_features:
                powerline_group = folium.FeatureGroup(name="Power Lines")
                for plf in powerline_features:
                    voltage = plf["properties"].get("voltage", "Unknown")
                    folium.GeoJson(
                        plf["geometry"],
                        style_function=lambda x: {
                            "color": "#FFD700",  # gold
                            "weight": 2,
                            "opacity": 0.8,
                            "dashArray": "6, 3",
                        },
                        tooltip=f"Power Line ({voltage})",
                    ).add_to(powerline_group)
                powerline_group.add_to(m)

        # Constraint layers: vital infrastructure (vitale infrastructuur) - PDOK NCTV Categories
        if show_vital_infrastructure:
            if vital_features:
                # Separate Category A and B for different styling
                vital_group_a = folium.FeatureGroup(name="Vitale Infrastructuur - Category A (Critical)")
                vital_group_b = folium.FeatureGroup(name="Vitale Infrastructuur - Category B (Severe)")
                
                for vf in vital_features:
                    geom = vf["geometry"]
                    props = vf["properties"]
                    inside = props.get("inside_grb", False)
                    name = props.get("name", "Infrastructure")
                    infra_type = props.get("infrastructure_type", "Unknown")
                    category = props.get("category", "B")  # Default to B if not specified
                    
                    # Category A = Red, Category B = Blue
                    if category == "A":
                        icon_color = "#FF0000"  # Red
                        line_color = "#FF0000"
                        group = vital_group_a
                        category_label = "Category A (Critical)"
                    else:
                        icon_color = "#0066CC"  # Blue
                        line_color = "#0066CC"
                        group = vital_group_b
                        category_label = "Category B (Severe)"
                    
                    # Handle different geometry types
                    geom_type = geom.get("type", "Point")
                    
                    if geom_type == "Point":
                        # Point geometry - use marker
                        lon, lat = geom["coordinates"]
                        icon_html = f"""
                        <div style="
                            display:flex;
                            align-items:center;
                            justify-content:center;
                            width:24px;
                            height:24px;
                            background-color:{icon_color};
                            color:white;
                            font-weight:bold;
                            border-radius:50%;
                            border:3px solid white;
                            box-shadow:0 0 5px rgba(0,0,0,0.6);
                            font-size:12px;
                            ">
                            ⚠
                        </div>
                        """
                        icon = folium.DivIcon(html=icon_html)
                        folium.Marker(
                            location=[lat, lon],
                            icon=icon,
                            tooltip=f"{name} ({infra_type}) - {category_label}",
                            popup=folium.Popup(
                                f"<b>{name}</b><br>Type: {infra_type}<br>Category: {category_label}<br>{'🚨 Inside GRB' if inside else 'Near GRB'}",
                                max_width=250
                            ),
                        ).add_to(group)
                    else:
                        # LineString or Polygon - use GeoJson
                        folium.GeoJson(
                            geom,
                            style_function=lambda x, cat_color=line_color: {
                                "color": cat_color,
                                "weight": 3 if category == "A" else 2,
                                "opacity": 0.9 if category == "A" else 0.7,
                                "fillOpacity": 0.2 if category == "A" else 0.1,
                                "dashArray": "8, 4" if category == "A" else "4, 4",
                            },
                            tooltip=f"{name} ({infra_type}) - {category_label}",
                            popup=folium.Popup(
                                f"<b>{name}</b><br>Type: {infra_type}<br>Category: {category_label}<br>{'🚨 Inside GRB' if inside else 'Near GRB'}",
                                max_width=250
                            ),
                        ).add_to(group)
                
                vital_group_a.add_to(m)
                vital_group_b.add_to(m)

        # Grids
        if grid_features:
            grid_group = folium.FeatureGroup(name="Analysis Grids")
            for grid in grid_features:
                pop = grid['properties']['viz_pop']
                color = grid['properties']['viz_color']
                is_max = grid['properties']['is_hotspot']
                if is_max: color = "#0000FF"
                style = {'color': color, 'weight': 3 if is_max else 1.5, 'fillOpacity': 0.1, 'dashArray': '5, 5'}
                folium.GeoJson(grid['geometry'], style_function=lambda x, s=style: s, tooltip=f"Pop: {pop}").add_to(grid_group)
            grid_group.add_to(m)

        fg_buff = folium.FeatureGroup(name="SORA Buffers")
        folium.GeoJson(geoms.GRB, style_function=lambda x: {'color': '#ff0000', 'weight': 1, 'fillOpacity': 0.1}, tooltip="GRB").add_to(fg_buff)
        folium.GeoJson(geoms.CV, style_function=lambda x: {'color': '#ffff00', 'weight': 1, 'fillOpacity': 0.2}, tooltip="CV").add_to(fg_buff)
        folium.GeoJson(geoms.FG, style_function=lambda x: {'color': '#00ff00', 'weight': 2, 'fillOpacity': 0.4}, tooltip="FG").add_to(fg_buff)
        fg_buff.add_to(m)
        
        if hotspot:
            folium.GeoJson(hotspot, name="Hotspot", style_function=lambda x:{'color':'#0000FF', 'weight':4, 'fillOpacity':0.0, 'dashArray':'5,5'}).add_to(m)

    folium.LayerControl().add_to(m)
    output = st_folium(m, width=800, height=500)

    if output.get("all_drawings"):
        last = output["all_drawings"][-1]
        raw = last['geometry']['coordinates']
        if last['geometry']['type'] == "Polygon": new_c = [[p[1], p[0]] for p in raw[0]]
        else: new_c = [[p[1], p[0]] for p in raw]
        if new_c != st.session_state['drawn_coords']:
            st.session_state['drawn_coords'] = new_c
            st.rerun()

    with col2:
        st.metric("Max Density", f"{int(density)} ppl/km²")
        st.caption(f"Cells Intersecting: {count} (Analysis Range)")

        # Dataset provenance
        source_meta = debug_info.get("source", {})
        if source_meta:
            src_label = source_meta.get("name", "Unknown source")
            vintage = source_meta.get("vintage", "?")
            res = source_meta.get("resolution_m", "?")
            st.caption(f"Dataset: {src_label} – {vintage}, {res} m grid")

        # Roads constraint summary
        if roads_overlap_m > 0:
            st.caption(f"Road overlap (within GRB): ~{int(roads_overlap_m)} m")
        else:
            st.caption("Road overlap (within GRB): none detected")

        # Local roads debug summary (for now, to verify data pull)
        if 'local_roads_count' in locals() and local_roads_count > 0:
            st.caption(f"Local roads fetched (context): {local_roads_count}")
        elif show_local_roads:
            st.caption("Local roads fetched (context): 0")

        # Hospitals / ER summary
        if show_hospitals:
            if hospitals_intersecting > 0:
                st.caption(f"Hospitals intersecting GRB: {hospitals_intersecting}")
            if nearest_hospital_name and nearest_hospital_dist_m is not None:
                st.caption(
                    f"Nearest hospital: {nearest_hospital_name} (~{int(nearest_hospital_dist_m)} m from GRB)"
                )

        # Airports / Airfields summary
        if show_airports:
            if airports_intersecting > 0:
                st.warning(f"⚠️ {airports_intersecting} airport(s) intersect GRB")
            if nearest_airport_name and nearest_airport_dist_m is not None:
                st.caption(
                    f"Nearest airport: {nearest_airport_name} (~{int(nearest_airport_dist_m)} m from GRB)"
                )
            if airport_metadata:
                st.caption(f"Source: {airport_metadata.get('name', 'Unknown')}")

        # Protected Areas summary
        if show_protected_areas:
            if protected_areas_intersecting > 0:
                st.warning(f"⚠️ {protected_areas_intersecting} protected area(s) intersect GRB")
            if protected_areas_overlap_km2 > 0:
                st.caption(f"Protected area overlap: ~{protected_areas_overlap_km2:.2f} km²")
            if protected_areas_metadata:
                st.caption(f"Source: {protected_areas_metadata.get('name', 'Unknown')} ({protected_areas_metadata.get('vintage', '?')})")

        # Power Lines summary
        if show_power_lines:
            if powerlines_intersecting > 0:
                st.warning(f"⚠️ {powerlines_intersecting} power line segment(s) intersect GRB")
            if powerlines_overlap_m > 0:
                st.caption(f"Power line overlap: ~{int(powerlines_overlap_m)} m")
            if powerlines_metadata:
                st.caption(f"Source: {powerlines_metadata.get('name', 'Unknown')}")

        # Vital Infrastructure summary (NCTV Categories)
        if show_vital_infrastructure:
            if vital_infrastructure_intersecting > 0:
                st.error(f"🚨 {vital_infrastructure_intersecting} critical infrastructure facility/facilities intersect GRB")
            
            # Show category breakdown
            if vital_infrastructure_metadata and "category_counts" in vital_infrastructure_metadata:
                cat_counts = vital_infrastructure_metadata["category_counts"]
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Category A (Critical)", cat_counts.get("A", 0), delta=None, delta_color="off")
                with col_b:
                    st.metric("Category B (Severe)", cat_counts.get("B", 0), delta=None, delta_color="off")
            
            if nearest_vital_name and nearest_vital_dist_m is not None:
                st.caption(
                    f"Nearest: {nearest_vital_name} (~{int(nearest_vital_dist_m)} m from GRB)"
                )
            if vital_infrastructure_types:
                with st.expander("📋 Infrastructure Types (NCTV)", expanded=False):
                    # Group by category
                    st.markdown("**Category A (Critical - Red):**")
                    cat_a_types = ["High Voltage Grid", "Primary Dike", "Pumping Station"]
                    for infra_type in cat_a_types:
                        if infra_type in vital_infrastructure_types:
                            st.caption(f"• {infra_type}: {vital_infrastructure_types[infra_type]}")
                    
                    st.markdown("**Category B (Severe - Blue):**")
                    cat_b_types = ["Fire Station", "Police Station", "Railway", "Main Road (State)", "Low Voltage Grid"]
                    for infra_type in cat_b_types:
                        if infra_type in vital_infrastructure_types:
                            st.caption(f"• {infra_type}: {vital_infrastructure_types[infra_type]}")
                    
                    # Show any other types
                    other_types = {k: v for k, v in vital_infrastructure_types.items() 
                                 if k not in cat_a_types + cat_b_types}
                    if other_types:
                        st.markdown("**Other:**")
                        for infra_type, count in sorted(other_types.items(), key=lambda x: x[1], reverse=True):
                            st.caption(f"• {infra_type}: {count}")
            
            if vital_infrastructure_metadata:
                st.caption(f"Source: {vital_infrastructure_metadata.get('name', 'Unknown')} ({vital_infrastructure_metadata.get('vintage', '?')})")
                if "categories" in vital_infrastructure_metadata:
                    with st.expander("ℹ️ Category Definitions", expanded=False):
                        cats = vital_infrastructure_metadata["categories"]
                        st.caption(f"**{cats.get('A', 'Category A')}**")
                        st.caption(f"**{cats.get('B', 'Category B')}**")

        # Grid colour legend (applies to CBS + GHSL)
        with st.expander("🗺️ Grid colour legend", expanded=False):
            st.markdown(
                "- **Teal** (`#008080`): 1–99 people per cell  \n"
                "- **Purple** (`#800080`): 100–199 people per cell  \n"
                "- **Black** (`#000000`): ≥200 people per cell  \n"
                "- **Blue** (`#0000FF`): Hotspot (max intersecting cell)"
            )
        
        # Show Adjacent Density if Strict is 0
        adj_max = debug_info.get('adjacent_max', 0)
        if density == 0 and adj_max > 0:
             st.info(f"✅ Flight area clear (0 intersection).")
             st.caption(f"Adjacent Density (Context): {adj_max*100} ppl/km²")
        
        if hotspot:
            st.info("Blue Box = Risk Driver")
            if st.button("🔍 Zoom to Hotspot"):
                hs_shape = shape(hotspot)
                st.session_state['map_center'] = [hs_shape.centroid.y, hs_shape.centroid.x]
                st.session_state['map_zoom'] = 18
                st.rerun()

        # Calculate Initial GRC from population density
        initial_grc = 1
        if density >= 50000: initial_grc = 6
        elif density >= 5000: initial_grc = 5
        elif density >= 500: initial_grc = 4
        elif density >= 50: initial_grc = 3
        elif density >= 1: initial_grc = 2
        
        # Get mitigation set from session state
        mitigation_set = st.session_state.get("mitigation_set", MitigationSet())
        total_mitigation = mitigation_set.calculate_total_reduction()
        
        # Calculate Final GRC
        final_grc_value = max(1, initial_grc + total_mitigation)
        
        st.metric(
            "Initial GRC",
            f"{initial_grc}",
            help="Based on population density"
        )
        
        st.metric(
            "Final GRC",
            f"{final_grc_value}",
            delta=int(total_mitigation),
            delta_color="inverse",
            help="Initial GRC + mitigation reductions"
        )
        
        # Show breakdown if mitigations applied
        if total_mitigation < 0:
            with st.expander("📊 Mitigation Breakdown", expanded=False):
                if mitigation_set.m1a:
                    st.write(f"**M1(A)**: {mitigation_set.m1a.corrective_factor} (Level: {mitigation_set.m1a.level.value})")
                if mitigation_set.m1b:
                    st.write(f"**M1(B)**: {mitigation_set.m1b.corrective_factor} (Level: {mitigation_set.m1b.level.value})")
                if mitigation_set.m1c:
                    st.write(f"**M1(C)**: {mitigation_set.m1c.corrective_factor} (Level: {mitigation_set.m1c.level.value})")
                if mitigation_set.m2:
                    st.write(f"**M2**: {mitigation_set.m2.corrective_factor} (Level: {mitigation_set.m2.level.value})")
                    st.info("⚠️ M2 applied: GRB recalculation required (step-back procedure)")
        
        if geoms:
            kml = create_kml_download(
                {"FG": geoms.FG, "CV": geoms.CV, "GRB": geoms.GRB},
                {"h_fg": specs.h_fg, "h_cv": geoms.metrics.h_cv},
                hotspot_geom=hotspot,
            )
            st.download_button("📥 Export KML", kml, "sora.kml")
        
        with st.expander("📊 Intersection Table", expanded=True):
            if 'cell_analysis' in debug_info and len(debug_info['cell_analysis']) > 0:
                df = pd.DataFrame(debug_info['cell_analysis'])
                st.dataframe(df[["Population", "Overlap %"]])
            else:
                st.write("No direct intersection.")