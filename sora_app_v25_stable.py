import streamlit as st
import folium
from streamlit_folium import st_folium
from folium.plugins import Draw
import math
import requests
import json
import pandas as pd
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple, Protocol
from shapely.geometry import Polygon, LineString, Point, shape, mapping
from shapely.ops import transform
import pyproj
import xml.etree.ElementTree as ET

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

# --- 4. EXPORT KML ---
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

# --- 5. APP LAYOUT ---
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
        m1 = st.checkbox("M1 Sheltering")
        m2 = st.selectbox("M2 Parachute", ["None", "Medium", "High"])
        
        applied = st.form_submit_button("Apply")

    m2_val = -1 if "Medium" in m2 else -2 if "High" in m2 else 0
    total_mitigation = ( -1 if m1 else 0 ) + m2_val
    
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

st.sidebar.metric("Total GRC Reduction", f"{total_mitigation}")

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

    if active_coords:
        engine = SORACalculator(specs)
        geoms = engine.get_buffers(active_coords)
        if geoms:
            with st.spinner("Calculating..."):
                # For now we use CBS as the default PopulationSource implementation
                cbs_source = CbsPopulationSource()
                density, count, mode, hotspot, grid_features, debug_info = cbs_source.query(geoms.GRB)

    m = folium.Map(location=st.session_state['map_center'], zoom_start=st.session_state['map_zoom'], tiles="CartoDB positron")
    Draw(export=True, draw_options={'polyline': True, 'polygon': True, 'rectangle': True, 'circle': False, 'marker': False}).add_to(m)

    if geoms:
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

        final_grc = 1
        if density >= 50000: final_grc = 6
        elif density >= 5000: final_grc = 5
        elif density >= 500: final_grc = 4
        elif density >= 50: final_grc = 3
        elif density >= 1: final_grc = 2
        
        st.metric("Final GRC", f"{max(1, final_grc + total_mitigation)}", delta=total_mitigation, delta_color="inverse")
        
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