# map_app.py
import json, re
from pathlib import Path
from collections import defaultdict

import pandas as pd
import streamlit as st
import pydeck as pdk
from geonamescache import GeonamesCache
from unidecode import unidecode

st.set_page_config(page_title="Class Map", layout="wide")

DATA_PATH = Path("summaries.json")  # <-- changed

# -------------------- Load data --------------------
@st.cache_data
def load_records():
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    return data

# -------------------- Build location -> people --------------------
def build_location_people(records):
    loc_to_people = defaultdict(set)
    for r in records:
        person = (r.get("Name") or "").strip()
        for loc in r.get("Location(s)") or []:
            loc_clean = re.sub(r"\s+", " ", str(loc)).strip(" ,")
            if loc_clean and person:
                loc_to_people[loc_clean].add(person)
    return loc_to_people

# -------------------- Offline geocoder --------------------
US_STATES = {
 "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA","KS",
 "KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ","NM","NY",
 "NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT","VA","WA","WV",
 "WI","WY","DC"
}
CITY_ST = re.compile(r"^(.+?),\s*([A-Za-z]{2})$")
CITY_COUNTRY = re.compile(r"^(.+?),\s*([A-Za-z .'\-]{2,40})$")

gc = GeonamesCache()
cities_all = list(gc.get_cities().values())
countries = gc.get_countries()
countries_by_name = {unidecode(v["name"]).lower(): k for k, v in countries.items()}
countries_by_alt = {}
for iso, v in countries.items():
    for alt in v.get("alternatenames", []):
        countries_by_alt[unidecode(alt).lower()] = iso

def norm(s): return unidecode((s or "")).lower().strip()
name_index = defaultdict(list)
for c in cities_all:
    name_index[norm(c["name"])].append(c)

def country_iso_from_name(name: str):
    n = norm(name)
    if n in countries_by_name: return countries_by_name[n]
    if n in countries_by_alt: return countries_by_alt[n]
    aliases = {"uk":"GB","u.k.":"GB","united kingdom":"GB","usa":"US","u.s.":"US","u.s.a.":"US",
               "south korea":"KR","north korea":"KP","czech republic":"CZ","cote d'ivoire":"CI",
               "ivory coast":"CI","viet nam":"VN"}
    return aliases.get(n)

@st.cache_data
def geocode_locations(loc_to_people: dict[str, set[str]]):
    rows = []
    for loc, people in loc_to_people.items():
        lat = lon = None
        m_us = CITY_ST.match(loc)
        if m_us and m_us.group(2).upper() in US_STATES:
            city = m_us.group(1).strip()
            candidates = [c for c in name_index.get(norm(city), []) if c.get("countrycode") == "US"]
            if candidates:
                best = max(candidates, key=lambda x: x.get("population") or 0)
                lat, lon = float(best["latitude"]), float(best["longitude"])
        else:
            m_world = CITY_COUNTRY.match(loc)
            if m_world:
                city = m_world.group(1).strip()
                country_name = m_world.group(2).strip()
                iso = country_iso_from_name(country_name)
                if iso:
                    candidates = [c for c in name_index.get(norm(city), []) if c.get("countrycode") == iso]
                    if candidates:
                        best = max(candidates, key=lambda x: x.get("population") or 0)
                        lat, lon = float(best["latitude"]), float(best["longitude"])

        if lat is not None and lon is not None:
            rows.append({
                "loc": loc,
                "lat": lat,
                "lon": lon,
                "count": len(people),
                "names": ", ".join(sorted(people))
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.groupby(["lat","lon","loc"], as_index=False).agg({
            "count":"sum",
            "names": lambda s: ", ".join(sorted(set(", ".join(s).split(", "))))
        })
    return df

# -------------------- UI --------------------
st.title("Class Map")
left, right = st.columns([2,1], gap="large")

with st.sidebar:
    st.header("Options")
    offline = st.checkbox("Offline mode (no basemap)", value=False,
                          help="If your network blocks basemap tiles, use this.")
    max_names = st.slider("Max names in tooltip", 10, 500, 120, 10)
    show_table = st.checkbox("Show table preview", value=True)
    base_radius_km = st.slider("Dot base size (km)", 5, 200, 60, 5)
    growth_per_person = st.slider("Growth per person (km)", 0, 50, 8, 1)

records = load_records()
loc_to_people = build_location_people(records)
df = geocode_locations(loc_to_people)

# Diagnostics
with right:
    st.metric("Unique locations in JSON", len(loc_to_people))
    st.metric("Mappable locations", len(df))
    st.caption("If 'Mappable' is 0, check that locations look like 'City, ST' or 'City, Country'.")

if df.empty:
    st.warning("No mappable rows. Try adjusting your locations or inspect the JSON.")
    st.json(list(loc_to_people.items())[:10])
    st.stop()

# Trim tooltips so the map doesn't choke on huge strings
df["names_trimmed"] = df["names"].apply(lambda s: (s[:max_names] + "…") if len(s) > max_names else s)

# View center
view_state = pdk.ViewState(
    latitude=float((df["lat"] * df["count"]).sum() / df["count"].sum()),
    longitude=float((df["lon"] * df["count"]).sum() / df["count"].sum()),
    zoom=2
)

# Layers
# Convert “km” sliders to meters (if you want to use them dynamically)
df["radius"] = df["count"].apply(lambda n: 60000 * max(1, min(6, n)))  # meters
scatter = pdk.Layer(
    "ScatterplotLayer",
    data=df,
    get_position="[lon, lat]",
    get_radius="radius",
    pickable=True,
    get_fill_color="[30, 144, 255, 180]",
)

tooltip = {
    "html": "<b>{loc}</b><br/>People: {count}<br/>{names_trimmed}",
    "style": {"backgroundColor": "white", "color": "black"}
}

map_style = None if offline else "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
deck = pdk.Deck(map_style=map_style, initial_view_state=view_state, layers=[scatter], tooltip=tooltip)

with left:
    st.pydeck_chart(deck)

if show_table:
    st.subheader("Geocoded Locations")
    st.dataframe(df[["loc","count","lat","lon","names"]], use_container_width=True, height=320)
