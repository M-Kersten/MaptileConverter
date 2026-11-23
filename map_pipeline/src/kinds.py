"""What a building is, from every register that says.

The BAG says what a building is used for, in eleven words that cannot tell a
church from a concert hall or a school from a hospital: both pairs share a
purpose. Others say more:

* **OpenStreetMap** tags the building itself, and in the Netherlands its
  buildings were imported from the BAG with the building id as ``ref:bag``,
  so a tag joins to a 3DBAG building on its key, exactly. ``building=church``,
  ``building=garages``, ``man_made=water_tower``, ``amenity=fuel``.
* **OpenStreetMap points** inside a building say what its ground floor is: a
  restaurant, a hotel lobby, a supermarket, a fire station's doors.
* **TOP10NL** types its buildings: kerk, school, ziekenhuis, kas, silo.
* **The monument register** says which are churches and which are mills.

They are read in that order, each filling in only what an earlier one left
empty, and the BAG's own purposes and the building's shape come last. The
result is a `kind` per building ("church", "school", "barn", "garages") and a
`ground_kind` for its street level ("horeca", "hotel"), and the facade stage
builds from those.
"""

from __future__ import annotations

# Every kind a building can be given here.
KINDS = frozenset({
    "church", "mosque", "synagogue", "tower", "castle", "fort", "bunker",
    "prison", "town_hall", "museum", "station", "school", "hospital", "hall",
    "greenhouse", "parking", "fuel_canopy", "canopy", "water_tower",
    "lighthouse", "chimney", "silo", "tank", "technical", "windmill", "barn",
    "shed", "garages", "industrial", "hotel", "supermarket", "fire_station",
})

# TOP10NL typegebouw.
TOP10NL_KINDS: dict[str, str] = {
    "kerk": "church", "kapel": "church", "klooster, abdij": "church",
    "moskee": "mosque", "synagoge": "synagogue",
    "toren": "tower", "klokkentoren": "tower",
    "watertoren": "water_tower", "vuurtoren": "lighthouse", "lichttoren": "lighthouse",
    "schoorsteen": "chimney", "koeltoren": "technical",
    "silo": "silo", "tank": "tank",
    "windmolen": "windmill", "windmolen: korenmolen": "windmill",
    "windmolen: watermolen": "windmill",
    "kas, warenhuis": "greenhouse",
    "parkeerdak, parkeerdek, parkeergarage": "parking",
    "tankstation": "fuel_canopy",
    "school": "school", "universiteit": "school",
    "ziekenhuis": "hospital", "psychiatrisch ziekenhuis, psychiatrisch centrum": "hospital",
    "kliniek, inrichting, sanatorium": "hospital",
    "sporthal": "hall", "zwembad": "hall", "stadion": "hall", "manege": "hall",
    "kunstijsbaan": "hall",
    "stationsgebouw": "station", "gemeentehuis": "town_hall",
    "stadskantoor, hulpsecretarie": "town_hall", "museum": "museum",
    "kasteel": "castle", "paleis": "castle", "fort": "fort", "bunker": "bunker",
    "gevangenis": "prison", "brandweerkazerne": "fire_station", "hotel": "hotel",
    "transformatorstation": "technical", "elektriciteitscentrale": "technical",
    "gemaal": "technical", "pompstation": "technical",
    "fabriek": "industrial", "werf": "industrial", "loods": "industrial",
    "remise": "industrial", "veiling": "industrial",
}

# OpenStreetMap building=* values that say something. The generic ones the
# BAG import wrote (yes, house, apartments, residential) are left out.
OSM_BUILDING_KINDS: dict[str, str] = {
    "church": "church", "cathedral": "church", "chapel": "church",
    "monastery": "church", "mosque": "mosque", "synagogue": "synagogue",
    "school": "school", "kindergarten": "school", "university": "school",
    "college": "school", "hospital": "hospital",
    "sports_hall": "hall", "sports_centre": "hall", "stadium": "hall",
    "grandstand": "hall", "riding_hall": "hall",
    "greenhouse": "greenhouse", "glasshouse": "greenhouse",
    "garage": "shed", "carport": "canopy", "garages": "garages",
    "shed": "shed", "hut": "shed", "cabin": "shed", "kiosk": "shed",
    "toilets": "shed",
    "barn": "barn", "farm_auxiliary": "barn", "cowshed": "barn", "stable": "barn",
    "sty": "barn", "slurry_tank": "tank",
    "silo": "silo", "storage_tank": "tank", "tank": "tank",
    "water_tower": "water_tower", "transformer_tower": "technical",
    "bunker": "bunker", "castle": "castle", "fort": "fort",
    "windmill": "windmill", "train_station": "station", "transportation": "station",
    "parking": "parking", "roof": "canopy",
    "hotel": "hotel", "supermarket": "supermarket", "fire_station": "fire_station",
    "industrial": "industrial", "warehouse": "industrial", "manufacture": "industrial",
    "service": "technical", "digester": "tank",
}

# Other tags on a building way, in the order they are tried.
OSM_TAG_KINDS: tuple[tuple[str, str, str], ...] = (
    ("man_made", "windmill", "windmill"),
    ("man_made", "water_tower", "water_tower"),
    ("man_made", "lighthouse", "lighthouse"),
    ("man_made", "chimney", "chimney"),
    ("man_made", "silo", "silo"),
    ("man_made", "storage_tank", "tank"),
    ("man_made", "gasometer", "tank"),
    ("historic", "castle", "castle"),
    ("historic", "fort", "fort"),
    ("military", "bunker", "bunker"),
    ("amenity", "fuel", "fuel_canopy"),
    ("amenity", "prison", "prison"),
    ("amenity", "school", "school"),
    ("amenity", "hospital", "hospital"),
    ("amenity", "townhall", "town_hall"),
    ("amenity", "fire_station", "fire_station"),
    ("leisure", "sports_hall", "hall"),
    ("leisure", "swimming_pool", "hall"),
    ("leisure", "stadium", "hall"),
    ("leisure", "ice_rink", "hall"),
    ("tourism", "museum", "museum"),
    ("tourism", "hotel", "hotel"),
    ("shop", "supermarket", "supermarket"),
)

RELIGION_KINDS = {"christian": "church", "muslim": "mosque", "jewish": "synagogue"}

# What the model calls each kind where someone clicks on it.
KIND_LABELS = {
    "church": "kerk", "mosque": "moskee", "synagogue": "synagoge", "tower": "toren",
    "castle": "kasteel", "fort": "fort", "bunker": "bunker", "prison": "gevangenis",
    "town_hall": "gemeentehuis", "museum": "museum", "station": "station",
    "school": "school", "hospital": "ziekenhuis", "hall": "hal of zaal",
    "greenhouse": "kas", "parking": "parkeergarage", "fuel_canopy": "tankstation",
    "canopy": "overkapping", "water_tower": "watertoren", "lighthouse": "vuurtoren",
    "chimney": "schoorsteen", "silo": "silo", "tank": "tank",
    "technical": "technisch gebouw", "windmill": "molen", "barn": "schuur of stal",
    "shed": "bijgebouw", "garages": "garageboxen", "industrial": "bedrijfshal",
    "hotel": "hotel", "supermarket": "supermarkt", "fire_station": "brandweerkazerne",
}

# What a point inside a building says about its ground floor.
HORECA = frozenset({
    "restaurant", "cafe", "bar", "pub", "fast_food", "ice_cream", "biergarten",
})
GROUND_PRIORITY = ("fire_station", "hotel", "supermarket", "horeca", "shop")


def kind_from_osm(tags: dict) -> str:
    """The kind an OpenStreetMap building's tags give it, or ""."""
    building = str(tags.get("building") or "")
    if building == "roof" and tags.get("amenity") == "fuel":
        return "fuel_canopy"
    if tags.get("amenity") == "place_of_worship":
        religion = RELIGION_KINDS.get(str(tags.get("religion") or ""))
        if religion:
            return religion
    for key, value, kind in OSM_TAG_KINDS:
        if tags.get(key) == value:
            # A petrol station's shop is a building tagged amenity=fuel
            # too; only the roof over the pumps is a canopy.
            if kind == "fuel_canopy" and building not in ("roof", ""):
                continue
            return kind
    return OSM_BUILDING_KINDS.get(building, "")


def kind_from_top10(types) -> str:
    for value in types:
        kind = TOP10NL_KINDS.get(str(value).strip())
        if kind:
            return kind
    return ""


def kind_from_monument(category: str, subcategory: str) -> str:
    sub = (subcategory or "").lower()
    if "molen" in sub and "water" not in sub:
        return "windmill"
    if "schuur" in sub or "hooiberg" in sub or "stal" in sub.split():
        return "barn"
    if (category or "").startswith("Religieuze gebouwen"):
        return "church"
    if (category or "").startswith("Kastelen"):
        return "castle"
    if (category or "").startswith("Bestuursgebouwen"):
        return "town_hall"
    return ""


def ground_kind_from_osm(tags: dict) -> str:
    """What a point inside a building says about its ground floor, or ""."""
    if tags.get("amenity") == "fire_station":
        return "fire_station"
    if tags.get("tourism") in ("hotel", "hostel", "guest_house", "motel"):
        return "hotel"
    if tags.get("shop") == "supermarket":
        return "supermarket"
    if tags.get("amenity") in HORECA:
        return "horeca"
    if tags.get("shop"):
        return "shop"
    return ""


def stronger_ground(current: str, new: str) -> str:
    """The more telling of two ground-floor kinds: a hotel's lobby outranks
    the cafe in it."""
    if not current:
        return new
    if not new:
        return current
    return min(current, new, key=lambda k: GROUND_PRIORITY.index(k) if k in GROUND_PRIORITY else 99)


def bag_key(identifier: str) -> str:
    """A BAG building id without its prefix or leading zeros, for joining:
    3DBAG writes NL.IMBAG.Pand.0344100000000004, OpenStreetMap 344100000000004."""
    return str(identifier).rsplit(".", 1)[-1].lstrip("0")


__all__ = [
    "GROUND_PRIORITY",
    "HORECA",
    "KINDS",
    "KIND_LABELS",
    "OSM_BUILDING_KINDS",
    "OSM_TAG_KINDS",
    "TOP10NL_KINDS",
    "bag_key",
    "ground_kind_from_osm",
    "kind_from_monument",
    "kind_from_osm",
    "kind_from_top10",
    "stronger_ground",
]
