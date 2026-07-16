from __future__ import annotations

import asyncio
import math
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import httpx

from .cache import SQLiteCache


class GeoResearchError(RuntimeError):
    pass


_CATEGORY_FILTERS = {
    "restaurant": '["amenity"="restaurant"]',
    "food": '["amenity"~"^(restaurant|fast_food|cafe)$"]',
    "lodging": '["tourism"~"^(hotel|motel|hostel|guest_house)$"]',
    "retail": '["shop"]',
    "attraction": '["tourism"~"^(attraction|museum|gallery|theme_park)$"]',
    "named_place": '["name"]',
}
_NAME_WORDS = re.compile(r"[a-z0-9]+")


def _haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6_371_008.8
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    value = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    return 2 * radius * math.asin(math.sqrt(value))


def _entity_key(value: str) -> str:
    return " ".join(_NAME_WORDS.findall(value.casefold()))


class GeoResearchClient:
    """Cached public-OSM research for location and walking-distance clues."""

    def __init__(
        self,
        cache_path: Path,
        *,
        client: httpx.AsyncClient | None = None,
        photon_base_url: str = "https://photon.komoot.io/api/",
        nominatim_base_url: str = "https://nominatim.openstreetmap.org/search",
        overpass_base_url: str = "https://overpass-api.de/api/interpreter",
        overpass_fallback_urls: tuple[str, ...] = (
            "https://overpass.kumi.systems/api/interpreter",
            "https://overpass.private.coffee/api/interpreter",
        ),
        valhalla_base_url: str = "https://valhalla1.openstreetmap.de/sources_to_targets",
        timeout_seconds: float = 30,
    ) -> None:
        self._owns_client = client is None
        self.client = client or httpx.AsyncClient(
            timeout=timeout_seconds,
            headers={"User-Agent": "BrowseComp250-Star/0.1"},
        )
        self.photon_base_url = photon_base_url
        self.nominatim_base_url = nominatim_base_url
        self.overpass_base_url = overpass_base_url
        self.overpass_urls = list(dict.fromkeys((overpass_base_url, *overpass_fallback_urls)))
        self.valhalla_base_url = valhalla_base_url
        self.timeout_seconds = timeout_seconds
        self.geocode_cache = SQLiteCache(cache_path, "geo:photon")
        self.places_cache = SQLiteCache(cache_path, "geo:overpass")
        self.route_cache = SQLiteCache(cache_path, "geo:valhalla")
        self._overpass_slots = asyncio.Semaphore(2)

    async def close(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    async def explore(
        self,
        anchors: list[dict[str, Any]],
        *,
        category: str = "named_place",
        max_results: int = 50,
        include_walking_routes: bool = True,
    ) -> dict[str, Any]:
        if category not in _CATEGORY_FILTERS:
            raise GeoResearchError(f"Unsupported geo category: {category}")
        if not 1 <= len(anchors) <= 4:
            raise GeoResearchError("geo_search requires between one and four anchors")
        max_results = max(1, min(int(max_results), 100))
        explored = await asyncio.gather(
            *(
                self._explore_anchor(
                    anchor,
                    category=category,
                    max_results=max_results,
                    include_walking_routes=include_walking_routes,
                )
                for anchor in anchors
            ),
            return_exceptions=True,
        )
        rows: list[dict[str, Any]] = []
        for anchor, result in zip(anchors, explored, strict=True):
            if isinstance(result, Exception):
                rows.append(
                    {
                        "query": str(anchor.get("query") or ""),
                        "ok": False,
                        "error": f"{type(result).__name__}: {result}",
                    }
                )
            else:
                rows.append(result)

        entity_occurrences: dict[str, dict[str, Any]] = {}
        for anchor_index, row in enumerate(rows):
            for place in row.get("places") or []:
                labels = {
                    str(place.get("name") or "").strip(),
                    str(place.get("brand") or "").strip(),
                }
                for label in labels - {""}:
                    key = _entity_key(label)
                    if not key:
                        continue
                    occurrence = entity_occurrences.setdefault(
                        key,
                        {"label": label, "matches_by_anchor": {}},
                    )
                    match = {
                        "anchor_index": anchor_index,
                        "name": place.get("name"),
                        "brand": place.get("brand"),
                        "straight_line_miles": place.get("straight_line_miles"),
                        "walking_miles": place.get("walking_miles"),
                        "distance_error_miles": place.get("distance_error_miles"),
                    }
                    previous = occurrence["matches_by_anchor"].get(anchor_index)
                    if previous is None or self._match_sort_key(match) < self._match_sort_key(
                        previous
                    ):
                        occurrence["matches_by_anchor"][anchor_index] = match
        shared_entities: list[dict[str, Any]] = []
        for item in entity_occurrences.values():
            matches = list(item["matches_by_anchor"].values())
            if len(matches) < 2:
                continue
            errors = [
                float(match["distance_error_miles"])
                for match in matches
                if match.get("distance_error_miles") is not None
            ]
            shared_entities.append(
                {
                    "label": item["label"],
                    "anchor_count": len(matches),
                    "expected_distance_anchor_count": len(errors),
                    "total_distance_error_miles": round(sum(errors), 3) if errors else None,
                    "matches": sorted(matches, key=lambda match: match["anchor_index"]),
                }
            )
        shared_entities.sort(
            key=lambda item: (
                -item["anchor_count"],
                -item["expected_distance_anchor_count"],
                (
                    item["total_distance_error_miles"]
                    if item["total_distance_error_miles"] is not None
                    else float("inf")
                ),
                item["label"].casefold(),
            )
        )
        return {
            "ok": any(row.get("ok") for row in rows),
            "category": category,
            "distance_method": (
                "OpenStreetMap/Overpass candidates; straight-line distance plus Valhalla "
                "pedestrian routing when available. Historical map state may differ."
            ),
            "anchors": rows,
            "shared_entities": shared_entities[:100],
        }

    @staticmethod
    def _match_sort_key(match: dict[str, Any]) -> tuple[float, float]:
        error = match.get("distance_error_miles")
        walking = match.get("walking_miles")
        straight = match.get("straight_line_miles")
        return (
            float(error) if error is not None else float("inf"),
            float(walking if walking is not None else straight or float("inf")),
        )

    async def _explore_anchor(
        self,
        anchor: dict[str, Any],
        *,
        category: str,
        max_results: int,
        include_walking_routes: bool,
    ) -> dict[str, Any]:
        query = " ".join(str(anchor.get("query") or "").split()).strip()
        if not query:
            raise GeoResearchError("Every geo anchor requires a non-empty query")
        radius_m = max(100, min(int(anchor.get("radius_m") or 5_000), 25_000))
        expected_distance_miles = anchor.get("expected_distance_miles")
        if expected_distance_miles is not None:
            expected_distance_miles = float(expected_distance_miles)
            if expected_distance_miles < 0:
                raise GeoResearchError("expected_distance_miles must be non-negative")
        point = await self._geocode(query)
        places = await self._nearby_places(
            point["lat"],
            point["lon"],
            radius_m=radius_m,
            category=category,
        )
        for place in places:
            meters = _haversine_meters(
                point["lat"],
                point["lon"],
                place["lat"],
                place["lon"],
            )
            place["straight_line_miles"] = round(meters / 1_609.344, 3)
        places.sort(key=lambda item: item["straight_line_miles"])
        if expected_distance_miles is None:
            places = places[:max_results]
        elif len(places) > 500:
            nearest = places[:100]
            route_likely = sorted(
                places,
                key=lambda item: abs(
                    float(item["straight_line_miles"]) - expected_distance_miles * 0.8
                ),
            )
            selected: dict[tuple[str, Any], dict[str, Any]] = {}
            for place in nearest + route_likely:
                key = (str(place.get("osm_type") or ""), place.get("osm_id"))
                selected.setdefault(key, place)
                if len(selected) >= 500:
                    break
            places = list(selected.values())
        if include_walking_routes and places:
            routes = await self._walking_matrix(point, places)
            for place, route in zip(places, routes, strict=True):
                if route is None:
                    continue
                place["walking_miles"] = round(float(route["distance"]), 3)
                place["walking_minutes"] = round(float(route["time"]) / 60, 1)
                if expected_distance_miles is not None:
                    place["distance_error_miles"] = round(
                        abs(float(route["distance"]) - expected_distance_miles),
                        3,
                    )
            if expected_distance_miles is not None:
                places.sort(
                    key=lambda item: (
                        item.get("distance_error_miles", float("inf")),
                        item["straight_line_miles"],
                    )
                )
                places = places[:max_results]
        return {
            "ok": True,
            "query": query,
            "radius_m": radius_m,
            "expected_distance_miles": expected_distance_miles,
            "location": point,
            "places": places,
        }

    async def _geocode(self, query: str) -> dict[str, Any]:
        request = {"query": query, "limit": 5}
        cached = self.geocode_cache.get(request)
        if cached is not None:
            return dict(cached)
        result = None
        for candidate_query in self._geocode_query_variants(query):
            response = await self._request_with_retries(
                "GET",
                self.photon_base_url,
                params={"q": candidate_query, "limit": 5},
            )
            payload = response.json()
            features = payload.get("features") if isinstance(payload, dict) else None
            if not isinstance(features, list) or not features:
                continue
            feature = features[0]
            coordinates = (feature.get("geometry") or {}).get("coordinates") or []
            if len(coordinates) < 2:
                continue
            properties = feature.get("properties") or {}
            result = {
                "lat": float(coordinates[1]),
                "lon": float(coordinates[0]),
                "display_name": ", ".join(
                    str(value)
                    for value in (
                        properties.get("name"),
                        properties.get("street"),
                        properties.get("city"),
                        properties.get("state"),
                        properties.get("country"),
                    )
                    if value
                ),
                "matched_query": candidate_query,
                "source_url": self.photon_base_url
                + "?"
                + urlencode({"q": candidate_query, "limit": 5}),
            }
            break
        if result is None:
            response = await self._request_with_retries(
                "GET",
                self.nominatim_base_url,
                params={"q": query, "format": "jsonv2", "limit": 5},
            )
            payload = response.json()
            if not isinstance(payload, list) or not payload:
                raise GeoResearchError(f"No geocoding result for {query!r}")
            first = payload[0]
            if first.get("lat") is None or first.get("lon") is None:
                raise GeoResearchError(f"Geocoder omitted coordinates for {query!r}")
            result = {
                "lat": float(first["lat"]),
                "lon": float(first["lon"]),
                "display_name": str(first.get("display_name") or query),
                "matched_query": query,
                "source_url": self.nominatim_base_url
                + "?"
                + urlencode({"q": query, "format": "jsonv2", "limit": 5}),
            }
        self.geocode_cache.put(request, result)
        return result

    @staticmethod
    def _geocode_query_variants(query: str) -> list[str]:
        variants = [query]
        parts = [part.strip() for part in query.split(",")]
        if len(parts) >= 2 and not any(character.isdigit() for character in parts[0]):
            tail = ", ".join(parts[1:]).strip()
            if any(character.isdigit() for character in tail):
                variants.append(tail)
        return list(dict.fromkeys(variant for variant in variants if variant))

    async def _nearby_places(
        self,
        lat: float,
        lon: float,
        *,
        radius_m: int,
        category: str,
    ) -> list[dict[str, Any]]:
        request = {
            "lat": round(lat, 7),
            "lon": round(lon, 7),
            "radius_m": radius_m,
            "category": category,
        }
        cached = self.places_cache.get(request)
        if cached is not None:
            return [dict(item) for item in cached]
        query = (
            f"[out:json][timeout:25];nwr{_CATEGORY_FILTERS[category]}"
            f"(around:{radius_m},{lat:.7f},{lon:.7f});out center tags;"
        )
        async with self._overpass_slots:
            last_error: GeoResearchError | None = None
            for endpoint in self.overpass_urls:
                try:
                    response = await self._request_with_retries(
                        "POST",
                        endpoint,
                        attempts=1,
                        data={"data": query},
                    )
                    break
                except GeoResearchError as exc:
                    last_error = exc
            else:
                raise GeoResearchError(
                    f"All Overpass endpoints failed: {last_error}"
                ) from last_error
        payload = response.json()
        elements = payload.get("elements") if isinstance(payload, dict) else None
        if not isinstance(elements, list):
            raise GeoResearchError("Overpass response omitted elements")
        places: list[dict[str, Any]] = []
        seen: set[tuple[str, float, float]] = set()
        for element in elements:
            tags = element.get("tags") or {}
            name = str(tags.get("name") or tags.get("brand") or "").strip()
            center = element.get("center") or {}
            place_lat = element.get("lat", center.get("lat"))
            place_lon = element.get("lon", center.get("lon"))
            if not name or place_lat is None or place_lon is None:
                continue
            place_lat = float(place_lat)
            place_lon = float(place_lon)
            key = (_entity_key(name), round(place_lat, 6), round(place_lon, 6))
            if key in seen:
                continue
            seen.add(key)
            address = " ".join(
                part
                for part in (
                    str(tags.get("addr:housenumber") or "").strip(),
                    str(tags.get("addr:street") or "").strip(),
                )
                if part
            )
            places.append(
                {
                    "name": name,
                    "brand": str(tags.get("brand") or "").strip() or None,
                    "address": address or None,
                    "lat": place_lat,
                    "lon": place_lon,
                    "osm_type": element.get("type"),
                    "osm_id": element.get("id"),
                }
            )
        self.places_cache.put(request, places)
        return places

    async def _walking_matrix(
        self,
        source: dict[str, Any],
        places: list[dict[str, Any]],
    ) -> list[dict[str, float] | None]:
        batches = [places[index : index + 100] for index in range(0, len(places), 100)]
        rows = await asyncio.gather(
            *(self._walking_matrix_batch(source, batch) for batch in batches)
        )
        return [route for row in rows for route in row]

    async def _walking_matrix_batch(
        self,
        source: dict[str, Any],
        places: list[dict[str, Any]],
    ) -> list[dict[str, float] | None]:
        request = {
            "sources": [{"lat": source["lat"], "lon": source["lon"]}],
            "targets": [{"lat": item["lat"], "lon": item["lon"]} for item in places],
            "costing": "pedestrian",
            "units": "miles",
        }
        cached = self.route_cache.get(request)
        if cached is not None:
            return list(cached)
        try:
            response = await self._request_with_retries(
                "POST",
                self.valhalla_base_url,
                json=request,
            )
            payload = response.json()
            matrix = payload.get("sources_to_targets") or []
            row = matrix[0] if isinstance(matrix, list) and matrix else []
            routes = [
                (
                    {"distance": float(item["distance"]), "time": float(item["time"])}
                    if isinstance(item, dict)
                    and item.get("distance") is not None
                    and item.get("time") is not None
                    else None
                )
                for item in row
            ]
        except (httpx.HTTPError, ValueError, KeyError, TypeError, GeoResearchError):
            routes = [None] * len(places)
        if len(routes) != len(places):
            routes = [None] * len(places)
        self.route_cache.put(request, routes)
        return routes

    async def _request_with_retries(
        self,
        method: str,
        url: str,
        *,
        attempts: int = 3,
        **kwargs: Any,
    ) -> httpx.Response:
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                response = await self.client.request(
                    method,
                    url,
                    timeout=self.timeout_seconds,
                    **kwargs,
                )
                response.raise_for_status()
                return response
            except httpx.HTTPError as exc:
                last_error = exc
                if attempt == attempts - 1:
                    break
                await asyncio.sleep(2**attempt)
        raise GeoResearchError(f"Geo request failed: {last_error}") from last_error
