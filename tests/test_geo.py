import json
from collections import Counter
from pathlib import Path

import httpx
import pytest

from browsecomp250.geo import GeoResearchClient


@pytest.mark.asyncio
async def test_geo_research_ranks_shared_candidate_and_reuses_cache(tmp_path: Path) -> None:
    requests: Counter[str] = Counter()

    def handler(request: httpx.Request) -> httpx.Response:
        requests[request.url.path] += 1
        if request.url.path == "/api/":
            query = request.url.params["q"]
            latitude = 10.0 if query == "Anchor A" else 20.0
            return httpx.Response(
                200,
                json={
                    "features": [
                        {
                            "geometry": {"coordinates": [-70.0, latitude]},
                            "properties": {"name": query, "country": "Testland"},
                        }
                    ]
                },
            )
        if request.url.path == "/overpass":
            return httpx.Response(
                200,
                json={
                    "elements": [
                        {
                            "type": "node",
                            "id": 42,
                            "lat": 15.0,
                            "lon": -70.0,
                            "tags": {
                                "name": "Shared Cafe",
                                "brand": "Shared Cafe",
                                "amenity": "restaurant",
                            },
                        }
                    ]
                },
            )
        if request.url.path == "/matrix":
            payload = json.loads(request.content)
            assert "source" not in payload
            assert len(payload["sources"]) == 1
            distance = 1.0 if payload["sources"][0]["lat"] == 10.0 else 2.0
            return httpx.Response(
                200,
                json={"sources_to_targets": [[{"distance": distance, "time": 600}]]},
            )
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = GeoResearchClient(
        tmp_path / "geo.sqlite3",
        client=http,
        photon_base_url="https://geo.test/api/",
        overpass_base_url="https://geo.test/overpass",
        valhalla_base_url="https://geo.test/matrix",
    )
    anchors = [
        {"query": "Anchor A", "radius_m": 5000, "expected_distance_miles": 1.0},
        {"query": "Anchor B", "radius_m": 5000, "expected_distance_miles": 2.0},
    ]

    first = await client.explore(anchors, category="restaurant")
    second = await client.explore(anchors, category="restaurant")

    assert first == second
    assert requests == {"/api/": 2, "/overpass": 2, "/matrix": 2}
    assert first["shared_entities"][0] == {
        "label": "Shared Cafe",
        "anchor_count": 2,
        "expected_distance_anchor_count": 2,
        "total_distance_error_miles": 0.0,
        "matches": [
            {
                "anchor_index": 0,
                "name": "Shared Cafe",
                "brand": "Shared Cafe",
                "straight_line_miles": 345.467,
                "walking_miles": 1.0,
                "distance_error_miles": 0.0,
            },
            {
                "anchor_index": 1,
                "name": "Shared Cafe",
                "brand": "Shared Cafe",
                "straight_line_miles": 345.467,
                "walking_miles": 2.0,
                "distance_error_miles": 0.0,
            },
        ],
    }
    await http.aclose()


@pytest.mark.asyncio
async def test_geo_research_rotates_overpass_endpoint_after_throttle(tmp_path: Path) -> None:
    requests: Counter[str] = Counter()

    def handler(request: httpx.Request) -> httpx.Response:
        requests[request.url.path] += 1
        if request.url.path == "/primary":
            return httpx.Response(429, text="rate limited")
        if request.url.path == "/fallback":
            return httpx.Response(200, json={"elements": []})
        raise AssertionError(f"Unexpected request: {request.url}")

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = GeoResearchClient(
        tmp_path / "geo.sqlite3",
        client=http,
        overpass_base_url="https://geo.test/primary",
        overpass_fallback_urls=("https://geo.test/fallback",),
    )

    places = await client._nearby_places(1.0, 2.0, radius_m=1000, category="restaurant")

    assert places == []
    assert requests == {"/primary": 1, "/fallback": 1}
    await http.aclose()
