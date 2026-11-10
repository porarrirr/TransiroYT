import hashlib

import httpx
from transiroyt.app import create_app
from transiroyt.config import Settings


async def test_health_identifies_storage_and_restart_without_disclosing_secrets(tmp_path):
    identities = []
    for directory in [tmp_path / "development", tmp_path / "consumer", tmp_path / "development"]:
        app = create_app(Settings(data=directory, token="private-connection-token"))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url="http://127.0.0.1") as client:
            response = await client.get("/health")
            assert response.status_code == 200
            identity = response.json()
            assert identity["data_id"] == hashlib.sha256(str(directory.resolve()).encode()).hexdigest()
            assert "private-connection-token" not in response.text
            assert str(tmp_path) not in response.text
            assert (await client.get("/health")).json() == identity
            identities.append(identity)
    assert identities[0]["data_id"] == identities[2]["data_id"] != identities[1]["data_id"]
    assert len({identity["instance_id"] for identity in identities}) == 3
