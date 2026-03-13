"""
Unit tests for the Production FastAPI (api/).

Run with:  pytest tests/test_api.py -v
Requires:  pip install httpx pytest

Note: These tests use the FastAPI TestClient and do NOT require a running server.
Models are loaded via ModelRegistry at startup — if pkl files are missing,
the registry gracefully returns None and endpoints fall back to a 503 response.
"""

import pytest
from fastapi.testclient import TestClient

# Import the app — if api/main.py imports fail (missing deps), all tests are skipped
try:
    from api.main import app
    API_AVAILABLE = True
except Exception:
    API_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not API_AVAILABLE,
    reason="api/main.py could not be imported (missing dependency or syntax error)",
)

client = TestClient(app, raise_server_exceptions=False)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

VALID_SMILES = "c1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1"  # Imatinib
KNOWN_EGFR_SMILES = "C#Cc1cccc(NC2=NC=NC3=CC(OCCO)=C(OCC)C=C23)c1"  # Erlotinib


# ─────────────────────────────────────────────────────────────────────────────
# Smoke / health endpoint tests
# ─────────────────────────────────────────────────────────────────────────────

def test_root_returns_200():
    response = client.get("/")
    assert response.status_code == 200


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert "status" in body


def test_docs_endpoint_reachable():
    """Swagger UI docs should be served."""
    response = client.get("/docs")
    assert response.status_code == 200


# ─────────────────────────────────────────────────────────────────────────────
# /predict/toxicity
# ─────────────────────────────────────────────────────────────────────────────

class TestToxicityEndpoint:
    def test_valid_smiles_returns_200_or_503(self):
        payload = {"smiles": VALID_SMILES}
        response = client.post("/predict/toxicity", json=payload)
        assert response.status_code in (200, 503), (
            f"Unexpected status {response.status_code}: {response.text}"
        )

    def test_200_response_has_correct_shape(self):
        payload = {"smiles": VALID_SMILES}
        response = client.post("/predict/toxicity", json=payload)
        if response.status_code == 503:
            pytest.skip("Model not loaded — skipping shape assertion")
        body = response.json()
        assert "smiles" in body
        assert "predictions" in body
        assert isinstance(body["predictions"], dict)

    def test_invalid_smiles_returns_422(self):
        payload = {"smiles": "NOTASMILES!!!###"}
        response = client.post("/predict/toxicity", json=payload)
        assert response.status_code == 422

    def test_empty_smiles_returns_422(self):
        payload = {"smiles": ""}
        response = client.post("/predict/toxicity", json=payload)
        assert response.status_code == 422

    def test_missing_smiles_field_returns_422(self):
        response = client.post("/predict/toxicity", json={})
        assert response.status_code == 422

    def test_egfr_inhibitor_smiles_accepted(self):
        payload = {"smiles": KNOWN_EGFR_SMILES}
        response = client.post("/predict/toxicity", json=payload)
        assert response.status_code in (200, 503)


# ─────────────────────────────────────────────────────────────────────────────
# /predict/admet
# ─────────────────────────────────────────────────────────────────────────────

class TestADMETEndpoint:
    def test_valid_smiles_accepted(self):
        payload = {"smiles": VALID_SMILES}
        response = client.post("/predict/admet", json=payload)
        assert response.status_code in (200, 503)

    def test_200_response_contains_admet_fields(self):
        payload = {"smiles": VALID_SMILES}
        response = client.post("/predict/admet", json=payload)
        if response.status_code == 503:
            pytest.skip("Model not loaded")
        body = response.json()
        assert "solubility_esol" in body or "properties" in body

    def test_invalid_smiles_rejected(self):
        payload = {"smiles": "XYZ_INVALID"}
        response = client.post("/predict/admet", json=payload)
        assert response.status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# /predict/activity
# ─────────────────────────────────────────────────────────────────────────────

class TestActivityEndpoint:
    def test_valid_smiles_accepted(self):
        payload = {"smiles": KNOWN_EGFR_SMILES}
        response = client.post("/predict/activity", json=payload)
        assert response.status_code in (200, 503)

    def test_200_response_has_probability(self):
        payload = {"smiles": KNOWN_EGFR_SMILES}
        response = client.post("/predict/activity", json=payload)
        if response.status_code == 503:
            pytest.skip("Model not loaded")
        body = response.json()
        assert "active_probability" in body or "probability" in body

    def test_probability_in_valid_range(self):
        payload = {"smiles": KNOWN_EGFR_SMILES}
        response = client.post("/predict/activity", json=payload)
        if response.status_code == 503:
            pytest.skip("Model not loaded")
        body = response.json()
        prob = body.get("active_probability", body.get("probability", None))
        if prob is not None:
            assert 0.0 <= float(prob) <= 1.0

    def test_invalid_smiles_rejected(self):
        payload = {"smiles": "NOT_A_MOLECULE"}
        response = client.post("/predict/activity", json=payload)
        assert response.status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# /predict/full (combined endpoint)
# ─────────────────────────────────────────────────────────────────────────────

class TestFullPredictionEndpoint:
    def test_valid_smiles_accepted(self):
        payload = {"smiles": VALID_SMILES}
        response = client.post("/predict/full", json=payload)
        assert response.status_code in (200, 503)

    def test_full_response_structure(self):
        payload = {"smiles": VALID_SMILES}
        response = client.post("/predict/full", json=payload)
        if response.status_code == 503:
            pytest.skip("Model not loaded")
        body = response.json()
        assert "smiles" in body
        # Should contain results from all three prediction types
        assert any(k in body for k in ("toxicity", "admet", "activity", "predictions"))

    def test_invalid_smiles_rejected(self):
        payload = {"smiles": "GARBAGE_INPUT"}
        response = client.post("/predict/full", json=payload)
        assert response.status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# /batch/predict
# ─────────────────────────────────────────────────────────────────────────────

class TestBatchEndpoint:
    BATCH_SMILES = [
        "c1ccccc1",           # Benzene
        VALID_SMILES,         # Imatinib
        KNOWN_EGFR_SMILES,    # Erlotinib
    ]

    def test_batch_endpoint_accepts_list(self):
        payload = {"smiles_list": self.BATCH_SMILES}
        response = client.post("/batch/predict", json=payload)
        assert response.status_code in (200, 503)

    def test_batch_response_is_list_or_dict(self):
        payload = {"smiles_list": self.BATCH_SMILES}
        response = client.post("/batch/predict", json=payload)
        if response.status_code == 503:
            pytest.skip("Model not loaded")
        body = response.json()
        assert isinstance(body, (list, dict))

    def test_batch_response_length_matches_input(self):
        payload = {"smiles_list": self.BATCH_SMILES}
        response = client.post("/batch/predict", json=payload)
        if response.status_code == 503:
            pytest.skip("Model not loaded")
        body = response.json()
        if isinstance(body, list):
            assert len(body) == len(self.BATCH_SMILES)

    def test_empty_batch_handled(self):
        payload = {"smiles_list": []}
        response = client.post("/batch/predict", json=payload)
        assert response.status_code in (200, 422)

    def test_batch_with_mixed_valid_invalid(self):
        payload = {"smiles_list": [VALID_SMILES, "INVALID_SMILES_XYZ"]}
        response = client.post("/batch/predict", json=payload)
        # Should either reject the batch (422) or return partial results (200)
        assert response.status_code in (200, 422, 503)


# ─────────────────────────────────────────────────────────────────────────────
# Request validation edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestRequestValidation:
    ENDPOINTS = ["/predict/toxicity", "/predict/admet", "/predict/activity", "/predict/full"]

    @pytest.mark.parametrize("endpoint", ENDPOINTS)
    def test_wrong_content_type(self, endpoint):
        """Sending plain text instead of JSON should return 422 or 415."""
        response = client.post(endpoint, content="not json", headers={"Content-Type": "text/plain"})
        assert response.status_code in (415, 422, 400)

    @pytest.mark.parametrize("endpoint", ENDPOINTS)
    def test_null_smiles_rejected(self, endpoint):
        response = client.post(endpoint, json={"smiles": None})
        assert response.status_code == 422

    @pytest.mark.parametrize("endpoint", ENDPOINTS)
    def test_numeric_smiles_rejected(self, endpoint):
        response = client.post(endpoint, json={"smiles": 12345})
        assert response.status_code == 422

    def test_known_safe_molecule_benzene(self):
        """Benzene is a valid SMILES and should be accepted by all predict endpoints."""
        for endpoint in self.ENDPOINTS:
            response = client.post(endpoint, json={"smiles": "c1ccccc1"})
            assert response.status_code in (200, 503), (
                f"Endpoint {endpoint} returned {response.status_code} for benzene"
            )


# ─────────────────────────────────────────────────────────────────────────────
# GET /molecule/info
# ─────────────────────────────────────────────────────────────────────────────

class TestMoleculeInfoEndpoint:
    def test_valid_smiles_returns_200(self):
        response = client.get("/molecule/info", params={"smiles": VALID_SMILES})
        assert response.status_code == 200

    def test_response_has_required_fields(self):
        response = client.get("/molecule/info", params={"smiles": VALID_SMILES})
        assert response.status_code == 200
        body = response.json()
        for field in ("smiles", "valid", "molecular_weight", "logP", "QED",
                      "lipinski_pass", "molecular_formula"):
            assert field in body, f"Missing field: {field}"

    def test_valid_flag_is_true_for_valid_smiles(self):
        response = client.get("/molecule/info", params={"smiles": "c1ccccc1"})
        body = response.json()
        assert body["valid"] is True

    def test_invalid_smiles_returns_200_with_valid_false(self):
        """GET endpoint returns 200 but valid=False for bad SMILES."""
        response = client.get("/molecule/info", params={"smiles": "NOTASMILES!!!"})
        assert response.status_code == 200
        body = response.json()
        assert body["valid"] is False
        assert body.get("error") is not None

    def test_aspirin_molecular_formula(self):
        response = client.get("/molecule/info", params={"smiles": "CC(=O)Oc1ccccc1C(=O)O"})
        body = response.json()
        assert body["molecular_formula"] == "C9H8O4"

    def test_aspirin_lipinski_pass(self):
        response = client.get("/molecule/info", params={"smiles": "CC(=O)Oc1ccccc1C(=O)O"})
        body = response.json()
        assert body["lipinski_pass"] is True
        assert body["lipinski_violations"] == []

    def test_qed_in_range(self):
        response = client.get("/molecule/info", params={"smiles": VALID_SMILES})
        body = response.json()
        assert 0.0 <= body["QED"] <= 1.0

    def test_inchikey_returned(self):
        response = client.get("/molecule/info", params={"smiles": "CC(=O)Oc1ccccc1C(=O)O"})
        body = response.json()
        # InChIKey is 27-char string in standard format
        ikey = body.get("InChIKey")
        if ikey is not None:
            assert len(ikey) == 27
