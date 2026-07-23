import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture
def client():
    """Fixture providing a TestClient configured with FastAPI lifespan startup/shutdown events."""
    with TestClient(app) as c:
        yield c

# -------------------------------------------------------------
# 1. MECHANICAL TESTS (4 tests)
# -------------------------------------------------------------

def test_health_check_returns_200(client):
    """Mechanical Test 1: GET /health returns HTTP 200 and model loaded status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert "algorithm" in data
    assert "optimal_threshold" in data


def test_valid_prediction_returns_200_and_schema(client):
    """Mechanical Test 2: Valid payload returns HTTP 200 and correct response schema."""
    valid_payload = {
        "Warehouse_block": "F",
        "Mode_of_Shipment": "Flight",
        "Customer_care_calls": 4,
        "Customer_rating": 3,
        "Cost_of_the_Product": 210.0,
        "Prior_purchases": 3,
        "Product_importance": "high",
        "Gender": "F",
        "Discount_offered": 5.0,
        "Weight_in_gms": 4500.0
    }
    response = client.post("/predict-keterlambatan", json=valid_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["prediction_label"] in ["terlambat", "tepat_waktu"]
    assert isinstance(data["is_delayed"], bool)
    assert 0.0 <= data["delay_probability"] <= 1.0
    assert "applied_threshold" in data
    assert "recommendation" in data


def test_missing_field_returns_422(client):
    """Mechanical Test 3: Missing required field (Discount_offered) returns HTTP 422."""
    invalid_payload = {
        "Warehouse_block": "A",
        "Mode_of_Shipment": "Ship",
        "Customer_care_calls": 3,
        "Customer_rating": 4,
        "Cost_of_the_Product": 150.0,
        "Prior_purchases": 2,
        "Product_importance": "low",
        "Gender": "M"
        # Discount_offered and Weight_in_gms missing
    }
    response = client.post("/predict-keterlambatan", json=invalid_payload)
    assert response.status_code == 422


def test_invalid_enum_returns_422(client):
    """Mechanical Test 4: Invalid enum value ('Teleportation' / 'Z') returns HTTP 422."""
    invalid_enum_payload = {
        "Warehouse_block": "Z",  # Invalid warehouse block
        "Mode_of_Shipment": "Teleportation",  # Invalid mode of shipment
        "Customer_care_calls": 3,
        "Customer_rating": 4,
        "Cost_of_the_Product": 150.0,
        "Prior_purchases": 2,
        "Product_importance": "low",
        "Gender": "M",
        "Discount_offered": 10.0,
        "Weight_in_gms": 3000.0
    }
    response = client.post("/predict-keterlambatan", json=invalid_enum_payload)
    assert response.status_code == 422


# -------------------------------------------------------------
# 2. BEHAVIORAL TESTS (2 tests)
# -------------------------------------------------------------

def test_behavioral_high_risk_vs_low_risk_package(client):
    """
    Behavioral Test 1 (Kasus A):
    High-risk package (low discount, high weight, many care calls) must have
    a HIGHER delay probability than a low-risk package (high discount, lower weight, few calls).
    """
    high_risk_payload = {
        "Warehouse_block": "F",
        "Mode_of_Shipment": "Ship",
        "Customer_care_calls": 6,
        "Customer_rating": 1,
        "Cost_of_the_Product": 280.0,
        "Prior_purchases": 6,
        "Product_importance": "low",
        "Gender": "M",
        "Discount_offered": 2.0,   # Very low discount
        "Weight_in_gms": 6500.0    # Heavy package
    }
    
    low_risk_payload = {
        "Warehouse_block": "F",
        "Mode_of_Shipment": "Ship",
        "Customer_care_calls": 2,
        "Customer_rating": 5,
        "Cost_of_the_Product": 120.0,
        "Prior_purchases": 2,
        "Product_importance": "high",
        "Gender": "M",
        "Discount_offered": 45.0,  # High discount (incentivized express handling)
        "Weight_in_gms": 2000.0    # Light package
    }
    
    resp_high = client.post("/predict-keterlambatan", json=high_risk_payload)
    resp_low = client.post("/predict-keterlambatan", json=low_risk_payload)
    
    assert resp_high.status_code == 200
    assert resp_low.status_code == 200
    
    prob_high = resp_high.json()["delay_probability"]
    prob_low = resp_low.json()["delay_probability"]
    
    print(f"\n[BEHAVIORAL TEST 1 RESULTS]: High Risk Prob = {prob_high}, Low Risk Prob = {prob_low}")
    assert prob_high > prob_low, (
        f"Behavioral failure: Expected high risk prob ({prob_high}) > low risk prob ({prob_low})"
    )


def test_behavioral_discount_sensitivity(client):
    """
    Behavioral Test 2:
    Decreasing the discount offered on an otherwise identical package
    should increase or maintain high delay risk probability.
    """
    base_high_discount = {
        "Warehouse_block": "A",
        "Mode_of_Shipment": "Flight",
        "Customer_care_calls": 3,
        "Customer_rating": 4,
        "Cost_of_the_Product": 200.0,
        "Prior_purchases": 3,
        "Product_importance": "medium",
        "Gender": "F",
        "Discount_offered": 40.0,
        "Weight_in_gms": 3500.0
    }
    
    base_low_discount = base_high_discount.copy()
    base_low_discount["Discount_offered"] = 3.0  # Reduced discount
    
    resp_high_disc = client.post("/predict-keterlambatan", json=base_high_discount)
    resp_low_disc = client.post("/predict-keterlambatan", json=base_low_discount)
    
    assert resp_high_disc.status_code == 200
    assert resp_low_disc.status_code == 200
    
    prob_high_disc = resp_high_disc.json()["delay_probability"]
    prob_low_disc = resp_low_disc.json()["delay_probability"]
    
    print(f"\n[BEHAVIORAL TEST 2 RESULTS]: High Disc Prob = {prob_high_disc}, Low Disc Prob = {prob_low_disc}")
    assert prob_low_disc >= prob_high_disc, (
        f"Behavioral sensitivity failure: Low discount prob ({prob_low_disc}) should be >= high discount prob ({prob_high_disc})"
    )
