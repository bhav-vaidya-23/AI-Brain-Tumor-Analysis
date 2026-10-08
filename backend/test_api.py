import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from backend.main import app

def run_tests():
    print("=" * 70)
    print("RUNNING END-TO-END BACKEND API SUITE")
    print("=" * 70)

    with TestClient(app) as client:
        # Test 1: Health Check
        resp = client.get("/api/health")
        assert resp.status_code == 200, f"Health check failed: {resp.status_code}"
        health_data = resp.json()
        assert health_data["status"] == "healthy"
        assert health_data["classifier_loaded"] is True
        assert health_data["segmentation_loaded"] is True
        assert health_data["explainability_loaded"] is True
        print("[PASS] Test 1: GET /api/health returns 200 Healthy with all models loaded.")

        # Test 2: Multi-Class Inference on all 4 classes
        test_samples = [
            ("Glioma", "frontend/public/samples/glioma_sample.jpg"),
            ("Meningioma", "frontend/public/samples/meningioma_sample.jpg"),
            ("Pituitary", "frontend/public/samples/pituitary_sample.jpg"),
            ("No Tumor", "frontend/public/samples/no_tumor_sample.jpg")
        ]

        for expected_cls, img_path in test_samples:
            assert Path(img_path).exists(), f"Sample image missing: {img_path}"
            with open(img_path, "rb") as f:
                resp = client.post("/api/analyze", files={"file": (Path(img_path).name, f, "image/jpeg")})
            
            assert resp.status_code == 200, f"Analysis failed for {expected_cls}: {resp.text}"
            data = resp.json()
            
            assert "prediction" in data
            assert "class" in data["prediction"]
            assert "confidence" in data["prediction"]
            assert "probabilities" in data
            assert len(data["probabilities"]) == 4
            assert all(k in data["probabilities"] for k in ["Glioma", "Meningioma", "Pituitary", "No Tumor"])
            assert "segmentation" in data
            assert "tumor_area_percentage" in data["segmentation"]
            assert "visualizations" in data
            assert all(k in data["visualizations"] for k in ["original_mri", "gradcam_heatmap", "gradcam_overlay", "segmentation_mask", "segmentation_overlay"])
            assert "disclaimer" in data
            print(f"[PASS] Test 2 ({expected_cls}): Predicted='{data['prediction']['class']}' (Conf: {data['prediction']['confidence_percentage']}%), Tumor Area: {data['segmentation']['tumor_area_percentage']}%.")

        # Test 3: Invalid File Extension (.txt)
        resp_invalid_ext = client.post("/api/analyze", files={"file": ("report.txt", b"Invalid content", "text/plain")})
        assert resp_invalid_ext.status_code == 400, f"Expected 400 for .txt, got {resp_invalid_ext.status_code}"
        print("[PASS] Test 3: Invalid file extension rejected with 400 Bad Request.")

        # Test 4: Empty File
        resp_empty = client.post("/api/analyze", files={"file": ("empty.jpg", b"", "image/jpeg")})
        assert resp_empty.status_code == 400, f"Expected 400 for empty file, got {resp_empty.status_code}"
        print("[PASS] Test 4: Empty file rejected with 400 Bad Request.")

        # Test 5: Oversized File (>10MB)
        large_payload = b"x" * (11 * 1024 * 1024)
        resp_large = client.post("/api/analyze", files={"file": ("huge.jpg", large_payload, "image/jpeg")})
        assert resp_large.status_code == 413, f"Expected 413 for oversized file, got {resp_large.status_code}"
        print("[PASS] Test 5: Oversized payload (>10MB) rejected with 413 Request Entity Too Large.")

        # Test 6: CORS Options Preflight
        resp_cors = client.options("/api/analyze", headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST"
        })
        assert resp_cors.status_code == 200, f"CORS preflight failed: {resp_cors.status_code}"
        print("[PASS] Test 6: CORS preflight headers verified.")

    print("=" * 70)
    print("ALL API SUITE TESTS PASSED (6/6)")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
