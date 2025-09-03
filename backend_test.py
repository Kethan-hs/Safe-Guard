import requests
import sys
import json
from datetime import datetime

class SafeGuardAPITester:
    def __init__(self, base_url="https://safeguard-ai-4.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.failed_tests = []

    def run_test(self, name, method, endpoint, expected_status, data=None, timeout=30):
        """Run a single API test"""
        url = f"{self.api_url}/{endpoint}" if not endpoint.startswith('http') else endpoint
        headers = {'Content-Type': 'application/json'}

        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        print(f"   URL: {url}")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, timeout=timeout)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=headers, timeout=timeout)

            success = response.status_code == expected_status
            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code}")
                try:
                    response_data = response.json()
                    print(f"   Response keys: {list(response_data.keys()) if isinstance(response_data, dict) else 'Non-dict response'}")
                    return True, response_data
                except:
                    return True, response.text
            else:
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                print(f"   Response: {response.text[:200]}...")
                self.failed_tests.append(f"{name}: Expected {expected_status}, got {response.status_code}")
                return False, {}

        except requests.exceptions.Timeout:
            print(f"❌ Failed - Request timeout after {timeout}s")
            self.failed_tests.append(f"{name}: Request timeout")
            return False, {}
        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            self.failed_tests.append(f"{name}: {str(e)}")
            return False, {}

    def test_root_endpoint(self):
        """Test root API endpoint"""
        return self.run_test("Root API Endpoint", "GET", "", 200)

    def test_locations_endpoint(self):
        """Test locations endpoint"""
        success, response = self.run_test("Indian Locations", "GET", "locations", 200)
        if success and isinstance(response, dict):
            locations = response.get('locations', {})
            print(f"   Found {len(locations)} states")
            if 'Maharashtra' in locations:
                print(f"   Maharashtra cities: {locations['Maharashtra']}")
        return success, response

    def test_crime_types_endpoint(self):
        """Test crime types endpoint"""
        success, response = self.run_test("Crime Types", "GET", "crime-types", 200)
        if success and isinstance(response, dict):
            crime_types = response.get('crime_types', {})
            print(f"   Found {len(crime_types)} crime types")
            print(f"   Crime types: {list(crime_types.keys())}")
        return success, response

    def test_crime_data_get(self):
        """Test getting crime data"""
        success, response = self.run_test("Get Crime Data", "GET", "crime-data?limit=10", 200)
        if success and isinstance(response, list):
            print(f"   Retrieved {len(response)} crime records")
            if response:
                sample = response[0]
                print(f"   Sample crime: {sample.get('crime_type', 'N/A')} in {sample.get('city', 'N/A')}")
        return success, response

    def test_dashboard_stats(self):
        """Test dashboard statistics"""
        success, response = self.run_test("Dashboard Stats", "GET", "dashboard-stats", 200)
        if success and isinstance(response, dict):
            print(f"   Total crimes: {response.get('total_crimes', 0)}")
            print(f"   Crime by type entries: {len(response.get('crime_by_type', []))}")
            print(f"   Crime by state entries: {len(response.get('crime_by_state', []))}")
            print(f"   Recent crimes: {len(response.get('recent_crimes', []))}")
        return success, response

    def test_safety_analysis(self):
        """Test safety analysis for Mumbai location"""
        mumbai_location = {
            "latitude": 19.0760,
            "longitude": 72.8777,
            "radius": 5.0
        }
        success, response = self.run_test("Safety Analysis", "POST", "safety-analysis", 200, mumbai_location)
        if success and isinstance(response, dict):
            print(f"   Risk level: {response.get('risk_level', 'N/A')}")
            print(f"   Risk score: {response.get('risk_score', 0)}/100")
            print(f"   Recommendations: {len(response.get('recommendations', []))}")
            print(f"   Nearby crimes: {len(response.get('nearby_crimes', []))}")
        return success, response

    def test_crime_prediction(self):
        """Test ML crime prediction for Delhi location"""
        delhi_location = {
            "latitude": 28.7041,
            "longitude": 77.1025
        }
        success, response = self.run_test("Crime Prediction", "POST", "predict-crime", 200, delhi_location)
        if success and isinstance(response, dict):
            print(f"   Predicted risk score: {response.get('predicted_risk_score', 0)}/100")
            print(f"   Predicted crime types: {response.get('predicted_crime_types', [])}")
            print(f"   Confidence: {response.get('confidence', 0)}")
        return success, response

    def test_crime_heatmap(self):
        """Test crime heatmap data"""
        success, response = self.run_test("Crime Heatmap", "GET", "crime-heatmap?days=30", 200)
        if success and isinstance(response, dict):
            heatmap_data = response.get('heatmap_data', [])
            print(f"   Heatmap points: {len(heatmap_data)}")
        return success, response

    def test_crime_data_post(self):
        """Test submitting new crime data"""
        new_crime = {
            "location": {"lat": 19.0760, "lng": 72.8777},
            "crime_type": "theft",
            "severity": 6,
            "state": "Maharashtra",
            "city": "Mumbai",
            "description": "Test crime report for API testing"
        }
        success, response = self.run_test("Submit Crime Data", "POST", "crime-data", 200, new_crime)
        if success and isinstance(response, dict):
            print(f"   Created crime ID: {response.get('id', 'N/A')}")
            print(f"   Crime type: {response.get('crime_type', 'N/A')}")
        return success, response

def main():
    print("🚀 Starting Safe Guard API Testing...")
    print("=" * 60)
    
    tester = SafeGuardAPITester()
    
    # Run all tests
    test_results = []
    
    # Basic endpoint tests
    test_results.append(tester.test_root_endpoint())
    test_results.append(tester.test_locations_endpoint())
    test_results.append(tester.test_crime_types_endpoint())
    test_results.append(tester.test_crime_data_get())
    test_results.append(tester.test_dashboard_stats())
    
    # Advanced functionality tests
    test_results.append(tester.test_safety_analysis())
    test_results.append(tester.test_crime_prediction())
    test_results.append(tester.test_crime_heatmap())
    test_results.append(tester.test_crime_data_post())
    
    # Print final results
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"Tests run: {tester.tests_run}")
    print(f"Tests passed: {tester.tests_passed}")
    print(f"Tests failed: {tester.tests_run - tester.tests_passed}")
    print(f"Success rate: {(tester.tests_passed / tester.tests_run * 100):.1f}%")
    
    if tester.failed_tests:
        print("\n❌ FAILED TESTS:")
        for failure in tester.failed_tests:
            print(f"   • {failure}")
    else:
        print("\n✅ All tests passed!")
    
    return 0 if tester.tests_passed == tester.tests_run else 1

if __name__ == "__main__":
    sys.exit(main())