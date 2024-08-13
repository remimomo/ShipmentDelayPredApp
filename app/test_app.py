import unittest
import json
from app import app

class FlaskTest(unittest.TestCase):
    # Test case for the prediction route
    def test_predict(self):
        #with self.app as c:
        tester = app.test_client(self)
        #tester = self.app
        data = {
            "shipment_number": "12345",
            "vehicle_size": "L",
            "vehicle_build_up": "Closed",
            "first_collection_latitude": 51.509865,
            "first_collection_longitude": -0.118092,
            "last_delivery_latitude": 52.486244,
            "last_delivery_longitude": -1.890401,
            "last_delivery_schedule_earliest": "2023-11-01T08:00:00",
            "last_delivery_schedule_latest": "2023-11-01T12:00:00",
            "shipper_id": "Shipper_01",
            "carrier_id": "Carrier_01"
        }
        # Send POST request to the root route with form data
        response = tester.post('/', data=data)
        
        # Check that the response status code is 200 (OK)
        self.assertEqual(response.status_code, 200)
        
        # Check that the response contains the expected result
        self.assertIn(b'Prediction Result:', response.data)

if __name__ == '__main__':
    unittest.main()
