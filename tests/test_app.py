import unittest
from app import app

class FlaskTestCase(unittest.TestCase):
    # Ensure that the Flask app is running
    def test_index(self):
        tester = app.test_client(self)
        response = tester.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Shipment Delay Predictor', response.data)

    # Ensure the prediction route works and returns a response
    def test_prediction(self):
        tester = app.test_client(self)
        response = tester.post('/predict', data=dict(
            vehicle_size="Tractor Unit",
            vehicle_build_up="Curtain-Side 13.6M Trailer",
            first_collection_time="2024-01-11T15:03:00",
            last_delivery_time="2024-01-12T15:03:00",
            destination_lat="51.5074",
            destination_lon="-0.1278"
        ))
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Prediction:', response.data)

if __name__ == '__main__':
    unittest.main()
