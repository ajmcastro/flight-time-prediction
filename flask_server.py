"""Simple Flask server."""
import json

import joblib
import numpy as np
import requests
from flask import Flask, abort, jsonify, request

from src import consts as const

# Reading configurations
with open('src/settings.json') as settings:
    props = json.load(settings)

imputed = props['dataset']['imputed']
target = props['dataset']['target']
cat_encoding = props['dataset']['cat_encoding']
fleet_type = props['dataset']['fleet_type']

model_name = props['model']['model_name']

# Loading model
model = joblib.load(const.MODEL_DIR / 'best_performers' /
                    f'{model_name}_{target}_{fleet_type}.sav')


# Init server app
app = Flask(__name__)

# Define endpoint route
@app.route('/airtime', methods=['POST'])
def make_air_time_prediction():
    data = request.get_json(force=True)

    # input error checking
    assert(all(isinstance(x, float) for x in [
        data['origin_air_temperature'], data['origin_wind_speed'],
        data['origin_visibility'], data['origin_cloud_height'],
        data['destination_air_temperature'], data['destination_wind_speed'],
        data['destination_visibility'], data['destination_cloud_height'],
        data['scheduled_rotation_time']
    ]))
    assert(all(isinstance(x, int) for x in [
        data['flight_number'], data['scheduled_block_time'], data['is_night'],
        data['scheduled_departure_date_year'], data['scheduled_departure_date_month'],
        data['scheduled_departure_date_day'], data['scheduled_departure_date_hour'],
        data['scheduled_departure_date_minute'], data['scheduled_arrival_date_year'],
        data['scheduled_arrival_date_month'], data['scheduled_arrival_date_day'],
        data['scheduled_arrival_date_hour'], data['scheduled_arrival_date_minute']
    ]))

    # convert json into numpy array
    predict_request = [
        data['flight_number'], data['tail_number'], data['aircraft_model'],
        data['fleet'], data['origin_airport'], data['destination_airport'],
        data['origin_air_temperature'], data['origin_wind_direction'],
        data['origin_wind_speed'], data['origin_visibility'],
        data['origin_cloud_coverage'], data['origin_cloud_height'],
        data['destination_air_temperature'], data['destination_wind_direction'],
        data['destination_wind_speed'], data['destination_visibility'],
        data['destination_cloud_coverage'], data['destination_cloud_height'],
        data['scheduled_block_time'], data['is_night'], data['scheduled_rotation_time'],
        data['prev_delay_code'], data['scheduled_departure_date_year'],
        data['scheduled_departure_date_month'], data['scheduled_departure_date_day'],
        data['scheduled_departure_date_hour'], data['scheduled_departure_date_minute'],
        data['scheduled_arrival_date_year'], data['scheduled_arrival_date_month'],
        data['scheduled_arrival_date_day'], data['scheduled_arrival_date_hour'],
        data['scheduled_arrival_date_minute']
    ]
    predict_request = np.array(predict_request).reshape(1, -1)

    # make inference with loaded model
    y_pred = model.predict(predict_request).flatten()

    print('y_pred -->' + str(y_pred))

    # return prediction
    output = [y_pred[0]]
    return jsonify(results=output)


if __name__ == '__main__':
    app.run(port=9000, debug=True)


# %%
# Run test for air time
import json
import requests

url = "http://127.0.0.1:9000/airtime"
data = json.dumps({
    'flight_number': 1231, 'tail_number': 'CSTNP',
    'aircraft_model': 'AIRBUS A320-211', 'fleet': 'NB',
    'origin_airport': 'DME', 'destination_airport': 'LIS',
    'origin_air_temperature': 12.0, 'origin_wind_direction': 'SE',
    'origin_wind_speed': 9.72, 'origin_visibility': 6.21,
    'origin_cloud_coverage': 'OVC', 'origin_cloud_height': 900.0,
    'destination_air_temperature': 13.0, 'destination_wind_direction': 'NW',
    'destination_wind_speed': 7.0, 'destination_visibility': 6.21,
    'destination_cloud_coverage': 'SCT', 'destination_cloud_height': 1800.0,
    'scheduled_block_time': 335, 'is_night': 1, 'scheduled_rotation_time': 155.0,
    'prev_delay_code': '1', 'scheduled_departure_date_year': 2016,
    'scheduled_departure_date_month': 4, 'scheduled_departure_date_day': 27,
    'scheduled_departure_date_hour': 2, 'scheduled_departure_date_minute': 20,
    'scheduled_arrival_date_year': 2016, 'scheduled_arrival_date_month': 4,
    'scheduled_arrival_date_day': 27, 'scheduled_arrival_date_hour': 7,
    'scheduled_arrival_date_minute': 55
})
r = requests.post(url, data)

print(r.json())
# true value is 316. Result value is 322

