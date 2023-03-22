import dill
import os
import json
from datetime import datetime
import pandas as pd


def predict():

    path = os.environ.get('PROJECT_PATH', '..')

    folder = os.listdir(f'{path}/data/models')
    files = [os.path.join(f'{path}/data/models', file) for file in folder if os.path.isfile(os.path.join(f'{path}/data/models', file))]
    pickle = max(files, key=os.path.getctime)
    with open(pickle, 'rb') as file:
        model = dill.load(file)

    res = pd.DataFrame(columns=['car_id','pred'])
    for filename in os.listdir(f'{path}/data/test/'):
        if filename.endswith(".json"):
            with open(f'{path}/data/test/{filename}', "r") as my_file:
                current_file = my_file.read()
            current_file_dict = json.loads(current_file)
            df = pd.DataFrame.from_dict([current_file_dict])
            y = model.predict(df)
            res_dict = {'car_id': filename.split('.')[0], 'pred': y[0]}
            res = res.append(res_dict,ignore_index=True)

    res.to_csv(f'{path}/data/predictions/Prediction_{datetime.now().strftime("%Y%m%d%H%M")}.csv')


if __name__ == '__main__':
    predict()
