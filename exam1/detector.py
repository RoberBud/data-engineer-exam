import os
import csv
import shutil
import zipfile

from algorithm import Detector
from plot import plot


def detect(path, detector):
    pred = []
    vals = []
    with open(path, newline='') as csvfile:
        rows = csv.DictReader(csvfile)
        
        for row in rows:
            val = row.get('value')
            vals.append(float(val))
        pred = detector.fit_predict(path)
    return vals, pred


if __name__ == '__main__':

    if not os.path.exists('result'):
        os.mkdir('result')

    zip_path = 'data/time-series.zip'
    archive = zipfile.ZipFile(zip_path, 'r')

    for i in range(1, 68):
        detector = Detector()

        name = f'time-series/real_{i}.csv'
        path = archive.extract(member=name)
        vals, pred = detect(path, detector)
        plot(name, vals, pred)

    shutil.rmtree('time-series')
