import subprocess as sp
from datetime import datetime
import time 
import os
import csv

if not os.path.exists('power_log.csv'):
    with open('power_log.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["datetime", "power_readout"])
        file.close()

while True:
    try:
        nom_power = float(sp.getoutput('apcaccess status | grep NOMPOWER')[11:15])
        load_percent = float(sp.getoutput('apcaccess status | grep LOADPCT')[11:15])
        power = nom_power * load_percent / 100

        entry = [datetime.now(), power]

        with open('power_log.csv', 'a', newline='') as file:
            writer_object = csv.writer(file)
            writer_object.writerow(entry)
            file.close()

        time.sleep(60)
    except:
        print('PSU comms error')