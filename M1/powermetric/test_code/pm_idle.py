# this part is used to calulate the idle power consumption of the system

import subprocess
import time
from powermetric.test_code.pm_calculate import txt_data_process
from powermetric.test_code.pm_calculate import run_powermetrics

'''
This code is to calcule the idle situation of the system for 2 minutes,
and then calculate the average power consumption of the system.
'''

def main():
    # create the file name: powermetric/pm_idle.txt
    file_path = "powermetric/pm_idle.txt"

    # run powermetrics
    idle_process = run_powermetrics(file_path)

    # wait for 2 minutes
    time.sleep(120)

    # kill the powermetrics process
    idle_process.terminate()
    idle_process.wait()

    # calculate the average power consumptions
    idle_energy_consumption, idle_list = txt_data_process(file_path)
    avg_idel_energy_consumption = idle_energy_consumption / 120
    print('The average idle energy consuption per second is ', avg_idel_energy_consumption, 'kWh')


if __name__ == "__main__":
    main()