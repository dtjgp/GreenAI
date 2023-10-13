# this code is to test automatically start powermetric when using
import subprocess
import time
# import datetime

def run_powermetrics(file_path):

    # Define the command as a list of arguments
    cmd = ["sudo", "powermetrics", "-i", '1000', "--samplers", "cpu_power,gpu_power", "-o", file_path]

    # "-n", str(count),
    
    # # Run the command and retrieve the output
    # try:
    #     output = subprocess.check_output(cmd, text=True)
    #     return output
    # except subprocess.CalledProcessError as e:
    #     print(f"Error occurred: {str(e)}")
    #     return None
    process = subprocess.Popen(cmd)
    return process

# Example usage
if __name__ == "__main__":

    # interval = 1000  # 1 second
    # count = 5  # Retrieve 5 samples
    
    # Fetch powermetrics data
    # data = run_powermetrics(interval, count)
    
    # # Do something with the data
    # if data is not None:
    #     print(data)


    # Create the file name by appending the time string to the file path
    file_path = "powermetric/output_test5.txt"
    print(file_path)

    # run powermetrics
    powermetrics_process = run_powermetrics(file_path)

    # set a time sleep to stop powermetrics
    time.sleep(10)

    # stop powermetrics
    powermetrics_process.terminate()
    powermetrics_process.wait()

