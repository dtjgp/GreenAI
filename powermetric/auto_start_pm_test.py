# this code is to test automatically start powermetric when using
import subprocess
import datetime

def run_powermetrics(interval, count):
    """
    Run powermetrics and retrieve the output.

    :param interval: Sampling interval in milliseconds.
    :param count: Number of samples to retrieve.
    :return: The output from powermetrics.
    """

    # Get the current time
    now = datetime.datetime.now()

    # Format the time as a string that can be used as a file name
    time_str = now.strftime("pm_output" + " "+ "%Y-%m-%d_%H-%M")

    # Create the file name by appending the time string to the file path
    file_path = "/Users/dtjgp/Learning/Thesis/GreenAI/powermetric/" + time_str + ".txt"
    print(file_path)

    # Define the command as a list of arguments
    cmd = ["sudo", "powermetrics", "-i", str(interval), "--samplers", "cpu_power,gpu_power", "-a", "-o", file_path]
    
    # Run the command and retrieve the output
    try:
        output = subprocess.check_output(cmd, text=True)
        return output
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {str(e)}")
        return None

# Example usage
if __name__ == "__main__":

    interval = 1000  # 1 second
    count = 5  # Retrieve 5 samples
    
    # Fetch powermetrics data
    data = run_powermetrics(interval, count)
    
    # Do something with the data
    if data is not None:
        print(data)

