import datetime

# Get the current time
now = datetime.datetime.now()

# Format the time as a string that can be used as a file name
time_str = now.strftime("pm_output" + " "+ "%Y-%m-%d_%H-%M")

# Create the file name by appending the time string to the file path
file_path = "/Users/dtjgp/Learning/Thesis/GreenAI/powermetric/" + time_str + ".txt"

# Open the file and write the current time to it
with open(file_path, "w") as f:
    f.write(str(now))
