# Description: Read the output file of powermetric and extract the power value

list_power = []
with open('powermetric/output_test1.txt', 'r') as f:
    for line in f:
        if 'Combined Power' in line:
            power_value = line.split(':')[1].strip()
            print(power_value)

            # Remove the unit
            power_value = power_value.replace('mW', '')

            # Convert to integer
            power_value = int(power_value)
            list_power.append(power_value)

print(list_power)

