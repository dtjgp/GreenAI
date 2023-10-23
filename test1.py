'''
This is a test python file that used to test the code.
'''
# a = [98, 'bottle of neck', ['on', 'the', 'wall'], 99]
# a[1:2] = [ 'bottle', 'of', 'neck']
# print(a)

energy_consumption = 0.005205
# change the kWh to J
energy_consumption = energy_consumption * 3600000
print(energy_consumption)
assumed_time = energy_consumption / 6 # the power is 5W
print(assumed_time)

time_cost = 1644
energy_consumption_RAM = time_cost * 6 / 3600000
print(energy_consumption_RAM)

a = energy_consumption_RAM + 0.00441
print(a)