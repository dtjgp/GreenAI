# the code is to create a class for data center
# the class has the following attributes:
# 1. the location of the data center
# 2. the capacity of the data center
# 3. the energy consumption of the data center
# 4. the energy efficiency of the data center

class DataCenter:
    def __init__(self, location, capacity, energy_consumption, energy_efficiency):
        self.location = None
        self.capacity = capacity
        self.energy_consumption = energy_consumption
        self.energy_efficiency = energy_efficiency