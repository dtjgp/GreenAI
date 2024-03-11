# 计算参数
macs_per_image = 1.75e9  # MACs per image
bits_per_mac = 64  # 32 bits for multiplication, 32 bits for addition
energy_per_bit = 1.757e-13  # Joules per bit operation
images_per_epoch = 6e4  # Number of images per epoch
epochs = 10  # Number of epochs

# print(macs_per_image)
# 计算每张图片的位操作数
bits_per_image = macs_per_image * bits_per_mac

# 计算每个epoch的总位操作数
total_bits_per_epoch = bits_per_image * images_per_epoch

# 计算总电能消耗 (Joules)
total_energy = total_bits_per_epoch * energy_per_bit * epochs

# 将能量从焦耳转换为千瓦时 (1 Joule = 2.77778e-7 kWh)
total_energy_kWh = total_energy * 2.77778e-7

print(total_energy, total_energy_kWh)
