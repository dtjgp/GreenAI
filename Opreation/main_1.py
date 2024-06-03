from instance_gen.process_gen import generate_instance
from solve import *
import json
import pandas as pd


# process_operations = generate_instance()
# contributions = [1/32, 1/28, 1/26, 1/25, 1/24, 1/23, 1/22]

# start_hour = 98
# Hours = 26
# end_hour = start_hour + Hours
# solar_data = pd.read_csv('/Users/jingsichen/Politecnico Di Torino Studenti Dropbox/Jingsi Chen/Mac/Desktop/Green AI/AI/solve/pvwatts_hourly.csv')
# G_max_solar = solar_data['AC System Output (W)'][1:169].tolist()
# GPU_power_options = [100, 125, 150, 175, 200, 225, 250]
# electricity_price = [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.45, 0.45, 0.45, 0.45, 0.45,
#                      0.45, 0.45, 0.45, 0.45, 0.45, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.25]*30
# print(electricity_price)
# sell_back_price = 0.3

# results_df = solve_gurobi(
#     start_hour = start_hour,
#     end_hour = end_hour,
#     instance = process_operations,
#     contributions = contributions,
#     E_max=250,
#     alpha=0.9,
#     beta=0.9,
#     theta=0.005,
#     G_max_solar=G_max_solar,
#     G_max_grid= 1000,
#     GPU_power_options = GPU_power_options,
#     electricity_price = electricity_price,
#     sell_back_price = sell_back_price
# )


from instance_gen.process_gen import generate_instance
from solve import *
import json
import pandas as pd

# 读取太阳能数据
solar_data = pd.read_csv('/Users/jingsichen/Politecnico Di Torino Studenti Dropbox/Jingsi Chen/Mac/Desktop/Green AI/AI/solve/pvwatts_hourly.csv')
G_max_solar = solar_data['AC System Output (W)'][1:8761].tolist()

# 定义其他参数
process_operations = generate_instance()
contributions = [1/32, 1/28, 1/26, 1/25, 1/24, 1/23, 1/22]
GPU_power_options = [100, 125, 150, 175, 200, 225, 250]
electricity_price = [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.45, 0.45, 0.45, 0.45, 0.45,
                     0.45, 0.45, 0.45, 0.45, 0.45, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.25]*30
sell_back_price = 0.3

# # 存储所有结果
# all_results = []

# 循环计算不同持续时间和开始时间的电费支出
for Hours in range(26, 33):
    for start_hour in range(72, 144):
        end_hour = start_hour + Hours
        result = solve_gurobi(
            start_hour=start_hour,
            end_hour=end_hour,
            instance=process_operations,
            contributions=contributions,
            E_max=250,
            alpha=0.9,
            beta=0.9,
            theta=0.005,
            G_max_solar=G_max_solar,
            G_max_grid=1000,
            GPU_power_options=GPU_power_options,
            electricity_price=electricity_price,
            sell_back_price=sell_back_price
        )
        result['Hours'] = Hours
        result['Start Hour'] = start_hour
        # all_results.append(result)

# 将所有结果保存到一个DataFrame中
# results_df = pd.DataFrame(all_results)

# 保存结果到文件（可选）
# results_df.to_csv('energy_costs_results.csv', index=False)

# 输出结果
# print(results_df)

