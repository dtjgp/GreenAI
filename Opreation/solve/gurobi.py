import time
import random
from gurobipy import Model, GRB, quicksum
import numpy as np
import pandas as pd

def solve_gurobi(start_hour, end_hour, instance,contributions,E_max, alpha, beta, theta, G_max_solar, G_max_grid, GPU_power_options, electricity_price, sell_back_price):
    
    model = Model("Power_Management")

    # 创建变量
    # Hours = end_hour - start_hour  # 确保Hours是定义的持续时间
    I = range(len(instance)) # 进程的数量
    F = {i: range(len(instance[i]['operations'])) for i in I} # 每个进程的操作数量
    operations_info = {i: instance[i]['operations'] for i in I}
    operation_power = {i: {k: operations_info[i][k]['power'] for k in F[i]} for i in I}
    # operation_time = {i: {k: operations_info[i][k]['duration'] for k in F[i]} for i in I}
    # process_max_time = {i: instance[i]['max_time'] for i in I}

    G_solar = model.addVars(range(start_hour, end_hour), lb=0, name="G_solar")
    P_solar = model.addVars(range(start_hour, end_hour), lb=0, name="P_solar")
    R_solar = model.addVars(range(start_hour, end_hour), lb=0, name="R_solar")
    W_solar = model.addVars(range(start_hour, end_hour), lb=0, name="W_solar")

    for t in range(start_hour, end_hour):
        G_solar[t].ub = G_max_solar[t]
        P_solar[t].ub = G_max_solar[t]
        R_solar[t].ub = G_max_solar[t]
        W_solar[t].ub = G_max_solar[t]

    # 电网
    G_grid = model.addVars(range(start_hour, end_hour), lb=0, ub=G_max_grid, name="G_grid")
    P_grid = model.addVars(range(start_hour, end_hour), lb=0, ub=G_max_grid, name="P_grid")
    R_grid = model.addVars(range(start_hour, end_hour), lb=0, ub=G_max_grid, name="R_grid")
    W_grid = model.addVars(range(start_hour, end_hour), lb=0, ub=G_max_grid, name="W_grid")

    # 电池
    D_dc = model.addVars(range(start_hour, end_hour), lb=0, ub=E_max, name="D_dc")
    D_grid = model.addVars(range(start_hour, end_hour), lb=0, ub=E_max, name="D_grid")
    ESD = model.addVars(range(start_hour, end_hour+1), lb=0, ub=E_max, name="ESD")

    # GPU功率
    # y = model.addVars(range(start_hour, end_hour), I, {i: F[i] for i in I}, vtype=GRB.BINARY, name="y") # 操作选择变量 第t时刻第i个进程的第k个操作是否执行
    y = model.addVars(range(start_hour, end_hour), I, F[0], vtype=GRB.BINARY, name="y")
    gpu_power = model.addVars(range(start_hour, end_hour), lb=0, ub=max(GPU_power_options), name="gpu_power")
    power_selection = model.addVars(range(start_hour, end_hour), len(GPU_power_options), vtype=GRB.BINARY, name="power_selection")

    # 目标函数：最小化总电力成本
    objective =  quicksum((G_grid[t] * electricity_price[t] - W_grid[t] * sell_back_price) for t in range(start_hour, end_hour))
    model.setObjective(objective, sense=GRB.MINIMIZE)

    model.addConstrs(P_solar[t] + R_solar[t] + W_solar[t] == G_solar[t] for t in range(start_hour, end_hour)) # 太阳能给出去的总功率 = data center + battery + back to grid 
    model.addConstrs(P_grid[t] + R_grid[t] == G_grid[t] for t in range(start_hour, end_hour)) # 电网给出去的总功率 = data center + battery
    model.addConstrs(W_solar[t] + beta * D_grid[t] == W_grid[t] for t in range(start_hour, end_hour)) # 卖回电网的功率等于太阳能卖回电网的功率加上电池卖回电网的功率
    model.addConstrs(R_solar[t]+R_grid[t] <= (E_max - ESD[t]) for t in range(start_hour, end_hour)) # 电池容量限制
    model.addConstrs(ESD[t] == (1-theta) * (ESD[t-1] - D_dc[t-1] - D_grid[t-1] + alpha * (R_solar[t-1] + R_grid[t-1])) for t in range(start_hour+1, end_hour+1)) # 电池容量更新
    model.addConstrs(D_dc[t] + D_grid[t] <= ESD[t] for t in range(start_hour, end_hour)) # 电池放电功率不得超过电池容量
    model.addConstrs(D_dc[t] + D_grid[t] >= 0 for t in range(start_hour, end_hour)) # 电池放电功率不得为负

    # 每个时间槽的job完成量变量
    completion_per_slot = model.addVars(range(start_hour, end_hour), name="completion_per_slot", vtype=GRB.CONTINUOUS)

    # 每个时间槽中只能执行一个operation
    for t in range(start_hour, end_hour):
        model.addConstr(completion_per_slot[t] == quicksum(y[t, i, k] * contributions[i] for i in I for k in F[i]), name=f"CompletionTime_{t}")
        model.addConstr(quicksum(y[t, i, k] for i in I for k in F[i]) == 1, name=f"OneOperationAtTime_{t}")

    # 定义完成率变量
    model.addConstr(quicksum(completion_per_slot[t] for t in range(start_hour, end_hour)) >= 1, name="CompleteJob")

    model.addConstrs((P_solar[t] + P_grid[t] + beta * D_dc[t] >= quicksum(y[t, i, k] * instance[i]['operations'][k]['power'] for i in I for k in F[i]) for t in range(start_hour, end_hour)), "PowerBalance")
    model.addConstrs((P_solar[t] + P_grid[t] + beta * D_dc[t] >= gpu_power[t] for t in range(start_hour, end_hour)), "DataCenterPowerSupplyMatchesGPUPower")
    model.addConstrs((quicksum(power_selection[t, j] * GPU_power_options[j] for j in range(len(GPU_power_options))) == gpu_power[t] for t in range(start_hour, end_hour)),"SelectGPUPower")
    model.addConstrs((quicksum(power_selection[t, j] for j in range(len(GPU_power_options))) == 1 for t in range(start_hour, end_hour)),"OnePowerOption")

    model.addConstrs(
        (quicksum(y[t, i, k] * operation_power[i][k] for i in I for k in F[i]) <= gpu_power[t] 
        for t in range(start_hour, end_hour)),"GPUPowerLimit")

    # 初始条件和容量限制
    ESD[start_hour].setAttr(GRB.Attr.LB, 0)
    ESD[start_hour].setAttr(GRB.Attr.UB, 0)

    # 求解模型
    model.optimize()

    # 输出结果
    total_completion = sum(completion_per_slot[t].X for t in range(start_hour, end_hour))
    minimum_cost = model.objVal
    results = {}
    if model.status == GRB.Status.OPTIMAL:
        # results['status'] = "Optimal Solution Found"
        # results['details'] = []
        results = {
        'status': 'Optimal Solution Found',
        'details': [],
        'total_job_completion': f"{total_completion:.2%}",  # Format as a percentage
        'minimum_cost': minimum_cost
    }
        for t in range(start_hour, end_hour):
            time_details = {
                'time': t,
                'G_grid': G_grid[t].X,
                'W_grid': W_grid[t].X,
                'P_grid': P_grid[t].X,
                'R_grid': R_grid[t].X,
                'G_solar': G_solar[t].X,
                'W_solar': W_solar[t].X,
                'P_solar': P_solar[t].X,
                'R_solar': R_solar[t].X,
                'D_dc': D_dc[t].X,
                'D_grid': D_grid[t].X,
                'GPU_power': gpu_power[t].X,
                'operations': [],
                'job_completion': completion_per_slot[t].X 
        
            }
            for i in I:
                for k in range(len(operation_power[i])):
                    if y[t, i, k].X > 0.5:
                        time_details['operations'].append(f"Process {i} is executed")
            results['details'].append(time_details)
    else:
        results['status'] = "No Optimal Solution Found"


    energy_dispatch = results['details']
    energy_dispatch_df = pd.DataFrame(energy_dispatch)

    energy_dispatch_df['total_job_completion'] = f"{total_completion:.2%}"
    energy_dispatch_df['minimum_cost'] = f"{minimum_cost:.2f}"

    energy_dispatch_df.to_csv('/Users/jingsichen/Politecnico Di Torino Studenti Dropbox/Jingsi Chen/Mac/Desktop/Green AI/AI/results/Duration{}_start{}.csv'.format(end_hour-start_hour, start_hour), index=False)

    return results

