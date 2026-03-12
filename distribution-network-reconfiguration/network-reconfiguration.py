# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 23:00:00 2026

@author: Chuhong
"""

#%% 定义变量
from gurobipy import *  # 导入Gurobi优化求解器的所有模块
from readData import *  # 导入数据读取模块中的所有变量和数据
import pandas as pd
import networkx as nx  # 导入networkx库，用于图论分析和网络拓扑绘制
from pathlib import Path  # 导入Path类，用于处理文件路径

OUTPUT_DIR = BASE_DIR if 'BASE_DIR' in globals() else Path(__file__).resolve().parent  # 设置输出目录，优先使用readData中定义的BASE_DIR
model = Model('distflow')  # 创建Gurobi优化模型，命名为'distflow'（配电网潮流模型）
objective_type = 'cost'  # 可选: 'loss' 或 'cost'
# 外部购电
P_balance_it = model.addVars(balance_node, T_set, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='P_balance_it')
Q_balance_it = model.addVars(balance_node, T_set, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='Q_balance_it')
# 支路电流
L_ijt = model.addVars(branch_ij,T_set,ub=I_ijmax,name='L_ijt')  # 添加支路电流平方变量，上界为电流平方上限值
# 节点电压
v_it = model.addVars(
    B_set,
    T_set,
    lb={(i, t): float(ppc['bus'][i, 12] ** 2) for i in B_set for t in T_set},
    ub={(i, t): float(ppc['bus'][i, 11] ** 2) for i in B_set for t in T_set},
    name='v_it'
)  # 添加节点电压平方变量，上下界直接使用ppc节点电压限制的平方
# 支路有功
P_ijt = model.addVars(branch_ij, T_set,lb=-GRB.INFINITY,name='P_ijt')  # 添加支路有功功率变量，允许为负（双向潮流）
# 支路无功
Q_ijt = model.addVars(branch_ij, T_set,lb=-GRB.INFINITY,name='Q_ijt')  # 添加支路无功功率变量，允许为负
# 支路开断
alpha_ij = model.addVars(branch_ij,T_set,lb=0,ub=1,name='alpha')  # 添加支路开断状态变量（连续松弛），0=断开，1=闭合
beta_ij = model.addVars(branch_ij,T_set,vtype=GRB.BINARY,name='beta_ij')  # 添加支路正向虚拟功率流方向变量（0-1整数变量），用于辐射网约束
beta_ji = model.addVars(branch_ji,T_set,vtype=GRB.BINARY,name='beta_ji')  # 添加支路反向虚拟功率流方向变量（0-1整数变量），用于辐射网约束
#%% 约束
# 平衡节点机组出力上下限（按节点给定，不含时间索引）
model.addConstrs((P_balance_it[i, t] <= balancePmax_node[i] for i in balance_node for t in T_set), name='BalancePmax')
model.addConstrs((P_balance_it[i, t] >= balancePmin_node[i] for i in balance_node for t in T_set), name='BalancePmin')
model.addConstrs((Q_balance_it[i, t] <= balanceQmax_node[i] for i in balance_node for t in T_set), name='BalanceQmax')
model.addConstrs((Q_balance_it[i, t] >= balanceQmin_node[i] for i in balance_node for t in T_set), name='BalanceQmin')

# 潮流方程
# 普通节点
model.addConstrs((-P_in_it[i,t]==quicksum(P_ijt[i,j,t] for j in Ninsert_set[i] if (i,j) in branch_ij)  # 普通节点有功功率平衡：-负荷 = 流出功率之和 - (流入功率 - 线路损耗)之和
                  - quicksum(P_ijt[k,i,t] - r_ij[k,i]*L_ijt[k,i,t] for k in Nout_set[i]if (k,i) in branch_ij) 
                  for t in T_set for i in comment_set) ,name='nodePbalance3')
model.addConstrs((-Q_in_it[i,t]==quicksum(Q_ijt[i,j,t] for j in Ninsert_set[i])  # 普通节点无功功率平衡：-无功负荷 = 流出无功之和 - (流入无功 - 线路无功损耗)之和
                  - quicksum(Q_ijt[k,i,t]- x_ij[k,i]*L_ijt[k,i,t] for k in Nout_set[i]) 
                  for t in T_set for i in comment_set),name='nodeQbalance4')
# 对外购电
model.addConstrs((P_balance_it[i,t]-P_in_it[i,t]==quicksum(P_ijt[i,j,t] for j in Ninsert_set[i])  # 平衡节点有功功率平衡：平衡出力 - 负荷 = 流出功率之和 - (流入功率 - 线路损耗)之和
                  - quicksum(P_ijt[k,i,t]- r_ij[k,i]*L_ijt[k,i,t] for k in Nout_set[i]) 
                  for t in T_set for i in balance_node ),name='firstnode1')
model.addConstrs((Q_balance_it[i,t]-Q_in_it[i,t]==quicksum(Q_ijt[i,j,t] for j in Ninsert_set[i])  # 平衡节点无功功率平衡：平衡无功出力 - 负荷 = 流出无功之和 - (流入无功 - 线路无功损耗)之和
                  - quicksum(Q_ijt[k,i,t] - x_ij[k,i]*L_ijt[k,i,t] for k in Nout_set[i]) 
                  for t in T_set for i in balance_node ),name='firstnode2')
# 电压约束
model.addConstrs((v_it[j,t]<=bigM*(1-alpha_ij[i,j,t])+v_it[i,t]-2*(r_ij[i,j]*P_ijt[i,j,t]  # 电压降落方程上界约束（DistFlow模型），利用大M法处理支路开断状态
                +x_ij[i,j]*Q_ijt[i,j,t]+(r_ij[i,j]**2+x_ij[i,j]**2)*L_ijt[i,j,t] ) for (i,j) in 
                  branch_ij for t in T_set),name='V-2')
model.addConstrs((v_it[j,t]>=-bigM*(1-alpha_ij[i,j,t])+v_it[i,t]-2*(r_ij[i,j]*P_ijt[i,j,t]  # 电压降落方程下界约束（DistFlow模型），利用大M法处理支路开断状态
                +x_ij[i,j]*Q_ijt[i,j,t]+(r_ij[i,j]**2+x_ij[i,j]**2)*L_ijt[i,j,t] ) for (i,j) in 
                  branch_ij for t in T_set),name='V-3')
model.addConstrs((v_it[0,t]==1 for t in T_set),name='balancenode')  # 平衡节点（节点0）电压平方固定为1（即1.0 p.u.）
# 电流约束
model.addConstrs((L_ijt[i,j,t]*v_it[i,t]>=(P_ijt[i,j,t]**2+Q_ijt[i,j,t]**2) for (i,j) in branch_ij  # 二阶锥松弛约束：电流平方×电压平方 >= 有功平方+无功平方（SOC松弛）
                  for t in T_set),name='SOC')
model.addConstrs((L_ijt[i,j,t]<=alpha_ij[i,j,t]*I_ijmax for (i,j) in branch_ij for t in T_set),name='Iconstrs')  # 支路电流上限约束：当支路断开(alpha=0)时电流为0
# 辐射网约束
model.addConstrs((quicksum(alpha_ij[i,j,t] for (i,j) in branch_ij) ==B_num-1 for t in T_set),name='radical-1')  # 辐射网约束1：闭合支路数 = 节点数-1（树形结构条件）
model.addConstrs((beta_ij[i,j,t]+beta_ji[j,i,t]==alpha_ij[i,j,t] for (i,j) in branch_ij for t in T_set),name='radical-2')  # 辐射网约束2：若支路闭合则正向或反向虚拟流恰好选一个
model.addConstrs((quicksum(beta_ij[j,i,t] for j in Nout_set[i])==1 for i in B_set if i not in [0] for t in T_set),name='radical-3')  # 辐射网约束3：每个非根节点恰好有一个父节点（保证树形结构）
# model.addConstrs((beta_ij[0,j,t]==0 for j in Ninsert_set[0] for t in T_set),name='redical-4')  # （已注释）辐射网约束4：根节点不作为子节点

#%% 目标函数
def _poly_cost(gencost_row, power_var):
    """按 MATPOWER gencost 多项式形式计算单个平衡节点购电成本。"""
    n = int(gencost_row[3])
    power_mw = power_var * baseMVA
    expr = 0
    for k in range(n):
        coeff = float(gencost_row[4 + k])
        expr += coeff * (power_mw ** (n - 1 - k))
    return expr

if objective_type == 'loss':
    obj = quicksum(L_ijt[i,j,t]*r_ij[i,j] for (i,j) in branch_ij for t in T_set)  # 目标函数：最小化所有支路在所有时段的有功网损之和（线路电流平方 × 电阻）
elif objective_type == 'cost':
    obj = quicksum(
        _poly_cost(ppc['gencost'][gen_idx], P_balance_it[int(ppc['gen'][gen_idx, 0]), t])
        for gen_idx in range(len(ppc['gen'])) for t in T_set
    )
else:
    raise ValueError(f"不支持的 objective_type: {objective_type}，可选值为 'loss' 或 'cost'")

model.update()  # 更新模型，使新添加的变量和约束生效
model.setObjective(obj,GRB.MINIMIZE)  # 设置优化方向为最小化目标函数
# model.Params.MIPGap = 0.01  # （已注释）设置MIP求解间隙为1%
model.update()  # 再次更新模型
model.optimize()  # 调用Gurobi求解器进行求解
# # model.write('abf.lp')  # （已注释）将模型写出为LP格式文件
# model.computeIIS()  # （已注释）计算不可行子系统（调试不可行模型时使用）
# model.write('abc.ilp')  # （已注释）将不可行子系统写出为ILP文件
#%% 输出数据
import matplotlib.pyplot as plt  # 导入matplotlib绑图库
import matplotlib  # 导入matplotlib模块
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # 设置中文字体为黑体或微软雅黑，解决中文显示问题
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号"-"显示为方块的问题


def print_bus_data_table(period=0):
    print("\n" + "=" * 110)
    print(f"{'Bus Data':^110}")
    print(f"{'t=' + str(period):^110}")
    print("=" * 110)
    print(f"{'Bus':>3} {'Voltage':>15} {'Generation':>20} {'Load':>15} {'Net Injection':>20}")
    print(f"{'#':>3} {'Mag(pu)':>7} {'Ang(deg)':>8} {'P (MW)':>8} {'Q (MVAr)':>12} {'P (MW)':>7} {'Q (MVAr)':>8} {'P':>10} {'Q':>10}")
    print("-" * 110)

    total_gen_p = 0.0
    total_gen_q = 0.0
    total_load_p = 0.0
    total_load_q = 0.0

    balance_set = set(balance_node)

    for bus in B_set:
        v_mag = v_it[bus, period].x ** 0.5
        v_ang_str = "-"

        gen_p = 0.0
        gen_q = 0.0
        if bus in balance_set:
            gen_p += P_balance_it[bus, period].x * baseMVA
            gen_q += Q_balance_it[bus, period].x * baseMVA

        load_p = P_in_it[bus, period] * baseMVA
        load_q = Q_in_it[bus, period] * baseMVA
        net_p = gen_p - load_p
        net_q = gen_q - load_q

        total_gen_p += gen_p
        total_gen_q += gen_q
        total_load_p += load_p
        total_load_q += load_q

        gen_p_str = f"{gen_p:.2f}" if abs(gen_p) > 1e-8 else "-"
        gen_q_str = f"{gen_q:.2f}" if abs(gen_q) > 1e-8 else "-"
        load_p_str = f"{load_p:.2f}" if abs(load_p) > 1e-8 else "-"
        load_q_str = f"{load_q:.2f}" if abs(load_q) > 1e-8 else "-"

        print(f"{bus+1:>3} {v_mag:>7.3f} {v_ang_str:>8} {gen_p_str:>8} {gen_q_str:>12} {load_p_str:>7} {load_q_str:>8} {net_p:>10.3f} {net_q:>10.3f}")

    print("-" * 110)
    print(f"{'Total:':>20} {total_gen_p:>8.2f} {total_gen_q:>12.2f} {total_load_p:>7.2f} {total_load_q:>8.2f} {'':>10} {'':>10}")


def print_branch_data_table(period=0):
    print("\n" + "=" * 110)
    print(f"{'Branch Data':^110}")
    print(f"{'t=' + str(period):^110}")
    print("=" * 110)
    INDEX_W = 5
    BUS_W = 4
    SEP2 = "  "
    SEP4 = "    "
    NUM_W = 10
    GROUP_W = NUM_W * 2 + len(SEP2)
    print(f"{'Brnch':>{INDEX_W}}{SEP2}{'From':>{BUS_W}}{SEP2}{'To':>{BUS_W}}{SEP4}{'From Bus Injection':^{GROUP_W}}{SEP4}{'To Bus Injection':^{GROUP_W}}{SEP4}{'Loss (I^2 * Z)':^{GROUP_W}}")
    print(f"{'#':>{INDEX_W}}{SEP2}{'Bus':>{BUS_W}}{SEP2}{'Bus':>{BUS_W}}{SEP4}{'P (MW)':>{NUM_W}}{SEP2}{'Q (MVAr)':>{NUM_W}}{SEP4}{'P (MW)':>{NUM_W}}{SEP2}{'Q (MVAr)':>{NUM_W}}{SEP4}{'P (MW)':>{NUM_W}}{SEP2}{'Q (MVAr)':>{NUM_W}}")
    print("-" * 110)

    total_p_loss = 0.0
    total_q_loss = 0.0
    for idx, (i, j) in enumerate(branch_ij):
        p_from = P_ijt[i, j, period].x * baseMVA
        q_from = Q_ijt[i, j, period].x * baseMVA
        i_sq = L_ijt[i, j, period].x
        p_loss = i_sq * r_ij[i, j] * baseMVA
        q_loss = i_sq * x_ij[i, j] * baseMVA
        p_to = -(P_ijt[i, j, period].x - r_ij[i, j] * i_sq) * baseMVA
        q_to = -(Q_ijt[i, j, period].x - x_ij[i, j] * i_sq) * baseMVA

        total_p_loss += p_loss
        total_q_loss += q_loss
        print(f"{idx+1:>{INDEX_W}d}{SEP2}{i+1:>{BUS_W}d}{SEP2}{j+1:>{BUS_W}d}{SEP4}{p_from:>{NUM_W}.2f}{SEP2}{q_from:>{NUM_W}.2f}{SEP4}{p_to:>{NUM_W}.2f}{SEP2}{q_to:>{NUM_W}.2f}{SEP4}{p_loss:>{NUM_W}.3f}{SEP2}{q_loss:>{NUM_W}.2f}")

    print("-" * 110)
    print(f"{'':>{INDEX_W}}{SEP2}{'':>{BUS_W}}{SEP2}{'':>{BUS_W}}{SEP4}{'':>{NUM_W}}{SEP2}{'':>{NUM_W}}{SEP4}{'':>{NUM_W-1}}{SEP2}{'':>{1}}{SEP4}Total:{SEP4}{total_p_loss:>{NUM_W}.3f}{SEP2}{total_q_loss:>{NUM_W}.2f}")


def _split_edges_for_detour(edgelist, pos):
    """将会与同排中间节点重叠的长边拆分为折线绘制，其余保持直线。"""
    straight_edges = []
    detour_edges = []
    tol = 1e-9
    for u, v in edgelist:
        x1, y1 = pos[u]
        x2, y2 = pos[v]

        is_horizontal_long = abs(y1 - y2) <= tol and abs(x1 - x2) > 1.0
        if not is_horizontal_long:
            straight_edges.append((u, v))
            continue

        xmin, xmax = sorted((x1, x2))
        has_middle_node = any(
            n not in (u, v) and abs(py - y1) <= tol and xmin < px < xmax
            for n, (px, py) in pos.items()
        )

        if has_middle_node:
            detour_edges.append((u, v))
        else:
            straight_edges.append((u, v))

    return straight_edges, detour_edges


def _draw_edges_with_detour(graph, ax, pos, edgelist, edge_color, width, style='solid', alpha=1.0):
    """优先画直线；对易重叠长边画梯形折线，避免与主干线叠线。"""
    straight_edges, detour_edges = _split_edges_for_detour(edgelist, pos)

    if straight_edges:
        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=straight_edges,
            ax=ax,
            edge_color=edge_color,
            width=width,
            style=style,
            alpha=alpha,
        )

    for u, v in detour_edges:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        sign = 1 if x2 >= x1 else -1

        # 折线抬升/下探方向：上排往上拐，下排往下拐，主干线默认往上拐
        offset = 0.45 if y1 >= 0 else -0.45
        shoulder = min(0.9, max(0.35, 0.18 * abs(x2 - x1)))

        xs = [x1, x1 + sign * shoulder, x2 - sign * shoulder, x2]
        ys = [y1, y1 + offset, y2 + offset, y2]
        ax.plot(
            xs,
            ys,
            color=edge_color,
            linewidth=width,
            linestyle=style,
            alpha=alpha,
            solid_capstyle='round',
        )

o = t  # 将支路终止节点数组赋值给变量o（用于后续构建DataFrame时作为'to'列）
if model.status == GRB.Status.OPTIMAL:  # 如果模型求解成功（找到最优解）
    dictalpha = {}  # 初始化支路开断状态结果字典
    dictv = {'bus_index': B_set}  # 初始化节点电压结果字典，包含节点编号索引
    dictpij = {'line_index': branch_ij}  # 初始化支路有功功率结果字典，包含支路编号索引
    dictbeta_ij = {'line_index': branch_ij}  # 初始化正向虚拟流结果字典
    dictbeta_ji = {'line_index': branch_ji}  # 初始化反向虚拟流结果字典
    dictL_ij = {'line_index': branch_ij}  # 初始化支路电流平方结果字典
    for t in T_set:  # 遍历每个时段
        dictalpha['t={}'.format(t)] = [alpha_ij[L+(t,) ].x for L in branch_ij]  # 提取每个时段各支路的开断状态值
        dictv['t={}'.format(t)] = [v_it[i,t].x for i in B_set]  # 提取每个时段各节点的电压平方值
        dictpij['t={}'.format(t)] = [P_ijt[L+(t,)].x for L in branch_ij]  # 提取每个时段各支路的有功功率值
        dictbeta_ij['t={}'.format(t)] = [beta_ij[L+(t,) ].x for L in branch_ij]  # 提取每个时段各支路的正向虚拟流值
        dictbeta_ji['t={}'.format(t)] = [beta_ji[L+(t,) ].x for L in branch_ji]  # 提取每个时段各支路的反向虚拟流值
        dictL_ij['t={}'.format(t)] = [L_ijt[L+(t,) ].x for L in branch_ij]  # 提取每个时段各支路的电流平方值

    print(f"\nDistFlow求解成功: 目标值={model.ObjVal:.6f}, 求解时间={model.Runtime:.3f}s")

    print_bus_data_table(period=0)
    print_branch_data_table(period=0)

    dfalpha = pd.DataFrame(dictalpha)  # 将开断状态字典转换为DataFrame
    dL_ij = pd.DataFrame(dictL_ij)  # 将电流平方字典转换为DataFrame
    dfalpha1 = dfalpha.copy()  # 复制一份开断状态DataFrame用于后续导出
    dfalpha.index = branch_ij  # 设置开断状态DataFrame的索引为支路元组(i,j)
    alpha_path = OUTPUT_DIR / 'alpha.xlsx'  # 构建alpha结果文件的输出路径
    dfalpha1.to_excel(alpha_path)  # 将开断状态数据导出为Excel文件
    dfalpha.iloc[:,0] = f  # 将DataFrame第一列替换为支路起始节点数组（from节点）
    dfalpha.iloc[:,1] = o  # 将DataFrame第二列替换为支路终止节点数组（to节点）
    flag = 0  # 初始化辐射网验证标志
    for i in T_set[3:]:  # 从第3个时段开始遍历（跳过前几个时段）
        frm = dfalpha[dfalpha.iloc[:,i]==1].iloc[:,0].values.tolist()  # 获取当前时段闭合支路的起始节点列表
        to = dfalpha[dfalpha.iloc[:,i]==1].iloc[:,1].values.tolist()  # 获取当前时段闭合支路的终止节点列表
        df = pd.DataFrame({'Source': frm, 'Target': to})  # 构建边列表DataFrame
        G = nx.Graph()  # 创建无向图对象
        G.add_edges_from(df.values.tolist())  # 将边添加到图中
        if not nx.is_directed_acyclic_graph(G):  # 检查是否为有向无环图（此处用无向图检查，实际判断连通性）
            flag += 1  # 如果不满足条件，标志加1
    if flag:  # 如果存在不满足辐射网条件的时段
        print("重构后的电网为辐射网")  # 打印重构成功信息
        result_path = OUTPUT_DIR / '电网重构结果.xlsx'  # 构建重构结果文件路径
        dfalpha1.to_excel(result_path)  # 将重构结果导出为Excel文件
    else:  # 如果所有时段都满足条件
        print("重构错误，重构后的电网存在环网")  # 打印重构失败信息（存在环路）

    #%% 绘制配电网拓扑图（重构前 vs 重构后）

    # --- IEEE 33节点固定布局坐标 ---
    pos_fixed = {  # 定义33个节点的二维坐标位置，用于网络拓扑可视化
        0: (0, 0), 1: (1, 0), 2: (2, 0), 3: (3, 0), 4: (4, 0), 5: (5, 0),  # 主干线节点（水平排列）
        6: (6, 0), 7: (7, 0), 8: (8, 0), 9: (9, 0), 10: (10, 0), 11: (11, 0),  # 主干线节点（续）
        12: (12, 0), 13: (13, 0), 14: (14, 0), 15: (15, 0), 16: (16, 0), 17: (17, 0),  # 主干线节点（续）
        18: (2, -1.5), 19: (3, -1.5), 20: (4, -1.5), 21: (5, -1.5),  # 下侧分支节点
        22: (3, 1.5), 23: (4, 1.5), 24: (5, 1.5),  # 上侧分支节点（左段）
        25: (7, 1.5), 26: (8, 1.5), 27: (9, 1.5), 28: (10, 1.5), 29: (11, 1.5),  # 上侧分支节点（右段）
        30: (12, 1.5), 31: (13, 1.5), 32: (14, 1.5),  # 上侧分支节点（最右段）
    }

    # --- 构建重构前拓扑（按原始status区分闭合/断开）---
    G_before = nx.Graph()  # 创建无向图对象，表示重构前的配电网
    G_before.add_nodes_from(B_set)  # 添加所有节点
    G_before.add_edges_from(branch_ij)  # 添加所有候选支路（含联络线）

    # readData.py中提供了原始status接口：初始闭合支路和初始断开支路。
    # 若接口不可用，则回退为“全部闭合”的旧逻辑。
    closed_branches_before = list(globals().get('initial_closed_branch_ij', branch_ij))
    opened_branches_before = list(globals().get('initial_open_branch_ij', []))
    spare_branches_before = [
        (int(row[0]), int(row[1]))
        for row in ppc['branch'][-5:, :2]
    ]

    # --- 构建重构后的图（取 t=0 时段） ---
    t_show = 0  # 选择展示的时段为t=0
    col = 't={}'.format(t_show)  # 构建对应列名
    closed_branches = [branch_ij[idx] for idx in range(len(branch_ij))  # 获取重构后闭合的支路列表（alpha>0.5视为闭合）
                       if dictalpha[col][idx] > 0.5]
    opened_branches = [branch_ij[idx] for idx in range(len(branch_ij))  # 获取重构后断开的支路列表（alpha<=0.5视为断开）
                       if dictalpha[col][idx] <= 0.5]

    G_after = nx.Graph()  # 创建无向图对象，表示重构后的配电网
    G_after.add_nodes_from(B_set)  # 添加所有节点
    G_after.add_edges_from(closed_branches)  # 仅添加闭合的支路

    # --- 节点分类颜色 ---
    def get_node_colors(graph):  # 定义函数：根据节点类型返回颜色列表
        colors = []  # 初始化颜色列表
        for n in graph.nodes():  # 遍历图中的所有节点
            if n == 0:  # 如果是电源节点（变电站，节点0）
                colors.append('#E74C3C')  # 红色
            else:  # 其他普通负荷节点
                colors.append('#3498DB')  # 蓝色
        return colors  # 返回颜色列表

    # --- 绘图 ---
    fig, axes = plt.subplots(1, 2, figsize=(28, 8))  # 创建1行2列的子图，总尺寸28x8英寸

    # ====== 重构前拓扑 ======
    ax1 = axes[0]  # 获取左侧子图轴对象
    ax1.set_title('重构前配电网拓扑（IEEE 33节点）', fontsize=16, fontweight='bold', pad=15)  # 设置左侧子图标题
    node_colors_before = get_node_colors(G_before)  # 获取重构前各节点的颜色
    _draw_edges_with_detour(G_before, ax1, pos_fixed, closed_branches_before, edge_color='#2C3E50',  # 绘制重构前闭合支路（黑色加粗，重叠边用梯形折线）
                            width=2.2, style='solid', alpha=0.9)
    _draw_edges_with_detour(G_before, ax1, pos_fixed, opened_branches_before, edge_color='#E74C3C',  # 绘制重构前断开支路（红色虚线）
                            width=2.0, style='dashed', alpha=0.5)
    nx.draw_networkx_nodes(G_before, pos_fixed, ax=ax1, node_color=node_colors_before,  # 绘制重构前所有节点（按类型着色）
                           node_size=400, edgecolors='white', linewidths=1.5)
    nx.draw_networkx_labels(G_before, pos_fixed, ax=ax1,  # 绘制节点编号标签（白色加粗字体）
                            font_size=8, font_color='white', font_weight='bold')
    ax1.legend(handles=[  # 添加图例
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#E74C3C',  # 图例项：电源节点（红色）
                   markersize=12, label='电源节点 (节点0)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498DB',  # 图例项：负荷节点（蓝色）
                   markersize=12, label='负荷节点'),
        plt.Line2D([0], [0], color='#2C3E50', linewidth=2, label='闭合支路'),  # 图例项：闭合支路（深色实线）
        plt.Line2D([0], [0], color='#E74C3C', linewidth=2, linestyle='dashed',  # 图例项：断开支路（红色虚线）
                   label='断开支路'),
    ], loc='lower left', fontsize=10, framealpha=0.9)  # 图例放在左下角，半透明背景
    ax1.set_axis_off()  # 关闭坐标轴显示
    spare_str = ', '.join([f'{e[0]}-{e[1]}' for e in spare_branches_before])
    ax1.text(0.5, -0.05, f'备用支路: {spare_str}',
             transform=ax1.transAxes, ha='center', fontsize=11, color='#E74C3C',
             fontweight='bold')

    # ====== 重构后拓扑 ======
    ax2 = axes[1]  # 获取右侧子图轴对象
    ax2.set_title(f'重构后配电网拓扑（t={t_show}时段）', fontsize=16, fontweight='bold', pad=15)  # 设置右侧子图标题
    node_colors_after = get_node_colors(G_after)  # 获取重构后各节点的颜色

    _draw_edges_with_detour(G_after, ax2, pos_fixed, closed_branches, edge_color='#2C3E50',  # 绘制闭合支路（重叠边改为梯形折线）
                            width=2.2, style='solid', alpha=0.9)
    _draw_edges_with_detour(G_before, ax2, pos_fixed, opened_branches, edge_color='#E74C3C',  # 绘制断开支路（红色虚线）
                            width=2.0, style='dashed', alpha=0.5)
    nx.draw_networkx_nodes(G_after, pos_fixed, ax=ax2, node_color=node_colors_after,  # 绘制重构后所有节点
                           node_size=400, edgecolors='white', linewidths=1.5)
    nx.draw_networkx_labels(G_after, pos_fixed, ax=ax2,  # 绘制节点编号标签
                            font_size=8, font_color='white', font_weight='bold')

    opened_str = ', '.join([f'{e[0]}-{e[1]}' for e in opened_branches])  # 将断开支路列表格式化为字符串
    ax2.text(0.5, -0.05, f'断开支路: {opened_str}',  # 在图下方标注断开的支路信息
             transform=ax2.transAxes, ha='center', fontsize=11, color='#E74C3C',
             fontweight='bold')

    ax2.legend(handles=[  # 添加图例
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#E74C3C',  # 图例项：电源节点
                   markersize=12, label='电源节点 (节点0)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498DB',  # 图例项：负荷节点
                   markersize=12, label='负荷节点'),
        plt.Line2D([0], [0], color='#2C3E50', linewidth=2, label='闭合支路'),  # 图例项：闭合支路（深色实线）
        plt.Line2D([0], [0], color='#E74C3C', linewidth=2, linestyle='dashed',  # 图例项：断开支路（红色虚线）
                   label='断开支路'),
    ], loc='lower left', fontsize=10, framealpha=0.9)  # 图例放在左下角
    ax2.set_axis_off()  # 关闭坐标轴显示

    plt.tight_layout()  # 自动调整子图间距，避免重叠
    topo_compare_path = OUTPUT_DIR / '配电网重构拓扑对比.png'  # 构建拓扑对比图的输出路径
    plt.savefig(topo_compare_path, dpi=200, bbox_inches='tight',  # 保存图片，分辨率200dpi，紧凑裁剪
                facecolor='white', edgecolor='none')
    plt.show()  # 显示图片
    print(f"拓扑图已保存至: {topo_compare_path}")  # 打印保存路径

    # --- 多时段重构结果展示 ---
    show_periods = [0, 6, 12, 18]  # 选择展示的4个代表性时段
    fig2, axes2 = plt.subplots(2, 2, figsize=(28, 16))  # 创建2x2子图布局
    axes2 = axes2.flatten()  # 将二维子图数组展平为一维，方便遍历

    for idx, t_p in enumerate(show_periods):  # 遍历每个展示时段
        ax = axes2[idx]  # 获取当前子图轴对象
        col_p = 't={}'.format(t_p)  # 构建对应列名
        closed_p = [branch_ij[k] for k in range(len(branch_ij))  # 获取当前时段闭合的支路列表
                    if dictalpha[col_p][k] > 0.5]
        opened_p = [branch_ij[k] for k in range(len(branch_ij))  # 获取当前时段断开的支路列表
                    if dictalpha[col_p][k] <= 0.5]

        G_p = nx.Graph()  # 创建当前时段的无向图对象
        G_p.add_nodes_from(B_set)  # 添加所有节点
        G_p.add_edges_from(closed_p)  # 仅添加闭合的支路

        nc = get_node_colors(G_p)  # 获取节点颜色
        _draw_edges_with_detour(G_p, ax, pos_fixed, closed_p, edge_color='#2C3E50',  # 绘制闭合支路（重叠边改为梯形折线）
                    width=2.0, style='solid', alpha=0.9)
        _draw_edges_with_detour(G_before, ax, pos_fixed, opened_p, edge_color='#E74C3C',  # 绘制断开支路（红色虚线）
                    width=1.8, style='dashed', alpha=0.5)
        nx.draw_networkx_nodes(G_p, pos_fixed, ax=ax, node_color=nc,  # 绘制节点
                               node_size=350, edgecolors='white', linewidths=1.2)
        nx.draw_networkx_labels(G_p, pos_fixed, ax=ax,  # 绘制节点编号标签
                                font_size=7, font_color='white', font_weight='bold')

        opened_s = ', '.join([f'{e[0]}-{e[1]}' for e in opened_p])  # 格式化断开支路字符串
        ax.set_title(f't={t_p}时段 重构后拓扑', fontsize=14, fontweight='bold')  # 设置子图标题
        ax.text(0.5, -0.03, f'断开: {opened_s}', transform=ax.transAxes,  # 在子图下方标注断开支路
                ha='center', fontsize=10, color='#E74C3C', fontweight='bold')
        ax.set_axis_off()  # 关闭坐标轴显示

    fig2.suptitle('IEEE 33节点配电网 多时段重构结果', fontsize=18, fontweight='bold', y=0.98)  # 设置总标题
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 调整子图布局，为总标题留出空间
    topo_multi_path = OUTPUT_DIR / '配电网多时段重构拓扑.png'  # 构建多时段拓扑图的输出路径
    plt.savefig(topo_multi_path, dpi=200, bbox_inches='tight',  # 保存图片
                facecolor='white', edgecolor='none')
    plt.show()  # 显示图片
    print(f"多时段拓扑图已保存至: {topo_multi_path}")  # 打印保存路径
