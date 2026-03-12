# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 23:00:00 2026

@author: Chuhong
"""

#%% 读取IEEE 33配电网数据
import sys
import numpy as np  # 导入numpy库，用于数值计算
from pathlib import Path  # 导入Path类，用于处理文件路径
from pypower.ext2int import ext2int

BASE_DIR = Path(__file__).resolve().parent  # 获取当前脚本所在目录的绝对路径
ROOT_DIR = BASE_DIR.parent
if str(ROOT_DIR) not in sys.path:
	sys.path.insert(0, str(ROOT_DIR))

from from_mpc_to_ppc import mpc2ppc
# bigM
bigM = 100  # 大M法中的M值，用于线性化约束中的松弛系数
# 读取并预处理IEEE 33节点系统（与ACOPF.py的case_33_mpc处理一致）
case_33_mpc = mpc2ppc(str(BASE_DIR / 'case33bw.m'))
# 从case文件读取基准值，避免硬编码
baseMVA = float(case_33_mpc['baseMVA'])  # 基准功率，单位MVA
basekV = float(case_33_mpc['bus'][0, 9])  # 基准电压，单位kV
baseI = baseMVA * 1000 / basekV  # 基准电流，单位A，用于标幺化
_Vbase = case_33_mpc['bus'][0, 9] * 1e3  # 基准电压 (V)
_Sbase = case_33_mpc['baseMVA'] * 1e6  # 基准功率 (VA)
_Zbase = _Vbase ** 2 / _Sbase  # 基准阻抗 (Ohm)
case_33_mpc['branch'][:, 2] /= _Zbase  # 支路电阻: Ohm -> p.u.
case_33_mpc['branch'][:, 3] /= _Zbase  # 支路电抗: Ohm -> p.u.
case_33_mpc['bus'][:, 2] /= 1e3  # 有功负荷: kW -> MW
case_33_mpc['bus'][:, 3] /= 1e3  # 无功负荷: kVAr -> MVAr

# 关键处理：将末尾5条备用联络线统一置为可参与重构的候选支路。
case_33_mpc['branch'][:, 10] = 1

# 保存原始联络线状态(1=闭合,0=断开)，用于“重构前拓扑”展示。
_raw_branch = case_33_mpc['branch'].copy()
initial_closed_branch_ij = [
	(int(row[0]) - 1, int(row[1]) - 1)
	for row in _raw_branch
	if int(row[10]) == 1
]
initial_open_branch_ij = [
	(int(row[0]) - 1, int(row[1]) - 1)
	for row in _raw_branch
	if int(row[10]) == 0
]

ppc = ext2int(case_33_mpc)

# 节点数据
T_set = np.arange(24)  # 时间段集合，24个时段（0~23），代表一天24小时
dT = 1  # 每个时段的时间长度，单位小时
B_set = ppc['bus'][:, 0].astype(int).tolist()  # 节点编号集合（来自ppc.bus第一列）
B_num = len(B_set)  # 配电网节点总数
_bus_row = {int(ppc['bus'][row, 0]): row for row in range(B_num)}  # 节点号到矩阵行号的映射
# 支路数据
# line
branchData = ppc['branch'][:, [0, 1, 2, 3, 10]].copy()  # [from, to, r, x, status]
f = branchData[:, 0].astype('int')  # 支路起始节点编号数组（from节点）
t = branchData[:, 1].astype('int')  # 支路终止节点编号数组（to节点）
branch_num = len(f)  # 支路总数
r = branchData[:, 2]  # 支路电阻标幺值
x = branchData[:, 3]  # 支路电抗标幺值
branch_ij = list(zip(f,t))  # 支路正向(i,j)元组列表，即从i到j
branch_ji = list(zip(t,f))  # 支路反向(j,i)元组列表，即从j到i
branch_ij_all = branch_ij+branch_ji  # 所有支路（正向+反向）的合并列表
I_ijmax = (500*500)/baseI**2  # 支路电流上限的平方标幺值（最大电流500A）
# r
r_ij = dict(zip(branch_ij,r))  # 支路电阻字典，键为(i,j)元组，值为对应标幺电阻
# x
x_ij = dict(zip(branch_ij,x))  # 支路电抗字典，键为(i,j)元组，值为对应标幺电抗

# 读取有功无功数据
P_in_it = {(i, t): float(ppc['bus'][_bus_row[i], 2] / baseMVA) for i in B_set for t in T_set}  # 各节点各时段有功负荷字典（标幺值）
Q_in_it = {(i, t): float(ppc['bus'][_bus_row[i], 3] / baseMVA) for i in B_set for t in T_set}  # 各节点各时段无功负荷字典（标幺值）

# 读取购电节点数据（来自ppc）
balance_node = [i for i in B_set if int(ppc['bus'][_bus_row[i], 1]) == 3]  # 平衡节点作为外部购电节点
if not balance_node:
	balance_node = [B_set[0]]

# 平衡节点出力上下限（来自ppc.gen，按母线聚合）
balancePmax_node = {i: 0.0 for i in balance_node}
balancePmin_node = {i: 0.0 for i in balance_node}
balanceQmax_node = {i: 0.0 for i in balance_node}
balanceQmin_node = {i: 0.0 for i in balance_node}
for row in ppc['gen']:
	bus_i = int(row[0])
	gen_status = int(row[7])
	if gen_status != 1 or bus_i not in balancePmax_node:
		continue
	balancePmax_node[bus_i] += float(row[8] / baseMVA)
	balancePmin_node[bus_i] += float(row[9] / baseMVA)
	balanceQmax_node[bus_i] += float(row[3] / baseMVA)
	balanceQmin_node[bus_i] += float(row[4] / baseMVA)

comment_set = sorted(set(B_set) - set(balance_node))  # 普通负荷节点集合（去除购电节点）
# 节点连接关系
Ninsert_set = {node:branchData[branchData[:,0]==node][:,1].astype('int').tolist() for node in B_set }  # 每个节点的下游邻接节点集合（从该节点出发的支路终点）
Nout_set = {node:branchData[branchData[:,1]==node][:,0].astype('int').tolist() for node in B_set }  # 每个节点的上游邻接节点集合（到达该节点的支路起点）
N_all_set = {node:branchData[branchData[:,0]==node][:,1].astype('int').tolist()+branchData[branchData[:,1]==node][:,0].astype('int').tolist()for node in B_set}  # 每个节点的所有邻接节点集合（上游+下游）
