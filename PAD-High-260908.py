# -*- coding: utf-8 -*-
"""
台积电 Fab 10 扩散区大规模调度 - 基于真实外挂数据的 ALNS 调度系统
【核心特性】：
1. 文件系统 I/O：直接从指定的 CSV 文件目录读取现场的 5 张关系型数据表。
2. 缺失值鲁棒性：自动处理 INTIME 的正负逻辑、缺失的 DISP_BAYID 等现场数据特有情况。
3. Lot 级 CNS 资质校验：打破传统“配方-机台”映射，实装精确到“工件-机台”的专属白名单校验。
"""
import os
import random
import math
import time
import copy
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors


# ==========================================
# 1. 核心数据结构定义
# ==========================================
class Operation:
    def __init__(self, op_id, op_type, recipe, qtime_limit=99999):
        self.op_id = op_id
        self.op_type = op_type
        self.recipe = recipe
        self.qtime_limit = qtime_limit


class Lot:
    def __init__(self, lot_id, arr_time, bay0, qty, priority, ops):
        self.lot_id = lot_id
        self.arr_time = arr_time
        self.bay0 = bay0
        self.qty = qty
        self.priority = priority
        self.ops = ops


class Machine:
    def __init__(self, mach_id, m_type, bay, capacity, available_time=0.0):
        self.mach_id = mach_id
        self.m_type = m_type
        self.bay = bay
        self.capacity = capacity
        self.available_time = available_time


class Batch:
    def __init__(self, machine_id, recipe, start_time, end_time, is_new=False):
        self.machine_id = machine_id
        self.recipe = recipe
        self.lots = []
        self.start_time = start_time
        self.end_time = end_time
        self.is_new = is_new

    def is_full(self, capacity=6):
        return len(self.lots) >= capacity


class ScheduleState:
    def __init__(self):
        self.machine_batches = defaultdict(list)
        self.unassigned_lots = set()
        self.lot_record = {}
        self.total_cycle_time = 0.0
        self.total_bay_penalty = 0.0
        self.obj_value = float('inf')


# ==========================================
# 2. 真实数据文件读取与解析引擎 (I/O & Parser)
# ==========================================
def load_tsmc_data_from_csv(data_dir="./tsmc_data"):
    """
    从指定目录读取真实现场的 CSV 关系表
    """
    print(f"[*] 📂 正在从目录 '{data_dir}' 加载现场关系型数据表...")

    try:
        df_mach_status = pd.read_csv(os.path.join(data_dir, "machine_status.csv"))
        df_mach_loc = pd.read_csv(os.path.join(data_dir, "machine_location.csv"))
        df_cns = pd.read_csv(os.path.join(data_dir, "cns_check.csv"))
        df_recipe_rt = pd.read_csv(os.path.join(data_dir, "recipe_rt.csv"))
        df_lots = pd.read_csv(os.path.join(data_dir, "lots_pool.csv"))

        print(f"   ✅ 数据读取成功! Lots: {len(df_lots)}, 机台状态: {len(df_mach_status)}, CNS规则: {len(df_cns)}")
        return df_mach_status, df_mach_loc, df_cns, df_recipe_rt, df_lots
    except Exception as e:
        print(f"   ❌ 读取数据文件失败: {str(e)}")
        print("   ⚠️ 请确保对应目录下存在所需的 5 张 CSV 表格。")
        raise e


def parse_and_enrich_tsmc_data(df_mach_status, df_mach_loc, df_cns, df_recipe_rt, df_lots):
    """
    清洗并解析现场数据，映射为 ALNS 的物理结构，同时处理缺失值与动态时效
    """
    print("[*] 🧬 正在将关系型数据投影为 ALNS 运筹模型 (执行数据清洗与 CNS 匹配)...")

    # 1. 解析 Recipe 处理时间 (现场单位为小时，转为分钟)
    recipes_pt = {row['AUTOMATEDRECIPE']: row['RT'] * 60 for _, row in df_recipe_rt.iterrows()}

    # 2. 解析机台状态与位置
    machines = []
    df_mach_full = pd.merge(df_mach_loc, df_mach_status, on='EQPID', how='left')
    for _, row in df_mach_full.iterrows():
        # 宕机状态剔除
        if row.get('STATUS', 'DOWN') == 'DOWN': continue

        m_id = row['EQPID']
        # 智能推断机台类型
        m_type = "PAD-OX" if "PAD" in str(row.get('MAINGROUP', '')) else "HIGH-TEMP"
        bay = str(row.get('INTERBAYID', 'UNKNOWN'))

        # 剩余时间处理 (小时 -> 分钟)
        rem_rt_hrs = row.get('REMAIN_RUNTIME', 0.0)
        avail_t = rem_rt_hrs * 60 if pd.notna(rem_rt_hrs) else 0.0

        machines.append(Machine(m_id, m_type, bay, capacity=6, available_time=avail_t))

    # 3. 解析 CNS 校验表，构建 [LotID + Recipe] -> 允许机台集合的白名单字典
    cns_map = defaultdict(set)
    for _, row in df_cns.iterrows():
        cns_map[(row['LOTID'], row['AUTORECP'])].add(row['EQPID'])

    # 4. 解析 Lot 列表及补全隐藏信息
    lots = []
    for _, row in df_lots.iterrows():
        lot_id = row['LOTID']
        qty = row.get('QTY', 25)
        # 优先读取 EXTERNALPRIORITY，兼容 PRIORITY
        priority = row.get('EXTERNALPRIORITY', row.get('PRIORITY', 500))
        recipe = row['AUTORECP']
        intime = row['INTIME']

        # 补全缺失的区域信息 (未到站的 Lot 可能没有 DISP_BAYID)
        bay0 = row.get('DISP_BAYID', 'UNKNOWN')
        if pd.isna(bay0) or bay0 == 'UNKNOWN':
            bay0 = 'Transit_Zone'

        # 处理时间：负数代表已到站，正数代表还在路上
        arr_time = 0.0 if intime < 0 else intime * 60

        # Q-Time 消耗补偿计算 (对于高优先级 1XX 且已到站的，计算剩余 Q-Time)
        q_limit = 99999
        if str(priority).startswith('1'):
            consumed = abs(intime) * 60 if intime < 0 else 0
            base_qtime = random.uniform(240, 480)  # 模拟原厂标定 Q-Time
            q_limit = max(10, base_qtime - consumed)

        op_type = "PAD-OX" if "PAD" in str(row.get('STAGE', '')) or "OPA" in recipe else "HIGH-TEMP"
        ops = [Operation(op_id=0, op_type=op_type, recipe=recipe, qtime_limit=q_limit)]

        lots.append(Lot(lot_id, arr_time, bay0, qty, priority, ops))

    # 5. 生成物理转移时间矩阵 (跨区 15 分钟)
    all_bays = set(m.bay for m in machines) | set(l.bay0 for l in lots) | {'Transit_Zone'}
    t_trans = {(b1, b2): 0 if b1 == b2 else 15 for b1 in all_bays for b2 in all_bays}

    return lots, machines, recipes_pt, cns_map, t_trans


# ==========================================
# 3. 严格零违约 ALNS 求解器 (加入 CNS 防线)
# ==========================================
class ALNS_Strict_Scheduler:
    def __init__(self, lots, machines, recipes_pt, cns_map, t_trans):
        self.lots = {l.lot_id: l for l in lots}
        self.machines = {m.mach_id: m for m in machines}
        self.recipes_pt = recipes_pt
        self.cns_map = cns_map
        self.t_trans = t_trans

        self.PENALTY_UNASSIGNED = 1000000
        self.PENALTY_BAY = 50

        self.best_solution = None
        self.history = []

    def evaluate_state(self, state: ScheduleState):
        ct_total, bay_pen = 0.0, 0.0
        for l_id, ops_record in state.lot_record.items():
            lot = self.lots[l_id]
            last_op = max(ops_record.keys())
            ct_total += (ops_record[last_op]['end'] - lot.arr_time)

            curr_bay = lot.bay0
            for o_idx in range(len(lot.ops)):
                m_bay = self.machines[ops_record[o_idx]['mach']].bay
                if curr_bay != m_bay: bay_pen += self.PENALTY_BAY
                curr_bay = m_bay

        state.total_cycle_time = ct_total
        state.total_bay_penalty = bay_pen
        state.obj_value = len(state.unassigned_lots) * self.PENALTY_UNASSIGNED + ct_total + bay_pen
        return state.obj_value

    def get_candidate_insertions(self, state: ScheduleState, lot: Lot, op_req: Operation, earliest_start: float):
        """【增强】基于 CNS 资质表获取候选插槽"""
        candidates = []
        # 处理现场可能缺少配方加工时间的情况，赋缺省值
        pt = self.recipes_pt.get(op_req.recipe, 180)

        # 提取当前 Lot + Recipe 在 CNS 表中允许的机台
        allowed_machines = self.cns_map.get((lot.lot_id, op_req.recipe))

        for m_id, m in self.machines.items():
            if m.m_type != op_req.op_type: continue

            # 【CNS 硬防线】如果 CNS 列表中有数据，且当前机器不在白名单中，则强行跳过！
            # 如果 cns_map 没有任何记录，为了程序的健壮性，允许向同类型机台派发
            if allowed_machines and (m_id not in allowed_machines):
                continue

            batches = state.machine_batches[m_id]
            for b_idx, b in enumerate(batches):
                if not b.is_full() and b.recipe == op_req.recipe and b.start_time >= earliest_start:
                    candidates.append(
                        {'m_id': m_id, 'b_idx': b_idx, 'start': b.start_time, 'end': b.end_time, 'is_new': False})

            mach_avail = max(batches[-1].end_time if batches else 0.0, m.available_time)
            start_t = max(earliest_start, mach_avail)
            candidates.append(
                {'m_id': m_id, 'b_idx': len(batches), 'start': start_t, 'end': start_t + pt, 'is_new': True})

        return candidates

    def try_atomic_insertion(self, state: ScheduleState, lot: Lot):
        best_plan = None
        best_cost_inc = float('inf')

        if len(lot.ops) == 1:
            cands = self.get_candidate_insertions(state, lot, lot.ops[0], lot.arr_time)
            for c in cands:
                cost = c['end'] - lot.arr_time
                if cost < best_cost_inc:
                    best_cost_inc = cost
                    best_plan = [c]

        elif len(lot.ops) == 2:
            op0, op1 = lot.ops[0], lot.ops[1]
            cands0 = self.get_candidate_insertions(state, lot, op0, lot.arr_time)

            for c0 in cands0:
                t_trans1 = self.t_trans.get((lot.bay0, self.machines[c0['m_id']].bay), 15)
                cands1 = self.get_candidate_insertions(state, lot, op1, c0['end'] + t_trans1)

                for c1 in cands1:
                    actual_wait = c1['start'] - c0['end'] - t_trans1
                    valid = False
                    adj_c0_start, adj_c0_end = c0['start'], c0['end']

                    if actual_wait <= op0.qtime_limit:
                        valid = True
                    else:
                        if c0['is_new']:
                            shift = actual_wait - op0.qtime_limit
                            adj_c0_start += shift
                            adj_c0_end += shift
                            if adj_c0_end + t_trans1 <= c1['start']:
                                valid = True

                    if valid:
                        cost = c1['end'] - lot.arr_time
                        if cost < best_cost_inc:
                            best_cost_inc = cost
                            best_plan = [
                                {'m_id': c0['m_id'], 'b_idx': c0['b_idx'], 'start': adj_c0_start, 'end': adj_c0_end,
                                 'is_new': c0['is_new']},
                                {'m_id': c1['m_id'], 'b_idx': c1['b_idx'], 'start': c1['start'], 'end': c1['end'],
                                 'is_new': c1['is_new']}]

        return best_plan

    def repair_operator(self, state: ScheduleState, tasks_to_insert):
        for l_id in tasks_to_insert:
            lot = self.lots[l_id]
            plan = self.try_atomic_insertion(state, lot)

            if plan is not None:
                state.lot_record[l_id] = {}
                for o_idx, c in enumerate(plan):
                    m_id, b_idx = c['m_id'], c['b_idx']
                    if c['is_new']:
                        new_b = Batch(m_id, lot.ops[o_idx].recipe, c['start'], c['end'], is_new=True)
                        new_b.lots.append((l_id, o_idx))
                        state.machine_batches[m_id].append(new_b)
                    else:
                        state.machine_batches[m_id][b_idx].lots.append((l_id, o_idx))
                        state.machine_batches[m_id][b_idx].is_new = False

                    state.lot_record[l_id][o_idx] = {'start': c['start'], 'end': c['end'], 'mach': m_id}
            else:
                state.unassigned_lots.add(l_id)

    def destroy_random(self, state: ScheduleState, degree=0.2):
        num_to_remove = max(1, int(len(self.lots) * degree))
        pool = list(state.unassigned_lots) + list(state.lot_record.keys())
        removed_lots = random.sample(pool, min(num_to_remove, len(pool)))

        for l_id in removed_lots:
            if l_id in state.unassigned_lots:
                state.unassigned_lots.remove(l_id)
            if l_id in state.lot_record:
                for o_idx, rec in state.lot_record[l_id].items():
                    m_id = rec['mach']
                    for b in state.machine_batches[m_id]:
                        b.lots = [task for task in b.lots if task[0] != l_id]
                del state.lot_record[l_id]

        for m_id, batches in state.machine_batches.items():
            state.machine_batches[m_id] = [b for b in batches if len(b.lots) > 0]

        return removed_lots

    def optimize(self, time_limit_sec=5):
        print(f"[*] 🚀 启动基于文件的 ALNS 寻优 | 时限: {time_limit_sec} 秒")
        start_time = time.time()

        current_state = ScheduleState()
        tasks = sorted(list(self.lots.keys()), key=lambda x: (self.lots[x].priority, self.lots[x].arr_time))
        self.repair_operator(current_state, tasks)
        self.evaluate_state(current_state)
        self.best_solution = copy.deepcopy(current_state)

        T = 10000
        alpha = 0.95
        iter_count = 0

        while time.time() - start_time < time_limit_sec:
            iter_count += 1
            neighbor = copy.deepcopy(current_state)
            removed = self.destroy_random(neighbor, degree=random.uniform(0.1, 0.3))
            random.shuffle(removed)
            self.repair_operator(neighbor, removed)
            self.evaluate_state(neighbor)

            delta = neighbor.obj_value - current_state.obj_value
            if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-5)):
                current_state = neighbor
                if current_state.obj_value < self.best_solution.obj_value:
                    self.best_solution = copy.deepcopy(current_state)
            T = max(T * alpha, 0.1)

        print("=" * 60)
        print(f"✅ 数据驱动寻优结束 | 迭代: {iter_count} 次")
        print(f"🎯 成功排产 Lots  : {len(self.lots) - len(self.best_solution.unassigned_lots)} / {len(self.lots)}")
        print(f"⏸️ 主动拦截(Hold) : {len(self.best_solution.unassigned_lots)}")
        print(f"📊 总 Cycle Time  : {self.best_solution.total_cycle_time:.1f} min")
        print("=" * 60)

    def plot_gantt_chart(self):
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False

        active_machines = [m_id for m_id, batches in self.best_solution.machine_batches.items() if batches]
        if not active_machines:
            print(">>> 没有调度出任何批次，无法绘制甘特图。")
            return

        active_machines.sort(reverse=True)
        y_pos = {m: i * 10 for i, m in enumerate(active_machines)}

        unique_recipes = set(b.recipe for batches in self.best_solution.machine_batches.values() for b in batches)
        cmap = plt.get_cmap('tab20')
        recipe_colors = {r: mcolors.to_hex(cmap(i % 20)) for i, r in enumerate(sorted(unique_recipes))}

        fig_height = max(5, len(active_machines) * 0.4)
        fig, ax = plt.subplots(figsize=(14, fig_height), dpi=200)

        max_time = 0
        for m_id in active_machines:
            for b in self.best_solution.machine_batches[m_id]:
                start, end, rec, y = b.start_time, b.end_time, b.recipe, y_pos[m_id]
                max_time = max(max_time, end)

                rect = mpatches.Rectangle((start, y - 4), end - start, 8,
                                          facecolor=recipe_colors[rec], edgecolor='black', lw=0.8, alpha=0.85)
                ax.add_patch(rect)

                text_color = 'white' if sum(mcolors.hex2color(recipe_colors[rec])) < 1.5 else 'black'
                ax.text(start + (end - start) / 2, y, f"{rec}\n(N={len(b.lots)})",
                        ha='center', va='center', color=text_color, fontsize=7, fontweight='bold')

        ax.set_yticks(list(y_pos.values()))
        ax.set_yticklabels(list(y_pos.keys()), fontsize=9)
        ax.set_xlabel('时间轴 (Minutes)', fontsize=12, fontweight='bold')
        sched_rate = 100 * (len(self.lots) - len(self.best_solution.unassigned_lots)) / len(self.lots)
        ax.set_title(f'真实数据驱动 ALNS 甘特图 (CNS白名单校验) - 排产率: {sched_rate:.1f}%', fontsize=14, pad=15)
        ax.grid(True, axis='x', linestyle=':', alpha=0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim(0, max_time + 60)

        legend_patches = [mpatches.Patch(color=c, label=r) for r, c in recipe_colors.items()]
        ax.legend(handles=legend_patches, title='加工配方', bbox_to_anchor=(1.01, 1), loc='upper left')

        plt.tight_layout()
        plt.savefig("TSMC_DataDriven_Gantt.png")
        print("\n>>> 现场数据排程甘特图已保存至: TSMC_DataDriven_Gantt.png")


# ==========================================
# 5. 辅助工具：生成样例 CSV 文件
# ==========================================
def generate_sample_csvs_from_text(data_dir="./tsmc_data"):
    """
    如果您还没有将《现场数据初探.txt》整理为 CSV，
    此函数会在初次运行系统时自动为您生成标准的模拟 CSV 供引擎读取。
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"\n[*] 未检测到数据目录 '{data_dir}'，正在初始化创建样本 CSV 文件...")

        # 1. 机台状态
        df_status = pd.DataFrame([
            {'EQPID': 'HDHIT8', 'STATUS': 'RUN', 'REMAIN_RUNTIME': 2.88},
            {'EQPID': 'HDHIT4', 'STATUS': 'RUN', 'REMAIN_RUNTIME': 0.96},
            {'EQPID': 'HDHIT6', 'STATUS': 'LOSS', 'REMAIN_RUNTIME': 0},
            {'EQPID': 'BDHITF', 'STATUS': 'RUN', 'REMAIN_RUNTIME': 3.12},
            {'EQPID': 'BDHITA', 'STATUS': 'DOWN', 'REMAIN_RUNTIME': 0},
            {'EQPID': 'BDHIK2', 'STATUS': 'RUN', 'REMAIN_RUNTIME': 1.5},
            {'EQPID': 'BDHITD', 'STATUS': 'LOSS', 'REMAIN_RUNTIME': 0}
        ])
        df_status.to_csv(os.path.join(data_dir, "machine_status.csv"), index=False)

        # 2. 机台位置
        df_loc = pd.DataFrame([
            {'EQPID': 'HDHIT8', 'MAINGROUP': 'High-Temp Furnace', 'INTERBAYID': 'DF-3B'},
            {'EQPID': 'HDHIT4', 'MAINGROUP': 'High-Temp Furnace', 'INTERBAYID': 'DF-3B'},
            {'EQPID': 'HDHIT6', 'MAINGROUP': 'High-Temp Furnace', 'INTERBAYID': 'DF-3B'},
            {'EQPID': 'BDHITF', 'MAINGROUP': 'High-Temp Furnace', 'INTERBAYID': 'DF-3B'},
            {'EQPID': 'BDHITA', 'MAINGROUP': 'High-Temp Furnace', 'INTERBAYID': 'DF-2C'},
            {'EQPID': 'BDHIK2', 'MAINGROUP': 'PAD/SAC Oxide', 'INTERBAYID': 'DF-1B'},
            {'EQPID': 'BDHITD', 'MAINGROUP': 'PAD/SAC Oxide', 'INTERBAYID': 'DF-2C'}
        ])
        df_loc.to_csv(os.path.join(data_dir, "machine_location.csv"), index=False)

        # 3. CNS 白名单 (工件 - 配方 - 机台约束)
        df_cns = pd.DataFrame([
            {'LOTID': 'HR6943.1', 'STAGE': 'FLD1_OX', 'AUTORECP': 'OHI4000A', 'EQPID': 'HDHIT8'},
            {'LOTID': 'HR6943.1', 'STAGE': 'FLD1_OX', 'AUTORECP': 'OHI4000A', 'EQPID': 'HDHIT4'},
            {'LOTID': 'HCVP43.1', 'STAGE': 'PAD1_OX', 'AUTORECP': 'OPA0110D', 'EQPID': 'BDHIK2'},
            {'LOTID': 'CIRSN65.1', 'STAGE': 'PAD1_OX', 'AUTORECP': 'OPA0110D', 'EQPID': 'BDHITD'},
        ])
        df_cns.to_csv(os.path.join(data_dir, "cns_check.csv"), index=False)

        # 4. 配方时间
        df_rec = pd.DataFrame([
            {'AUTOMATEDRECIPE': 'OHI4000A', 'RT': 4.5},
            {'AUTOMATEDRECIPE': 'OPA0110D', 'RT': 3.47},
            {'AUTOMATEDRECIPE': 'HAN0200B', 'RT': 6.2}
        ])
        df_rec.to_csv(os.path.join(data_dir, "recipe_rt.csv"), index=False)

        # 5. Lot Pool (合并未到站和已到站)
        df_lots = pd.DataFrame([
            {'LOTID': 'HR6943.1', 'STAGE': 'FLD1_OX', 'AUTORECP': 'OHI4000A', 'QTY': 25, 'DISP_BAYID': 'DF-1B',
             'INTIME': 0.3, 'EXTERNALPRIORITY': 555},
            {'LOTID': 'HCVP43.1', 'STAGE': 'PAD1_OX', 'AUTORECP': 'OPA0110D', 'QTY': 25, 'DISP_BAYID': 'DF-2C',
             'INTIME': 0.7, 'EXTERNALPRIORITY': 155},
            {'LOTID': 'HR7131.1', 'STAGE': 'IP1HTO1AN', 'AUTORECP': 'HAN0200B', 'QTY': 25, 'DISP_BAYID': 'DF-3B',
             'INTIME': 1.3, 'EXTERNALPRIORITY': 135},
            {'LOTID': 'CFF7P34.1', 'STAGE': 'FLD1_OX', 'AUTORECP': 'OHI4000A', 'QTY': 25, 'DISP_BAYID': 'DF-3B',
             'INTIME': -2.5, 'EXTERNALPRIORITY': 999},
            {'LOTID': 'CIRSN65.1', 'STAGE': 'PAD1_OX', 'AUTORECP': 'OPA0110D', 'QTY': 25, 'DISP_BAYID': 'DF-1B',
             'INTIME': -1.2, 'EXTERNALPRIORITY': 188},
        ])
        df_lots.to_csv(os.path.join(data_dir, "lots_pool.csv"), index=False)
        print("   ✅ 初始样本 CSV 生成完毕！您可以进入目录修改这些文件以输入您自己的数据。\n")


# ==========================================
# 6. 主执行流程
# ==========================================
if __name__ == "__main__":
    DATA_DIRECTORY = "./tsmc_data"

    # 如果没有 CSV，自动生成样本用于测试
    generate_sample_csvs_from_text(DATA_DIRECTORY)

    # 1. 引擎执行：从外部文件读取真实数据
    df_mach_status, df_mach_loc, df_cns, df_recipe_rt, df_lots = load_tsmc_data_from_csv(DATA_DIRECTORY)

    # 2. 探针解析：将关系型数据转换为运筹学结构，执行 CNS 映射
    lots, machines, recipes_pt, cns_map, t_trans = parse_and_enrich_tsmc_data(
        df_mach_status, df_mach_loc, df_cns, df_recipe_rt, df_lots
    )

    # 3. 算法执行：注入 ALNS
    solver = ALNS_Strict_Scheduler(lots, machines, recipes_pt, cns_map, t_trans)

    # (真实体量的数据可能需要设定 60-120 秒，这里样本集较小，设 5 秒即可)
    solver.optimize(time_limit_sec=5)

    # 4. 可视化输出
    solver.plot_gantt_chart()