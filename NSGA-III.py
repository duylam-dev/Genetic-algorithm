import random
import json
import numpy as np
import networkx as nx
from collections import defaultdict
from deap import algorithms, base, creator, tools
from scipy.spatial import distance
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go

# Lớp VNFPlacementProblem (giữ nguyên)
class VNFPlacementProblem:
    def __init__(self, network_data):
        self.V = network_data['V']
        self.E = network_data['E']
        self.F = network_data['F']
        self.R = sorted(network_data['R'], key=lambda x: x.get('T', 0))
        
        # Initialize max_time based on the requests
        raw_max = max(req.get('T', 0) + req.get('d_max', 0) for req in self.R)
        self.max_time = int(raw_max) + 1
        self.time_slots = list(range(1, self.max_time + 1))

        self.G = nx.Graph()
        self.vm_nodes = []
        for nid, info in self.V.items():
            node = int(nid)
            self.G.add_node(node)
            if info.get('server', False):
                self.vm_nodes.append(node)
        for e in self.E:
            self.G.add_edge(e['u'], e['v'], bandwidth=e['b_l'], delay=e['d_l'])

    def create_individual(self):
        ind = []
        for r in self.R:
            x = random.randint(0,1)
            ind.append(x)
            if x == 1:
                for _ in r['F_r']:
                    ind.append(random.choice(self.vm_nodes))
                min_t = int(r.get('T', 1))
                d_max = int(r.get('d_max', 1))
                if d_max <= 0:
                    max_t = min_t + 1
                else:
                    max_t = min(min_t + d_max, self.max_time)
                ind.append(random.randint(min_t, max_t))
            else:
                for _ in r['F_r']:
                    ind.append(random.choice(self.vm_nodes))
                ind.append(0)
        return ind

    def decode(self, ind):
        sol = {'x':{}, 'y':{}, 'tau':{}, 'route':{}}
        idx = 0
        for i, r in enumerate(self.R):
            key = f'r{i}'
            x = ind[idx]; sol['x'][key] = x; idx += 1
            if x == 1:
                y_seq = {}
                for k in range(len(r['F_r'])):
                    vm = ind[idx]
                    y_seq[k] = vm; idx += 1
                sol['y'][key] = y_seq
                tau = max(int(ind[idx]), int(r.get('T',1)))
                sol['tau'][key] = tau; idx += 1
                sol['route'][key] = self._route(r, y_seq)
            else:
                y_seq = {}
                for k in range(len(r['F_r'])):
                    vm = ind[idx]
                    y_seq[k] = vm; idx += 1
                sol['y'][key] = y_seq
                sol['tau'][key] = None
                sol['route'][key] = {}
                idx += 1
        return sol

    def _route(self, r, y_seq):
        routing = {}
        prev = r['st_r']
        for k, node in y_seq.items():
            try:
                path = nx.shortest_path(self.G, prev, node)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                path = []
            routing[k] = path
            if path:
                prev = node
        try:
            path = nx.shortest_path(self.G, prev, r['d_r'])
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            path = []
        routing['end'] = path
        return routing

    def evaluate(self, ind):
        sol = self.decode(ind)
        total_cost = 0
        accepted = 0
        total_delay = 0
        cnt = 0
        penalty = 0
        cpu_usage = defaultdict(lambda: defaultdict(int))
        bw_usage = defaultdict(lambda: defaultdict(int))

        for i, r in enumerate(self.R):
            key = f'r{i}'
            if sol['x'][key] == 1:
                accepted += 1
                tau = sol['tau'][key]
                dly = tau - int(r.get('T',1))
                total_delay += dly; cnt += 1
                for k, vm in sol['y'][key].items():
                    info_f = self.F[r['F_r'][k]]
                    info_v = self.V[str(vm)]
                    base = (info_f['c_f']*info_v['cost_c'] + info_f['r_f']*info_v['cost_r'] + info_f['h_f']*info_v['cost_h'])
                    total_cost += base * (1 + 0.1 * dly)
                t_cursor = tau
                for step, path in sol['route'][key].items():
                    if not path:
                        continue
                    delay_path = int(sum(self.G[u][v]['delay'] for u,v in zip(path,path[1:])))
                    for t in range(t_cursor, min(t_cursor+delay_path, self.max_time)+1):
                        for u,v in zip(path,path[1:]):
                            bw_usage[t][(u,v)] += r['b_r']
                    t_cursor += delay_path
                    if step == 'end':
                        continue
                    vm_node = sol['y'][key][step]
                    info_f = self.F[r['F_r'][step]]
                    d_v = int(self.V[str(vm_node)]['d_v'])
                    df_raw = info_f['d_f']
                    d_f = sum(int(val) for val in df_raw.values()) if isinstance(df_raw, dict) else int(df_raw)
                    d_proc = d_v + d_f
                    for t in range(t_cursor, min(t_cursor+d_proc, self.max_time)+1):
                        cpu_usage[t][vm_node] += info_f['c_f']
                    t_cursor += d_proc
                if 'd_max' in r and dly > r['d_max']:
                    penalty += (dly - r['d_max']) * 1000

        for t in self.time_slots:
            for vm in self.vm_nodes:
                cap = self.V[str(vm)]['c_v']
                if cpu_usage[t][vm] > cap:
                    penalty += (cpu_usage[t][vm]-cap)*1000
            for (u,v), used in bw_usage[t].items():
                cap = self.G[u][v]['bandwidth']
                if used > cap:
                    penalty += (used-cap)*1000

        avg_delay = (total_delay/cnt) if cnt>0 else 0
        return total_cost+penalty, -accepted, avg_delay

# Hàm đột biến (mutate_individual)
def mutate_individual(ind, problem):
    new_ind = creator.Individual(ind[:])
    idx = 0
    for r in problem.R:
        if random.random() < 0.1:
            new_ind[idx] = 1 - new_ind[idx]
        x = new_ind[idx]; idx += 1
        if x == 1:
            for _ in r['F_r']:
                if random.random() < 0.2:
                    new_ind[idx] = random.choice(problem.vm_nodes)
                idx += 1
            if random.random() < 0.2:
                min_t = int(r.get('T',1))
                d_max = int(r.get('d_max',1))
                if d_max <= 0:
                    max_t = min_t + 1
                else:
                    max_t = min(min_t + d_max, problem.max_time)
                new_ind[idx] = random.randint(min_t, max_t)
            idx += 1
        else:
            idx += len(r['F_r']) + 1
    return new_ind,

# Hàm tạo điểm tham chiếu cho NSGA-III
def generate_reference_points(num_objectives: int, num_divisions: int) -> np.ndarray:
    def recursive_reference_points(p: int, n: int, m: int, points: list, current: list, index: int) -> None:
        if index == num_objectives - 1:
            current[index] = p / n
            points.append(current[:])
        else:
            for i in range(p + 1):
                current[index] = i / n
                recursive_reference_points(p - i, n, m, points, current, index + 1)
    
    points = []
    recursive_reference_points(num_divisions, num_divisions, num_objectives, points, [0] * num_objectives, 0)
    return np.array(points)

# Hàm chuẩn hóa mục tiêu
def normalize_objectives(objectives: np.ndarray, ideal_point: np.ndarray, worst_point: np.ndarray) -> np.ndarray:
    return (objectives - ideal_point) / (worst_point - ideal_point + 1e-6)

# Hàm lựa chọn NSGA-III
def sel_nsga3(individuals, k, ref_points):
    # Non-dominated sorting
    fronts = tools.sortNondominated(individuals, len(individuals), first_front_only=False)
    
    # Chọn các cá thể cho đến front cuối cùng
    selected = []
    last_front = None
    for i, front in enumerate(fronts):
        if len(selected) + len(front) <= k:
            selected.extend(front)
            for ind in front:
                ind.rank = i
        else:
            last_front = front
            for ind in last_front:
                ind.rank = i
            break
    
    # Nếu cần thêm cá thể từ front cuối cùng
    if len(selected) < k and last_front:
        # Tìm điểm lý tưởng và điểm tệ nhất
        objectives = np.array([ind.fitness.values for ind in individuals])
        ideal_point = np.min(objectives, axis=0)
        worst_point = np.max(objectives, axis=0)
        
        # Chuẩn hóa mục tiêu của front cuối cùng
        norm_objs = normalize_objectives(np.array([ind.fitness.values for ind in last_front]), ideal_point, worst_point)
        
        # Gán cá thể vào điểm tham chiếu
        for ind_idx, ind in enumerate(last_front):
            distances = [distance.euclidean(norm_objs[ind_idx], ref) for ref in ref_points]
            ind.associated_ref = np.argmin(distances)
            ind.distance_to_ref = distances[ind.associated_ref]
        
        # Đếm số cá thể trong mỗi niche
        niche_count = {i: 0 for i in range(len(ref_points))}
        for ind in selected:
            if hasattr(ind, 'associated_ref') and ind.associated_ref is not None:
                niche_count[ind.associated_ref] += 1
        
        # Chọn cá thể từ front cuối cùng dựa trên niche
        sorted_last_front = sorted(last_front, key=lambda x: (niche_count[x.associated_ref], x.distance_to_ref))
        selected.extend(sorted_last_front[:k - len(selected)])
    
    return selected

# Hàm vẽ Pareto Front 3D tĩnh (matplotlib)
def plot_pareto_front(hof):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Lấy giá trị mục tiêu từ hall of fame
    objectives = np.array([ind.fitness.values for ind in hof])
    costs = objectives[:, 0]
    accepted = -objectives[:, 1]  # Chuyển -accepted thành accepted
    delays = objectives[:, 2]
    
    # Vẽ scatter plot 3D
    ax.scatter(costs, accepted, delays, c='blue', marker='o')
    
    ax.set_xlabel('Total Cost')
    ax.set_ylabel('Accepted Requests')
    ax.set_zlabel('Average Delay')
    ax.set_title('Pareto Front (NSGA-III)')
    
    # Lưu biểu đồ
    plt.savefig('pareto_front_nsga3.png')
    plt.close()

# Hàm vẽ Pareto Front 3D tương tác (plotly)
def plot_pareto_front_interactive(hof):
    # Lấy giá trị mục tiêu từ hall of fame
    objectives = np.array([ind.fitness.values for ind in hof])
    costs = objectives[:, 0]
    accepted = -objectives[:, 1]  # Chuyển -accepted thành accepted
    delays = objectives[:, 2]
    
    # Tạo scatter plot 3D tương tác
    fig = go.Figure(data=[
        go.Scatter3d(
            x=costs,
            y=accepted,
            z=delays,
            mode='markers',
            marker=dict(
                size=5,
                color='blue',
                opacity=0.8
            ),
            text=[f'Cost: {c:.2f}, Accepted: {a:.0f}, Delay: {d:.2f}' for c, a, d in zip(costs, accepted, delays)],
            hoverinfo='text'
        )
    ])
    
    # Cài đặt layout
    fig.update_layout(
        title='Interactive Pareto Front (NSGA-III)',
        scene=dict(
            xaxis_title='Total Cost',
            yaxis_title='Accepted Requests',
            zaxis_title='Average Delay'
        ),
        width=800,
        height=600
    )
    
    # Lưu biểu đồ dưới dạng HTML
    fig.write_html('pareto_front_nsga3_interactive.html')

# Cài đặt NSGA-III
def setup_nsga3():
    if hasattr(creator, 'FitnessMulti'):
        del creator.FitnessMulti
    if hasattr(creator, 'Individual'):
        del creator.Individual
    creator.create('FitnessMulti', base.Fitness, weights=(-1.0, -1.0, -1.0))
    creator.create('Individual', list, fitness=creator.FitnessMulti, associated_ref=None, distance_to_ref=None)

# Chạy NSGA-III
def run_nsga3(network_data, pop_size=50, gen=100, num_divisions=12):
    problem = VNFPlacementProblem(network_data)
    setup_nsga3()
    
    # Tạo điểm tham chiếu
    num_objectives = 3  # cost, -accepted, avg_delay
    ref_points = generate_reference_points(num_objectives, num_divisions)
    
    # Cài đặt toolbox
    toolbox = base.Toolbox()
    toolbox.register('individual', lambda: creator.Individual(problem.create_individual()))
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    toolbox.register('evaluate', problem.evaluate)
    toolbox.register('mate', tools.cxTwoPoint)
    toolbox.register('mutate', lambda ind: mutate_individual(ind, problem))
    toolbox.register('select', sel_nsga3, ref_points=ref_points)
    
    # Tạo quần thể ban đầu
    pop = toolbox.population(n=pop_size)
    invalid = [ind for ind in pop if not ind.fitness.valid]
    for ind in invalid:
        ind.fitness.values = toolbox.evaluate(ind)
    
    # Khởi tạo hall of fame và thống kê
    hof = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register('avg', np.mean, axis=0)
    stats.register('min', np.min, axis=0)
    
    # Chạy thuật toán NSGA-III
    for g in range(gen):
        offspring = algorithms.varAnd(pop, toolbox, cxpb=0.8, mutpb=0.2)
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = toolbox.evaluate(ind)
        pop = toolbox.select(pop + offspring, pop_size)
        hof.update(pop)
        record = stats.compile(pop)
    
    # Vẽ Pareto Front (tĩnh và tương tác)
    plot_pareto_front(hof)
    plot_pareto_front_interactive(hof)
    
    return pop, hof, problem

# Main
def main():
    with open('cogent_centers_easy_s1.json') as f:
        data = json.load(f)
    pop, hof, prob = run_nsga3(data, pop_size=30, gen=50, num_divisions=12)
    with open('vnf_time_aware_nsga3_output.txt', 'w') as f:
        f.write(f'Found {len(hof)} Pareto solutions\n')
        for i, ind in enumerate(hof):
            sol = prob.decode(ind)
            cost, acc_neg, avg_d = ind.fitness.values
            f.write(f'--- Solution {i+1} ---\n')
            f.write(f'Cost: {cost:.2f}\n')
            f.write(f'Accepted: {-acc_neg}\n')
            f.write(f'Avg Delay: {avg_d:.2f}\n')
            f.write(f"x_r: {sol['x']}\n")
            f.write(f"y_r: {sol['y']}\n")
            f.write(f"z (routes): {sol['route']}\n")
            f.write(f"tau: {sol['tau']}\n")

if __name__ == '__main__':
    main()