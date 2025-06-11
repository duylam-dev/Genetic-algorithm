import random
import networkx as nx
from collections import defaultdict

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
        #add nodes to the graph
        for nid, info in self.V.items():
            node = int(nid)
            self.G.add_node(node)
            if info.get('server', False):
                self.vm_nodes.append(node)
        #add edges to the graph
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
                    penalty += (dly - r['d_max']) * 100

        for t in self.time_slots:
            for vm in self.vm_nodes:
                cap = self.V[str(vm)]['c_v']
                if cpu_usage[t][vm] > cap:
                    penalty += (cpu_usage[t][vm]-cap)*100
            for (u,v), used in bw_usage[t].items():
                cap = self.G[u][v]['bandwidth']
                if used > cap:
                    penalty += (used-cap)*100

        avg_delay = (total_delay/cnt) if cnt>0 else 0
        return total_cost+penalty, -accepted, avg_delay
