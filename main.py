import json
import random
import numpy as np
from deap import algorithms, base, creator, tools
from vnf_ploblem import VNFPlacementProblem
from visualize import plot_pareto_2d, draw_network_graph


# NSGA-II setup
# create fitness class and individual class
def setup_nsga2():
    if hasattr(creator, 'FitnessMulti'):
        del creator.FitnessMulti
    if hasattr(creator, 'Individual'):
        del creator.Individual
    creator.create('FitnessMulti', base.Fitness, weights=(-1.0, -1.0, -1.0))
    creator.create('Individual', list, fitness=creator.FitnessMulti)

# Mutation operator

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

# Run NSGA-II

def run_nsga2(network_data, pop_size=50, gen=30):
    # Initialize the problem instance
    problem = VNFPlacementProblem(network_data)
    draw_network_graph(problem, filename='network_graph.png')
    # Setup DEAP NSGA-II components
    setup_nsga2()
    # Create DEAP toolbox
    toolbox = base.Toolbox()
    toolbox.register('individual', lambda: creator.Individual(problem.create_individual()))
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    toolbox.register('evaluate', problem.evaluate)
    toolbox.register('mate', tools.cxTwoPoint)
    toolbox.register('mutate', lambda ind: mutate_individual(ind, problem))
    toolbox.register('select', tools.selNSGA2)

    # Create initial population
    pop = toolbox.population(n=pop_size)
    # Evaluate the initial population
    invalid = [ind for ind in pop if not ind.fitness.valid]
    for ind in invalid:
        ind.fitness.values = toolbox.evaluate(ind)
    # Initialize hall of fame and statistics
    hof = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register('avg', np.mean, axis=0)
    stats.register('min', np.min, axis=0)
    # Run the NSGA-II algorithm
    for g in range(gen):
        offspring = algorithms.varAnd(pop, toolbox, cxpb=0.8, mutpb=0.2)
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = toolbox.evaluate(ind)
        pop = toolbox.select(pop + offspring, pop_size)
        hof.update(pop)
        record = stats.compile(pop)
    return pop, hof, problem




# Evaluation Metrics for Pareto Front
def evaluate_pareto_front(hof):
    if len(hof) < 2:
        return {
            'Cardinality': len(hof),
            'Spacing': None,
            'Spread': None,
            'Convergence': None
        }

    fitnesses = np.array([ind.fitness.values for ind in hof])
    cardinality = len(hof)

    # Spacing
    distances = []
    for i in range(len(fitnesses)):
        dists = [np.linalg.norm(fitnesses[i] - fitnesses[j]) for j in range(len(fitnesses)) if i != j]
        distances.append(min(dists))
    spacing = np.std(distances)

    # Spread (Δ - Spread metric)
    df = fitnesses - fitnesses.min(axis=0)
    ideal = np.zeros(fitnesses.shape[1])  # Assuming minimization problem
    d_f = np.max(np.linalg.norm(fitnesses - ideal, axis=1))
    d_l = np.min(np.linalg.norm(fitnesses - ideal, axis=1))
    mean_dist = np.mean(distances)
    spread = (d_f + d_l + sum(abs(d - mean_dist) for d in distances)) / (d_f + d_l + (len(distances) * mean_dist))

    # Convergence: distance to ideal point (0,0,0)
    convergence = np.mean(np.linalg.norm(fitnesses - ideal, axis=1))

    return {
        'Cardinality': cardinality,
        'Spacing': spacing,
        'Spread': spread,
        'Convergence': convergence
    }


# Main

def main():
    #load data
    with open('input/input_25/cogent_centers_easy_s2.json') as f:
        data = json.load(f)
    # Run NSGA-II
    pop, hof, prob = run_nsga2(data, pop_size=30, gen=300)
    metrics = evaluate_pareto_front(hof)
    # Save results
    with open('vnf_time_aware_output.txt', 'w') as f:
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
        f.write('\n=== Pareto Front Metrics ===\n')
        for key, value in metrics.items():
            f.write(f'{key}: {value}\n')
    plot_pareto_2d(hof)

if __name__ == '__main__':
    main()
