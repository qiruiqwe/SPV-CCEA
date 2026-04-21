import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
import shutil
import matplotlib
import numpy as np
from algorithm.SPV_CCEA import SPV_CCEA
from problem.EVCS import EVCS, EVCSProblem
from utilities.load import load_points_information

matplotlib.use('TkAgg')


def set_random_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    print(f"The random seed has been set to {seed}.")


def main(args):
    N = args.N
    R = args.R
    stop_FEs = args.FEs
    pop_size = args.population_num
    num_probes = args.num_probes
    algorithm_name = args.algorithm_name
    run_name = args.run_name
    train = args.train
    seed = args.run_num
    set_random_seed(seed=seed)

    points_path = f"data/guangzhou_foshan_conditate_position_in_road_N{N}/guangzhou_foshan_conditate_position_in_road_N{N}.shp"
    point_gdf = load_points_information(points_path)
    point_coords = np.array([(point.x, point.y) for point in point_gdf.geometry])

    traffic_flow_path = f'./data/{N}_candidate_position_month_traffic_flow'
    month_names = ['一月', '四月', '七月', '十一月']

    evcs_model = EVCS(point_coords, traffic_flow_path, month_names, n_samples=R, ev_penetration=0.58, xl=10, xu=25)

    problem = EVCSProblem(evcs_model)

    print(f'{"=" * 15} {seed} {"=" * 15}')
    set_random_seed(seed=seed)
    if args.run_name == 'parameter_tunning':
        base_result_path = f'results/{algorithm_name}/{run_name}/{N}_{R}/{pop_size}_{num_probes}/{seed}'
    else:
        base_result_path = f'results/{algorithm_name}/{run_name}/{N}_{R}/{seed}'
    solution_path = os.path.join(base_result_path, 'solution.txt')
    piles_path = os.path.join(base_result_path, 'piles_count.txt')
    summary_path = os.path.join(base_result_path, 'summary.txt')
    fitness_history_path = os.path.join(base_result_path, 'fitness_history.csv')

    if train:
        if os.path.exists(base_result_path):
            shutil.rmtree(base_result_path)
            print(f"Delete result: {base_result_path}")
        os.makedirs(base_result_path, exist_ok=True)

        with open(fitness_history_path, 'w', encoding='utf-8') as f:
            f.write('evaluation_count,best_fitness\n')

        solution = None
        piles = None
        best_fitness = None

        if algorithm_name == 'SPV_CCEA':
            algorithm = SPV_CCEA(problem, stop_FEs, pop_size, num_probes, fitness_history_path)
            solution, piles, best_fitness = algorithm.solve()

        np.savetxt(solution_path, solution.reshape(1, -1), fmt='%d',
                   header=f'Switch | Best Fitness: {best_fitness:.6f}', comments='')
        np.savetxt(piles_path, piles.reshape(1, -1), fmt='%d',
                   header=f'Piles | Best Fitness: {best_fitness:.6f}', comments='')

        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"Algorithm: {algorithm_name}\n")
            f.write(f"Seed: {seed}\n")
            f.write(f"Best Fitness: {best_fitness:.6f}\n")
            f.write(f"Total candidate positions: {len(solution)}\n")
            f.write(f"Number of opened stations: {np.sum(solution)}\n")
            f.write(f"Total piles: {np.sum(solution * piles)}\n")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser(description='Training Script')
    parser.add_argument('--algorithm_name', type=str, default='SPV_CCEA')
    parser.add_argument('--run_name', type=str, default='test', help='Name of the run')
    parser.add_argument('--run_num', type=int, default=0, help='seed')
    parser.add_argument('--N', type=int, default=2000, help='Number of candidate position')
    parser.add_argument('--R', type=int, default=25000, help='Number of O-D trips')
    parser.add_argument('--FEs', type=int, default=10000, help='Number of FEs')
    parser.add_argument('--population_num', type=int, default=200)
    parser.add_argument('--train', type=str2bool, default=True)
    parser.add_argument('--num_probes', type=int, default=20)
    parser.add_argument('--GA_max_FEs', type=int, default=2000)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print("Parameter list:")
    for arg in vars(args):
        print(f"{arg:15s} : {getattr(args, arg)}")
    main(args)

