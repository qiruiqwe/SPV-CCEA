import copy

import numpy as np


class UnifiedGA:

    def __init__(self, func, x, y, fitness, pos, lb, ub, start_FE, pop_size,
                 max_fes, fitness_history, survival_probs, opt_target='siting'):
        self.func = func
        self.x = np.copy(x)
        self.y = np.copy(y)
        self.y_index = pos
        self.dim = len(pos)
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.pop_size = max(10, pop_size)
        self.max_fes = max_fes
        self.start_fes = start_FE
        self.fes_now = 0
        self.fitness_history = fitness_history
        self.survival_probs = survival_probs

        self.opt_target = opt_target
        self.population = np.zeros((self.pop_size, self.dim), dtype=int)
        self.fitness = np.full(self.pop_size, np.inf)
        self.gbest_fitness = copy.deepcopy(fitness)

        self.pc = 0.8
        self.pm = 1.0 / max(1, self.dim)

    def _init(self):
        if self.opt_target == 'siting':
            seed = np.copy(self.x[self.y_index])
            self.population[0] = seed
            self.gbest_solution = np.copy(seed)

            init_perturb_rate = 2.0 / max(1, self.dim)

            for i in range(1, self.pop_size):
                if i < self.pop_size // 2:
                    mutated_seed = np.copy(seed)
                    mask = np.random.rand(self.dim) < init_perturb_rate

                    for j in range(self.dim):
                        if mask[j]:
                            prob_stay_alive = self.survival_probs[j]
                            if mutated_seed[j] == 0:
                                if np.random.rand() < prob_stay_alive:
                                    mutated_seed[j] = 1
                            else:
                                if np.random.rand() < (1.0 - prob_stay_alive):
                                    mutated_seed[j] = 0

                    self.population[i] = mutated_seed
                else:
                    self.population[i] = (np.random.rand(self.dim) < self.survival_probs).astype(int)

    def _evaluate(self, chrom):
        if self.opt_target == 'siting':
            temp_switches = np.copy(self.x)
            temp_switches[self.y_index] = chrom
            return self.func(temp_switches, self.y)

    def solve(self):
        print('    >>> Starting GA ...')
        self._init()

        for i in range(self.pop_size):
            if self.fes_now >= self.max_fes:
                break
            self.fitness[i] = self._evaluate(self.population[i])
            self.fes_now += 1
            if self.fitness[i] < self.gbest_fitness:
                self.gbest_fitness = self.fitness[i]
                self.gbest_solution = np.copy(self.population[i])
            print(f"    >>> FEs: {self.start_fes+self.fes_now}, fitness: {self.gbest_fitness:.4f}")

            with open(self.fitness_history, 'a', encoding='utf-8') as f:
                f.write(f'{self.start_fes+self.fes_now},{self.gbest_fitness:.6f}\n')

        while self.fes_now < self.max_fes:
            new_population = np.zeros_like(self.population)
            new_population[0] = np.copy(self.population[np.argmin(self.fitness)])

            for i in range(1, self.pop_size, 2):
                p1, p2 = self._tournament_selection(), self._tournament_selection()
                c1, c2 = self._uniform_crossover(self.population[p1], self.population[p2])
                c1, c2 = self._mutate(c1), self._mutate(c2)

                new_population[i] = c1
                if i + 1 < self.pop_size:
                    new_population[i + 1] = c2

            self.population = new_population

            for i in range(1, self.pop_size):
                if self.fes_now >= self.max_fes:
                    break
                self.fitness[i] = self._evaluate(self.population[i])
                self.fes_now += 1
                if self.fitness[i] < self.gbest_fitness:
                    self.gbest_fitness = self.fitness[i]
                    self.gbest_solution = np.copy(self.population[i])

                print(f"    >>> FEs: {self.start_fes+self.fes_now}, fitness: {self.gbest_fitness:.4f}")

                with open(self.fitness_history, 'a', encoding='utf-8') as f:
                    f.write(f'{self.start_fes+self.fes_now},{self.gbest_fitness:.6f}\n')

        return self.gbest_solution, self.gbest_fitness, self.fes_now

    def _tournament_selection(self, k=3):
        indices = np.random.choice(self.pop_size, size=k, replace=False)
        return indices[np.argmin(self.fitness[indices])]

    def _uniform_crossover(self, p1, p2):
        if np.random.rand() < self.pc:
            mask = np.random.rand(self.dim) < 0.5
            return np.where(mask, p1, p2), np.where(mask, p2, p1)
        return np.copy(p1), np.copy(p2)

    def _mutate(self, chrom):
        mask = np.random.rand(self.dim) < self.pm
        if not np.any(mask):
            return chrom

        if self.opt_target == 'siting':
            mask = np.random.rand(self.dim) < self.pm

            for i in range(self.dim):
                if mask[i]:
                    prob_stay_alive = self.survival_probs[i]

                    if chrom[i] == 0:
                        if np.random.rand() < prob_stay_alive:
                            chrom[i] = 1
                    else:
                        if np.random.rand() < (1.0 - prob_stay_alive):
                            chrom[i] = 0
        return chrom