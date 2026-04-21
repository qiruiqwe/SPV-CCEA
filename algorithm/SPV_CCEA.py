import copy
import numpy as np
from scipy.spatial import cKDTree
from algorithm.UnifiedGA import UnifiedGA


class Individual:
    def __init__(self, x, y, fitness):
        self.m = len(x)
        self.solution = x
        self.piles = y
        self.fitness = fitness


class SPV_CCEA:
    """survival probability vector-guided cooperative co-evolutionary algorithm(SPV-CCEA)"""
    def __init__(self, problem, stop_FEs, pop_size, num_probes, fitness_history_path):
        self.individuals = None
        self.best_individual = None
        self.problem = problem
        self.xl = problem.xl
        self.xu = problem.xu
        self.n = int(problem.n_var / 2)
        self.stop_FEs = stop_FEs
        self.pop_size = pop_size
        self.num_probes = num_probes
        self.current_fe = 0
        self.epsilon = 1e-4
        self.fitness_history = fitness_history_path
        self.coords = self.problem.evcs.coords

    def _log_progress(self, stage, outer_gen=None, fitness=None, fe_used=None):
        """Unified log output format"""
        log_parts = [
            f"[SPV-CCEA] {stage}",
            f"Outer Iteration: {outer_gen if outer_gen is not None else 'N/A'}",
            f"Current Fitness: {fitness:.4f}" if fitness is not None else "",
            f"Cumulative FEs: {self.current_fe}/{self.stop_FEs}",
            f"FEs Used in This Stage: {fe_used}" if fe_used is not None else ""
        ]
        log_str = " | ".join([p for p in log_parts if p])
        print(log_str)

    def calculate_fitness(self, x, y):
        x = np.floor(x + 0.5)
        y = np.floor(y + 0.5)
        [total_annual_objective, _, _] = self.problem.evcs._evaluate(x * y)
        return total_annual_objective

    def TEP(self):
        """
        Topology-varied Ensemble Probing Mechanism
        """
        num_probes = self.num_probes
        print(f"    >>> [TEP] Starting Topology-varied Ensemble Probing ({num_probes} probes)...")

        total_utilizations = np.zeros(self.n)
        open_counts = np.zeros(self.n)
        best_probe_switches = None
        best_probe_piles = None
        best_probe_fit = float('inf')

        sparsities = np.linspace(0.1, 0.9, num_probes)

        for sparsity in sparsities:
            probe_switches = (np.random.rand(self.n) < sparsity).astype(int)

            if np.sum(probe_switches) == 0:
                probe_switches[np.random.randint(0, self.n)] = 1

            probe_piles = np.copy(self.xu[self.n:])
            [probe_fit, util_rates, _] = self.problem.evcs._evaluate(probe_switches * probe_piles)

            self.current_fe += 1

            if probe_fit < best_probe_fit:
                best_probe_fit = probe_fit
                best_probe_switches = np.copy(probe_switches)
                best_probe_piles = np.copy(probe_piles)

            print(f"    >>> [TEP] FEs: {self.current_fe}, fitness: {best_probe_fit:.4f}")
            with open(self.fitness_history, 'a', encoding='utf-8') as f:
                f.write(f'{self.current_fe},{best_probe_fit:.6f}\n')

            active_mask = (probe_switches > 0)
            total_utilizations[active_mask] += util_rates[active_mask]
            open_counts[active_mask] += 1
        self.best_individual = Individual(best_probe_switches, best_probe_piles, best_probe_fit)

        avg_utils = np.zeros(self.n)
        valid_mask = (open_counts > 0)
        avg_utils[valid_mask] = total_utilizations[valid_mask] / open_counts[valid_mask]

        survival_probs = np.full(self.n, 0.05)

        active_avg_utils = avg_utils[valid_mask]
        if len(active_avg_utils) > 0:
            max_u = np.max(active_avg_utils)
            min_u = np.min(active_avg_utils)
            if max_u > min_u:
                scaled_probs = 0.2 + 0.7 * (avg_utils[valid_mask] - min_u) / (max_u - min_u)
                survival_probs[valid_mask] = scaled_probs
            else:
                survival_probs[valid_mask] = 0.5

        return survival_probs

    def spatial_relocation_search(self, current_switches, current_piles, fitness, K=5):
        cumulative_fes = 0
        print('     >>> Starting Spatial Relocation Search (SRO)...')

        switches = np.copy(current_switches)
        piles = np.copy(current_piles)

        global_best_fitness = float(fitness) if fitness is not None else float('inf')
        global_best_switches = np.copy(switches)
        global_best_piles = np.copy(piles)

        active_indices = np.where(switches > 0)[0]
        if len(active_indices) == 0:
            return global_best_switches, global_best_piles, global_best_fitness, cumulative_fes

        [_, utilization_rates, _] = self.problem.evcs._evaluate(switches * piles)
        cumulative_fes += 1

        active_utilizations = utilization_rates[active_indices]
        sorted_relative_indices = np.argsort(-active_utilizations)
        sorted_active_indices = active_indices[sorted_relative_indices]

        coords = self.coords

        for idx in sorted_active_indices:
            no_active_indices = np.where(switches == 0)[0]
            if len(no_active_indices) == 0:
                break

            current_coord = coords[idx]
            no_active_coords = coords[no_active_indices]

            kdtree = cKDTree(no_active_coords)
            K_NEIGHBORS = min(K, len(no_active_indices))
            distances, local_indices = kdtree.query(current_coord, k=K_NEIGHBORS)

            if K_NEIGHBORS == 1:
                local_indices = [local_indices]

            local_best_fitness = global_best_fitness
            local_best_switches = np.copy(switches)
            local_best_piles = np.copy(piles)
            improved = False

            for local_i in local_indices:
                candidate_idx = no_active_indices[local_i]

                temp_switches = np.copy(switches)
                temp_piles = np.copy(piles)

                temp_switches[idx] = 0
                temp_switches[candidate_idx] = 1

                transferred_piles = piles[idx]
                new_lb = int(self.xl[self.n + candidate_idx])
                new_ub = int(self.xu[self.n + candidate_idx])

                temp_piles[candidate_idx] = np.clip(transferred_piles, new_lb, new_ub)
                temp_piles[idx] = int(self.xu[self.n + idx])

                new_fitness = self.calculate_fitness(temp_switches, temp_piles)
                cumulative_fes += 1

                if new_fitness < local_best_fitness:
                    improved = True
                    local_best_fitness = new_fitness
                    local_best_switches = np.copy(temp_switches)
                    local_best_piles = np.copy(temp_piles)

                    print(
                        f'     >>> FEs: {self.current_fe + cumulative_fes}, Relocated station {idx} -> {candidate_idx}, New fitness: {local_best_fitness:.4f}')

                if self.current_fe + cumulative_fes >= self.stop_FEs:
                    with open(self.fitness_history, 'a', encoding='utf-8') as f:
                        f.write(f'{self.current_fe + cumulative_fes},{global_best_fitness:.6f}\n')
                    return global_best_switches, global_best_piles, global_best_fitness, cumulative_fes

            if improved:
                switches = np.copy(local_best_switches)
                piles = np.copy(local_best_piles)

                if local_best_fitness < global_best_fitness:
                    global_best_fitness = local_best_fitness
                    global_best_switches = np.copy(local_best_switches)
                    global_best_piles = np.copy(local_best_piles)
            with open(self.fitness_history, 'a', encoding='utf-8') as f:
                f.write(f'{self.current_fe + cumulative_fes},{global_best_fitness:.6f}\n')
        print(f'     <<< SRO finished. FEs consumed: {cumulative_fes}, Final fitness: {global_best_fitness:.4f}')
        return global_best_switches, global_best_piles, global_best_fitness, cumulative_fes

    def modify_piles(self, current_switches, current_piles, fitness):
        cumulative_fes = 0
        best_fitness = fitness
        active_station_indices = np.where(current_switches > 0)[0]

        for global_idx in active_station_indices:

            if self.current_fe + cumulative_fes >= self.stop_FEs:
                return current_switches, current_piles, best_fitness, cumulative_fes

            lb = int(self.xl[self.n + global_idx])
            ub = int(self.xu[self.n + global_idx])
            current_c = current_piles[global_idx]

            if current_c < lb:
                current_c = lb
                current_piles[global_idx] = lb
                best_fitness = self.calculate_fitness(current_switches, current_piles)
                cumulative_fes += 1
                if self.current_fe + cumulative_fes >= self.stop_FEs:
                    return current_switches, current_piles, best_fitness, cumulative_fes

            if current_c > ub:
                current_c = ub
                current_piles[global_idx] = ub
                best_fitness = self.calculate_fitness(current_switches, current_piles)
                cumulative_fes += 1
                if self.current_fe + cumulative_fes >= self.stop_FEs:
                    return current_switches, current_piles, best_fitness, cumulative_fes

            if lb == ub:
                continue

            dir_decrease = False
            dir_increase = False

            fit_minus_1 = float('inf')
            fit_plus_1 = float('inf')

            if current_c > lb:
                temp_piles = copy.deepcopy(current_piles)
                temp_piles[global_idx] = current_c - 1
                fit_minus_1 = self.calculate_fitness(current_switches, temp_piles)
                cumulative_fes += 1

                if self.current_fe + cumulative_fes >= self.stop_FEs:
                    return current_switches, current_piles, best_fitness, cumulative_fes

                if fit_minus_1 < best_fitness:
                    dir_decrease = True

            if not dir_decrease and current_c < ub:
                temp_piles = copy.deepcopy(current_piles)
                temp_piles[global_idx] = current_c + 1
                fit_plus_1 = self.calculate_fitness(current_switches, temp_piles)
                cumulative_fes += 1

                if self.current_fe + cumulative_fes >= self.stop_FEs:
                    return current_switches, current_piles, best_fitness, cumulative_fes

                if fit_plus_1 < best_fitness:
                    dir_increase = True

            if dir_decrease:
                L, R = lb, current_c - 1
                local_best_c = current_c - 1
                local_best_fit = fit_minus_1
            elif dir_increase:
                L, R = current_c + 1, ub
                local_best_c = current_c + 1
                local_best_fit = fit_plus_1
            else:
                continue

            while L < R:
                mid = (L + R) // 2

                temp_piles = copy.deepcopy(current_piles)
                temp_piles[global_idx] = mid
                fit_mid = self.calculate_fitness(current_switches, temp_piles)

                temp_piles[global_idx] = mid + 1
                fit_mid_plus = self.calculate_fitness(current_switches, temp_piles)
                cumulative_fes += 2

                if self.current_fe + cumulative_fes >= self.stop_FEs:
                    return current_switches, current_piles, best_fitness, cumulative_fes

                if fit_mid < local_best_fit:
                    local_best_fit, local_best_c = fit_mid, mid
                if fit_mid_plus < local_best_fit:
                    local_best_fit, local_best_c = fit_mid_plus, mid + 1

                if fit_mid < fit_mid_plus:
                    R = mid
                else:
                    L = mid + 1

            if local_best_fit < best_fitness:
                direction_str = "decreased" if local_best_c < current_c else "increased"
                print(
                    f'    >>> FEs: {self.current_fe + cumulative_fes} | Station {global_idx} piles {direction_str} from {current_c} to {local_best_c}. Best fitness: {local_best_fit:.4f}')

                current_piles[global_idx] = local_best_c
                best_fitness = local_best_fit

            with open(self.fitness_history, 'a', encoding='utf-8') as f:
                f.write(f'{self.current_fe + cumulative_fes},{best_fitness:.6f}\n')

        return current_switches, current_piles, best_fitness, cumulative_fes

    def close_station(self, current_switches, current_piles):
        cumulative_fes = 0

        current_switches = np.round(current_switches).astype(int)
        current_piles = np.round(current_piles).astype(int)

        [best_fitness, utilization_rates, _] = self.problem.evcs._evaluate(current_switches * current_piles)
        cumulative_fes += 1

        with open(self.fitness_history, 'a', encoding='utf-8') as f:
            f.write(f'{self.current_fe + cumulative_fes},{best_fitness:.6f}\n')

        if self.current_fe + cumulative_fes >= self.stop_FEs:
            return current_switches, current_piles, best_fitness, cumulative_fes

        active_station_indices = np.where(current_switches > 0)[0]
        zero_utilization_mask = utilization_rates[active_station_indices] == 0
        zero_utilization_indices = active_station_indices[zero_utilization_mask]

        if len(zero_utilization_indices) > 0:
            current_switches[zero_utilization_indices] = 0
            [best_fitness, utilization_rates, _] = self.problem.evcs._evaluate(current_switches * current_piles)
            cumulative_fes += 1

            if self.current_fe + cumulative_fes >= self.stop_FEs:
                return current_switches, current_piles, best_fitness, cumulative_fes
            print(
                f'    >>> FEs: {self.current_fe + cumulative_fes}, Closing {len(zero_utilization_indices)} stations with strictly zero utilization, fitness: {best_fitness:.4f}')
            with open(self.fitness_history, 'a', encoding='utf-8') as f:
                f.write(f'{self.current_fe + cumulative_fes},{best_fitness:.6f}\n')

        active_station_indices = np.where(current_switches > 0)[0]
        active_utilizations = utilization_rates[active_station_indices]
        lowest_util_relative_indices = np.argsort(active_utilizations)
        candidates_to_close = active_station_indices[lowest_util_relative_indices]

        for close_idx in candidates_to_close:
            temp_switches = np.copy(current_switches)
            temp_piles = np.copy(current_piles)
            temp_switches[close_idx] = 0
            new_fitness = self.calculate_fitness(temp_switches, temp_piles)
            cumulative_fes += 1

            if new_fitness < best_fitness:
                current_switches = np.copy(temp_switches)
                current_piles = np.copy(temp_piles)
                best_fitness = new_fitness

                print(
                    f'     >>> FEs: {self.current_fe + cumulative_fes}, Closed  station ID: {close_idx}, fitness: {best_fitness:.4f}')

            with open(self.fitness_history, 'a', encoding='utf-8') as f:
                f.write(f'{self.current_fe + cumulative_fes},{best_fitness:.6f}\n')

            if self.current_fe + cumulative_fes >= self.stop_FEs:
                return current_switches, current_piles, best_fitness, cumulative_fes

        return current_switches, current_piles, best_fitness, cumulative_fes

    def solve(self):
        generation = 0
        self._log_progress(
            stage="Starting probing"
        )
        survival_probs = self.TEP()

        self._log_progress(
            stage="Ending probing",
            outer_gen=generation,
            fitness=self.best_individual.fitness,
            fe_used=self.num_probes
        )

        COEVOLUTION_CONFIG = {
            "MAX_OUTER_LOOP": 1500,
            "INNER_FES": 2000,
            "INNER_POP_SIZE": self.pop_size
        }

        for outer_gen in range(1, COEVOLUTION_CONFIG["MAX_OUTER_LOOP"] + 1):
            generation += 1

            if self.current_fe >= self.stop_FEs:
                break

            # ================= Stage 1: Siting Space Search =================
            self._log_progress(stage="Starting siting space search",
                               outer_gen=outer_gen,
                               fitness=self.best_individual.fitness)

            pos = np.arange(self.n)
            lb = self.xl[:self.n]
            ub = self.xu[:self.n]

            optimizer = UnifiedGA(
                func=self.calculate_fitness,
                x=self.best_individual.solution,
                y=self.best_individual.piles,
                fitness=self.best_individual.fitness,
                pos=pos,
                lb=lb,
                ub=ub,
                start_FE=self.current_fe,
                pop_size=COEVOLUTION_CONFIG["INNER_POP_SIZE"],
                max_fes=COEVOLUTION_CONFIG["INNER_FES"],
                fitness_history=self.fitness_history,
                opt_target='siting',
                survival_probs=survival_probs[pos]
            )
            solution, best_fit, cumulative_fes = optimizer.solve()
            self.best_individual.solution[pos] = copy.deepcopy(solution)
            self.best_individual.fitness = copy.deepcopy(best_fit)
            self.current_fe += cumulative_fes

            self._log_progress(
                stage="Ending Siting Space Search",
                outer_gen=outer_gen,
                fitness=self.best_individual.fitness,
                fe_used=COEVOLUTION_CONFIG["INNER_FES"]
            )

            if self.current_fe >= self.stop_FEs:
                break

            # ================= Stage 2: Close station local search =================
            self._log_progress(stage="Starting close station local search",
                               outer_gen=outer_gen,
                               fitness=self.best_individual.fitness)
            best_x, best_y, best_fit, cumulative_fes = self.close_station(
                current_switches=self.best_individual.solution,
                current_piles=self.best_individual.piles,
            )
            self.current_fe += cumulative_fes
            self.best_individual.solution = copy.deepcopy(best_x)
            self.best_individual.piles = copy.deepcopy(best_y)
            self.best_individual.fitness = best_fit
            self._log_progress(stage="Ending close station local search",
                               outer_gen=outer_gen,
                               fitness=self.best_individual.fitness,
                               fe_used=cumulative_fes)

            # ================= Stage 3: Piles local search =================
            self._log_progress(stage="Starting piles local search",
                               outer_gen=outer_gen,
                               fitness=self.best_individual.fitness)
            best_x, best_y, best_fit, cumulative_fes = self.modify_piles(
                current_switches=self.best_individual.solution,
                current_piles=self.best_individual.piles,
                fitness=self.best_individual.fitness
            )
            self.current_fe += cumulative_fes
            self.best_individual.solution = copy.deepcopy(best_x)
            self.best_individual.piles = copy.deepcopy(best_y)
            self.best_individual.fitness = best_fit
            self._log_progress(
                stage="Ending piles local search",
                outer_gen=outer_gen,
                fitness=self.best_individual.fitness,
                fe_used=cumulative_fes,
            )

            # ================= Stage 4: Stations Neighborhood Replacement =================
            self._log_progress(stage="Starting stations neighborhood replacement",
                               outer_gen=outer_gen,
                               fitness=self.best_individual.fitness)
            best_x, best_y, best_fit, cumulative_fes = self.spatial_relocation_search(
                current_switches=self.best_individual.solution,
                current_piles=self.best_individual.piles,
                fitness=self.best_individual.fitness,
                K=10
            )
            self.current_fe += cumulative_fes
            self.best_individual.solution = copy.deepcopy(best_x)
            self.best_individual.piles = copy.deepcopy(best_y)
            self.best_individual.fitness = best_fit
            self._log_progress(
                stage="Ending stations neighborhood replacement",
                outer_gen=outer_gen,
                fitness=self.best_individual.fitness,
                fe_used=cumulative_fes,
            )

            # ================= Stage 5  =================
            [_, latest_utils, _] = self.problem.evcs._evaluate(
                self.best_individual.solution * self.best_individual.piles)

            active_mask = (latest_utils > 0)
            fresh_probs = np.full(self.n, 0.05)

            if np.any(active_mask):
                max_u = np.max(latest_utils[active_mask])
                min_u = np.min(latest_utils[active_mask])
                if max_u > min_u:
                    fresh_probs[active_mask] = 0.2 + 0.7 * (latest_utils[active_mask] - min_u) / (max_u - min_u)
                else:
                    fresh_probs[active_mask] = 0.5

            alpha = 0.2
            survival_probs = (1 - alpha) * survival_probs + alpha * fresh_probs

            self._log_progress(stage="Knowledge Matrix Updated", outer_gen=outer_gen)

        # ================= Final Results Output =================
        final_switches_index = np.where(self.best_individual.solution > 0)[0]
        total_piles = np.sum(self.best_individual.piles[final_switches_index])

        print("\n" + "=" * 60)
        print("[SPV-CCEA] Optimization Completed | Final Results Summary")
        print("=" * 60)
        print(f"Final Active Stations: {len(final_switches_index)}")
        print(f"Total Deployed Piles: {total_piles:.0f}")
        print(f"Optimal Social Cost: {self.best_individual.fitness:.4f}")
        print(f"Cumulative FEs: {self.current_fe}/{self.stop_FEs}")
        print("=" * 60)

        return self.best_individual.solution, self.best_individual.piles, self.best_individual.fitness
