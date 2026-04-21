import os
import pickle
from scipy.spatial import cKDTree
import numpy as np
from pymoo.core.problem import ElementwiseProblem


class EVCSProblem(ElementwiseProblem):
    def __init__(self, evcs_instance, **kwargs):
        self.evcs = evcs_instance
        n_locations = evcs_instance.n_var
        n_var_total = 2 * n_locations

        xl = np.concatenate([
            np.zeros(n_locations),
            np.ones(n_locations) * evcs_instance.xl
        ])

        xu = np.concatenate([
            np.ones(n_locations),
            np.ones(n_locations) * evcs_instance.xu
        ])

        super().__init__(n_var=n_var_total,
                         n_obj=1,
                         n_ieq_constr=0,
                         xl=xl,
                         xu=xu,
                         **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        x_int = np.round(x).astype(int)

        n = self.evcs.n_var
        decision_switches = x_int[:n]
        pile_counts = x_int[n:]

        final_piles = decision_switches * pile_counts

        infor = self.evcs._evaluate(final_piles)

        out["F"] = infor[0]


class EVCS:
    def __init__(self, coords, traffic_flow_path, month_names,
                 n_samples=1000, ev_penetration=0.58, xl=10, xu=25, cache_dir="./evcs_cache"):
        self.coords = coords
        self.traffic_flow_path = traffic_flow_path
        self.month_names = month_names
        self.ev_penetration = ev_penetration
        self.n_var = len(coords)
        self.xl, self.xu = xl, xu
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.trajectories, self.traffic, self.traj_ev_flows = self._load_and_process_monthly_traffic_data(n_samples)

    def _load_and_process_monthly_traffic_data(self, n_samples):
        cache_filename = f"N_{self.n_var}_R_{n_samples}.pkl"
        cache_path = os.path.join(self.cache_dir, cache_filename)

        if os.path.exists(cache_path):
            print(f"📁 加载缓存数据: {cache_path}")
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            return data['trajectories'], data['traffic'], data['traj_ev_flows']

    def calculate_mmc_wait(self, arrival_rate, c, mu=1.25):
        W_max = 2.0
        phi = 100.0

        arrival_rate = np.atleast_1d(arrival_rate)
        c = np.maximum(1, np.round(c).astype(int))

        rho = arrival_rate / (c * mu)
        traffic = arrival_rate / mu

        wait_time = np.zeros_like(rho, dtype=float)

        mask_saturated = (rho >= 0.95)
        wait_time[mask_saturated] = W_max + phi * (rho[mask_saturated] - 0.95) ** 2
        mask_normal = (rho < 0.95) & (traffic > 0)

        if np.any(mask_normal):
            t_n = traffic[mask_normal]
            c_n = c[mask_normal]
            rho_n = rho[mask_normal]
            arr_n = arrival_rate[mask_normal]
            B_c = np.ones_like(t_n, dtype=float)
            max_c = np.max(c_n)

            for k in range(1, max_c + 1):
                active_k = (k <= c_n)
                B_c[active_k] = (t_n[active_k] * B_c[active_k]) / (k + t_n[active_k] * B_c[active_k])

            prob_queue = B_c / (1.0 - rho_n * (1.0 - B_c))
            wait_time[mask_normal] = prob_queue / (c_n * mu - arr_n)

        return np.maximum(0, wait_time)

    def _simulate_trip_behavior(self, trajs, ev_flows, node_traffic, cs_kdtree, cs_indices,
                                params, cs_coords):
        BATTERY_CAP = params['BATTERY_CAP']
        KWH_PER_KM = params['KWH_PER_KM']
        SOC_THRESHOLD = params['SOC_THRESHOLD']
        V_FREE = params['V_FREE']
        CHARGING_POWER = params['CHARGING_POWER']
        DIST_COST = params['DIST_COST']
        n_active_cs = len(cs_indices)

        station_arrival_rates_24h = np.zeros((n_active_cs, 24))
        total_detour_cost = 0.0
        total_lost_flow = 0.0

        for traj_info, flow in zip(trajs, ev_flows):
            current_soc = traj_info.get('init_soc')
            current_time = traj_info.get('start_hour')
            traj_path = traj_info['path']
            trip_failed = False

            for i in range(len(traj_path) - 1):
                idx_a, idx_b = traj_path[i], traj_path[i + 1]
                coord_a, coord_b = self.coords[idx_a], self.coords[idx_b]
                dist_ab = np.linalg.norm(coord_a - coord_b) / 1000.0
                req_kwh_ab = dist_ab * KWH_PER_KM

                need_charge = (current_soc * BATTERY_CAP < req_kwh_ab) or (current_soc < SOC_THRESHOLD)
                if need_charge:
                    dist_as_m, idx_s = cs_kdtree.query(coord_a, k=1)
                    dist_as = dist_as_m / 1000.0
                    coord_s = cs_coords[idx_s]
                    req_kwh_as = dist_as * KWH_PER_KM
                    if current_soc * BATTERY_CAP < req_kwh_as:
                        trip_failed = True
                        break

                    current_soc = (current_soc * BATTERY_CAP - req_kwh_as) / BATTERY_CAP

                    avg_flow_as = (node_traffic[idx_a] + node_traffic[cs_indices[idx_s]]) / 2.0
                    congestion_penalty_as = 1.0 + np.log1p(avg_flow_as) / 10.0
                    travel_time_as = (dist_as / V_FREE) * congestion_penalty_as
                    arrival_time = current_time + travel_time_as
                    arrival_hour = int(arrival_time) % 24
                    station_arrival_rates_24h[idx_s, arrival_hour] += flow
                    dist_sb = np.linalg.norm(coord_s - coord_b) / 1000.0
                    avg_flow_sb = (node_traffic[cs_indices[idx_s]] + node_traffic[idx_b]) / 2.0
                    congestion_penalty_sb = 1.0 + np.log1p(avg_flow_sb) / 10.0
                    travel_time_sb = (dist_sb / V_FREE) * congestion_penalty_sb
                    detour_km = max(0.0, (dist_as + dist_sb) - dist_ab)
                    total_detour_cost += detour_km * DIST_COST * flow
                    charging_soc_target = 0.9
                    charging_kwh = (charging_soc_target - current_soc) * BATTERY_CAP
                    charging_time = charging_kwh / CHARGING_POWER
                    current_soc = charging_soc_target
                    current_soc -= (dist_sb * KWH_PER_KM) / BATTERY_CAP
                    current_time += travel_time_as + travel_time_sb + charging_time

                else:
                    current_soc -= req_kwh_ab / BATTERY_CAP
                    avg_flow_ab = (node_traffic[idx_a] + node_traffic[idx_b]) / 2.0
                    congestion_penalty_ab = 1.0 + np.log1p(avg_flow_ab) / 10.0
                    travel_time_ab = (dist_ab / V_FREE) * congestion_penalty_ab
                    current_time += travel_time_ab

            if trip_failed:
                total_lost_flow += flow

        return station_arrival_rates_24h, total_detour_cost, total_lost_flow

    def _calculate_monthly_annual_cost(self, station_arrival_rates_24h, total_detour_cost,
                                       total_lost_flow, cs_piles, params):
        MU = params['MU']
        TIME_VAL = params['TIME_VAL']
        PENALTY_LOST = params['PENALTY_LOST']
        FIXED_COST = params['FIXED_COST']
        PILE_COST = params['PILE_COST']
        OPERATION_COST = params['OPERATION_COST']
        MAINTEMANCE_COST = params['MAINTEMANCE_COST']
        INTEREST_RATE = params['INTEREST_RATE']
        OPERATING_YEARS = params['OPERATING_YEARS']
        DAYS_PER_YEAR = params['DAYS_PER_YEAR']
        n_active_cs = len(cs_piles)
        total_queue_wait_cost = 0.0
        queue_wait_cost_list = []
        for hour in range(24):
            arrival_rates_h = station_arrival_rates_24h[:, hour]
            wait_times_h = self.calculate_mmc_wait(arrival_rates_h, cs_piles, MU)
            queue_wait_cost = np.sum(wait_times_h * arrival_rates_h * TIME_VAL)
            queue_wait_cost_list.append(queue_wait_cost)
            total_queue_wait_cost += queue_wait_cost

        monthly_user_cost = total_detour_cost + total_queue_wait_cost + (total_lost_flow * PENALTY_LOST)
        annual_user_cost = (monthly_user_cost * DAYS_PER_YEAR)

        total_investment = (n_active_cs * FIXED_COST) + (np.sum(cs_piles) * PILE_COST)
        r = INTEREST_RATE
        n = OPERATING_YEARS
        if r == 0:
            annual_construction_cost = total_investment / n
        else:
            annual_construction_cost = total_investment * (r * (1 + r) ** n) / ((1 + r) ** n - 1)

        annual_operation_cost = OPERATION_COST * n_active_cs
        annual_maintenance_cost = MAINTEMANCE_COST * np.sum(cs_piles)
        total_annual_cost = (
                annual_construction_cost
                + annual_operation_cost
                + annual_maintenance_cost
                + annual_user_cost
        )

        return total_annual_cost, total_queue_wait_cost, annual_user_cost

    def _evaluate(self, x, epsilon=1):
        params = {
            'BATTERY_CAP': 60.0,
            'MAX_TRAVEL': 500.0,
            'CHARGING_POWER': 80,
            'KWH_PER_KM': 0.12,
            'SOC_THRESHOLD': 0.25,
            'V_FREE': 60.0,
            'MU': 1.25,
            'TIME_VAL': 0.005,
            'DIST_COST': 0.0002,
            'PENALTY_LOST': 0.02,
            'FIXED_COST': 100.0,
            'PILE_COST': 5.0,
            'OPERATION_COST': 10,
            'MAINTEMANCE_COST': 1,
            'OPERATING_YEARS': 20,
            'INTEREST_RATE': 0.05,
            'DAYS_PER_YEAR': 365,
        }

        active_mask = x >= 1
        cs_indices = np.where(active_mask)[0]
        n_active_cs = len(cs_indices)

        if n_active_cs == 0:
            return [np.inf, np.zeros(self.n_var), []]

        cs_coords = self.coords[active_mask]
        cs_piles = x[active_mask]
        cs_kdtree = cKDTree(cs_coords)

        annual_total_costs = []
        annual_user_costs = []
        max_rho_active = np.zeros(n_active_cs)

        for trajs, ev_flows, node_traffic in zip(self.trajectories, self.traj_ev_flows, self.traffic):
            arrival_rates, detour_cost, lost_flow = self._simulate_trip_behavior(
                trajs, ev_flows, node_traffic, cs_kdtree, cs_indices, params, cs_coords
            )
            month_rho = arrival_rates / (cs_piles[:, None] * params['MU'])
            max_rho_active = np.maximum(max_rho_active, np.max(month_rho, axis=1))
            monthly_annual_cost, queue_cost, user_cost = self._calculate_monthly_annual_cost(
                arrival_rates, detour_cost, lost_flow, cs_piles, params
            )
            annual_total_costs.append(monthly_annual_cost)
            annual_user_costs.append(user_cost)

        total_costs_np = np.array(annual_total_costs)
        expected_cost = np.mean(total_costs_np)
        cost_risk = np.std(total_costs_np)
        final_cost = expected_cost + epsilon * cost_risk

        full_utilization = np.zeros(self.n_var)
        full_utilization[active_mask] = max_rho_active

        return [final_cost, full_utilization, annual_user_costs]