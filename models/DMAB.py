import numpy as np
import math

class MultiArmedBandit(object):

    def __init__(self, num_arms, k, delta, c=1.0, q=0.5):
        self.num_arms = int(num_arms)
        self.k = int(k)
        self.delta = float(delta)
        self.c = float(c)
        self.q = float(q)

        self.Q = np.zeros(self.num_arms, dtype=float)
        self.N = np.zeros(self.num_arms, dtype=float)
        self.t = 0.0

        self.energy_avg = np.zeros(self.num_arms, dtype=float)

    def ucb2(self,
             current_epoch=None,
             edge_server=None,
             edge_global_ids=None,
             select_k=None,
             delta=None,
             c=None):
        if select_k is None:
            select_k = self.k
        if delta is None:
            delta = self.delta
        if c is None:
            c = self.c

        eps = 1e-9
        logsafe = max(float(delta), 1e-12)
        ucb_vec = self.Q + c * np.sqrt(2.0 * math.log(1.0 / logsafe) / (self.N + eps))

        if edge_server is not None:
            gids = list(edge_global_ids) if edge_global_ids is not None else list(
                getattr(edge_server, "clients_idx", []))
            cur_round_idx = int(getattr(getattr(edge_server, "args", None), "current_epoch", 0))
            t_display = int(cur_round_idx + 1)

            dop_vals = []
            lids = []
            for g in gids:
                lid = edge_server.get_local_index(g)
                if lid is None or lid < 0:
                    continue
                lids.append(lid)
                dop_vals.append(float(t_display - self.N[lid]))
            max_dop = max(dop_vals) if dop_vals else 0.0

            print("\n[MAB][UCB] Edge {} | t={} | delta={:.3g} | c={:.3g} | q={:.3f}".format(
                getattr(edge_server, "edge_id", "NA"),
                t_display, float(delta), float(c), float(self.q)))
            print("{:<6} {:>8} {:>10} {:>10} {:>6} {:>8} {:>10} {:>10}".format(
                "gid", "lid", "energy", "fairness", "Lk", "DoPraw", "Reward", "UCB"))

            for g in gids:
                lid = edge_server.get_local_index(g)
                if lid is None or lid < 0:
                    continue

                Lk = float(self.N[lid])
                DoP_raw = float(t_display - Lk)
                #fairness = (DoP_raw / max_dop) if max_dop > 1e-12 else 1.0
                fairness = DoP_raw

                energy_term = float(self.energy_avg[lid])*10
                Q_prev = float(self.Q[lid])
                u = float(ucb_vec[lid])

                print("{:<6d} {:>8d} {:>10.4f} {:>10.4f} {:>6.0f} {:>8.0f} {:>10.4f} {:>10.4f}".format(
                    int(g), int(lid),
                    energy_term, fairness,
                    Lk, DoP_raw, Q_prev, u
                ))

        topk_local = np.argsort(ucb_vec)[-int(select_k):][::-1]
        if edge_server is not None:
            chosen_gids = [edge_server.get_global_id(int(lid)) for lid in topk_local]
            print("[MAB][UCB] Selected (top-{}): {}\n".format(int(select_k), chosen_gids))
        return list(map(int, topk_local))

    def ucb(self, current_epoch):
        eps = 1e-6
        ucb = self.Q + self.c * np.sqrt(2.0 * np.log(1.0 / self.delta) / (self.N + eps))
        topk = np.argpartition(ucb, -self.k)[-self.k:]
        return topk

    def accumulate_energy(self, lid: int, e_hat: float):
        """Update the running average energy estimate for one local arm."""
        n_old = float(self.N[lid])
        self.energy_avg[lid] = (n_old * self.energy_avg[lid] + float(e_hat)) / (n_old + 1.0)

    def build_rewards_for_all(self, edge_server, t_completed: int, selected_set=None, q: float = None):
        """Build per-client DMAB rewards from energy and selection freshness.

        Lk_after is the selection count after the current round, and the
        degree-of-participation term grows when a client has not been selected
        recently. The final reward is a weighted sum controlled by q.
        """
        if q is None:
            q = self.q
        if selected_set is None:
            selected_set = set()

        gids = list(getattr(edge_server, "clients_idx", []))
        lids = []
        for g in gids:
            lid = edge_server.get_local_index(g)
            if lid is None or lid < 0:
                continue
            lids.append(lid)

        Lk_after = []
        for lid in lids:
            gid = edge_server.get_global_id(lid)
            inc = 1.0 if gid in selected_set else 0.0
            Lk_after.append(float(self.N[lid]) + inc)

        dop_vals = [float(t_completed+1) - Lk_after[i] for i in range(len(lids))]
        max_dop = max(dop_vals) if len(dop_vals) > 0 else 0.0
        #fairness_vec = [(dop_vals[i] / max_dop) if max_dop > 1e-12 else 1.0 for i in range(len(dop_vals))]
        fairness_vec = dop_vals

        rewards = {}
        for i, lid in enumerate(lids):
            gid = edge_server.get_global_id(lid)
            energy_term = float(self.energy_avg[lid])*10
            fairness = float(fairness_vec[i])
            reward = (1.0 - q) * fairness + q * energy_term
            rewards[gid] = reward

            if gid in selected_set:
                print(f"[MAB][REWARD-ALL] gid={gid} lid={lid} | Ebar={energy_term:.4f} DoP={fairness:.4f} "
                      f"q={q:.3f} -> Reward={(1.0 - q) * fairness:.4f} + {q * energy_term:.4f} = {reward:.4f}")

        return rewards

    def update(self, edge_server, selected_clients, rewards):
        """Store the latest reward for every arm and increment selected counts."""
        def _take_reward(rwds, gid, default_val):
            if isinstance(rwds, dict):
                return rwds.get(gid, default_val)
            arr = np.asarray(rwds)
            if gid < 0 or gid >= len(arr):
                return default_val
            return float(arr[gid])

        cur_round_idx = int(getattr(getattr(edge_server, "args", None), "current_epoch", self.t))

        for gid in getattr(edge_server, "clients_idx", []):
            lid = edge_server.get_local_index(gid)
            if lid is None or lid < 0:
                continue
            self.Q[lid] = _take_reward(rewards, gid, self.Q[lid])

        for gid in selected_clients:
            lid = edge_server.get_local_index(gid)
            if lid is None or lid < 0:
                continue
            self.N[lid] += 1.0

        self.t = float(cur_round_idx)

    def MAB_update_mean(self, *args, **kwargs):
        """Deprecated compatibility method for the old monolithic update path."""
        raise NotImplementedError("Use accumulate_energy() + build_rewards_for_all() + update() in the new pipeline.")
