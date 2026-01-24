# Market Camera v2 — Sliding-window structure aggregator
# NO core_system changes. Consumes raw ticks, emits STRUCTURE FRAMES.

from collections import deque, defaultdict
import math
from typing import Optional, Dict, List, Any, Tuple

# Env defaults (override via os.environ where used)
CAM_W = 300
CAM_STRIDE = 10
CAM_EDGE_CAP = 2000
CAM_PROX_EPS_PCT = 0.003


def _mid_from_actions(actions: Dict[int, Dict]) -> Optional[float]:
    """Compute mid = median(action.price) for side in {buy, sell}. Fallback None if none."""
    prices = []
    for aid, a in actions.items():
        s = a.get("side")
        if s in ("buy", "sell") and "price" in a:
            try:
                prices.append(float(a["price"]))
            except (TypeError, ValueError):
                pass
    if not prices:
        return None
    prices.sort()
    n = len(prices)
    if n % 2 == 1:
        return prices[n // 2]
    return (prices[n // 2 - 1] + prices[n // 2]) / 2.0


def _matches_pairs(matches: List) -> List[Tuple[int, int]]:
    """Return list of (a,b) with b != -1. Accept list of (a,b) or [{"a":a,"b":b}]."""
    out = []
    for m in matches:
        if isinstance(m, (list, tuple)) and len(m) >= 2:
            a, b = m[0], m[1]
        elif isinstance(m, dict) and "a" in m and "b" in m:
            a, b = m["a"], m["b"]
        else:
            continue
        if b is not None and b != -1:
            out.append((min(a, b), max(a, b)))
    return out


class SlidingWindowCamera:
    def __init__(
        self,
        W: int = CAM_W,
        stride: int = CAM_STRIDE,
        eps_pct: float = CAM_PROX_EPS_PCT,
        edge_cap: int = CAM_EDGE_CAP,
    ):
        self.W = W
        self.stride = stride
        self.eps_pct = eps_pct
        self.edge_cap = edge_cap
        self.ring_ticks: deque = deque(maxlen=W)
        self._last_mid: Optional[float] = None

    def ingest(self, raw_tick: Dict[str, Any]) -> None:
        """Append one raw_tick to ring. raw_tick = {t, mid, agents, actions, matches}."""
        self.ring_ticks.append(raw_tick)
        m = raw_tick.get("mid")
        if m is not None:
            self._last_mid = m

    def maybe_build_frame(self, t: int) -> Optional[Dict[str, Any]]:
        """Emit frame only when t >= W and (t % stride == 0). window t0 = t - W + 1, t1 = t."""
        if len(self.ring_ticks) < self.W:
            return None
        if t % self.stride != 0:
            return None

        t0 = t - self.W + 1
        t1 = t
        ticks = list(self.ring_ticks)

        # --- Nodes ---
        agents_in_window = set()
        for tk in ticks:
            for aid in tk.get("agents", {}):
                agents_in_window.add(aid)
            for aid in tk.get("actions", {}):
                agents_in_window.add(aid)

        nodes: List[Dict] = []
        for aid in sorted(agents_in_window):
            node_id = f"agent:{aid}@t:{t0}-{t1}"
            features = self._node_features(aid, ticks, t0, t1)
            nodes.append({
                "node_id": node_id,
                "agent_id": aid,
                "t0": t0,
                "t1": t1,
                "features": features,
            })

        # --- Edges ---
        match_edges, prox_edges = self._build_edges(ticks)
        all_edges: List[Dict] = []
        for (a, b), cnt in match_edges.items():
            w = cnt / self.W
            all_edges.append({
                "src": f"agent:{a}@t:{t0}-{t1}",
                "dst": f"agent:{b}@t:{t0}-{t1}",
                "type": "match",
                "w": round(w, 6),
            })
        for (a, b), cnt in prox_edges.items():
            w = cnt / self.W
            all_edges.append({
                "src": f"agent:{a}@t:{t0}-{t1}",
                "dst": f"agent:{b}@t:{t0}-{t1}",
                "type": "quote_proximity",
                "w": round(w, 6),
            })

        # Edge cap: sort by w desc, keep top K = min(edge_cap, 8*N)
        K = min(self.edge_cap, max(0, 8 * len(nodes)))
        all_edges.sort(key=lambda e: e["w"], reverse=True)
        all_edges = all_edges[:K]

        return {
            "type": "frame",
            "t": t,
            "window": {"W": self.W, "t0": t0, "t1": t1, "stride": self.stride},
            "nodes": nodes,
            "edges": all_edges,
            "events": {"bursts": [], "shocks": []},
        }

    def _node_features(
        self, aid: int, ticks: List[Dict], t0: int, t1: int
    ) -> Dict[str, Any]:
        buy_count = sell_count = none_count = 0
        quotes: List[float] = []
        mids: List[float] = []
        match_count = 0
        exp_start = exp_end = None

        for tk in ticks:
            mid = tk.get("mid")
            if mid is not None:
                mids.append(mid)
            agents = tk.get("agents", {})
            actions = tk.get("actions", {})
            if aid in agents:
                ex = agents[aid]
                try:
                    ex = float(ex)
                except (TypeError, ValueError):
                    ex = None
                if exp_start is None:
                    exp_start = ex
                exp_end = ex

            ac = actions.get(aid, {})
            side = ac.get("side", "none")
            if side == "buy":
                buy_count += 1
            elif side == "sell":
                sell_count += 1
            else:
                none_count += 1
            price = ac.get("price")
            if price is not None and side in ("buy", "sell"):
                try:
                    quotes.append(float(price))
                except (TypeError, ValueError):
                    pass

            for (a, b) in _matches_pairs(tk.get("matches", [])):
                if a == aid or b == aid:
                    match_count += 1

        # dominant_side
        dominant = "none"
        if buy_count >= sell_count and buy_count >= none_count and buy_count > 0:
            dominant = "buy"
        elif sell_count >= buy_count and sell_count >= none_count and sell_count > 0:
            dominant = "sell"

        quote_price_mean = float(sum(quotes)) / len(quotes) if quotes else 0.0
        quote_price_std = 0.0
        if len(quotes) > 1:
            v = sum((x - quote_price_mean) ** 2 for x in quotes) / len(quotes)
            quote_price_std = math.sqrt(v)

        quote_dist_to_mid_mean = 0.0
        if quotes and mids:
            mid_avg = sum(mids) / len(mids)
            if mid_avg != 0:
                dists = [abs(p - mid_avg) / abs(mid_avg) for p in quotes]
                quote_dist_to_mid_mean = sum(dists) / len(dists)

        match_rate = match_count / self.W if self.W else 0.0
        experience_delta = 0.0
        if exp_start is not None and exp_end is not None:
            try:
                experience_delta = float(exp_end) - float(exp_start)
            except (TypeError, ValueError):
                pass

        return {
            "buy_count": buy_count,
            "sell_count": sell_count,
            "none_count": none_count,
            "match_count": match_count,
            "match_rate": round(match_rate, 6),
            "quote_price_mean": round(quote_price_mean, 6),
            "quote_price_std": round(quote_price_std, 6),
            "quote_dist_to_mid_mean": round(quote_dist_to_mid_mean, 6),
            "experience_start": exp_start if exp_start is None else round(float(exp_start), 6),
            "experience_end": exp_end if exp_end is None else round(float(exp_end), 6),
            "experience_delta": round(experience_delta, 6),
            "dominant_side": dominant,
        }

    def _build_edges(
        self, ticks: List[Dict]
    ) -> Tuple[Dict[Tuple[int, int], int], Dict[Tuple[int, int], int]]:
        match_counts: Dict[Tuple[int, int], int] = defaultdict(int)
        prox_counts: Dict[Tuple[int, int], int] = defaultdict(int)

        for tk in ticks:
            mid = tk.get("mid")
            if mid is None or mid <= 0:
                continue
            eps = self.eps_pct * abs(mid)
            actions = tk.get("actions", {})

            for (a, b) in _matches_pairs(tk.get("matches", [])):
                key = (min(a, b), max(a, b))
                match_counts[key] += 1

            for side in ("buy", "sell"):
                entries = [
                    (aid, float(ac["price"]))
                    for aid, ac in actions.items()
                    if ac.get("side") == side and "price" in ac
                ]
                try:
                    entries.sort(key=lambda x: x[1], reverse=(side == "buy"))
                except Exception:
                    continue
                for i in range(len(entries) - 1):
                    a1, p1 = entries[i]
                    a2, p2 = entries[i + 1]
                    if abs(p1 - p2) <= eps:
                        key = (min(a1, a2), max(a1, a2))
                        prox_counts[key] += 1

        return dict(match_counts), dict(prox_counts)
