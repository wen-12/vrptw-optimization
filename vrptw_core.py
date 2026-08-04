"""
vrptw_core.py

Modul inti VRPTW: Solomon I1 Insertion Heuristic, Nearest Neighbor (NN)
Construction, dan Dragonfly Algorithm (DA).
"""

import math
import random
import time
import csv
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable


# =============================================================================
# PEMINDAIAN DATASET
# =============================================================================

def list_datasets(dataset_folder: str = "dataset") -> dict:
    """Pindai semua file .csv di `dataset_folder`."""
    folder = Path(dataset_folder)
    if not folder.exists():
        return {}
    return {p.stem: str(p) for p in sorted(folder.glob("*.csv"))}


# =============================================================================
# STRUKTUR DATA: CUSTOMER & INSTANCE VRPTW
# =============================================================================

# Class ini adalah struktur data satu customer (atau depot) pada VRPTW
@dataclass
class Customer:
    id: int
    x: float
    y: float
    demand: float
    ready: float
    due: float
    service: float


# Class ini adalah struktur data satu instance/problem VRPTW secara utuh
@dataclass
class VRPTWInstance:
    name: str
    capacity: float
    num_vehicles: int
    customers: List[Customer]
    dist_matrix: List[List[float]] = field(default_factory=list)

    @property
    def n_customers(self) -> int:
        # Properti ini untuk menghitung jumlah customer (tidak termasuk depot)
        return len(self.customers) - 1


# Fungsi ini untuk menghitung jarak Euclidean antara dua titik/customer
def euclidean(a: Customer, b: Customer) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


# Fungsi ini untuk membangun matriks jarak antar semua customer
def build_distance_matrix(customers: List[Customer]) -> List[List[float]]:
    n = len(customers)
    mat = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                mat[i][j] = euclidean(customers[i], customers[j])
    return mat


# Ini adalah daftar nama kolom yang wajib ada di file CSV dataset
_REQUIRED_COLUMNS = ["id", "x", "y", "demand", "ready", "due", "service"]

# Ini adalah daftar alias nama kolom yang dikenali saat membaca CSV
_COLUMN_ALIASES = {
    "id":       ["custno", "cust_no", "cust no.", "cust no", "id", "customer", "customerid",
                 "customer_id", "no", "no."],
    "x":        ["xcoord", "xcoord.", "x_coord", "x coord.", "x", "xcoordinate"],
    "y":        ["ycoord", "ycoord.", "y_coord", "y coord.", "y", "ycoordinate"],
    "demand":   ["demand"],
    "ready":    ["readytime", "ready_time", "ready time", "ready"],
    "due":      ["duedate", "due_date", "due date", "duetime", "due_time", "due time", "due"],
    "service":  ["servicetime", "_service_time", "service time", "service"],
    "capacity": ["capacity"],
    "vehicle":  ["vehicle", "num_vehicle", "num_vehicles", "numvehicles"],
}


# Fungsi ini untuk menormalkan (lowercase & trim) satu nama header kolom
def _normalize_header(header: str) -> str:
    return header.strip().lower()


# Fungsi ini untuk memetakan header CSV ke nama kolom standar (id, x, y, dst.)
def _map_columns(header_row: List[str]) -> dict:
    normalized = [_normalize_header(h) for h in header_row]
    col_map = {}
    for standard_name, aliases in _COLUMN_ALIASES.items():
        found_idx = None
        for i, h in enumerate(normalized):
            if h in aliases:
                found_idx = i
                break
        if found_idx is None and standard_name in _REQUIRED_COLUMNS:
            raise ValueError(
                f"Kolom '{standard_name}' tidak ditemukan di header CSV: {header_row}\n"
                f"Nama kolom yang dikenali untuk '{standard_name}': {_COLUMN_ALIASES[standard_name]}"
            )
        col_map[standard_name] = found_idx
    return col_map


# Fungsi ini untuk membaca file CSV dan membangun objek VRPTWInstance
def read_csv_instance(filepath: str, capacity: Optional[float] = None,
                       num_vehicles: Optional[int] = None,
                       n_customers: Optional[int] = None,
                       delimiter: str = ",", name: Optional[str] = None) -> VRPTWInstance:
    with open(filepath, "r", newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        rows = [row for row in reader if any(cell.strip() != "" for cell in row)]

    if len(rows) < 2:
        raise ValueError(f"File CSV '{filepath}' tidak memiliki data yang cukup.")

    header_row = rows[0]
    col_map = _map_columns(header_row)
    data_rows = rows[1:]

    # Fungsi ini untuk mengubah satu baris CSV menjadi objek Customer
    def parse_row(row) -> Customer:
        return Customer(
            id=int(float(row[col_map["id"]])),
            x=float(row[col_map["x"]]),
            y=float(row[col_map["y"]]),
            demand=float(row[col_map["demand"]]),
            ready=float(row[col_map["ready"]]),
            due=float(row[col_map["due"]]),
            service=float(row[col_map["service"]]),
        )

    all_customers = [parse_row(row) for row in data_rows]

    # Ini adalah kandidat depot (customer dengan demand = 0)
    depot_candidates = [c for c in all_customers if c.demand == 0.0]
    if not depot_candidates:
        raise ValueError(f"Tidak ditemukan depot (demand=0) di file CSV: {filepath}")
    depot = depot_candidates[0]
    customer_pool = [c for c in all_customers if c.id != depot.id]
    available = len(customer_pool)

    if capacity is None:
        cap_idx = col_map.get("capacity")
        if cap_idx is not None:
            for row in data_rows:
                raw_cap = row[cap_idx].strip()
                if raw_cap != "":
                    capacity = float(raw_cap)
                    break
        if capacity is None:
            raise ValueError("Kapasitas kendaraan tidak diberikan dan tidak ditemukan di CSV.")

    if num_vehicles is None:
        veh_idx = col_map.get("vehicle")
        if veh_idx is not None:
            for row in data_rows:
                raw_veh = row[veh_idx].strip()
                if raw_veh != "":
                    num_vehicles = int(float(raw_veh))
                    break
        if num_vehicles is None:
            num_vehicles = len(customer_pool)

    if n_customers is not None:
        if n_customers > available:
            raise ValueError(
                f"Jumlah customer yang diminta ({n_customers}) melebihi yang tersedia ({available})."
            )
        if n_customers < 1:
            raise ValueError(f"n_customers harus >= 1, diberikan: {n_customers}")
        customer_pool = customer_pool[:n_customers]

    customers = [depot] + customer_pool
    inst_name = name if name is not None else Path(filepath).name
    inst = VRPTWInstance(name=inst_name, capacity=capacity, num_vehicles=num_vehicles,
                          customers=customers)
    inst.dist_matrix = build_distance_matrix(customers)
    return inst


# Fungsi ini untuk mengecek kelayakan satu rute (batas kapasitas dan time window)
def route_feasibility(inst: VRPTWInstance, route: List[int]) -> Tuple[bool, float, float]:
    id_to_cust = {c.id: c for c in inst.customers}
    load, time_, dist = 0.0, 0.0, 0.0
    feasible = True
    current = id_to_cust[route[0]]
    for nxt_id in route[1:]:
        nxt = id_to_cust[nxt_id]
        travel = euclidean(current, nxt)
        dist += travel
        arrival = max(time_ + travel, nxt.ready)
        if arrival > nxt.due:
            feasible = False
        time_ = arrival + nxt.service
        load += nxt.demand
        current = nxt
    if load > inst.capacity:
        feasible = False
    return feasible, dist, load


# Fungsi ini untuk menghitung fitness solusi: jumlah kendaraan dan total jarak
def evaluate_fitness(inst: VRPTWInstance, routes: Optional[List[List[int]]]) -> Tuple[int, float]:
    if routes is None:
        return (float("inf"), float("inf"))
    num_vehicles = len(routes)
    total_distance = sum(route_feasibility(inst, r)[1] for r in routes)
    return num_vehicles, total_distance


# Fungsi ini untuk menghitung total waktu tempuh (termasuk tunggu & layanan) seluruh rute
def evaluate_total_time(inst: VRPTWInstance, routes: List[List[int]]) -> float:
    id_to_cust = {c.id: c for c in inst.customers}
    total_time = 0.0
    for route in routes:
        t = 0.0
        current = id_to_cust[route[0]]
        for nxt_id in route[1:]:
            nxt = id_to_cust[nxt_id]
            travel = euclidean(current, nxt)
            arrival = max(t + travel, nxt.ready)
            t = arrival + nxt.service
            current = nxt
        total_time += t
    return total_time


# Class ini adalah parameter untuk algoritma Solomon I1 Insertion
@dataclass
class I1Parameters:
    mu: float = 1.0
    lam: float = 2.0
    alpha1: float = 1.0
    alpha2: float = 0.0


# Fungsi ini untuk menghitung biaya penyisipan (insertion cost) satu customer ke satu rute
def _insertion_cost(inst: VRPTWInstance, route: List[int], pos: int, cust: Customer,
                     params: I1Parameters) -> Optional[Tuple[float, float]]:
    id_to_cust = {c.id: c for c in inst.customers}
    prev_id = route[pos - 1]
    next_id = route[pos]
    prev_c = id_to_cust[prev_id]
    next_c = id_to_cust[next_id]

    current_route_load = sum(id_to_cust[cid].demand for cid in route[1:-1])
    if current_route_load + cust.demand > inst.capacity:
        return None

    t = 0.0
    cur = id_to_cust[route[0]]
    for cid in route[1:pos]:
        nxt = id_to_cust[cid]
        travel = euclidean(cur, nxt)
        arr = max(t + travel, nxt.ready)
        if arr > nxt.due:
            return None
        t = arr + nxt.service
        cur = nxt

    d_prev_u = euclidean(prev_c, cust)
    d_u_next = euclidean(cust, next_c)
    d_prev_next = euclidean(prev_c, next_c)
    # Ini adalah c11: selisih jarak akibat penyisipan customer
    c11 = d_prev_u + d_u_next - d_prev_next

    arrival_u = max(t + d_prev_u, cust.ready)
    if arrival_u > cust.due:
        return None

    departure_u = arrival_u + cust.service
    old_arrival_next = max(t + d_prev_next, next_c.ready)
    new_arrival_next = max(departure_u + d_u_next, next_c.ready)
    push_forward = new_arrival_next - old_arrival_next
    # Ini adalah c12: pergeseran waktu kedatangan (push forward) akibat penyisipan
    c12 = push_forward

    time_cursor = departure_u
    prev_for_check = cust
    for cid in route[pos:]:
        nxt = id_to_cust[cid]
        travel = euclidean(prev_for_check, nxt)
        arr = max(time_cursor + travel, nxt.ready)
        if arr > nxt.due:
            return None
        time_cursor = arr + nxt.service
        prev_for_check = nxt

    c1 = params.alpha1 * c11 + params.alpha2 * c12
    return c1, arrival_u


# Fungsi ini untuk membangun rute VRPTW dengan Solomon I1 Insertion Heuristic
def solomon_i1_insertion(inst: VRPTWInstance, params: I1Parameters) -> List[List[int]]:
    depot = inst.customers[0]
    unassigned = {c.id: c for c in inst.customers[1:]}
    routes: List[List[int]] = []

    while unassigned:
        seed = min(unassigned.values(), key=lambda c: c.due)
        route = [depot.id, seed.id, depot.id]
        del unassigned[seed.id]

        improved = True
        while improved and unassigned:
            improved = False
            best_c2_overall = -float("inf")
            best_choice = None

            for cust in list(unassigned.values()):
                local_best_c1 = float("inf")
                local_best_pos = None
                for pos in range(1, len(route)):
                    result = _insertion_cost(inst, route, pos, cust, params)
                    if result is None:
                        continue
                    c1, _ = result
                    if c1 < local_best_c1:
                        local_best_c1 = c1
                        local_best_pos = pos

                if local_best_pos is None:
                    continue

                d0u = euclidean(depot, cust)
                c2 = params.lam * d0u - local_best_c1

                if c2 > best_c2_overall:
                    best_c2_overall = c2
                    best_choice = (cust, local_best_pos)

            if best_choice is not None:
                cust, pos = best_choice
                route.insert(pos, cust.id)
                del unassigned[cust.id]
                improved = True

        routes.append(route)

    return routes


# =============================================================================
# NEAREST NEIGHBOR (NN) CONSTRUCTION HEURISTIC
# =============================================================================

def nearest_neighbor_construction(inst: VRPTWInstance) -> List[List[int]]:
    """Bangun rute awal VRPTW dengan Nearest Neighbor (NN) heuristic."""
    depot = inst.customers[0]
    unassigned = {c.id: c for c in inst.customers[1:]}
    routes: List[List[int]] = []

    while unassigned:
        route = [depot.id]
        load, t = 0.0, 0.0
        current = depot

        while True:
            best_cust = None
            best_dist = float("inf")

            for cust in unassigned.values():
                if load + cust.demand > inst.capacity:
                    continue
                travel = euclidean(current, cust)
                arrival = max(t + travel, cust.ready)
                if arrival > cust.due:
                    continue
                if travel < best_dist:
                    best_dist = travel
                    best_cust = cust

            if best_cust is None:
                break

            travel = euclidean(current, best_cust)
            arrival = max(t + travel, best_cust.ready)
            t = arrival + best_cust.service
            load += best_cust.demand
            route.append(best_cust.id)
            current = best_cust
            del unassigned[best_cust.id]

        if len(route) == 1:
            stuck = next(iter(unassigned.values()))
            raise ValueError(
                f"Nearest Neighbor gagal: customer id={stuck.id} tidak bisa "
                f"dilayani oleh kendaraan manapun (kapasitas atau time "
                f"window tidak terpenuhi meski rute masih kosong)."
            )

        route.append(depot.id)
        routes.append(route)

    return routes


# Fungsi ini untuk mengubah rute menjadi vektor random-key (dipakai sebagai seed Dragonfly Algorithm)
def encode_routes_to_vector(inst: VRPTWInstance, routes: List[List[int]], noise: float = 0.0) -> List[float]:
    n = inst.n_customers
    vector = [0.0] * n
    id_to_pos = {c.id: i for i, c in enumerate(inst.customers[1:])}

    visit_order = []
    for route in routes:
        for cid in route[1:-1]:
            visit_order.append(cid)

    for pos, cid in enumerate(visit_order):
        base = pos / max(n - 1, 1)
        idx = id_to_pos[cid]
        vector[idx] = max(0.0, min(1.0, base + random.uniform(-noise, noise)))
    return vector


# Fungsi ini untuk mengubah vektor random-key menjadi rute (metode split sederhana/greedy)
def decode_vector_to_routes_simple(inst: VRPTWInstance, vector: List[float]) -> List[List[int]]:
    custs = inst.customers[1:]
    order = sorted(range(len(custs)), key=lambda i: vector[i])
    giant_tour = [custs[i] for i in order]

    depot = inst.customers[0]
    routes, route = [], [depot.id]
    load, t = 0.0, 0.0
    current = depot

    for cust in giant_tour:
        travel = euclidean(current, cust)
        arrival = max(t + travel, cust.ready)
        if (load + cust.demand) <= inst.capacity and arrival <= cust.due:
            route.append(cust.id)
            load += cust.demand
            t = arrival + cust.service
            current = cust
        else:
            route.append(depot.id)
            routes.append(route)
            route = [depot.id, cust.id]
            load = cust.demand
            arrival = max(euclidean(depot, cust), cust.ready)
            t = arrival + cust.service
            current = cust

    route.append(depot.id)
    routes.append(route)
    return routes


# Fungsi ini untuk membelah giant tour menjadi rute-rute optimal (shortest-path split)
def split_giant_tour_vrptw(inst: VRPTWInstance, giant_tour: List[Customer]) -> Optional[List[List[int]]]:
    depot = inst.customers[0]
    n = len(giant_tour)
    INF = float("inf")

    V = [INF] * (n + 1)
    pred = [None] * (n + 1)
    V[0] = 0.0

    for i in range(n):
        if V[i] == INF:
            continue
        load = 0.0
        t = 0.0
        cur = depot
        running_dist = 0.0
        for j in range(i, n):
            cust = giant_tour[j]
            travel = euclidean(cur, cust)
            arrival = max(t + travel, cust.ready)

            if load + cust.demand > inst.capacity or arrival > cust.due:
                break

            load += cust.demand
            running_dist += travel
            t = arrival + cust.service
            cur = cust

            route_cost = running_dist + euclidean(cur, depot)
            if V[i] + route_cost < V[j + 1]:
                V[j + 1] = V[i] + route_cost
                pred[j + 1] = i

    if V[n] == INF:
        return None

    routes = []
    idx = n
    while idx > 0:
        i = pred[idx]
        route = [depot.id] + [c.id for c in giant_tour[i:idx]] + [depot.id]
        routes.append(route)
        idx = i
    routes.reverse()
    return routes


# Fungsi ini untuk mengubah vektor random-key menjadi rute memakai metode split optimal
def decode_vector_to_routes_split(inst: VRPTWInstance, vector: List[float]) -> List[List[int]]:
    custs = inst.customers[1:]
    order = sorted(range(len(custs)), key=lambda i: vector[i])
    giant_tour = [custs[i] for i in order]

    routes = split_giant_tour_vrptw(inst, giant_tour)
    if routes is None:
        return decode_vector_to_routes_simple(inst, vector)
    return routes


# =============================================================================
# DRAGONFLY ALGORITHM (DA) CANONICAL - MIRJALILI 2016
# =============================================================================

class PureDragonflyAlgorithm:
    """Implementasi Dragonfly Algorithm (DA) Canonical (Mirjalili, 2016)."""

    def __init__(self, instance: VRPTWInstance, num_agents: int = 40, max_iter: int = 300,
                 weights: Tuple[float, float, float, float] = (0.15, 0.35, 0.3, 0.2),
                 seed: Optional[int] = None,
                 seed_vector: Optional[List[float]] = None):
        if seed is not None:
            random.seed(seed)

        self.seed_vector = seed_vector
        self.inst = instance
        self.num_agents = num_agents
        self.max_iter = max_iter
        # Ini adalah bobot untuk separation, alignment, cohesion, dan food factor
        self.s, self.a, self.c, self.f = weights
        # Ini adalah bobot untuk enemy factor
        self.e = 0.1
        # Ini adalah bobot inertia (pengaruh step sebelumnya)
        self.w = 0.4
        self.n = instance.n_customers

        # Ini adalah posisi (vektor random-key) tiap agen dragonfly
        self.positions: List[List[float]] = []
        # Ini adalah vektor langkah/kecepatan (step vector) tiap agen dragonfly
        self.step_vectors: List[List[float]] = []
        # Ini adalah nilai fitness tiap agen
        self.fitness: List[Tuple[int, float]] = []

        # Ini adalah agen dan solusi terbaik (food source)
        self.best_agent_idx = None
        self.best_fitness = (float("inf"), float("inf"))
        self.best_routes: List[List[int]] = []
        self.best_position: Optional[List[float]] = None
        # Ini adalah agen terburuk (enemy)
        self.worst_agent_idx = None
        self.worst_fitness = (-1, -1.0)

    # Fungsi ini untuk menginisialisasi populasi awal agen dragonfly
    def initialize_population(self):
        for i in range(self.num_agents):
            if self.seed_vector is not None:
                vector = list(self.seed_vector)
                if i > 0:
                    k = random.randint(2, 10)
                    for idx in random.sample(range(self.n), k):
                        vector[idx] = random.random()
            else:
                vector = [random.random() for _ in range(self.n)]
            self.positions.append(vector)
            self.step_vectors.append([0.0] * self.n)
        self._evaluate_population()

    # Fungsi ini untuk mengevaluasi fitness seluruh populasi agen
    def _evaluate_population(self):
        self.fitness = []
        routes_cache = []
        for vector in self.positions:
            routes = decode_vector_to_routes_split(self.inst, vector)
            fit = evaluate_fitness(self.inst, routes)
            self.fitness.append(fit)
            routes_cache.append(routes)
        self._update_food_and_enemy(routes_cache)

    # Fungsi ini untuk memperbarui agen terbaik (food) dan agen terburuk (enemy)
    def _update_food_and_enemy(self, routes_cache):
        for i, fit in enumerate(self.fitness):
            if fit < self.best_fitness:
                self.best_fitness = fit
                self.best_agent_idx = i
                self.best_routes = routes_cache[i]
                self.best_position = list(self.positions[i])
            if fit > self.worst_fitness:
                self.worst_fitness = fit
                self.worst_agent_idx = i

    # Fungsi ini untuk mencari agen-agen tetangga dalam radius tertentu
    def _get_neighbors(self, i: int, radius: float) -> List[int]:
        scaled_radius = radius
        return [j for j in range(self.num_agents)
                if j != i and math.dist(self.positions[i], self.positions[j]) <= scaled_radius]

    # Fungsi ini untuk menghitung separation: dorongan menjauh dari agen tetangga terdekat
    def _separation(self, i, neighbors):
        if not neighbors:
            return [0.0] * self.n
        s_vec = [0.0] * self.n
        for j in neighbors:
            for k in range(self.n):
                s_vec[k] -= (self.positions[j][k] - self.positions[i][k])
        return s_vec

    # Fungsi ini untuk menghitung alignment: penyamaan arah gerak dengan agen tetangga
    def _alignment(self, i, neighbors):
        if not neighbors:
            return [0.0] * self.n
        a_vec = [0.0] * self.n
        for j in neighbors:
            for k in range(self.n):
                a_vec[k] += self.step_vectors[j][k]
        return [v / len(neighbors) for v in a_vec]

    # Fungsi ini untuk menghitung cohesion: tarikan mendekat ke pusat massa agen tetangga
    def _cohesion(self, i, neighbors):
        if not neighbors:
            return [0.0] * self.n
        c_vec = [0.0] * self.n
        for j in neighbors:
            for k in range(self.n):
                c_vec[k] += self.positions[j][k]
        mean_pos = [v / len(neighbors) for v in c_vec]
        return [mean_pos[k] - self.positions[i][k] for k in range(self.n)]

    # Fungsi ini untuk menghitung food factor: tarikan menuju agen/solusi terbaik
    def _food_factor(self, i):
        if self.best_agent_idx is None:
            return [0.0] * self.n
        food = self.positions[self.best_agent_idx]
        return [food[k] - self.positions[i][k] for k in range(self.n)]

    # Fungsi ini untuk menghitung enemy factor: tolakan menjauh dari agen/solusi terburuk
    def _enemy_factor(self, i):
        if self.worst_agent_idx is None:
            return [0.0] * self.n
        enemy = self.positions[self.worst_agent_idx]
        return [enemy[k] + self.positions[i][k] for k in range(self.n)]

    # Fungsi ini untuk menghasilkan gerakan acak Levy flight (dipakai saat agen tidak punya tetangga)
    def _levy_flight(self):
        beta = 1.5
        sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
                 (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        return [0.01 * random.gauss(0, sigma) / (abs(random.gauss(0, 1)) ** (1 / beta) + 1e-9)
                for _ in range(self.n)]

    # Fungsi ini untuk memperbarui posisi satu agen dragonfly berdasarkan S, A, C, F, E
    def _update_agent(self, i, radius):
        neighbors = self._get_neighbors(i, radius)
        if neighbors:
            # S, A, C, F, E berturut-turut adalah separation, alignment,
            # cohesion, food factor, dan enemy factor untuk agen ke-i
            S = self._separation(i, neighbors)
            A = self._alignment(i, neighbors)
            C = self._cohesion(i, neighbors)
            F = self._food_factor(i)
            E = self._enemy_factor(i)
            new_step = [
                self.s * S[k] + self.a * A[k] + self.c * C[k] +
                self.f * F[k] + self.e * E[k] + self.w * self.step_vectors[i][k]
                for k in range(self.n)
            ]
            self.step_vectors[i] = new_step
            self.positions[i] = [max(0.0, min(1.0, self.positions[i][k] + new_step[k]))
                                  for k in range(self.n)]
        else:
            levy = self._levy_flight()
            self.positions[i] = [max(0.0, min(1.0, self.positions[i][k] + levy[k]))
                                  for k in range(self.n)]

    # Fungsi ini untuk menjalankan seluruh iterasi Dragonfly Algorithm sampai selesai
    def run(self, verbose: bool = False) -> Tuple[List[List[int]], Tuple[int, float]]:
        self.initialize_population()
        for t in range(self.max_iter):
            radius = 0.5 * (1 - t / self.max_iter) + 0.05
            self.w = 0.9 - 0.5 * (t / self.max_iter)
            for i in range(self.num_agents):
                self._update_agent(i, radius)
            self._evaluate_population()

            

        return self.best_routes, self.best_fitness


# =============================================================================
# VISUALISASI RUTE
# =============================================================================

# Fungsi ini untuk menggambar visualisasi rute VRPTW ke dalam Figure matplotlib
def visualize_routes(inst: VRPTWInstance, routes: List[List[int]], title: str):
    import matplotlib.pyplot as plt

    id_to_cust = {c.id: c for c in inst.customers}
    fig, ax = plt.subplots(figsize=(9, 6.5))

    depot = inst.customers[0]
    ax.scatter([depot.x], [depot.y], c="black", marker="*", s=260, label="Depot", zorder=5)

    for cust in inst.customers[1:]:
        ax.scatter(cust.x, cust.y, c="gray", marker="o", s=45, zorder=4)
        ax.text(cust.x, cust.y, str(cust.id), fontsize=7, ha="center", va="bottom",
                color="black", zorder=6, fontweight="bold")

    colors = plt.cm.tab20.colors
    for idx, route in enumerate(routes):
        xs = [id_to_cust[cid].x for cid in route]
        ys = [id_to_cust[cid].y for cid in route]
        ax.plot(xs, ys, marker="", linewidth=1.4,
                color=colors[idx % len(colors)], label=f"Rute {idx + 1}")

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Koordinat X", fontsize=11)
    ax.set_ylabel("Koordinat Y", fontsize=11)
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1), ncol=1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
