"""
app_unit_vrptw.py

Antarmuka Streamlit untuk pengujian unit (tanpa integrasi): Solomon I1
Insertion, Dragonfly Algorithm, dan hybrid Nearest Neighbor + Dragonfly
Algorithm dijalankan terpisah lalu dibandingkan.

Jalankan dengan:
    streamlit run app_unit_vrptw.py
"""

import time
import streamlit as st

from vrptw_core import (
    list_datasets, read_csv_instance, visualize_routes,
    solomon_i1_insertion, I1Parameters, PureDragonflyAlgorithm,
    evaluate_fitness, evaluate_total_time,
    nearest_neighbor_construction, encode_routes_to_vector,
)

st.set_page_config(page_title="Pengujian VRPTW", layout="centered")

# Ini adalah state untuk menyimpan hasil pengujian antar rerun Streamlit
if "results" not in st.session_state:
    st.session_state.results = None
# Ini adalah state untuk menyimpan instance VRPTW yang sedang diuji
if "inst" not in st.session_state:
    st.session_state.inst = None
# Ini adalah state untuk menyimpan nama/label instance yang sedang diuji
if "inst_name" not in st.session_state:
    st.session_state.inst_name = None
# Ini adalah state untuk menyimpan mode pengujian yang terakhir dijalankan
if "test_mode_used" not in st.session_state:
    st.session_state.test_mode_used = None

st.title("Pengujian Optimasi VRPTW")

# ---------------------------------------------------------------------------
# MODE PENGUJIAN: Tanpa Integrasi
# ---------------------------------------------------------------------------
st.subheader("Tanpa Integrasi")
st.caption(
    "Baseline Solomon I1 dan Hybrid Nearest Neighbor dengan Algoritma Dragonfly"
    "Hasil tiap unit ditampilkan sendiri "
    "lalu dibandingkan."
)

# ---------------------------------------------------------------------------
# TAHAP 1 - INPUT DATA
# ---------------------------------------------------------------------------
st.header("Tahap 1 — Input data")

# Ini adalah daftar dataset (.csv) yang ditemukan di folder dataset
datasets = list_datasets("dataset")

if not datasets:
    st.warning(
        "Belum ada dataset (.csv) di folder 'dataset/'. Jalankan "
        "generate_sample_dataset.py untuk membuat contoh dataset, atau "
        "salin file dataset kamu sendiri ke folder tersebut."
    )
    st.stop()

col1, col2 = st.columns(2)
with col1:
    selected_dataset = st.selectbox("Pilih dataset", sorted(datasets.keys()))
with col2:
    n_customers = st.selectbox("Jumlah customers", [25, 50, 100, 200], index=1)

col3, col4, col5 = st.columns(3)
with col3:
    da_runs = st.number_input("Jumlah run DA", min_value=1, max_value=50, value=10)
with col4:
    da_max_iter = st.number_input("Maks. iterasi DA", min_value=10, max_value=2000, value=300, step=10)
with col5:
    da_num_agents = st.number_input("Agents DA", min_value=5, max_value=200, value=40, step=5)
    st.caption("jumlah agen/populasi")

start = st.button("▶  Mulai pengujian", type="primary", use_container_width=True)


# ---------------------------------------------------------------------------
# Fungsi bantu: render Tahap 3
# ---------------------------------------------------------------------------
def _route_color_hex(idx: int) -> str:
    """Warna yang dipakai untuk Rute ke-(idx+1)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex
    colors = plt.cm.tab20.colors
    return to_hex(colors[idx % len(colors)])


def render_route_list(inst, routes):
    """Tampilkan daftar customer yang dilalui tiap rute."""
    lines = []
    for idx, route in enumerate(routes):
        color_hex = _route_color_hex(idx)
        last = len(route) - 1
        path_str = " → ".join(
            "Depot" if pos in (0, last) else str(cid)
            for pos, cid in enumerate(route)
        )
        lines.append(
            f'<div style="margin-bottom:4px;">'
            f'<span style="display:inline-block;width:12px;height:12px;'
            f'background-color:{color_hex};margin-right:6px;'
            f'border-radius:2px;vertical-align:middle;"></span>'
            f'<b>Rute {idx + 1}</b> = {path_str}'
            f'</div>'
        )
    st.markdown("".join(lines), unsafe_allow_html=True)


# Fungsi ini untuk menampilkan seluruh hasil perbandingan Tahap 3
def render_tahap3(results, inst, inst_name):
    st.header("Tahap 3 — Hasil")
    st.subheader("Ringkasan hasil perbandingan")

    # i1, da, nn_da adalah hasil dari masing-masing unit: Solomon I1,
    # Dragonfly Algorithm murni, dan hybrid Nearest Neighbor + Dragonfly
    i1 = results["i1"]
    da = results["da"]
    nn_da = results["nn_da"]

    # Ini adalah baris-baris tabel ringkasan hasil tiap metode
    table_rows = [
        {
            "Metode": "Solomon I1 Insertion",
            "Kendaraan": i1["fitness"][0],
            "Total jarak": round(i1["fitness"][1], 2),
            "Total waktu kunjungan": round(i1["total_time"], 1),
        },
        {
            "Metode": "Dragonfly Algorithm (terbaik)",
            "Kendaraan": da["best_fitness"][0],
            "Total jarak": round(da["best_fitness"][1], 2),
            "Total waktu kunjungan": round(da["best_total_time"], 1),
        },
        {
            "Metode": "Nearest Neighbor (rute awal, sebelum DA)",
            "Kendaraan": nn_da["nn_fitness"][0],
            "Total jarak": round(nn_da["nn_fitness"][1], 2),
            "Total waktu kunjungan": round(nn_da["nn_total_time"], 1),
        },
        {
            "Metode": "NN + Dragonfly Algorithm (terbaik)",
            "Kendaraan": nn_da["best_fitness"][0],
            "Total jarak": round(nn_da["best_fitness"][1], 2),
            "Total waktu kunjungan": round(nn_da["best_total_time"], 1),
        },
    ]
    st.dataframe(table_rows, hide_index=True, use_container_width=True)

    st.subheader("Perbandingan Solomon I1 vs NN + Dragonfly Algorithm")

    # Ini adalah hasil Solomon I1 (kendaraan, jarak, waktu) sebagai acuan
    solomon_kendaraan = i1["fitness"][0]
    solomon_jarak = i1["fitness"][1]
    solomon_waktu = i1["total_time"]
    # Ini adalah hasil terbaik NN + Dragonfly Algorithm
    nnda_kendaraan = nn_da["best_fitness"][0]
    nnda_jarak = nn_da["best_fitness"][1]
    nnda_waktu = nn_da["best_total_time"]

    # Ini adalah baris-baris tabel perbandingan Solomon I1 vs NN + DA
    perbandingan_table = [
        {
            "Metode": "Solomon I1",
            "Total kendaraan": solomon_kendaraan,
            "Total jarak": round(solomon_jarak, 2),
            "Total waktu kunjungan": round(solomon_waktu, 1),
        },
        {
            "Metode": "NN + DA (terbaik)",
            "Total kendaraan": nnda_kendaraan,
            "Total jarak": round(nnda_jarak, 2),
            "Total waktu kunjungan": round(nnda_waktu, 1),
        },
    ]
    st.dataframe(perbandingan_table, hide_index=True, use_container_width=True)

    # Fungsi ini untuk menghitung gap (%) hasil terhadap acuan (Solomon I1)
    def _tdg(acuan, hasil):
        return (hasil - acuan) / acuan * 100

    st.subheader("Gap Solomon I1 (acuan) vs Dragonfly Algorithm & NN + Dragonfly Algorithm")
    st.caption(
        "Format mengikuti gaya Tabel Perbandingan Kinerja ODA vs ODA-VNS: "
        "NV = jumlah kendaraan, NVG = selisih NV terhadap Solomon I1 (acuan), "
        "TD = total jarak, TDG (%) = gap jarak terhadap Solomon I1. "
        "Nilai positif berarti lebih besar/panjang dari Solomon I1."
    )

    # Ini adalah nilai NV (jumlah kendaraan) dan TD (total jarak) tiap metode
    solomon_nv = i1["fitness"][0]
    solomon_td = i1["fitness"][1]
    da_nv = da["best_fitness"][0]
    da_td = da["best_fitness"][1]
    nnda_nv = nn_da["best_fitness"][0]
    nnda_td = nn_da["best_fitness"][1]

    # Fungsi ini untuk menghitung NVG: selisih jumlah kendaraan terhadap Solomon I1
    def _nvg(nv_hasil):
        diff = nv_hasil - solomon_nv
        return "0" if diff == 0 else f"{diff:+d}"

    # Fungsi ini untuk memformat TDG: gap jarak (%) terhadap acuan
    def _fmt_tdg(acuan, hasil):
        gap = _tdg(acuan, hasil)
        return "0.00" if gap == 0 else f"{gap:+.2f}"

    # Ini adalah HTML tabel Gap Solomon I1 vs DA & NN+DA
    html_table = f"""
    <style>
    .tabel47 {{ border-collapse: collapse; width: 100%; font-size: 0.9rem; }}
    .tabel47 th, .tabel47 td {{ border: 1px solid #999; padding: 4px 8px; text-align: center; color: #111827; }}
    .tabel47 th {{ background-color: #e6e9ef; font-weight: 600; }}
    .tabel47 td {{ background-color: #ffffff; }}
    </style>
    <table class="tabel47">
      <thead>
        <tr>
          <th rowspan="2">Instance</th>
          <th rowspan="2">Capacity</th>
          <th colspan="2">Solomon I1 (acuan)</th>
          <th colspan="4">Dragonfly Algorithm</th>
          <th colspan="4">NN + Dragonfly Algorithm</th>
        </tr>
        <tr>
          <th>NV</th><th>TD</th>
          <th>NV</th><th>NVG</th><th>TD</th><th>TDG (%)</th>
          <th>NV</th><th>NVG</th><th>TD</th><th>TDG (%)</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>{inst_name}</td>
          <td>{inst.capacity}</td>
          <td>{solomon_nv}</td>
          <td>{solomon_td:.2f}</td>
          <td>{da_nv}</td>
          <td>{_nvg(da_nv)}</td>
          <td>{da_td:.2f}</td>
          <td>{_fmt_tdg(solomon_td, da_td)}</td>
          <td>{nnda_nv}</td>
          <td>{_nvg(nnda_nv)}</td>
          <td>{nnda_td:.2f}</td>
          <td>{_fmt_tdg(solomon_td, nnda_td)}</td>
        </tr>
      </tbody>
    </table>
    """
    st.markdown(html_table, unsafe_allow_html=True)

    st.subheader("Visualisasi rute")
    # Ini adalah pilihan metode rute yang ingin ditampilkan penggunanya
    view = st.radio(
        "Tampilkan rute",
        ["Dragonfly Algorithm (terbaik)", "NN + Dragonfly Algorithm (terbaik)", "Solomon I1 Insertion"],
        horizontal=True, key="view_radio",
    )

    # fig adalah grafik rute yang ditampilkan; active_routes adalah rute yang aktif dipilih
    if view.startswith("NN"):
        fig = visualize_routes(inst, nn_da["routes"], f"VRPTW - NN + Dragonfly Algorithm ({inst_name})")
        active_routes = nn_da["routes"]
    elif view.startswith("Dragonfly"):
        fig = visualize_routes(inst, da["routes"], f"VRPTW - Dragonfly Algorithm ({inst_name})")
        active_routes = da["routes"]
    else:
        fig = visualize_routes(inst, i1["routes"], f"VRPTW - Solomon I1 ({inst_name})")
        active_routes = i1["routes"]

    st.pyplot(fig)

    st.subheader("Detail rute")
    render_route_list(inst, active_routes)


# ---------------------------------------------------------------------------
# TAHAP 2 - UNIT TESTING
# ---------------------------------------------------------------------------
if start:
    st.header("Tahap 2 — Proses (Unit Testing)")

    try:
        # inst adalah instance VRPTW yang dibaca dari file CSV dataset terpilih
        inst = read_csv_instance(
            filepath=datasets[selected_dataset],
            capacity=None,
            num_vehicles=None,
            n_customers=int(n_customers),
        )
    except ValueError as e:
        st.error(str(e))
        st.stop()

    # Ini adalah parameter untuk algoritma Solomon I1 Insertion
    params = I1Parameters(mu=1.0, lam=2.0, alpha1=1.0, alpha2=0.0)

    # --- UNIT 1: Solomon I1 Insertion ---
    st.subheader("Unit 1 — Solomon I1 Insertion (dijalankan sendiri)")
    with st.spinner("Menjalankan solomon_i1_insertion()..."):
        t0 = time.perf_counter()
        # i1_routes adalah rute hasil Solomon I1 Insertion
        i1_routes = solomon_i1_insertion(inst, params)
        i1_time = time.perf_counter() - t0
        # i1_fitness adalah (jumlah kendaraan, total jarak) dari i1_routes
        i1_fitness = evaluate_fitness(inst, i1_routes)
        i1_total_time = evaluate_total_time(inst, i1_routes)

    st.dataframe([{
        "Kendaraan": i1_fitness[0],
        "Total jarak": round(i1_fitness[1], 2),
        "Total waktu kunjungan": round(i1_total_time, 1),
        "Waktu komputasi (detik)": round(i1_time, 4),
    }], hide_index=True, use_container_width=True)
    st.caption(f"solomon_i1_insertion() menghasilkan {len(i1_routes)} rute, dijalankan 1x (deterministik).")

    # --- UNIT 2: Dragonfly Algorithm (populasi acak murni) ---
    st.subheader("Unit 2 — Dragonfly Algorithm (dijalankan sendiri, populasi acak murni)")
    st.caption("seed_vector TIDAK diisi di sini -- DA tidak mengambil solusi awal dari Unit 1, murni diuji berdiri sendiri.")

    status_text = st.empty()
    progress_bar = st.progress(0.0)

    # Ini adalah penampung hasil (fitness, waktu) tiap run Dragonfly Algorithm
    da_results = []
    da_times = []
    da_total_times = []
    # Ini adalah rute dan fitness terbaik dari seluruh run
    da_best_routes = None
    da_best_fitness = (float("inf"), float("inf"))

    for run_idx in range(int(da_runs)):
        status_text.info(f"Menjalankan PureDragonflyAlgorithm secara langsung (tanpa seed I1)... run ke-{run_idx + 1} dari {int(da_runs)}")
        t0 = time.perf_counter()
        da = PureDragonflyAlgorithm(
            inst, num_agents=int(da_num_agents), max_iter=int(da_max_iter),
            seed=42 + run_idx, seed_vector=None,
        )
        best_routes, best_fitness = da.run(verbose=False)
        elapsed = time.perf_counter() - t0

        da_results.append(best_fitness)
        da_times.append(elapsed)
        da_total_times.append(evaluate_total_time(inst, best_routes))
        if best_fitness < da_best_fitness:
            da_best_fitness = best_fitness
            da_best_routes = best_routes

        progress_bar.progress((run_idx + 1) / int(da_runs))

    status_text.empty()
    progress_bar.empty()

    # avg_* adalah rata-rata hasil seluruh run; best_run adalah hasil terbaik
    avg_vehicles = sum(r[0] for r in da_results) / int(da_runs)
    avg_distance = sum(r[1] for r in da_results) / int(da_runs)
    avg_total_time = sum(da_total_times) / int(da_runs)
    best_run = min(da_results)
    best_run_idx = da_results.index(best_run)
    best_total_time = da_total_times[best_run_idx]
    avg_time = sum(da_times) / int(da_runs)

    st.dataframe([
        {
            "Metode": "Dragonfly Algorithm (rata-rata)",
            "Kendaraan": round(avg_vehicles, 2),
            "Total jarak": round(avg_distance, 2),
            "Total waktu kunjungan": round(avg_total_time, 1),
            "Waktu komputasi rata-rata (detik)": round(avg_time, 4),
        },
        {
            "Metode": "Dragonfly Algorithm (terbaik)",
            "Kendaraan": best_run[0],
            "Total jarak": round(best_run[1], 2),
            "Total waktu kunjungan": round(best_total_time, 1),
            "Waktu komputasi (detik)": round(da_times[best_run_idx], 4),
        },
    ], hide_index=True, use_container_width=True)
    st.caption(f"PureDragonflyAlgorithm.run() dipanggil {int(da_runs)}x secara independen (unit terpisah dari Solomon I1).")

    # --- UNIT 3: Nearest Neighbor + Dragonfly Algorithm (hybrid) ---
    st.subheader("Unit 3 — Nearest Neighbor + Dragonfly Algorithm (hybrid)")
    st.caption(
        "nearest_neighbor_construction() dijalankan 1x untuk membangun rute "
        "awal, lalu di-encode jadi seed_vector -- SELURUH populasi awal DA "
        "diturunkan dari rute NN tsb (format hybrid NN + DA)."
    )

    with st.spinner("Menjalankan nearest_neighbor_construction()..."):
        t0 = time.perf_counter()
        # nn_routes adalah rute awal hasil Nearest Neighbor construction
        nn_routes = nearest_neighbor_construction(inst)
        nn_time = time.perf_counter() - t0
        nn_fitness = evaluate_fitness(inst, nn_routes)
        nn_total_time = evaluate_total_time(inst, nn_routes)
        # nn_seed_vector adalah encoding rute NN, dipakai sebagai seed populasi awal DA
        nn_seed_vector = encode_routes_to_vector(inst, nn_routes)

    st.dataframe([{
        "Kendaraan": nn_fitness[0],
        "Total jarak": round(nn_fitness[1], 2),
        "Total waktu kunjungan": round(nn_total_time, 1),
        "Waktu komputasi (detik)": round(nn_time, 4),
    }], hide_index=True, use_container_width=True)
    st.caption(f"nearest_neighbor_construction() menghasilkan {len(nn_routes)} rute awal (sebelum DA), dijalankan 1x (deterministik).")

    status_text = st.empty()
    progress_bar = st.progress(0.0)

    # Ini adalah penampung hasil (fitness, waktu) tiap run NN + Dragonfly Algorithm
    nnda_results = []
    nnda_times = []
    nnda_total_times = []
    # Ini adalah rute dan fitness terbaik dari seluruh run
    nnda_best_routes = None
    nnda_best_fitness = (float("inf"), float("inf"))

    for run_idx in range(int(da_runs)):
        status_text.info(f"Menjalankan PureDragonflyAlgorithm dengan seed dari NN... run ke-{run_idx + 1} dari {int(da_runs)}")
        t0 = time.perf_counter()
        nnda = PureDragonflyAlgorithm(
            inst, num_agents=int(da_num_agents), max_iter=int(da_max_iter),
            seed=42 + run_idx, seed_vector=nn_seed_vector,
        )
        best_routes, best_fitness = nnda.run(verbose=False)
        elapsed = time.perf_counter() - t0

        nnda_results.append(best_fitness)
        nnda_times.append(elapsed)
        nnda_total_times.append(evaluate_total_time(inst, best_routes))
        if best_fitness < nnda_best_fitness:
            nnda_best_fitness = best_fitness
            nnda_best_routes = best_routes

        progress_bar.progress((run_idx + 1) / int(da_runs))

    status_text.empty()
    progress_bar.empty()

    # nnda_avg_* adalah rata-rata hasil seluruh run; nnda_best_run adalah hasil terbaik
    nnda_avg_vehicles = sum(r[0] for r in nnda_results) / int(da_runs)
    nnda_avg_distance = sum(r[1] for r in nnda_results) / int(da_runs)
    nnda_avg_total_time = sum(nnda_total_times) / int(da_runs)
    nnda_best_run = min(nnda_results)
    nnda_best_run_idx = nnda_results.index(nnda_best_run)
    nnda_best_total_time = nnda_total_times[nnda_best_run_idx]
    nnda_avg_time = sum(nnda_times) / int(da_runs)

    st.dataframe([
        {
            "Metode": "NN + Dragonfly Algorithm (rata-rata)",
            "Kendaraan": round(nnda_avg_vehicles, 2),
            "Total jarak": round(nnda_avg_distance, 2),
            "Total waktu kunjungan": round(nnda_avg_total_time, 1),
            "Waktu komputasi rata-rata (detik)": round(nnda_avg_time, 4),
        },
        {
            "Metode": "NN + Dragonfly Algorithm (terbaik)",
            "Kendaraan": nnda_best_run[0],
            "Total jarak": round(nnda_best_run[1], 2),
            "Total waktu kunjungan": round(nnda_best_total_time, 1),
            "Waktu komputasi (detik)": round(nnda_times[nnda_best_run_idx], 4),
        },
    ], hide_index=True, use_container_width=True)
    st.caption(f"PureDragonflyAlgorithm.run() dipanggil {int(da_runs)}x dengan seed_vector dari NN (unit terpisah dari Unit 1 & 2).")

    # --- Gabungkan hasil ketiga unit (Tahap 3) ---
    # results adalah gabungan seluruh data dari 3 unit, dipakai render_tahap3()
    results = {
        "i1": {"fitness": i1_fitness, "time": i1_time, "routes": i1_routes,
               "total_time": i1_total_time},
        "da": {"avg_fitness": (avg_vehicles, avg_distance), "best_fitness": best_run,
               "times": da_times, "all_results": da_results, "routes": da_best_routes,
               "avg_total_time": avg_total_time, "best_total_time": best_total_time,
               "avg_time": avg_time},
        "nn_da": {"nn_fitness": nn_fitness, "nn_time": nn_time, "nn_routes": nn_routes,
                  "nn_total_time": nn_total_time,
                  "avg_fitness": (nnda_avg_vehicles, nnda_avg_distance), "best_fitness": nnda_best_run,
                  "times": nnda_times, "all_results": nnda_results, "routes": nnda_best_routes,
                  "avg_total_time": nnda_avg_total_time, "best_total_time": nnda_best_total_time,
                  "avg_time": nnda_avg_time},
    }
    st.session_state.results = results
    st.session_state.inst = inst
    st.session_state.inst_name = f"{selected_dataset} ({n_customers} customers) — Unit Testing"
    st.session_state.test_mode_used = "Unit Testing"


# ---------------------------------------------------------------------------
# TAHAP 3 - HASIL
# ---------------------------------------------------------------------------
if st.session_state.results is not None:
    render_tahap3(st.session_state.results, st.session_state.inst, st.session_state.inst_name)
    st.caption(f"Mode pengujian yang menghasilkan tabel di atas: **{st.session_state.test_mode_used}**")
