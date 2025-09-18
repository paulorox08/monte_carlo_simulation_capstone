import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.cm as cm

# Load app parameters
PARAMS_PATH = Path(__file__).parent / "household_traffic_app_params_(1).json"
with open(PARAMS_PATH, "r") as f:
    APP_PARAMS = json.load(f)

APP_NAMES = [app["name"] for app in APP_PARAMS]  # preserve ordering

# Generate a consistent color map for apps
colors = cm.get_cmap('tab20', len(APP_NAMES))
APP_COLORS = {app: colors(i) for i, app in enumerate(APP_NAMES)}

def triangular_sample(min_val, mode_val, max_val):
    return np.random.triangular(min_val, mode_val, max_val)

def simulate_one_day(n_users=4, minutes_per_day=1440):
    results = []
    for app in APP_PARAMS:
        lam = 1.0 / app["interarrival_mean_min_per_user"]
        active_users = int(np.ceil(n_users * app["daily_active_users_fraction"]))
        for user in range(active_users):
            t = 0.0
            while t < minutes_per_day:
                hour = int(t // 60)
                weight = app.get("diurnal_profile", [1.0]*24)[hour]
                adjusted_lam = lam * weight
                if adjusted_lam <= 0:
                    t += 1
                    continue
                t += np.random.exponential(scale=1/adjusted_lam)
                if t >= minutes_per_day:
                    break
                session_length = max(1, int(np.random.lognormal(
                    mean=np.log(app["session_length"]["mean"]),
                    sigma=app["session_length"]["sigma"]) * 60))
                bitrate = triangular_sample(
                    app["bitrate"]["min"],
                    app["bitrate"]["mode"],
                    app["bitrate"]["max"])

                results.append({
                    "app": app["name"],
                    "user": user,
                    "start_min": int(t),
                    "duration_min": min(session_length, minutes_per_day - int(t)),
                    "down_Mbps": bitrate,
                    "up_Mbps": bitrate * app["upstream_factor"]
                })
    return pd.DataFrame(results)

def monte_carlo_simulation(iterations=10000, n_users_range=(2,6)):
    peak_demands = []
    for i in range(iterations):
        n_users = np.random.randint(n_users_range[0], n_users_range[1]+1)
        df = simulate_one_day(n_users=n_users)
        minutes_per_day = 1440
        down_total = np.zeros(minutes_per_day)
        for _, row in df.iterrows():
            start, end = row["start_min"], row["start_min"] + row["duration_min"]
            down_total[start:end] += row["down_Mbps"]
        peak_demands.append(down_total.max())
    return np.array(peak_demands)

def analyze_results(peak_demands):
    stats = {
        "mean": np.mean(peak_demands),
        "median": np.median(peak_demands),
        "variance": np.var(peak_demands),
        "std_dev": np.std(peak_demands),
        "p95": np.percentile(peak_demands, 95),
        "p99": np.percentile(peak_demands, 99)
    }
    return stats

def plot_analysis(peak_demands):
    plt.figure(figsize=(10,5))
    plt.hist(peak_demands, bins=50, density=True, alpha=0.6)
    plt.title("Histogram of Peak Bandwidth Demand")
    plt.xlabel("Peak Mbps")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig("peak_histogram.png")
    plt.show()

    # CDF Plot
    sorted_demands = np.sort(peak_demands)
    cdf = np.arange(len(sorted_demands)) / float(len(sorted_demands))
    plt.figure(figsize=(10,5))
    plt.plot(sorted_demands, cdf)
    plt.title("Cumulative Distribution of Peak Demand")
    plt.xlabel("Peak Mbps")
    plt.ylabel("CDF")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("peak_cdf.png")
    plt.show()

def plot_stacked_area(df):
    minutes_per_day = 1440
    app_totals = {app: np.zeros(minutes_per_day) for app in APP_NAMES}

    for _, row in df.iterrows():
        start, end = row["start_min"], row["start_min"] + row["duration_min"]
        app_totals[row["app"]][start:end] += row["down_Mbps"]

    # Sort apps by total traffic for a cleaner legend
    app_totals = dict(sorted(app_totals.items(), key=lambda x: x[1].sum(), reverse=True))

    plt.figure(figsize=(12,6))
    times = np.arange(minutes_per_day) / 60.0
    colors_ordered = [APP_COLORS[app] for app in app_totals.keys()]
    plt.stackplot(times, *app_totals.values(), labels=app_totals.keys(), colors=colors_ordered, alpha=0.8)
    plt.legend(loc="upper left", fontsize=8, ncol=2)
    plt.title("Stacked Area Chart of Bandwidth Usage by Application")
    plt.xlabel("Hour of Day")
    plt.ylabel("Downstream Mbps")
    plt.tight_layout()
    plt.savefig("stacked_area_apps.png")
    plt.show()

    # Hourly summary table
    hourly_summary = pd.DataFrame(index=np.arange(24))
    for app, data in app_totals.items():
        hourly_summary[app] = data.reshape(24,60).mean(axis=1)
    print("\n=== Hourly Average Bandwidth per App (Mbps) ===")
    print(hourly_summary)

def plot_hourly_profile(all_down_profiles):
    hourly_means = all_down_profiles.reshape(all_down_profiles.shape[0], 24, 60).mean(axis=(0,2))
    plt.figure(figsize=(10,5))
    plt.plot(range(24), hourly_means, marker="o")
    plt.title("Average Hourly Downstream Load")
    plt.xlabel("Hour of Day")
    plt.ylabel("Average Mbps")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("hourly_profile.png")
    plt.show()

def main():
    print("Running Monte Carlo simulation with 10,000 iterations...")
    peak_demands = monte_carlo_simulation()
    stats = analyze_results(peak_demands)
    print("\n=== Summary Statistics ===")
    for k,v in stats.items():
        print(f"{k}: {v:.2f}")

    df_one_day = simulate_one_day()
    plot_analysis(peak_demands)
    plot_stacked_area(df_one_day)
    plot_hourly_profile(df_one_day)

if __name__ == "__main__":
    main()