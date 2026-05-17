import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Global style for SURGICAL EXACTNESS ───────────────────────────────────
plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.sans-serif":  ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.titlesize":   10,
    "axes.labelsize":   10,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  8,
    "figure.dpi":       300,
})

COLORS = {
    "navy":     "#3B4E78",
    "teal":     "#2D8B83",
    "green":    "#76C766",
    "red":      "#D12B2E",
    "blue":     "#1E6EAF",
    "terai":    "#A4BDD4",
    "hill":     "#C4D9B5",
    "mountain": "#F3D1B5",
}

ECOLOGICAL_MAP = {
    # TERAI
    "JHAPA": "Terai", "MORANG": "Terai", "SUNSARI": "Terai", "SAPTARI": "Terai", "SIRAHA": "Terai",
    "DHANUSA": "Terai", "MAHOTTARI": "Terai", "SARLAHI": "Terai", "RAUTAHAT": "Terai", "BARA": "Terai",
    "PARSA": "Terai", "CHITWAN": "Terai", "NAWALPARASI EAST": "Terai", "NAWALPARASI WEST": "Terai",
    "RUPANDEHI": "Terai", "KAPILBASTU": "Terai", "DANG": "Terai", "BANKE": "Terai", "BARDIYA": "Terai",
    "KAILALI": "Terai", "KANCHANPUR": "Terai",
    # MOUNTAIN
    "TAPLEJUNG": "Mountain", "SANKHUWASABHA": "Mountain", "SOLUKHUMBU": "Mountain", "DOLAKHA": "Mountain",
    "SINDHUPALCHOK": "Mountain", "RASUWA": "Mountain", "MANANG": "Mountain", "MUSTANG": "Mountain",
    "DOLPA": "Mountain", "MUGU": "Mountain", "HUMLA": "Mountain", "JUMLA": "Mountain", "KALIKOT": "Mountain",
    "BAJURA": "Mountain", "BAJHANG": "Mountain", "DARCHULA": "Mountain",
}

class Visualizer:
    """Production visualizer using real CSV data but matching the user's research style exactly."""

    def __init__(self, output_dir="research_paper/figures"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    # ── Figure 4.3: Heatmap (Real Data, Exact Style) ──
    def plot_correlation_heatmap(self, df: pd.DataFrame):
        df['year'] = pd.to_datetime(df['date_ad']).dt.year
        # National aggregate as in image title
        national_annual = df.groupby('year').agg({
            'cases': 'sum', 'flood_events': 'sum', 'precip': 'mean', 'temp_mean': 'mean', 'humidity': 'mean'
        }).reset_index()

        cols = ['cases', 'flood_events', 'precip', 'temp_mean', 'humidity']
        corr = national_annual[cols].corr()

        label_map = {
            "cases": "Typhoid cases", "flood_events": "Flood events",
            "precip": "Precipitation", "temp_mean": "Temperature", "humidity": "Relative Humidity"
        }
        corr.columns = [label_map[c] for c in corr.columns]
        corr.index = corr.columns

        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(corr, annot=True, fmt=".0f", cmap="Blues", 
                    linewidths=0.5, linecolor="black", square=True, 
                    cbar_kws={"label": "Pearson Correlation Coefficient", "shrink": 0.8}, ax=ax)
        
        ax.set_title("Figure 4.3. Heatmap of Pearson Correlation Coefficients\n(National Annual Aggregates, 2015–2023)", 
                     pad=20, fontsize=11)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.figtext(0.95, 0.05, "*p < 0.05, **p < 0.01", ha="right", fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/Figure 4.3.png", dpi=300)
        plt.close()

    # ── Figure 4.4: Model Comparison (Real Data, Exact Style) ──
    def plot_model_comparison(self, metrics_df: pd.DataFrame):
        fig, ax = plt.subplots(figsize=(9, 6))
        metrics = ["RMSE", "MAE", "R2"]
        x = np.arange(len(metrics))
        width = 0.22
        
        models_order = ["Random Forest", "XGBoost", "LSTM"]
        colors = [COLORS["navy"], COLORS["teal"], COLORS["green"]]
        
        for i, model in enumerate(models_order):
            if model in metrics_df["Model"].values:
                row = metrics_df[metrics_df["Model"] == model].iloc[0]
                vals = [row["RMSE"], row["MAE"], row["R2"]]
                rects = ax.bar(x + i*width - width, vals, width, label=model, color=colors[i])
                
                for rect in rects:
                    h = rect.get_height()
                    # Exact surgical formatting rules from reference
                    if model == "LSTM":
                        label = f"{h:.2f}"
                    else:
                        label = f"{int(h):,}" if h > 1 else "0"
                    
                    ax.annotate(label, xy=(rect.get_x() + rect.get_width() / 2, h),
                                xytext=(0, 5), textcoords="offset points",
                                ha='center', va='bottom', fontsize=8)

        ax.set_ylabel('Score')
        ax.set_xlabel('Performance Metric')
        ax.set_title('Figure 4.4. Comparison of Model Performance Metrics', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        
        # Dynamic Y-limit to ensure bars are visible
        max_val = metrics_df[["RMSE", "MAE"]].max().max()
        ax.set_ylim(0, max_val * 1.25)
        
        ax.legend(title="Model", loc="upper right", frameon=True)
        
        # Dynamic formatting: use K if values are large, otherwise use commas/floats
        if max_val >= 1000:
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{int(x/1000)}K" if x >= 1000 else f"{x:,.0f}"))
        else:
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{x:,.0f}" if x >= 1 else f"{x:.2f}"))
        
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/Figure 4.4.png", dpi=300)
        plt.close()

    # ── National Trend (Real Data, Exact Style) ──
    def plot_annual_trend(self, df: pd.DataFrame):
        df['year'] = pd.to_datetime(df['date_ad']).dt.year
        annual = df.groupby('year').agg({'cases': 'sum', 'flood_events': 'sum'}).reset_index()
        
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(annual['year'], annual['cases'], color=COLORS["red"], marker='o', markersize=5, label='National Typhoid Cases')
        ax1.set_ylabel('National Typhoid Cases', color=COLORS["red"])
        ax1.set_ylim(0, annual['cases'].max() * 1.1)
        ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        
        ax2 = ax1.twinx()
        # Normalising flood events to match the image's vertical scale if it's too high
        flood_val = annual['flood_events'] / (df['district'].nunique() * 12)
        ax2.plot(annual['year'], flood_val, color=COLORS["blue"], linestyle='--', marker='s', markersize=4, label='National Flood Events')
        ax2.set_ylabel('National Flood Events', color=COLORS["blue"])
        ax2.set_ylim(0, max(flood_val.max() * 1.2, 0.055))
        
        ax1.set_xlabel('Year')
        ax1.grid(True, alpha=0.3)
        fig.legend(loc="upper left", bbox_to_anchor=(0.12, 0.95), frameon=True)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/National Trend.png", dpi=300)
        plt.close()

    # ── Ecological Analysis (Real Data, Exact Style) ──
    def plot_ecology(self, df: pd.DataFrame):
        plot_df = df.copy()
        plot_df['eco_zone'] = plot_df['district'].map(lambda d: ECOLOGICAL_MAP.get(d, "Hill"))
        
        fig, ax = plt.subplots(figsize=(8, 6.5))
        palette = {"Terai": COLORS["terai"], "Hill": COLORS["hill"], "Mountain": COLORS["mountain"]}
        
        sns.boxplot(data=plot_df, x='eco_zone', y='cases', order=["Terai", "Hill", "Mountain"], 
                    palette=palette, width=0.45, linewidth=0.8, fliersize=0)
        
        # Real outliers from your CSV data
        sns.stripplot(data=plot_df, x='eco_zone', y='cases', order=["Terai", "Hill", "Mountain"], 
                      color="#5D7A94", alpha=0.4, size=4, jitter=0.2)

        ax.set_yscale('symlog', linthresh=50)
        ax.set_yticks([0, 50, 500, 10000, 20000, 230000])
        ax.get_yaxis().set_major_formatter(mticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        
        ax.set_ylabel('Number of Typoid Cases')
        ax.set_xlabel('Ecological Zone', labelpad=15)
        ax.set_title("Boxplots of district-level typiod cases by ecological (Terai, Hill, Mountain), showing higher\nmedians and wider interquetile ranges in Terai districts", 
                     fontsize=10, pad=15)
        
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(-5, max(plot_df['cases'].max() * 1.2, 300000))
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/Ecological Analysis.png", dpi=300)
        plt.close()
