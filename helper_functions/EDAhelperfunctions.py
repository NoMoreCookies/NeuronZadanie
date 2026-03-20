# =========================
# IMPORT BIBLIOTEK
# =========================

# Operacje na danych
import pandas as pd
import numpy as np

# Wizualizacja
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

# Analiza sygnałów i statystyka
from scipy.signal import find_peaks
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests

# =========================
# WIZUALIZACJA SYGNAŁU
# =========================

def visualize_signals(
    df: pd.DataFrame,
    id_value: str,
    fs: int,
    seconds: int = 120,
    offset: float = 500,
) -> None:
    """Rysuje fragment wielokanałowego sygnału dla wybranego ID."""

    channels = [c for c in df.columns if c not in ["ID", "Class"]]

    signal_df = df.loc[df["ID"] == id_value, channels].reset_index(drop=True)

    segment_df = signal_df.iloc[: fs * seconds]
    t = np.arange(len(segment_df)) / fs

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, ch in enumerate(channels):
        sig = segment_df[ch].to_numpy()
        ax.plot(t, sig + i * offset, lw=0.2, color=f"C{i % 10}")

    ax.set_yticks(np.arange(len(channels)) * offset)
    ax.set_yticklabels(channels)
    ax.set_xlabel("Czas [s]")
    ax.set_ylabel("Kanały")
    ax.set_title(f"ID: {id_value}")
    ax.grid(axis="x", alpha=0.2)

    plt.tight_layout()
    plt.show()

# =========================
# FILTR 1 - NISKA ZMIENNOŚĆ
# =========================

def filtr_1(
    df: pd.DataFrame,
    channels: list[str],
    low_var_q: float = 0.20,
) -> pd.DataFrame:
    """Usuwa kanały o niskiej zmienności."""

    if not 0 <= low_var_q <= 1:
        raise ValueError("low_var_q musi być z przedziału [0, 1].")

    channels = [c for c in channels if c in df.columns]

    # Wariancja liczona osobno dla każdego ID
    chan_var = df.groupby("ID")[channels].var(ddof=0).quantile(0.7)

    # Próg odcięcia
    var_cut = chan_var.quantile(low_var_q)

    # Kanały do zachowania
    keep = chan_var[chan_var >= var_cut].index.tolist()

    base_cols = [c for c in ["ID", "Class"] if c in df.columns]
    return df[base_cols + keep].copy()

# =========================
# FILTR 2 - REDUNDANCJA KANAŁÓW
# =========================

def filtr_2(
    df: pd.DataFrame,
    high_corr: float = 0.7,
    std_thresh: float = 0.2
) -> pd.DataFrame:
    """Usuwa redundantne kanały na podstawie korelacji między kanałami."""

    channels = [c for c in df.columns if c not in ["ID", "Class"]]

    # Kolejność kanałów: od większej zmienności do mniejszej
    chan_var = (
        df.groupby("ID")[channels]
        .var(ddof=0)
        .quantile(0.7)
        .sort_values(ascending=False)
    )
    chan_order = chan_var.index.tolist()

    # Korelacja liczona osobno dla każdego ID
    corr_obj = df.groupby("ID")[channels].apply(lambda x: x.corr())
    corr_per_id = corr_obj.stack()

    # Średnia i odchylenie standardowe korelacji po ID
    mean_corr = corr_per_id.groupby(level=[1, 2]).mean()
    std_corr = corr_per_id.groupby(level=[1, 2]).std()

    mean_corr_mat = mean_corr.unstack().reindex(index=channels, columns=channels)
    std_corr_mat = std_corr.unstack().reindex(index=channels, columns=channels)

    stable_mask = (mean_corr_mat.abs() >= high_corr) & (std_corr_mat <= std_thresh)
    stable_mask = stable_mask.fillna(False)

    # Bez przekątnej
    for ch in channels:
        stable_mask.loc[ch, ch] = False

    removed = set()
    kept = []

    for ch in chan_order:
        if ch in removed:
            continue

        kept.append(ch)
        redundant = stable_mask.columns[stable_mask.loc[ch]].tolist()
        removed.update(redundant)

    base_cols = [c for c in ["ID", "Class"] if c in df.columns]
    return df[base_cols + kept].copy()

# =========================
# BUDOWANIE DODATKOWYCH WSKAŹNIKÓW
# =========================

def build_features(df: pd.DataFrame, fs: int = 128) -> pd.DataFrame:
    """Buduje cechy dla każdej sesji (ID) na podstawie sygnałów kanałów."""

    channels = [c for c in df.columns if c not in ["ID", "Class"]]
    features = []

    bands = {
        "delta": (0, 4),
        "theta": (4, 8),
        "alpha": (8, 12),
        "beta": (12, 30),
    }

    for id_val, session_data in df.groupby("ID"):
        class_val = session_data["Class"].iloc[0]
        duration = len(session_data) / fs

        row = {"ID": id_val, "Class": class_val, "duration": duration}

        for ch in channels:
            series = session_data[ch].to_numpy()
            s = pd.Series(series)

            row[f"{ch}_mean"] = np.mean(series)
            row[f"{ch}_std"] = np.std(series)
            row[f"{ch}_skew"] = s.skew()
            row[f"{ch}_kurt"] = s.kurtosis()
            row[f"{ch}_rms"] = np.sqrt(np.mean(series ** 2))
            row[f"{ch}_energy"] = np.sum(series ** 2)

            peaks, _ = find_peaks(series)
            row[f"{ch}_peaks"] = len(peaks)

            fft = np.fft.rfft(series)
            power = np.abs(fft) ** 2
            freqs = np.fft.rfftfreq(len(series), d=1 / fs)

            total_power = np.sum(power)
            if total_power == 0:
                total_power = 1e-10

            for band, (low, high) in bands.items():
                mask = (freqs >= low) & (freqs < high)
                row[f"{ch}_{band}_power"] = np.sum(power[mask]) / total_power

            power_norm = power / total_power
            row[f"{ch}_spectral_entropy"] = -np.sum(
                power_norm * np.log2(power_norm + 1e-10)
            )

        features.append(row)

    return pd.DataFrame(features)

# =========================
# RANKING CECH
# =========================

def rank_features_mannwhitney(
    features_df: pd.DataFrame,
    class_col: str = "Class",
    id_col: str = "ID",
    positive_class: str = "ADHD",
    negative_class: str = "Control",
    correction_method: str = "fdr_bh",
) -> pd.DataFrame:
    """Porządkuje cechy na podstawie p-value, effect size i AUC."""

    positive_data = features_df[features_df[class_col] == positive_class]
    negative_data = features_df[features_df[class_col] == negative_class]

    feature_cols = [c for c in features_df.columns if c not in [id_col, class_col]]
    results = []

    for col in feature_cols:
        x = positive_data[col].dropna()
        y = negative_data[col].dropna()

        u_stat, p_val = mannwhitneyu(x, y, alternative="two-sided")

        n1 = len(x)
        n2 = len(y)
        effect_size = 2 * u_stat / (n1 * n2) - 1

        tmp = features_df[[class_col, col]].dropna().copy()
        tmp["target"] = (tmp[class_col] == positive_class).astype(int)

        auc = roc_auc_score(tmp["target"], tmp[col])
        auc_sep = max(auc, 1 - auc)

        results.append({
            "feature": col,
            "p_value": p_val,
            "effect_size": effect_size,
            "auc_sep": auc_sep,
        })

    results_df = pd.DataFrame(results)

    results_df["p_adj"] = multipletests(
        results_df["p_value"],
        method=correction_method
    )[1]

    results_df = results_df.sort_values(
        by=["p_adj", "auc_sep"],
        ascending=[True, False]
    ).reset_index(drop=True)

    return results_df

# =========================
# FILTROWANIE CECH
# =========================

def filter_features_by_statistics(
    features_df: pd.DataFrame,
    results_df: pd.DataFrame,
    id_col: str = "ID",
    class_col: str = "Class",
    p_adj_threshold: float = 0.10,
    effect_size_threshold: float = 0.20,
    auc_threshold: float = 0.60
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """Zostawia cechy spełniające zadane progi statystyczne."""

    keep_df = results_df[
        (results_df["p_adj"] <= p_adj_threshold)
        & (results_df["effect_size"].abs() >= effect_size_threshold)
        & (results_df["auc_sep"] >= auc_threshold)
    ].copy()

    drop_df = results_df[~results_df["feature"].isin(keep_df["feature"])].copy()

    keep_feats = keep_df["feature"].tolist()
    drop_feats = drop_df["feature"].tolist()

    features_df_filtered = features_df[[id_col, class_col] + keep_feats].copy()

    return features_df_filtered, keep_df, keep_feats, drop_feats

# =========================
# WIZUALIZACJA CECH
# =========================

def draw_plot(df: pd.DataFrame, feature: str) -> None:
    """Rysuje histogram z KDE i boxplot dla wybranej cechy."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    palette = sns.color_palette("deep", 2)

    sns.histplot(
        data=df,
        x=feature,
        hue="Class",
        kde=True,
        stat="density",
        common_norm=False,
        ax=axes[0],
        palette=palette,
    )

    sns.boxplot(
        data=df,
        x=feature,
        hue="Class",
        ax=axes[1],
        palette=palette,
    )

    if axes[0].get_legend() is not None:
        axes[0].get_legend().remove()
    if axes[1].get_legend() is not None:
        axes[1].get_legend().remove()

    handles = [
        Patch(facecolor=palette[0], label="Control"),
        Patch(facecolor=palette[1], label="ADHD"),
    ]

    fig.legend(handles=handles, title="Klasa", loc="lower center", ncol=2)
    fig.suptitle(f"Porównanie rozkładu cechy: {feature}", fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.88])
    plt.show()

# =========================
# USUWANIE SILNIE SKORELOWANYCH CECH
# =========================

def drop_highly_correlated_features(
    features_df: pd.DataFrame,
    corr_threshold: float = 0.8,
    id_col: str = "ID",
    target_col: str = "Class",
) -> pd.DataFrame:
    """Usuwa silnie skorelowane cechy, zostawiając lepszą z każdej pary."""

    feature_cols = [c for c in features_df.columns if c not in [id_col, target_col]]
    num_df = features_df[feature_cols].copy()

    classes = features_df[target_col].dropna().unique()
    class_0, class_1 = classes

    df_0 = features_df[features_df[target_col] == class_0]
    df_1 = features_df[features_df[target_col] == class_1]

    y_true = (features_df[target_col] == class_1).astype(int)

    quality_rows = []

    for feature in feature_cols:
        x0 = df_0[feature].dropna()
        x1 = df_1[feature].dropna()

        _, p_value = mannwhitneyu(x0, x1, alternative="two-sided")

        values = features_df[feature]
        auc = roc_auc_score(y_true, values)
        auc_sep = max(auc, 1 - auc)

        quality_rows.append({
            "feature": feature,
            "p_value": p_value,
            "auc_sep": auc_sep,
        })

    quality_df = pd.DataFrame(quality_rows)
    quality_df["p_adj"] = multipletests(quality_df["p_value"], method="fdr_bh")[1]

    feature_quality = quality_df.set_index("feature")[["p_adj", "auc_sep"]]

    corr = num_df.corr(method="spearman").abs()

    to_drop = set()
    cols = corr.columns.tolist()

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            f1, f2 = cols[i], cols[j]

            if corr.loc[f1, f2] > corr_threshold:
                q1 = feature_quality.loc[f1]
                q2 = feature_quality.loc[f2]

                if (q1["p_adj"] < q2["p_adj"]) or (
                    q1["p_adj"] == q2["p_adj"] and q1["auc_sep"] >= q2["auc_sep"]
                ):
                    to_drop.add(f2)
                else:
                    to_drop.add(f1)

    print(f"Usuwam {len(to_drop)} cech przez wysoką korelację")

    filtered_df = features_df.drop(columns=list(to_drop)).copy()
    print(f"Nowy wymiar danych: {filtered_df.shape}")

    return filtered_df