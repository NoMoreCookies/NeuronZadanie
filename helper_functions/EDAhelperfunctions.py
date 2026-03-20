# IMPORTY
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks
from scipy.stats import mannwhitneyu
import statsmodels.stats.multitest as smm
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

# FUNKCJA DO RYSOWANIA WYKRESÓW
def visualize_signals(
    df: pd.DataFrame,
    id_value: str,
    fs: int,
    seconds: int = 120,
    offset: float = 500,
) -> None:
    """Wizualizuje fragment wielokanałowego sygnału dla wybranego ID.

    Args:
        df: Ramka danych zawierająca kolumnę `ID` oraz kanały sygnałowe.
        id_value: Identyfikator obserwacji w kolumnie `ID`.
        channels: Lista nazw kanałów do narysowania.
        fs: Częstotliwość próbkowania sygnału.
        seconds: Długość fragmentu sygnału do wyświetlenia w sekundach.
        offset: Stałe przesunięcie pionowe pomiędzy kolejnymi kanałami.

    Returns:
        None.

    Raises:
        ValueError: Gdy nie znaleziono danych dla podanego ID lub `seconds <= 0`.
        KeyError: Gdy brakuje wymaganych kanałów w ramce danych.
    """
    
    numeric_cols = df.columns[:-2]

    channels = list(numeric_cols)

    signal_df = (
        df.loc[df["ID"] == id_value, channels]
        .reset_index(drop=True)
    )

    if signal_df.empty:
        raise ValueError(f"Nie znaleziono danych dla ID={id_value}.")

    segment_df = signal_df.iloc[: fs * seconds]
    t = np.arange(len(segment_df)) / fs

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, ch in enumerate(channels):
        sig = segment_df[ch].to_numpy()
        ax.plot(t, sig + i * offset, lw=0.2, color=f"C{i % 10}")

    ax.set_yticks(np.arange(len(channels)) * offset)
    ax.set_yticklabels(channels)
    ax.set_xlabel("Czas [s]")
    ax.set_ylabel("Kanały (offset)")
    ax.set_title(f"ID: {id_value}")
    ax.grid(axis="x", alpha=0.2)

    fig.tight_layout()
    plt.show()

# FILTR 1
def filtr_1(
    df: pd.DataFrame,
    channels: list[str],
    low_var_q: float = 0.20,
) -> pd.DataFrame:
    """Usuwa kanały o niskiej wariancji na podstawie mediany wariancji po ID.

    Dla każdego kanału funkcja liczy wariancję osobno w każdej grupie `ID`,
    następnie wyznacza medianę tych wariancji. Kanały, których mediana
    wariancji jest mniejsza od kwantyla `low_var_q`, są odrzucane.

    Args:
        df: Ramka danych zawierająca kolumnę `ID` oraz kanały sygnałowe.
        channels: Lista nazw kanałów do rozważenia.
        low_var_q: Kwantyl wyznaczający próg odrzucenia kanałów
            o niskiej wariancji. Wartość powinna należeć do przedziału [0, 1].

    Returns:
        DataFrame zawierający kolumny bazowe (`ID`, `Class`, jeśli istnieją)
        oraz kanały o medianie wariancji większej lub równej wyznaczonemu progowi.

    Raises:
        KeyError: Gdy w `df` brakuje kolumny `ID`.
        ValueError: Gdy `low_var_q` nie należy do przedziału [0, 1]
            albo gdy nie znaleziono żadnego poprawnego kanału.
    """

    if not 0 <= low_var_q <= 1:
        raise ValueError("Parametr 'low_var_q' musi należeć do przedziału [0, 1].")

    channels = [c for c in channels if c in df.columns]
    
    if not channels:
        raise ValueError("Nie znaleziono żadnych kanałów z listy 'channels' w df.")

    # tutaj ddof = 0, bo mamy całą populację
    chan_var = df.groupby("ID")[channels].var(ddof=0).quantile(0.7)

    var_cut = chan_var.quantile(low_var_q)

    keep_set = set(chan_var[chan_var >= var_cut].index)

    keep = [c for c in channels if c in keep_set]

    base_cols = [c for c in ["ID", "Class"] if c in df.columns]
    return df[base_cols + keep]

# FILTR 2
def filtr_2(
    df: pd.DataFrame,
    channels: list[str] | None = None,
    high_corr: float = 0.7,
    std_thresh: float = 0.2
) -> pd.DataFrame:
    """Usuwa redundantne kanały na podstawie korelacji liczonej osobno dla każdego ID.

    Funkcja najpierw wyznacza kolejność kanałów metodą zachłanną na podstawie
    kwantylu 0.7 wariancji w grupach `ID` (od największej do najmniejszej). Następnie
    dla każdego `ID` liczy macierz korelacji między kanałami, a potem dla każdej
    pary kanałów oblicza średnią korelację i odchylenie standardowe po wszystkich
    identyfikatorach. Jeśli para kanałów ma wysoką średnią korelację oraz niskie
    odchylenie standardowe, uznawana jest za redundantną. W takim przypadku
    zachowywany jest pierwszy kanał w kolejności greedy, a pozostałe są usuwane.

    Args:
        df: Ramka danych zawierająca kolumnę `ID` oraz kanały sygnałowe.
        channels: Lista kanałów do rozważenia. Jeśli `None`, funkcja używa
            wszystkich kolumn poza `ID` i `Class`.
        high_corr: Minimalna wartość bezwzględnej średniej korelacji, od której
            para kanałów jest uznawana za silnie skorelowaną.
        std_thresh: Maksymalne dopuszczalne odchylenie standardowe korelacji
            po `ID`, poniżej którego zależność uznawana jest za stabilną.


    Returns:
        DataFrame zawierający kolumny bazowe (`ID`, `Class`, jeśli istnieją)
        oraz kanały wybrane po usunięciu redundancji.


    """
    
    channels = [c for c in df.columns if c not in ["ID", "Class"]]

    # Kolejność greedy: od największej mediany wariancji po ID.
    chan_var = df.groupby("ID")[channels].var(ddof=0).quantile(0.7).sort_values(ascending=False)
    chan_order = chan_var.index.tolist()

    # Korelacja osobno dla każdego ID.
    corr_obj = df.groupby("ID")[channels].apply(lambda x: x.corr())
    corr_per_id = corr_obj.stack()

    # Średnia i odchylenie standardowe korelacji dla każdej pary kanałów po ID.
    mean_corr = corr_per_id.groupby(level=[1, 2]).mean()
    std_corr = corr_per_id.groupby(level=[1, 2]).std()

    mean_corr_mat = mean_corr.unstack().reindex(index=channels, columns=channels)
    std_corr_mat = std_corr.unstack().reindex(index=channels, columns=channels)

    # Maska stabilnej wysokiej korelacji.
    stable_mask = (mean_corr_mat.abs() >= high_corr) & (std_corr_mat <= std_thresh)
    stable_mask = stable_mask.fillna(False)

    # Ominięcie przekątnej
    common = stable_mask.index.intersection(stable_mask.columns)
    for ch in common:
        stable_mask.loc[ch, ch] = False

    removed: set[str] = set()
    kept: list[str] = []

    for ch in chan_order:
        if ch in removed:
            continue

        kept.append(ch)
        redundant = stable_mask.columns[stable_mask.loc[ch]].tolist()
        removed.update(redundant)

    kept = kept[: len(kept)]

    base_cols = [c for c in ["ID", "Class"] if c in df.columns]
    return df[base_cols + kept]

# DODANIE WSKAŹNIKÓW
def build_features(df, fs=128):
    """Buduje wektor cech dla każdej sesji (`ID`) na podstawie sygnałów kanałów."""


    channels = df.columns[2:]

    features = []

    for id_val, session_data in df.groupby("ID"):
        class_val = session_data["Class"].iloc[0]
        duration = len(session_data) / fs

        row = {"ID": id_val, "Class": class_val, "duration": duration}

        for ch in channels:
            series = session_data[ch].to_numpy()

            row[f"{ch}_mean"] = np.mean(series)
            row[f"{ch}_std"] = np.std(series)
            row[f"{ch}_skew"] = pd.Series(series).skew()
            row[f"{ch}_kurt"] = pd.Series(series).kurtosis()
            row[f"{ch}_rms"] = np.sqrt(np.mean(series ** 2))
            row[f"{ch}_energy"] = np.sum(series ** 2)

            peaks, _ = find_peaks(series)
            row[f"{ch}_peaks"] = len(peaks)

            fft = np.fft.fft(series)
            power = np.abs(fft) ** 2
            freqs = np.fft.fftfreq(len(series), d=1 / fs)

            total_power = np.sum(power)
            if total_power == 0:
                total_power = 1e-10

            bands = {
                "delta": (0, 4),
                "theta": (4, 8),
                "alpha": (8, 12),
                "beta": (12, 30),
            }

            for band, (low, high) in bands.items():
                mask = (freqs >= low) & (freqs < high)
                row[f"{ch}_{band}_power"] = np.sum(power[mask]) / total_power

            power_norm = power / total_power
            row[f"{ch}_spectral_entropy"] = -np.sum(
                power_norm * np.log2(power_norm + 1e-10)
            )

        features.append(row)

    return pd.DataFrame(features)

# TABELA PRZYDATNYCH WSKAŹNIKÓW
def rank_features_mannwhitney(
    features_df: pd.DataFrame,
    class_col: str = "Class",
    id_col: str = "ID",
    positive_class: str = "ADHD",
    negative_class: str = "Control",
    correction_method: str = "fdr_bh",
) -> pd.DataFrame:
    """Porządkuje cechy numeryczne na podstawie testu Manna-Whitneya i AUC.

    Dla każdej cechy funkcja:
    1. Dzieli dane na dwie grupy na podstawie etykiety klasy.
    2. Liczy p-value z testu Manna-Whitneya.
    3. Liczy rank-biserial correlation jako miarę wielkości efektu.
    4. Liczy ROC AUC dla pojedynczej cechy.
    5. Wykonuje korektę wielokrotnego testowania dla p-value.

    Zwrócona tabela jest sortowana najpierw po skorygowanym p-value,
    a następnie po sile separacji cechy (AUC).

    Args:
        features_df: DataFrame wejściowy zawierający cechy oraz kolumny
            pomocnicze, np. identyfikator i etykietę klasy.
        class_col: Nazwa kolumny z etykietą klasy.
        id_col: Nazwa kolumny identyfikatora, która ma zostać pominięta
            w analizie.
        positive_class: Klasa traktowana jako pozytywna przy liczeniu AUC
            oraz interpretacji znaku effect size.
        negative_class: Klasa odniesienia (negatywna) przy porównaniu.
        correction_method: Metoda korekty wielokrotnych testów używana
            w funkcji `multipletests`, np. "fdr_bh" lub "bonferroni".

    Returns:
        DataFrame z jednym wierszem na każdą cechę i kolumnami:
        - feature
        - p_value
        - effect_size
        - auc_sep
        - p_adj

        Tabela jest posortowana według:
        1. `p_adj` rosnąco,
        2. `auc_sep` malejąco.

    Raises:
        ValueError: Gdy w danych brakuje wymaganych kolumn.
        ValueError: Gdy jedna z porównywanych klas nie występuje
            w kolumnie z etykietami.

    Example:
        >>> results_df = rank_features_mannwhitney(features_df)
        >>> print(results_df.head())
    """

    classes_present = set(features_df[class_col].unique())

    positive_data = features_df[features_df[class_col] == positive_class]

    negative_data = features_df[features_df[class_col] == negative_class]

    results = []

    feature_cols = features_df.columns.difference([id_col, class_col])

    for col in feature_cols:

        # Pobranie wartości cechy dla obu klas.
        x = positive_data[col]
        y = negative_data[col]

        # Test Manna-Whitneya dla różnicy między grupami.
        u_stat, p_val = mannwhitneyu(x, y, alternative="two-sided")

        # Liczebności grup potrzebne do wyznaczenia effect size.
        n1 = len(x)
        n2 = len(y)

        # Rank-biserial correlation:
        # wartość dodatnia oznacza zwykle większe wartości w positive_class.
        effect_size = 2 * u_stat / (n1 * n2) - 1

        # AUC dla pojedynczej cechy.
        tmp = features_df[[class_col, col]].dropna().copy()
        tmp["target"] = (tmp[class_col] == positive_class).astype(int)

        auc = roc_auc_score(tmp["target"], tmp[col])

        # Siła separacji niezależna od kierunku.
        auc_sep = max(auc, 1 - auc)

        results.append(
            {
                "feature": col,
                "p_value": p_val,
                "effect_size": effect_size,
                "auc_sep": auc_sep,
            }
        )

    results_df = pd.DataFrame(results)

    # Korekta wielokrotnego testowania.
    results_df["p_adj"] = multipletests(
        results_df["p_value"],
        method=correction_method,
    )[1]

    # Sortowanie: najpierw istotność, potem siła separacji.
    results_df = results_df.sort_values(
        by=["p_adj", "auc_sep"],
        ascending=[True, False],
    ).reset_index(drop=True)

    return results_df

# PRZEFILTROWANE WSKAŹNIKI
def filter_features_by_statistics(
        
    features_df: pd.DataFrame,
    results_df: pd.DataFrame,
    id_col: str = "ID",
    class_col: str = "Class",
    p_adj_threshold: float = 0.10,
    effect_size_threshold: float = 0.20,
    auc_threshold: float = 0.60
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """Filtruje cechy na podstawie istotności, wielkości efektu i AUC.


    Funkcja:
    1. Wykonuje korektę wielokrotnego testowania dla kolumny `p_value`,
       jeśli kolumna `p_adj` nie istnieje.
    2. Sortuje tabelę wyników po `p_adj` oraz `auc_sep`.
    3. Wybiera cechy spełniające zadane progi dla:
       - skorygowanego p-value,
       - bezwzględnej wartości effect size,
       - AUC separacji.
    4. Zwraca przefiltrowany zbiór danych zawierający tylko wybrane cechy.

    Args:
        features_df: Oryginalny DataFrame z kolumnami identyfikatora,
            klasy oraz cechami.
        results_df: DataFrame z wynikami rankingu cech. Powinien zawierać
            co najmniej kolumny:
            - `feature`
            - `p_value`
            - `effect_size`
            - `auc_sep`
            Opcjonalnie może już zawierać kolumnę `p_adj`.
        id_col: Nazwa kolumny identyfikatora.
        class_col: Nazwa kolumny klasy.
        p_adj_threshold: Maksymalna dopuszczalna wartość skorygowanego
            p-value dla pozostawienia cechy.
        effect_size_threshold: Minimalna bezwzględna wartość effect size
            dla pozostawienia cechy.
        auc_threshold: Minimalna wartość `auc_sep` dla pozostawienia cechy.
        correction_method: Metoda korekty wielokrotnych testów używana
            w funkcji `multipletests`, np. "fdr_bh" lub "bonferroni".

    Returns:
        Krotka zawierająca:
        1. `features_df_filtered` - DataFrame z kolumnami `id_col`,
           `class_col` oraz wybranymi cechami,
        2. `keep_df` - tabela wyników dla pozostawionych cech,
        3. `keep_feats` - lista nazw pozostawionych cech,
        4. `drop_feats` - lista nazw usuniętych cech.

    Raises:
        ValueError: Gdy brakuje wymaganych kolumn w `results_df`.
        ValueError: Gdy brakuje kolumn `id_col` lub `class_col` w `features_df`.

    Example:
        >>> features_df_filtered, keep_df, keep_feats, drop_feats = (
        ...     filter_features_by_statistics(features_df, results_df)
        ... )
        >>> print(keep_feats[:10])
    """

    results_df = results_df.copy()

    # Sortowanie: najpierw po istotności, potem po sile separacji.
    results_df = results_df.sort_values(
        by=["p_adj", "auc_sep"],
        ascending=[True, False],
    ).reset_index(drop=True)

    # Wybór cech spełniających progi.
    keep_df = results_df[
        (results_df["p_adj"] <= p_adj_threshold)
        & (results_df["effect_size"].abs() >= effect_size_threshold)
        & (results_df["auc_sep"] >= auc_threshold)
    ].copy()

    drop_df = results_df.drop(keep_df.index).copy()

    keep_feats = keep_df["feature"].tolist()
    drop_feats = drop_df["feature"].tolist()

    features_df_filtered = features_df[[id_col, class_col] + keep_feats].copy()

    return features_df_filtered, keep_df, keep_feats, drop_feats

# RYSOWANIE WYKRESÓW
def draw_plot(df: pd.DataFrame, channel: str) -> None:
    """Rysuje dwa wykresy porównujące rozkład cechy między klasami.

    Funkcja tworzy figurę z dwoma panelami:
    1. histogramem z krzywą KDE,
    2. boxplotem.

    Oba wykresy porównują rozkład wskazanej cechy (`channel`)
    pomiędzy klasami zapisanymi w kolumnie `Class`.
    Dodatkowo funkcja usuwa lokalne legendy z osi i tworzy
    jedną wspólną legendę dla całej figury.

    Args:
        df: DataFrame zawierający co najmniej kolumnę `Class`
            oraz kolumnę wskazaną przez parametr `channel`.
        channel: Nazwa cechy, która ma zostać zwizualizowana.

    Returns:
        None. Funkcja wyświetla wykres i nic nie zwraca.

    Raises:
        KeyError: Gdy w DataFrame brakuje kolumny `Class`
            lub kolumny wskazanej w parametrze `channel`.

    Example:
        >>> draw_plot(features_df, "Fp2_alpha_power")
    """
    # Tworzenie figury z dwoma panelami obok siebie.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Ustalenie wspólnej palety kolorów dla obu klas.
    palette = sns.color_palette("deep", 2)

    # Histogram z KDE dla badanej cechy.
    sns.histplot(
        data=df,
        x=channel,
        hue="Class",
        kde=True,
        stat="density",
        common_norm=False,
        ax=axes[0],
        palette=palette,
    )

    # Boxplot dla tej samej cechy.
    sns.boxplot(
        data=df,
        x=channel,
        hue="Class",
        ax=axes[1],
        palette=palette,
    )

    # Usunięcie lokalnych legend z obu osi.
    if axes[0].get_legend() is not None:
        axes[0].get_legend().remove()
    if axes[1].get_legend() is not None:
        axes[1].get_legend().remove()

    # Ręczne utworzenie wspólnej legendy dla całej figury.
    handles = [
        Patch(facecolor=palette[0], label="Control"),
        Patch(facecolor=palette[1], label="ADHD"),
    ]

    fig.legend(handles=handles, title="Klasa", loc="lower center", ncol=2)

    # Dodanie wspólnego tytułu dla całej figury.
    fig.suptitle(f"Porównanie rozkładu cechy {channel}", fontsize=14)

    # Dopasowanie układu z miejscem na tytuł i legendę.
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    plt.show()

# USUNIĘCIE WYSOKO SKORELOWANYCH CECH
def drop_highly_correlated_features(
    features_df: pd.DataFrame,
    corr_threshold: float = 0.75,
    id_col: str = "ID",
    target_col: str = "Class",
) -> pd.DataFrame:
    """Usuwa silnie skorelowane cechy, zachowując lepszą z każdej pary.

    Dla każdej cechy funkcja wewnętrznie liczy:
    - p-value z testu Mann-Whitneya między dwiema klasami,
    - skorygowane p-value (Benjamini-Hochberg),
    - AUC separacji dla pojedynczej cechy.

    Jeśli dwie cechy są silnie skorelowane (|rho Spearmana| > corr_threshold),
    funkcja zostawia tę o niższym p_adj, a przy remisie tę o wyższym auc_sep.

    Args:
        features_df: DataFrame zawierający kolumny z cechami oraz kolumny
            identyfikacyjne/docelowe.
        corr_threshold: Próg korelacji bezwzględnej Spearmana.
        id_col: Nazwa kolumny identyfikatora.
        target_col: Nazwa kolumny zmiennej docelowej.

    Returns:
        DataFrame po usunięciu nadmiarowych, silnie skorelowanych cech.

    """

    # Wybór samych cech numerycznych
    num_df = features_df.drop(columns=[id_col, target_col]).copy()


    classes = features_df[target_col].dropna().unique()

    class_0, class_1 = classes

    # Dane rozdzielone wg klas
    df_0 = features_df[features_df[target_col] == class_0]
    df_1 = features_df[features_df[target_col] == class_1]

    # Liczenie jakości cech
    quality_rows = []

    # Zamiana klas na 0/1 do AUC
    y_true = (features_df[target_col] == class_1).astype(int)

    for feature in num_df.columns:
        x0 = df_0[feature].dropna()
        x1 = df_1[feature].dropna()

        # p-value z testu Mann-Whitneya
        _, p_value = mannwhitneyu(x0, x1, alternative="two-sided")

        # AUC dla pojedynczej cechy
        values = features_df[feature]
        auc = roc_auc_score(y_true, values)

        # Jeżeli AUC < 0.5, odwracamy interpretację
        auc_sep = max(auc, 1 - auc)

        quality_rows.append({
            "feature": feature,
            "p_value": p_value,
            "auc_sep": auc_sep,
        })

    quality_df = pd.DataFrame(quality_rows)

    # Korekta wielokrotnych testów
    quality_df["p_adj"] = multipletests(quality_df["p_value"], method="fdr_bh")[1]

    # Ranking jakości cech
    feature_quality = quality_df.set_index("feature")[["p_adj", "auc_sep"]]

    # Macierz korelacji
    corr = num_df.corr(method="spearman").abs()

    to_drop = set()
    cols = corr.columns.tolist()

    # Porównywanie par cech
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            f1, f2 = cols[i], cols[j]

            if corr.loc[f1, f2] > corr_threshold:
                q1 = feature_quality.loc[f1]
                q2 = feature_quality.loc[f2]

                # Zostaw lepszą cechę
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