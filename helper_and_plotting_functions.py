import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import itertools
import seaborn as sns


def compute_proportions(df):
    """Berechnet den Anteil schlechter Gesundheit innerhalb jeder Altersgruppe.

    Args:
        df (DataFrame): mit den Spalten ['_AGEG5YR', "label", "'INCOME3'"]

    Returns:
        Series: Prozentsatz der Personen mit schlechter Gesundheit pro Altersgruppe.
    """
    age_bins = {
        1: "18-24",
        2: "25-29",
        3: "30-34",
        4: "35-39",
        5: "40-44",
        6: "45-49",
        7: "50-54",
        8: "55-59",
        9: "60-64",
        10: "65-69",
        11: "70-74",
        12: "75-79",
        13: "80+",
        14: "Missing",
    }
    # Kopie des DataFrames erstellen, damit das Original unverändert bleibt
    df_copy = df.copy()
    # Altersgruppen mit .loc zuordnen, um eine SettingWithCopyWarning zu vermeiden
    df_copy.loc[:, "_AGEG5YR"] = df["_AGEG5YR"].map(age_bins)
    # Personen mit schlechtem Gesundheitszustand im kopierten DataFrame filtern
    df_poor_health = df_copy[df_copy["label"] == "Poor Health"]
    # Gesamtzahl der Personen in jeder Altersgruppe im DataFrame berechnen
    total_counts = df_copy.groupby("_AGEG5YR").size()
    # Anzahl der Personen mit schlechter Gesundheit in jeder Altersgruppe in df berechnen
    poor_health_counts = df_poor_health.groupby("_AGEG5YR").size()
    # Anteil schlechter Gesundheit in jeder Altersgruppe berechnen
    proportions = (poor_health_counts / total_counts) * 100
    # Sicherstellen, dass alle Altersgruppen vorkommen (fehlende mit 0 auffüllen)
    proportions = proportions.reindex(age_bins.values(), fill_value=0)
    return proportions


def plot_label_distribution(label_percentages):
    """Stellt die Verteilung des Gesundheitszustands der Teilnehmenden dar.

    Args:
        label_percentages (Series): Prozentsätze für Personen
                                    mit guter vs schlechter Gesundheit.
    """
    colors = ["#252629", "#FF4A11"]

    # Balkendiagramm erstellen
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(
        label_percentages.index, label_percentages, color=colors, edgecolor=colors[0]
    )

    # Prozentsätze über den Balken einfügen
    for bar, percentage in zip(bars, label_percentages):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1,
            f"{percentage:.0f}%",
            ha="center",
            va="bottom",
            fontsize=12,
            color="black",
        )

    # Plot anpassen
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.yaxis.set_ticks_position("none")
    ax.xaxis.set_ticks_position("none")
    ax.xaxis.set_tick_params(width=0)
    ax.yaxis.set_tick_params(width=0)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="grey", linestyle="-", linewidth=0.25, alpha=0.5)
    ax.xaxis.grid(False)

    # Beschriftungen und Titel mit bestimmten Schriftgrößen festlegen
    ax.set_xlabel("Health Status", fontsize=12, weight="bold", color=colors[0])
    ax.set_ylabel("Percentage", fontsize=12, weight="bold", color=colors[0])
    ax.set_title(
        "Health Status Distribution \n", fontsize=14, weight="bold", color=colors[0]
    )

    # Schriftgrößen der Achsenmarkierungen anpassen
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Plot anzeigen
    plt.tight_layout()
    plt.show()


def plot_income_health(df, poverty, ax):
    """Stellt den Anteil der Personen mit schlechter Gesundheit nach Altersgruppe dar,
        für die Untergruppen der Menschen, die in Armut leben oder nicht.
        Haushalte mit einem Jahreseinkommen von weniger als 25.000 $ gelten als arm.

    Args:
        df (DataFrame): mit den Spalten ['_AGEG5YR', "label", "'INCOME3'"]
        poverty (bool): über die Spalte "InCOM3" definiert
        ax (matplotlib ax): legt die Achse des Subplots fest
    """
    # Grenze der y-Achse festlegen
    ax.set_ylim(0, 60)
    # Beschriftungen und Titel hinzufügen
    ax.set_xlabel("Age Group", fontsize=12)
    ax.set_ylabel("Percentage of Poor Health", fontsize=12)
    # Dynamischen Titel je nach Einkommensniveau festlegen
    # Haushalt mit weniger als 25.000 Dollar pro Jahr
    if poverty:
        income_level = [1, 2, 3, 4]
        ax.set_title("In Poverty", fontsize=14, weight="bold")
    else:
        income_level = [5, 6, 7, 8, 9, 10, 11]
        ax.set_title("Not in Poverty", fontsize=14, weight="bold")
    # Personen mit dem angegebenen Einkommensniveau filtern
    df_income = df[df["INCOME3"].isin(income_level)]
    # Anteile für das angegebene Einkommensniveau berechnen
    proportions_income = compute_proportions(df_income)
    # Anteile als Balkendiagramm darstellen
    bars = ax.bar(proportions_income.index, proportions_income.values, color="#FF4A11")
    # Prozentsatz über jedem Balken anzeigen
    for bar in bars:
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 1,
            f"{yval:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    # Beschriftungen der Achsenmarkierungen auf der x-Achse anpassen
    ax.set_xticks(range(len(proportions_income.index)))
    ax.set_xticklabels(proportions_income.index, rotation=45, fontsize=10)
    # Plot für ein aufgeräumteres Aussehen anpassen
    ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.7)
    ax.tick_params(axis="both", which="major", labelsize=10)


def evaluate_model(
    test_predictions,
    test_probs,
    baseline_predictions,
    train_predictions,
    train_probs,
    train_labels,
    test_labels,
    ):
    """Vergleicht ein Machine-Learning-Modell mit der Leistung der Baseline.
    Berechnet Kennzahlen und zeigt die ROC-Kurve."""

    baseline = {}

    baseline["recall"] = recall_score(
        test_labels, baseline_predictions, pos_label="Poor Health"
    )
    baseline["precision"] = precision_score(
        test_labels, baseline_predictions, pos_label="Poor Health"
    )
    baseline["roc"] = roc_auc_score(
        test_labels, np.where(baseline_predictions == "Good Health", 0, 1)
    )

    results = {}

    results["recall"] = recall_score(
        test_labels, test_predictions, pos_label="Poor Health"
    )
    results["precision"] = precision_score(
        test_labels, test_predictions, pos_label="Poor Health"
    )
    results["roc"] = roc_auc_score(test_labels, test_probs)

    train_results = {}
    train_results["recall"] = recall_score(
        train_labels, train_predictions, pos_label="Poor Health"
    )
    train_results["precision"] = precision_score(
        train_labels, train_predictions, pos_label="Poor Health"
    )
    train_results["roc"] = roc_auc_score(train_labels, train_probs)

    for metric in ["recall", "precision", "roc"]:
        print(f"{metric.capitalize()} \n")
        print(
            f"for Baseline: {round(baseline[metric], 2)},"
            f" Model on Test: {round(results[metric], 2)} "
            f" and on Train: {round(train_results[metric], 2)}"
        )
        print("---" * 15)

    # False Positive Rates und True Positive Rates berechnen
    random_fpr, random_tpr, _ = roc_curve(
        test_labels, [1 for _ in range(len(test_labels))], pos_label="Poor Health"
    )
    base_fpr, base_tpr, _ = roc_curve(
        np.where(test_labels == "Good Health", 0, 1),
        np.where(np.array(baseline_predictions) == "Good Health", 0, 1),
    )
    model_fpr, model_tpr, _ = roc_curve(
        test_labels, test_probs, pos_label="Poor Health"
    )

    plt.figure(figsize=(8, 6))
    plt.rcParams["font.size"] = 16

    # Beide Kurven darstellen
    plt.plot(model_fpr, model_tpr, "#FF4A11", label="model")
    plt.plot(base_fpr, base_tpr, "b", label="baseline")
    plt.plot(random_fpr, random_tpr, "#252629", label="random guess")
    plt.legend()
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")


def plot_confusion_matrix(
    cm,
    classes,
    normalize=False,
    title="Confusion Matrix",
    cmap_range=("#F3F5F9", "#FF4A11"),
):
    """
    Diese Funktion gibt die Confusion Matrix aus und stellt sie dar.
    Mit `normalize=True` lässt sich eine Normalisierung anwenden.
    Quelle: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    cmap = LinearSegmentedColormap.from_list(
        "custom_cmap", [cmap_range[0], cmap_range[1]], N=256
    )

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title, size=18)
    plt.colorbar(aspect=4)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size=12)
    plt.yticks(tick_marks, classes, size=12)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0

    # Plot beschriften
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            fontsize=18,
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.grid(None)
    plt.tight_layout()
    plt.ylabel("True label", size=14)
    plt.xlabel("Predicted label", size=14)


def map_race_column(df, race_col="_IMPRACE"):
    """
    Ordnet die Codes der ethnischen Zugehörigkeit in der angegebenen Spalte eines DataFrames beschreibenden Bezeichnungen der ethnischen Zugehörigkeit zu.

    Parameters:
    df (pd.DataFrame): Der DataFrame mit den Codes der ethnischen Zugehörigkeit.
    race_col (str): Der Name der Spalte im DataFrame, die die Codes der ethnischen Zugehörigkeit enthält. Der Standardwert ist '_IMPRACE'.

    Returns:
    pd.DataFrame: Der ursprüngliche DataFrame mit einer zusätzlichen Spalte 'race_description', die die zugeordneten Bezeichnungen der ethnischen Zugehörigkeit enthält.
    """
    RACE_MAPPING = {
        1: "White",
        2: "Black",
        3: "Asian",
        4: "American Indian/\nAlaskan Native",
        5: "Hispanic",
        6: "Other race",
    }
    df["race_description"] = df[race_col].map(RACE_MAPPING)
    return df


def combine_data_metadata_predictions(data, labels, predictions, meta_data):
    """
    Kombiniert Daten, Labels, Vorhersagen und Metadaten zu einem einzigen DataFrame, einschließlich der Bezeichnungen der ethnischen Zugehörigkeit.

    Parameters:
    data (pd.DataFrame): Die ursprünglichen Daten.
    labels (pd.Series or np.ndarray): Die tatsächlichen Labels zu den Daten.
    predictions (pd.Series or np.ndarray): Die Vorhersagen eines Modells.
    meta_data (pd.DataFrame): Zusätzliche Metadaten, die mit den Daten kombiniert werden.

    Returns:
    pd.DataFrame: Ein DataFrame mit den ursprünglichen Daten, den tatsächlichen Labels, den Vorhersagen, den Metadaten und den Bezeichnungen der ethnischen Zugehörigkeit.
    """
    data_with_labels_predictions = data.copy()
    data_with_labels_predictions["predictions"] = np.where(
        predictions == "Good Health", 0, 1
    )
    data_with_labels_predictions["true_labels"] = np.where(
        labels == "Good Health", 0, 1
    )
    meta_data_analysis = pd.merge(
        data_with_labels_predictions, meta_data, left_index=True, right_index=True
    )
    meta_data_analysis = map_race_column(meta_data_analysis)
    return meta_data_analysis


def plot_bar_with_labels(data, x, y, title, xlabel, ylabel, ax, color="#252629"):
    """
    Stellt ein Balkendiagramm mit Beschriftungen über jedem Balken dar.

    Parameters:
    data (pd.DataFrame): Die Eingabedaten mit den x- und y-Werten für das Balkendiagramm.
    x (str): Der Spaltenname in `data`, dessen Werte auf der x-Achse stehen.
    y (str): Der Spaltenname in `data`, dessen Werte auf der y-Achse stehen.
    title (str): Der Titel des Plots.
    xlabel (str): Die Beschriftung der x-Achse.
    ylabel (str): Die Beschriftung der y-Achse.
    ax (matplotlib.axes.Axes): Das Axes-Objekt, in das der Plot gezeichnet wird.
    color (str): Die Farbe der Balken. Der Standardwert ist '#252629'.

    Returns:
    None
    """
    if data.empty:
        print("The input data is empty. Cannot create a plot.")
        return

    sns.barplot(data=data, x=x, y=y, color=color, edgecolor=".6", ax=ax)
    for index, row in data.iterrows():
        label = f"{row[y]:.2f}" if isinstance(row[y], float) else f"{row[y]}"
        ax.text(index, row[y] + 0.02, label, color="black", ha="center", fontsize=12)
    ax.set_title(title, fontsize=16, fontweight="bold", color="#333333")
    ax.set_xlabel(xlabel, fontsize=14, color="#333333")
    ax.set_ylabel(ylabel, fontsize=14, color="#333333")
    ax.tick_params(width=0.5, color="#333333")
    ax.grid(axis="y", color="gray", linestyle="-", linewidth=0.5, alpha=0.7)
    ax.set_xticklabels(
        ax.get_xticklabels(), rotation=45, ha="right", fontsize=12, color="#333333"
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.spines["bottom"].set_color("#333333")


def plot_roc_auc_by_race(
    df,
    true_label_col="true_labels",
    score_col="predictions",
    race_col="race_description",
):
    """
    Stellt die ROC-AUC-Scores nach Kategorien der ethnischen Zugehörigkeit dar.

    Parameters:
    df (pd.DataFrame): Der DataFrame mit den Daten samt Vorhersagen und tatsächlichen Labels.
    true_label_col (str): Der Spaltenname in `df` mit den tatsächlichen Labels. Der Standardwert ist 'true_labels'.
    score_col (str): Der Spaltenname in `df` mit den vorhergesagten Scores. Der Standardwert ist 'predictions'.
    race_col (str): Der Spaltenname in `df` mit den Kategorien der ethnischen Zugehörigkeit. Der Standardwert ist "race_description".

    Returns:
    None
    """
    race_roc_auc = (
        df.groupby(race_col)
        .apply(lambda x: roc_auc_score(x[true_label_col], x[score_col]))
        .reset_index()
    )
    race_roc_auc.columns = ["Race", "ROC AUC"]
    race_roc_auc = race_roc_auc.sort_values(by="ROC AUC", ascending=False).reset_index(
        drop=True
    )

    plt.figure(figsize=(7, 5))
    sns.set_style("whitegrid")
    sns.set_context("notebook")

    ax = plt.gca()
    plot_bar_with_labels(
        race_roc_auc,
        "Race",
        "ROC AUC",
        "ROC AUC by Race Category",
        "Race",
        "ROC AUC",
        ax,
    )

    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()


def plot_race_frequency(train_df, test_df, race_col="race_description"):
    """
    Stellt die Häufigkeit der Kategorien der ethnischen Zugehörigkeit im Trainings- und im Testdatensatz dar.

    Parameters:
    train_df (pd.DataFrame): Der Trainingsdatensatz mit Angaben zur Kategorie der ethnischen Zugehörigkeit.
    test_df (pd.DataFrame): Der Testdatensatz mit Angaben zur Kategorie der ethnischen Zugehörigkeit.
    race_col (str): Der Spaltenname in `train_df` und `test_df` mit den Kategorien der ethnischen Zugehörigkeit. Der Standardwert ist "race_description".

    Returns:
    None
    """
    train_race_freq = (
        train_df[race_col]
        .value_counts()
        .reset_index()
        .rename(columns={"index": "Race", race_col: "Frequency"})
    )
    test_race_freq = (
        test_df[race_col]
        .value_counts()
        .reset_index()
        .rename(columns={"index": "Race", race_col: "Frequency"})
    )
    train_race_freq = train_race_freq.sort_values(
        by="Frequency", ascending=False
    ).reset_index(drop=True)
    test_race_freq = test_race_freq.sort_values(
        by="Frequency", ascending=False
    ).reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    sns.set_style("whitegrid")
    sns.set_context("notebook")

    plot_bar_with_labels(
        train_race_freq,
        "Race",
        "Frequency",
        "Frequency of Race Categories (Train)",
        "Race",
        "Frequency",
        axes[0],
    )
    plot_bar_with_labels(
        test_race_freq,
        "Race",
        "Frequency",
        "Frequency of Race Categories (Test)",
        "Race",
        "Frequency",
        axes[1],
    )

    plt.tight_layout()
    plt.show()


def meta_data_race_analysis(
    test_data,
    test_labels,
    test_predictions,
    train_data,
    train_labels,
    train_predictions,
    meta_data,
):
    """
    Führt eine Fehleranalyse des Modells mit den Metadaten aus dem Test- und dem Trainingsdatensatz durch.

    Parameters:
    test_data (pd.DataFrame): Der zu analysierende Testdatensatz.
    test_labels (pd.Series or np.ndarray): Die tatsächlichen Labels zu den Testdaten.
    test_predictions (pd.Series or np.ndarray): Die Vorhersagen für die Testdaten.
    train_data (pd.DataFrame): Der zu analysierende Trainingsdatensatz.
    train_labels (pd.Series or np.ndarray): Die tatsächlichen Labels zu den Trainingsdaten.
    train_predictions (pd.Series or np.ndarray): Die Vorhersagen für die Trainingsdaten.
    meta_data (pd.DataFrame): Zusätzliche Metadaten, die mit den Datensätzen kombiniert werden.

    Returns:
    None
    """
    test_meta_data_analysis = combine_data_metadata_predictions(
        test_data, test_labels, test_predictions, meta_data
    )
    train_meta_data_analysis = combine_data_metadata_predictions(
        train_data, train_labels, train_predictions, meta_data
    )

    plot_roc_auc_by_race(test_meta_data_analysis)
    plot_race_frequency(train_meta_data_analysis, test_meta_data_analysis)
