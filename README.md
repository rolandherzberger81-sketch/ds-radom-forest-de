# Random Forest

Eine praxisnahe Einführung in Random Forests für die Klassifikation. Du erstellst einen Random Forest mit scikit-learn, siehst, inwiefern er einem einzelnen Entscheidungsbaum überlegen ist, evaluierst und tunst ihn und wendest dann den vollständigen Workflow selbst an, um anhand von Lending-Club-Daten Kreditausfälle vorherzusagen.

## Lernziele

Am Ende dieses Repositorys kannst du:

- einen Random-Forest-Klassifikator mit scikit-learn erstellen und trainieren.
- beschreiben, inwiefern ein Random Forest einem einzelnen Entscheidungsbaum überlegen ist.
- echte Daten für baumbasierte Modelle vorbereiten.
- ein Klassifikationsmodell evaluieren und tunen.
- Modellergebnisse und Feature Importances interpretieren.
- den vollständigen Workflow auf ein neues Vorhersageproblem anwenden.

## Lernpfad

Bearbeite die Notebooks der Reihe nach:

| Datei / Ordner | Beschreibung |
|---|---|
| [**1 - Random-Forest-Codealong**](1_random_forest_codealong.ipynb) | Einen Random Forest mit scikit-learn implementieren und seine Vorhersagen auswerten. |
| [**2 - Random-Forest-Tutorial**](2_random_forest_tutorial.ipynb) | Einen Entscheidungsbaum in einem vollständigen Workflow mit einem Random Forest vergleichen: die Daten bereinigen, fehlende Werte imputieren, eine Baseline trainieren und dann mit Randomized Search tunen. |
| [**3 - Random-Forest-Aufgabe**](3_random_forest_exercise.ipynb) | Den Workflow selbst auf die Kreditdaten von Lending Club anwenden und dabei sowohl einen Entscheidungsbaum als auch einen Random Forest erstellen. |

### Weitere Ordner und Dateien

| Datei / Ordner | Beschreibung |
|---|---|
| [**Hilfs- und Plotfunktionen**](helper_and_plotting_functions.py) | Gemeinsam genutzte Funktionen zum Evaluieren von Modellen und zum Darstellen der Ergebnisse. |
| [**Solutions**](solutions/) | Musterlösungen. |
| [**data.zip**](data.zip) | Die Datensätze, gebündelt als ZIP-Datei (entpack sie beim Setup). |
| [**pyproject.toml**](pyproject.toml) | Projektkonfiguration und Abhängigkeiten. |
| [**uv.lock**](uv.lock) | Lock-Datei für die Abhängigkeiten. |

## Setup

> [!NOTE]
> In allen Schritten ist Text in spitzen Klammern wie `<repo-name>` ein **Platzhalter**. Ersetz ihn samt den Klammern `< >` durch deinen eigenen Wert. Aus `cd <repo-name>` wird zum Beispiel `cd ds-random-forest`.

### 1. Repository aus dem Template erstellen

Klick auf GitHub auf **Use this template**.

Beim Erstellen des Repositorys:

- Leg dich selbst als **Owner** fest
- Wähl einen Namen für das Repository
- Deaktiviere **Include all branches**
- Klick auf **Create repository**

> [!IMPORTANT]
> Wenn du zu zweit oder in einer Gruppe arbeitest, erledigt nur **eine Person** diesen Schritt.

---

### 2. Collaborators hinzufügen (nur bei Paar- oder Gruppenarbeit)

Wenn du mit Teammitgliedern arbeitest:

1. Öffne das Repository auf GitHub
2. Geh zu **Settings → Collaborators**
3. Füg deine Teammitglieder als Collaborators hinzu
4. Teil den Link zum Repository mit deinem Team

Die Teammitglieder sollten die Einladung annehmen, bevor sie weitermachen.

---

### 3. Repository klonen

Kopiere die SSH-URL über den Button **Code** auf GitHub und führ dann Folgendes aus:

```bash
git clone <copied-ssh-url>
```

Die kopierte SSH-URL hat die Form `git@github.com:<your-username>/<repo-name>.git`.

---

### 4. In den Projektordner wechseln und Abhängigkeiten installieren

Das installiert alle Abhängigkeiten und erstellt eine virtuelle Umgebung in `.venv/`.

```bash
cd <repo-name>
uv sync
```

---

### 5. Daten entpacken

Die Datensätze sind in `data.zip` gebündelt. Entpack sie in einen Ordner `data/`, bevor du die Notebooks ausführst. Dieser Befehl verwendet die Umgebung aus `uv sync`:

```bash
uv run python -c "import zipfile; zipfile.ZipFile('data.zip').extractall()"
```

---

### 6. Notebooks öffnen

> [!NOTE]
> Achte darauf, VS Code im Stammverzeichnis des Projekts zu öffnen, damit es die von `uv sync` erstellte Umgebung automatisch erkennt.

Starte VS Code im Stammverzeichnis des Projekts:

```bash
code .
```

Öffne dann ein Notebook und wähl die von `uv sync` erstellte Python-Umgebung als Kernel aus.

## Quellen & weiterführende Literatur

- [**Scikit-learn: Ensemble methods**](https://scikit-learn.org/stable/modules/ensemble.html): Wie Wälder aus randomisierten Bäumen, Bagging und Boosting funktionieren.
- [**Scikit-learn: RandomForestClassifier**](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html): Die API-Referenz für den hier verwendeten Klassifikator.
- [**Scikit-learn: Feature importances with a forest of trees**](https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html): Unreinheitsbasierte und permutationsbasierte Feature Importances eines Random Forests lesen.
- [**Scikit-learn: Tuning hyperparameters**](https://scikit-learn.org/stable/modules/grid_search.html): Grid Search, Randomized Search und Successive Halving.
