Credit Card Fraud Detection API

 This repo is a small, opinionated system for training and serving a fraud detection model.
It’s not meant to be clever. It’s meant to be **predictable, inspectable, and easy to reason about**.
 The service trains a model on startup, stores every artifact it produces, and exposes both metrics and predictions through a FastAPI interface. Everything you see in the UI comes from files on disk — no hidden state, no magic.

  What this does

* Trains a **Balanced Random Forest** on an imbalanced credit card dataset
* Splits data into **train / validation / test** and saves them as artifacts
* Evaluates the model using **weighted F1, ROC AUC, confusion matrix**
* Generates **PNG reports** (metrics, heatmaps, ROC curve)
* Renders a **3D PCA visualization** in the browser
* Serves predictions for individual transactions via an API
* Runs fully inside Docker with a persistent output directory

If the container dies, the results don’t. You can inspect everything after the fact.

 Why Balanced Random Forest

  This dataset is extremely skewed. Accuracy is meaningless here.
Balanced Random Forest handles class imbalance at the algorithm level instead of relying on post-hoc threshold tricks. The goal is not leaderboard performance — it’s **stable recall on rare events** without turning the model into a false-positive machine.

Weighted F1 is the primary signal used during evaluation.
<img width="1880" height="833" alt="Screenshot 2025-08-18 094921" src="https://github.com/user-attachments/assets/e3a9523d-8a7c-4aff-8037-d00b037a846b" />


   How the pipeline works

1.Startup

   * Dataset is loaded using a memory-efficient reader
   * Data is split with stratification
   * Artifacts are written to disk immediately

2.Training

   * If a model exists, it’s reused
   * Otherwise a new model is trained and persisted

3.Evaluation

   * Metrics are computed on held-out data
   * All plots and reports are saved as files
   * PCA is used only for visualization, never training

4.Serving

   * HTML dashboard renders directly from stored artifacts
   * Prediction endpoint loads the same saved model

No global state is trusted unless it’s on disk.

 Running it

Docker (recommended)

bash
docker compose up --build

The API will be available at:
http://localhost:8888

Artifacts will be written to the mounted output directory.

  What I cared about while building this

* Reproducibility over raw performance
* Artifacts over logs — everything inspectable later
* Explicit trade-offs instead of silent defaults
* Treating ML as software, not a notebook

This is closer to how models are handled in production than in tutorials.

  What I’d do next with more time

* Replace local disk with object storage
* Add threshold tuning tied to business cost
* Track model versions explicitly
* Add basic data drift checks
* Move training behind a background task

None of those change the core structure — they extend it.

  Final note

This repo is intentionally quiet.
If something is happening, you can find it on disk.

That’s the point.

