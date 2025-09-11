from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import plotly.express as px
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import traceback
from datetime import datetime
from sklearn.metrics import roc_curve, auc
import pickle

BASE_OUTPUT = os.environ.get("OUTPUT_FOLDER", "staj_data")
DATE_FOLDER = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_FOLDER = os.path.join(BASE_OUTPUT, DATE_FOLDER)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

LOG_PATH = os.path.join(OUTPUT_FOLDER, "logs.txt")
REPORT_TXT_PATH = os.path.join(OUTPUT_FOLDER, "report.txt")
MODEL_PATH = os.path.join(OUTPUT_FOLDER, "model.pkl")

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Eğitim + Tahmin + Raporlama (PNG/TXT) + 3D PCA görselleştirme",
    version="1.0.0"
)

training_logs = ""
cm = None
report = ""
f1 = 0.0
plot_html = ""
model_cache = None

DATA_PATH = "data/creditcard.csv"

test_data_cache = None

def log(msg: str):
    global training_logs
    training_logs += msg + "\n"
    print(msg, flush=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def save_png_confusion_matrix(cm_local, f1_local):
    plt.figure(figsize=(6,5))
    sns.heatmap(cm_local, annot=True, fmt="d", cmap="viridis")
    plt.title(f"Confusion Matrix (F1={f1_local:.2f})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, "confusion_matrix.png"))
    plt.close()

def save_png_f1(f1_local):
    plt.figure(figsize=(4,2))
    plt.axis('off')
    plt.table(cellText=[[f1_local]], colLabels=["Weighted F1 Score"], loc='center')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, "f1_score.png"))
    plt.close()

def save_png_heatmap(X):
    plt.figure(figsize=(12, 8))
    corr = X.corr()
    sns.heatmap(corr, cmap="viridis", annot=False)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, "heatmap.png"))
    plt.close()

def save_png_class_report(report_dict):
    rows = [
        [k, v['precision'], v['recall'], v['f1-score']]
        for k, v in report_dict.items() if isinstance(v, dict)
    ]
    plt.figure(figsize=(10, 4))
    plt.axis('off')
    plt.table(cellText=rows, colLabels=["Class","Precision","Recall","F1"], loc='center')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, "classification_report.png"))
    plt.close()

def save_txt_report(f1_local, cm_local, report_str):
    with open(REPORT_TXT_PATH, "w", encoding="utf-8") as f:
        f.write("=== Model Report ===\n")
        f.write(f"F1 Score (weighted): {f1_local:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm_local) + "\n\n")
        f.write("Classification Report:\n")
        f.write(report_str + "\n")

def make_pca_plot_html(X, y):
    if len(X) > 50000:
        X_sample = X.sample(50000, random_state=42)
        y_sample = y.loc[X_sample.index]
    else:
        X_sample = X
        y_sample = y
    pca = PCA(n_components=3, random_state=42)
    pca_result = pca.fit_transform(X_sample)
    df_vis = pd.DataFrame({
        'PCA1': pca_result[:, 0],
        'PCA2': pca_result[:, 1],
        'PCA3': pca_result[:, 2],
        'Class': y_sample
    })
    fig = px.scatter_3d(
        df_vis, x='PCA1', y='PCA2', z='PCA3',
        color='Class',
        color_discrete_map={0: '#00ff00', 1: '#ff0000'},
        opacity=0.9,
        title="3D Anomaly Visualization"
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="#000000", plot_bgcolor="#000000",
        scene=dict(
            xaxis=dict(backgroundcolor="#000", gridcolor="#0f0", color="#0f0", title="PCA1"),
            yaxis=dict(backgroundcolor="#000", gridcolor="#0f0", color="#0f0", title="PCA2"),
            zaxis=dict(backgroundcolor="#000", gridcolor="#0f0", color="#0f0", title="PCA3"),
        ),
        title=dict(font=dict(color="#00ff00", size=24)),
        legend=dict(font=dict(color="#00ff00")),
    )
    fig.update_traces(marker=dict(size=4, line=dict(width=0.5, color="#ffffff")))
    return fig.to_html(full_html=False, include_plotlyjs='cdn')

def load_data_fast():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset yok: {DATA_PATH}")
    df_head = pd.read_csv(DATA_PATH, nrows=5)
    cols = list(df_head.columns)
    usecols = [c for c in cols if c != 'Time']
    df = pd.read_csv(DATA_PATH, usecols=usecols, engine="c", memory_map=True, low_memory=False)
    return df

def split_and_save_data(df):
    global test_data_cache
    log("🔪 Dataset train, validation, test olarak ayrılıyor...")
    y = df['Class']
    X = df.drop(columns=['Class'])

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    train_df.to_csv(os.path.join(OUTPUT_FOLDER, "train.csv"), index=False)
    val_df.to_csv(os.path.join(OUTPUT_FOLDER, "validation.csv"), index=False)
    test_df.to_csv(os.path.join(OUTPUT_FOLDER, "test.csv"), index=False)
    
    log(f"💾 train.csv kaydedildi: {X_train.shape}")
    log(f"💾 validation.csv kaydedildi: {X_val.shape}")
    log(f"💾 test.csv kaydedildi: {X_test.shape}")
    
    test_data_cache = test_df
    
    return X_train, X_test, y_train, y_test

def evaluate_and_report(model, X, y):
    log("📊 Model değerlendiriliyor...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1] 
    
    f1_local = f1_score(y_test, y_pred, average='weighted')
    cm_local = confusion_matrix(y_test, y_pred)
    report_str = classification_report(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    log(f"📊 Weighted F1: {f1_local:.4f}")
    log(f"🔢 Confusion Matrix:\n{cm_local}")

    log("📝 Rapor ve görseller staj_data/ içine yazılıyor...")
    save_png_confusion_matrix(cm_local, f1_local)
    save_png_f1(f1_local)
    save_png_class_report(report_dict)
    save_txt_report(f1_local, cm_local, report_str)
    save_png_heatmap(X)
    save_roc_curve(y_test, y_pred_prob)

    plot_html_local = make_pca_plot_html(X, y)
    
    return f1_local, cm_local, report_str, plot_html_local

def save_roc_curve(y_test, y_pred_prob):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, "roc_curve.png"))
    plt.close()

@app.on_event("startup")
def startup_pipeline():
    global cm, report, f1, plot_html, model_cache
    try:
        log("=== Startup pipeline ===")
        df = load_data_fast()
        log(f"Dataset shape: {df.shape}")
        
        X_train, X_test, y_train, y_test = split_and_save_data(df)

        if os.path.exists(MODEL_PATH):
            log("✅ Mevcut model yükleniyor...")
            model_cache = joblib.load(MODEL_PATH)
        else:
            log("🛠️ Model eğitiliyor (BalancedRandomForest, n_jobs=-1)...")
            model_cache = BalancedRandomForestClassifier(
                n_estimators=100, 
                criterion="gini", 
                max_depth=None, 
                random_state=42,
                n_jobs=-1
            )
            model_cache.fit(X_train, y_train)
            joblib.dump(model_cache, MODEL_PATH)
            log(f"💾 Model kaydedildi: {MODEL_PATH}")

        f1_v, cm_v, report_v, plot_html_v = evaluate_and_report(model_cache, df.drop(columns=['Class']), df['Class'])
        f1, cm, report, plot_html = f1_v, cm_v, report_v, plot_html_v
        log("✅ Pipeline tamam. Raporlar staj_data/ klasöründe.")

    except Exception:
        log("🔥 Hata:\n" + traceback.format_exc())

@app.get("/", include_in_schema=False, tags=["Train"])
def main_page():
    if cm is None or model_cache is None:
        return HTMLResponse(
            "<h1 style='color:#39ff14'>❌ Model hazır değil. staj_data/logs.txt kontrol edin.</h1>",
            status_code=200
        )
    cm_html = "<table class='matrix'>" + "".join(
        ["<tr>" + "".join([f"<td>{val}</td>" for val in row]) + "</tr>" for row in cm]
    ) + "</table>"

    model_name = type(model_cache).__name__
    model_params = model_cache.get_params()
    params_html_items = "".join([f"<li><span>{k}</span><code>{v}</code></li>" for k, v in model_params.items()])
    params_html = f"<details open><summary>Parameters</summary><ul class='params'>{params_html_items}</ul></details>"
    html = f"""
    <html>
    <head>
      <title>Fraud Detection Report</title>
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <style>
        :root {{
          --bg: #0a0f14;
          --panel: #0e141b;
          --accent: #00e5ff;       /* neon blue lines / borders */
          --text: #39ff14;         /* neon green text */
          --grid: rgba(0,229,255,0.2);
        }}
        * {{ box-sizing: border-box; }}
        body {{
          margin: 0;
          background: var(--bg);
          color: var(--text);
          font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
        }}
        a {{ color: var(--text); text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        /* Layout */
        .wrap {{
          display: grid;
          grid-template-columns: 320px 1fr;
          gap: 14px;
          padding: 16px;
        }}
        @media (max-width: 960px) {{
          .wrap {{ grid-template-columns: 1fr; }}
        }}
        .card {{
          background: var(--panel);
          border: 1px solid var(--accent);
          border-radius: 16px;
          padding: 12px 14px;
          box-shadow: 0 0 24px rgba(0,229,255,0.08), inset 0 0 0 1px rgba(255,255,255,0.02);
        }}
        .glow-title {{
          color: var(--text);
          text-shadow: 0 0 8px #39ff14;
          margin: 0 0 8px 0;
          font-size: 20px;
        }}
        header {{
          grid-column: 1 / -1;
          display: flex; align-items: center; justify-content: space-between;
          padding: 10px 16px; margin: 6px 16px 0; border-bottom: 1px solid var(--accent);
        }}
        header h1 {{
          color: var(--text);
          text-shadow: 0 0 10px #39ff14;
          font-size: 22px; margin: 0;
        }}
        /* Sidebar */
        .sidebar .meta {{ font-size: 13px; opacity: 0.9; }}
        .params {{ list-style: none; padding: 0; margin: 8px 0 0 0; max-height: 260px; overflow: auto; }}
        .params li {{ display: flex; justify-content: space-between; gap: 10px; padding: 4px 0; border-bottom: 1px dashed var(--accent); }}
        .params li span {{ color: var(--text); }}
        .params li code {{ color: var(--text); opacity: 0.9; }}
        .downloads a {{
          display: inline-block; margin: 4px 6px 0 0; padding: 6px 10px; border-radius: 10px;
          border: 1px solid var(--accent); background: rgba(0,229,255,0.06);
          color: var(--text);
        }}
        /* Horizontal tiles */
        .row {{
          display: flex; gap: 12px; overflow-x: auto; padding-bottom: 6px;
        }}
        .tile {{ min-width: 320px; max-width: 520px; flex: 0 0 auto; }}
        .tile .content {{ max-height: 380px; overflow: auto; border-radius: 10px; }}
        /* Tables */
        table.matrix {{
          border-collapse: collapse; width: 100%;
          background: rgba(0,229,255,0.04);
          color: var(--text);
        }}
        table.matrix td {{
          border: 1px solid var(--accent);
          padding: 6px 10px;
          text-align: center;
        }}
        pre {{
          margin: 0;
          padding: 10px;
          background: rgba(0,229,255,0.04);
          border: 1px solid var(--accent);
          border-radius: 10px;
          color: var(--text);
          overflow: auto;
        }}
        .sidebar, .tile .content, .pca-wrap {{
          color: var(--text);
        }}
        .pca-wrap {{ height: 68vh; min-height: 420px; }}
      </style>
    </head>
    <body>
      <header>
        <h1>💳 Credit Card Fraud — Neon Report</h1>
        <div class="hint">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
      </header>

      <div class="wrap">
        <aside class="sidebar card">
          <h2 class="glow-title">Model</h2>
          <div class="meta"><b>Name:</b> <span>{model_name}</span></div>
          {params_html}

          <h2 class="glow-title" style="margin-top:12px;">Downloads</h2>
          <div class="downloads">
            <a href="/matrix_png" target="_blank">confusion_matrix.png</a>
            <a href="/f1_png" target="_blank">f1_score.png</a>
            <a href="/report_png" target="_blank">classification_report.png</a>
            <a href="/heatmap_png" target="_blank">heatmap.png</a>
            <a href="/roc_curve_png" target="_blank">roc_curve.png</a>
            <a href="/report_txt" target="_blank">report.txt</a>
            <a href="/data/train.csv" target="_blank">train.csv</a>
            <a href="/data/validation.csv" target="_blank">validation.csv</a>
            <a href="/data/test.csv" target="_blank">test.csv</a>
          </div>
        </aside>

        <main class="main">
          <div class="row">
            <section class="card tile">
              <h2 class="glow-title">Weighted F1</h2>
              <div class="content"><pre>{f1:.4f}</pre></div>
            </section>

            <section class="card tile">
              <h2 class="glow-title">Confusion Matrix</h2>
              <div class="content">{cm_html}</div>
            </section>

            <section class="card tile">
              <h2 class="glow-title">Classification Report</h2>
              <div class="content"><pre>{report}</pre></div>
            </section>
          </div>

          <section class="card pca-wrap" style="margin-top:12px;">
            <h2 class="glow-title">3D PCA</h2>
            <div style="height: calc(100% - 34px);">{plot_html}</div>
          </section>
        </main>
      </div>
    </body>
    </html>
    """
    return HTMLResponse(html)

@app.post("/train", tags=["Training"])
def train(
    n_estimators: int = Query(100, description="Number of trees in the forest"),
    criterion: str = Query("gini", description="Splitting criterion: gini or entropy"),
    max_depth: int | None = Query(None, description="Maximum depth of trees")
):
    global model_cache, f1, cm, report, plot_html
    
    params_txt_path = os.path.join(OUTPUT_FOLDER, "model_parameters.txt")
    with open(params_txt_path, "w") as f:
        f.write(f"n_estimators: {n_estimators}\n")
        f.write(f"criterion: {criterion}\n")
        f.write(f"max_depth: {max_depth}\n")
    
    model_cache = BalancedRandomForestClassifier(
        n_estimators=n_estimators,
        criterion=criterion,
        max_depth=max_depth,
        n_jobs=-1,
        random_state=42
    )
    
    df = load_data_fast()
    X_train, X_test, y_train, y_test = split_and_save_data(df)
    model_cache.fit(X_train, y_train)
    
    f1_v, cm_v, report_v, plot_html_v = evaluate_and_report(model_cache, df.drop(columns=['Class']), df['Class'])
    f1, cm, report, plot_html = f1_v, cm_v, report_v, plot_html_v
    
    y_pred_prob = model_cache.predict_proba(X_test)[:, 1]
    from sklearn.metrics import roc_auc_score
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    with open(params_txt_path, "a") as f:
        f.write(f"ROC AUC: {roc_auc:.4f}\n")
    
    joblib.dump(model_cache, MODEL_PATH)
    
    return {"message": "Model trained", "ROC_AUC": roc_auc}

@app.get("/metrics", include_in_schema=False)
def metrics():
    if cm is None or model_cache is None:
        raise HTTPException(status_code=503, detail="Model not ready")
    return {"message": "Metrics available internally"}

@app.get("/matrix_png", include_in_schema=False)
def get_matrix_png():
    return FileResponse(os.path.join(OUTPUT_FOLDER, "confusion_matrix.png"), media_type="image/png")

@app.get("/f1_png", include_in_schema=False)
def get_f1_png():
    return FileResponse(os.path.join(OUTPUT_FOLDER, "f1_score.png"), media_type="image/png")

@app.get("/report_png", include_in_schema=False)
def get_report_png():
    return FileResponse(os.path.join(OUTPUT_FOLDER, "classification_report.png"), media_type="image/png")

@app.get("/heatmap_png", include_in_schema=False)
def get_heatmap_png():
    return FileResponse(os.path.join(OUTPUT_FOLDER, "heatmap.png"), media_type="image/png")

@app.get("/roc_curve_png", include_in_schema=False)
def get_roc_curve_png():
    return FileResponse(os.path.join(OUTPUT_FOLDER, "roc_curve.png"), media_type="image/png")

@app.get("/report_txt", include_in_schema=False)
def get_report_txt():
    return FileResponse(REPORT_TXT_PATH, media_type="text/plain")
    
@app.get("/data/train.csv", include_in_schema=False)
def get_train_csv():
    return FileResponse(os.path.join(OUTPUT_FOLDER, "train.csv"), media_type="text/csv")

@app.get("/data/validation.csv", include_in_schema=False)
def get_validation_csv():
    return FileResponse(os.path.join(OUTPUT_FOLDER, "validation.csv"), media_type="text/csv")

@app.get("/data/test.csv", include_in_schema=False)
def get_test_csv():
    return FileResponse(os.path.join(OUTPUT_FOLDER, "test.csv"), media_type="text/csv")

@app.post("/predict", tags=["Prediction"])
def predict(transaction_index: int = Query(..., description="Index of the transaction in the test.csv file.")):
    global test_data_cache
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=500, detail="Model not ready")
    if test_data_cache is None:
        test_path = os.path.join(OUTPUT_FOLDER, "test.csv")
        if not os.path.exists(test_path):
            raise HTTPException(status_code=500, detail="Test data not found. Please train the model first.")
        try:
            test_data_cache = pd.read_csv(test_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load test data: {str(e)}")
            
    try:
        if transaction_index < 0 or transaction_index >= len(test_data_cache):
            raise HTTPException(status_code=404, detail="Index out of range for test data.")
        
        transaction_row = test_data_cache.iloc[[transaction_index]].drop(columns=['Class'])

        features_with_names = transaction_row.values.tolist()
        
        model = joblib.load(MODEL_PATH)
        pred = model.predict(features_with_names)
        proba = model.predict_proba(features_with_names)
        
        
        return {
            "transaction_index": transaction_index,
            "prediction": int(pred[0]),
            "probabilities": {
                "class_0": float(proba[0][0]),
                "class_1": float(proba[0][1])
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))