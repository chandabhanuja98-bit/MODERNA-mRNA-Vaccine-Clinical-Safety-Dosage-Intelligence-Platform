import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve , auc
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import json
from pathlib import Path



data_path = Path("data/patient_records_clean.csv")
model_path = Path("outputs/risk_model.joblib")
latency_path = Path("outputs/reports/latencies.json")
attention_path = Path("outputs/figures/figures.png")
output_path = Path("outputs/figures/dashboard.png")


numeric_columns = ['Age','Biomarker_Level','Vaccine_Dose_mcg']
categorical_columns = 'Prior_Reaction_History'
target = 'Reaction_Score'
feature_columns = numeric_columns + [categorical_columns,'Pre_Existing_Conditions'] 


def load_roc_data():
    df = pd.read_csv(data_path)
    scaler = StandardScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    encoder = LabelEncoder()
    df[categorical_columns] = encoder.fit_transform(df[categorical_columns])
    x=df[feature_columns]
    y=df[target]
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)
    return x_test,y_test

def load_latencies():
    
    if not latency_path.exists():
        raise FileNotFoundError(
            f"Latency file not found: {latency_path}\n"
            "Run report_generator.py with latency logging enabled."
        )

    with latency_path.open("r", encoding="utf-8") as file:
        latencies = json.load(file)

    required_keys = [
        "rag_query_seconds",
        "Gemini_report_seconds",
    ]

    for key in required_keys:
        if key not in latencies:
            raise ValueError(
                f"Missing '{key}' in latencies.json"
            )

        value = latencies[key]

        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not np.isfinite(value)
            or value < 0
        ):
            raise ValueError(
                f"Invalid latency value for '{key}': {value}"
            )

    return latencies


def build_dashboard() :
    model = joblib.load(model_path)
    x_test,y_test = load_roc_data()

    class_index = list(model.classes_).index(1)
    y_scores = model.predict_proba(x_test)[:,class_index]
    if not attention_path.exists() :
        raise FileNotFoundError("attention heatmap not found")
    attention_image = plt.imread(attention_path)

    latencies = load_latencies()
    fig,axes = plt.subplots(1,3,figsize=(18,6))

    fpr,tpr,_ = roc_curve(y_test,y_scores,pos_label=1)
    roc_auc = auc(fpr,tpr)
    axes[0].plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    axes[0].plot([0,1],[0,1],linestyle="--",color = "gray",label = "Random Base Line")
    axes[0].set_title("Week 1: Risk Classifier ROC Curve")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].imshow(attention_image)
    axes[1].set_title("Week 2: mRNA Self-Attention Heatmap")
    axes[1].axis("off")

    labels = ["rag_retrival","Gemini_report"]
    values = [latencies["rag_query_seconds"],latencies["Gemini_report_seconds"]]
    bars = axes[2].bar(labels,values)
    axes[2].set_title("Week 3–4: Pipeline Latency")
    axes[2].set_ylabel("Time (seconds)")
    axes[2].tick_params(axis="x", rotation=12)
    axes[2].grid(axis="y", alpha=0.25)

    fig.suptitle(
        "mRNA-VIP Project Dashboard",
        fontsize=16,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.94])

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("\nDashboard generated successfully!")
    print(f"Saved image: {output_path}")
    print(f"ROC AUC: {roc_auc:.3f}")
    print(
        f"RAG retrieval: "
        f"{values[0]:.3f} seconds"
    )
    print(
        f"Gemini report generation: "
        f"{values[1]:.3f} seconds"
    )



if __name__ == "__main__":

    build_dashboard()


