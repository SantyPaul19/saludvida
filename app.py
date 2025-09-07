import math
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
from settings import settings

app = FastAPI(title="Riesgo de DM (PRE)")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Conexión a la base de datos
engine = create_engine(settings.sql_url, poolclass=QueuePool, pool_size=5, max_overflow=10)

# Cargar el modelo entrenado
def load_model():
    try:
        return joblib.load(settings.model_path)
    except Exception as e:
        print("⚠️ ADVERTENCIA: No se pudo cargar el modelo:", e)
        return None

model = load_model()

# Lista de features esperadas por el modelo
FEATURES = [
    "ex_fum","consum_alcoh","consum_alcoh_30","niveldeactividadesemanal","act_fis_frisk",
    "diet_frisk","med_hta_fr","glu_alta","parien_dm","edad","mets","sedentarismo",
    "talla","peso","imc","mme"
]

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,
    ex_fum: float = Form(...),
    consum_alcoh: float = Form(...),
    consum_alcoh_30: float = Form(...),
    niveldeactividadesemanal: float = Form(...),
    act_fis_frisk: float = Form(...),
    diet_frisk: float = Form(...),
    med_hta_fr: float = Form(...),
    glu_alta: float = Form(...),
    parien_dm: float = Form(...),
    edad: float = Form(...),
    mets: float = Form(...),
    sedentarismo: float = Form(...),
    talla: float = Form(...),
    peso: float = Form(...),
    imc: float = Form(...),
    mme: float = Form(...)
):
    values = [ex_fum, consum_alcoh, consum_alcoh_30, niveldeactividadesemanal,
              act_fis_frisk, diet_frisk, med_hta_fr, glu_alta, parien_dm,
              edad, mets, sedentarismo, talla, peso, imc, mme]

    proba, label = 0.0, 0
    if model:
        X = np.array(values).reshape(1, -1)
        proba = float(model.predict_proba(X)[:,1][0])
        label = int(proba >= settings.risk_threshold)

    # Guardar en BD
    with engine.begin() as conn:
        res = conn.execute(
            text("""
                insert into patients (ex_fum, consum_alcoh, consum_alcoh_30, niveldeactividadesemanal,
                act_fis_frisk, diet_frisk, med_hta_fr, glu_alta, parien_dm, edad,
                mets, sedentarismo, talla, peso, imc, mme)
                values (:ex_fum,:consum_alcoh,:consum_alcoh_30,:niveldeactividadesemanal,
                        :act_fis_frisk,:diet_frisk,:med_hta_fr,:glu_alta,:parien_dm,:edad,
                        :mets,:sedentarismo,:talla,:peso,:imc,:mme)
                returning id;
            """),
            dict(zip(FEATURES, values))
        )
        patient_id = res.scalar_one()
        conn.execute(
            text("""
                insert into predictions (patient_id, predicted_proba, predicted_label, threshold)
                values (:patient_id,:predicted_proba,:predicted_label,:threshold)
            """),
            {"patient_id": patient_id, "predicted_proba": proba,
             "predicted_label": label, "threshold": settings.risk_threshold}
        )

    return templates.TemplateResponse("result.html", {
        "request": request,
        "proba": round(proba,4),
        "label": "RIESGOSO" if label==1 else "NO RIESGOSO",
        "threshold": settings.risk_threshold
    })

# HISTORICO
from datetime import datetime

@app.get("/history", response_class=HTMLResponse)
def history(request: Request):
    rows = []
    with engine.connect() as conn:
        result = conn.execute(text("""
            select id, created_at, ex_fum, consum_alcoh, consum_alcoh_30, niveldeactividadesemanal,
                   act_fis_frisk, diet_frisk, med_hta_fr, glu_alta, parien_dm, edad, mets, sedentarismo,
                   talla, peso, imc, mme
            from patients
            order by created_at desc
            limit 500
        """))
        patients = result.mappings().all()

        if patients:
            # Trae la última predicción por paciente usando la vista
            pred_result = conn.execute(text("""
                select p.id as patient_id, lp.predicted_proba, lp.predicted_label, lp.predicted_at
                from patients p
                join latest_predictions lp on lp.id = p.id
                order by lp.predicted_at desc
                limit 500
            """))
            preds = {r["patient_id"]: r for r in pred_result.mappings().all()}

            for p in patients:
                pr = preds.get(p["id"])
                rows.append({
                    "id": p["id"],
                    "created_at": p["created_at"].strftime("%Y-%m-%d %H:%M"),
                    "edad": p["edad"],
                    "imc": p["imc"],
                    "peso": p["peso"],
                    "proba": round(pr["predicted_proba"], 4) if pr else None,
                    "label": "RIESGOSO" if (pr and pr["predicted_proba"] >= settings.risk_threshold) else "NO RIESGOSO",
                    "predicted_at": pr["predicted_at"].strftime("%Y-%m-%d %H:%M") if pr else None
                })

    # Para el gráfico de líneas (probabilidades ordenadas por fecha de predicción)
    chart_labels = [r["predicted_at"] for r in rows if r["predicted_at"]]
    chart_values = [r["proba"] for r in rows if r["proba"] is not None]

    return templates.TemplateResponse("history.html", {
        "request": request,
        "rows": rows,
        "chart_labels": chart_labels,
        "chart_values": chart_values,
        "threshold": settings.risk_threshold
    })

# CSV
from fastapi.responses import PlainTextResponse

@app.get("/export.csv", response_class=PlainTextResponse)
def export_csv():
    import csv, io
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["id","created_at","edad","peso","imc","proba","label","predicted_at"])
    with engine.connect() as conn:
        res = conn.execute(text("""
            select p.id, p.created_at, p.edad, p.peso, p.imc,
                   pr.predicted_proba, pr.predicted_label, pr.created_at as predicted_at
            from patients p
            join predictions pr on pr.patient_id = p.id
            order by pr.created_at desc
        """))
        for row in res:
            writer.writerow([row[0], row[1], row[2], row[3], row[4], round(row[5],4), row[6], row[7]])
    return output.getvalue()


# app.py
import os
from flask import Flask, render_template

app = Flask(__name__, template_folder='templates', static_folder='static')

@app.get("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  # Railway te pone PORT
    app.run(host="0.0.0.0", port=port)
