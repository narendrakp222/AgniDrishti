import pandas as pd
import numpy as np
import os
from flask import Flask, render_template, send_file
from pathlib import Path
from io import BytesIO
from datetime import datetime

app = Flask(__name__)
DATA_DIR = Path(__file__).resolve().parent / "data" / "processed"


def load_dataset(filename):
    path = DATA_DIR / filename
    if path.exists():
        return pd.read_csv(path)

    hotspots = pd.read_csv(DATA_DIR / "india_hotspots.csv")
    frp_score = hotspots["frp"].clip(upper=80) / 80
    brightness_score = (hotspots["brightness"] - 295) / (367 - 295)
    hotspots["risk_score"] = ((0.6 * frp_score + 0.4 * brightness_score) * 100).round(2)
    hotspots["georisk_score"] = hotspots["risk_score"]
    hotspots["event_type"] = "Fire"
    hotspots["anomaly_status"] = "Normal"
    hotspots["escalation_level"] = pd.cut(
        hotspots["georisk_score"],
        bins=[-float("inf"), 40, 70, 100, float("inf")],
        labels=["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    ).astype(str)
    hotspots["notification_message"] = hotspots["escalation_level"].map({
        "CRITICAL": "Critical alert: immediate emergency response required.",
        "HIGH": "High alert: dispatch monitoring team.",
        "MEDIUM": "Medium alert: continue monitoring.",
        "LOW": "Low alert: no action required."
    })
    return hotspots


@app.route("/")
def home():

    escalation_df = load_dataset("escalation_dataset.csv")

    events_df = load_dataset("classified_events.csv")

    alerts_df = load_dataset("alert_dataset.csv")

    total_hotspots = len(escalation_df)

    avg_risk = round(
        escalation_df["risk_score"].mean(),
        2
    )

    avg_georisk = round(
        escalation_df["georisk_score"].mean(),
        2
    )

    # Alert Summary

    critical_alerts = len(
        alerts_df[
            alerts_df["alert_level"] == "CRITICAL"
        ]
    )

    high_alerts = len(
        alerts_df[
            alerts_df["alert_level"] == "HIGH"
        ]
    )

    medium_alerts = len(
        alerts_df[
            alerts_df["alert_level"] == "MEDIUM"
        ]
    )

    low_alerts = len(
        alerts_df[
            alerts_df["alert_level"] == "LOW"
        ]
    )

    # Escalation Summary

    critical_escalations = len(
        escalation_df[
            escalation_df["escalation_level"] == "CRITICAL"
        ]
    )

    high_escalations = len(
        escalation_df[
            escalation_df["escalation_level"] == "HIGH"
        ]
    )

    medium_escalations = len(
        escalation_df[
            escalation_df["escalation_level"] == "MEDIUM"
        ]
    )

    low_escalations = len(
        escalation_df[
            escalation_df["escalation_level"] == "LOW"
        ]
    )

    # Anomalies

    total_anomalies = len(
        escalation_df[
            escalation_df["anomaly_status"] == "Anomaly"
        ]
    )

    # Event Classification

    event_summary = (
        events_df["event_type"]
        .value_counts()
        .reset_index()
    )

    event_summary.columns = [
        "event_type",
        "count"
    ]

    classification_columns = [
        "latitude",
        "longitude",
        "event_type"
    ]
    if all(column in events_df.columns for column in classification_columns):
        escalation_df = escalation_df.drop(
            columns=["event_type"],
            errors="ignore"
        ).merge(
            events_df[classification_columns].drop_duplicates(
                subset=["latitude", "longitude"]
            ),
            on=["latitude", "longitude"],
            how="left"
        )
        escalation_df["event_type"] = escalation_df[
            "event_type"
        ].fillna("Unknown Thermal Source")

    # Keep the command board diverse: show at most two incidents per class.
    top_escalations = (
        escalation_df
        .sort_values("georisk_score", ascending=False)
        .groupby("event_type", group_keys=False)
        .head(2)
        .copy()
    )
    priority_map = {
        "CRITICAL": "🔴",
        "HIGH": "🟠",
        "MEDIUM": "🟡",
        "LOW": "🟢"
    }
    top_escalations["priority"] = top_escalations[
        "escalation_level"
    ].map(priority_map).fillna("⚪")

    notification_count = len(
    escalation_df[
        escalation_df["escalation_level"].isin(
            ["CRITICAL", "HIGH"]
        )
    ]
)
    prediction_df = load_dataset("prediction_dataset.csv")
    high_probability = len(
        prediction_df[
            prediction_df["escalation_probability"] > 80
        ]
    )
    return render_template(
        "index.html",

        total_hotspots=total_hotspots,
        avg_risk=avg_risk,
        avg_georisk=avg_georisk,

        critical_alerts=critical_alerts,
        high_alerts=high_alerts,
        medium_alerts=medium_alerts,
        low_alerts=low_alerts,

        critical_escalations=critical_escalations,
        high_escalations=high_escalations,
        medium_escalations=medium_escalations,
        low_escalations=low_escalations,

        total_anomalies=total_anomalies,

        event_summary=event_summary.to_dict(
            "records"
        ),

        top_escalations=top_escalations.to_dict(
            "records",
        ),
        recent_alerts=top_escalations.head(5).to_dict("records"),
        notification_count=notification_count,
        high_probability=high_probability
    )


@app.route("/mapper")
def map_view():

    df = probability_data().sort_values(
        by="georisk_score",
        ascending=False
    )
    classified_path = DATA_DIR / "classified_events.csv"
    if classified_path.exists():
        classified = pd.read_csv(classified_path)
        df = df.drop(columns=["event_type"], errors="ignore").merge(
            classified[["latitude", "longitude", "event_type"]].drop_duplicates(
                ["latitude", "longitude"]
            ),
            on=["latitude", "longitude"],
            how="left"
        )
    weather = pd.read_csv(DATA_DIR / "weather_dataset.csv")
    population = pd.read_csv(DATA_DIR / "population_dataset.csv")
    df = df.merge(
        weather[["latitude", "longitude", "weather_risk_score"]],
        on=["latitude", "longitude"], how="left"
    ).merge(
        population[["latitude", "longitude", "population_exposed"]],
        on=["latitude", "longitude"], how="left"
    )
    incident = df.iloc[0].to_dict()
    incident["anomaly_label"] = "Yes" if incident.get("anomaly_status") == "Anomaly" else "No"
    incident["weather_label"] = (
        "HIGH" if incident.get("weather_risk_score", 0) >= 25
        else "MEDIUM" if incident.get("weather_risk_score", 0) >= 15
        else "LOW"
    )

    return render_template(
        "map_workspace.html",
        incident=incident,
        incident_count=len(df),
        critical_count=int((df["escalation_level"] == "CRITICAL").sum()),
        high_count=int((df["escalation_level"] == "HIGH").sum())
    )


@app.route("/map-document")
def map_document():
    return render_template("hotspot_map.html")


@app.route("/infrastructure")
def infrastructure():

    df = load_dataset("escalation_dataset.csv")

    infra_data = df[
        [
            "latitude",
            "longitude",
            "hospital_km",
            "industry_km",
            "factory_km",
            "powerplant_km",
            "georisk_score",
            "escalation_level"
        ]
    ]

    return render_template(
        "infrastructure.html",
        infra_data=infra_data.to_dict(
            "records"
        )
    )


@app.route("/anomalies")
def anomalies():

    df = load_dataset("anomaly_dataset.csv")

    anomaly_data = df[
        df["anomaly_status"] == "Anomaly"
    ]

    return render_template(
        "anomalies.html",
        anomaly_data=anomaly_data.to_dict(
            "records"
        )
    )
@app.route("/alerts")
def alerts():

    df = load_dataset("escalation_dataset.csv")

    critical = df[
        df["escalation_level"] == "CRITICAL"
    ]

    high = df[
        df["escalation_level"] == "HIGH"
    ]

    medium = df[
        df["escalation_level"] == "MEDIUM"
    ]

    return render_template(
        "alerts.html",

        critical_count=len(critical),
        high_count=len(high),
        medium_count=len(medium),

        critical_alerts=critical
        .sort_values(
            by="georisk_score",
            ascending=False
        )
        .head(20)
        .to_dict("records"),

        high_alerts=high
        .sort_values(
            by="georisk_score",
            ascending=False
        )
        .head(20)
        .to_dict("records"),

        medium_alerts=medium
        .sort_values(
            by="georisk_score",
            ascending=False
        )
        .head(20)
        .to_dict("records")
    )
@app.route("/notifications")
def notifications():

    df = load_dataset("notification_dataset.csv")

    notifications = df.sort_values(
        by="georisk_score",
        ascending=False
    ).head(20)

    def recommended_action(level):
        actions = {
            "CRITICAL": "Dispatch emergency response and notify district authorities.",
            "HIGH": "Deploy a field team and increase monitoring frequency.",
            "MEDIUM": "Review the incident and continue active monitoring.",
            "LOW": "Maintain routine surveillance."
        }
        return actions.get(level, "Review incident details.")

    notification_records = notifications.to_dict("records")
    for record in notification_records:
        record["recommended_action"] = recommended_action(
            record["escalation_level"]
        )

    return render_template(
        "notifications.html",
        notifications=notification_records,
        total_notifications=len(df),
        critical_alerts=int((df["escalation_level"] == "CRITICAL").sum()),
        high_alerts=int((df["escalation_level"] == "HIGH").sum()),
        medium_alerts=int((df["escalation_level"] == "MEDIUM").sum())
    )


@app.route("/predictions")
def predictions():

    df = probability_data()

    if "escalation_probability" not in df:
        def predict_probability(row):
            score = row["georisk_score"]

            if score >= 100:
                return 95
            elif score >= 90:
                return 85
            elif score >= 75:
                return 70
            elif score >= 60:
                return 50
            return 25

        df["escalation_probability"] = df.apply(
            predict_probability,
            axis=1
        )

    top_predictions = df.sort_values(
        by="escalation_probability",
        ascending=False
    )

    def recommended_action(level):
        return {
            "CRITICAL": "Dispatch emergency response team immediately.",
            "HIGH": "Deploy field team and increase monitoring.",
            "MEDIUM": "Review incident and continue active monitoring.",
            "LOW": "Maintain routine surveillance."
        }.get(level, "Review incident details.")

    prediction_records = top_predictions.to_dict("records")
    level_classes = {
        "CRITICAL": ("danger", "prediction-critical"),
        "HIGH": ("warning text-dark", "prediction-high"),
        "MEDIUM": ("warning text-dark", "prediction-medium"),
        "LOW": ("success", "prediction-low")
    }
    for record in prediction_records:
        record["recommended_action"] = recommended_action(
            record["escalation_level"]
        )
        record["badge_class"], record["card_class"] = level_classes.get(
            record["escalation_level"], ("secondary", "prediction-low")
        )
        record["progress_class"] = record["badge_class"].split(" ")[0]

    historical = pd.read_csv(DATA_DIR / "historical_trends.csv")
    trend_delta = float(historical["avg_georisk"].iloc[-1] - historical["avg_georisk"].iloc[-2]) if len(historical) > 1 else 0
    risk_trend = "Rising" if trend_delta > 0 else "Falling" if trend_delta < 0 else "Stable"
    average_prediction = round(df["escalation_probability"].mean(), 2)
    distribution_levels = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]

    return render_template(
        "predictions.html",
        predictions=prediction_records,
        highest_prediction=round(df["escalation_probability"].max(), 2),
        average_prediction=average_prediction,
        critical_incidents=int((df["escalation_level"] == "CRITICAL").sum()),
        high_risk_incidents=int((df["escalation_level"] == "HIGH").sum()),
        above_90=int((df["escalation_probability"] > 90).sum()),
        highest_georisk=round(df["georisk_score"].max(), 2),
        risk_trend=risk_trend,
        recommended_action=("Immediate intervention recommended" if average_prediction > 80 else "Continue active monitoring"),
        distribution_labels=distribution_levels,
        distribution_values=[int((df["escalation_level"] == level).sum()) for level in distribution_levels]
    )


@app.route("/explainability")
def explainability():

    df = probability_data()
    classified_path = DATA_DIR / "classified_events.csv"
    if classified_path.exists():
        classified = pd.read_csv(classified_path)
        df = df.drop(columns=["event_type"], errors="ignore").merge(
            classified[["latitude", "longitude", "event_type"]].drop_duplicates(
                ["latitude", "longitude"]
            ),
            on=["latitude", "longitude"],
            how="left"
        )
    weather = pd.read_csv(DATA_DIR / "weather_dataset.csv")
    population = pd.read_csv(DATA_DIR / "population_dataset.csv")
    threats = pd.read_csv(DATA_DIR / "infrastructure_threat_dataset.csv")
    df = df.merge(
        weather[["latitude", "longitude", "weather_risk_score"]],
        on=["latitude", "longitude"], how="left"
    ).merge(
        population[["latitude", "longitude", "population_exposed"]],
        on=["latitude", "longitude"], how="left"
    )
    threat_values = {"LOW": 25, "MEDIUM": 50, "HIGH": 75, "CRITICAL": 100}
    threat_scores = threats[
        ["latitude", "longitude", "hospital_threat", "industry_threat", "factory_threat", "powerplant_threat"]
    ].copy()
    threat_scores["infrastructure_score"] = threat_scores.iloc[:, 2:].replace(threat_values).max(axis=1)
    df = df.merge(
        threat_scores[["latitude", "longitude", "infrastructure_score"]],
        on=["latitude", "longitude"], how="left"
    )
    explanations = []

    for _, row in df.iterrows():
        factors = []

        if row.get("frp", 0) > 8:
            factors.append("High FRP")

        if row.get("brightness", 0) > 345:
            factors.append("High Brightness")

        if row.get("anomaly_status") == "Anomaly":
            factors.append("Detected Anomaly")

        if row.get("industry_km", float("inf")) < 20:
            factors.append("Near Industry")

        explanations.append({
            "latitude": row["latitude"],
            "longitude": row["longitude"],
            "event_type": row.get("event_type", "Unknown Thermal Source"),
            "georisk": row["georisk_score"],
            "escalation": row["escalation_level"],
            "probability": row["escalation_probability"],
            "factors": ", ".join(factors) or "No dominant factors detected"
        })

    feature_importance = [
        {"name": "FRP", "value": 30, "color": "danger"},
        {"name": "Brightness", "value": 25, "color": "warning"},
        {"name": "Weather", "value": 18, "color": "info"},
        {"name": "Population", "value": 15, "color": "primary"},
        {"name": "Infrastructure", "value": 12, "color": "success"}
    ]

    return render_template(
        "explainability.html",
        explanations=explanations,
        feature_importance=feature_importance,
        ai_explanation="Prediction generated because high thermal intensity and infrastructure proximity increased escalation probability."
    )


@app.route("/temporal")
def temporal():

    temporal_path = DATA_DIR / "historical_trends.csv"
    if temporal_path.exists():
        df = pd.read_csv(temporal_path)
        df = df.rename(columns={
            "critical": "critical_count",
            "high": "high_count",
            "medium": "medium_count",
            "avg_georisk": "avg_georisk"
        })
    else:
        source = load_dataset("escalation_dataset.csv")
        source["date"] = pd.to_datetime(source["acq_date"]).dt.strftime("%Y-%m-%d")
        df = (
            source.groupby("date", as_index=False)
            .agg(
                total_hotspots=("georisk_score", "size"),
                critical_count=("escalation_level", lambda values: (values == "CRITICAL").sum()),
                high_count=("escalation_level", lambda values: (values == "HIGH").sum()),
                medium_count=("escalation_level", lambda values: (values == "MEDIUM").sum()),
                avg_georisk=("georisk_score", "mean")
            )
        )
        df["avg_georisk"] = df["avg_georisk"].round(2)

    records = df.to_dict("records")
    if len(df) < 2:
        hotspot_trend = "Insufficient history"
        growth_rate = None
        georisk_trend = "Insufficient history"
    else:
        previous = df.iloc[-2]
        current = df.iloc[-1]
        growth_rate = round(
            ((current["total_hotspots"] - previous["total_hotspots"])
             / previous["total_hotspots"] * 100),
            1
        ) if previous["total_hotspots"] else None
        hotspot_trend = (
            "Rising" if current["total_hotspots"] > previous["total_hotspots"]
            else "Falling" if current["total_hotspots"] < previous["total_hotspots"]
            else "Stable"
        )
        georisk_trend = (
            "Increasing" if current["avg_georisk"] > previous["avg_georisk"]
            else "Decreasing" if current["avg_georisk"] < previous["avg_georisk"]
            else "Stable"
        )

    return render_template(
        "temporal.html",
        records=records,
        chart_labels=[record["date"] for record in records],
        hotspot_values=[record["total_hotspots"] for record in records],
        georisk_values=[record["avg_georisk"] for record in records],
        critical_values=[record["critical_count"] for record in records],
        high_values=[record["high_count"] for record in records],
        medium_values=[record["medium_count"] for record in records],
        hotspot_trend=hotspot_trend,
        growth_rate=growth_rate,
        georisk_trend=georisk_trend
    )


@app.route("/probability")
def probability():

    probability_path = DATA_DIR / "probability_dataset.csv"
    if probability_path.exists():
        df = pd.read_csv(probability_path)
    else:
        df = load_dataset("escalation_dataset.csv")
        anomaly_bonus = df["anomaly_status"].eq("Anomaly").where(
            df["anomaly_status"].notna(), False
        ) * 10
        df["escalation_probability"] = (
            df["georisk_score"] + anomaly_bonus
        ).clip(upper=100).round(2)

    top_probability = df.sort_values(
        by="escalation_probability",
        ascending=False
    ).head(20)

    average_probability = round(df["escalation_probability"].mean(), 2)
    distribution = [
        int((df["escalation_level"] == level).sum())
        for level in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    ]
    if average_probability > 80:
        insight = "High likelihood of severe escalation in next 48 hours"
    elif average_probability > 60:
        insight = "Moderate escalation risk detected"
    else:
        insight = "Low escalation risk"

    return render_template(
        "probability.html",
        incidents=top_probability.to_dict("records"),
        total_critical=int((df["escalation_level"] == "CRITICAL").sum()),
        average_probability=average_probability,
        highest_probability=round(df["escalation_probability"].max(), 2),
        distribution=distribution,
        insight=insight
    )


def probability_data():
    path = DATA_DIR / "probability_dataset.csv"
    if path.exists():
        df = pd.read_csv(path)
    else:
        df = load_dataset("escalation_dataset.csv")

    if "escalation_probability" not in df:
        anomaly_bonus = df["anomaly_status"].eq("Anomaly").where(
            df["anomaly_status"].notna(), False
        ) * 10
        df["escalation_probability"] = (
            df["georisk_score"] + anomaly_bonus
        ).clip(upper=100).round(2)
    return df


def create_pdf(lines):
    def escape(text):
        return str(text).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    commands = ["BT", "/F1 10 Tf", "50 760 Td"]
    for index, line in enumerate(lines):
        if index:
            commands.append("0 -14 Td")
        commands.append(f"({escape(line)}) Tj")
    commands.append("ET")
    stream = "\n".join(commands).encode("latin-1", "replace")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream"
    ]
    pdf = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{number} 0 obj\n".encode())
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")
    xref = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets:
        pdf.extend(f"{offset:010d} 00000 n \n".encode())
    pdf.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    )
    return bytes(pdf)


@app.route("/report")
def report():

    df = probability_data().sort_values(
        by="escalation_probability",
        ascending=False
    )
    incident = df.iloc[0]

    def related_record(filename):
        path = DATA_DIR / filename
        if not path.exists():
            return {}
        related = pd.read_csv(path)
        match = related[
            (related["latitude"] == incident["latitude"])
            & (related["longitude"] == incident["longitude"])
        ]
        return match.iloc[0].to_dict() if not match.empty else {}

    weather = related_record("weather_dataset.csv")
    population = related_record("population_dataset.csv")
    threats = related_record("infrastructure_threat_dataset.csv")
    georisk = float(incident["georisk_score"])
    probability = float(incident["escalation_probability"])
    risk_category = "EXTREME" if georisk >= 100 else "HIGH" if georisk >= 80 else "MODERATE"
    weather_score = float(weather.get("weather_risk_score", 0))
    weather_impact = "HIGH" if weather_score >= 25 else "MODERATE" if weather_score >= 15 else "LOW"
    threat_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
    threat_values = [
        threats.get("hospital_threat", "LOW"),
        threats.get("industry_threat", "LOW"),
        threats.get("factory_threat", "LOW"),
        threats.get("powerplant_threat", "LOW")
    ]
    infrastructure_threat = max(threat_values, key=lambda value: threat_order.get(value, 0))
    resource_scale = 3 if probability >= 90 else 2 if probability >= 70 else 1
    anomaly_status = "Anomaly Detected" if incident.get("anomaly_status") == "Anomaly" else "Normal"
    likelihood = "HIGH" if probability > 80 else "MODERATE" if probability > 60 else "LOW"
    incident_id = f"GR-{datetime.now().year}-0001"

    lines = [
        "AGNIDRISHTI INCIDENT REPORT",
        "====================================================",
        "INCIDENT STATUS: " + str(incident["escalation_level"]),
        f"GeoRisk Score: {georisk:.2f}",
        f"Escalation Probability: {probability:.0f}%",
        f"Population Exposed: {int(population.get('population_exposed', 0)):,}",
        "",
        f"Incident ID: {incident_id}",
        f"Generated Time: {datetime.now().strftime('%Y-%m-%d %I:%M %p')}",
        "----------------------------------------------------",
        "INCIDENT LOCATION",
        f"Latitude: {incident['latitude']}",
        f"Longitude: {incident['longitude']}",
        "----------------------------------------------------",
        "RISK ASSESSMENT",
        f"GeoRisk Score: {georisk:.2f}",
        f"Risk Category: {risk_category}",
        f"Escalation Level: {incident['escalation_level']}",
        f"Escalation Probability: {probability:.0f}%",
        "----------------------------------------------------",
        "EVENT ANALYSIS",
        f"Event Type: {incident.get('event_type', 'Wildfire Hotspot')}",
        f"Anomaly Status: {anomaly_status}",
        "Weather Conditions:",
        f"Temperature: {weather.get('temperature', 'N/A')} C",
        f"Humidity: {weather.get('humidity', 'N/A')}%",
        f"Wind Speed: {weather.get('wind_speed', 'N/A')} km/h",
        f"Rainfall: {weather.get('rainfall', 'N/A')} mm",
        f"Weather Impact: {weather_impact}",
        "----------------------------------------------------",
        "POPULATION IMPACT",
        f"Population Exposed: {int(population.get('population_exposed', 0)):,}",
        f"Exposure Level: {population.get('exposure_level', 'N/A')}",
        "----------------------------------------------------",
        "CRITICAL INFRASTRUCTURE",
        f"Nearest Hospital: {incident.get('hospital_km', 'N/A')} km",
        f"Nearest Industry: {incident.get('industry_km', 'N/A')} km",
        f"Nearest Factory: {incident.get('factory_km', 'N/A')} km",
        f"Nearest Power Plant: {incident.get('powerplant_km', 'N/A')} km",
        f"Infrastructure Threat: {infrastructure_threat}",
        "----------------------------------------------------",
        "RECOMMENDED RESPONSE",
        "Priority Level: IMMEDIATE" if probability >= 80 else "Priority Level: HIGH",
        f"Fire Trucks: {resource_scale}",
        f"Ambulances: {max(1, resource_scale - 1)}",
        f"Response Teams: {max(1, resource_scale - 1)}",
        "- Dispatch emergency response team",
        "- Continue hotspot monitoring",
        "- Notify district authorities",
        "- Issue precautionary alerts",
        "- Review nearby infrastructure status",
        "----------------------------------------------------",
        "AI DECISION SUMMARY",
        f"AgniDrishti predicts a {likelihood} likelihood of",
        "incident escalation within the next 48 hours.",
        "Immediate monitoring and response coordination are recommended.",
        "====================================================",
        "END OF REPORT"
    ]
    output = create_pdf(
        lines
    )
    return send_file(
        BytesIO(output),
        mimetype="application/pdf",
        as_attachment=True,
        download_name="agnidrishti_incident_report.pdf"
    )


@app.route("/decision-support")
def decision_support():

    df = probability_data()
    population = pd.read_csv(DATA_DIR / "population_dataset.csv")
    weather = pd.read_csv(DATA_DIR / "weather_dataset.csv")
    threats = pd.read_csv(DATA_DIR / "infrastructure_threat_dataset.csv")
    for related in [population, weather, threats]:
        additions = [
            column for column in related.columns
            if column not in df.columns or column.endswith("_threat")
        ]
        df = df.merge(
            related[["latitude", "longitude"] + additions].drop_duplicates(
                ["latitude", "longitude"]
            ),
            on=["latitude", "longitude"],
            how="left"
        )

    ranked = df.sort_values("georisk_score", ascending=False).head(10).copy()
    critical = ranked[ranked["escalation_level"] == "CRITICAL"].head(5).copy()
    infrastructure = df[
        (df["hospital_km"] < 20)
        | (df["industry_km"] < 20)
        | (df["factory_km"] < 20)
        | (df["powerplant_km"] < 20)
    ].sort_values(by="escalation_probability", ascending=False).head(5)

    def action_for_risk(risk):
        if risk > 100:
            return "Emergency Response"
        if risk > 80:
            return "Deploy Field Team"
        if risk > 60:
            return "Increase Monitoring"
        return "Routine Surveillance"

    ranked["priority_action"] = ranked["georisk_score"].apply(action_for_risk)
    critical["priority_action"] = critical["georisk_score"].apply(action_for_risk)
    infrastructure["priority_action"] = infrastructure["georisk_score"].apply(action_for_risk)
    max_population = int(ranked["population_exposed"].max()) if not ranked.empty else 0
    emergency_teams = 5 if max_population > 40000 else 3 if max_population > 20000 else 1

    return render_template(
        "decision_support.html",
        ranked_incidents=ranked.to_dict("records"),
        critical_incidents=critical.to_dict("records"),
        infrastructure= infrastructure.to_dict("records"),
        recommended_action=(
            "Dispatch emergency response and notify infrastructure operators."
            if not critical.empty
            else "Continue monitoring and collect additional hotspot history."
        ),
        notification_status=("Active critical notifications" if not critical.empty else "No critical notifications"),
        escalation_probability=(round(df["escalation_probability"].mean(), 2) if not df.empty else 0),
        emergency_teams=emergency_teams,
        medical_units=max(1, emergency_teams - 1),
        monitoring_drones=max(1, emergency_teams - 2),
        police_support="Yes" if not critical.empty else "Standby",
        critical_alert=not critical.empty,
        infrastructure_count=len(infrastructure),
        recommended_actions=[
            "Dispatch emergency response team",
            "Alert local authorities",
            "Secure nearby infrastructure",
            "Monitor escalation hourly"
        ],
        emergency_checklist=[
            "Verify incident coordinates",
            "Confirm field team availability",
            "Notify district control room",
            "Review hospital and power-plant proximity",
            "Log response decision"
        ]
    )


@app.route("/analytics")
def analytics():

    df = probability_data()
    risk_levels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    escalation_levels = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    event_counts = df["event_type"].value_counts()
    anomaly_counts = df["anomaly_status"].value_counts()

    weather_df = pd.read_csv(DATA_DIR / "weather_dataset.csv")
    population_df = pd.read_csv(DATA_DIR / "population_dataset.csv")
    threat_df = pd.read_csv(DATA_DIR / "infrastructure_threat_dataset.csv")
    threat_values = {"LOW": 25, "MEDIUM": 50, "HIGH": 75, "CRITICAL": 100}
    infrastructure_scores = threat_df[
        ["hospital_threat", "industry_threat", "factory_threat", "powerplant_threat"]
    ].replace(threat_values).mean(axis=1)

    historical = pd.read_csv(DATA_DIR / "historical_trends.csv")
    historical = historical.sort_values("date")
    trend_values = historical["avg_georisk"].astype(float).tolist()
    trend_labels = historical["date"].tolist()
    escalation_trend_values = {
        "critical": historical["critical"].astype(int).tolist(),
        "high": historical["high"].astype(int).tolist(),
        "medium": historical["medium"].astype(int).tolist()
    }
    if len(trend_values) >= 2:
        daily_delta = trend_values[-1] - trend_values[-2]
    else:
        daily_delta = 0
    predicted_values = [round(trend_values[-1] + daily_delta * step, 2) for step in range(1, 4)]
    predicted_labels = [f"Day +{step}" for step in range(1, 4)]

    insights = [
        f"Critical hotspot detected near {df.iloc[0]['latitude']}, {df.iloc[0]['longitude']}",
        f"Average escalation probability is {df['escalation_probability'].mean():.1f}%",
        f"Population exposure reaches {int(population_df['population_exposed'].max()):,} in the highest-risk record",
        "Infrastructure vulnerability is high" if infrastructure_scores.max() >= 75 else "Infrastructure vulnerability is currently low",
        "Immediate intervention recommended" if (df["escalation_probability"] > 80).any() else "Continue routine surveillance"
    ]

    return render_template(
        "analytics.html",
        risk_labels=["Low Risk", "Medium Risk", "High Risk", "Critical Risk"],
        risk_values=[int((df["risk_level"].str.upper() == level).sum()) for level in risk_levels],
        risk_category_labels=["Weather Risk", "Population Risk", "Infrastructure Risk", "Escalation Risk"],
        risk_category_values=[
            round(float(weather_df["weather_risk_score"].clip(lower=0).mean() / 35 * 100), 2),
            round(float(population_df["exposure_level"].map({"LOW": 25, "MEDIUM": 60, "HIGH": 100}).mean()), 2),
            round(float(infrastructure_scores.mean()), 2),
            round(float(df["escalation_probability"].mean()), 2)
        ],
        trend_labels=trend_labels,
        trend_values=trend_values,
        escalation_trend_values=escalation_trend_values,
        predicted_labels=predicted_labels,
        predicted_values=predicted_values,
        insights=insights,
        event_labels=event_counts.index.tolist(),
        event_values=event_counts.tolist(),
        escalation_labels=escalation_levels,
        escalation_values=[
            int((df["escalation_level"] == level).sum())
            for level in escalation_levels
        ],
        anomaly_labels=anomaly_counts.index.tolist(),
        anomaly_values=anomaly_counts.tolist()
    )


@app.route("/population")
def population():

    population_path = DATA_DIR / "population_dataset.csv"
    if population_path.exists():
        df = pd.read_csv(population_path)
    else:
        df = load_dataset("escalation_dataset.csv")
        np.random.seed(42)
        df["population_exposed"] = np.random.randint(1000, 50000, len(df))
        df["exposure_level"] = pd.cut(
            df["population_exposed"],
            bins=[-float("inf"), 10000, 30000, float("inf")],
            labels=["LOW", "MEDIUM", "HIGH"]
        ).astype(str)

    top_population = df.sort_values(
        by="population_exposed",
        ascending=False
    ).head(20)
    critical_exposure = df[df["population_exposed"] > 40000]

    return render_template(
        "population.html",
        population_data=top_population.to_dict("records"),
        total_population=int(df["population_exposed"].sum()),
        high_exposure_zones=int((df["exposure_level"] == "HIGH").sum()),
        critical_exposure_zones=len(critical_exposure),
        average_exposure=round(df["population_exposed"].mean()),
        chart_labels=[
            f"{row['latitude']}, {row['longitude']}"
            for _, row in top_population.head(10).iterrows()
        ],
        chart_values=top_population.head(10)["population_exposed"].tolist()
    )


@app.route("/weather")
def weather():

    weather_path = DATA_DIR / "weather_dataset.csv"
    if weather_path.exists():
        df = pd.read_csv(weather_path)
    else:
        df = pd.read_csv(DATA_DIR / "population_dataset.csv")
        np.random.seed(42)
        df["temperature"] = np.random.randint(25, 45, len(df))
        df["humidity"] = np.random.randint(20, 90, len(df))
        df["wind_speed"] = np.random.randint(5, 40, len(df))
        df["rainfall"] = np.random.randint(0, 50, len(df))
        df["weather_risk_score"] = (
            (df["temperature"] > 35) * 10
            + (df["humidity"] < 30) * 10
            + (df["wind_speed"] > 20) * 15
            - (df["rainfall"] > 10) * 10
        )

    def risk_band(score):
        if score >= 25:
            return "HIGH"
        if score >= 15:
            return "MEDIUM"
        return "LOW"

    df["weather_risk_level"] = df["weather_risk_score"].apply(risk_band)
    ranked_weather = df.sort_values(
        by="weather_risk_score",
        ascending=False
    )
    bands = ["LOW", "MEDIUM", "HIGH"]

    return render_template(
        "weather.html",
        weather_data=ranked_weather.to_dict("records"),
        highest_temperature=int(df["temperature"].max()),
        highest_wind_speed=int(df["wind_speed"].max()),
        highest_rainfall=int(df["rainfall"].max()),
        highest_weather_risk=int(df["weather_risk_score"].max()),
        risk_labels=["Low", "Medium", "High"],
        risk_values=[int((df["weather_risk_level"] == band).sum()) for band in bands]
    )


@app.route("/threats")
def threats():

    threat_path = DATA_DIR / "infrastructure_threat_dataset.csv"
    if threat_path.exists():
        df = pd.read_csv(threat_path)
    else:
        df = pd.read_csv(DATA_DIR / "weather_dataset.csv")

        def threat_level(distance):
            if distance < 20:
                return "CRITICAL"
            elif distance < 50:
                return "HIGH"
            elif distance < 100:
                return "MEDIUM"
            return "LOW"

        for infrastructure in ["hospital", "industry", "factory", "powerplant"]:
            df[f"{infrastructure}_threat"] = df[
                f"{infrastructure}_km"
            ].apply(threat_level)

    threat_columns = [
        "hospital_threat",
        "industry_threat",
        "factory_threat",
        "powerplant_threat"
    ]
    severity = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
    df["overall_threat"] = df[threat_columns].apply(
        lambda row: max(row, key=lambda value: severity.get(value, 0)),
        axis=1
    )
    threat_counts = {
        level: int((df["overall_threat"] == level).sum())
        for level in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    }

    return render_template(
        "threats.html",
        threat_data=df.sort_values(
            by="georisk_score",
            ascending=False
        ).to_dict("records"),
        total_incidents=len(df),
        critical_threats=threat_counts["CRITICAL"],
        high_threats=threat_counts["HIGH"],
        medium_threats=threat_counts["MEDIUM"],
        low_threats=threat_counts["LOW"]
    )


@app.route("/classification")
def classification():

    classification_path = DATA_DIR / "classified_events.csv"
    if classification_path.exists():
        df = pd.read_csv(classification_path)
    else:
        df = load_dataset("escalation_dataset.csv")

    return render_template(
        "classification.html",
        events=df.to_dict("records")
    )

if __name__ == "__main__":

    app.run(
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG", "0") == "1"
    )
