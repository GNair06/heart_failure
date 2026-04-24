import os
import pickle
import numpy as np
from datetime import datetime
from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, abort
from flask_login import login_required, current_user
from app import db
from app.models import Prediction, DoctorPatient

patient_bp = Blueprint('patient', __name__, url_prefix='/patient')

ML_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ml')


def _require_patient():
    if current_user.role != 'patient':
        abort(403)


def _load_model():
    model_path = os.path.join(ML_DIR, 'best_model.pkl')
    scaler_path = os.path.join(ML_DIR, 'scaler.pkl')
    if not os.path.exists(model_path):
        return None, None
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler


def _risk_level(prob):
    if prob < 0.35:
        return 'Low'
    elif prob < 0.65:
        return 'Medium'
    return 'High'


@patient_bp.route('/dashboard')
@login_required
def dashboard():
    _require_patient()
    predictions = (Prediction.query
                   .filter_by(patient_id=current_user.id)
                   .order_by(Prediction.created_at.desc())
                   .all())

    total = len(predictions)
    high_risk = sum(1 for p in predictions if p.risk_level == 'High')
    latest = predictions[0] if predictions else None

    doctors = (DoctorPatient.query
               .filter_by(patient_id=current_user.id)
               .all())

    return render_template('patient/dashboard.html',
                           predictions=predictions,
                           total=total,
                           high_risk=high_risk,
                           latest=latest,
                           doctors=doctors)


@patient_bp.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    _require_patient()

    if request.method == 'POST':
        model, scaler = _load_model()
        if model is None:
            flash('Prediction model not trained yet. Please contact the administrator.', 'warning')
            return redirect(url_for('patient.dashboard'))

        try:
            features = [
                float(request.form['age']),
                int(request.form['anaemia']),
                float(request.form['creatinine_phosphokinase']),
                int(request.form['diabetes']),
                float(request.form['ejection_fraction']),
                int(request.form['high_blood_pressure']),
                float(request.form['platelets']),
                float(request.form['serum_creatinine']),
                float(request.form['serum_sodium']),
                int(request.form['sex']),
                int(request.form['smoking']),
                float(request.form['time']),
            ]
        except (KeyError, ValueError):
            flash('Please fill in all fields with valid values.', 'danger')
            return render_template('patient/predict.html')

        X = np.array(features).reshape(1, -1)
        X_scaled = scaler.transform(X)
        result = int(model.predict(X_scaled)[0])
        prob = float(model.predict_proba(X_scaled)[0][1])
        risk = _risk_level(prob)

        pred = Prediction(
            patient_id=current_user.id,
            age=features[0], anaemia=features[1],
            creatinine_phosphokinase=features[2], diabetes=features[3],
            ejection_fraction=features[4], high_blood_pressure=features[5],
            platelets=features[6], serum_creatinine=features[7],
            serum_sodium=features[8], sex=features[9],
            smoking=features[10], time=features[11],
            result=result, probability=round(prob, 4),
            risk_level=risk,
        )
        db.session.add(pred)
        db.session.commit()

        flash(f'Prediction complete — Risk Level: {risk} ({prob*100:.1f}%)', 'success')
        return redirect(url_for('patient.result', pred_id=pred.id))

    return render_template('patient/predict.html')


@patient_bp.route('/result/<int:pred_id>')
@login_required
def result(pred_id):
    _require_patient()
    pred = Prediction.query.get_or_404(pred_id)
    if pred.patient_id != current_user.id:
        abort(403)
    return render_template('patient/result.html', pred=pred)


@patient_bp.route('/history/json')
@login_required
def history_json():
    _require_patient()
    preds = (Prediction.query
             .filter_by(patient_id=current_user.id)
             .order_by(Prediction.created_at.asc())
             .all())
    return jsonify([p.to_dict() for p in preds])


@patient_bp.route('/profile')
@login_required
def profile():
    _require_patient()
    predictions = Prediction.query.filter_by(patient_id=current_user.id).all()
    doctors = DoctorPatient.query.filter_by(patient_id=current_user.id).all()
    return render_template('patient/profile.html',
                           predictions=predictions,
                           doctors=doctors)
