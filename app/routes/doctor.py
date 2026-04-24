from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, abort
from flask_login import login_required, current_user
from app import db
from app.models import User, Prediction, DoctorPatient

doctor_bp = Blueprint('doctor', __name__, url_prefix='/doctor')


def _require_doctor():
    if current_user.role != 'doctor':
        abort(403)


@doctor_bp.route('/dashboard')
@login_required
def dashboard():
    _require_doctor()

    assignments = DoctorPatient.query.filter_by(doctor_id=current_user.id).all()
    patient_ids = [a.patient_id for a in assignments]

    patients = User.query.filter(User.id.in_(patient_ids)).all() if patient_ids else []

    all_predictions = (Prediction.query
                       .filter(Prediction.patient_id.in_(patient_ids))
                       .order_by(Prediction.created_at.desc())
                       .all()) if patient_ids else []

    total_patients = len(patients)
    total_predictions = len(all_predictions)
    high_risk = sum(1 for p in all_predictions if p.risk_level == 'High')
    pending_review = sum(1 for p in all_predictions if not p.doctor_notes)

    # Latest predictions per patient for overview
    latest_per_patient = {}
    for pred in all_predictions:
        if pred.patient_id not in latest_per_patient:
            latest_per_patient[pred.patient_id] = pred

    patient_overview = []
    for pat in patients:
        latest = latest_per_patient.get(pat.id)
        patient_overview.append({'user': pat, 'latest': latest})

    recent_predictions = all_predictions[:10]

    return render_template('doctor/dashboard.html',
                           patient_overview=patient_overview,
                           recent_predictions=recent_predictions,
                           total_patients=total_patients,
                           total_predictions=total_predictions,
                           high_risk=high_risk,
                           pending_review=pending_review)


@doctor_bp.route('/patient/<int:patient_id>')
@login_required
def patient_detail(patient_id):
    _require_doctor()

    assignment = DoctorPatient.query.filter_by(
        doctor_id=current_user.id, patient_id=patient_id).first_or_404()

    patient = User.query.get_or_404(patient_id)
    predictions = (Prediction.query
                   .filter_by(patient_id=patient_id)
                   .order_by(Prediction.created_at.desc())
                   .all())

    return render_template('doctor/patient_detail.html',
                           patient=patient,
                           predictions=predictions)


@doctor_bp.route('/patient/<int:patient_id>/note/<int:pred_id>', methods=['POST'])
@login_required
def add_note(patient_id, pred_id):
    _require_doctor()

    DoctorPatient.query.filter_by(
        doctor_id=current_user.id, patient_id=patient_id).first_or_404()

    pred = Prediction.query.get_or_404(pred_id)
    if pred.patient_id != patient_id:
        abort(403)

    note = request.form.get('note', '').strip()
    if note:
        pred.doctor_notes = note
        pred.reviewed_by = current_user.id
        db.session.commit()
        flash('Note saved successfully.', 'success')
    else:
        flash('Note cannot be empty.', 'danger')

    return redirect(url_for('doctor.patient_detail', patient_id=patient_id))


@doctor_bp.route('/stats/json')
@login_required
def stats_json():
    _require_doctor()

    assignments = DoctorPatient.query.filter_by(doctor_id=current_user.id).all()
    patient_ids = [a.patient_id for a in assignments]

    if not patient_ids:
        return jsonify({'risk_counts': {}, 'prediction_trend': [], 'patient_risk': []})

    predictions = Prediction.query.filter(Prediction.patient_id.in_(patient_ids)).all()

    risk_counts = {'Low': 0, 'Medium': 0, 'High': 0}
    for p in predictions:
        risk_counts[p.risk_level] = risk_counts.get(p.risk_level, 0) + 1

    from collections import defaultdict
    monthly = defaultdict(int)
    for p in predictions:
        key = p.created_at.strftime('%Y-%m')
        monthly[key] += 1
    trend = [{'month': k, 'count': v} for k, v in sorted(monthly.items())]

    patients = User.query.filter(User.id.in_(patient_ids)).all()
    patient_risk = []
    for pat in patients:
        pat_preds = [p for p in predictions if p.patient_id == pat.id]
        if pat_preds:
            latest = max(pat_preds, key=lambda x: x.created_at)
            patient_risk.append({
                'name': pat.username,
                'risk': latest.risk_level,
                'prob': latest.probability,
            })

    return jsonify({
        'risk_counts': risk_counts,
        'prediction_trend': trend,
        'patient_risk': patient_risk,
    })
