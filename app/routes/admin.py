from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, abort
from flask_login import login_required, current_user
from collections import defaultdict
from app import db
from app.models import User, Prediction, DoctorPatient

admin_bp = Blueprint('admin', __name__, url_prefix='/admin')


def _require_admin():
    if current_user.role != 'admin':
        abort(403)


@admin_bp.route('/dashboard')
@login_required
def dashboard():
    _require_admin()

    total_users = User.query.count()
    patients = User.query.filter_by(role='patient').all()
    doctors = User.query.filter_by(role='doctor').all()
    admins = User.query.filter_by(role='admin').all()
    total_predictions = Prediction.query.count()
    high_risk_predictions = Prediction.query.filter_by(risk_level='High').count()

    recent_users = User.query.order_by(User.created_at.desc()).limit(5).all()
    recent_predictions = Prediction.query.order_by(Prediction.created_at.desc()).limit(10).all()

    unassigned_patients = []
    for p in patients:
        if not DoctorPatient.query.filter_by(patient_id=p.id).first():
            unassigned_patients.append(p)

    return render_template('admin/dashboard.html',
                           total_users=total_users,
                           num_patients=len(patients),
                           num_doctors=len(doctors),
                           num_admins=len(admins),
                           total_predictions=total_predictions,
                           high_risk=high_risk_predictions,
                           recent_users=recent_users,
                           recent_predictions=recent_predictions,
                           unassigned_patients=unassigned_patients,
                           patients=patients,
                           doctors=doctors)


@admin_bp.route('/users')
@login_required
def users():
    _require_admin()
    all_users = User.query.order_by(User.created_at.desc()).all()
    doctors = User.query.filter_by(role='doctor').all()
    return render_template('admin/users.html', users=all_users, doctors=doctors)


@admin_bp.route('/users/create', methods=['POST'])
@login_required
def create_user():
    _require_admin()
    username = request.form.get('username', '').strip()
    email = request.form.get('email', '').strip()
    password = request.form.get('password', '').strip()
    role = request.form.get('role', 'patient')

    if not username or not email or not password:
        flash('All fields are required.', 'danger')
        return redirect(url_for('admin.users'))

    if role not in ('patient', 'doctor', 'admin'):
        flash('Invalid role.', 'danger')
        return redirect(url_for('admin.users'))

    if User.query.filter_by(email=email).first():
        flash(f'Email {email} is already registered.', 'danger')
        return redirect(url_for('admin.users'))

    if User.query.filter_by(username=username).first():
        flash(f'Username {username} is already taken.', 'danger')
        return redirect(url_for('admin.users'))

    user = User(username=username, email=email, role=role)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()
    flash(f'User "{username}" created successfully as {role}.', 'success')
    return redirect(url_for('admin.users'))


@admin_bp.route('/users/<int:user_id>/assign', methods=['POST'])
@login_required
def assign_doctor(user_id):
    _require_admin()
    patient = User.query.get_or_404(user_id)
    if patient.role != 'patient':
        flash('User is not a patient.', 'danger')
        return redirect(url_for('admin.users'))

    doctor_id = request.form.get('doctor_id', type=int)
    if not doctor_id:
        flash('Please select a doctor.', 'danger')
        return redirect(url_for('admin.users'))

    doctor = User.query.get_or_404(doctor_id)
    if doctor.role != 'doctor':
        flash('Selected user is not a doctor.', 'danger')
        return redirect(url_for('admin.users'))

    existing = DoctorPatient.query.filter_by(
        doctor_id=doctor_id, patient_id=user_id).first()
    if existing:
        flash('Patient is already assigned to this doctor.', 'info')
    else:
        assignment = DoctorPatient(doctor_id=doctor_id, patient_id=user_id)
        db.session.add(assignment)
        db.session.commit()
        flash(f'Patient {patient.username} assigned to Dr. {doctor.username}.', 'success')

    return redirect(url_for('admin.users'))


@admin_bp.route('/users/<int:user_id>/role', methods=['POST'])
@login_required
def change_role(user_id):
    _require_admin()
    if user_id == current_user.id:
        flash('You cannot change your own role.', 'danger')
        return redirect(url_for('admin.users'))

    user = User.query.get_or_404(user_id)
    new_role = request.form.get('role')
    if new_role not in ('patient', 'doctor', 'admin'):
        flash('Invalid role.', 'danger')
        return redirect(url_for('admin.users'))

    user.role = new_role
    db.session.commit()
    flash(f'Role updated to {new_role} for {user.username}.', 'success')
    return redirect(url_for('admin.users'))


@admin_bp.route('/users/<int:user_id>/delete', methods=['POST'])
@login_required
def delete_user(user_id):
    _require_admin()
    if user_id == current_user.id:
        flash('You cannot delete your own account.', 'danger')
        return redirect(url_for('admin.users'))

    user = User.query.get_or_404(user_id)
    DoctorPatient.query.filter(
        (DoctorPatient.doctor_id == user_id) | (DoctorPatient.patient_id == user_id)
    ).delete()
    Prediction.query.filter_by(patient_id=user_id).delete()
    db.session.delete(user)
    db.session.commit()
    flash(f'User {user.username} deleted.', 'success')
    return redirect(url_for('admin.users'))


@admin_bp.route('/stats/json')
@login_required
def stats_json():
    _require_admin()

    role_counts = {
        'patient': User.query.filter_by(role='patient').count(),
        'doctor': User.query.filter_by(role='doctor').count(),
        'admin': User.query.filter_by(role='admin').count(),
    }

    risk_counts = {
        'Low': Prediction.query.filter_by(risk_level='Low').count(),
        'Medium': Prediction.query.filter_by(risk_level='Medium').count(),
        'High': Prediction.query.filter_by(risk_level='High').count(),
    }

    all_preds = Prediction.query.order_by(Prediction.created_at.asc()).all()
    monthly = defaultdict(int)
    for p in all_preds:
        key = p.created_at.strftime('%Y-%m')
        monthly[key] += 1
    trend = [{'month': k, 'count': v} for k, v in sorted(monthly.items())]

    all_users = User.query.order_by(User.created_at.asc()).all()
    user_monthly = defaultdict(int)
    for u in all_users:
        key = u.created_at.strftime('%Y-%m')
        user_monthly[key] += 1
    user_trend = [{'month': k, 'count': v} for k, v in sorted(user_monthly.items())]

    return jsonify({
        'role_counts': role_counts,
        'risk_counts': risk_counts,
        'prediction_trend': trend,
        'user_trend': user_trend,
    })
