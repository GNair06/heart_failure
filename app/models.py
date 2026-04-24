from datetime import datetime
from app import db, bcrypt


class User(db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(20), default='patient', nullable=False)  # patient | doctor | admin
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    predictions = db.relationship('Prediction', foreign_keys='Prediction.patient_id',
                                  backref='patient', lazy='dynamic')

    def set_password(self, password):
        self.password_hash = bcrypt.generate_password_hash(password).decode('utf-8')

    def check_password(self, password):
        return bcrypt.check_password_hash(self.password_hash, password)

    # Flask-Login interface
    @property
    def is_authenticated(self):
        return True

    @property
    def is_active(self):
        return True

    @property
    def is_anonymous(self):
        return False

    def get_id(self):
        return str(self.id)

    def __repr__(self):
        return f'<User {self.username} ({self.role})>'


class DoctorPatient(db.Model):
    __tablename__ = 'doctor_patient'

    doctor_id = db.Column(db.Integer, db.ForeignKey('users.id'), primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('users.id'), primary_key=True)
    assigned_at = db.Column(db.DateTime, default=datetime.utcnow)

    doctor = db.relationship('User', foreign_keys=[doctor_id], backref='assigned_patients')
    patient = db.relationship('User', foreign_keys=[patient_id], backref='assigned_doctors')


class Prediction(db.Model):
    __tablename__ = 'predictions'

    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)

    # Clinical features
    age = db.Column(db.Float, nullable=False)
    anaemia = db.Column(db.Integer, nullable=False)
    creatinine_phosphokinase = db.Column(db.Float, nullable=False)
    diabetes = db.Column(db.Integer, nullable=False)
    ejection_fraction = db.Column(db.Float, nullable=False)
    high_blood_pressure = db.Column(db.Integer, nullable=False)
    platelets = db.Column(db.Float, nullable=False)
    serum_creatinine = db.Column(db.Float, nullable=False)
    serum_sodium = db.Column(db.Float, nullable=False)
    sex = db.Column(db.Integer, nullable=False)
    smoking = db.Column(db.Integer, nullable=False)
    time = db.Column(db.Float, nullable=False)

    # Results
    result = db.Column(db.Integer, nullable=False)          # 0 = survived, 1 = died
    probability = db.Column(db.Float, nullable=False)
    risk_level = db.Column(db.String(10), nullable=False)   # Low | Medium | High

    # Doctor interaction
    doctor_notes = db.Column(db.Text, nullable=True)
    reviewed_by = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    reviewer = db.relationship('User', foreign_keys=[reviewed_by], backref='reviewed_predictions')

    def to_dict(self):
        return {
            'id': self.id,
            'age': self.age,
            'anaemia': self.anaemia,
            'creatinine_phosphokinase': self.creatinine_phosphokinase,
            'diabetes': self.diabetes,
            'ejection_fraction': self.ejection_fraction,
            'high_blood_pressure': self.high_blood_pressure,
            'platelets': self.platelets,
            'serum_creatinine': self.serum_creatinine,
            'serum_sodium': self.serum_sodium,
            'sex': self.sex,
            'smoking': self.smoking,
            'time': self.time,
            'result': self.result,
            'probability': self.probability,
            'risk_level': self.risk_level,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M'),
        }

    def __repr__(self):
        return f'<Prediction id={self.id} patient={self.patient_id} result={self.result}>'
