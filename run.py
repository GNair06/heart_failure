from flask import redirect, url_for
from app import create_app, db
from app.models import User

app = create_app()

@app.route('/')
def index():
    return redirect(url_for('auth.login'))

with app.app_context():
    db.create_all()
    if not User.query.filter_by(role='admin').first():
        admin = User(username='admin', email='admin@heartcare.com', role='admin')
        admin.set_password('admin123')
        db.session.add(admin)
        db.session.commit()
        print("Default admin created — email: admin@heartcare.com | password: admin123")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
