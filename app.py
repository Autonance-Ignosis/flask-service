from flask import Flask
from routes.kyc_routes import kyc_bp
# from routes.loan_routes import loan_bp
from routes.mandate_routes import mandate_bp



def create_app():
    app = Flask(__name__)
    # CORS(app)
    app.register_blueprint(kyc_bp, url_prefix='/kyc')
    # app.register_blueprint(loan_bp, url_prefix='/loan')
    app.register_blueprint(mandate_bp, url_prefix='/mandate')
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
