from flask import Flask
from routes.kyc_routes import kyc_bp

def create_app():
    app = Flask(__name__)
    app.register_blueprint(kyc_bp, url_prefix='/kyc')
    return app



if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
