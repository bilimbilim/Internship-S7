from flask import Flask
from app_llmops_routes import llmops_bp

app = Flask(__name__)

# 🔹 Enregistre les routes LLMOps
app.register_blueprint(llmops_bp)

# 🔹 Route test (optionnel)
@app.route("/")
def home():
    return "Serveur LLMOps actif "


if __name__ == "__main__":
    app.run(debug=True)