from flask import Flask, request, jsonify
from recommendation_model import generate_learning_path  # Import your recommendation model

app = Flask(__name__)

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    user_input = data.get('query')
    if not user_input:
        return jsonify({"error": "Query is required"}), 400

    try:
        learning_path = generate_learning_path(user_input)
        return jsonify({"learning_path": learning_path})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)