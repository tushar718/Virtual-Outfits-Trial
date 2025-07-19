from flask import Flask, request, jsonify
import subprocess

app = Flask(__name__)

@app.route('/try-on', methods=['POST'])
def try_on():
    item_name = request.json.get('item_name')
    if not item_name:
        return jsonify({"error": "No item name provided"}), 400
    
    
    try:
        subprocess.Popen(["python", "try_on.py", item_name])
        return jsonify({"message": "Try-on started"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
