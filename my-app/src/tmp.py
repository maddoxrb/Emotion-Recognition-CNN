from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/run-script', methods=['POST'])
def run_script():
    # Execute your Python script logic here
    print("cat")
    return jsonify({'model1corrrect': int(100)})

@app.route('/run-script-dog', methods=['POST'])
def run_scriptdog():
    # Execute your Python script logic here
    print("dog")
    return str(5)

if __name__ == '__main__':
    app.run()


# Randomly generate an image
# Correct Answer
# Model Guess