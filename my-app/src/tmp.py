from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/run3u', methods=['POST'])
def run_script():
    # Execute your Python script logic here
    return jsonify({'answer': 'lotus', 
                    'model1correct': 1, 
                    'model1incorrect': 0,
                    'model2correct': 0,
                    'model2incorrect': 1,
                    'model3correct': 1,
                    'model3incorrect': 1000,
                    'filename': '/daisies/daisys1.jpg',
                    })

if __name__ == '__main__':
    app.run()


# Randomly generate an image
# Correct Answer
# Model Guess