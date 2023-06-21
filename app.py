from flask import Flask, request
import subprocess

app = Flask(__name__)
    

@app.route('/execute-script',  methods=['GET', 'POST'])

def execute_script():
    try:
        # Run the Python script using subprocess module
        subprocess.run(['python', 'Model-Predictions.py'], check=True)
        return 'Script executed successfully'
    except Exception as e:
        return f'Error executing script: {str(e)}'
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)