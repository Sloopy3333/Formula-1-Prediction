from flask import Flask,render_template,request
import predictor


app = Flask(__name__)

@app.route('/')
def get_input():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict_position():
    driver = request.form['driver']
    constructor = request.form['constructor']
    quali = request.form['grid']
    circuit = request.form['circuit']
    my_prediction = predictor.pred(driver,constructor,quali,circuit)
    return render_template('index.html',prediction=my_prediction)
if __name__ == '__main__':
    app.run(debug=False)

# prediction = predictor.pred('British Grand Prix',3,'Red Bull','Max Verstappen')
# print(prediction)


