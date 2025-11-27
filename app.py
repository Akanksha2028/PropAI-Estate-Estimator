from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load Model
with open('models/rent_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        
        city = request.form['city']
        furnishing = request.form['furnishing']
        bhk = float(request.form['bhk'])           
        bathroom = float(request.form['bathroom']) 
        floor_input = request.form['floor']        
        size = float(request.form['size'])        

        floor_level = 0
        if 'Ground' in floor_input:
            floor_level = 0
        else:
            try: floor_level = int(floor_input)
            except: floor_level = 0

        
        input_data = pd.DataFrame([[bhk, size, floor_level, city, furnishing, bathroom]],
                                  columns=['BHK', 'Size', 'Floor_Level', 'City', 'Furnishing Status', 'Bathroom'])

        prediction = model.predict(input_data)[0]
        final_price = round(prediction, 2)

        price_per_sqft = final_price / size
        
        if price_per_sqft < 15:
            verdict = "💎 STEAL DEAL (Great Price!)"
            css_class = "result-good"
        elif price_per_sqft < 45:
            verdict = "⚖️ FAIR MARKET VALUE"
            css_class = "result-fair"
        else:
            verdict = "⚠️ PREMIUM / EXPENSIVE"
            css_class = "result-bad"

        return render_template('index.html', 
                               prediction_text=f"₹ {final_price:,.0f}",
                               verdict_text=verdict,
                               result_class=css_class)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
