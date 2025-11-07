from flask import Flask, render_template, request
import pandas as pd
import geocoder
import requests
import calendar
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load dataset and train model once at startup
df = pd.read_csv("merged_dataset.csv")
le_state = LabelEncoder()
df["state_encoded"] = le_state.fit_transform(df["state"])
le_label = LabelEncoder()
df["label_encoded"] = le_label.fit_transform(df["label"])

X = df[["state_encoded", "rainfall", "humidity", "temperature", "N", "P", "K", "ph"]]
y = df["label_encoded"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Form data
            N = float(request.form["N"])
            P = float(request.form["P"])
            K = float(request.form["K"])
            pH = float(request.form["pH"])

            # Get geolocation
            g = geocoder.ip('me')
            lat, lon = g.latlng if g.latlng else (28.6139, 77.2090)

            # Get state using reverse geocoding
            url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json"
            headers = {"User-Agent": "Kisan.io/1.0"}
            location_data = requests.get(url, headers=headers).json()
            state = location_data["address"].get("state", "Delhi")

            # Get weather data
            weather_url = f"https://power.larc.nasa.gov/api/temporal/climatology/point"
            params = {
                "parameters": "T2M,RH2M,PRECTOTCORR",
                "community": "AG",
                "longitude": lon,
                "latitude": lat,
                "format": "JSON"
            }
            response = requests.get(weather_url, params=params).json()
            data = response["properties"]["parameter"]
            months = list(data["T2M"].keys())[:6]

            avg_temp = sum(data["T2M"][m] for m in months) / 6
            avg_humidity = (sum(data["RH2M"][m] for m in months) / 6) * 2
            avg_rainfall = sum(
                data["PRECTOTCORR"][m] * calendar.monthrange(2022, i + 1)[1]
                for i, m in enumerate(months)
            )

            # 🛑 Try transforming the state (may raise ValueError)
            try:
                state_encoded = le_state.transform([state])[0]
            except ValueError:
                return render_template("error.html", message=f"Sorry, the model doesn't support predictions for your location: '{state}'.")

            # Prepare input
            input_data = pd.DataFrame([{
                "state_encoded": state_encoded,
                "rainfall": avg_rainfall,
                "humidity": avg_humidity,
                "temperature": avg_temp,
                "N": N,
                "P": P,
                "K": K,
                "ph": pH
            }])

            # Predict
            probs = clf.predict_proba(input_data)[0]
            top_indices = probs.argsort()[-3:][::-1]
            top_crops = le_label.inverse_transform(top_indices)

            results = []
            for crop in top_crops:
                price = df.loc[df["label"] == crop, "modal_price"].mean()
                results.append((crop, price))

            results = sorted(results, key=lambda x: x[1], reverse=True)

            return render_template("index.html", results=results)

        except Exception as e:
            return render_template("error.html", message=f"Unexpected error: {str(e)}")

    return render_template("index.html", results=None)
    if request.method == "POST":
        # User inputs from form
        N = float(request.form["N"])
        P = float(request.form["P"])
        K = float(request.form["K"])
        pH = float(request.form["pH"])

        # Get geolocation
        g = geocoder.ip('me')
        lat, lon = g.latlng if g.latlng else (28.6139, 77.2090)  # fallback: Delhi

        # Get state using reverse geocoding
        url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json"
        headers = {"User-Agent": "Kisan.io/1.0 (contact@example.com)"}
        response = requests.get(url, headers=headers).json()
        state = response["address"].get("state", "Delhi")

        # Get climate data from NASA POWER API
        weather_url = f"https://power.larc.nasa.gov/api/temporal/climatology/point"
        params = {
            "parameters": "T2M,RH2M,PRECTOTCORR",
            "community": "AG",
            "longitude": lon,
            "latitude": lat,
            "format": "JSON"
        }
        response = requests.get(weather_url, params=params).json()
        data = response["properties"]["parameter"]

        months = list(data["T2M"].keys())[:6]

        avg_temp = sum(data["T2M"][m] for m in months) / 6
        avg_humidity = (sum(data["RH2M"][m] for m in months) / 6) * 2
        rainfall_vals = [
            data["PRECTOTCORR"][m] * calendar.monthrange(2022, i + 1)[1]
            for i, m in enumerate(months)
        ]
        avg_rainfall = sum(rainfall_vals)

        # Prepare input
        input_data = pd.DataFrame([{
            "state_encoded": le_state.transform([state])[0],
            "rainfall": avg_rainfall,
            "humidity": avg_humidity,
            "temperature": avg_temp,
            "N": N,
            "P": P,
            "K": K,
            "ph": pH
        }])

        # Prediction
        probs = clf.predict_proba(input_data)[0]
        top_indices = probs.argsort()[-3:][::-1]
        top_crops = le_label.inverse_transform(top_indices)

        # Get average modal prices
        results = []
        for crop in top_crops:
            price = df.loc[df["label"] == crop, "modal_price"].mean()
            results.append((crop, price))

        results = sorted(results, key=lambda x: x[1], reverse=True)

        return render_template("index.html", results=results)

    return render_template("index.html", results=None)

if __name__ == "__main__":
    app.run(debug=True)
