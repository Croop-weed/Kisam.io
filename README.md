# 🌱 Kisan.io – Intelligent Crop Recommendation System

**Kisan.io** is a Flask-based web application that helps farmers and agriculture enthusiasts determine the best crops to grow based on **soil nutrients, location, and live weather data**.  
It uses **machine learning (Random Forest)** and integrates with **NASA POWER API** and **OpenStreetMap** for weather and geolocation data.

---

## 🚀 Features

- 🌍 **Auto-detects your location and state** using IP-based geolocation.  
- ☁️ **Fetches live climate data** (temperature, humidity, rainfall) from NASA POWER API.  
- 🌾 **Predicts top 3 suitable crops** for your soil and environment using a trained Random Forest model.  
- 💰 **Displays estimated market price** (from dataset averages).  
- 🧠 Machine learning model trained on **merged crop and soil datasets**.  

---

## 🧩 Tech Stack

| Category | Technologies Used |
|-----------|------------------|
| **Frontend** | HTML, CSS, Jinja2 Templates |
| **Backend** | Python, Flask |
| **Machine Learning** | scikit-learn (Random Forest Classifier) |
| **Data Handling** | Pandas, LabelEncoder |
| **APIs** | NASA POWER API, OpenStreetMap (Nominatim), Geocoder |
| **Dataset** | `merged_dataset.csv` (contains state, weather, NPK, pH, and crop data) |

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/<your-username>/Kisan.io.git
cd Kisan.io
