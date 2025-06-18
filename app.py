from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import os
import uuid
from PIL import Image
import io

# Initialize FastAPI app
app = FastAPI(title="Gaugehaus Real Estate Price Prediction API")

# Mount static directory to serve images
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load model, encoders, and dataset
try:
    model = joblib.load('model.joblib')
    label_encoders = joblib.load('encoders.joblib')
    df_cleaned = pd.read_csv('cleaned_data.csv')
except FileNotFoundError as e:
    raise FileNotFoundError(f"Missing file: {str(e)}. Ensure 'model.joblib', 'encoders.joblib', and 'cleaned_data.csv' are in the project root.")

# Verify required columns in cleaned_data.csv
required_columns = ['City', 'Price_per_sqm']
for col in required_columns:
    if col not in df_cleaned.columns:
        raise ValueError(f"Column '{col}' missing in cleaned_data.csv. Run prepare_cleaned_data.py to generate it.")

# Define input schema
class RealEstateInput(BaseModel):
    city: str
    property_type: str
    furnished: str
    delivery_term: str
    bedrooms: int
    bathrooms: int
    area: float
    level: int

# Define predict endpoint
@app.post("/predict")
async def predict(input_data: RealEstateInput):
    try:
        # Validate numerical inputs
        if input_data.bedrooms < 0:
            raise HTTPException(status_code=400, detail="Bedrooms must be non-negative")
        if input_data.bathrooms < 0:
            raise HTTPException(status_code=400, detail="Bathrooms must be non-negative")
        if input_data.area <= 0:
            raise HTTPException(status_code=400, detail="Area must be positive")
        if input_data.level < 0:
            raise HTTPException(status_code=400, detail="Level must be non-negative")

        # Encode categorical inputs
        categorical_inputs = {
            'City': input_data.city,
            'Type': input_data.property_type,
            'Furnished': input_data.furnished,
            'Delivery_Term': input_data.delivery_term
        }
        encoded_inputs = {}
        for col, value in categorical_inputs.items():
            if col not in label_encoders:
                raise HTTPException(status_code=500, detail=f"LabelEncoder for {col} not found")
            valid_options = label_encoders[col].classes_.tolist()
            if value not in valid_options:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid {col}: '{value}'. Valid options: {', '.join(valid_options)}"
                )
            encoded_inputs[col] = label_encoders[col].transform([value])[0]

        # Compute Price_per_sqm
        city_encoded = encoded_inputs['City']
        city_data = df_cleaned[df_cleaned["City"] == city_encoded]
        if city_data.empty:
            valid_cities = label_encoders['City'].classes_.tolist()
            raise HTTPException(
                status_code=400,
                detail=f"No data for city code {city_encoded} (decoded: {input_data.city}). Valid cities: {', '.join(valid_cities)}"
            )
        city_avg_price_per_sqm = city_data["Price_per_sqm"].mean()
        if np.isnan(city_avg_price_per_sqm) or not np.isfinite(city_avg_price_per_sqm):
            # Fallback to a default Price_per_sqm (based on dataset median)
            default_price_per_sqm = df_cleaned["Price_per_sqm"].median()
            if np.isnan(default_price_per_sqm) or not np.isfinite(default_price_per_sqm):
                raise HTTPException(
                    status_code=500,
                    detail="Cannot compute Price_per_sqm: invalid data in cleaned_data.csv"
                )
            city_avg_price_per_sqm = default_price_per_sqm

        # Create input DataFrame
        input_df = pd.DataFrame({
            'Type': [encoded_inputs['Type']],
            'Bedrooms': [input_data.bedrooms],
            'Bathrooms': [input_data.bathrooms],
            'Area': [input_data.area],
            'Furnished': [encoded_inputs['Furnished']],
            'Level': [input_data.level],
            'Delivery_Term': [encoded_inputs['Delivery_Term']],
            'City': [encoded_inputs['City']],
            'Price_per_sqm': [city_avg_price_per_sqm]
        })

        # Make prediction
        predicted_price = model.predict(input_df)[0]
        if predicted_price < 0:
            raise HTTPException(status_code=500, detail="Predicted price is negative, which is invalid")

        # Initialize SHAP explainer
        explainer = shap.TreeExplainer(model)

        # Generate SHAP explanation
        shap_values = explainer.shap_values(input_df)

        # Create SHAP waterfall plot
        plt.figure(figsize=(10, 5))
        shap.plots.waterfall(shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=input_df.iloc[0],
            feature_names=input_df.columns.tolist()
        ), show=False)

        # Save plot to buffer as PNG
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
        plt.close()
        buf.seek(0)

        # Convert PNG (RGBA) to JPEG (RGB)
        img = Image.open(buf)
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        unique_id = str(uuid.uuid4())
        jpeg_path = f"static/shap_explanation_{unique_id}.jpeg"
        img.save(jpeg_path, "JPEG", quality=95)
        image_url = f"/static/shap_explanation_{unique_id}.jpeg"

        # Generate textual description
        description = "Factors influencing the predicted price:\n"
        for feature_name, shap_value, feature_value in zip(input_df.columns, shap_values[0], input_df.iloc[0]):
            direction = "increased" if shap_value > 0 else "decreased"
            description += f"- {feature_name}: Value = {feature_value}, {direction} price by {abs(shap_value):.2f} units.\n"

        # Return prediction and explanations
        return {
            "predicted_price": float(predicted_price),
            "image_url": image_url,
            "factors_description": description
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Input error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)