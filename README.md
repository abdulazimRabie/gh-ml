# Real Estate Price Prediction API

This project is a FastAPI-based API that predicts real estate prices using a RandomForestRegressor model. It accepts property details (e.g., city, property type, area) and returns the predicted price, a SHAP explanation image (as a `.jpeg` file URL), and a textual description of factors influencing the prediction.

## Prerequisites

Before running the API locally, ensure you have the following installed:

- **Python 3.8 or higher**: Download and install from [python.org](https://www.python.org/downloads/).
- **pip**: Python's package manager (usually included with Python).
- **Git** (optional): To clone the repository, install from [git-scm.com](https://git-scm.com/downloads).
- A code editor (e.g., VS Code) is recommended but not required.

## Project Structure

Ensure you have the following files and directories in your project folder:

```
project/
├── models/
├── model.joblib
├── encoders.joblib
├── static/  # Created automatically when running the API
├── cleaned_data.csv
├── app.py
├── requirements.txt
```

- `model.joblib`: Pre-trained RandomForestRegressor model.
- `encoders.joblib`: Label encoders for categorical features.
- `cleaned_data.csv`: Dataset for computing `Price_per_sqm`.
- `app.py`: FastAPI application code.
- `requirements.txt`: Python dependencies.
- `static/`: Directory for storing generated `.jpeg` SHAP explanation images (created at runtime).

## Installation Instructions

Follow these steps to set up and run the API locally on Windows, macOS, or Linux.

### 1. Clone or Download the Project

If you have Git installed, clone the repository:

```bash
git clone git@github.com:abdulazimRabie/gh-ml.git
cd gh-ml
```

Alternatively, download the project as a ZIP file and extract it to a folder.

### 2. Set Up a Virtual Environment

Create and activate a Python virtual environment to isolate dependencies:

#### On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

#### On macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

After activation, your terminal prompt should indicate the virtual environment (e.g., `(venv)`).

### 3. Install Dependencies

Install the required Python packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

This installs FastAPI, Uvicorn, pandas, numpy, scikit-learn, joblib, pydantic, SHAP, Matplotlib, and Pillow.

If you encounter issues, ensure `pip` is up-to-date:

```bash
pip install --upgrade pip
```

### 4. Verify Project Files

Ensure the following files are in the project root:
- `app.py`
- `requirements.txt`
- `cleaned_data.csv`
- `models/model.joblib`
- `models/encoders.joblib`

The `static/` directory will be created automatically when you run the API.

### 5. Run the API

Start the FastAPI server using Uvicorn:

```bash
python app.py
or
python3 app.py
or
uvicorn app:app --reload
```

This runs the API on `http://localhost:8000`. You should see output like:

```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### 6. Test the API

To verify the API is working, send a POST request to `http://localhost:8000/predict` with a JSON payload. You can use tools like **cURL**, **Postman**, or a Python script.

#### Example Using cURL:
```bash
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{
    "city": "New Cairo - El Tagamoa",
    "property_type": "Apartment",
    "furnished": "No",
    "delivery_term": "Semi Finished",
    "bedrooms": 3,
    "bathrooms": 2,
    "area": 150.0,
    "level": 2
}'
```

#### Example Using Python (Optional):
Save the following as `test_client.py`:

```python
import requests

url = "http://localhost:8000/predict"
input_data = {
    "city": "New Cairo - El Tagamoa",
    "property_type": "Apartment",
    "furnished": "No",
    "delivery_term": "Semi Finished",
    "bedrooms": 3,
    "bathrooms": 2,
    "area": 150.0,
    "level": 2
}

response = requests.post(url, json=input_data)
response.raise_for_status()
result = response.json()

print("Predicted Price:", result["predicted_price"])
print("Image URL:", f"http://localhost:8000{result['image_url']}")
print("Factors Description:\n", result["factors_description"])

# Download the JPEG image
image_response = requests.get(f"http://localhost:8000{result['image_url']}")
image_response.raise_for_status()
with open("shap_explanation.jpg", "wb") as f:
    f.write(image_response.content)
print("Image saved as shap_explanation.jpg")
```

Run the script:
```bash
pip install requests
python test_client.py
```

**Expected Response**:
```json
{
    "predicted_price": 2500000.0,
    "image_url": "/static/shap_explanation_123e4567-e89b-12d3-a456-426614174000.jpeg",
    "factors_description": "Factors influencing the predicted price:\n- Type: Value = 0, increased price by 4014.99 units.\n- Bedrooms: Value = 3, decreased price by 141611.64 units.\n..."
}
```

- The `image_url` (e.g., `http://localhost:8000/static/shap_explanation_<uuid>.jpeg`) links to the SHAP explanation image.
- Open the URL in a browser to view the `.jpeg` file or check the `static/` directory.
- If using `test_client.py`, the image is saved as `shap_explanation.jpg`.

### Troubleshooting

- **Port Conflict**: If port 8000 is in use, change the port in `app.py` (e.g., `port=8001`) or kill the conflicting process:
  ```bash
  # On macOS/Linux
  lsof -i :8000
  kill -9 <PID>
  # On Windows
  netstat -ano | findstr :8000
  taskkill /PID <PID> /F
  ```

- **Missing Files**: Ensure `model.joblib`, `encoders.joblib`, and `cleaned_data.csv` are in the correct locations. If missing, contact the project maintainer.

- **Invalid Input**: If you get a 400 error (e.g., "Invalid City"), ensure input values match valid options in `cleaned_data.csv`. Check valid values by inspecting `encoders.joblib` or the dataset.

- **Dependency Issues**: If `pip install` fails, try:
  ```bash
  pip install --no-cache-dir -r requirements.txt
  ```

- **Image Not Found**: Verify the `static/` directory is writable and contains the `.jpeg` file. Check the `image_url` in the response.

### Additional Notes

- **File Cleanup**: The API generates a new `.jpeg` file per request in `static/`. Periodically delete old files to free disk space:
  ```bash
  rm static/*.jpg  # macOS/Linux
  del static\*.jpg  # Windows
  ```

- **Production**: For production, use a WSGI server like Gunicorn, configure a reverse proxy (e.g., Nginx), and implement automatic file cleanup. Do not run `python app.py` directly in production.

- **Valid Input Values**: Ensure input fields (`city`, `property_type`, `furnished`, `delivery_term`) match values in `cleaned_data.csv`. Example valid values might include:
  - `city`: "New Cairo - El Tagamoa", "6th of October"
  - `property_type`: "Apartment", "Villa"
  - `furnished`: "No", "Yes"
  - `delivery_term`: "Semi Finished", "Finished"

## Support

For issues, contact the project maintainer or open an issue in the repository (if applicable).