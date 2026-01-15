# ZPhones – FastAPI Backend

ZPhones-FastApi is the **backend service** for the ZPhones project.  
It provides a REST API built with **FastAPI** that runs multiple **machine-learning pipelines** to detect and classify smartphone images, predict prices, and manage an inventory stored in **MongoDB with GridFS**.

## Features
- Image-based phone detection and classification
- Multiple ML pipelines:
  - **Splitter** (no phone / single phone / multiple phones)
  - **Brand → Model pipeline**
  - **Direct model classifier**
  - **Manual classification** fallback
- Confidence-based validation for predictions
- **Price prediction** using a regression model
- Inventory management:
  - add new products
  - update quantities
  - list products
- Image storage and streaming using **MongoDB GridFS**
- Designed to work with a React frontend

## Tech Stack
- FastAPI
- TensorFlow / Keras
- NumPy, Pandas
- MongoDB + Motor + GridFS
- Joblib (regression model)
- Uvicorn

## Project Structure
