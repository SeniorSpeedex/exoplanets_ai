# ExoAI

**ExoAI** is an open-source platform designed to tackle the NASA Space Apps Challenge 2025 task: "A World Away: Hunting for Exoplanets with AI". The project combines interactive Jupyter Notebooks, Python scripts, HTML visualizations, and a FastAPI-powered backend to solve the challenge's core problems using modern AI and data science tools.

---

## Requirements

- **Python 3.8+**
- **Jupyter Notebook**
- **FastAPI** (for backend server)
- **pip** (for package management)

---

## Quick Start Guide

1. **Clone the repository:**
   ```sh
   git clone https://github.com/SeniorSpeedex/exoai.git
   cd exoai
   ```

2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
   *(If `requirements.txt` is missing, manually install the recommended packages above.)*

3. **Run the FastAPI backend:**
   ```sh
   uvicorn backend.main:app --reload
   ```
   *(Assuming your FastAPI app is in `backend/main.py`)*  
   Visit [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for interactive API documentation.

4. **Launch Jupyter Notebook:**
   ```sh
   jupyter notebook
   ```
   Open `.ipynb` files to run and modify AI workflows for exoplanet analysis.

5. **View visualizations:**  
   Open generated HTML files in your browser for interactive results.

---

## Deployment

- ## **Local:**  
### **To run the project locally, follow these steps:**

## **1. Virtual Environment Setup**

*Create a virtual environment:*
```bash
python -m venv venv
```
**Activate the virtual environment:**

*Windows:*
```bash
venv\Scripts\activate
```
*Linux/MacOS:*
```bash
source venv/bin/activate
```
## **2. Install Dependencies**
*Install required packages from requirements.txt:*
```bash
pip install -r requirements.txt
```
## **3. Configuration Check and Initialization**

*Run the Docker setup script:*
```bash
python docker-setup.py
```
## **4. Run Docker Containers**

*Build and start services using Docker Compose:*
```bash
docker-compose up --build
```

## **P.S**

*1. Make sure Docker and Docker Compose are installed on your system*

*2. Verify that requirements.txt and docker-setup.py files are in the project root*

*3. After completing all steps, the application will be available at - http://127.0.0.1:8000*

- **Cloud:**  
  -

---

## Features

- **Solves NASA Space Apps Exoplanet AI Challenge:**  
  Implements AI and ML solutions for exoplanet detection and analysis.
- **FastAPI Backend:**  
  Provides RESTful API endpoints for data processing, model inference, and integration.
- **Modular Jupyter Notebooks:**  
  Extensible workflows for data science and experimentation.
- **Python Data Science Stack:**  
  Uses popular libraries for data handling, modeling, and visualization.
- **Interactive HTML Visualizations:**  
  Share and explore results easily in any browser.
- **Easy Local & Cloud Deployment:**  
  Minimal setup for running locally or in cloud environments.
