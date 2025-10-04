# ExoAI

**ExoAI** harnesses the power of artificial intelligence to improve the search for exoplanets. Developed for NASA's Space Apps Challenge in 2025, our open-source platform transforms complex astronomical data into actionable insights. Combining exploratory Jupyter Notebook, a scalable FastAPI backend and interactive visualizations, ExoAI provides a powerful and accessible toolkit for identifying and verifying planetary candidates outside our solar system.

---

## Project demo website
http://84.54.29.144:8000/ - all functions are working here! Try and send your feedback to us!

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
   git clone https://github.com/SeniorSpeedex/exoplanets_ai.git
   cd exoplanets_ai
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

- ## **Cloud:**   
  ## Server Setup with Docker

### Prerequisites
- Linux VPS with Docker and Docker Compose
- Root SSH access


1. **Connect to server:**
   ```bash
   ssh root@your-server-ip
   ```

2. **Create project directory:**
   ```bash
   mkdir -p /home/speedsense
   cd /home/speedsense
   ```

3. **Upload all project files to `/home/speedsense/`**

4. **Deploy:**
   ```bash
   docker-compose up -d
   ```

5. **Verify:**
   ```bash
   docker ps
   docker logs exoplanet_app
   ```

### Access Application
- Web interface: `http://your-server-ip:8000`

### Management Commands
```bash
# Stop application
docker-compose down

# Restart
docker-compose restart

# Update with changes
docker-compose up -d --build

# View logs
docker logs exoplanet_app -f
```

### Domain Setup
Point your domain to server IP with A records:
- `@` → `your-server-ip`
- `www` → `your-server-ip`



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


## 🎯 Challenge requirements

| **Requirements**                                              | **ExoAI Implementation**                                                                                         |
|---------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| "Develop AI/ML solutions for exoplanet detection and analysis" | **Production CatBoost model** (`catboost_model.cbm`) + `knn_imput.sav` = high accuracy   |
| "Process and analyze Kepler mission data"                     | **Complete data pipeline** in `model.ipynb` processing `cumulative_2025.09.20_00:18:02.csv` dataset              |
| "Create accessible tools and interfaces"                      | **FastAPI backend** (`main.py`) + **Interactive web interface** (`static/index.html`) + **RESTful API endpoints** |
| "Ensure reproducibility and scalability"                      | **Docker containerization** (`Dockerfile`, `docker-compose.yml`) + **MongoDB integration** (`init-mongo.js`)     |
| "Provide data visualization capabilities"                     | **Jupyter notebooks** (`model.ipynb`) + **HTML visualizations** + **Real-time prediction interface**             |
| "Implement end-to-end exoplanet hunting workflow"             | **From raw data → feature engineering → model training → prediction serving** in unified platform                |
| "Leverage modern AI/ML frameworks"                            | **CatBoost Classifier** + **Scikit-learn KNN** + **Full Python data science stack** (`requirements.txt`)         |

---

### **They worked for you:**

**Timur Veliev - teamlead, github.com/SeniorSpeedex**

**Dmitriy Susin - backend developer, github.com/F1ling**

**Mikhail Novoselov - ML Engineer, telegram: @Redgrave23** 

**Artem Emelyanov - Data Analyst, github.com/pnby**

---
