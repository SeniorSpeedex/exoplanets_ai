from fastapi import FastAPI, HTTPException, BackgroundTasks, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from datetime import datetime
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json
import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
import uuid
from catboost import CatBoostClassifier
import pickle
from sklearn.impute import KNNImputer
import pandas as pd
#from predict import ModelNasa  
import secrets
from typing import Optional
from predict import ModelNasa
app = FastAPI(version="1.0", title="NASA Exoplanet Analysis API")

# Конфигурация email для ОС
EMAIL_CONFIG = {
    "address": "minobra52@gmail.com",
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "password": os.getenv("EMAIL_PASSWORD") or "Sigmaboy123"  # Используем пароль приложения
}

# Проверяем наличие пароля при запуске
if EMAIL_CONFIG["password"] == "Sigmaboy123":
    print("WARNING: Email password not set. Feedback emails will not be sent.")

# Глобальные настройки приложения
app_settings = {
    "language": "ru",
    "theme": "dark"
}

origins = [
    "http://localhost",  # Для доступа по локальному хосту
    "http://localhost:8080",  
    "http://127.0.0.1:5500",  
    "http://127.0.0.1:8000", 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for HTML frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

# In-memory storage (в продe заменить на базу данных)
user_sessions = {}
exoplanet_history = []
feedback_data = []
educational_content = {}
user_accounts = {}  # email -> user_data
active_sessions = {}  # session_token -> user_id

# Модели данных Pydantic (расширенные)
class UserSettings(BaseModel):
    language: str = "ru"
    theme: str = "dark"
    user_id: Optional[str] = None  # Исправлено: | на Optional

class UserRegistration(BaseModel):
    username: str
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

class UserProfile(BaseModel):
    user_id: str
    username: str
    email: str
    registration_date: str
    searches_count: int

class ExoplanetData(BaseModel):
    # Основные параметры (расширенный список согласно фронту)
    orbital_period: float
    transit_epoch: float
    impact_parameter: float
    transit_duration: float
    transit_depth: float
    planetary_radius: float
    equilibrium_temperature: float
    insolation_flux: float
    transit_snr: float
    tce_planet_number: int
    stellar_temperature: float
    stellar_surface_gravity: float
    ra: float
    dec: float
    kepler_band: float
    star_system: str

    # Дополнительные гиперпараметры для улучшения анализа
    stellar_metallicity: float = 0.0
    stellar_mass: float = 1.0
    stellar_radius: float = 1.0
    age_of_system: float = 5.0  # в миллиардах лет

class FeedbackRequest(BaseModel):
    name: str
    email: str
    message: str
    user_id: Optional[str] = None  # Исправлено: None на Optional[str]

class SearchResponse(BaseModel):
    habitable: bool
    confidence: float
    analysis: str
    details: dict
    search_id: str

class PDFRequest(BaseModel):
    search_id: str
    language: str = "ru"


# Хэлпер-функции для работы с паролями
def hash_password(password: str) -> str:
    import hashlib
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return hash_password(plain_password) == hashed_password

def create_session(user_id: str) -> str:
    session_token = secrets.token_urlsafe(32)
    active_sessions[session_token] = {
        "user_id": user_id,
        "created_at": datetime.now().isoformat()
    }
    return session_token

# Загрузка образовательного контента при старте
def load_educational_content():
    global educational_content
    educational_content = {
        "ru": {
            "habitability": {
                "title": "Что делает планету обитаемой?",
                "content": """
                <h3>Ключевые факторы обитаемости планет:</h3>
                <ul>
                <li><strong>Зона обитаемости</strong> - расстояние от звезды, где возможна жидкая вода</li>
                <li><strong>Атмосфера</strong> - наличие защитного газового слоя</li>
                <li><strong>Магнитное поле</strong> - защита от звездной радиации</li>
                <li><strong>Стабильность орбиты</strong> - предсказуемые климатические условия</li>
                <li><strong>Состав планеты</strong> - наличие необходимых химических элементов</li>
                </ul>
                """
            },
            "methods": {
                "title": "Методы обнаружения экзопланет",
                "content": """
                <h3>Основные методы поиска экзопланет:</h3>
                <ul>
                <li><strong>Транзитный метод</strong> - обнаружение по затемнению звезды</li>
                <li><strong>Метод Доплера</strong> - измерение колебаний звезды</li>
                <li><strong>Прямое наблюдение</strong> - использование коронографов</li>
                <li><strong>Гравитационное микролинзирование</strong> - использование эффекта линзы</li>
                </ul>
                """
            }
        },
        "en": {
            "habitability": {
                "title": "What Makes a Planet Habitable?",
                "content": """
                <h3>Key factors for planetary habitability:</h3>
                <ul>
                <li><strong>Habitable zone</strong> - distance from star allowing liquid water</li>
                <li><strong>Atmosphere</strong> - presence of protective gas layer</li>
                <li><strong>Magnetic field</strong> - protection from stellar radiation</li>
                <li><strong>Orbital stability</strong> - predictable climate conditions</li>
                <li><strong>Planetary composition</strong> - availability of necessary elements</li>
                </ul>
                """
            },
            "methods": {
                "title": "Exoplanet Detection Methods",
                "content": """
                <h3>Main methods for exoplanet discovery:</h3>
                <ul>
                <li><strong>Transit method</strong> - detection via star dimming</li>
                <li><strong>Doppler method</strong> - measuring star wobbles</li>
                <li><strong>Direct imaging</strong> - using coronagraphs</li>
                <li><strong>Gravitational microlensing</strong> - using lensing effect</li>
                </ul>
                """
            }
        }
    }

class MockExoplanetModel(ModelNasa):
    def __init__(self):
        super().__init__('catboost_model.cbm', 'knn_imput.sav')
        self.hyperparameters = {
            "temperature_range": (200, 350),
            "radius_range": (0.5, 2.5),
            "insolation_range": (0.3, 1.8),
            "period_range": (50, 400),
            "gravity_range": (2.5, 4.5),
            'orbital_period': (0.241842544, 129995.7784),
            'transit_epoch': (120.5159138, 1472.522306),
            'impact_parameter': (0.0, 100.806),
            'transit_duration': (0.052, 138.54),
            'transit_depth': (0.0, 1541400.0),
            'planetary_radius': (0.08, 200346.0),
            'equilibrium_temperature': (25.0, 14667.0),
            'insolation_flux': (0.0, 10947554.55),
            'transit_snr': (0.0, 9054.7),
            'tce_planet_number': (1.0, 8.0),
            'stellar_temperature': (2661.0, 15896.0),
            'stellar_surface_gravity': (0.047, 5.364),
            'stellar_radius': (0.109, 229.908),
            'ra': (279.85272, 301.72076),
            'dec': (36.577381, 52.33601),
            'kepler_band': (6.966, 20.003)
        }
    def analys(self,data: ExoplanetData):
        x = pd.DataFrame(columns=self.columns)
        x.loc[0] = [np.nan] * 16
        for i in self.names:
            x.loc[0, self.comp[i]] = getattr(data, i)
        return self.analys_feat(x)
        
    def predict_habitability(self, data: ExoplanetData):
        x = pd.DataFrame(columns=self.columns)
        x.loc[0] = [np.nan] * 16
        for i in self.names:
            x.loc[0, self.comp[i]] = getattr(data, i)

        confidence = self.prediction(x)  # Исправлено: model.prediction на self.prediction
        if confidence > 0.5:
            habitable = 1
        else:
            habitable = 0

        return bool(habitable), confidence * 100  # Исправлено: возвращаем bool

    def _evaluate_temperature(self, temp: float):
        optimal_range = self.hyperparameters["temperature_range"]
        if optimal_range[0] <= temp <= optimal_range[1]:
            return 1.0
        elif temp < optimal_range[0] - 50 or temp > optimal_range[1] + 50:
            return 0.0
        else:
            return 0.5

    def _evaluate_radius(self, radius: float):
        optimal_range = self.hyperparameters["radius_range"]
        if optimal_range[0] <= radius <= optimal_range[1]:
            return 1.0
        return 0.0

    def _evaluate_flux(self, flux: float):
        optimal_range = self.hyperparameters["insolation_range"]
        if optimal_range[0] <= flux <= optimal_range[1]:
            return 1.0
        return 0.0

    def _evaluate_period(self, period: float):
        optimal_range = self.hyperparameters["period_range"]
        if optimal_range[0] <= period <= optimal_range[1]:
            return 1.0
        return 0.0

    def _evaluate_gravity(self, gravity: float):
        optimal_range = self.hyperparameters["gravity_range"]
        if optimal_range[0] <= gravity <= optimal_range[1]:
            return 1.0
        return 0.0

    def _evaluate_additional(self, data: ExoplanetData):
        # Оценка дополнительных параметров
        factors = []

        # SNR транзита
        if data.transit_snr > 10:
            factors.append(1.0)
        else:
            factors.append(0.5)

        # Глубина транзита
        if data.transit_depth > 100:  # ppm
            factors.append(1.0)
        else:
            factors.append(0.7)

        return sum(factors) / len(factors) if factors else 0.5

# Инициализация модели
model = MockExoplanetModel()

# УЛУЧШЕННАЯ функция отправки email с обработкой ошибок
async def send_feedback_email(feedback: FeedbackRequest):
    """Безопасная отправка email с обработкой всех возможных ошибок"""
    try:
        # Проверяем конфигурацию email
        if not all([EMAIL_CONFIG["address"], EMAIL_CONFIG["password"]]):
            print("Email configuration incomplete. Skipping email send.")
            return False

        if EMAIL_CONFIG["password"] == "your_app_specific_password_here":
            print("Default email password detected. Please set EMAIL_PASSWORD environment variable.")
            return False

        # Создаем сообщение
        msg = MIMEMultipart()
        msg['From'] = EMAIL_CONFIG["address"]
        msg['To'] = EMAIL_CONFIG["address"]
        msg['Subject'] = f"Exoplanet AI Feedback from {feedback.name}"

        body = f"""
        New feedback received from Exoplanet AI:

        📧 Contact Information:
        Name: {feedback.name}
        Email: {feedback.email}
        User ID: {feedback.user_id or 'Not provided'}
        Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

        💬 Message:
        {feedback.message}

        ---
        This email was sent automatically from NASA Exoplanet AI System.
        """

        msg.attach(MIMEText(body, 'plain', 'utf-8'))

        # Устанавливаем соединение и отправляем
        print("📧 Attempting to send feedback email...")
        server = smtplib.SMTP(EMAIL_CONFIG["smtp_server"], EMAIL_CONFIG["smtp_port"])
        server.ehlo()
        server.starttls()
        server.ehlo()

        # Безопасная аутентификация
        server.login(EMAIL_CONFIG["address"], EMAIL_CONFIG["password"])

        text = msg.as_string()
        server.sendmail(EMAIL_CONFIG["address"], EMAIL_CONFIG["address"], text)
        server.quit()

        print("✅ Feedback email sent successfully!")
        return True

    except smtplib.SMTPAuthenticationError:
        print("❌ SMTP Authentication Error: Check email and password")
        return False
    except smtplib.SMTPException as e:
        print(f"❌ SMTP Error: {str(e)}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error sending email: {str(e)}")
        return False

# Функция генерации PDF
def generate_pdf_report(search_data: dict, language: str = "ru"):
    try:
        filename = f"exoplanet_report_{search_data['search_id']}.pdf"
        doc = SimpleDocTemplate(filename, pagesize=letter)
        styles = getSampleStyleSheet()

        # Стили для разных языков
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=30,
            alignment=1  # center
        )

        content = []

        # Заголовок
        title_text = "Exoplanets AI Analyze Report " if language == "en" else "Отчет об экзопланете"
        content.append(Paragraph(title_text, title_style))

        # Информация о системе
        params = search_data['parameters']
        feat = search_data['shap']
        
        system_info = [
            ["Parameter", "Value","Features importance"],
            ["Star System", params.get('star_system', 'N/A'),' '],
            ["Planetary Radius", f"{params.get('planetary_radius', 'N/A')} R⊕",feat.loc[0,'planetary_radius']],
            ["Equilibrium Temperature", f"{params.get('equilibrium_temperature', 'N/A')} K",feat.loc[0,'equilibrium_temperature']],
            ["Orbital Period", f"{params.get('orbital_period', 'N/A')} days",feat.loc[0,'orbital_period']],
            ["Transit Epoch", f"{params.get('transit_epoch', 'N/A')}",feat.loc[0,'transit_epoch']],
            ["Impact Parameter", f"{params.get('impact_parameter', 'N/A')}",feat.loc[0,'impact_parameter']],
            ["Transit Duration", f"{params.get('transit_duration', 'N/A')} hours",feat.loc[0,'transit_duration']],
            ["Transit Depth", f"{params.get('transit_depth', 'N/A')} ppm",feat.loc[0,'transit_depth']],
            ["Insolation Flux", f"{params.get('insolation_flux', 'N/A')} F⊕",feat.loc[0,'insolation_flux']],
            ["Transit SNR", f"{params.get('transit_snr', 'N/A')}",feat.loc[0,'transit_snr']],
            ["TCE Planet Number", f"{params.get('tce_planet_number', 'N/A')}",feat.loc[0,'tce_planet_number']],
            ["Stellar Temperature", f"{params.get('stellar_temperature', 'N/A')} K",feat.loc[0,'stellar_temperature']],
            ["Stellar Surface Gravity", f"{params.get('stellar_surface_gravity', 'N/A')} log(cm/s²)",feat.loc[0,'stellar_surface_gravity']],
            ["Right Ascension", f"{params.get('ra', 'N/A')}°",feat.loc[0,'ra']],
            ["Declination", f"{params.get('dec', 'N/A')}°",feat.loc[0,'dec']],
            ["Kepler Band Magnitude", f"{params.get('kepler_band', 'N/A')}",feat.loc[0,'kepler_band']],
            ["Stellar Radius", f"{params.get('stellar_radius', 'N/A')} R☉",feat.loc[0,'stellar_radius']],
            ["Stellar Mass", f"{params.get('stellar_mass', 'N/A')} M☉"," "],
            ["Stellar Metallicity", f"{params.get('stellar_metallicity', 'N/A')}"," "],
            ["System Age", f"{params.get('age_of_system', 'N/A')} billion years"," "]
        ]

        system_table = Table(system_info)
        system_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        content.append(system_table)
        content.append(Spacer(1, 20))

        # Результаты анализа
        result_text = "Analysis Results" if language == "en" else "Результаты анализа"
        content.append(Paragraph(result_text, styles['Heading2']))

        result_data = search_data['result']
        habitable_text = "Successful" if result_data['habitable'] else "Not an exoplanet"
        habitable_text_ru = "Является экзопланетой" if result_data['habitable'] else "Не является экзопланетой"

        result_info = [
            ["Metric", "Value"],
            ["Analysis status", habitable_text if language == "en" else habitable_text_ru],
            ["Confidence Level", f"{result_data.get('confidence', 0):.1f}%"],
            ["Analysis", result_data.get('analysis', 'N/A')]
        ]

        result_table = Table(result_info)
        result_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        content.append(result_table)

        doc.build(content)
        return filename
    except Exception as e:
        print(f"Error generating PDF: {str(e)}")
        return None

# Инициализация при запуске
@app.on_event("startup")
async def startup_event():
    load_educational_content()
    print("NASA Exoplanet AI Backend started successfully")

# Serve main page
@app.get("/")
async def read_index():
    return FileResponse('static/index.html')

# Эндпоинт для регистрации пользователя
@app.post("/api/register")
async def register_user(user_data: UserRegistration):
    if user_data.email in user_accounts:
        raise HTTPException(status_code=400, detail="Email already registered")

    user_id = str(uuid.uuid4())

    user_accounts[user_data.email] = {
        "user_id": user_id,
        "username": user_data.username,
        "email": user_data.email,
        "password_hash": hash_password(user_data.password),
        "registration_date": datetime.now().isoformat(),
        "searches_count": 0
    }

    # Создаем сессию автоматически после регистрации
    session_token = create_session(user_id)

    return {
        "status": "success",
        "message": "User registered successfully",
        "user_id": user_id,
        "session_token": session_token,
        "username": user_data.username
    }

# Эндпоинт для авторизации пользователя
@app.post("/api/login")
async def login_user(login_data: UserLogin):
    if login_data.email not in user_accounts:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    user_data = user_accounts[login_data.email]

    if not verify_password(login_data.password, user_data["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    session_token = create_session(user_data["user_id"])

    return {
        "status": "success",
        "message": "Login successful",
        "user_id": user_data["user_id"],
        "session_token": session_token,
        "username": user_data["username"]
    }

# Эндпоинт для выхода
@app.post("/api/logout")
async def logout_user(session_token: str):
    if session_token in active_sessions:
        del active_sessions[session_token]

    return {"status": "success", "message": "Logout successful"}

# Эндпоинт для получения ID пользователя
@app.get("/api/user/id")
async def get_user_id():
    """Генерация уникального ID пользователя для отслеживания сессии"""
    user_id = str(uuid.uuid4())
    user_sessions[user_id] = {
        "created_at": datetime.now().isoformat(),
        "settings": UserSettings().dict(),
        "search_count": 0
    }
    return {"user_id": user_id}

# Эндпоинт для сохранения настроек
@app.post("/settings")
async def save_user_settings(settings: UserSettings):
    # Сохранение пользовательских настроек (язык, тема)
    if settings.user_id and settings.user_id in user_sessions:
        user_sessions[settings.user_id]["settings"] = settings.dict()

    # Глобальные настройки приложения (для демонстрации)
    app_settings.update(settings.dict())

    return {"status": "success", "message": "Settings saved successfully"}

# Эндпоинт для получения образовательного контента
@app.get("/api/education/{topic}")
async def get_educational_content(topic: str, language: str = "ru"):
    if language not in educational_content:
        language = "ru"

    if topic in educational_content[language]:
        return educational_content[language][topic]
    else:
        raise HTTPException(status_code=404, detail="Educational topic not found")

# Эндпоинт анализа экзопланеты (обновленный)
@app.post("/search", response_model=SearchResponse)
async def analyze_exoplanet(
    data: ExoplanetData, 
    session_token: Optional[str] = None,
    authorization: Optional[str] = Header(None)  # Добавлена поддержка заголовка Authorization
):
    """Расширенный анализ экзопланеты на обитаемость с гиперпараметрами"""

    # Получаем session_token из заголовка Authorization, если не передан в теле
    if not session_token and authorization and authorization.startswith("Bearer "):
        session_token = authorization.replace("Bearer ", "")

    user_id = None
    if session_token and session_token in active_sessions:
        user_id = active_sessions[session_token]["user_id"]
        # Обновляем счетчик поисков в аккаунте
        for account in user_accounts.values():
            if account["user_id"] == user_id:
                account["searches_count"] += 1
                break

    # Определяем язык для анализа
    language = "ru"
    if user_id and user_id in user_sessions:
        user_settings = user_sessions[user_id].get("settings", {})
        language = user_settings.get("language", "ru")
    shap_values = model.analys(data)
    # Анализ с использованием модели и гиперпараметров
    habitable, confidence = model.predict_habitability(data)

    # Генерация анализа на основе результатов
    if language == "en":
        analysis = generate_english_analysis(habitable, confidence, data)
    else:
        analysis = generate_russian_analysis(habitable, confidence, data)

    # Создание записи в истории
    search_id = str(uuid.uuid4())
    search_record = {
        "search_id": search_id,
        "timestamp": datetime.now().isoformat(),
        "user_id": user_id,
        "parameters": data.dict(),
        "result": {
            "habitable": habitable,
            "confidence": confidence,
            "analysis": analysis
        },
        "shap":shap_values
    }
    exoplanet_history.append(search_record)

    return SearchResponse(
        habitable=habitable,
        confidence=confidence,
        analysis=analysis,
        details=data.dict(),
        search_id=search_id
    )

# Эндпоинт для генерации PDF отчета
@app.post("/api/generate-pdf")
async def generate_pdf(request: PDFRequest):
    # Поиск данных анализа по ID
    search_data = None
    for search in exoplanet_history:
        if search["search_id"] == request.search_id:
            search_data = search
            break

    if not search_data:
        raise HTTPException(status_code=404, detail="Analysis not found")

    # Генерация PDF
    pdf_filename = generate_pdf_report(search_data, request.language)

    if pdf_filename and os.path.exists(pdf_filename):
        return FileResponse(
            pdf_filename, 
            filename=f"exoplanet_report_{request.search_id}.pdf",
            media_type='application/pdf'
        )
    else:
        raise HTTPException(status_code=500, detail="Error generating PDF report")

# Эндпоинт истории поисков (обновленный)
@app.get("/history")
async def get_search_history(user_id: Optional[str] = None):  # Исправлено: None на Optional[str]
    """Получение истории поисков с фильтрацией по пользователю"""
    if user_id:
        user_searches = [s for s in exoplanet_history if s.get("user_id") == user_id]
        searches = user_searches[-10:]  # Последние 10 поисков пользователя
    else:
        searches = exoplanet_history[-10:]  # Последние 10 поисков вообще

    return {
        "count": len(searches),
        "searches": searches
    }

# Эндпоинт профиля пользователя (обновленный)
@app.get("/me")
async def get_user_profile(
    session_token: Optional[str] = None, 
    authorization: Optional[str] = Header(None)
):
    """Получение профиля пользователя с статистикой"""
    user_data = None

    # Получаем session_token из заголовка Authorization
    if not session_token and authorization:
        if authorization.startswith("Bearer "):
            session_token = authorization.replace("Bearer ", "")

    if session_token and session_token in active_sessions:
        user_id = active_sessions[session_token]["user_id"]
        # Находим данные пользователя по user_id
        for account in user_accounts.values():
            if account["user_id"] == user_id:
                user_data = account
                break

    if user_data:
        return {
            "username": user_data["username"],
            "email": user_data["email"],
            "searches_count": user_data["searches_count"],
            "member_since": user_data["registration_date"].split("T")[0],
            "preferences": {
                "temperature_unit": "Kelvin",
                "mass_unit": "Earth masses",
                "language": "ru"
            }
        }
    else:
        # Возвращаем анонимный профиль с временным ID
        user_id = str(uuid.uuid4())[:8]
        return {
            "username": f"Guest_Researcher_{user_id}",
            "searches_count": 0,
            "member_since": datetime.now().strftime("%Y-%m-%d"),
            "preferences": {
                "temperature_unit": "Kelvin",
                "mass_unit": "Earth masses",
                "language": "ru"
            }
        }

# Эндпоинт обратной связи (с отправкой email)
@app.post("/help")
async def submit_feedback(feedback: FeedbackRequest, background_tasks: BackgroundTasks):
    """Отправка обратной связи с уведомлением на email"""

    feedback_record = {
        "timestamp": datetime.now().isoformat(),
        "user_id": feedback.user_id,
        **feedback.dict()
    }
    feedback_data.append(feedback_record)

    # Отправка email в фоновом режиме
    background_tasks.add_task(send_feedback_email, feedback)

    return {
        "status": "success",
        "message": "Thank you for your feedback!",
        "received_data": feedback.dict()
    }

# Эндпоинт статистики обратной связи
@app.get("/feedback/stats")
async def get_feedback_stats():
    return {
        "total_feedback": len(feedback_data),
        "latest_feedback": feedback_data[-5:] if feedback_data else []
    }

# Вспомогательные функции для генерации текста анализа
def generate_russian_analysis(habitable: bool, confidence: float, data: ExoplanetData) -> str:
    if habitable:
        return f"Planet in {data.star_system} system is an exoplanet! " \
               f"Analysis confidence: {confidence:.1f}%. Further study recommended."
    else:
        return f"Planet in {data.star_system} system is not an exoplanet! " \
               f"Analysis confidence: {confidence:.1f}%."

def generate_english_analysis(habitable: bool, confidence: float, data: ExoplanetData) -> str:
    if habitable:
        return f"Planet in {data.star_system} system is an exoplanet! " \
               f"Analysis confidence: {confidence:.1f}%. Further study recommended."
    else:
        return f"Planet in {data.star_system} system is not an exoplanet! " \
               f"Analysis confidence: {confidence:.1f}%."

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
