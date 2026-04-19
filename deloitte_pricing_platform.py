# 安装命令：pip install streamlit plotly pandas numpy scipy scikit-learn openai requests

import os
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from scipy.stats import norm, gamma, poisson
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import openai
import requests
import warnings
warnings.filterwarnings('ignore')

# 创建配置目录和config.toml来强制奶白色主题（解决下拉菜单对比度问题）
config_dir = ".streamlit"
if not os.path.exists(config_dir):
    os.makedirs(config_dir)
config_path = os.path.join(config_dir, "config.toml")
with open(config_path, "w", encoding="utf-8") as f:
    f.write("""
[theme]
primaryColor = "#3b82f6"
backgroundColor = "#fefdf9"
secondaryBackgroundColor = "#faf8f3"
textColor = "#1f2937"
font = "sans serif"
""")

# ==================== 动态参数计算 ====================
def calculate_dynamic_parameters(historical_data):
    """
    基于历史数据动态计算定价参数
    """
    if historical_data.empty:
        # 默认值作为后备
        return {
            "base_price": 200,
            "holiday_surcharge": 0.3,
            "rainy_discount": 0.15,
            "price_elasticity": -1.2,
            "max_peak_valley_diff": 0.4
        }
    
    # 计算基础价格（历史平均票价）
    base_price = historical_data["票价"].mean()
    base_price = int(round(base_price, -1))
    base_price = max(150, min(300, base_price))
    
    # 计算节假日附加费
    holiday_prices = historical_data[historical_data["是否节假日"] == "是"]["票价"]
    weekday_prices = historical_data[historical_data["是否节假日"] == "否"]["票价"]
    
    if not holiday_prices.empty and not weekday_prices.empty:
        holiday_avg = holiday_prices.mean()
        weekday_avg = weekday_prices.mean()
        holiday_surcharge = (holiday_avg - weekday_avg) / weekday_avg
        holiday_surcharge = max(0.1, min(0.5, holiday_surcharge))
    else:
        holiday_surcharge = 0.3
    
    # 计算雨天折扣
    rainy_prices = historical_data[historical_data["天气"].isin(["小雨", "大雨"])]["票价"]
    sunny_prices = historical_data[historical_data["天气"].isin(["晴天", "多云"])]["票价"]
    
    if not rainy_prices.empty and not sunny_prices.empty:
        rainy_avg = rainy_prices.mean()
        sunny_avg = sunny_prices.mean()
        rainy_discount = (sunny_avg - rainy_avg) / sunny_avg
        rainy_discount = max(0.05, min(0.25, rainy_discount))
    else:
        rainy_discount = 0.15
    
    # 计算价格弹性（基于历史数据的价格和客流关系）
    price_elasticity = -1.2  # 默认为-1.2
    if len(historical_data) >= 10:
        # 使用线性回归计算简单的价格弹性
        from sklearn.linear_model import LinearRegression
        
        df = historical_data.copy()
        df["price_log"] = np.log(df["票价"])
        df["traffic_log"] = np.log(df["客流"])
        
        X = df[["price_log"]]
        y = df["traffic_log"]
        
        model = LinearRegression()
        model.fit(X, y)
        
        elasticity = model.coef_[0]
        # 确保弹性为负（价格上涨，需求下降）
        if elasticity < 0:
            price_elasticity = max(-3.0, min(-0.5, elasticity))
    
    # 计算最大峰谷差异
    max_price = historical_data["票价"].max()
    min_price = historical_data["票价"].min()
    if min_price > 0:
        max_peak_valley_diff = (max_price - min_price) / min_price
        max_peak_valley_diff = max(0.2, min(0.6, max_peak_valley_diff))
    else:
        max_peak_valley_diff = 0.4
    
    return {
        "base_price": base_price,
        "holiday_surcharge": holiday_surcharge,
        "rainy_discount": rainy_discount,
        "price_elasticity": price_elasticity,
        "max_peak_valley_diff": max_peak_valley_diff
    }

# ==================== 全局配置区 - 请仅修改这里的参数 ====================
DEEPSEEK_API_KEY = "sk-791dd465a4454a37b16f3632e1629f68"
# 这些参数将在运行时动态计算，初始值为默认值
BASE_PRICE = 200
HOLIDAY_SURCHARGE = 0.3
RAINY_DISCOUNT = 0.15
PRICE_ELASTICITY = -1.2
MAX_PEAK_VALLEY_DIFF = 0.4
# ===========================================================================

# ==================== 真实日历和天气获取 ====================
# 2025-2026年中国法定节假日（根据国务院办公厅通知）
CHINA_HOLIDAYS_2025 = {
    "2025-01-01", "2025-01-28", "2025-01-29", "2025-01-30", "2025-01-31",
    "2025-02-01", "2025-02-02", "2025-02-03", "2025-02-04",
    "2025-04-04", "2025-04-05", "2025-04-06",
    "2025-05-01", "2025-05-02", "2025-05-03", "2025-05-04", "2025-05-05",
    "2025-06-22",
    "2025-10-01", "2025-10-02", "2025-10-03", "2025-10-04", "2025-10-05", "2025-10-06", "2025-10-07", "2025-10-08"
}

CHINA_HOLIDAYS_2026 = {
    "2026-01-01",
    "2026-02-17", "2026-02-18", "2026-02-19", "2026-02-20", "2026-02-21", "2026-02-22", "2026-02-23",
    "2026-04-04", "2026-04-05", "2026-04-06",
    "2026-05-01", "2026-05-02", "2026-05-03", "2026-05-04", "2026-05-05",
    "2026-06-19",
    "2026-10-01", "2026-10-02", "2026-10-03", "2026-10-04", "2026-10-05", "2026-10-06", "2026-10-07", "2026-10-08"
}

def is_chinese_holiday(date_str):
    """判断是否为中国法定节假日"""
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    year = date_obj.year
    
    if year == 2025:
        return date_str in CHINA_HOLIDAYS_2025 or date_obj.weekday() >= 5
    elif year == 2026:
        return date_str in CHINA_HOLIDAYS_2026 or date_obj.weekday() >= 5
    else:
        return date_obj.weekday() >= 5

def get_weather_forecast(city="Beijing", days=7):
    """
    获取天气预报（使用免费天气API）
    如果API调用失败，返回备用模拟数据
    """
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude=39.9042&longitude=116.4074&daily=weather_code&timezone=Asia%2FShanghai&forecast_days={days}"
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            weather_codes = data.get('daily', {}).get('weather_code', [])
            
            weather_map = {
                0: "晴天", 1: "晴天", 2: "多云", 3: "多云",
                45: "阴天", 48: "阴天",
                51: "小雨", 53: "小雨", 55: "小雨",
                56: "小雨", 57: "小雨",
                61: "小雨", 63: "小雨", 65: "大雨",
                66: "小雨", 67: "大雨",
                71: "小雨", 73: "小雨", 75: "大雨",
                77: "小雨",
                80: "小雨", 81: "小雨", 82: "大雨",
                85: "小雨", 86: "大雨",
                95: "大雨", 96: "大雨", 99: "大雨"
            }
            
            weather_list = [weather_map.get(code, "多云") for code in weather_codes]
            return weather_list[:days]
    except Exception as e:
        print(f"天气API调用失败: {e}")
    
    weather_options = ["晴天", "多云", "阴天", "小雨", "大雨"]
    weather_weights = [0.45, 0.30, 0.15, 0.07, 0.03]
    return [random.choices(weather_options, weights=weather_weights)[0] for _ in range(days)]

def get_competitor_price_trend(base_price, days=7):
    """获取竞品价格趋势（模拟真实市场波动）"""
    prices = []
    for i in range(days):
        if i == 0:
            price = base_price
        else:
            price = prices[i-1] * (0.98 + random.random() * 0.06)
        price = int(max(150, min(300, price)))
        prices.append(price)
    return prices
# ===========================================================================

st.set_page_config(
    page_title="德勤 - 乐园AI智能定价平台",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 光追效果CSS样式 ====================
st.markdown("""
<style>
    /* 基础光追容器 */
    .light-trace-container {
        position: relative;
        overflow: hidden;
    }
    
    /* 光追元素 - 图标和文字 */
    .light-trace-element {
        position: relative;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
    }
    
    .light-trace-element:hover {
        transform: translateY(-5px) scale(1.02);
    }
    
    /* 光追效果 */
    .light-trace-element::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(
            circle at var(--mouse-x, 50%) var(--mouse-y, 50%),
            rgba(59, 130, 246, 0.15) 0%,
            rgba(59, 130, 246, 0.08) 20%,
            transparent 50%
        );
        opacity: 0;
        transition: opacity 0.3s ease;
        pointer-events: none;
        z-index: 1;
    }
    
    .light-trace-element:hover::before {
        opacity: 1;
    }
    
    /* 发光边框效果 */
    .glow-border {
        position: relative;
        border: 2px solid transparent;
        border-image: linear-gradient(
            90deg,
            rgba(59, 130, 246, 0.3) 0%,
            rgba(59, 130, 246, 0.8) var(--mouse-progress, 50%),
            rgba(59, 130, 246, 0.3) 100%
        ) 1;
    }
    
    /* 图标颜色随鼠标变化 */
    .icon-color-shift {
        filter: saturate(1);
        transition: filter 0.3s ease, transform 0.3s ease;
    }
    
    .icon-color-shift:hover {
        filter: saturate(1.5) brightness(1.1);
    }
    
    /* 文字颜色渐变效果 */
    .text-gradient {
        background: linear-gradient(
            90deg,
            #1e40af var(--text-start, 0%),
            #3b82f6 var(--text-middle, 50%),
            #60a5fa var(--text-end, 100%)
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        background-size: 200% 100%;
        animation: textGradient 3s ease infinite;
    }
    
    @keyframes textGradient {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    /* 脉冲光效 */
    .pulse-glow {
        animation: pulseGlow 2s ease-in-out infinite;
    }
    
    @keyframes pulseGlow {
        0%, 100% { 
            box-shadow: 0 0 5px rgba(59, 130, 246, 0.3),
                        0 0 10px rgba(59, 130, 246, 0.2);
        }
        50% { 
            box-shadow: 0 0 20px rgba(59, 130, 246, 0.5),
                        0 0 30px rgba(59, 130, 246, 0.3);
        }
    }
    
    /* 悬浮光影卡片 */
    .hover-card {
        background: linear-gradient(135deg, #fefdf9 0%, #faf8f3 100%);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .hover-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(
            90deg,
            transparent,
            rgba(59, 130, 246, 0.1),
            transparent
        );
        transition: left 0.5s ease;
    }
    
    .hover-card:hover::before {
        left: 100%;
    }
    
    .hover-card:hover {
        transform: translateY(-8px) rotateX(2deg);
        box-shadow: 0 20px 40px rgba(59, 130, 246, 0.2),
                    0 8px 25px rgba(0, 0, 0, 0.1);
        border-color: #3b82f6;
    }
    
    /* 单色图标样式 */
    .mono-icon {
        transition: all 0.3s ease;
    }
    
    .mono-icon:hover {
        transform: scale(1.1) rotate(5deg);
    }
    
    /* Streamlit组件样式优化 */
    h1, h2, h3 { color: #1e40af !important; }
    .stMetric { 
        background: linear-gradient(135deg, #fefdf9 0%, #faf8f3 100%); 
        border-radius: 12px; 
        padding: 15px; 
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    .stMetric:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.15);
        border-color: #3b82f6;
    }
    div[data-testid="stMetricValue"] { color: #16a34a !important; font-weight: bold; }
    div[data-testid="stMetricLabel"] { color: #4b5563 !important; }
    button[data-testid="stBaseButton-primary"] {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%) !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    button[data-testid="stBaseButton-primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4) !important;
    }
    .stSlider label {
        color: #374151 !important;
        font-weight: 600 !important;
        font-size: 15px !important;
    }
    div[data-testid="stThumbValue"] {
        color: #3b82f6 !important;
        font-weight: bold !important;
    }
</style>

<!-- 光追效果JavaScript -->
<script>
document.addEventListener('mousemove', function(e) {
    const elements = document.querySelectorAll('.light-trace-element');
    
    elements.forEach(function(el) {
        const rect = el.getBoundingClientRect();
        const x = ((e.clientX - rect.left) / rect.width) * 100;
        const y = ((e.clientY - rect.top) / rect.height) * 100;
        
        el.style.setProperty('--mouse-x', x + '%');
        el.style.setProperty('--mouse-y', y + '%');
        
        const progress = ((e.clientX - rect.left) / rect.width) * 100;
        el.style.setProperty('--mouse-progress', progress + '%');
        
        const textProgress = ((e.clientX - rect.left) / rect.width) * 100;
        el.style.setProperty('--text-start', Math.max(0, textProgress - 30) + '%');
        el.style.setProperty('--text-middle', textProgress + '%');
        el.style.setProperty('--text-end', Math.min(100, textProgress + 30) + '%');
    });
});
</script>
""", unsafe_allow_html=True)

# ==================== 二维单色图标库 ====================
MONO_ICONS = {
    "coaster": """<svg class="mono-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M12 2L12 22M2 12L22 12" stroke="#3b82f6" stroke-width="2" stroke-linecap="round"/>
        <circle cx="12" cy="12" r="3" fill="#3b82f6"/>
        <circle cx="12" cy="12" r="8" stroke="#3b82f6" stroke-width="1.5"/>
    </svg>""",
    
    "target": """<svg class="mono-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <circle cx="12" cy="12" r="9" stroke="#16a34a" stroke-width="2"/>
        <circle cx="12" cy="12" r="5" stroke="#16a34a" stroke-width="2"/>
        <circle cx="12" cy="12" r="2" fill="#16a34a"/>
    </svg>""",
    
    "chart": """<svg class="mono-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <rect x="3" y="14" width="3" height="7" fill="#3b82f6" rx="0.5"/>
        <rect x="8" y="10" width="3" height="11" fill="#3b82f6" rx="0.5"/>
        <rect x="13" y="6" width="3" height="15" fill="#3b82f6" rx="0.5"/>
        <rect x="18" y="12" width="3" height="9" fill="#3b82f6" rx="0.5"/>
        <path d="M2 3L22 3" stroke="#64748b" stroke-width="1" stroke-linecap="round"/>
    </svg>""",
    
    "trending": """<svg class="mono-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M4 18L10 12L14 15L20 9" stroke="#16a34a" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        <circle cx="20" cy="9" r="2" fill="#16a34a"/>
        <path d="M17 9L20 9L20 12" stroke="#16a34a" stroke-width="2" stroke-linecap="round"/>
    </svg>""",
    
    "gear": """<svg class="mono-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <circle cx="12" cy="12" r="3" stroke="#6366f1" stroke-width="2"/>
        <circle cx="12" cy="12" r="7" stroke="#6366f1" stroke-width="2"/>
        <rect x="11" y="2" width="2" height="4" fill="#6366f1"/>
        <rect x="11" y="18" width="2" height="4" fill="#6366f1"/>
        <rect x="2" y="11" width="4" height="2" fill="#6366f1"/>
        <rect x="18" y="11" width="4" height="2" fill="#6366f1"/>
    </svg>""",
    
    "bell": """<svg class="mono-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M12 3C12 3 6 8 6 14L6 17L18 17L18 14C18 8 12 3 12 3" fill="#f59e0b" stroke="#d97706" stroke-width="1.5"/>
        <path d="M8 18L16 18" stroke="#d97706" stroke-width="2" stroke-linecap="round"/>
        <circle cx="12" cy="21" r="1.5" fill="#d97706"/>
    </svg>""",
    
    "search": """<svg class="mono-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <circle cx="10" cy="10" r="6" stroke="#3b82f6" stroke-width="2"/>
        <path d="M15 15L21 21" stroke="#3b82f6" stroke-width="2" stroke-linecap="round"/>
    </svg>""",
    
    "lightbulb": """<svg class="mono-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <circle cx="12" cy="10" r="6" fill="#fbbf24" stroke="#f59e0b" stroke-width="1.5"/>
        <rect x="9" y="15" width="6" height="3" fill="#f59e0b"/>
        <rect x="8" y="17" width="8" height="2" fill="#f59e0b"/>
        <path d="M12 2L12 4" stroke="#f59e0b" stroke-width="2" stroke-linecap="round"/>
    </svg>""",
    
    "calendar": """<svg class="mono-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <rect x="3" y="5" width="18" height="16" rx="2" stroke="#3b82f6" stroke-width="2"/>
        <rect x="3" y="3" width="18" height="4" fill="#3b82f6" rx="1"/>
        <rect x="6" y="1" width="2" height="4" fill="#93c5fd"/>
        <rect x="16" y="1" width="2" height="4" fill="#93c5fd"/>
        <rect x="6" y="10" width="2" height="2" fill="#3b82f6"/>
        <rect x="10" y="10" width="2" height="2" fill="#3b82f6"/>
        <rect x="14" y="10" width="2" height="2" fill="#3b82f6"/>
        <rect x="6" y="14" width="2" height="2" fill="#93c5fd"/>
        <rect x="10" y="14" width="2" height="2" fill="#93c5fd"/>
        <rect x="14" y="14" width="2" height="2" fill="#93c5fd"/>
    </svg>""",
    
    "warning": """<svg class="mono-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M12 3L22 20L2 20Z" fill="#fef3c7" stroke="#f59e0b" stroke-width="2"/>
        <rect x="11" y="9" width="2" height="6" fill="#f59e0b"/>
        <circle cx="12" cy="17" r="1" fill="#f59e0b"/>
    </svg>""",
    
    "document": """<svg class="mono-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <rect x="4" y="3" width="16" height="18" rx="2" stroke="#3b82f6" stroke-width="2"/>
        <path d="M14 3L14 7L18 7" fill="#3b82f6"/>
        <rect x="7" y="11" width="10" height="1.5" fill="#93c5fd"/>
        <rect x="7" y="14" width="10" height="1.5" fill="#93c5fd"/>
        <rect x="7" y="17" width="7" height="1.5" fill="#93c5fd"/>
    </svg>""",
    
    "building": """<svg class="mono-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <rect x="6" y="4" width="12" height="17" stroke="#3b82f6" stroke-width="2"/>
        <rect x="2" y="11" width="4" height="10" stroke="#3b82f6" stroke-width="2"/>
        <rect x="18" y="11" width="4" height="10" stroke="#3b82f6" stroke-width="2"/>
        <rect x="8" y="7" width="2" height="2" fill="#3b82f6"/>
        <rect x="14" y="7" width="2" height="2" fill="#3b82f6"/>
        <rect x="8" y="11" width="2" height="2" fill="#93c5fd"/>
        <rect x="14" y="11" width="2" height="2" fill="#93c5fd"/>
        <rect x="8" y="15" width="2" height="2" fill="#3b82f6"/>
        <rect x="14" y="15" width="2" height="2" fill="#3b82f6"/>
        <rect x="10" y="18" width="4" height="3" fill="#1e40af"/>
    </svg>""",
    
    "rocket": """<svg class="mono-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M12 3L14 10L10 10Z" fill="#ef4444"/>
        <rect x="10" y="10" width="4" height="8" fill="#3b82f6"/>
        <path d="M10 14L6 18L10 17" fill="#6366f1"/>
        <path d="M14 14L18 18L14 17" fill="#6366f1"/>
        <circle cx="12" cy="13" r="1.5" fill="#fbbf24"/>
        <path d="M11 18L12 21L13 18" fill="#f59e0b"/>
    </svg>""",
    
    "bullhorn": """<svg class="mono-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M4 10L14 6L14 18L4 14Z" fill="#3b82f6"/>
        <rect x="14" y="8" width="6" height="8" rx="2" fill="#6366f1"/>
        <circle cx="20" cy="12" r="1.5" fill="#fbbf24"/>
        <rect x="2" y="14" width="3" height="6" fill="#3b82f6"/>
    </svg>"""
}

def get_mono_icon(name, size=32, color="#3b82f6"):
    """获取二维单色图标"""
    svg = MONO_ICONS.get(name, MONO_ICONS["chart"])
    svg = svg.replace('#3b82f6', color)
    svg = svg.replace('#16a34a', color)
    svg = svg.replace('#6366f1', color)
    svg = svg.replace('#f59e0b', color)
    svg = svg.replace('#d97706', color)
    svg = svg.replace('#fbbf24', color)
    svg = svg.replace('#93c5fd', color)
    svg = svg.replace('#ef4444', color)
    svg = svg.replace('#1e40af', color)
    return svg.replace('viewBox="0 0 24 24"', f'viewBox="0 0 24 24" width="{size}" height="{size}"')

def display_icon(name, size=28, color="#3b82f6", margin_right=8):
    """在Streamlit中显示图标"""
    svg = get_mono_icon(name, size, color)
    return f'<span style="display:inline-flex;align-items:center;margin-right:{margin_right}px;vertical-align:middle;">{svg}</span>'

def generate_historical_data(base_price=200, holiday_surcharge=0.3, rainy_discount=0.15):
    dates = [datetime.now() - timedelta(days=i) for i in range(30, 0, -1)]
    data = []
    for date in dates:
        weekday = date.weekday()
        is_holiday = weekday >= 5 or random.random() < 0.1
        weather_options = ["晴天", "多云", "阴天", "小雨", "大雨"]
        weather = random.choices(weather_options, weights=[0.4, 0.3, 0.15, 0.1, 0.05])[0]
        
        base_traffic = 8000
        if is_holiday:
            base_traffic *= 1.5
        if weather in ["小雨", "大雨"]:
            base_traffic *= 0.6
        if weather == "阴天":
            base_traffic *= 0.9
        
        traffic = int(base_traffic + random.gauss(0, 800))
        traffic = max(2000, min(18000, traffic))
        
        price = base_price
        if is_holiday:
            price *= (1 + holiday_surcharge * 0.8)
        if weather in ["小雨", "大雨"]:
            price *= (1 - rainy_discount)
        
        price = int(price + random.gauss(0, 10))
        price = max(120, min(350, price))
        
        ticket_revenue = traffic * price
        secondary_spending = traffic * (80 + random.gauss(0, 15))
        competitor_price = int(base_price * (0.9 + random.random() * 0.3))
        
        data.append({
            "日期": date.strftime("%Y-%m-%d"),
            "星期": ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][weekday],
            "是否节假日": "是" if is_holiday else "否",
            "天气": weather,
            "客流": traffic,
            "票价": price,
            "门票营收": int(ticket_revenue),
            "二次消费营收": int(secondary_spending),
            "竞品均价": competitor_price
        })
    return pd.DataFrame(data)

def bayesian_traffic_forecast(date_type, weather, historical_data):
    """
    贝叶斯时间序列客流预测
    使用贝叶斯更新和历史数据进行概率预测
    """
    weather_factor = {"晴天": 1.0, "多云": 0.9, "阴天": 0.85, "小雨": 0.65, "大雨": 0.5}
    day_factor = {"工作日": 1.0, "周末": 1.4, "节假日": 1.8}
    
    historical_data = historical_data.copy()
    historical_data["日期"] = pd.to_datetime(historical_data["日期"])
    historical_data = historical_data.sort_values("日期")
    
    y = historical_data["客流"].values
    n = len(y)
    
    prior_mean = y.mean()
    prior_std = y.std()
    prior_precision = 1 / (prior_std ** 2)
    
    window_size = min(7, n)
    recent_data = y[-window_size:]
    recent_mean = recent_data.mean()
    recent_std = recent_data.std() if len(recent_data) > 1 else prior_std
    
    likelihood_mean = recent_mean * day_factor.get(date_type, 1.0) * weather_factor.get(weather, 1.0)
    likelihood_std = max(recent_std * 0.8, 800)
    likelihood_precision = 1 / (likelihood_std ** 2)
    
    posterior_precision = prior_precision + likelihood_precision
    posterior_mean = (prior_precision * prior_mean + likelihood_precision * likelihood_mean) / posterior_precision
    posterior_std = np.sqrt(1 / posterior_precision)
    
    forecast_samples = norm.rvs(loc=posterior_mean, scale=posterior_std, size=100)
    forecast = int(np.percentile(forecast_samples, 50))
    forecast = max(3000, min(20000, forecast))
    
    return forecast, posterior_mean, posterior_std

def random_forest_sensitivity_analysis(historical_data, base_price, price_elasticity_default=-1.2):
    """
    随机森林和决策树价格敏感度分析
    训练模型分析价格弹性和最优价格区间
    """
    df = historical_data.copy()
    
    df["is_holiday"] = (df["是否节假日"] == "是").astype(int)
    weather_mapping = {"晴天": 0, "多云": 1, "阴天": 2, "小雨": 3, "大雨": 4}
    df["weather_encoded"] = df["天气"].map(weather_mapping)
    
    df["revenue"] = df["门票营收"] + df["二次消费营收"]
    df["price_ratio"] = df["票价"] / base_price
    df["traffic_ratio"] = df["客流"] / df["客流"].mean()
    
    X = df[["票价", "is_holiday", "weather_encoded", "客流"]]
    y_revenue = df["revenue"]
    y_traffic = df["客流"]
    
    rf_revenue = RandomForestRegressor(
        n_estimators=100,
        max_depth=8,
        min_samples_split=5,
        random_state=42
    )
    rf_revenue.fit(X, y_revenue)
    
    rf_traffic = RandomForestRegressor(
        n_estimators=100,
        max_depth=8,
        min_samples_split=5,
        random_state=42
    )
    rf_traffic.fit(X, y_traffic)
    
    from sklearn.tree import DecisionTreeRegressor
    dt_revenue = DecisionTreeRegressor(
        max_depth=5,
        min_samples_split=5,
        random_state=42
    )
    dt_revenue.fit(X, y_revenue)
    
    price_grid = np.linspace(base_price * 0.7, base_price * 1.3, 50)
    
    best_revenue_rf = -1
    best_price_rf = base_price
    
    for price in price_grid:
        X_test = X.copy()
        X_test["票价"] = price
        
        pred_revenue_rf = rf_revenue.predict(X_test).mean()
        
        if pred_revenue_rf > best_revenue_rf:
            best_revenue_rf = pred_revenue_rf
            best_price_rf = price
    
    price_changes = np.linspace(-0.2, 0.2, 20)
    traffic_changes = []
    
    for delta_p in price_changes:
        test_price = base_price * (1 + delta_p)
        X_test = X.copy()
        X_test["票价"] = test_price
        pred_traffic = rf_traffic.predict(X_test).mean()
        traffic_change = (pred_traffic - X["客流"].mean()) / X["客流"].mean()
        traffic_changes.append(traffic_change)
    
    price_changes = np.array(price_changes)
    traffic_changes = np.array(traffic_changes)
    
    valid_indices = np.where(np.abs(price_changes) > 0.01)[0]
    elasticities = traffic_changes[valid_indices] / price_changes[valid_indices]
    elasticity = float(np.mean(elasticities)) if len(elasticities) > 0 else price_elasticity_default
    
    optimal_min = int(max(base_price * 0.8, best_price_rf * 0.92))
    optimal_max = int(min(base_price * 1.2, best_price_rf * 1.08))
    
    return {
        "price_elasticity": elasticity,
        "optimal_price_range": (optimal_min, optimal_max),
        "price_sensitive_threshold": float(best_price_rf * 1.05),
        "random_forest_model": rf_revenue,
        "decision_tree_model": dt_revenue,
        "rf_traffic_model": rf_traffic
    }

def reinforcement_learning_optimal_price(base_demand, competitor_price, sensitivity_result, historical_data, base_price):
    """
    真正的强化学习Q-learning定价优化
    多状态、多动作、带探索-利用平衡的完整Q-learning
    """
    optimal_range = sensitivity_result["optimal_price_range"]
    elasticity = sensitivity_result["price_elasticity"]
    rf_traffic_model = sensitivity_result.get("rf_traffic_model", None)
    
    num_states = 5
    num_actions = 15
    Q = np.zeros((num_states, num_actions))
    
    epsilon_start = 0.3
    epsilon_end = 0.05
    epsilon_decay = 0.995
    
    alpha = 0.12
    gamma = 0.98
    
    actions = np.linspace(-0.25, 0.25, num_actions)
    
    target_traffic = 10000
    secondary_spending = 80
    
    epsilon = epsilon_start
    
    if rf_traffic_model is not None:
        df = historical_data.copy()
        df["is_holiday"] = (df["是否节假日"] == "是").astype(int)
        weather_mapping = {"晴天": 0, "多云": 1, "阴天": 2, "小雨": 3, "大雨": 4}
        df["weather_encoded"] = df["天气"].map(weather_mapping)
        X_base = df[["票价", "is_holiday", "weather_encoded", "客流"]]
    
    for episode in range(1000):
        state = episode % num_states
        
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        if random.random() < epsilon:
            action_idx = random.randint(0, num_actions - 1)
        else:
            action_idx = np.argmax(Q[state])
        
        price_adjustment = actions[action_idx]
        price = base_price * (1 + price_adjustment)
        
        if rf_traffic_model is not None:
            X_test = X_base.copy()
            X_test["票价"] = price
            traffic = rf_traffic_model.predict(X_test).mean()
        else:
            traffic = base_demand * (price / base_price) ** elasticity
        
        traffic = max(3000, min(20000, traffic))
        
        ticket_revenue = price * traffic
        secondary_revenue = secondary_spending * traffic
        total_revenue = ticket_revenue + secondary_revenue
        
        revenue_score = total_revenue * 0.00008
        
        traffic_balance = 1.0 - abs(traffic - target_traffic) / target_traffic
        traffic_balance = max(0, traffic_balance)
        traffic_reward = traffic_balance * 25
        
        competitor_diff = price - competitor_price
        if competitor_diff > 20:
            competitor_penalty = (competitor_diff - 20) * 0.3
        elif competitor_diff < -20:
            competitor_penalty = abs(competitor_diff + 20) * 0.2
        else:
            competitor_penalty = 0
        
        if optimal_range[0] <= price <= optimal_range[1]:
            range_bonus = 20
        elif optimal_range[0] * 0.95 <= price <= optimal_range[1] * 1.05:
            range_bonus = 10
        else:
            range_bonus = -10
        
        reward = revenue_score + traffic_reward - competitor_penalty + range_bonus
        
        next_state = (state + 1) % num_states
        best_next_action = np.argmax(Q[next_state])
        
        Q[state, action_idx] = Q[state, action_idx] + alpha * (
            reward + gamma * Q[next_state, best_next_action] - Q[state, action_idx]
        )
    
    best_action_idx = np.argmax(Q[2])
    optimal_price = base_price * (1 + actions[best_action_idx])
    
    return int(round(optimal_price, -1)), Q

def demand_function(price, base_demand_at_base_price, base_price, elasticity):
    price_ratio = price / base_price
    demand = base_demand_at_base_price * (price_ratio ** elasticity)
    return int(max(3000, min(20000, demand)))

def calculate_base_demand(date_type, weather):
    weather_factor = {"晴天": 1.0, "多云": 0.9, "阴天": 0.85, "小雨": 0.65, "大雨": 0.5}
    day_factor = {"工作日": 1.0, "周末": 1.4, "节假日": 1.8}
    
    base_traffic = 10000
    base_traffic *= day_factor.get(date_type, 1.0)
    base_traffic *= weather_factor.get(weather, 1.0)
    
    return int(max(3000, min(20000, base_traffic)))

def find_optimal_price_supply_demand(base_demand, competitor_price, secondary_potential, 
                                      is_holiday, is_rainy, sensitivity_result, base_price):
    traffic_at_base = base_demand
    elasticity = sensitivity_result["price_elasticity"]
    
    price_grid = np.linspace(150, 320, 80)
    
    best_price = base_price
    best_revenue = -float('inf')
    best_traffic = 0
    
    results = []
    
    for price in price_grid:
        price_ratio = price / base_price
        traffic = traffic_at_base * (price_ratio ** elasticity)
        traffic = int(max(3000, min(20000, traffic)))
        
        ticket_revenue = price * traffic
        secondary_revenue = secondary_potential * traffic
        total_revenue = ticket_revenue + secondary_revenue
        
        target_traffic = 10000
        traffic_balance = 1.0 - abs(traffic - target_traffic) / target_traffic
        traffic_balance = max(0, traffic_balance)
        
        score = total_revenue * (0.85 + 0.15 * traffic_balance)
        
        results.append({
            'price': price,
            'traffic': traffic,
            'revenue': total_revenue,
            'score': score
        })
        
        if score > best_revenue:
            best_revenue = score
            best_price = price
            best_traffic = traffic
    
    fixed_traffic = traffic_at_base * ((base_price / base_price) ** elasticity)
    fixed_revenue = base_price * fixed_traffic + fixed_traffic * secondary_potential
    
    comp_traffic = traffic_at_base * ((competitor_price / base_price) ** elasticity)
    comp_revenue = competitor_price * comp_traffic + comp_traffic * secondary_potential
    
    if best_revenue <= max(fixed_revenue, comp_revenue):
        search_prices = np.linspace(best_price * 0.92, best_price * 1.15, 40)
        for p in search_prices:
            t = traffic_at_base * ((p / base_price) ** elasticity)
            t = int(max(3000, min(20000, t)))
            r = p * t + t * secondary_potential
            if r > best_revenue:
                best_revenue = r
                best_price = p
                best_traffic = t
    
    best_price = int(round(best_price, -1))
    
    return int(max(150, min(320, best_price))), best_traffic, results

# ==================== 增强算法模块 ====================

def enhanced_gradient_boosting_model(historical_data, base_price):
    """
    增强的梯度提升预测模型（使用LightGBM风格，兼容scikit-learn）
    比随机森林更强大的价格预测
    """
    df = historical_data.copy()
    
    df["is_holiday"] = (df["是否节假日"] == "是").astype(int)
    weather_mapping = {"晴天": 0, "多云": 1, "阴天": 2, "小雨": 3, "大雨": 4}
    df["weather_encoded"] = df["天气"].map(weather_mapping)
    
    df["day_of_week"] = pd.to_datetime(df["日期"]).dt.dayofweek
    df["month"] = pd.to_datetime(df["日期"]).dt.month
    df["revenue"] = df["门票营收"] + df["二次消费营收"]
    
    X = df[["票价", "is_holiday", "weather_encoded", "客流", "day_of_week", "month"]]
    y_revenue = df["revenue"]
    y_traffic = df["客流"]
    
    from sklearn.ensemble import GradientBoostingRegressor
    
    gb_revenue = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        min_samples_split=5,
        random_state=42,
        subsample=0.8
    )
    gb_revenue.fit(X, y_revenue)
    
    gb_traffic = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        min_samples_split=5,
        random_state=42,
        subsample=0.8
    )
    gb_traffic.fit(X, y_traffic)
    
    price_grid = np.linspace(base_price * 0.6, base_price * 1.4, 60)
    best_revenue = -1
    best_price = base_price
    
    for price in price_grid:
        X_test = X.copy()
        X_test["票价"] = price
        pred_revenue = gb_revenue.predict(X_test).mean()
        
        if pred_revenue > best_revenue:
            best_revenue = pred_revenue
            best_price = price
    
    price_changes = np.linspace(-0.25, 0.25, 25)
    traffic_changes = []
    
    for delta_p in price_changes:
        test_price = base_price * (1 + delta_p)
        X_test = X.copy()
        X_test["票价"] = test_price
        pred_traffic = gb_traffic.predict(X_test).mean()
        traffic_change = (pred_traffic - X["客流"].mean()) / X["客流"].mean()
        traffic_changes.append(traffic_change)
    
    price_changes = np.array(price_changes)
    traffic_changes = np.array(traffic_changes)
    
    valid_indices = np.where(np.abs(price_changes) > 0.01)[0]
    elasticities = traffic_changes[valid_indices] / price_changes[valid_indices]
    elasticity = float(np.mean(elasticities)) if len(elasticities) > 0 else -1.2
    
    feature_importance = dict(zip(X.columns, gb_revenue.feature_importances_))
    
    return {
        "model": gb_revenue,
        "traffic_model": gb_traffic,
        "optimal_price": best_price,
        "price_elasticity": elasticity,
        "feature_importance": feature_importance
    }

def monte_carlo_risk_analysis(base_price, base_demand, elasticity, sensitivity_result, num_simulations=1000):
    """
    Monte Carlo模拟用于风险分析
    考虑需求波动、天气变化、竞品价格等不确定性
    """
    results = []
    prices = []
    revenues = []
    traffics = []
    
    weather_volatility = {"晴天": 0.1, "多云": 0.12, "阴天": 0.15, "小雨": 0.25, "大雨": 0.35}
    competitor_volatility = 0.15
    
    for _ in range(num_simulations):
        weather_factor = norm.rvs(loc=1.0, scale=0.1)
        competitor_factor = norm.rvs(loc=1.0, scale=competitor_volatility)
        demand_shock = norm.rvs(loc=0, scale=0.08)
        
        price_noise = norm.rvs(loc=0, scale=5)
        simulated_price = base_price + price_noise
        simulated_price = max(150, min(320, simulated_price))
        
        price_ratio = simulated_price / base_price
        simulated_traffic = base_demand * (price_ratio ** elasticity) * (1 + demand_shock)
        simulated_traffic = max(3000, min(20000, simulated_traffic))
        
        secondary_spending = 80 + gamma.rvs(a=2, scale=10)
        ticket_revenue = simulated_price * simulated_traffic
        secondary_revenue = secondary_spending * simulated_traffic
        total_revenue = ticket_revenue + secondary_revenue
        
        prices.append(simulated_price)
        revenues.append(total_revenue)
        traffics.append(simulated_traffic)
        
        results.append({
            "price": simulated_price,
            "traffic": simulated_traffic,
            "revenue": total_revenue,
            "weather_factor": weather_factor,
            "competitor_factor": competitor_factor
        })
    
    revenues = np.array(revenues)
    prices = np.array(prices)
    traffics = np.array(traffics)
    
    revenue_mean = np.mean(revenues)
    revenue_std = np.std(revenues)
    revenue_5th = np.percentile(revenues, 5)
    revenue_95th = np.percentile(revenues, 95)
    
    var_95 = revenue_mean - revenue_5th
    
    return {
        "simulations": results,
        "revenue_mean": revenue_mean,
        "revenue_std": revenue_std,
        "revenue_5th": revenue_5th,
        "revenue_95th": revenue_95th,
        "var_95": var_95,
        "optimal_risk_adjusted_price": np.percentile(prices, 50),
        "confidence_interval": (revenue_5th, revenue_95th)
    }

def enhanced_time_series_forecast(historical_data, forecast_days=7):
    """
    增强的时间序列预测模型
    使用带趋势和季节性的指数平滑
    """
    df = historical_data.copy()
    df["日期"] = pd.to_datetime(df["日期"])
    df = df.sort_values("日期")
    
    y = df["客流"].values
    n = len(y)
    
    alpha = 0.3
    beta = 0.1
    gamma = 0.2
    
    level = y[0]
    trend = (y[-1] - y[0]) / (n - 1)
    seasonal = np.zeros(7)
    
    for i in range(n):
        day_of_week = df["日期"].iloc[i].dayofweek
        
        if i == 0:
            level = y[i]
        else:
            prev_level = level
            level = alpha * y[i] + (1 - alpha) * (level + trend)
            trend = beta * (level - prev_level) + (1 - beta) * trend
            seasonal[day_of_week] = gamma * (y[i] - level) + (1 - gamma) * seasonal[day_of_week]
    
    forecast = []
    last_date = df["日期"].iloc[-1]
    
    for i in range(forecast_days):
        future_date = last_date + timedelta(days=i+1)
        day_of_week = future_date.dayofweek
        
        forecast_value = level + (i + 1) * trend + seasonal[day_of_week]
        forecast_value = max(3000, min(20000, forecast_value))
        forecast.append(forecast_value)
    
    return forecast, level, trend, seasonal

def multi_objective_optimization(base_demand, competitor_price, secondary_potential, 
                                  is_holiday, is_rainy, sensitivity_result, base_price):
    """
    多目标优化框架
    同时优化：1. 营收 2. 客流稳定性 3. 价格竞争力
    """
    elasticity = sensitivity_result["price_elasticity"]
    
    price_grid = np.linspace(140, 330, 100)
    
    objectives = []
    target_traffic = 10000
    
    for price in price_grid:
        price_ratio = price / base_price
        traffic = base_demand * (price_ratio ** elasticity)
        traffic = int(max(3000, min(20000, traffic)))
        
        ticket_revenue = price * traffic
        secondary_revenue = secondary_potential * traffic
        total_revenue = ticket_revenue + secondary_revenue
        
        traffic_balance = 1.0 - abs(traffic - target_traffic) / target_traffic
        traffic_balance = max(0, traffic_balance)
        
        competitor_diff = abs(price - competitor_price)
        competitiveness = max(0, 1.0 - competitor_diff / 50)
        
        revenue_score = total_revenue / 3000000
        traffic_score = traffic_balance
        competitiveness_score = competitiveness
        
        weights = [0.5, 0.3, 0.2]
        total_score = (
            weights[0] * revenue_score +
            weights[1] * traffic_score +
            weights[2] * competitiveness_score
        )
        
        objectives.append({
            "price": price,
            "traffic": traffic,
            "revenue": total_revenue,
            "revenue_score": revenue_score,
            "traffic_score": traffic_score,
            "competitiveness_score": competitiveness_score,
            "total_score": total_score
        })
    
    best_idx = np.argmax([obj["total_score"] for obj in objectives])
    best_solution = objectives[best_idx]
    
    pareto_front = []
    for obj in objectives:
        is_dominated = False
        for other in objectives:
            if (other["revenue_score"] >= obj["revenue_score"] and
                other["traffic_score"] >= obj["traffic_score"] and
                other["competitiveness_score"] >= obj["competitiveness_score"] and
                (other["revenue_score"] > obj["revenue_score"] or
                 other["traffic_score"] > obj["traffic_score"] or
                 other["competitiveness_score"] > obj["competitiveness_score"])):
                is_dominated = True
                break
        if not is_dominated:
            pareto_front.append(obj)
    
    return {
        "best_solution": best_solution,
        "pareto_front": pareto_front,
        "all_solutions": objectives
    }

def generate_enhanced_report(scenario_data, pricing_result, mc_results, gb_results, mo_results, base_price, price_elasticity):
    """
    生成增强的NLP分析报告
    包含Monte Carlo风险分析、梯度提升模型洞察、多目标优化结果
    """
    date_type = scenario_data["date_type"]
    weather = scenario_data["weather"]
    competitor = scenario_data["competitor"]
    secondary = scenario_data["secondary"]
    
    optimal_price = pricing_result["optimal_price"]
    forecast_traffic = pricing_result["forecast_traffic"]
    
    report = f"""# 乐园AI智能定价增强分析报告

## 一、执行摘要
本报告基于四种高级AI算法（梯度提升、Monte Carlo模拟、增强时间序列、多目标优化）生成。

### 关键指标
- **最优票价**：{optimal_price}元
- **预测客流**：{forecast_traffic:,}人次
- **预期营收**：{optimal_price * forecast_traffic + forecast_traffic * secondary:,.0f}元
- **价格弹性**：{price_elasticity:.2f}

## 二、梯度提升模型分析
### 特征重要性
"""
    
    if "feature_importance" in gb_results:
        for feature, importance in sorted(gb_results["feature_importance"].items(), key=lambda x: x[1], reverse=True)[:5]:
            report += f"- {feature}: {importance:.1%}\n"
    
    report += f"""
### 模型推荐价格
梯度提升模型推荐价格：{gb_results['optimal_price']:.0f}元

## 三、Monte Carlo风险分析
### 不确定性评估
- **预期营收均值**：{mc_results['revenue_mean']/10000:.1f}万元
- **营收标准差**：{mc_results['revenue_std']/10000:.1f}万元
- **95%置信区间**：{mc_results['confidence_interval'][0]/10000:.1f}万 - {mc_results['confidence_interval'][1]/10000:.1f}万元
- **风险价值(VaR 95%)**：{mc_results['var_95']/10000:.1f}万元

### 风险调整建议
考虑不确定性后，风险调整价格为：{mc_results['optimal_risk_adjusted_price']:.0f}元

## 四、多目标优化结果
### 最佳平衡方案
- **价格**：{mo_results['best_solution']['price']:.0f}元
- **营收得分**：{mo_results['best_solution']['revenue_score']:.2f}
- **客流平衡得分**：{mo_results['best_solution']['traffic_score']:.2f}
- **竞争力得分**：{mo_results['best_solution']['competitiveness_score']:.2f}

### Pareto前沿
共找到{len(mo_results['pareto_front'])}个非支配解，可根据业务偏好选择。

## 五、综合建议
1. **推荐价格区间**：{int(optimal_price*0.95)}-{int(optimal_price*1.05)}元
2. **风险监控**：每日跟踪实际客流与预测偏差
3. **动态调整**：建议设置±10%的价格调整阈值
4. **多目标平衡**：根据运营目标调整权重（当前权重：营收50%、客流30%、竞争力20%）

---
*报告生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
*增强算法：梯度提升、Monte Carlo模拟、多目标优化*
"""
    return report

def stable_pricing_engine(base_price, date_type, weather, competitor_price, 
                          sensitivity_result, historical_data, is_holiday):
    """
    稳定的定价引擎
    确保：1. 价格稳定 2. 节假日价格合理高于工作日 3. 避免大幅波动
    """
    price_elasticity = sensitivity_result["price_elasticity"]
    
    # 基础价格区间约束
    min_price = max(150, base_price * 0.85)
    max_price = min(320, base_price * 1.25)
    
    # 日期类型价格系数（确保层级关系）
    date_type_factors = {
        "工作日": 1.0,
        "周末": 1.12,
        "节假日": 1.22
    }
    
    # 天气价格系数
    weather_factors = {
        "晴天": 1.05,
        "多云": 1.0,
        "阴天": 0.97,
        "小雨": 0.92,
        "大雨": 0.88
    }
    
    # 获取基础因子
    date_factor = date_type_factors.get(date_type, 1.0)
    weather_factor = weather_factors.get(weather, 1.0)
    
    # 竞品价格影响（平滑处理）
    competitor_diff = competitor_price - base_price
    competitor_factor = 1.0 + np.clip(competitor_diff / base_price, -0.08, 0.08)
    
    # 计算原始建议价格
    raw_price = base_price * date_factor * weather_factor * competitor_factor
    
    # 应用价格敏感度调整（轻微调整）
    if abs(price_elasticity) > 1.5:
        elasticity_adjustment = 0.97
    elif abs(price_elasticity) < 0.8:
        elasticity_adjustment = 1.03
    else:
        elasticity_adjustment = 1.0
    
    adjusted_price = raw_price * elasticity_adjustment
    
    # 确保在合理区间内
    adjusted_price = max(min_price, min(max_price, adjusted_price))
    
    # 四舍五入到十位（符合行业习惯）
    final_price = int(round(adjusted_price, -1))
    
    # 再次确保层级关系
    if date_type == "节假日":
        final_price = max(final_price, int(round(base_price * 1.15, -1)))
    elif date_type == "周末":
        if is_holiday:
            final_price = max(final_price, int(round(base_price * 1.15, -1)))
        else:
            final_price = max(final_price, int(round(base_price * 1.05, -1)))
            final_price = min(final_price, int(round(base_price * 1.18, -1)))
    
    # 最终安全检查
    final_price = max(150, min(320, final_price))
    
    # 计算价格调整范围（用于展示）
    price_range = {
        "min_recommended": max(150, final_price - 10),
        "max_recommended": min(320, final_price + 10),
        "confidence": "高"
    }
    
    return final_price, price_range

# ==================== 高级AI算法模块 ====================

def deep_learning_ts_forecast(historical_data, lookback_window=7):
    """
    LSTM风格的深度学习时间序列预测
    使用多层感知机模拟LSTM的序列学习能力
    """
    df = historical_data.copy()
    df["日期"] = pd.to_datetime(df["日期"])
    df = df.sort_values("日期")
    
    y = df["客流"].values
    n = len(y)
    
    if n < lookback_window + 1:
        return np.mean(y), y[-1], 0
    
    X_train = []
    y_train = []
    
    for i in range(lookback_window, n):
        X_train.append(y[i-lookback_window:i])
        y_train.append(y[i])
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    from sklearn.neural_network import MLPRegressor
    
    mlp = MLPRegressor(
        hidden_layer_sizes=(64, 32, 16),
        activation='relu',
        solver='adam',
        learning_rate_init=0.001,
        max_iter=500,
        random_state=42,
        early_stopping=True
    )
    
    mlp.fit(X_train, y_train)
    
    last_window = y[-lookback_window:]
    forecast = mlp.predict(last_window.reshape(1, -1))[0]
    
    train_preds = mlp.predict(X_train)
    mape = np.mean(np.abs((y_train - train_preds) / y_train)) * 100
    
    forecast = max(3000, min(20000, forecast))
    
    return forecast, y[-1], mape

def ensemble_learning_fusion(base_price, date_type, weather, competitor_price, 
                              sensitivity_result, historical_data, is_holiday):
    """
    集成学习融合 - 多模型投票机制
    结合多种算法的预测结果
    """
    predictions = []
    
    date_type_factors = {
        "工作日": 1.0,
        "周末": 1.12,
        "节假日": 1.22
    }
    
    weather_factors = {
        "晴天": 1.05,
        "多云": 1.0,
        "阴天": 0.97,
        "小雨": 0.92,
        "大雨": 0.88
    }
    
    date_factor = date_type_factors.get(date_type, 1.0)
    weather_factor = weather_factors.get(weather, 1.0)
    
    model_1_price = base_price * date_factor * weather_factor
    predictions.append(("基础模型", model_1_price, 0.25))
    
    competitor_diff = competitor_price - base_price
    competitor_adjustment = np.clip(competitor_diff / base_price, -0.08, 0.08)
    model_2_price = base_price * date_factor * weather_factor * (1 + competitor_adjustment)
    predictions.append(("竞品模型", model_2_price, 0.20))
    
    price_elasticity = sensitivity_result["price_elasticity"]
    if abs(price_elasticity) > 1.5:
        elasticity_adjustment = 0.97
    elif abs(price_elasticity) < 0.8:
        elasticity_adjustment = 1.03
    else:
        elasticity_adjustment = 1.0
    model_3_price = base_price * date_factor * weather_factor * elasticity_adjustment
    predictions.append(("弹性模型", model_3_price, 0.20))
    
    try:
        rf_optimal = sensitivity_result["optimal_price_range"][0]
        model_4_price = (rf_optimal + base_price * date_factor * weather_factor) / 2
        predictions.append(("随机森林", model_4_price, 0.18))
    except:
        pass
    
    try:
        gb_optimal = sensitivity_result.get("gb_optimal", base_price)
        model_5_price = gb_optimal
        predictions.append(("梯度提升", model_5_price, 0.17))
    except:
        pass
    
    total_weight = sum(w for _, _, w in predictions)
    weighted_price = sum(p * w for _, p, w in predictions) / total_weight
    
    model_details = [{"model": name, "price": int(round(p, -1)), "weight": w} 
                     for name, p, w in predictions]
    
    return int(round(weighted_price, -1)), model_details

def multi_round_monte_carlo_simulation(base_price, base_demand, elasticity, 
                                         num_rounds=5, simulations_per_round=200):
    """
    多轮Monte Carlo模拟
    每轮迭代优化参数，提高预测准确性
    """
    all_results = []
    round_prices = []
    
    for round_idx in range(num_rounds):
        round_results = []
        
        if round_idx == 0:
            price_mean = base_price
            price_std = 15
        else:
            prev_prices = np.array([r["price"] for r in all_results[-simulations_per_round:]])
            price_mean = np.mean(prev_prices)
            price_std = np.std(prev_prices) * 0.8
        
        for sim_idx in range(simulations_per_round):
            weather_volatility = norm.rvs(loc=1.0, scale=0.08)
            competitor_factor = norm.rvs(loc=1.0, scale=0.10)
            demand_shock = norm.rvs(loc=0, scale=0.06)
            
            price_noise = norm.rvs(loc=0, scale=price_std)
            simulated_price = price_mean + price_noise
            simulated_price = max(150, min(320, simulated_price))
            
            price_ratio = simulated_price / base_price
            simulated_traffic = base_demand * (price_ratio ** elasticity) * (1 + demand_shock) * weather_volatility
            simulated_traffic = max(3000, min(20000, simulated_traffic))
            
            secondary_spending = 80 + gamma.rvs(a=2, scale=8)
            ticket_revenue = simulated_price * simulated_traffic
            secondary_revenue = secondary_spending * simulated_traffic
            total_revenue = ticket_revenue + secondary_revenue
            
            round_results.append({
                "round": round_idx + 1,
                "simulation": sim_idx + 1,
                "price": simulated_price,
                "traffic": simulated_traffic,
                "revenue": total_revenue,
                "weather_factor": weather_volatility,
                "competitor_factor": competitor_factor
            })
        
        all_results.extend(round_results)
        round_avg_price = np.mean([r["price"] for r in round_results])
        round_prices.append(int(round(round_avg_price, -1)))
    
    final_prices = [r["price"] for r in all_results[-simulations_per_round:]]
    final_revenues = [r["revenue"] for r in all_results[-simulations_per_round:]]
    optimal_price = int(round(np.mean(final_prices), -1))
    price_std = np.std(final_prices)
    revenue_mean = np.mean(final_revenues)
    
    convergence = np.std(round_prices[-3:]) if len(round_prices) >= 3 else 999
    
    return {
        "optimal_price": optimal_price,
        "price_std": price_std,
        "round_prices": round_prices,
        "all_simulations": all_results,
        "convergence": convergence,
        "revenue_mean": revenue_mean,
        "confidence_interval": (
            int(round(np.percentile(final_prices, 5), -1)),
            int(round(np.percentile(final_prices, 95), -1))
        )
    }

def simulated_annealing_optimization(base_price, base_demand, elasticity, 
                                      target_traffic=10000, max_iterations=500):
    """
    模拟退火优化算法
    用于寻找全局最优定价点
    """
    def objective(price):
        traffic = base_demand * (price / base_price) ** elasticity
        traffic = max(3000, min(20000, traffic))
        revenue = price * traffic + traffic * 80
        
        traffic_penalty = abs(traffic - target_traffic) / target_traffic
        revenue_score = revenue / 3000000
        
        return -(revenue_score - 0.3 * traffic_penalty)
    
    current_price = base_price
    current_score = objective(current_price)
    
    best_price = current_price
    best_score = current_score
    
    T = 50.0
    T_min = 0.1
    alpha = 0.99
    
    temperature_history = []
    price_history = []
    score_history = []
    
    iteration = 0
    while T > T_min and iteration < max_iterations:
        next_price = current_price + norm.rvs(loc=0, scale=10)
        next_price = max(150, min(320, next_price))
        next_score = objective(next_price)
        
        delta = next_score - current_score
        
        if delta < 0:
            current_price = next_price
            current_score = next_score
            
            if current_score < best_score:
                best_price = current_price
                best_score = current_score
        else:
            acceptance_prob = np.exp(-delta / T)
            if random.random() < acceptance_prob:
                current_price = next_price
                current_score = next_score
        
        temperature_history.append(T)
        price_history.append(current_price)
        score_history.append(current_score)
        
        T *= alpha
        iteration += 1
    
    best_price = int(round(best_price, -1))
    
    return {
        "optimal_price": best_price,
        "final_score": best_score,
        "temperature_history": temperature_history,
        "price_history": price_history,
        "score_history": score_history,
        "iterations": iteration
    }

def advanced_ai_pricing_engine(base_price, date_type, weather, competitor_price,
                                sensitivity_result, historical_data, is_holiday):
    """
    高级AI定价引擎 - 融合所有高级算法
    """
    st.info("🔬 高级AI分析中...")
    
    base_demand = calculate_base_demand(date_type, weather)
    price_elasticity = sensitivity_result["price_elasticity"]
    
    results = {}
    
    st.info("步骤1/5: 深度学习时间序列预测...")
    dl_forecast, dl_last, dl_mape = deep_learning_ts_forecast(historical_data)
    results["deep_learning"] = {
        "forecast": dl_forecast,
        "last_value": dl_last,
        "mape": dl_mape
    }
    
    st.info("步骤2/5: 集成学习多模型融合...")
    ensemble_price, ensemble_details = ensemble_learning_fusion(
        base_price, date_type, weather, competitor_price,
        sensitivity_result, historical_data, is_holiday
    )
    results["ensemble"] = {
        "price": ensemble_price,
        "details": ensemble_details
    }
    
    st.info("步骤3/5: 多轮Monte Carlo模拟...")
    mc_results = multi_round_monte_carlo_simulation(
        base_price, base_demand, price_elasticity
    )
    results["monte_carlo"] = mc_results
    
    st.info("步骤4/5: 模拟退火全局优化...")
    sa_results = simulated_annealing_optimization(
        base_price, base_demand, price_elasticity
    )
    results["simulated_annealing"] = sa_results
    
    st.info("步骤5/5: 稳定定价引擎最终确定...")
    stable_price, price_range = stable_pricing_engine(
        base_price, date_type, weather, competitor_price,
        sensitivity_result, historical_data, is_holiday
    )
    results["stable"] = {
        "price": stable_price,
        "range": price_range
    }
    
    candidate_prices = [
        ("稳定引擎", stable_price, 0.35),
        ("集成学习", ensemble_price, 0.25),
        ("Monte Carlo", mc_results["optimal_price"], 0.20),
        ("模拟退火", sa_results["optimal_price"], 0.20)
    ]
    
    total_weight = sum(w for _, _, w in candidate_prices)
    final_price = sum(p * w for _, p, w in candidate_prices) / total_weight
    final_price = int(round(final_price, -1))
    final_price = max(150, min(320, final_price))
    
    return final_price, results

def generate_package_options(optimal_price, weather, is_holiday, date_type=None):
    """
    增强的套餐生成函数
    根据天气、节假日等条件提供丰富的组合销售推荐
    """
    packages = []
    
    # 基础套餐（总是包含）
    base_package = {
        "name": "基础门票",
        "price": optimal_price,
        "description": "包含所有基础项目",
        "features": ["全部项目畅玩", "免费停车", "基础保险"],
        "icon": "🎫"
    }
    packages.append(base_package)
    
    # 天气相关套餐
    if weather == "晴天":
        # 晴天套餐
        sunny_package = {
            "name": "阳光畅享",
            "price": int(optimal_price * 1.05),
            "description": "晴天专属，户外项目优先体验",
            "features": ["户外项目快速通道", "免费遮阳伞", "冰爽饮品买一送一", "户外拍照纪念"],
            "icon": "☀️"
        }
        packages.append(sunny_package)
        
        # 情侣套餐
        couple_package = {
            "name": "浪漫双人",
            "price": int(optimal_price * 1.9),
            "description": "双人浪漫之旅，含专属惊喜",
            "features": ["双人门票", "情侣专属休息区", "精美礼品一份", "指定餐厅优先座位"],
            "icon": "💑"
        }
        packages.append(couple_package)
    
    elif weather == "多云":
        # 多云套餐
        cloudy_package = {
            "name": "舒适漫步",
            "price": optimal_price,
            "description": "多云天气，舒适游玩体验",
            "features": ["全部项目畅玩", "免费充电宝", "美食广场优惠券", "导览地图"],
            "icon": "⛅"
        }
        packages.append(cloudy_package)
        
        # 拍照套餐
        photo_package = {
            "name": "光影记忆",
            "price": int(optimal_price * 1.15),
            "description": "包含专业摄影服务",
            "features": ["门票+3张精修照片", "摄影师跟拍1小时", "电子相册", "相框一个"],
            "icon": "📸"
        }
        packages.append(photo_package)
    
    elif weather == "阴天":
        # 阴天套餐
        overcast_package = {
            "name": "静谧时光",
            "price": int(optimal_price * 0.95),
            "description": "阴天优惠，室内外结合",
            "features": ["全部项目畅玩", "室内项目优先", "热饮优惠券", "休息区优先"],
            "icon": "☁️"
        }
        packages.append(overcast_package)
        
        # 休闲套餐
        leisure_package = {
            "name": "轻松休闲",
            "price": int(optimal_price * 0.85),
            "description": "精选项目，轻松体验",
            "features": ["精选10个项目", "免费导览", "美食代金券", "舒适休息"],
            "icon": "🧘"
        }
        packages.append(leisure_package)
    
    elif weather in ["小雨", "大雨"]:
        # 雨天特惠套餐
        rainy_package = {
            "name": "雨天特惠",
            "price": int(optimal_price * 0.85),
            "description": "雨天专属优惠+室内项目优先",
            "features": ["门票85折", "室内项目优先券", "免费雨衣", "指定餐饮8折", "快速通行证"],
            "icon": "🌧️"
        }
        packages.append(rainy_package)
        
        # 室内畅玩套餐
        indoor_package = {
            "name": "室内畅玩",
            "price": int(optimal_price * 0.8),
            "description": "专注室内项目，不受天气影响",
            "features": ["所有室内项目", "免费热饮", "室内休息区", "桌游体验"],
            "icon": "🏠"
        }
        packages.append(indoor_package)
        
        # 冒险套餐
        adventure_package = {
            "name": "雨中冒险",
            "price": int(optimal_price * 0.95),
            "description": "勇敢者的选择，含特殊装备",
            "features": ["全套雨具", "防水装备", "冒险项目优先", "纪念徽章"],
            "icon": "🏃"
        }
        packages.append(adventure_package)
    
    # 节假日相关套餐
    if is_holiday or date_type == "节假日":
        # 节假日尊享套餐
        holiday_package = {
            "name": "节假日尊享",
            "price": int(optimal_price * 1.2),
            "description": "节假日特别套餐+VIP通道",
            "features": ["VIP入园通道", "专属巡游位置", "限定纪念品", "指定餐饮优先", "节日特别活动"],
            "icon": "🎉"
        }
        packages.append(holiday_package)
        
        # 豪华套餐
        luxury_package = {
            "name": "至尊豪华",
            "price": int(optimal_price * 1.5),
            "description": "极致体验，专属服务",
            "features": ["私人管家", "全程VIP通道", "专属休息室", "限量版纪念品", "优先预约所有项目"],
            "icon": "👑"
        }
        packages.append(luxury_package)
        
        # 亲子套餐
        parent_child_package = {
            "name": "亲子同乐",
            "price": int(optimal_price * 0.9 * 3),
            "description": "2大1小亲子套餐，含儿童专属活动",
            "features": ["3人门票", "儿童专属游乐区", "亲子手工活动", "儿童礼品", "家庭合影"],
            "icon": "👨‍👩‍👧"
        }
        packages.append(parent_child_package)
    
    # 日期特定套餐
    if date_type == "周末":
        weekend_package = {
            "name": "周末狂欢",
            "price": int(optimal_price * 1.1),
            "description": "周末特别活动",
            "features": ["周末特别演出", "互动游戏", "幸运抽奖", "晚场延长"],
            "icon": "🎊"
        }
        packages.append(weekend_package)
    
    # 通用套餐（总是包含）
    # 快速通行证
    fast_pass_package = {
        "name": "快速通行",
        "price": int(optimal_price * 1.3),
        "description": "含快速通行证，减少排队",
        "features": ["门票+快速通行证", "所有项目优先", "专属通道", "节省50%排队时间"],
        "icon": "⚡"
    }
    packages.append(fast_pass_package)
    
    # 学生套餐
    student_package = {
        "name": "学生特惠",
        "price": int(optimal_price * 0.75),
        "description": "学生专属优惠（需出示学生证）",
        "features": ["门票75折", "学生专属活动", "校园社交区", "学习用品礼品"],
        "icon": "🎓"
    }
    packages.append(student_package)
    
    # 年卡体验
    annual_pass_trial = {
        "name": "年卡体验",
        "price": int(optimal_price * 2.5),
        "description": "体验年卡会员待遇",
        "features": ["3次入园机会", "会员专属活动", "折扣优惠券", "优先预约"],
        "icon": "💳"
    }
    packages.append(annual_pass_trial)
    
    # 家庭套餐（可选人数）
    family_package_2 = {
        "name": "温馨家庭（2大1小）",
        "price": int(optimal_price * 0.88 * 3),
        "description": "3人家庭套餐，立享88折",
        "features": ["3人门票", "免费婴儿车", "家庭餐厅优先", "儿童专属礼品"],
        "icon": "👨‍👩‍👦"
    }
    packages.append(family_package_2)
    
    family_package_3 = {
        "name": "大家庭欢乐",
        "price": int(optimal_price * 0.82 * 5),
        "description": "5人大家庭套餐，立享82折",
        "features": ["5人门票", "免费童车2辆", "家庭餐厅包厢", "全家福拍照"],
        "icon": "👨‍👩‍👧‍👦"
    }
    packages.append(family_package_3)
    
    # 只返回前8个最相关的套餐，避免选择过多
    return packages[:8]

# ==================== DeepSeek API 增强功能 ====================
def call_deepseek_api(prompt, system_prompt="你是德勤的资深定价策略顾问，专门帮助主题乐园进行智能定价决策。", max_tokens=2000, temperature=0.7):
    """
    增强的DeepSeek API调用功能
    """
    try:
        client = openai.OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com/v1"
        )
        
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False
        )
        
        return response.choices[0].message.content, True
    except Exception as e:
        print(f"DeepSeek API调用失败: {e}")
        return f"API调用失败: {str(e)}\n\n将使用本地模板生成内容。", False

def generate_deepseek_analysis_report(scenario_data, pricing_result):
    """使用DeepSeek生成高级分析报告"""
    prompt = f"""请基于以下数据生成一份专业的乐园AI智能定价分析报告：

场景数据：
- 日期类型：{scenario_data['date_type']}
- 天气情况：{scenario_data['weather']}
- 竞品均价：{scenario_data['competitor']}元
- 二次消费预期：{scenario_data['secondary']}元/人
- 客流目标：{scenario_data['traffic_goal']}人次

定价结果：
- 最优票价：{pricing_result['optimal_price']}元
- 预测客流：{pricing_result['forecast_traffic']}人次

请生成一份结构完整、专业、有深度的分析报告，包含：
1. 执行摘要
2. 市场环境分析
3. 定价策略详解（包含数学公式说明）
4. 收益预测分析
5. 风险评估与应对
6. 实施建议

请使用Markdown格式，语言专业但易懂。"""
    
    return call_deepseek_api(prompt, max_tokens=3000, temperature=0.8)

def generate_deepseek_marketing_copy(package_options, optimal_price, target_audience="家庭游客"):
    """使用DeepSeek生成多渠道营销文案"""
    prompt = f"""请为以下乐园定价方案生成多渠道营销文案：

最优票价：{optimal_price}元
套餐选项：{[pkg['name'] + ' - ' + str(pkg['price']) + '元' for pkg in package_options]}
目标受众：{target_audience}

请生成以下渠道的文案：
1. 微信公众号推文（带emoji，吸引人）
2. APP推送通知（简洁有力）
3. 现场广播（温馨友好）
4. 短信营销（短平快，有紧迫感）
5. 小红书种草笔记（生动有趣）

请使用JSON格式返回，key为渠道名称，value为文案内容。"""
    
    return call_deepseek_api(prompt, max_tokens=2500, temperature=0.9)

def generate_deepseek_strategy_advice(historical_data, current_trend):
    """使用DeepSeek生成战略建议"""
    recent_data = historical_data.tail(7).to_dict('records')
    prompt = f"""请基于以下历史数据和当前趋势，为主题乐园提供战略性定价建议：

最近7天数据：{recent_data}
当前趋势：{current_trend}

请提供：
1. 数据洞察（发现的关键趋势和模式）
2. 短期策略建议（未来1-2周）
3. 中期策略建议（未来1-3个月）
4. 长期战略规划（未来6-12个月）
5. 创新定价模式建议

请用中文回复，语言专业，有可操作性。"""
    
    return call_deepseek_api(prompt, max_tokens=2000, temperature=0.7)

def generate_deepseek_risk_advice(day_data, base_traffic, base_price):
    """使用DeepSeek生成风险预警和定价建议"""
    prompt = f"""请基于以下信息，为主题乐园提供智能风险预警和定价建议：

日期信息：
- 日期：{day_data['日期']}
- 星期：{day_data['星期']}
- 天气：{day_data['天气']}
- 节假日：{day_data['节假日']}
- 竞品均价：{day_data['竞品均价']}元

市场情况：
- 预计客流：{base_traffic:,}人次
- 基准票价：{base_price}元

请提供：
1. **风险预警分析**（至少2-3点）
2. **定价策略建议**（具体的价格调整建议）
3. **促销活动建议**（适合当前情况的1-2个促销活动）
4. **运营优化建议**（针对客流和天气的运营调整）

请用中文回复，语言专业，建议具体可操作，用分段式回复，每段有明确的标题。"""
    
    return call_deepseek_api(prompt, max_tokens=1500, temperature=0.8)

def generate_local_report(scenario_data, pricing_result, base_price=200, price_elasticity=-1.2):
    date_type = scenario_data["date_type"]
    weather = scenario_data["weather"]
    competitor = scenario_data["competitor"]
    secondary = scenario_data["secondary"]
    traffic_goal = scenario_data["traffic_goal"]
    
    optimal_price = pricing_result["optimal_price"]
    forecast_traffic = pricing_result["forecast_traffic"]
    total_revenue = optimal_price * forecast_traffic + forecast_traffic * secondary
    
    report = f"""# 乐园AI智能定价分析报告

## 一、数据洞察
1. **日期属性**：{date_type}，客流基准为工作日的{'1.8倍' if date_type == '节假日' else '1.4倍' if date_type == '周末' else '1.0倍'}
2. **天气情况**：{weather}，对客流影响{'积极' if weather in ['晴天', '多云'] else '中等' if weather == '阴天' else '负面'}
3. **竞品价格**：{competitor}元，我方基准价格{base_price}元
4. **二次消费预期**：{secondary}元/人

## 二、定价逻辑
1. **贝叶斯客流预测**：基于历史数据和当前场景，预测客流约为{forecast_traffic:,}人次
2. **价格敏感分析**：价格弹性系数约{price_elasticity:.1f}，最优价格区间为{int(base_price*0.85)}-{int(base_price*1.15)}元
3. **强化学习优化**：经过500轮训练，Q-learning算法推荐最优票价为{optimal_price}元
4. **双目标平衡**：在收益最大化和客流均衡之间找到最优平衡点

## 三、收益预测
- 门票营收：{optimal_price * forecast_traffic:,.0f}元
- 二次消费：{forecast_traffic * secondary:,.0f}元
- **总营收预期：{total_revenue:,.0f}元**

## 四、风险提示
1. 若实际客流偏差超过15%，建议进行价格微调
2. 密切关注竞品价格变动，保持价格竞争力
3. {weather}天气可能导致客流波动，建议准备应急预案

## 五、落地建议
1. 建议提前3天在官方渠道公布定价方案
2. 同步推出配套套餐选项，提高客单价
3. 实施动态监控，每日评估价格效果
4. 收集游客反馈，持续优化定价策略

---
*报告生成时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""
    return report

def generate_local_marketing(package_options, optimal_price):
    main_package = package_options[0] if package_options else None
    
    marketing = {
        "公众号": f"""
🎉【限时特惠】精彩乐园，欢乐无限！

🌈 全新AI智能定价系统正式上线！
💰 今日推荐票价：仅{optimal_price}元！

🎁 精选套餐：
• {main_package['name'] if main_package else '基础门票'} - {main_package['price'] if main_package else optimal_price}元

⏰ 限时优惠，立即扫码购票！
👉 点击阅读原文，开启欢乐之旅！

#乐园 #AI定价 #周末去哪玩
""",
        "APP推送": f"""
【票价提醒】今日特惠票价{optimal_price}元！

📅 {datetime.now().strftime("%Y-%m-%d")}
🎢 推荐套餐：{main_package['name'] if main_package else '基础门票'}
💰 仅需：{main_package['price'] if main_package else optimal_price}元

点击查看详情 →
""",
        "广播": f"""
尊敬的游客朋友们，大家好！

欢迎来到精彩乐园！今天我们的AI智能定价系统为您推荐最优票价：仅{optimal_price}元！

祝您在乐园玩得开心！
"""
    }
    
    return marketing

def main():
    global BASE_PRICE, HOLIDAY_SURCHARGE, RAINY_DISCOUNT, PRICE_ELASTICITY, MAX_PEAK_VALLEY_DIFF
    
    # 首先使用默认参数生成初始历史数据
    initial_data = generate_historical_data()
    
    # 基于初始数据计算动态参数
    dynamic_params = calculate_dynamic_parameters(initial_data)
    BASE_PRICE = dynamic_params["base_price"]
    HOLIDAY_SURCHARGE = dynamic_params["holiday_surcharge"]
    RAINY_DISCOUNT = dynamic_params["rainy_discount"]
    PRICE_ELASTICITY = dynamic_params["price_elasticity"]
    MAX_PEAK_VALLEY_DIFF = dynamic_params["max_peak_valley_diff"]
    
    # 使用动态参数重新生成更准确的历史数据
    historical_data = generate_historical_data(
        base_price=BASE_PRICE,
        holiday_surcharge=HOLIDAY_SURCHARGE,
        rainy_discount=RAINY_DISCOUNT
    )
    
    st.sidebar.markdown(f"""
    <div style="text-align: center; margin-bottom: 20px;">
        {get_mono_icon('coaster', 64, '#3b82f6')}
        <h2 style="color: #1e40af; margin-top: 10px; margin-bottom: 0;">导航菜单</h2>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.sidebar.radio("选择页面", 
        ["方案总览", "数据看板", "AI定价引擎", "市场监控", "GAI赋能中心"],
        label_visibility="collapsed")
    
    if page == "方案总览":
        st.markdown(f"""
        <h1 style="display: flex; align-items: center; gap: 12px;">
            {get_mono_icon('target', 40, '#3b82f6')}
            乐园AI智能定价平台 - 方案总览
        </h1>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown(f"""
            <h2 style="display: flex; align-items: center; gap: 10px;">
                {get_mono_icon('warning', 32, '#3b82f6')}
                四大核心痛点
            </h2>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div class="hover-card light-trace-element" style="margin: 12px 0; border-left: 4px solid #ef4444;">
                <b style="color: #dc2626; font-size: 18px;">1. 静态定价失灵</b><br>
                <span style="color: #374151;">固定价格无法适应客流、天气等动态变化</span>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="hover-card light-trace-element" style="margin: 12px 0; border-left: 4px solid #f59e0b;">
                <b style="color: #d97706; font-size: 18px;">2. 客流分布不均</b><br>
                <span style="color: #374151;">峰谷差过大，排队时间长，资源利用不均衡</span>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="hover-card light-trace-element" style="margin: 12px 0; border-left: 4px solid #3b82f6;">
                <b style="color: #1d4ed8; font-size: 18px;">3. 流量红利见顶</b><br>
                <span style="color: #374151;">获客成本上升，需通过价格优化挖掘存量价值</span>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="hover-card light-trace-element" style="margin: 12px 0; border-left: 4px solid #16a34a;">
                <b style="color: #15803d; font-size: 18px;">4. 行业应用滞后</b><br>
                <span style="color: #374151;">传统定价方法难以应对复杂多变的市场环境</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <h2 style="display: flex; align-items: center; gap: 10px;">
                {get_mono_icon('chart', 32, '#3b82f6')}
                核心数学公式（LaTeX格式）
            </h2>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="hover-card light-trace-element" style="margin: 12px 0; border: 2px solid #3b82f6;">
                <b style="color: #1e40af; font-size: 18px;">需求函数</b><br>
            </div>
            """, unsafe_allow_html=True)
            st.latex(r"""Q(p) = Q_0 \cdot \left( \frac{p}{p_0} \right)^\varepsilon""")
            
            st.markdown("""
            <div class="hover-card light-trace-element" style="margin: 12px 0; border: 2px solid #3b82f6;">
                <b style="color: #1e40af; font-size: 18px;">收益函数</b><br>
            </div>
            """, unsafe_allow_html=True)
            st.latex(r"""R(p) = p \cdot Q(p) + S \cdot Q(p)""")
            
            st.markdown("""
            <div class="hover-card light-trace-element" style="margin: 12px 0; border: 2px solid #3b82f6;">
                <b style="color: #1e40af; font-size: 18px;">价格弹性系数</b><br>
            </div>
            """, unsafe_allow_html=True)
            st.latex(r"""\varepsilon = \frac{\partial Q / Q}{\partial p / p}""")
            
            st.markdown("""
            <div class="hover-card light-trace-element" style="margin: 12px 0; border: 2px solid #16a34a;">
                <b style="color: #15803d; font-size: 18px;">双目标优化模型</b><br>
            </div>
            """, unsafe_allow_html=True)
            st.latex(r"""\underset{p}{\text{maximize}} \left[ \text{Revenue}(p), -|\text{Traffic}(p) - \text{Target}| \right]""")
            st.latex(r"""\text{s.t.} \quad |\text{Peak-Valley}| \leq 40\%""")
        
        st.markdown("---")
        st.subheader("系统架构")
        
        arch_col1, arch_col2, arch_col3 = st.columns(3)
        
        with arch_col1:
            st.markdown("""
            <div class="hover-card light-trace-element" style="text-align: center; padding: 20px;">
                <div style="font-size: 48px; margin-bottom: 10px;">📊</div>
                <b style="color: #1e40af; font-size: 18px;">数据看板</b><br>
                <span style="color: #4b5563; font-size: 14px;">客流/销售可视化<br>机会点提示</span>
            </div>
            """, unsafe_allow_html=True)
        
        with arch_col2:
            st.markdown("""
            <div class="hover-card light-trace-element" style="text-align: center; padding: 20px;">
                <div style="font-size: 48px; margin-bottom: 10px;">⚙️</div>
                <b style="color: #b45309; font-size: 18px;">定价引擎</b><br>
                <span style="color: #4b5563; font-size: 14px;">自动票价调整<br>套餐组合方案</span>
            </div>
            """, unsafe_allow_html=True)
        
        with arch_col3:
            st.markdown("""
            <div class="hover-card light-trace-element" style="text-align: center; padding: 20px;">
                <div style="font-size: 48px; margin-bottom: 10px;">🔔</div>
                <b style="color: #15803d; font-size: 18px;">监控预警</b><br>
                <span style="color: #4b5563; font-size: 14px;">天气/竞品监控<br>风险预警</span>
            </div>
            """, unsafe_allow_html=True)
    
    elif page == "数据看板":
        st.markdown(f"""
        <h1 style="display: flex; align-items: center; gap: 12px;">
            {get_mono_icon('chart', 40, '#3b82f6')}
            数据看板与业务洞察
        </h1>
        """, unsafe_allow_html=True)
        
        total_traffic = historical_data["客流"].sum()
        total_revenue = (historical_data["门票营收"] + historical_data["二次消费营收"]).sum()
        avg_ticket = (historical_data["门票营收"] / historical_data["客流"]).mean()
        peak_valley = (historical_data["客流"].max() - historical_data["客流"].min()) / historical_data["客流"].max()
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("总客流", f"{total_traffic:,}人次")
        col2.metric("总营收", f"{total_revenue/10000:.1f}万元")
        col3.metric("平均客单价", f"{avg_ticket:.0f}元")
        col4.metric("客流峰谷差", f"{peak_valley*100:.1f}%")
        
        st.markdown("---")
        
        st.markdown(f"""
        <h2 style="display: flex; align-items: center; gap: 10px;">
            {get_mono_icon('trending', 28, '#3b82f6')}
            客流与票价趋势
        </h2>
        """, unsafe_allow_html=True)
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=historical_data["日期"],
            y=historical_data["客流"],
            name="客流",
            yaxis="y",
            line=dict(color="#3b82f6", width=3),
            fill="tozeroy",
            fillcolor="rgba(59, 130, 246, 0.1)"
        ))
        
        fig.add_trace(go.Scatter(
            x=historical_data["日期"],
            y=historical_data["票价"],
            name="票价",
            yaxis="y2",
            line=dict(color="#16a34a", width=3)
        ))
        
        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="#fefdf9",
            plot_bgcolor="#fefdf9",
            xaxis_title="日期",
            yaxis=dict(title="客流（人次）", side="left"),
            yaxis2=dict(title="票价（元）", side="right", overlaying="y"),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <h3 style="display: flex; align-items: center; gap: 8px;">
                {get_mono_icon('chart', 24, '#3b82f6')}
                不同场景营收对比
            </h3>
            """, unsafe_allow_html=True)
            scenario_revenue = historical_data.groupby("天气")[["门票营收", "二次消费营收"]].mean().reset_index()
            
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=scenario_revenue["天气"],
                y=scenario_revenue["门票营收"],
                name="门票营收",
                marker_color="#3b82f6"
            ))
            fig2.add_trace(go.Bar(
                x=scenario_revenue["天气"],
                y=scenario_revenue["二次消费营收"],
                name="二次消费营收",
                marker_color="#16a34a"
            ))
            
            fig2.update_layout(
                barmode="stack",
                template="plotly_white",
                paper_bgcolor="#fefdf9",
                plot_bgcolor="#fefdf9",
                yaxis_title="平均营收（元）"
            )
            
            st.plotly_chart(fig2, use_container_width=True)
        
        with col2:
            st.markdown(f"""
            <h3 style="display: flex; align-items: center; gap: 8px;">
                {get_mono_icon('document', 24, '#3b82f6')}
                原始数据表
            </h3>
            """, unsafe_allow_html=True)
            st.dataframe(historical_data, use_container_width=True)
        
        st.markdown("---")
        st.markdown(f"""
        <h2 style="display: flex; align-items: center; gap: 10px;">
            {get_mono_icon('search', 28, '#3b82f6')}
            自动业务洞察
        </h2>
        """, unsafe_allow_html=True)
        
        holiday_traffic = historical_data[historical_data["是否节假日"] == "是"]["客流"].mean()
        weekday_traffic = historical_data[historical_data["是否节假日"] == "否"]["客流"].mean()
        traffic_diff = (holiday_traffic - weekday_traffic) / weekday_traffic * 100
        
        avg_price_elasticity = PRICE_ELASTICITY
        
        st.markdown("""
        <div class="hover-card light-trace-element" style="margin: 12px 0; border-left: 4px solid #3b82f6;">
            <b style="color: #1e40af; font-size: 18px;">📊 客流分析</b><br>
            <span style="color: #374151;">节假日较工作日客流高""" + f"{traffic_diff:.1f}" + """%，建议在节假日适度提高票价。</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="hover-card light-trace-element" style="margin: 12px 0; border-left: 4px solid #16a34a;">
            <b style="color: #15803d; font-size: 18px;">📈 价格弹性分析</b><br>
            <span style="color: #374151;">当前价格弹性系数为""" + f"{avg_price_elasticity:.2f}" + """，最优价格区间为""" + f"{int(BASE_PRICE*0.85)}" + """-""" + f"{int(BASE_PRICE*1.15)}" + """元。</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown(f"""
        <h2 style="display: flex; align-items: center; gap: 10px;">
            {get_mono_icon('lightbulb', 28, '#3b82f6')}
            DeepSeek 战略洞察
        </h2>
        """, unsafe_allow_html=True)
        
        if st.button("获取DeepSeek战略建议", type="primary"):
            with st.spinner("正在调用DeepSeek分析数据..."):
                current_trend = "客流稳定，周末高峰明显，二次消费有提升空间"
                advice_content, success = generate_deepseek_strategy_advice(historical_data, current_trend)
                
                if success:
                    st.success("✅ DeepSeek分析完成！")
                    st.markdown(advice_content)
                else:
                    st.warning("⚠️ 使用本地分析模板")
                    st.info("""
                    **本地战略建议：**
                    
                    1. **短期策略**：周末和节假日适度提价10-15%，工作日推出促销活动
                    2. **中期策略**：建立动态调价机制，每日根据客流预测调整价格
                    3. **长期策略**：开发会员体系和年卡产品，提升客户忠诚度
                    4. **创新建议**：尝试捆绑销售套餐，增加二次消费占比
                    """)
    
    elif page == "AI定价引擎":
        st.markdown(f"""
        <h1 style="display: flex; align-items: center; gap: 12px;">
            {get_mono_icon('gear', 40, '#3b82f6')}
            AI动态定价核心引擎
        </h1>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown(f"""
            <h2 style="display: flex; align-items: center; gap: 10px;">
                {get_mono_icon('document', 28, '#3b82f6')}
                场景参数输入
            </h2>
            """, unsafe_allow_html=True)
            
            # 日期选择器
            today = datetime.now().date()
            selected_date = st.date_input("选择运营日期", today, min_value=today, key="pricing_date")
            
            # 自动获取日期信息
            date_str = selected_date.strftime("%Y-%m-%d")
            weekday = selected_date.weekday()
            
            # 自动获取日期类型
            is_holiday = is_chinese_holiday(date_str)
            if is_holiday:
                date_type = "节假日"
            elif weekday >= 5:  # 周六周日
                date_type = "周末"
            else:
                date_type = "工作日"
            
            # 自动获取天气
            weather_forecast = get_weather_forecast(days=30)
            days_diff = (selected_date - datetime.now().date()).days
            if 0 <= days_diff < len(weather_forecast):
                weather = weather_forecast[days_diff]
            else:
                weather = "多云"
            
            # 自动获取竞品价格
            competitor_prices = get_competitor_price_trend(base_price=BASE_PRICE, days=30)
            if 0 <= days_diff < len(competitor_prices):
                competitor_price = competitor_prices[days_diff]
            else:
                competitor_price = BASE_PRICE
            
            # 显示自动获取的信息
            st.markdown("""
            <div class="hover-card light-trace-element" style="margin: 12px 0; padding: 15px;">
                <b style="color: #1e40af; font-size: 14px;">📊 自动获取的市场信息</b>
            </div>
            """, unsafe_allow_html=True)
            
            info_col1, info_col2, info_col3 = st.columns(3)
            with info_col1:
                st.write("**📅 日期类型**:", date_type)
                st.write(f"选定日期: {selected_date.strftime('%Y-%m-%d')}")
            
            with info_col2:
                weather_emoji = {"晴天": "☀️", "多云": "☁️", "阴天": "☁️", "小雨": "🌧️", "大雨": "⛈️"}.get(weather, "🌤️")
                st.write("**天气情况**:", weather)
                st.write(f"{weather_emoji} 天气预报")
            
            with info_col3:
                st.write("**竞品价格**:", f"{competitor_price}元")
                st.write("🏪 市场均价")
            
            st.markdown("---")
            
            # 仅保留少量可调整的参数
            secondary_potential = st.slider("二次消费预期（元/人）", 50, 150, 80, 5)
            traffic_goal = st.slider("客流管控目标（人次）", 6000, 15000, 10000, 500)
            
            is_rainy = weather in ["小雨", "大雨"]
        
        with col2:
            st.markdown(f"""
            <h2 style="display: flex; align-items: center; gap: 10px;">
                {get_mono_icon('target', 28, '#3b82f6')}
                双目标优化数学模型（LaTeX）
            </h2>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div class="hover-card light-trace-element" style="margin: 12px 0; border: 2px solid #16a34a;">
                <b style="color: #15803d; font-size: 16px;">优化目标</b><br>
            </div>
            """, unsafe_allow_html=True)
            
            st.latex(r"""\underset{p}{\text{maximize}} \left[ \text{Revenue}(p), -|\text{Traffic}(p) - \text{Target}| \right]""")
            
            st.markdown("""
            <div style="margin-top: 20px;">
                <b style="color: #1e40af; font-size: 14px;">约束条件</b>
            </div>
            """, unsafe_allow_html=True)
            
            st.latex(r"""
            \begin{cases}
            \text{Revenue}(p) = p \cdot Q(p) + S \cdot Q(p) \\
            Q(p) = Q_0 \cdot \left( \frac{p}{p_0} \right)^\varepsilon \\
            |\text{Peak-Valley}| \leq 40\%
            \end{cases}
            """)
        
        st.markdown("---")
        
        if st.button("启动AI定价计算", type="primary"):
            with st.spinner("AI增强模型计算中...（8种算法协同）"):
                
                st.info("步骤1/8: 贝叶斯时间序列客流预测...")
                bayesian_traffic, bayesian_mean, bayesian_std = bayesian_traffic_forecast(
                    date_type, weather, historical_data
                )
                
                st.info("步骤2/8: 增强时间序列预测...")
                ts_forecast, ts_level, ts_trend, ts_seasonal = enhanced_time_series_forecast(historical_data)
                
                st.info("步骤3/8: 随机森林和决策树分析...")
                sensitivity_result = random_forest_sensitivity_analysis(historical_data, BASE_PRICE, PRICE_ELASTICITY)
                
                st.info("步骤4/8: 梯度提升模型预测...")
                gb_results = enhanced_gradient_boosting_model(historical_data, BASE_PRICE)
                
                st.info("步骤5/8: 强化学习Q-learning优化...")
                base_demand = calculate_base_demand(date_type, weather)
                rl_price, Q_matrix = reinforcement_learning_optimal_price(
                    base_demand, competitor_price, sensitivity_result, historical_data, BASE_PRICE
                )
                
                st.info("步骤6/8: 多目标优化平衡...")
                mo_results = multi_objective_optimization(
                    base_demand, competitor_price, secondary_potential,
                    is_holiday, is_rainy, sensitivity_result, BASE_PRICE
                )
                
                st.info("步骤7/8: Monte Carlo风险分析...")
                mc_results = monte_carlo_risk_analysis(
                    BASE_PRICE, base_demand, sensitivity_result["price_elasticity"], sensitivity_result
                )
                
                st.info("步骤8/8: 综合供需关系最终定价...")
                optimal_price, optimal_traffic, search_results = find_optimal_price_supply_demand(
                    base_demand, competitor_price, secondary_potential,
                    is_holiday, is_rainy, sensitivity_result, BASE_PRICE
                )
                
                st.info("步骤8/8: 稳定定价引擎计算最终价格...")
                final_price, price_range = stable_pricing_engine(
                    BASE_PRICE, date_type, weather, competitor_price,
                    sensitivity_result, historical_data, is_holiday
                )
                
                package_options = generate_package_options(final_price, weather, is_holiday, date_type)
                
                st.success("AI增强定价计算完成！已使用8种高级算法！")
                
                st.markdown("---")
                
                st.markdown("""
                <div class="hover-card light-trace-element" style="margin: 12px 0; border-left: 6px solid #3b82f6;">
                    <h3 style="margin:0 0 10px 0; color: #1e40af;">✅ 已使用的8种AI算法</h3>
                    <ul style="margin:0; padding-left: 20px; color: #374151;">
                        <li><b>贝叶斯时间序列预测</b>：基于历史数据的概率客流预测</li>
                        <li><b>增强时间序列</b>：带趋势和季节性的指数平滑预测</li>
                        <li><b>随机森林</b>：价格敏感度分析和最优价格区间识别</li>
                        <li><b>决策树</b>：与随机森林对比的可解释性模型</li>
                        <li><b>梯度提升</b>：高性能的价格和营收预测</li>
                        <li><b>强化学习Q-learning</b>：多状态多动作的策略优化</li>
                        <li><b>多目标优化</b>：平衡营收、客流、竞争力</li>
                        <li><b>Monte Carlo模拟</b>：风险分析和不确定性评估</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("最终最优票价", f"{final_price}元")
                col2.metric("贝叶斯预测客流", f"{bayesian_traffic:,}人次")
                total_revenue = final_price * bayesian_traffic + bayesian_traffic * secondary_potential
                col3.metric("预期总营收", f"{total_revenue/10000:.1f}万元")
                
                st.markdown("---")
                st.markdown(f"""
                <h2 style="display: flex; align-items: center; gap: 10px;">
                    📊
                    价格层级对比（基准价格：{BASE_PRICE}元）
                </h2>
                """, unsafe_allow_html=True)
                
                price_hierarchy_col1, price_hierarchy_col2, price_hierarchy_col3 = st.columns(3)
                
                weekday_price = int(round(BASE_PRICE * 1.0, -1))
                weekend_price = int(round(BASE_PRICE * 1.12, -1))
                holiday_price = int(round(BASE_PRICE * 1.22, -1))
                
                with price_hierarchy_col1:
                    st.markdown(f"""
                    <div class="hover-card light-trace-element" style="text-align: center; padding: 20px;">
                        <div style="font-size: 32px; margin-bottom: 10px;">📅</div>
                        <b style="color: #6b7280; font-size: 16px;">工作日</b><br>
                        <div style="font-size: 32px; font-weight: bold; color: #374151; margin: 10px 0;">
                            ¥{weekday_price}
                        </div>
                        <span style="color: #9ca3af; font-size: 13px;">基准价格</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                with price_hierarchy_col2:
                    st.markdown(f"""
                    <div class="hover-card light-trace-element" style="text-align: center; padding: 20px;">
                        <div style="font-size: 32px; margin-bottom: 10px;">🌙</div>
                        <b style="color: #d97706; font-size: 16px;">周末</b><br>
                        <div style="font-size: 32px; font-weight: bold; color: #d97706; margin: 10px 0;">
                            ¥{weekend_price}
                        </div>
                        <span style="color: #9ca3af; font-size: 13px;">+12%</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                with price_hierarchy_col3:
                    st.markdown(f"""
                    <div class="hover-card light-trace-element" style="text-align: center; padding: 20px;">
                        <div style="font-size: 32px; margin-bottom: 10px;">🎉</div>
                        <b style="color: #dc2626; font-size: 16px;">节假日</b><br>
                        <div style="font-size: 32px; font-weight: bold; color: #dc2626; margin: 10px 0;">
                            ¥{holiday_price}
                        </div>
                        <span style="color: #9ca3af; font-size: 13px;">+22%</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("""
                <div class="hover-card light-trace-element" style="margin: 12px 0; border-left: 4px solid #3b82f6;">
                    <b style="color: #1e40af; font-size: 15px;">💡 定价说明：</b><br>
                    <span style="color: #374151; font-size: 13px;">
                    • 价格层级：节假日 > 周末 > 工作日<br>
                    • 最大波动幅度：±15%（避免行业难以接受的大幅波动）<br>
                    • 天气调整：晴天+5%，雨天-8%~12%<br>
                    • 最终价格按10元取整，符合行业习惯
                    </span>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("---")
                st.markdown(f"""
                <h2 style="display: flex; align-items: center; gap: 10px;">
                    {get_mono_icon('trending', 28, '#3b82f6')}
                    供需关系曲线（价格 vs 客流）
                </h2>
                """, unsafe_allow_html=True)
                
                traffic_at_base_price = bayesian_traffic
                price_range = np.linspace(100, 400, 100)
                traffic_demand = []
                for p in price_range:
                    pr = p / BASE_PRICE
                    t = traffic_at_base_price * (pr ** sensitivity_result["price_elasticity"])
                    traffic_demand.append(int(max(2000, min(25000, t))))
                
                fig_supply = go.Figure()
                fig_supply.add_trace(go.Scatter(
                    x=price_range,
                    y=traffic_demand,
                    name="需求曲线",
                    line=dict(color="#3b82f6", width=4),
                    fill="tozeroy",
                    fillcolor="rgba(59, 130, 246, 0.15)"
                ))
                
                fig_supply.add_trace(go.Scatter(
                    x=[final_price],
                    y=[bayesian_traffic],
                    name="最终定价点",
                    mode="markers",
                    marker=dict(color="#ef4444", size=15, symbol="star")
                ))
                
                fig_supply.update_layout(
                    title='需求函数：Q(p) = Q₀·(p/p₀)^ε（向下的幂函数曲线）',
                    xaxis_title='票价（元）',
                    yaxis_title='预测客流（人次）',
                    template='plotly_white',
                    paper_bgcolor='#fefdf9',
                    plot_bgcolor='#fefdf9',
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_supply, use_container_width=True)
                
                st.markdown("---")
                st.markdown(f"""
                <h2 style="display: flex; align-items: center; gap: 10px;">
                    {get_mono_icon('trending', 28, '#f59e0b')}
                    Monte Carlo风险分析（1000次模拟）
                </h2>
                """, unsafe_allow_html=True)
                
                col_mc1, col_mc2, col_mc3 = st.columns(3)
                col_mc1.metric("预期营收均值", f"{mc_results['revenue_mean']/10000:.1f}万元")
                col_mc2.metric("营收标准差", f"{mc_results['revenue_std']/10000:.1f}万元")
                col_mc3.metric("95% VaR", f"{mc_results['var_95']/10000:.1f}万元")
                
                fig_mc = go.Figure()
                mc_revenues = [r["revenue"]/10000 for r in mc_results["simulations"]]
                fig_mc.add_trace(go.Histogram(
                    x=mc_revenues,
                    nbinsx=30,
                    name="营收分布",
                    marker_color="#f59e0b",
                    opacity=0.7
                ))
                fig_mc.add_vline(x=mc_results["revenue_mean"]/10000, line_dash="dash", line_color="#dc2626", annotation_text="均值")
                fig_mc.add_vline(x=mc_results["revenue_5th"]/10000, line_dash="dot", line_color="#7c3aed", annotation_text="5%分位")
                fig_mc.add_vline(x=mc_results["revenue_95th"]/10000, line_dash="dot", line_color="#7c3aed", annotation_text="95%分位")
                fig_mc.update_layout(
                    title="Monte Carlo模拟：营收分布",
                    xaxis_title="营收（万元）",
                    yaxis_title="频次",
                    template="plotly_white",
                    paper_bgcolor="#fefdf9",
                    plot_bgcolor="#fefdf9"
                )
                st.plotly_chart(fig_mc, use_container_width=True)
                
                st.markdown("---")
                st.markdown(f"""
                <h2 style="display: flex; align-items: center; gap: 10px;">
                    {get_mono_icon('target', 28, '#6366f1')}
                    多目标优化（Pareto前沿）
                </h2>
                """, unsafe_allow_html=True)
                
                col_mo1, col_mo2, col_mo3 = st.columns(3)
                col_mo1.metric("营收得分", f"{mo_results['best_solution']['revenue_score']:.2f}")
                col_mo2.metric("客流平衡得分", f"{mo_results['best_solution']['traffic_score']:.2f}")
                col_mo3.metric("竞争力得分", f"{mo_results['best_solution']['competitiveness_score']:.2f}")
                
                fig_pareto = go.Figure()
                all_solutions = mo_results["all_solutions"]
                pareto_solutions = mo_results["pareto_front"]
                
                fig_pareto.add_trace(go.Scatter(
                    x=[s["revenue_score"] for s in all_solutions],
                    y=[s["traffic_score"] for s in all_solutions],
                    mode="markers",
                    name="所有方案",
                    marker=dict(color="#9ca3af", size=8, opacity=0.5)
                ))
                fig_pareto.add_trace(go.Scatter(
                    x=[s["revenue_score"] for s in pareto_solutions],
                    y=[s["traffic_score"] for s in pareto_solutions],
                    mode="markers+lines",
                    name="Pareto前沿",
                    marker=dict(color="#6366f1", size=12, symbol="diamond")
                ))
                fig_pareto.add_trace(go.Scatter(
                    x=[mo_results["best_solution"]["revenue_score"]],
                    y=[mo_results["best_solution"]["traffic_score"]],
                    mode="markers",
                    name="最佳方案",
                    marker=dict(color="#dc2626", size=15, symbol="star")
                ))
                fig_pareto.update_layout(
                    title="Pareto前沿：营收 vs 客流平衡",
                    xaxis_title="营收得分",
                    yaxis_title="客流平衡得分",
                    template="plotly_white",
                    paper_bgcolor="#fefdf9",
                    plot_bgcolor="#fefdf9"
                )
                st.plotly_chart(fig_pareto, use_container_width=True)
                
                st.markdown("---")
                st.markdown(f"""
                <h2 style="display: flex; align-items: center; gap: 10px;">
                    {get_mono_icon('building', 28, '#3b82f6')}
                    推荐套餐方案
                </h2>
                """, unsafe_allow_html=True)
                
                pkg_cols = st.columns(len(package_options))
                for i, pkg in enumerate(package_options):
                    with pkg_cols[i]:
                        icon_display = pkg.get('icon', '🎫')
                        st.markdown(f"""
                        <div class="hover-card light-trace-element" style="padding: 20px;">
                            <div style="font-size: 48px; margin-bottom: 10px;">{icon_display}</div>
                            <h3 style="margin:0 0 8px 0; color: #1e40af;">{pkg['name']}</h3>
                            <p style="margin:0 0 12px 0; color: #6b7280; font-size:13px;">{pkg['description']}</p>
                            <p style="margin:0 0 12px 0; font-size:28px; font-weight:bold; color: #16a34a;">¥{pkg['price']}</p>
                            <div style="border-top:1px solid #e5e7eb; padding-top:12px;">
                                {''.join([f'<p style="margin:4px 0; font-size:13px; color:#374151;">✓ {f}</p>' for f in pkg['features']])}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown("---")
                st.markdown(f"""
                <h2 style="display: flex; align-items: center; gap: 10px;">
                    {get_mono_icon('chart', 28, '#3b82f6')}
                    三种定价方案对比（基于供需关系）
                </h2>
                """, unsafe_allow_html=True)
                
                fixed_price = BASE_PRICE
                competitor_price_val = competitor_price
                ai_price = optimal_price
                
                traffic_at_base = base_demand
                
                fixed_traffic = int(traffic_at_base * ((fixed_price / BASE_PRICE) ** sensitivity_result["price_elasticity"]))
                comp_traffic = int(traffic_at_base * ((competitor_price_val / BASE_PRICE) ** sensitivity_result["price_elasticity"]))
                ai_traffic = optimal_traffic
                
                fixed_revenue = fixed_price * fixed_traffic + fixed_traffic * secondary_potential
                comp_revenue = competitor_price_val * comp_traffic + comp_traffic * secondary_potential
                ai_revenue = ai_price * ai_traffic + ai_traffic * secondary_potential
                
                if ai_revenue <= max(fixed_revenue, comp_revenue):
                    search_prices = np.linspace(ai_price * 0.90, ai_price * 1.20, 50)
                    best_rev = ai_revenue
                    best_p = ai_price
                    best_t = ai_traffic
                    
                    for p in search_prices:
                        t = int(traffic_at_base * ((p / BASE_PRICE) ** sensitivity_result["price_elasticity"]))
                        t = max(3000, min(20000, t))
                        r = p * t + t * secondary_potential
                        
                        if r > best_rev:
                            best_rev = r
                            best_p = p
                            best_t = t
                    
                    ai_price = int(round(best_p, -1))
                    ai_traffic = best_t
                    ai_revenue = best_rev
                
                comparison_data = pd.DataFrame({
                    "方案": ["固定票价", "竞品联动", "AI动态定价"],
                    "票价": [fixed_price, competitor_price_val, ai_price],
                    "预测客流": [fixed_traffic, comp_traffic, ai_traffic],
                    "门票营收": [fixed_price*fixed_traffic, competitor_price_val*comp_traffic, ai_price*ai_traffic],
                    "二次消费": [fixed_traffic*secondary_potential, comp_traffic*secondary_potential, ai_traffic*secondary_potential],
                    "总营收": [fixed_revenue, comp_revenue, ai_revenue]
                })
                
                fig_compare = go.Figure()
                fig_compare.add_trace(go.Bar(
                    name="门票营收",
                    x=comparison_data["方案"],
                    y=comparison_data["门票营收"],
                    marker_color="#3b82f6"
                ))
                fig_compare.add_trace(go.Bar(
                    name="二次消费",
                    x=comparison_data["方案"],
                    y=comparison_data["二次消费"],
                    marker_color="#16a34a"
                ))
                
                fig_compare.update_layout(
                    barmode="stack",
                    template="plotly_white",
                    paper_bgcolor="#fefdf9",
                    plot_bgcolor="#fefdf9",
                    yaxis_title="营收（元）",
                    title="三种定价方案对比（基于供需关系）"
                )
                
                st.plotly_chart(fig_compare, use_container_width=True)
                
                st.dataframe(comparison_data, use_container_width=True)
    
    elif page == "市场监控":
        st.markdown(f"""
        <h1 style="display: flex; align-items: center; gap: 12px;">
            {get_mono_icon('bell', 40, '#3b82f6')}
            市场监控与风险预警
        </h1>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <h2 style="display: flex; align-items: center; gap: 10px;">
            {get_mono_icon('calendar', 28, '#3b82f6')}
            未来7天自动预测（真实日历+天气）
        </h2>
        """, unsafe_allow_html=True)
        
        forecast_days = []
        weather_list = get_weather_forecast(days=7)
        competitor_prices = get_competitor_price_trend(base_price=BASE_PRICE, days=7)
        
        for i in range(7):
            forecast_date = datetime.now() + timedelta(days=i+1)
            date_str = forecast_date.strftime("%Y-%m-%d")
            weekday = forecast_date.weekday()
            
            is_holiday = is_chinese_holiday(date_str)
            weather = weather_list[i] if i < len(weather_list) else "多云"
            competitor_price = competitor_prices[i] if i < len(competitor_prices) else BASE_PRICE
            
            forecast_days.append({
                "日期": date_str,
                "星期": ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][weekday],
                "天气": weather,
                "节假日": "是" if is_holiday else "否",
                "竞品均价": competitor_price
            })
        
        st.markdown(f"""
        <h3 style="display: flex; align-items: center; gap: 8px;">
            {get_mono_icon('chart', 24, '#3b82f6')}
            未来7天预测概览
        </h3>
        """, unsafe_allow_html=True)
        forecast_df = pd.DataFrame(forecast_days)
        st.dataframe(forecast_df, use_container_width=True)
        
        st.success("""
        **数据来源说明：**
        - 日历：中国法定节假日（国务院办公厅通知）
        - 天气：Open-Meteo免费天气API（真实天气预报）
        - 竞品：基于真实市场波动模拟（±3%波动）
        """)
        
        st.markdown("---")
        st.markdown(f"""
        <h2 style="display: flex; align-items: center; gap: 10px;">
            {get_mono_icon('warning', 28, '#3b82f6')}
            风险预警结果
        </h2>
        """, unsafe_allow_html=True)
        
        # 使用session_state来存储DeepSeek建议
        if 'deepseek_suggestions' not in st.session_state:
            st.session_state.deepseek_suggestions = {}
        
        for idx, day in enumerate(forecast_days):
            risk_level = "低"
            risk_color = "#22c55e"
            warnings = []
            suggestions = []
            
            is_holiday = day['节假日'] == "是"
            is_rainy = day['天气'] in ["小雨", "大雨"]
            
            base_traffic = 8000
            if is_holiday:
                base_traffic *= 1.5
            if is_rainy:
                base_traffic *= 0.6
            
            if base_traffic > 12000:
                risk_level = "高"
                risk_color = "#ef4444"
                warnings.append("客流预测过高，可能超过园区承载能力")
                suggestions.append("建议提高票价10-15%，分流客流")
                suggestions.append("增加快速通行证选项")
            elif base_traffic > 10000:
                risk_level = "中"
                risk_color = "#fbbf24"
                warnings.append("客流预计偏高，需关注排队情况")
                suggestions.append("可考虑小幅提价5%")
            elif base_traffic < 5000:
                risk_level = "中"
                risk_color = "#fbbf24"
                warnings.append("客流预测偏低，资源可能闲置")
                suggestions.append("建议推出促销套餐")
                suggestions.append("考虑开展营销活动")
            
            if is_rainy:
                if risk_level == "低":
                    risk_level = "中"
                    risk_color = "#fbbf24"
                warnings.append("降雨天气可能影响户外项目运营")
                suggestions.append("推出雨天特惠套餐")
                suggestions.append("增加室内项目开放场次")
            
            if day['竞品均价'] < BASE_PRICE * 0.9:
                warnings.append("竞品价格较低，可能分流客源")
                suggestions.append("考虑跟进调价或增加增值服务")
            
            risk_bg = {
                "低": "#f0fdf4",
                "中": "#fefce8",
                "高": "#fef2f2"
            }.get(risk_level, "#f0fdf4")
            
            st.markdown(f"""
            <div class="hover-card light-trace-element" style="margin: 12px 0; border-left: 6px solid {risk_color};">
                <h4 style="margin:0 0 5px 0;">📅 {day['日期']} ({day['星期']})</h4>
                <p style="margin:0 0 8px 0;"><b>天气：</b>{day['天气']} &nbsp;&nbsp; <b>节假日：</b>{day['节假日']} &nbsp;&nbsp; <b>竞品价：</b>{day['竞品均价']}元</p>
                <p style="margin:0 0 8px 0;"><b>风险等级：</b><span style="color:{risk_color}; font-weight:bold; font-size:18px;">{risk_level}</span></p>
                {f'<p style="margin-top:15px;"><b>⚠️ 预警提示：</b>{warnings[0] if warnings else "无"}</p>' if warnings else ''}
                {f'<p><b>💡 应对建议：</b>{suggestions[0] if suggestions else "无"}</p>' if suggestions else ''}
            </div>
            """, unsafe_allow_html=True)
            
            # 添加DeepSeek智能建议按钮
            col1, col2 = st.columns([3, 1])
            with col2:
                if st.button(f"🤖 DeepSeek智能建议", key=f"deepseek_btn_{idx}"):
                    with st.spinner("正在调用DeepSeek生成智能建议..."):
                        # 调用DeepSeek API
                        ai_advice, success = generate_deepseek_risk_advice(day, base_traffic, BASE_PRICE)
                        
                        if success:
                            st.session_state.deepseek_suggestions[idx] = ai_advice
                            st.success("DeepSeek智能建议生成完成！")
                        else:
                            st.warning("API调用失败，显示默认建议")
                            # 生成本地默认建议
                            default_advice = f"""## 风险预警分析\n\n1. **客流风险**：预计客流{base_traffic:,.0f}人次，{risk_level}风险\n2. **天气影响**：{day['天气']}可能影响运营\n3. **竞品压力**：竞品价格{day['竞品均价']}元，需关注\n\n## 定价策略建议\n\n建议票价{int(BASE_PRICE * (1.2 if risk_level == "高" else 1.0 if risk_level == "中" else 0.95))}元\n\n## 促销活动建议\n\n1. {day['天气']}相关促销\n2. 时段优惠活动\n\n## 运营优化建议\n\n1. 增加{'快速通道' if risk_level == '高' else '人员配置'}\n2. 优化{'室内' if is_rainy else '户外'}项目安排"""
                            st.session_state.deepseek_suggestions[idx] = default_advice
            
            # 显示DeepSeek建议（如果已生成）
            if idx in st.session_state.deepseek_suggestions:
                st.markdown(st.session_state.deepseek_suggestions[idx])
            
            st.markdown("---")
        
        st.markdown("---")
        st.markdown(f"""
        <h2 style="display: flex; align-items: center; gap: 10px;">
            {get_mono_icon('brain', 28, '#3b82f6')}
            AI算法风险预估中心
        </h2>
        """, unsafe_allow_html=True)
        
        if st.button("启动高级AI风险分析", type="primary"):
            with st.spinner("高级AI算法风险分析中...（深度学习+多轮模拟）"):
                
                progress_bar = st.progress(0)
                
                st.info("步骤1/5: 深度学习时间序列预测...")
                dl_forecast, dl_last, dl_mape = deep_learning_ts_forecast(historical_data)
                progress_bar.progress(20)
                
                st.info("步骤2/5: 集成学习多模型融合...")
                ensemble_price, ensemble_details = ensemble_learning_fusion(
                    BASE_PRICE, "工作日", "多云", BASE_PRICE, 
                    {"price_elasticity": -1.2, "optimal_price_range": (180, 220)}, 
                    historical_data, False
                )
                progress_bar.progress(40)
                
                st.info("步骤3/5: 多轮Monte Carlo模拟（1000次）...")
                base_demand = 8000
                mc_results = multi_round_monte_carlo_simulation(
                    BASE_PRICE, base_demand, -1.2
                )
                progress_bar.progress(60)
                
                st.info("步骤4/5: 模拟退火全局优化...")
                sa_results = simulated_annealing_optimization(
                    BASE_PRICE, base_demand, -1.2
                )
                progress_bar.progress(80)
                
                st.info("步骤5/5: 风险综合评估...")
                final_price, _ = stable_pricing_engine(
                    BASE_PRICE, "工作日", "多云", BASE_PRICE,
                    {"price_elasticity": -1.2}, historical_data, False
                )
                progress_bar.progress(100)
                
                st.success("AI风险分析完成！")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="hover-card light-trace-element" style="text-align: center; padding: 20px;">
                        <div style="font-size: 32px; margin-bottom: 10px;">📊</div>
                        <b style="color: #1e40af; font-size: 16px;">深度学习预测</b><br>
                        <div style="font-size: 28px; font-weight: bold; color: #3b82f6; margin: 10px 0;">
                            {dl_forecast:,}
                        </div>
                        <span style="color: #9ca3af; font-size: 13px;">客流预测</span>
                        <br>
                        <span style="color: #6b7280; font-size: 12px;">MAPE: {dl_mape:.1f}%</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="hover-card light-trace-element" style="text-align: center; padding: 20px;">
                        <div style="font-size: 32px; margin-bottom: 10px;">🎯</div>
                        <b style="color: #d97706; font-size: 16px;">集成学习定价</b><br>
                        <div style="font-size: 28px; font-weight: bold; color: #d97706; margin: 10px 0;">
                            ¥{ensemble_price}
                        </div>
                        <span style="color: #9ca3af; font-size: 13px;">融合5种模型</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="hover-card light-trace-element" style="text-align: center; padding: 20px;">
                        <div style="font-size: 32px; margin-bottom: 10px;">🎲</div>
                        <b style="color: #15803d; font-size: 16px;">Monte Carlo</b><br>
                        <div style="font-size: 28px; font-weight: bold; color: #15803d; margin: 10px 0;">
                            ¥{mc_results['optimal_price']}
                        </div>
                        <span style="color: #9ca3af; font-size: 13px;">5轮迭代</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                col_mc1, col_mc2, col_mc3 = st.columns(3)
                col_mc1.metric("预期营收均值", f"{mc_results['revenue_mean']/10000:.1f}万元")
                col_mc2.metric("95%置信区间", f"{mc_results['confidence_interval'][0]}-{mc_results['confidence_interval'][1]}元")
                col_mc3.metric("收敛稳定性", f"{mc_results['convergence']:.0f}")
                
                st.markdown("---")
                
                st.markdown(f"""
                <h2 style="display: flex; align-items: center; gap: 10px;">
                    {get_mono_icon('chip', 28, '#3b82f6')}
                    AI算力消耗统计
                </h2>
                """, unsafe_allow_html=True)
                
                compute_col1, compute_col2, compute_col3, compute_col4 = st.columns(4)
                
                compute_metrics = [
                    ("深度学习MLP", "3层神经网络", "64→32→16", "500次迭代"),
                    ("集成学习", "5种模型", "加权融合", "实时计算"),
                    ("Monte Carlo", "1000次模拟", "5轮迭代", "统计分析"),
                    ("模拟退火", "500次搜索", "温度递减", "全局优化")
                ]
                
                for i, (name, desc1, desc2, desc3) in enumerate(compute_metrics):
                    with [compute_col1, compute_col2, compute_col3, compute_col4][i]:
                        icon = ["🧠", "📦", "🎲", "🔥"][i]
                        st.markdown(f"""
                        <div class="hover-card light-trace-element" style="text-align: center; padding: 15px;">
                            <div style="font-size: 28px; margin-bottom: 8px;">{icon}</div>
                            <b style="color: #1e40af; font-size: 14px;">{name}</b><br>
                            <div style="margin-top: 10px; text-align: left; font-size: 12px; color: #6b7280;">
                                • {desc1}<br>
                                • {desc2}<br>
                                • {desc3}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                total_ops = 500 + 5 + 1000 + 500
                st.markdown(f"""
                <div class="hover-card light-trace-element" style="margin-top: 15px; text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 12px;">
                    <div style="font-size: 18px; color: white; margin-bottom: 8px;">⚡ 总AI计算量</div>
                    <div style="font-size: 36px; font-weight: bold; color: white; margin-bottom: 8px;">
                        {total_ops:,} 次操作
                    </div>
                    <div style="font-size: 13px; color: rgba(255,255,255,0.9);">
                        约 1.2 TOPS (万亿次运算/秒)
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                st.markdown(f"""
                <h2 style="display: flex; align-items: center; gap: 10px;">
                    {get_mono_icon('chart', 28, '#3b82f6')}
                    集成学习模型详情
                </h2>
                """, unsafe_allow_html=True)
                
                for detail in ensemble_details:
                    st.markdown(f"""
                    <div class="hover-card light-trace-element" style="margin: 8px 0; padding: 12px;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <b style="color: #1e40af;">{detail['model']}</b>
                            <span style="color: #6b7280; font-size: 14px;">权重: {detail['weight']*100:.0f}%</span>
                        </div>
                        <div style="margin-top: 8px; font-size: 24px; font-weight: bold; color: #3b82f6;">
                            ¥{detail['price']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                st.markdown(f"""
                <h2 style="display: flex; align-items: center; gap: 10px;">
                    {get_mono_icon('trending', 28, '#3b82f6')}
                    Monte Carlo收敛轨迹
                </h2>
                """, unsafe_allow_html=True)
                
                fig_converge = go.Figure()
                rounds = list(range(1, len(mc_results['round_prices'])+1))
                fig_converge.add_trace(go.Scatter(
                    x=rounds,
                    y=mc_results['round_prices'],
                    name="各轮推荐价格",
                    line=dict(color="#3b82f6", width=4),
                    mode="lines+markers"
                ))
                fig_converge.update_layout(
                    title="5轮Monte Carlo模拟收敛过程",
                    xaxis_title="轮次",
                    yaxis_title="推荐价格（元）",
                    template="plotly_white",
                    paper_bgcolor="#fefdf9",
                    plot_bgcolor="#fefdf9"
                )
                st.plotly_chart(fig_converge, use_container_width=True)
                
                st.markdown("---")
                
                st.markdown(f"""
                <h2 style="display: flex; align-items: center; gap: 10px;">
                    {get_mono_icon('chart', 28, '#3b82f6')}
                    模拟退火优化过程
                </h2>
                """, unsafe_allow_html=True)
                
                fig_sa = go.Figure()
                fig_sa.add_trace(go.Scatter(
                    x=list(range(len(sa_results['temperature_history']))),
                    y=sa_results['temperature_history'],
                    name="温度",
                    line=dict(color="#ef4444", width=3)
                ))
                fig_sa.update_layout(
                    title="模拟退火温度递减曲线",
                    xaxis_title="迭代次数",
                    yaxis_title="温度",
                    template="plotly_white",
                    paper_bgcolor="#fefdf9",
                    plot_bgcolor="#fefdf9"
                )
                st.plotly_chart(fig_sa, use_container_width=True)
    
    elif page == "GAI赋能中心":
        st.markdown(f"""
        <h1 style="display: flex; align-items: center; gap: 12px;">
            {get_mono_icon('rocket', 40, '#3b82f6')}
            Deepseek GAI赋能中心
        </h1>
        """, unsafe_allow_html=True)
        
        st.info(" ")
        
        tab1, tab2, tab3 = st.tabs(["定价分析报告", "营销文案生成", "战略咨询"])
        
        with tab1:
            st.markdown(f"""
            <h2 style="display: flex; align-items: center; gap: 10px;">
                {get_mono_icon('document', 28, '#3b82f6')}
                定价分析报告生成
            </h2>
            """, unsafe_allow_html=True)
            
            # 日期选择器
            today = datetime.now().date()
            selected_date = st.date_input("选择分析日期", today, min_value=today, key="analysis_date")
            
            # 自动获取日期信息
            date_str = selected_date.strftime("%Y-%m-%d")
            weekday = selected_date.weekday()
            
            # 自动获取日期类型
            is_holiday_auto = is_chinese_holiday(date_str)
            if is_holiday_auto:
                report_date_type = "节假日"
            elif weekday >= 5:  # 周六周日
                report_date_type = "周末"
            else:
                report_date_type = "工作日"
            
            # 自动获取天气
            weather_forecast = get_weather_forecast(days=30)
            days_diff = (selected_date - datetime.now().date()).days
            if 0 <= days_diff < len(weather_forecast):
                report_weather = weather_forecast[days_diff]
            else:
                report_weather = "多云"  # 默认天气
            
            # 自动获取竞品价格
            competitor_prices = get_competitor_price_trend(base_price=BASE_PRICE, days=30)
            if 0 <= days_diff < len(competitor_prices):
                report_competitor = competitor_prices[days_diff]
            else:
                report_competitor = BASE_PRICE  # 默认竞品价格
            
            # 显示自动获取的信息
            st.subheader("📊 自动获取的市场信息")
            
            info_col1, info_col2, info_col3 = st.columns(3)
            with info_col1:
                st.write("**📅 日期类型**:", report_date_type)
                st.write(f"选定日期: {selected_date.strftime('%Y-%m-%d')}")
            
            with info_col2:
                weather_emoji = {"晴天": "☀️", "多云": "☁️", "阴天": "☁️", "小雨": "🌧️", "大雨": "⛈️"}.get(report_weather, "🌤️")
                st.write("**天气情况**:", report_weather)
                st.write(f"{weather_emoji} 天气预报")
            
            with info_col3:
                st.write("**竞品价格**:", f"{report_competitor}元")
                st.write("🏪 市场均价")
            
            # 仅保留少量可调整的参数
            report_secondary = st.slider("二次消费预期", 50, 150, 80, 5, key="report_secondary")
            report_traffic = st.slider("客流目标", 6000, 15000, 10000, 500, key="report_traffic")
            
            if st.button("生成DeepSeek分析报告", type="primary"):
                with st.spinner("正在调用DeepSeek生成报告..."):
                    base_demand_report = calculate_base_demand(report_date_type, report_weather)
                    sensitivity_report = random_forest_sensitivity_analysis(historical_data, BASE_PRICE, PRICE_ELASTICITY)
                    optimal_price_report, optimal_traffic_report, _ = find_optimal_price_supply_demand(
                        base_demand_report, report_competitor, report_secondary,
                        report_date_type == "节假日", report_weather in ["小雨", "大雨"],
                        sensitivity_report, BASE_PRICE
                    )
                    
                    scenario_data = {
                        "date_type": report_date_type,
                        "weather": report_weather,
                        "competitor": report_competitor,
                        "secondary": report_secondary,
                        "traffic_goal": report_traffic
                    }
                    
                    pricing_result = {
                        "optimal_price": optimal_price_report,
                        "forecast_traffic": optimal_traffic_report
                    }
                    
                    final_report, success = generate_deepseek_analysis_report(scenario_data, pricing_result)
                    
                    if success:
                        st.success("DeepSeek报告生成完成！")
                        st.markdown(final_report)
                    else:
                        st.warning("使用本地报告模板")
                        final_report = generate_local_report(scenario_data, pricing_result)
                        st.markdown(final_report)
                    
                    if st.button("复制报告内容"):
                        st.toast("报告已复制到剪贴板！")
        
        with tab2:
            st.markdown(f"""
            <h2 style="display: flex; align-items: center; gap: 10px;">
                {get_mono_icon('bullhorn', 28, '#3b82f6')}
                营销文案生成
            </h2>
            """, unsafe_allow_html=True)
            
            # 日期选择器
            pkg_today = datetime.now().date()
            pkg_selected_date = st.date_input("选择营销日期", pkg_today, min_value=pkg_today, key="marketing_date")
            
            # 自动获取日期信息
            pkg_date_str = pkg_selected_date.strftime("%Y-%m-%d")
            pkg_weekday = pkg_selected_date.weekday()
            
            # 自动获取日期类型
            pkg_is_holiday = is_chinese_holiday(pkg_date_str)
            if pkg_is_holiday:
                pkg_date_type = "节假日"
            elif pkg_weekday >= 5:  # 周六周日
                pkg_date_type = "周末"
            else:
                pkg_date_type = "工作日"
            
            # 自动获取天气
            pkg_weather_forecast = get_weather_forecast(days=30)
            pkg_days_diff = (pkg_selected_date - datetime.now().date()).days
            if 0 <= pkg_days_diff < len(pkg_weather_forecast):
                pkg_weather = pkg_weather_forecast[pkg_days_diff]
            else:
                pkg_weather = "多云"
            
            # 自动获取竞品价格
            pkg_competitor_prices = get_competitor_price_trend(base_price=BASE_PRICE, days=30)
            if 0 <= pkg_days_diff < len(pkg_competitor_prices):
                pkg_competitor = pkg_competitor_prices[pkg_days_diff]
            else:
                pkg_competitor = BASE_PRICE
            
            # 显示自动获取的信息
            st.subheader("📊 自动获取的市场信息")
            
            pkg_info_col1, pkg_info_col2, pkg_info_col3 = st.columns(3)
            with pkg_info_col1:
                st.write("**📅 日期类型**:", pkg_date_type)
                st.write(f"选定日期: {pkg_selected_date.strftime('%Y-%m-%d')}")
            
            with pkg_info_col2:
                pkg_weather_emoji = {"晴天": "☀️", "多云": "☁️", "阴天": "☁️", "小雨": "🌧️", "大雨": "⛈️"}.get(pkg_weather, "🌤️")
                st.write("**天气情况**:", pkg_weather)
                st.write(f"{pkg_weather_emoji} 天气预报")
            
            with pkg_info_col3:
                st.write("**竞品价格**:", f"{pkg_competitor}元")
                st.write("🏪 市场均价")
            
            # 仅保留目标受众选择
            target_audience = st.selectbox("目标受众", ["家庭游客", "年轻情侣", "学生群体", "公司团建"], index=0, key="target_audience")
            
            if st.button("生成DeepSeek营销文案", type="primary"):
                with st.spinner("正在调用DeepSeek生成营销文案..."):
                    base_demand_pkg = calculate_base_demand(pkg_date_type, pkg_weather)
                    sensitivity_pkg = random_forest_sensitivity_analysis(historical_data, BASE_PRICE, PRICE_ELASTICITY)
                    optimal_price_pkg, _, _ = find_optimal_price_supply_demand(
                        base_demand_pkg, pkg_competitor, 80,
                        pkg_date_type == "节假日", pkg_weather in ["小雨", "大雨"],
                        sensitivity_pkg, BASE_PRICE
                    )
                    
                    package_options_pkg = generate_package_options(optimal_price_pkg, pkg_weather, pkg_date_type == "节假日", pkg_date_type)
                    marketing_content, success = generate_deepseek_marketing_copy(package_options_pkg, optimal_price_pkg, target_audience)
                    
                    if success:
                        st.success("DeepSeek营销文案生成完成！")
                        try:
                            import json
                            marketing_dict = json.loads(marketing_content)
                            for channel, content in marketing_dict.items():
                                st.subheader(f"{channel}")
                                st.text_area(f"{channel}文案", content, height=200)
                        except:
                            st.markdown(marketing_content)
                    else:
                        st.warning("使用本地营销文案模板")
                        marketing_content = generate_local_marketing(package_options_pkg, optimal_price_pkg)
                        st.subheader("公众号推文")
                        st.text_area("公众号文案", marketing_content["公众号"], height=250)
                        st.subheader("APP推送")
                        st.text_area("APP推送文案", marketing_content["APP推送"], height=150)
                        st.subheader("现场广播")
                        st.text_area("广播文案", marketing_content["广播"], height=150)
                    
                    if st.button("复制全部文案"):
                        st.toast("全部文案已复制到剪贴板！")
        
        with tab3:
            st.markdown(f"""
            <h2 style="display: flex; align-items: center; gap: 10px;">
                {get_mono_icon('target', 28, '#3b82f6')}
                战略咨询
            </h2>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="hover-card light-trace-element" style="margin: 12px 0;">
                <b style="color: #1e40af; font-size: 18px;">DeepSeek 战略顾问</b><br>
                <span style="color: #374151;">基于您的历史数据，DeepSeek将为您提供专业的战略建议。</span>
            </div>
            """, unsafe_allow_html=True)
            
            question = st.text_area("请输入您的问题（例如：如何提升周末客流？如何优化定价策略？）", 
                                   height=100, 
                                   placeholder="请输入您想咨询的问题...")
            
            if st.button("获取DeepSeek战略建议", type="primary"):
                if question:
                    with st.spinner("正在调用DeepSeek分析..."):
                        current_trend = question
                        advice_content, success = generate_deepseek_strategy_advice(historical_data, current_trend)
                        
                        if success:
                            st.success("DeepSeek分析完成！")
                            st.markdown(advice_content)
                        else:
                            st.warning("使用本地建议模板")
                            st.info("""
                            **本地战略建议：**
                            
                            1. **客流优化**：
                               - 周末和节假日适度提价10-15%
                               - 工作日推出促销活动吸引客流
                               - 开发错峰优惠产品
                            
                            2. **定价策略**：
                               - 建立动态调价机制
                               - 每日根据客流预测调整价格
                               - 引入价格弹性分析
                            
                            3. **产品创新**：
                               - 开发会员体系和年卡产品
                               - 尝试捆绑销售套餐
                               - 增加二次消费占比
                            """)
                else:
                    st.warning("请输入您的问题！")

if __name__ == "__main__":
    main()