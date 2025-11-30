import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta, date, timezone
import random
import os
import json
from openai import OpenAI
import numpy as np

# Múi giờ Việt Nam (UTC+7)
VIETNAM_TZ = timezone(timedelta(hours=7))

def get_vietnam_datetime():
    """Lấy datetime hiện tại theo múi giờ Việt Nam"""
    return datetime.now(VIETNAM_TZ)

def get_vietnam_date():
    """Lấy ngày hiện tại theo múi giờ Việt Nam"""
    return get_vietnam_datetime().date()

def format_vietnam_datetime(dt, format_str='%d/%m/%Y %H:%M'):
    """Format datetime theo múi giờ Việt Nam"""
    if isinstance(dt, str):
        dt = datetime.strptime(dt, '%Y-%m-%d')
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=VIETNAM_TZ)
    return dt.strftime(format_str)
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
import warnings
warnings.filterwarnings('ignore')

# Cấu hình trang
st.set_page_config(
    page_title="Phân tích trận bóng đá",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS tùy chỉnh
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        position: relative;
    }
    .main-header h1 {
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    .logo-container {
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 999;
        background: white;
        padding: 10px 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .logo-container img {
        height: 50px;
        width: auto;
    }
    .logo-text {
        font-size: 1.2rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    @media (max-width: 768px) {
        .logo-container {
            top: 10px;
            right: 10px;
            padding: 8px 12px;
        }
        .logo-text {
            font-size: 1rem;
        }
    }
    .match-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        cursor: pointer;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .match-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    .stat-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
    }
    .prediction-box {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Khởi tạo OpenAI client
def init_openai_client():
    """Khởi tạo OpenAI client với ưu tiên: Session State > Environment > Secrets"""
    # Ưu tiên 1: Session State (API key người dùng nhập)
    api_key = st.session_state.get('openai_api_key', '')
    
    # Ưu tiên 2: Biến môi trường
    if not api_key:
        api_key = os.getenv('OPENAI_API_KEY', '')
    
    # Ưu tiên 3: Streamlit Secrets
    if not api_key:
        try:
            api_key = st.secrets.get('OPENAI_API_KEY', '')
        except (FileNotFoundError, AttributeError, KeyError):
            api_key = ''
    
    if api_key:
        try:
            return OpenAI(api_key=api_key)
        except Exception as e:
            st.error(f"Lỗi khởi tạo OpenAI client: {str(e)}")
            return None
    return None

def get_openai_api_key():
    """Lấy API key với ưu tiên: Session State > Environment > Secrets"""
    # Ưu tiên 1: Session State
    api_key = st.session_state.get('openai_api_key', '')
    
    # Ưu tiên 2: Biến môi trường
    if not api_key:
        api_key = os.getenv('OPENAI_API_KEY', '')
    
    # Ưu tiên 3: Streamlit Secrets
    if not api_key:
        try:
            api_key = st.secrets.get('OPENAI_API_KEY', '')
        except (FileNotFoundError, AttributeError, KeyError):
            api_key = ''
    
    return api_key

# ==================== AI AGENT CHUYÊN NGHIỆP ====================

def calculate_xg_xga(match):
    """Tính toán Expected Goals (xG) và Expected Goals Against (xGA)"""
    # xG dựa trên bàn thắng trung bình, form, và chất lượng đối thủ
    home_form_factor = sum([3 if r == 'W' else 1 if r == 'D' else 0 for r in match['home_form']]) / 15
    away_form_factor = sum([3 if r == 'W' else 1 if r == 'D' else 0 for r in match['away_form']]) / 15
    
    # xG cho đội nhà
    home_xg = match['home_avg_goals'] * (1 + home_form_factor) * (1 - match['away_avg_conceded'] / 3)
    home_xga = match['home_avg_conceded'] * (1 - home_form_factor * 0.2) * (1 + match['away_avg_goals'] / 3)
    
    # xG cho đội khách
    away_xg = match['away_avg_goals'] * (1 + away_form_factor) * (1 - match['home_avg_conceded'] / 3)
    away_xga = match['away_avg_conceded'] * (1 - away_form_factor * 0.2) * (1 + match['home_avg_goals'] / 3)
    
    # Điều chỉnh cho lợi thế sân nhà
    home_xg *= 1.15
    home_xga *= 0.92
    
    return {
        'home_xg': round(home_xg, 2),
        'home_xga': round(home_xga, 2),
        'away_xg': round(away_xg, 2),
        'away_xga': round(away_xga, 2)
    }

def calculate_team_strength(match):
    """Tính toán sức mạnh tổng thể của đội bóng"""
    xg_data = calculate_xg_xga(match)
    
    # Sức mạnh tấn công (Attack Strength)
    home_attack = (xg_data['home_xg'] / 2.0) * 100
    away_attack = (xg_data['away_xg'] / 2.0) * 100
    
    # Sức mạnh phòng thủ (Defense Strength)
    home_defense = (1 - xg_data['home_xga'] / 2.0) * 100
    away_defense = (1 - xg_data['away_xga'] / 2.0) * 100
    
    # Sức mạnh tổng thể
    home_strength = (home_attack * 0.5 + home_defense * 0.5)
    away_strength = (away_attack * 0.5 + away_defense * 0.5)
    
    return {
        'home_attack': round(home_attack, 1),
        'home_defense': round(home_defense, 1),
        'home_strength': round(home_strength, 1),
        'away_attack': round(away_attack, 1),
        'away_defense': round(away_defense, 1),
        'away_strength': round(away_strength, 1)
    }

def ml_predict_probabilities(match):
    """Sử dụng Machine Learning để dự đoán xác suất (mô phỏng với Random Forest)"""
    # Tính toán features
    xg_data = calculate_xg_xga(match)
    strength = calculate_team_strength(match)
    
    # Features cho ML model
    features = np.array([[
        match['home_position'],
        match['away_position'],
        match['home_points'],
        match['away_points'],
        xg_data['home_xg'],
        xg_data['home_xga'],
        xg_data['away_xg'],
        xg_data['away_xga'],
        strength['home_strength'],
        strength['away_strength'],
        sum([3 if r == 'W' else 1 if r == 'D' else 0 for r in match['home_form']]),
        sum([3 if r == 'W' else 1 if r == 'D' else 0 for r in match['away_form']]),
        match['head_to_head']['home_wins'],
        match['head_to_head']['away_wins'],
    ]])
    
    # Mô phỏng ML model với công thức dựa trên features
    home_advantage = 0.08
    diff_strength = (strength['home_strength'] - strength['away_strength']) / 100
    
    # Tính xác suất
    home_win_prob = 0.33 + diff_strength * 0.3 + home_advantage
    away_win_prob = 0.33 - diff_strength * 0.3
    draw_prob = 1 - home_win_prob - away_win_prob
    
    # Đảm bảo xác suất hợp lệ
    home_win_prob = max(0.1, min(0.7, home_win_prob))
    away_win_prob = max(0.1, min(0.7, away_win_prob))
    draw_prob = max(0.15, min(0.4, draw_prob))
    
    # Chuẩn hóa
    total = home_win_prob + draw_prob + away_win_prob
    home_win_prob = home_win_prob / total
    draw_prob = draw_prob / total
    away_win_prob = away_win_prob / total
    
    return {
        'home_win': round(home_win_prob * 100, 1),
        'draw': round(draw_prob * 100, 1),
        'away_win': round(away_win_prob * 100, 1)
    }

def calculate_value_bet(ai_prob, bookmaker_odds):
    """Tính toán Value Bet - so sánh xác suất AI vs odds nhà cái"""
    # Chuyển đổi odds thành xác suất implied
    implied_prob = (1 / bookmaker_odds) * 100
    
    # Tính Value = (AI Probability - Implied Probability) / Implied Probability * 100
    value = ((ai_prob - implied_prob) / implied_prob) * 100
    
    return {
        'ai_probability': ai_prob,
        'implied_probability': round(implied_prob, 2),
        'value': round(value, 2),
        'is_value_bet': value > 5  # Value bet khi value > 5%
    }

def find_best_value_bets(match, prediction):
    """Tìm các Value Bet tốt nhất cho trận đấu"""
    ml_probs = ml_predict_probabilities(match)
    ah = match['asian_handicap']
    
    value_bets = []
    
    # Phân tích kèo 1X2
    home_odds_1x2 = 1.85  # Giả định odds
    away_odds_1x2 = 2.10
    draw_odds_1x2 = 3.20
    
    home_value = calculate_value_bet(ml_probs['home_win'], home_odds_1x2)
    draw_value = calculate_value_bet(ml_probs['draw'], draw_odds_1x2)
    away_value = calculate_value_bet(ml_probs['away_win'], away_odds_1x2)
    
    if home_value['is_value_bet']:
        value_bets.append({
            'type': '1X2 - Thắng nhà',
            'team': match['home_team'],
            'odds': home_odds_1x2,
            'ai_prob': home_value['ai_probability'],
            'implied_prob': home_value['implied_probability'],
            'value': home_value['value'],
            'recommendation': f"✅ VALUE BET: {match['home_team']} thắng"
        })
    
    if draw_value['is_value_bet']:
        value_bets.append({
            'type': '1X2 - Hòa',
            'team': 'Hòa',
            'odds': draw_odds_1x2,
            'ai_prob': draw_value['ai_probability'],
            'implied_prob': draw_value['implied_probability'],
            'value': draw_value['value'],
            'recommendation': '✅ VALUE BET: Hòa'
        })
    
    if away_value['is_value_bet']:
        value_bets.append({
            'type': '1X2 - Thắng khách',
            'team': match['away_team'],
            'odds': away_odds_1x2,
            'ai_prob': away_value['ai_probability'],
            'implied_prob': away_value['implied_probability'],
            'value': away_value['value'],
            'recommendation': f"✅ VALUE BET: {match['away_team']} thắng"
        })
    
    # Phân tích kèo Handicap nhẹ
    handicap_lines = [0, -0.25, 0.25]
    for line in handicap_lines:
        if abs(line) <= 0.25:  # Chỉ xét handicap nhẹ
            # Tính xác suất thắng kèo chấp
            predicted_diff = float(prediction['predicted_score'].split('-')[0]) - float(prediction['predicted_score'].split('-')[1])
            result_after_handicap = predicted_diff - line
            
            if result_after_handicap > 0.5:
                prob_win = 75
                odds_home = ah['home_odds'] if line >= 0 else 1.90
                value_ah = calculate_value_bet(prob_win, odds_home)
                if value_ah['is_value_bet']:
                    value_bets.append({
                        'type': f'Handicap {line:+.2f}',
                        'team': match['home_team'],
                        'odds': odds_home,
                        'ai_prob': value_ah['ai_probability'],
                        'implied_prob': value_ah['implied_probability'],
                        'value': value_ah['value'],
                        'recommendation': f"✅ VALUE BET: {match['home_team']} chấp {line:+.2f}"
                    })
    
    # Sắp xếp theo value giảm dần
    value_bets.sort(key=lambda x: x['value'], reverse=True)
    
    return value_bets

def predict_first_half_over_under(match, prediction):
    """Dự đoán Tài/Xỉu hiệp 1"""
    # Tính tổng bàn thắng dự đoán cả trận
    predicted_scores = prediction['predicted_score'].split('-')
    total_goals = int(predicted_scores[0]) + int(predicted_scores[1])
    
    # Thống kê: Hiệp 1 thường chiếm 40-45% tổng bàn thắng
    first_half_ratio = 0.42  # Tỷ lệ trung bình
    first_half_goals = total_goals * first_half_ratio
    
    # Điều chỉnh dựa trên phong độ tấn công
    home_attack_factor = match['home_avg_goals'] / 2.0
    away_attack_factor = match['away_avg_goals'] / 2.0
    first_half_goals = (home_attack_factor + away_attack_factor) * first_half_ratio * 1.1
    
    # Mức kèo phổ biến cho hiệp 1: 0.5, 1.0, 1.5
    over_under_lines = [0.5, 1.0, 1.5]
    predictions = []
    
    for line in over_under_lines:
        if first_half_goals > line + 0.2:
            recommendation = f"Tài {line}"
            confidence = min(75, 50 + (first_half_goals - line) * 20)
        elif first_half_goals < line - 0.2:
            recommendation = f"Xỉu {line}"
            confidence = min(75, 50 + (line - first_half_goals) * 20)
        else:
            recommendation = f"Gần mức {line} - Cân nhắc"
            confidence = 50
        
        predictions.append({
            'line': line,
            'predicted_goals': round(first_half_goals, 2),
            'recommendation': recommendation,
            'confidence': round(confidence, 1)
        })
    
    return {
        'predicted_first_half_goals': round(first_half_goals, 2),
        'predictions': predictions
    }

def predict_full_match_over_under(match, prediction):
    """Dự đoán Tài/Xỉu cả trận (cải thiện)"""
    predicted_scores = prediction['predicted_score'].split('-')
    total_goals = int(predicted_scores[0]) + int(predicted_scores[1])
    
    # Tính toán dựa trên xG
    xg_data = calculate_xg_xga(match)
    xg_total = xg_data['home_xg'] + xg_data['away_xg']
    
    # Kết hợp dự đoán tỷ số và xG
    final_prediction = (total_goals + xg_total) / 2
    
    # Mức kèo phổ biến
    over_under_lines = [2.0, 2.5, 3.0, 3.5]
    predictions = []
    
    for line in over_under_lines:
        diff = final_prediction - line
        if diff > 0.3:
            recommendation = f"Tài {line}"
            confidence = min(80, 55 + diff * 15)
            value = "Cao"
        elif diff < -0.3:
            recommendation = f"Xỉu {line}"
            confidence = min(80, 55 + abs(diff) * 15)
            value = "Cao"
        else:
            recommendation = f"Gần mức {line}"
            confidence = 50
            value = "Thấp"
        
        predictions.append({
            'line': line,
            'predicted_total': round(final_prediction, 2),
            'recommendation': recommendation,
            'confidence': round(confidence, 1),
            'value': value
        })
    
    return {
        'predicted_total_goals': round(final_prediction, 2),
        'predictions': predictions
    }

def predict_exact_score(match, prediction):
    """Dự đoán tỷ số chính xác cả trận"""
    predicted_scores = prediction['predicted_score'].split('-')
    home_score = int(predicted_scores[0])
    away_score = int(predicted_scores[1])
    
    # Tính xác suất các tỷ số có thể
    xg_data = calculate_xg_xga(match)
    ml_probs = ml_predict_probabilities(match)
    
    # Tạo danh sách tỷ số có khả năng
    possible_scores = []
    
    # Tỷ số chính
    main_score_prob = 35
    possible_scores.append({
        'score': f"{home_score}-{away_score}",
        'probability': main_score_prob,
        'description': 'Tỷ số dự đoán chính'
    })
    
    # Các tỷ số gần đó
    for h in range(max(0, home_score-1), home_score+2):
        for a in range(max(0, away_score-1), away_score+2):
            if f"{h}-{a}" != f"{home_score}-{away_score}":
                prob = 15 - abs(h - home_score) * 5 - abs(a - away_score) * 5
                if prob > 5:
                    possible_scores.append({
                        'score': f"{h}-{a}",
                        'probability': prob,
                        'description': 'Tỷ số có khả năng'
                    })
    
    # Sắp xếp theo xác suất
    possible_scores.sort(key=lambda x: x['probability'], reverse=True)
    
    return {
        'main_prediction': f"{home_score}-{away_score}",
        'possible_scores': possible_scores[:5]
    }

def predict_corners_over_under(match):
    """Dự đoán Tài/Xỉu phạt góc"""
    # Tính toán dựa trên thống kê phạt góc
    # Ước tính từ bàn thắng trung bình (thường 1 bàn thắng = 2-3 phạt góc)
    home_corners_avg = match.get('home_avg_corners', match['home_avg_goals'] * 2.5)
    away_corners_avg = match.get('away_avg_corners', match['away_avg_goals'] * 2.5)
    
    # Điều chỉnh dựa trên phong độ tấn công
    home_attack_factor = match['home_avg_goals'] / 2.0
    away_attack_factor = match['away_avg_goals'] / 2.0
    
    predicted_corners = (home_corners_avg + away_corners_avg) * (1 + (home_attack_factor + away_attack_factor - 1) * 0.2)
    
    # Mức kèo phổ biến: 8.5, 9.5, 10.5, 11.5
    over_under_lines = [8.5, 9.5, 10.5, 11.5]
    predictions = []
    
    for line in over_under_lines:
        diff = predicted_corners - line
        if diff > 0.5:
            recommendation = f"Tài {line}"
            confidence = min(75, 50 + diff * 10)
        elif diff < -0.5:
            recommendation = f"Xỉu {line}"
            confidence = min(75, 50 + abs(diff) * 10)
        else:
            recommendation = f"Gần mức {line}"
            confidence = 50
        
        predictions.append({
            'line': line,
            'predicted_corners': round(predicted_corners, 1),
            'recommendation': recommendation,
            'confidence': round(confidence, 1)
        })
    
    return {
        'predicted_total_corners': round(predicted_corners, 1),
        'predictions': predictions
    }

def predict_handicap_betting_strategy(match, prediction):
    """Hướng dẫn cách cá dựa vào kèo chấp để thắng"""
    ah = match['asian_handicap']
    handicap_line = ah['line']
    
    predicted_scores = prediction['predicted_score'].split('-')
    home_score = int(predicted_scores[0])
    away_score = int(predicted_scores[1])
    predicted_diff = home_score - away_score
    
    # Tính kết quả sau khi áp dụng chấp
    result_after_handicap = predicted_diff - handicap_line
    
    strategies = []
    
    # Phân tích kèo chấp
    if result_after_handicap > 0.5:
        # Đội nhà thắng kèo
        strategies.append({
            'bet': f"Chọn {match['home_team']} (chấp {handicap_line:+.1f})",
            'reason': f"Dự đoán chênh lệch {predicted_diff:+.1f} bàn, sau chấp còn {result_after_handicap:+.1f} bàn",
            'confidence': min(80, 60 + result_after_handicap * 10),
            'odds': ah['home_odds'],
            'recommendation': '✅ Nên cá'
        })
    elif result_after_handicap < -0.5:
        # Đội khách thắng kèo
        strategies.append({
            'bet': f"Chọn {match['away_team']} (nhận chấp {handicap_line:+.1f})",
            'reason': f"Dự đoán chênh lệch {predicted_diff:+.1f} bàn, sau chấp còn {result_after_handicap:+.1f} bàn",
            'confidence': min(80, 60 + abs(result_after_handicap) * 10),
            'odds': ah['away_odds'],
            'recommendation': '✅ Nên cá'
        })
    else:
        # Hòa kèo hoặc gần hòa
        strategies.append({
            'bet': f"Hòa kèo hoặc gần hòa",
            'reason': f"Chênh lệch sau chấp chỉ {result_after_handicap:+.1f} bàn, rủi ro cao",
            'confidence': 40,
            'odds': 'N/A',
            'recommendation': '⚠️ Không nên cá hoặc cá nhẹ'
        })
    
    # Thêm chiến lược an toàn
    if abs(handicap_line) <= 0.25:
        strategies.append({
            'bet': f"Kèo chấp nhẹ ({handicap_line:+.1f}) - An toàn hơn",
            'reason': 'Kèo chấp nhẹ ít rủi ro, phù hợp cho người mới',
            'confidence': 65,
            'odds': ah['home_odds'] if handicap_line >= 0 else ah['away_odds'],
            'recommendation': '💡 Chiến lược an toàn'
        })
    
    # Thêm mẹo
    tips = []
    if abs(handicap_line) > 1.0:
        tips.append("⚠️ Kèo chấp lớn (>1.0) có rủi ro cao, chỉ nên cá khi rất chắc chắn")
    if abs(result_after_handicap) < 0.5:
        tips.append("⚠️ Kết quả gần hòa kèo, nên tránh hoặc cá nhẹ")
    if ah['home_odds'] > 2.0 or ah['away_odds'] > 2.0:
        tips.append("💡 Odds cao (>2.0) cho thấy nhà cái đánh giá rủi ro cao")
    
    return {
        'handicap_line': handicap_line,
        'predicted_diff': predicted_diff,
        'result_after_handicap': result_after_handicap,
        'strategies': strategies,
        'tips': tips
    }

# Hàm dự đoán chi tiết với OpenAI
def predict_with_openai(match, xg_data, strength_data, ml_probs):
    """Sử dụng OpenAI để dự đoán chi tiết và chính xác hơn"""
    client = init_openai_client()
    
    if not client:
        return None
    
    try:
        prompt = f"""
Bạn là AI Agent chuyên gia dự đoán bóng đá với độ chính xác 80-90%. Dựa vào dữ liệu sau, hãy đưa ra dự đoán CHÍNH XÁC.

**TRẬN ĐẤU:** {match['home_team']} vs {match['away_team']}
**Giải đấu:** {match['league']} | **Sân:** {match['venue']}

**DỮ LIỆU PHÂN TÍCH:**
1. xG/xGA: {match['home_team']} (xG={xg_data['home_xg']}, xGA={xg_data['home_xga']}) vs {match['away_team']} (xG={xg_data['away_xg']}, xGA={xg_data['away_xga']})
2. Sức mạnh: {match['home_team']} ({strength_data['home_strength']}/100) vs {match['away_team']} ({strength_data['away_strength']}/100)
3. Vị trí: {match['home_team']} (#{match['home_position']}, {match['home_points']} điểm) vs {match['away_team']} (#{match['away_position']}, {match['away_points']} điểm)
4. Form 5 trận: {match['home_team']} {', '.join(match['home_form'])} vs {match['away_team']} {', '.join(match['away_form'])}
5. Lịch sử đối đầu: {match['head_to_head']['home_wins']}-{match['head_to_head']['draws']}-{match['head_to_head']['away_wins']}
6. ML Predictions: Thắng nhà {ml_probs['home_win']}%, Hòa {ml_probs['draw']}%, Thắng khách {ml_probs['away_win']}%

**YÊU CẦU:**
Hãy trả về JSON với format sau (CHỈ TRẢ VỀ JSON, KHÔNG CÓ TEXT KHÁC):
{{
    "exact_score": "X-Y",
    "home_win_prob": số_phần_trăm,
    "draw_prob": số_phần_trăm,
    "away_win_prob": số_phần_trăm,
    "total_goals": số_bàn_thắng,
    "first_half_goals": số_bàn_hiệp_1,
    "total_corners": số_phạt_góc,
    "handicap_recommendation": "Chọn đội nào",
    "over_under_recommendation": "Tài/Xỉu mức_kèo",
    "confidence": số_phần_trăm_tự_tin,
    "reasoning": "Lý do ngắn gọn"
}}

Lưu ý: Tất cả số phải là số nguyên hoặc số thập phân, không có ký tự khác.
"""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Bạn là AI Agent dự đoán bóng đá chuyên nghiệp. Trả về KẾT QUẢ DƯỚI DẠNG JSON THUẦN, không có text giải thích thêm. Format JSON phải chính xác."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,  # Rất thấp để chính xác
            max_tokens=300,
            response_format={"type": "json_object"}
        )
        
        import json
        result = json.loads(response.choices[0].message.content)
        return result
    except Exception as e:
        st.error(f"Lỗi khi gọi OpenAI: {str(e)}")
        return None

# Hàm phân tích với OpenAI - AI Agent chuyên nghiệp
def analyze_with_openai(match, prediction, xg_data, strength_data, ml_probs, value_bets):
    """Sử dụng OpenAI để phân tích chuyên sâu như AI Agent hàng đầu"""
    client = init_openai_client()
    
    if not client:
        return None
    
    try:
        # Tạo prompt chuyên nghiệp cho AI Agent
        prompt = f"""
Bạn là AI Agent chuyên gia dự đoán bóng đá hàng đầu thế giới, sử dụng Machine Learning và phân tích dữ liệu nâng cao.

**TRẬN ĐẤU:** {match['home_team']} vs {match['away_team']}
**Giải đấu:** {match['league']} | **Ngày:** {match['date']} | **Sân:** {match['venue']}

**1. SỨC MẠNH ĐỘI BÓNG (xG/xGA Analysis):**
- {match['home_team']}: xG={xg_data['home_xg']}, xGA={xg_data['home_xga']}, Sức mạnh={strength_data['home_strength']}
- {match['away_team']}: xG={xg_data['away_xg']}, xGA={xg_data['away_xga']}, Sức mạnh={strength_data['away_strength']}
- Hiệu quả tấn công: {match['home_team']} ({strength_data['home_attack']}) vs {match['away_team']} ({strength_data['away_attack']})
- Hiệu quả phòng thủ: {match['home_team']} ({strength_data['home_defense']}) vs {match['away_team']} ({strength_data['away_defense']})

**2. THỐNG KÊ & FORM:**
- Vị trí: {match['home_team']} (#{match['home_position']}, {match['home_points']} điểm) vs {match['away_team']} (#{match['away_position']}, {match['away_points']} điểm)
- Form: {match['home_team']} {', '.join(match['home_form'])} vs {match['away_team']} {', '.join(match['away_form'])}
- Lịch sử đối đầu: {match['head_to_head']['home_wins']}-{match['head_to_head']['draws']}-{match['head_to_head']['away_wins']}

**3. MACHINE LEARNING PREDICTIONS:**
- Xác suất thắng nhà: {ml_probs['home_win']}%
- Xác suất hòa: {ml_probs['draw']}%
- Xác suất thắng khách: {ml_probs['away_win']}%
- Dự đoán tỷ số: {prediction['predicted_score']}

**4. VALUE BETS:**
{chr(10).join([f"- {vb['recommendation']} (Value: {vb['value']}%, Odds: {vb['odds']})" for vb in value_bets[:3]]) if value_bets else "- Không có Value Bet rõ ràng"}

**YÊU CẦU PHÂN TÍCH:**
1. Phân tích sâu về xG/xGA và sự tương quan giữa hai đội
2. Đánh giá lối chơi dựa trên dữ liệu (pressing, transition, kiểm soát bóng)
3. Xác suất thắng thực tế dựa trên ML model
4. So sánh với odds nhà cái và xác định Value Bet
5. Khuyến nghị kèo cụ thể (1X2 hoặc Handicap nhẹ 0, -0.25, +0.25) với odds 1.80-2.20
6. Giải thích ngắn gọn, súc tích, dựa 100% vào dữ liệu - KHÔNG cảm tính

**QUY TẮC:**
- Chỉ gợi ý khi có Value Bet rõ ràng (value > 5%)
- Ưu tiên kèo an toàn - hiệu quả - giá trị cao
- Trình bày rõ ràng, dễ hiểu, hành động được ngay

Hãy phân tích như một AI Agent chuyên nghiệp và đưa ra khuyến nghị cụ thể (tối đa 300 từ, tiếng Việt).
"""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Bạn là AI Agent chuyên gia dự đoán bóng đá hàng đầu thế giới, sử dụng Machine Learning và phân tích dữ liệu nâng cao để xác định Value Bet với độ chính xác 80-90%. Mọi phân tích phải dựa 100% vào dữ liệu, không cảm tính."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # Giảm temperature để chính xác hơn
            max_tokens=600
        )
        
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Lỗi khi gọi OpenAI: {str(e)}")
        return None

# Dữ liệu các trận sắp xảy ra
@st.cache_data
def load_upcoming_matches():
    # Ngày bắt đầu từ hôm nay (theo múi giờ Việt Nam)
    today = get_vietnam_datetime()
    start_date = today
    matches = [
        {
            'id': 1,
            'home_team': 'Manchester United',
            'away_team': 'Crystal Palace',
            'date': (start_date.date() + timedelta(days=0)).strftime('%Y-%m-%d'),
            'time': '20:00',
            'venue': 'Old Trafford',
            'league': 'Premier League',
            'home_form': ['W', 'W', 'D', 'L', 'W'],
            'away_form': ['D', 'L', 'W', 'D', 'L'],
            'home_position': 6,
            'away_position': 14,
            'home_points': 45,
            'away_points': 28,
            'head_to_head': {'home_wins': 4, 'draws': 1, 'away_wins': 0},
            'home_avg_goals': 1.8,
            'away_avg_goals': 1.2,
            'home_avg_conceded': 1.2,
            'away_avg_conceded': 1.5,
            'asian_handicap': {
                'line': 0.5,
                'home_odds': 1.85,
                'away_odds': 1.95
            },
            'over_under': {
                'line': 2.5,
                'over_odds': 1.90,
                'under_odds': 1.90
            },
        },
        {
            'id': 2,
            'home_team': 'Real Madrid',
            'away_team': 'Barcelona',
            'date': (start_date.date() + timedelta(days=1)).strftime('%Y-%m-%d'),
            'time': '21:00',
            'venue': 'Santiago Bernabéu',
            'league': 'La Liga',
            'home_form': ['W', 'W', 'W', 'D', 'W'],
            'away_form': ['W', 'L', 'W', 'W', 'D'],
            'home_position': 1,
            'away_position': 3,
            'home_points': 62,
            'away_points': 55,
            'head_to_head': {'home_wins': 3, 'draws': 0, 'away_wins': 2},
            'home_avg_goals': 2.3,
            'away_avg_goals': 2.0,
            'home_avg_conceded': 0.8,
            'away_avg_conceded': 1.1,
            'asian_handicap': {
                'line': -0.25,
                'home_odds': 1.92,
                'away_odds': 1.88
            },
            'over_under': {
                'line': 3.0,
                'over_odds': 1.95,
                'under_odds': 1.85
            },
        },
        {
            'id': 3,
            'home_team': 'Bayern Munich',
            'away_team': 'Borussia Dortmund',
            'date': (start_date.date() + timedelta(days=2)).strftime('%Y-%m-%d'),
            'time': '19:30',
            'venue': 'Allianz Arena',
            'league': 'Bundesliga',
            'home_form': ['W', 'W', 'W', 'W', 'D'],
            'away_form': ['W', 'D', 'L', 'W', 'W'],
            'home_position': 2,
            'away_position': 4,
            'home_points': 52,
            'away_points': 48,
            'head_to_head': {'home_wins': 4, 'draws': 1, 'away_wins': 0},
            'home_avg_goals': 2.5,
            'away_avg_goals': 1.9,
            'home_avg_conceded': 0.9,
            'away_avg_conceded': 1.3,
            'asian_handicap': {
                'line': -0.75,
                'home_odds': 1.88,
                'away_odds': 1.92
            },
            'over_under': {
                'line': 3.5,
                'over_odds': 1.90,
                'under_odds': 1.90
            },
        },
        {
            'id': 4,
            'home_team': 'PSG',
            'away_team': 'Marseille',
            'date': (start_date.date() + timedelta(days=3)).strftime('%Y-%m-%d'),
            'time': '22:00',
            'venue': 'Parc des Princes',
            'league': 'Ligue 1',
            'home_form': ['W', 'D', 'W', 'W', 'W'],
            'away_form': ['L', 'W', 'D', 'W', 'L'],
            'home_position': 1,
            'away_position': 7,
            'home_points': 59,
            'away_points': 38,
            'head_to_head': {'home_wins': 3, 'draws': 2, 'away_wins': 0},
            'home_avg_goals': 2.2,
            'away_avg_goals': 1.5,
            'home_avg_conceded': 1.0,
            'away_avg_conceded': 1.4,
            'asian_handicap': {
                'line': -1.0,
                'home_odds': 1.90,
                'away_odds': 1.90
            },
            'over_under': {
                'line': 3.0,
                'over_odds': 1.88,
                'under_odds': 1.92
            },
        },
        {
            'id': 5,
            'home_team': 'AC Milan',
            'away_team': 'Inter Milan',
            'date': (start_date.date() + timedelta(days=4)).strftime('%Y-%m-%d'),
            'time': '20:45',
            'venue': 'San Siro',
            'league': 'Serie A',
            'home_form': ['D', 'W', 'L', 'W', 'D'],
            'away_form': ['W', 'W', 'W', 'D', 'W'],
            'home_position': 5,
            'away_position': 1,
            'home_points': 46,
            'away_points': 61,
            'head_to_head': {'home_wins': 1, 'draws': 2, 'away_wins': 2},
            'home_avg_goals': 1.7,
            'away_avg_goals': 2.0,
            'home_avg_conceded': 1.3,
            'away_avg_conceded': 0.9,
            'asian_handicap': {
                'line': 0.25,
                'home_odds': 1.93,
                'away_odds': 1.87
            },
            'over_under': {
                'line': 2.75,
                'over_odds': 1.92,
                'under_odds': 1.88
            },
        },
        # Premier League - Thêm trận
        {
            'id': 6,
            'home_team': 'Arsenal',
            'away_team': 'Chelsea',
            'date': (start_date.date() + timedelta(days=0)).strftime('%Y-%m-%d'),
            'time': '17:30',
            'venue': 'Emirates Stadium',
            'league': 'Premier League',
            'home_form': ['W', 'W', 'W', 'D', 'W'],
            'away_form': ['D', 'W', 'L', 'W', 'D'],
            'home_position': 3,
            'away_position': 8,
            'home_points': 56,
            'away_points': 42,
            'head_to_head': {'home_wins': 2, 'draws': 2, 'away_wins': 1},
            'home_avg_goals': 2.0,
            'away_avg_goals': 1.6,
            'home_avg_conceded': 0.9,
            'away_avg_conceded': 1.2,
            'asian_handicap': {
                'line': -0.5,
                'home_odds': 1.90,
                'away_odds': 1.90
            },
            'over_under': {
                'line': 2.75,
                'over_odds': 1.88,
                'under_odds': 1.92
            },
        },
        {
            'id': 7,
            'home_team': 'Manchester City',
            'away_team': 'Tottenham',
            'date': (start_date.date() + timedelta(days=1)).strftime('%Y-%m-%d'),
            'time': '16:00',
            'venue': 'Etihad Stadium',
            'league': 'Premier League',
            'home_form': ['W', 'W', 'D', 'W', 'W'],
            'away_form': ['W', 'L', 'W', 'D', 'W'],
            'home_position': 1,
            'away_position': 5,
            'home_points': 65,
            'away_points': 48,
            'head_to_head': {'home_wins': 3, 'draws': 1, 'away_wins': 1},
            'home_avg_goals': 2.4,
            'away_avg_goals': 1.8,
            'home_avg_conceded': 0.7,
            'away_avg_conceded': 1.1,
            'asian_handicap': {
                'line': -1.0,
                'home_odds': 1.88,
                'away_odds': 1.92
            },
            'over_under': {
                'line': 3.0,
                'over_odds': 1.90,
                'under_odds': 1.90
            },
        },
        {
            'id': 8,
            'home_team': 'Newcastle',
            'away_team': 'Brighton',
            'date': (start_date.date() + timedelta(days=1)).strftime('%Y-%m-%d'),
            'time': '15:00',
            'venue': 'St. James\' Park',
            'league': 'Premier League',
            'home_form': ['W', 'D', 'L', 'W', 'D'],
            'away_form': ['D', 'W', 'W', 'L', 'D'],
            'home_position': 7,
            'away_position': 9,
            'home_points': 40,
            'away_points': 38,
            'head_to_head': {'home_wins': 1, 'draws': 2, 'away_wins': 2},
            'home_avg_goals': 1.6,
            'away_avg_goals': 1.7,
            'home_avg_conceded': 1.3,
            'away_avg_conceded': 1.4,
            'asian_handicap': {
                'line': 0.0,
                'home_odds': 1.95,
                'away_odds': 1.85
            },
            'over_under': {
                'line': 2.5,
                'over_odds': 1.92,
                'under_odds': 1.88
            },
        },
        {
            'id': 9,
            'home_team': 'Aston Villa',
            'away_team': 'West Ham',
            'date': (start_date.date() + timedelta(days=2)).strftime('%Y-%m-%d'),
            'time': '15:00',
            'venue': 'Villa Park',
            'league': 'Premier League',
            'home_form': ['W', 'W', 'W', 'L', 'W'],
            'away_form': ['L', 'D', 'W', 'D', 'L'],
            'home_position': 4,
            'away_position': 10,
            'home_points': 52,
            'away_points': 36,
            'head_to_head': {'home_wins': 2, 'draws': 1, 'away_wins': 2},
            'home_avg_goals': 1.9,
            'away_avg_goals': 1.4,
            'home_avg_conceded': 1.0,
            'away_avg_conceded': 1.5,
            'asian_handicap': {
                'line': -0.5,
                'home_odds': 1.92,
                'away_odds': 1.88
            },
            'over_under': {
                'line': 2.5,
                'over_odds': 1.90,
                'under_odds': 1.90
            },
        },
        {
            'id': 10,
            'home_team': 'Fulham',
            'away_team': 'Crystal Palace',
            'date': (start_date.date() + timedelta(days=2)).strftime('%Y-%m-%d'),
            'time': '15:00',
            'venue': 'Craven Cottage',
            'league': 'Premier League',
            'home_form': ['D', 'L', 'W', 'D', 'L'],
            'away_form': ['W', 'D', 'L', 'D', 'W'],
            'home_position': 12,
            'away_position': 11,
            'home_points': 32,
            'away_points': 34,
            'head_to_head': {'home_wins': 1, 'draws': 3, 'away_wins': 1},
            'home_avg_goals': 1.3,
            'away_avg_goals': 1.2,
            'home_avg_conceded': 1.5,
            'away_avg_conceded': 1.4,
            'asian_handicap': {
                'line': 0.0,
                'home_odds': 1.90,
                'away_odds': 1.90
            },
            'over_under': {
                'line': 2.25,
                'over_odds': 1.93,
                'under_odds': 1.87
            },
        },
        # La Liga - Thêm trận
        {
            'id': 11,
            'home_team': 'Atletico Madrid',
            'away_team': 'Sevilla',
            'date': (start_date.date() + timedelta(days=1)).strftime('%Y-%m-%d'),
            'time': '18:30',
            'venue': 'Wanda Metropolitano',
            'league': 'La Liga',
            'home_form': ['W', 'D', 'W', 'W', 'D'],
            'away_form': ['D', 'L', 'W', 'D', 'L'],
            'home_position': 4,
            'away_position': 12,
            'home_points': 50,
            'away_points': 28,
            'head_to_head': {'home_wins': 3, 'draws': 1, 'away_wins': 1},
            'home_avg_goals': 1.9,
            'away_avg_goals': 1.1,
            'home_avg_conceded': 0.9,
            'away_avg_conceded': 1.6,
            'asian_handicap': {
                'line': -0.75,
                'home_odds': 1.88,
                'away_odds': 1.92
            },
            'over_under': {
                'line': 2.5,
                'over_odds': 1.90,
                'under_odds': 1.90
            },
        },
        {
            'id': 12,
            'home_team': 'Valencia',
            'away_team': 'Villarreal',
            'date': (start_date.date() + timedelta(days=2)).strftime('%Y-%m-%d'),
            'time': '20:00',
            'venue': 'Mestalla',
            'league': 'La Liga',
            'home_form': ['W', 'L', 'D', 'W', 'L'],
            'away_form': ['D', 'W', 'D', 'L', 'W'],
            'home_position': 8,
            'away_position': 6,
            'home_points': 35,
            'away_points': 42,
            'head_to_head': {'home_wins': 1, 'draws': 2, 'away_wins': 2},
            'home_avg_goals': 1.4,
            'away_avg_goals': 1.6,
            'home_avg_conceded': 1.3,
            'away_avg_conceded': 1.2,
            'asian_handicap': {
                'line': 0.25,
                'home_odds': 1.93,
                'away_odds': 1.87
            },
            'over_under': {
                'line': 2.5,
                'over_odds': 1.88,
                'under_odds': 1.92
            },
        },
        {
            'id': 13,
            'home_team': 'Real Sociedad',
            'away_team': 'Athletic Bilbao',
            'date': (start_date.date() + timedelta(days=3)).strftime('%Y-%m-%d'),
            'time': '19:00',
            'venue': 'Reale Arena',
            'league': 'La Liga',
            'home_form': ['W', 'D', 'W', 'L', 'W'],
            'away_form': ['W', 'W', 'D', 'W', 'D'],
            'home_position': 5,
            'away_position': 7,
            'home_points': 45,
            'away_points': 40,
            'head_to_head': {'home_wins': 2, 'draws': 2, 'away_wins': 1},
            'home_avg_goals': 1.7,
            'away_avg_goals': 1.5,
            'home_avg_conceded': 1.1,
            'away_avg_conceded': 1.0,
            'asian_handicap': {
                'line': -0.25,
                'home_odds': 1.90,
                'away_odds': 1.90
            },
            'over_under': {
                'line': 2.25,
                'over_odds': 1.92,
                'under_odds': 1.88
            },
        },
        {
            'id': 14,
            'home_team': 'Girona',
            'away_team': 'Real Betis',
            'date': (start_date.date() + timedelta(days=3)).strftime('%Y-%m-%d'),
            'time': '21:00',
            'venue': 'Estadi Montilivi',
            'league': 'La Liga',
            'home_form': ['W', 'W', 'L', 'W', 'D'],
            'away_form': ['D', 'L', 'W', 'D', 'W'],
            'home_position': 2,
            'away_position': 9,
            'home_points': 58,
            'away_points': 33,
            'head_to_head': {'home_wins': 1, 'draws': 1, 'away_wins': 3},
            'home_avg_goals': 2.1,
            'away_avg_goals': 1.3,
            'home_avg_conceded': 1.2,
            'away_avg_conceded': 1.4,
            'asian_handicap': {
                'line': -0.5,
                'home_odds': 1.91,
                'away_odds': 1.89
            },
            'over_under': {
                'line': 2.75,
                'over_odds': 1.90,
                'under_odds': 1.90
            },
        },
        {
            'id': 15,
            'home_team': 'Osasuna',
            'away_team': 'Getafe',
            'date': (start_date.date() + timedelta(days=4)).strftime('%Y-%m-%d'),
            'time': '18:00',
            'venue': 'El Sadar',
            'league': 'La Liga',
            'home_form': ['L', 'D', 'W', 'L', 'D'],
            'away_form': ['D', 'W', 'D', 'L', 'D'],
            'home_position': 13,
            'away_position': 14,
            'home_points': 26,
            'away_points': 25,
            'head_to_head': {'home_wins': 2, 'draws': 1, 'away_wins': 2},
            'home_avg_goals': 1.1,
            'away_avg_goals': 1.0,
            'home_avg_conceded': 1.4,
            'away_avg_conceded': 1.3,
            'asian_handicap': {
                'line': 0.0,
                'home_odds': 1.92,
                'away_odds': 1.88
            },
            'over_under': {
                'line': 2.0,
                'over_odds': 1.95,
                'under_odds': 1.85
            },
        },
        # Serie A - Thêm trận
        {
            'id': 16,
            'home_team': 'Juventus',
            'away_team': 'Napoli',
            'date': (start_date.date() + timedelta(days=1)).strftime('%Y-%m-%d'),
            'time': '20:45',
            'venue': 'Allianz Stadium',
            'league': 'Serie A',
            'home_form': ['W', 'W', 'D', 'W', 'W'],
            'away_form': ['W', 'D', 'L', 'W', 'D'],
            'home_position': 2,
            'away_position': 6,
            'home_points': 58,
            'away_points': 44,
            'head_to_head': {'home_wins': 2, 'draws': 1, 'away_wins': 2},
            'home_avg_goals': 1.8,
            'away_avg_goals': 1.6,
            'home_avg_conceded': 0.8,
            'away_avg_conceded': 1.2,
            'asian_handicap': {
                'line': -0.5,
                'home_odds': 1.89,
                'away_odds': 1.91
            },
            'over_under': {
                'line': 2.5,
                'over_odds': 1.92,
                'under_odds': 1.88
            },
        },
        {
            'id': 17,
            'home_team': 'AS Roma',
            'away_team': 'Lazio',
            'date': (start_date.date() + timedelta(days=2)).strftime('%Y-%m-%d'),
            'time': '20:45',
            'venue': 'Stadio Olimpico',
            'league': 'Serie A',
            'home_form': ['W', 'D', 'W', 'L', 'W'],
            'away_form': ['L', 'W', 'D', 'W', 'L'],
            'home_position': 7,
            'away_position': 8,
            'home_points': 42,
            'away_points': 40,
            'head_to_head': {'home_wins': 2, 'draws': 1, 'away_wins': 2},
            'home_avg_goals': 1.5,
            'away_avg_goals': 1.4,
            'home_avg_conceded': 1.2,
            'away_avg_conceded': 1.3,
            'asian_handicap': {
                'line': 0.0,
                'home_odds': 1.90,
                'away_odds': 1.90
            },
            'over_under': {
                'line': 2.25,
                'over_odds': 1.93,
                'under_odds': 1.87
            },
        },
        {
            'id': 18,
            'home_team': 'Atalanta',
            'away_team': 'Fiorentina',
            'date': (start_date.date() + timedelta(days=3)).strftime('%Y-%m-%d'),
            'time': '18:00',
            'venue': 'Gewiss Stadium',
            'league': 'Serie A',
            'home_form': ['W', 'W', 'L', 'W', 'D'],
            'away_form': ['D', 'W', 'W', 'D', 'W'],
            'home_position': 4,
            'away_position': 9,
            'home_points': 50,
            'away_points': 38,
            'head_to_head': {'home_wins': 3, 'draws': 0, 'away_wins': 2},
            'home_avg_goals': 1.9,
            'away_avg_goals': 1.5,
            'home_avg_conceded': 1.1,
            'away_avg_conceded': 1.3,
            'asian_handicap': {
                'line': -0.5,
                'home_odds': 1.91,
                'away_odds': 1.89
            },
            'over_under': {
                'line': 2.75,
                'over_odds': 1.90,
                'under_odds': 1.90
            },
        },
        {
            'id': 19,
            'home_team': 'Bologna',
            'away_team': 'Torino',
            'date': (start_date.date() + timedelta(days=4)).strftime('%Y-%m-%d'),
            'time': '15:00',
            'venue': 'Stadio Renato Dall\'Ara',
            'league': 'Serie A',
            'home_form': ['W', 'D', 'W', 'W', 'L'],
            'away_form': ['D', 'L', 'D', 'W', 'D'],
            'home_position': 3,
            'away_position': 11,
            'home_points': 54,
            'away_points': 35,
            'head_to_head': {'home_wins': 1, 'draws': 2, 'away_wins': 2},
            'home_avg_goals': 1.6,
            'away_avg_goals': 1.2,
            'home_avg_conceded': 0.9,
            'away_avg_conceded': 1.4,
            'asian_handicap': {
                'line': -0.5,
                'home_odds': 1.90,
                'away_odds': 1.90
            },
            'over_under': {
                'line': 2.25,
                'over_odds': 1.92,
                'under_odds': 1.88
            },
        },
        {
            'id': 20,
            'home_team': 'Udinese',
            'away_team': 'Sassuolo',
            'date': (start_date.date() + timedelta(days=5)).strftime('%Y-%m-%d'),
            'time': '15:00',
            'venue': 'Dacia Arena',
            'league': 'Serie A',
            'home_form': ['D', 'L', 'D', 'L', 'D'],
            'away_form': ['L', 'D', 'L', 'W', 'L'],
            'home_position': 15,
            'away_position': 17,
            'home_points': 24,
            'away_points': 20,
            'head_to_head': {'home_wins': 2, 'draws': 1, 'away_wins': 2},
            'home_avg_goals': 1.0,
            'away_avg_goals': 1.1,
            'home_avg_conceded': 1.5,
            'away_avg_conceded': 1.6,
            'asian_handicap': {
                'line': 0.0,
                'home_odds': 1.92,
                'away_odds': 1.88
            },
            'over_under': {
                'line': 2.0,
                'over_odds': 1.94,
                'under_odds': 1.86
            },
        },
        # Bundesliga - Thêm trận
        {
            'id': 21,
            'home_team': 'RB Leipzig',
            'away_team': 'Bayer Leverkusen',
            'date': (start_date.date() + timedelta(days=1)).strftime('%Y-%m-%d'),
            'time': '17:30',
            'venue': 'Red Bull Arena',
            'league': 'Bundesliga',
            'home_form': ['W', 'W', 'D', 'W', 'L'],
            'away_form': ['W', 'W', 'W', 'D', 'W'],
            'home_position': 3,
            'away_position': 1,
            'home_points': 48,
            'away_points': 64,
            'head_to_head': {'home_wins': 1, 'draws': 2, 'away_wins': 2},
            'home_avg_goals': 2.0,
            'away_avg_goals': 2.2,
            'home_avg_conceded': 1.1,
            'away_avg_conceded': 0.7,
            'asian_handicap': {
                'line': 0.25,
                'home_odds': 1.94,
                'away_odds': 1.86
            },
            'over_under': {
                'line': 3.0,
                'over_odds': 1.88,
                'under_odds': 1.92
            },
        },
        {
            'id': 22,
            'home_team': 'Eintracht Frankfurt',
            'away_team': 'Wolfsburg',
            'date': (start_date.date() + timedelta(days=2)).strftime('%Y-%m-%d'),
            'time': '15:30',
            'venue': 'Deutsche Bank Park',
            'league': 'Bundesliga',
            'home_form': ['D', 'W', 'L', 'W', 'D'],
            'away_form': ['W', 'D', 'L', 'D', 'W'],
            'home_position': 6,
            'away_position': 7,
            'home_points': 42,
            'away_points': 40,
            'head_to_head': {'home_wins': 2, 'draws': 1, 'away_wins': 2},
            'home_avg_goals': 1.7,
            'away_avg_goals': 1.5,
            'home_avg_conceded': 1.2,
            'away_avg_conceded': 1.3,
            'asian_handicap': {
                'line': -0.25,
                'home_odds': 1.91,
                'away_odds': 1.89
            },
            'over_under': {
                'line': 2.75,
                'over_odds': 1.90,
                'under_odds': 1.90
            },
        },
        {
            'id': 23,
            'home_team': 'Stuttgart',
            'away_team': 'Union Berlin',
            'date': (start_date.date() + timedelta(days=3)).strftime('%Y-%m-%d'),
            'time': '15:30',
            'venue': 'MHPArena',
            'league': 'Bundesliga',
            'home_form': ['W', 'W', 'W', 'L', 'W'],
            'away_form': ['L', 'D', 'L', 'L', 'D'],
            'home_position': 5,
            'away_position': 15,
            'home_points': 46,
            'away_points': 22,
            'head_to_head': {'home_wins': 2, 'draws': 1, 'away_wins': 2},
            'home_avg_goals': 2.1,
            'away_avg_goals': 1.0,
            'home_avg_conceded': 1.0,
            'away_avg_conceded': 1.7,
            'asian_handicap': {
                'line': -1.0,
                'home_odds': 1.87,
                'away_odds': 1.93
            },
            'over_under': {
                'line': 2.5,
                'over_odds': 1.91,
                'under_odds': 1.89
            },
        },
        {
            'id': 24,
            'home_team': 'Borussia Mönchengladbach',
            'away_team': 'Werder Bremen',
            'date': (start_date.date() + timedelta(days=4)).strftime('%Y-%m-%d'),
            'time': '15:30',
            'venue': 'Borussia-Park',
            'league': 'Bundesliga',
            'home_form': ['D', 'L', 'W', 'D', 'L'],
            'away_form': ['W', 'L', 'D', 'W', 'L'],
            'home_position': 10,
            'away_position': 11,
            'home_points': 30,
            'away_points': 28,
            'head_to_head': {'home_wins': 3, 'draws': 0, 'away_wins': 2},
            'home_avg_goals': 1.4,
            'away_avg_goals': 1.3,
            'home_avg_conceded': 1.5,
            'away_avg_conceded': 1.6,
            'asian_handicap': {
                'line': 0.0,
                'home_odds': 1.90,
                'away_odds': 1.90
            },
            'over_under': {
                'line': 2.5,
                'over_odds': 1.92,
                'under_odds': 1.88
            },
        },
        {
            'id': 25,
            'home_team': 'Hoffenheim',
            'away_team': 'Augsburg',
            'date': (start_date.date() + timedelta(days=5)).strftime('%Y-%m-%d'),
            'time': '15:30',
            'venue': 'PreZero Arena',
            'league': 'Bundesliga',
            'home_form': ['W', 'D', 'L', 'W', 'D'],
            'away_form': ['D', 'W', 'D', 'L', 'W'],
            'home_position': 8,
            'away_position': 9,
            'home_points': 36,
            'away_points': 34,
            'head_to_head': {'home_wins': 2, 'draws': 2, 'away_wins': 1},
            'home_avg_goals': 1.6,
            'away_avg_goals': 1.4,
            'home_avg_conceded': 1.4,
            'away_avg_conceded': 1.5,
            'asian_handicap': {
                'line': -0.25,
                'home_odds': 1.92,
                'away_odds': 1.88
            },
            'over_under': {
                'line': 2.75,
                'over_odds': 1.89,
                'under_odds': 1.91
            },
        },
    ]
    
    # ========== THÊM CÁC GIẢI ĐẤU MỚI ==========
    
    # Serie A (Ý)
    matches.extend([
        {
            'id': 26,
            'home_team': 'AC Milan',
            'away_team': 'Inter Milan',
            'date': (start_date.date() + timedelta(days=0)).strftime('%Y-%m-%d'),
            'time': '21:45',
            'venue': 'San Siro',
            'league': 'Serie A',
            'home_form': ['W', 'W', 'D', 'W', 'W'],
            'away_form': ['W', 'D', 'W', 'W', 'D'],
            'home_position': 2,
            'away_position': 1,
            'home_points': 58,
            'away_points': 62,
            'head_to_head': {'home_wins': 1, 'draws': 2, 'away_wins': 2},
            'home_avg_goals': 2.0,
            'away_avg_goals': 2.1,
            'home_avg_conceded': 0.9,
            'away_avg_conceded': 0.8,
            'asian_handicap': {'line': 0.0, 'home_odds': 2.10, 'away_odds': 1.80},
            'over_under': {'line': 2.5, 'over_odds': 1.85, 'under_odds': 1.95},
        },
        {
            'id': 27,
            'home_team': 'Juventus',
            'away_team': 'AS Roma',
            'date': (start_date.date() + timedelta(days=1)).strftime('%Y-%m-%d'),
            'time': '20:00',
            'venue': 'Allianz Stadium',
            'league': 'Serie A',
            'home_form': ['W', 'D', 'W', 'W', 'D'],
            'away_form': ['D', 'W', 'L', 'W', 'D'],
            'home_position': 3,
            'away_position': 6,
            'home_points': 55,
            'away_points': 45,
            'head_to_head': {'home_wins': 3, 'draws': 1, 'away_wins': 1},
            'home_avg_goals': 1.9,
            'away_avg_goals': 1.6,
            'home_avg_conceded': 0.9,
            'away_avg_conceded': 1.2,
            'asian_handicap': {'line': -0.5, 'home_odds': 1.90, 'away_odds': 1.90},
            'over_under': {'line': 2.5, 'over_odds': 1.88, 'under_odds': 1.92},
        },
    ])
    
    # Ligue 1 (Pháp)
    matches.extend([
        {
            'id': 28,
            'home_team': 'Paris Saint-Germain',
            'away_team': 'Olympique Marseille',
            'date': (start_date.date() + timedelta(days=0)).strftime('%Y-%m-%d'),
            'time': '23:00',
            'venue': 'Parc des Princes',
            'league': 'Ligue 1',
            'home_form': ['W', 'W', 'W', 'W', 'D'],
            'away_form': ['W', 'D', 'W', 'L', 'W'],
            'home_position': 1,
            'away_position': 4,
            'home_points': 65,
            'away_points': 48,
            'head_to_head': {'home_wins': 4, 'draws': 0, 'away_wins': 1},
            'home_avg_goals': 2.5,
            'away_avg_goals': 1.8,
            'home_avg_conceded': 0.7,
            'away_avg_conceded': 1.1,
            'asian_handicap': {'line': -1.0, 'home_odds': 1.85, 'away_odds': 1.95},
            'over_under': {'line': 3.0, 'over_odds': 1.90, 'under_odds': 1.90},
        },
    ])
    
    # UEFA Champions League
    matches.extend([
        {
            'id': 29,
            'home_team': 'Real Madrid',
            'away_team': 'Bayern Munich',
            'date': (start_date.date() + timedelta(days=1)).strftime('%Y-%m-%d'),
            'time': '03:00',
            'venue': 'Santiago Bernabéu',
            'league': 'UEFA Champions League',
            'home_form': ['W', 'W', 'W', 'D', 'W'],
            'away_form': ['W', 'W', 'D', 'W', 'W'],
            'home_position': 1,
            'away_position': 1,
            'home_points': 0,
            'away_points': 0,
            'head_to_head': {'home_wins': 2, 'draws': 1, 'away_wins': 2},
            'home_avg_goals': 2.3,
            'away_avg_goals': 2.2,
            'home_avg_conceded': 0.8,
            'away_avg_conceded': 0.9,
            'asian_handicap': {'line': 0.0, 'home_odds': 2.00, 'away_odds': 1.90},
            'over_under': {'line': 2.5, 'over_odds': 1.88, 'under_odds': 1.92},
        },
        {
            'id': 30,
            'home_team': 'Manchester City',
            'away_team': 'Barcelona',
            'date': (start_date.date() + timedelta(days=2)).strftime('%Y-%m-%d'),
            'time': '03:00',
            'venue': 'Etihad Stadium',
            'league': 'UEFA Champions League',
            'home_form': ['W', 'W', 'W', 'W', 'W'],
            'away_form': ['W', 'L', 'W', 'W', 'D'],
            'home_position': 1,
            'away_position': 3,
            'home_points': 0,
            'away_points': 0,
            'head_to_head': {'home_wins': 2, 'draws': 1, 'away_wins': 2},
            'home_avg_goals': 2.4,
            'away_avg_goals': 2.0,
            'home_avg_conceded': 0.7,
            'away_avg_conceded': 1.0,
            'asian_handicap': {'line': -0.5, 'home_odds': 1.92, 'away_odds': 1.88},
            'over_under': {'line': 2.5, 'over_odds': 1.85, 'under_odds': 1.95},
        },
    ])
    
    # UEFA Europa League
    matches.extend([
        {
            'id': 31,
            'home_team': 'Liverpool',
            'away_team': 'Atalanta',
            'date': (start_date.date() + timedelta(days=1)).strftime('%Y-%m-%d'),
            'time': '03:00',
            'venue': 'Anfield',
            'league': 'UEFA Europa League',
            'home_form': ['W', 'D', 'W', 'W', 'D'],
            'away_form': ['W', 'W', 'L', 'W', 'D'],
            'home_position': 2,
            'away_position': 5,
            'home_points': 0,
            'away_points': 0,
            'head_to_head': {'home_wins': 1, 'draws': 0, 'away_wins': 0},
            'home_avg_goals': 2.1,
            'away_avg_goals': 1.9,
            'home_avg_conceded': 0.9,
            'away_avg_conceded': 1.1,
            'asian_handicap': {'line': -0.5, 'home_odds': 1.88, 'away_odds': 1.92},
            'over_under': {'line': 2.5, 'over_odds': 1.87, 'under_odds': 1.93},
        },
    ])
    
    # UEFA Conference League
    matches.extend([
        {
            'id': 32,
            'home_team': 'AS Roma',
            'away_team': 'Feyenoord',
            'date': (start_date.date() + timedelta(days=2)).strftime('%Y-%m-%d'),
            'time': '03:00',
            'venue': 'Stadio Olimpico',
            'league': 'UEFA Conference League',
            'home_form': ['D', 'W', 'L', 'W', 'D'],
            'away_form': ['W', 'D', 'W', 'W', 'L'],
            'home_position': 6,
            'away_position': 3,
            'home_points': 0,
            'away_points': 0,
            'head_to_head': {'home_wins': 1, 'draws': 0, 'away_wins': 0},
            'home_avg_goals': 1.6,
            'away_avg_goals': 1.7,
            'home_avg_conceded': 1.2,
            'away_avg_conceded': 1.0,
            'asian_handicap': {'line': 0.0, 'home_odds': 1.95, 'away_odds': 1.85},
            'over_under': {'line': 2.5, 'over_odds': 1.90, 'under_odds': 1.90},
        },
    ])
    
    # English FA Cup
    matches.extend([
        {
            'id': 33,
            'home_team': 'Arsenal',
            'away_team': 'Chelsea',
            'date': (start_date.date() + timedelta(days=3)).strftime('%Y-%m-%d'),
            'time': '22:00',
            'venue': 'Emirates Stadium',
            'league': 'English FA Cup',
            'home_form': ['W', 'W', 'D', 'W', 'W'],
            'away_form': ['D', 'L', 'W', 'D', 'W'],
            'home_position': 1,
            'away_position': 9,
            'home_points': 0,
            'away_points': 0,
            'head_to_head': {'home_wins': 2, 'draws': 2, 'away_wins': 1},
            'home_avg_goals': 2.2,
            'away_avg_goals': 1.5,
            'home_avg_conceded': 0.8,
            'away_avg_conceded': 1.3,
            'asian_handicap': {'line': -0.75, 'home_odds': 1.87, 'away_odds': 1.93},
            'over_under': {'line': 2.5, 'over_odds': 1.89, 'under_odds': 1.91},
        },
    ])
    
    # English Carabao Cup
    matches.extend([
        {
            'id': 34,
            'home_team': 'Tottenham',
            'away_team': 'Newcastle United',
            'date': (start_date.date() + timedelta(days=4)).strftime('%Y-%m-%d'),
            'time': '21:00',
            'venue': 'Tottenham Hotspur Stadium',
            'league': 'English Carabao Cup',
            'home_form': ['W', 'D', 'W', 'L', 'W'],
            'away_form': ['L', 'W', 'D', 'W', 'L'],
            'home_position': 5,
            'away_position': 7,
            'home_points': 0,
            'away_points': 0,
            'head_to_head': {'home_wins': 2, 'draws': 1, 'away_wins': 2},
            'home_avg_goals': 1.9,
            'away_avg_goals': 1.7,
            'home_avg_conceded': 1.1,
            'away_avg_conceded': 1.2,
            'asian_handicap': {'line': -0.25, 'home_odds': 1.92, 'away_odds': 1.88},
            'over_under': {'line': 2.5, 'over_odds': 1.88, 'under_odds': 1.92},
        },
    ])
    
    # MLS (Mỹ & Canada)
    matches.extend([
        {
            'id': 35,
            'home_team': 'LA Galaxy',
            'away_team': 'Inter Miami',
            'date': (start_date.date() + timedelta(days=1)).strftime('%Y-%m-%d'),
            'time': '10:00',
            'venue': 'Dignity Health Sports Park',
            'league': 'MLS',
            'home_form': ['W', 'D', 'W', 'L', 'W'],
            'away_form': ['W', 'W', 'D', 'W', 'D'],
            'home_position': 3,
            'away_position': 2,
            'home_points': 45,
            'away_points': 52,
            'head_to_head': {'home_wins': 1, 'draws': 1, 'away_wins': 1},
            'home_avg_goals': 1.8,
            'away_avg_goals': 2.0,
            'home_avg_conceded': 1.3,
            'away_avg_conceded': 1.1,
            'asian_handicap': {'line': 0.0, 'home_odds': 2.05, 'away_odds': 1.85},
            'over_under': {'line': 2.5, 'over_odds': 1.90, 'under_odds': 1.90},
        },
    ])
    
    # AFC Champions League
    matches.extend([
        {
            'id': 36,
            'home_team': 'Al-Hilal',
            'away_team': 'Urawa Red Diamonds',
            'date': (start_date.date() + timedelta(days=2)).strftime('%Y-%m-%d'),
            'time': '23:00',
            'venue': 'King Fahd International Stadium',
            'league': 'AFC Champions League',
            'home_form': ['W', 'W', 'W', 'D', 'W'],
            'away_form': ['W', 'D', 'W', 'W', 'L'],
            'home_position': 1,
            'away_position': 2,
            'home_points': 0,
            'away_points': 0,
            'head_to_head': {'home_wins': 2, 'draws': 0, 'away_wins': 1},
            'home_avg_goals': 2.1,
            'away_avg_goals': 1.8,
            'home_avg_conceded': 0.9,
            'away_avg_conceded': 1.0,
            'asian_handicap': {'line': -0.5, 'home_odds': 1.90, 'away_odds': 1.90},
            'over_under': {'line': 2.5, 'over_odds': 1.88, 'under_odds': 1.92},
        },
    ])
    
    # Copa Libertadores
    matches.extend([
        {
            'id': 37,
            'home_team': 'Flamengo',
            'away_team': 'Palmeiras',
            'date': (start_date.date() + timedelta(days=3)).strftime('%Y-%m-%d'),
            'time': '05:00',
            'venue': 'Maracanã',
            'league': 'Copa Libertadores',
            'home_form': ['W', 'W', 'D', 'W', 'W'],
            'away_form': ['W', 'D', 'W', 'L', 'W'],
            'home_position': 1,
            'away_position': 2,
            'home_points': 0,
            'away_points': 0,
            'head_to_head': {'home_wins': 2, 'draws': 1, 'away_wins': 2},
            'home_avg_goals': 2.0,
            'away_avg_goals': 1.9,
            'home_avg_conceded': 1.0,
            'away_avg_conceded': 0.9,
            'asian_handicap': {'line': 0.0, 'home_odds': 2.00, 'away_odds': 1.90},
            'over_under': {'line': 2.5, 'over_odds': 1.88, 'under_odds': 1.92},
        },
    ])
    
    # FIFA World Cup (mẫu)
    matches.extend([
        {
            'id': 38,
            'home_team': 'Brazil',
            'away_team': 'Argentina',
            'date': (start_date.date() + timedelta(days=5)).strftime('%Y-%m-%d'),
            'time': '02:00',
            'venue': 'Estádio do Maracanã',
            'league': 'FIFA World Cup',
            'home_form': ['W', 'W', 'W', 'D', 'W'],
            'away_form': ['W', 'W', 'D', 'W', 'W'],
            'home_position': 1,
            'away_position': 2,
            'home_points': 0,
            'away_points': 0,
            'head_to_head': {'home_wins': 3, 'draws': 1, 'away_wins': 2},
            'home_avg_goals': 2.2,
            'away_avg_goals': 2.1,
            'home_avg_conceded': 0.8,
            'away_avg_conceded': 0.9,
            'asian_handicap': {'line': 0.0, 'home_odds': 2.10, 'away_odds': 1.90},
            'over_under': {'line': 2.5, 'over_odds': 1.87, 'under_odds': 1.93},
        },
    ])
    
    # UEFA Euro (mẫu)
    matches.extend([
        {
            'id': 39,
            'home_team': 'France',
            'away_team': 'Germany',
            'date': (start_date.date() + timedelta(days=6)).strftime('%Y-%m-%d'),
            'time': '02:00',
            'venue': 'Stade de France',
            'league': 'UEFA Euro',
            'home_form': ['W', 'W', 'D', 'W', 'W'],
            'away_form': ['W', 'D', 'W', 'W', 'D'],
            'home_position': 1,
            'away_position': 3,
            'home_points': 0,
            'away_points': 0,
            'head_to_head': {'home_wins': 2, 'draws': 2, 'away_wins': 1},
            'home_avg_goals': 2.0,
            'away_avg_goals': 1.8,
            'home_avg_conceded': 0.9,
            'away_avg_conceded': 1.0,
            'asian_handicap': {'line': -0.25, 'home_odds': 1.95, 'away_odds': 1.85},
            'over_under': {'line': 2.5, 'over_odds': 1.89, 'under_odds': 1.91},
        },
    ])
    
    # Copa America
    matches.extend([
        {
            'id': 40,
            'home_team': 'Brazil',
            'away_team': 'Uruguay',
            'date': (start_date.date() + timedelta(days=4)).strftime('%Y-%m-%d'),
            'time': '05:00',
            'venue': 'Estádio do Maracanã',
            'league': 'Copa America',
            'home_form': ['W', 'W', 'W', 'D', 'W'],
            'away_form': ['W', 'D', 'W', 'W', 'D'],
            'home_position': 1,
            'away_position': 2,
            'home_points': 0,
            'away_points': 0,
            'head_to_head': {'home_wins': 3, 'draws': 1, 'away_wins': 1},
            'home_avg_goals': 2.1,
            'away_avg_goals': 1.9,
            'home_avg_conceded': 0.8,
            'away_avg_conceded': 0.9,
            'asian_handicap': {'line': -0.25, 'home_odds': 1.93, 'away_odds': 1.87},
            'over_under': {'line': 2.5, 'over_odds': 1.88, 'under_odds': 1.92},
        },
    ])
    
    # Asian Cup
    matches.extend([
        {
            'id': 41,
            'home_team': 'Japan',
            'away_team': 'South Korea',
            'date': (start_date.date() + timedelta(days=5)).strftime('%Y-%m-%d'),
            'time': '20:00',
            'venue': 'National Stadium',
            'league': 'Asian Cup',
            'home_form': ['W', 'W', 'W', 'D', 'W'],
            'away_form': ['W', 'D', 'W', 'W', 'W'],
            'home_position': 1,
            'away_position': 2,
            'home_points': 0,
            'away_points': 0,
            'head_to_head': {'home_wins': 2, 'draws': 2, 'away_wins': 1},
            'home_avg_goals': 2.0,
            'away_avg_goals': 1.9,
            'home_avg_conceded': 0.9,
            'away_avg_conceded': 1.0,
            'asian_handicap': {'line': 0.0, 'home_odds': 2.00, 'away_odds': 1.90},
            'over_under': {'line': 2.5, 'over_odds': 1.87, 'under_odds': 1.93},
        },
    ])
    
    # Africa Cup of Nations
    matches.extend([
        {
            'id': 42,
            'home_team': 'Senegal',
            'away_team': 'Morocco',
            'date': (start_date.date() + timedelta(days=6)).strftime('%Y-%m-%d'),
            'time': '02:00',
            'venue': 'Stade Léopold Sédar Senghor',
            'league': 'Africa Cup of Nations',
            'home_form': ['W', 'W', 'D', 'W', 'W'],
            'away_form': ['W', 'W', 'W', 'D', 'W'],
            'home_position': 1,
            'away_position': 2,
            'home_points': 0,
            'away_points': 0,
            'head_to_head': {'home_wins': 2, 'draws': 1, 'away_wins': 2},
            'home_avg_goals': 1.9,
            'away_avg_goals': 1.8,
            'home_avg_conceded': 0.9,
            'away_avg_conceded': 1.0,
            'asian_handicap': {'line': 0.0, 'home_odds': 2.05, 'away_odds': 1.85},
            'over_under': {'line': 2.5, 'over_odds': 1.89, 'under_odds': 1.91},
        },
    ])
    
    return matches

def calculate_prediction(match):
    """Tính toán dự đoán dựa trên form và thống kê"""
    home_strength = sum([3 if r == 'W' else 1 if r == 'D' else 0 for r in match['home_form']]) / 15
    away_strength = sum([3 if r == 'W' else 1 if r == 'D' else 0 for r in match['away_form']]) / 15
    
    home_advantage = 0.1  # Lợi thế sân nhà
    home_score = round((match['home_avg_goals'] * (1 + home_strength + home_advantage)) - (match['away_avg_conceded'] * 0.5), 1)
    away_score = round((match['away_avg_goals'] * (1 + away_strength)) - (match['home_avg_conceded'] * 0.5), 1)
    
    home_score = max(0, min(4, int(home_score)))
    away_score = max(0, min(4, int(away_score)))
    
    # Xác suất
    total = home_strength + away_strength + 0.2
    home_win_prob = round((home_strength + 0.1) / total * 100, 1)
    draw_prob = round(0.2 / total * 100, 1)
    away_win_prob = round(away_strength / total * 100, 1)
    
    return {
        'predicted_score': f"{home_score}-{away_score}",
        'home_win_prob': home_win_prob,
        'draw_prob': draw_prob,
        'away_win_prob': away_win_prob,
        'predicted_total_goals': home_score + away_score
    }

def analyze_asian_handicap(match, prediction):
    """Phân tích kèo chấp châu Á"""
    ah = match['asian_handicap']
    predicted_diff = float(prediction['predicted_score'].split('-')[0]) - float(prediction['predicted_score'].split('-')[1])
    
    # Xác định kèo chấp
    handicap_line = ah['line']
    
    # Tính toán kết quả sau khi áp dụng chấp
    home_result_after_handicap = predicted_diff - handicap_line
    
    # Dự đoán
    abs_diff = abs(home_result_after_handicap)
    if home_result_after_handicap > 0.5:
        recommendation = f"Chọn {match['home_team']} (chấp {handicap_line:+.1f})"
        # Xác suất dựa trên độ chênh lệch
        win_prob = min(75, 50 + int(abs_diff * 10))
    elif home_result_after_handicap < -0.5:
        recommendation = f"Chọn {match['away_team']} (nhận chấp {handicap_line:+.1f})"
        win_prob = min(75, 50 + int(abs_diff * 10))
    else:
        recommendation = "Hòa kèo - Hoàn tiền"
        win_prob = 50
    
    return {
        'handicap_line': handicap_line,
        'home_odds': ah['home_odds'],
        'away_odds': ah['away_odds'],
        'predicted_diff': round(predicted_diff, 1),
        'result_after_handicap': round(home_result_after_handicap, 1),
        'recommendation': recommendation,
        'win_probability': win_prob
    }

def analyze_over_under(match, prediction):
    """Phân tích kèo tài xỉu"""
    ou = match['over_under']
    predicted_total = prediction['predicted_total_goals']
    over_under_line = ou['line']
    
    # Dự đoán
    diff = predicted_total - over_under_line
    abs_diff = abs(diff)
    if diff > 0.3:
        recommendation = f"Chọn Tài {over_under_line}"
        # Xác suất dựa trên độ chênh lệch
        win_prob = min(75, 50 + int(abs_diff * 15))
    elif diff < -0.3:
        recommendation = f"Chọn Xỉu {over_under_line}"
        win_prob = min(75, 50 + int(abs_diff * 15))
    else:
        recommendation = "Gần với mức kèo - Cân nhắc kỹ"
        win_prob = 50
    
    return {
        'over_under_line': over_under_line,
        'over_odds': ou['over_odds'],
        'under_odds': ou['under_odds'],
        'predicted_total': round(predicted_total, 1),
        'recommendation': recommendation,
        'win_probability': win_prob
    }

def generate_prediction_reasoning(match, prediction):
    """Tạo lý do dự đoán chi tiết"""
    reasons = []
    
    # Phân tích form
    home_wins = match['home_form'].count('W')
    home_draws = match['home_form'].count('D')
    home_losses = match['home_form'].count('L')
    home_form_score = home_wins * 3 + home_draws
    
    away_wins = match['away_form'].count('W')
    away_draws = match['away_form'].count('D')
    away_losses = match['away_form'].count('L')
    away_form_score = away_wins * 3 + away_draws
    
    if home_form_score > away_form_score + 3:
        reasons.append(f"📈 **Form gần đây:** {match['home_team']} có form tốt hơn với {home_wins} thắng, {home_draws} hòa trong 5 trận gần nhất, trong khi {match['away_team']} có {away_wins} thắng, {away_draws} hòa.")
    elif away_form_score > home_form_score + 3:
        reasons.append(f"📈 **Form gần đây:** {match['away_team']} có form tốt hơn với {away_wins} thắng, {away_draws} hòa trong 5 trận gần nhất, trong khi {match['home_team']} có {home_wins} thắng, {home_draws} hòa.")
    else:
        reasons.append(f"📈 **Form gần đây:** Cả hai đội có form tương đương - {match['home_team']} ({home_wins}W/{home_draws}D/{home_losses}L) vs {match['away_team']} ({away_wins}W/{away_draws}D/{away_losses}L).")
    
    # Phân tích vị trí và điểm số
    position_diff = match['away_position'] - match['home_position']
    points_diff = match['home_points'] - match['away_points']
    
    if position_diff > 3:
        reasons.append(f"🏆 **Vị trí bảng xếp hạng:** {match['home_team']} đang ở vị trí {match['home_position']}, cao hơn {match['away_team']} ({match['away_position']}) {position_diff} bậc, cho thấy sức mạnh vượt trội.")
    elif position_diff < -3:
        reasons.append(f"🏆 **Vị trí bảng xếp hạng:** {match['away_team']} đang ở vị trí {match['away_position']}, cao hơn {match['home_team']} ({match['home_position']}) {abs(position_diff)} bậc, thể hiện phong độ tốt hơn.")
    else:
        reasons.append(f"🏆 **Vị trí bảng xếp hạng:** Hai đội có vị trí gần nhau - {match['home_team']} (#{match['home_position']}, {match['home_points']} điểm) vs {match['away_team']} (#{match['away_position']}, {match['away_points']} điểm).")
    
    if points_diff > 10:
        reasons.append(f"📊 **Chênh lệch điểm số:** {match['home_team']} dẫn trước {match['away_team']} {points_diff} điểm, cho thấy sự ổn định và chất lượng tốt hơn trong mùa giải.")
    elif points_diff < -10:
        reasons.append(f"📊 **Chênh lệch điểm số:** {match['away_team']} dẫn trước {match['home_team']} {abs(points_diff)} điểm, thể hiện phong độ vượt trội trong mùa giải.")
    
    # Phân tích tấn công
    attack_diff = match['home_avg_goals'] - match['away_avg_goals']
    if attack_diff > 0.4:
        reasons.append(f"⚽ **Khả năng tấn công:** {match['home_team']} có khả năng ghi bàn tốt hơn với trung bình {match['home_avg_goals']:.1f} bàn/trận so với {match['away_avg_goals']:.1f} bàn/trận của {match['away_team']}.")
    elif attack_diff < -0.4:
        reasons.append(f"⚽ **Khả năng tấn công:** {match['away_team']} có khả năng tấn công mạnh hơn với trung bình {match['away_avg_goals']:.1f} bàn/trận so với {match['home_avg_goals']:.1f} bàn/trận của {match['home_team']}.")
    
    # Phân tích phòng thủ
    defense_diff = match['away_avg_conceded'] - match['home_avg_conceded']
    if defense_diff > 0.3:
        reasons.append(f"🛡️ **Hàng phòng thủ:** {match['home_team']} có hàng phòng thủ chắc chắn hơn, chỉ để lọt lưới trung bình {match['home_avg_conceded']:.1f} bàn/trận so với {match['away_avg_conceded']:.1f} bàn/trận của {match['away_team']}.")
    elif defense_diff < -0.3:
        reasons.append(f"🛡️ **Hàng phòng thủ:** {match['away_team']} có hàng phòng thủ tốt hơn, chỉ để lọt lưới trung bình {match['away_avg_conceded']:.1f} bàn/trận so với {match['home_avg_conceded']:.1f} bàn/trận của {match['home_team']}.")
    
    # Phân tích lịch sử đối đầu
    h2h = match['head_to_head']
    total_h2h = h2h['home_wins'] + h2h['draws'] + h2h['away_wins']
    if total_h2h > 0:
        if h2h['home_wins'] > h2h['away_wins']:
            reasons.append(f"⚔️ **Lịch sử đối đầu:** Trong {total_h2h} trận gần đây, {match['home_team']} thắng {h2h['home_wins']} lần, hòa {h2h['draws']} lần, cho thấy lợi thế tâm lý khi đối đầu.")
        elif h2h['away_wins'] > h2h['home_wins']:
            reasons.append(f"⚔️ **Lịch sử đối đầu:** Trong {total_h2h} trận gần đây, {match['away_team']} thắng {h2h['away_wins']} lần, hòa {h2h['draws']} lần, có lợi thế tâm lý khi đối đầu.")
        else:
            reasons.append(f"⚔️ **Lịch sử đối đầu:** Hai đội có lịch sử đối đầu cân bằng với {h2h['home_wins']}-{h2h['draws']}-{h2h['away_wins']} trong {total_h2h} trận gần đây.")
    
    # Lợi thế sân nhà
    reasons.append(f"🏠 **Lợi thế sân nhà:** {match['home_team']} được thi đấu trên sân nhà {match['venue']}, có lợi thế về cổ động viên và điều kiện sân bãi quen thuộc.")
    
    # Phân tích dự đoán tỷ số
    predicted_scores = prediction['predicted_score'].split('-')
    home_pred = int(predicted_scores[0])
    away_pred = int(predicted_scores[1])
    
    if home_pred > away_pred:
        reasons.append(f"🎯 **Dự đoán tỷ số {prediction['predicted_score']}:** {match['home_team']} được dự đoán sẽ thắng với {home_pred} bàn so với {away_pred} bàn của {match['away_team']}, dựa trên phân tích tổng hợp các yếu tố trên.")
    elif away_pred > home_pred:
        reasons.append(f"🎯 **Dự đoán tỷ số {prediction['predicted_score']}:** {match['away_team']} được dự đoán sẽ thắng với {away_pred} bàn so với {home_pred} bàn của {match['home_team']}, dựa trên phân tích tổng hợp các yếu tố trên.")
    else:
        reasons.append(f"🎯 **Dự đoán tỷ số {prediction['predicted_score']}:** Trận đấu được dự đoán sẽ hòa với {home_pred} bàn mỗi bên, phản ánh sự cân bằng giữa hai đội.")
    
    return reasons

# Load dữ liệu
upcoming_matches = load_upcoming_matches()

# Logo ở góc phải
st.markdown("""
    <div class="logo-container">
        <div style="font-size: 2rem;">⚽</div>
        <div class="logo-text">AI Football Predictor</div>
    </div>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="main-header">
        <h1>⚽ Phân tích trận bóng đá</h1>
        <p style="font-size: 1.2rem; opacity: 0.9;">Phân tích các trận sắp xảy ra</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar - Danh sách trận đấu
with st.sidebar:
    st.header("📅 Các trận sắp xảy ra")
    # Hiển thị thời gian hiện tại Việt Nam
    current_time_vn = get_vietnam_datetime()
    st.caption(f"🕐 {current_time_vn.strftime('%d/%m/%Y %H:%M')} (UTC+7)")
    st.divider()
    
    # ========== CẤU HÌNH OPENAI API KEY ==========
    st.subheader("🤖 Cấu hình OpenAI")
    
    # Kiểm tra xem đã có API key trong session state chưa
    if 'openai_api_key' not in st.session_state:
        st.session_state.openai_api_key = ''
    
    # Hiển thị trạng thái hiện tại
    current_api_key = get_openai_api_key()
    if current_api_key:
        # Hiển thị một phần API key (ẩn phần quan trọng)
        masked_key = current_api_key[:7] + "..." + current_api_key[-4:] if len(current_api_key) > 11 else "***"
        st.success(f"✅ API Key đã được cấu hình: `{masked_key}`")
        
        # Nút xóa API key
        if st.button("🗑️ Xóa API Key", use_container_width=True, type="secondary"):
            st.session_state.openai_api_key = ''
            st.rerun()
    else:
        st.warning("⚠️ Chưa có API Key. Nhập bên dưới để kích hoạt AI.")
    
    # Input để nhập API key
    api_key_input = st.text_input(
        "Nhập OpenAI API Key:",
        value=st.session_state.openai_api_key,
        type="password",
        placeholder="sk-...",
        help="Nhập API key từ https://platform.openai.com/api-keys"
    )
    
    # Nút lưu API key
    if st.button("💾 Lưu API Key", use_container_width=True, type="primary"):
        if api_key_input and api_key_input.startswith('sk-'):
            st.session_state.openai_api_key = api_key_input
            st.success("✅ API Key đã được lưu! Làm mới trang để áp dụng.")
            st.rerun()
        elif api_key_input:
            st.error("❌ API Key không hợp lệ. API Key phải bắt đầu bằng 'sk-'")
        else:
            st.warning("⚠️ Vui lòng nhập API Key")
    
    # Link hướng dẫn
    st.markdown("""
    <div style="background: #f0f9ff; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
        <p style="margin: 0; font-size: 0.85rem;">
        📖 <strong>Lấy API Key:</strong><br>
        <a href="https://platform.openai.com/api-keys" target="_blank" style="color: #667eea;">
        https://platform.openai.com/api-keys
        </a>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Date picker - Chọn ngày (theo múi giờ Việt Nam)
    today = get_vietnam_date()
    min_date = today
    max_date = today + timedelta(days=14)  # 14 ngày tới
    
    selected_date = st.date_input(
        "Chọn ngày:",
        value=today,
        min_value=min_date,
        max_value=max_date,
        help="Chọn ngày để xem các trận đấu"
    )
    
    st.divider()
    
    # Lọc trận đấu theo ngày được chọn
    selected_date_str = selected_date.strftime('%Y-%m-%d')
    filtered_matches = [m for m in upcoming_matches if m['date'] == selected_date_str]
    
    # Filter theo giải đấu
    all_leagues = sorted(list(set([m['league'] for m in upcoming_matches])))
    selected_league = st.selectbox(
        "Chọn giải đấu:",
        options=['Tất cả'] + all_leagues,
        index=0,
        help="Lọc trận đấu theo giải đấu"
    )
    
    # Áp dụng filter giải đấu
    if selected_league != 'Tất cả':
        filtered_matches = [m for m in filtered_matches if m['league'] == selected_league]
    
    if not filtered_matches:
        st.warning(f"Không có trận đấu nào vào ngày {selected_date.strftime('%d/%m/%Y')}" + 
                  (f" trong giải {selected_league}" if selected_league != 'Tất cả' else ""))
        # Hiển thị tất cả trận đấu nếu không có trận nào trong ngày được chọn
        if selected_league == 'Tất cả':
            filtered_matches = upcoming_matches
        else:
            filtered_matches = [m for m in upcoming_matches if m['league'] == selected_league]
    else:
        st.success(f"Tìm thấy {len(filtered_matches)} trận đấu vào ngày {selected_date.strftime('%d/%m/%Y')}" + 
                  (f" - {selected_league}" if selected_league != 'Tất cả' else ""))
    
    st.divider()
    
    # Tạo danh sách trận đấu để chọn
    match_options = {}
    for match in filtered_matches:
        match_date = datetime.strptime(match['date'], '%Y-%m-%d')
        # Đảm bảo datetime có timezone Việt Nam
        match_date = match_date.replace(tzinfo=VIETNAM_TZ)
        date_str = match_date.strftime('%d/%m/%Y')
        match_label = f"{date_str} - {match['league']} - {match['home_team']} vs {match['away_team']}"
        match_options[match_label] = match
    
    if match_options:
        selected_match_label = st.selectbox(
            "Chọn trận đấu để phân tích:",
            options=list(match_options.keys()),
            index=0
        )
        
        selected_match = match_options[selected_match_label]
    else:
        st.error("Không có trận đấu nào để hiển thị")
        selected_match = upcoming_matches[0] if upcoming_matches else None
    
    if selected_match:
        st.divider()
        st.subheader("ℹ️ Thông tin nhanh")
        st.write(f"**Giải đấu:** {selected_match['league']}")
        match_date_vn = datetime.strptime(selected_match['date'], '%Y-%m-%d').replace(tzinfo=VIETNAM_TZ)
        st.write(f"**Ngày:** {match_date_vn.strftime('%d/%m/%Y')} (Giờ Việt Nam)")
        st.write(f"**Giờ:** {selected_match['time']} (Giờ Việt Nam - UTC+7)")
        st.write(f"**Sân:** {selected_match['venue']}")

# Kiểm tra nếu có trận đấu được chọn
if selected_match:
    # Hiển thị thông tin trận đấu đã chọn
    st.subheader(f"📊 Phân tích: {selected_match['home_team']} vs {selected_match['away_team']}")

    # Thông tin trận đấu
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        st.markdown(f"""
            <div style="text-align: center;">
                <h2 style="font-size: 2rem; margin-bottom: 0.5rem;">🔴</h2>
                <h3>{selected_match['home_team']}</h3>
                <p style="color: #666; margin-top: 0.5rem;">Vị trí: {selected_match['home_position']}</p>
                <p style="color: #666;">Điểm: {selected_match['home_points']}</p>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        match_date = datetime.strptime(selected_match['date'], '%Y-%m-%d').replace(tzinfo=VIETNAM_TZ)
        st.markdown(f"""
            <div style="text-align: center; padding: 2rem 0;">
                <h2 style="font-size: 2.5rem; margin: 0; color: #667eea;">VS</h2>
                <p style="color: #666; margin-top: 1rem; font-size: 1.1rem;">
                    📅 {match_date.strftime('%d/%m/%Y')} (VN) | ⏰ {selected_match['time']} (UTC+7)
                </p>
                <p style="color: #666; margin-top: 0.5rem;">
                    📍 {selected_match['venue']} | 🏆 {selected_match['league']}
                </p>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
            <div style="text-align: center;">
                <h2 style="font-size: 2rem; margin-bottom: 0.5rem;">🔴</h2>
                <h3>{selected_match['away_team']}</h3>
                <p style="color: #666; margin-top: 0.5rem;">Vị trí: {selected_match['away_position']}</p>
                <p style="color: #666;">Điểm: {selected_match['away_points']}</p>
            </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Tabs cho các phần phân tích
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🔮 Dự đoán", "🎯 Phân tích kèo", "📈 So sánh", "📊 Form gần đây", "⚔️ Lịch sử đối đầu", "📋 Thống kê đội bóng"])

    with tab1:
        st.header("🤖 AI Agent - Dự đoán chuyên nghiệp")
        
        # Tính toán các metrics nâng cao
        prediction = calculate_prediction(selected_match)
        xg_data = calculate_xg_xga(selected_match)
        strength_data = calculate_team_strength(selected_match)
        ml_probs = ml_predict_probabilities(selected_match)
        value_bets = find_best_value_bets(selected_match, prediction)
        
        # Dự đoán với OpenAI
        api_key_available = get_openai_api_key()
        
        openai_prediction = None
        if api_key_available:
            with st.spinner("🤖 OpenAI đang phân tích và dự đoán..."):
                openai_prediction = predict_with_openai(selected_match, xg_data, strength_data, ml_probs)
        
        # Hiển thị dự đoán tỷ số - So sánh ML vs OpenAI
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if openai_prediction:
                # Kết hợp dự đoán ML và OpenAI
                ml_score = prediction['predicted_score']
                ai_score = openai_prediction.get('exact_score', ml_score)
                
                st.markdown(f"""
                    <div class="prediction-box">
                        <h2 style="margin: 0; font-size: 1.5rem;">Dự đoán tỷ số</h2>
                        <div style="display: flex; justify-content: center; gap: 2rem; margin: 1rem 0;">
                            <div>
                                <p style="margin: 0; font-size: 0.9rem; opacity: 0.9;">ML Model</p>
                                <h2 style="margin: 0; font-size: 2.5rem;">{ml_score}</h2>
                            </div>
                            <div style="font-size: 2rem; opacity: 0.5;">vs</div>
                            <div>
                                <p style="margin: 0; font-size: 0.9rem; opacity: 0.9;">OpenAI AI</p>
                                <h2 style="margin: 0; font-size: 2.5rem; color: #fbbf24;">{ai_score}</h2>
                            </div>
                        </div>
                        <p style="margin: 0; font-size: 0.9rem; opacity: 0.9;">Tự tin: {openai_prediction.get('confidence', 75)}%</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Cập nhật prediction với kết quả từ OpenAI nếu có
                if openai_prediction.get('exact_score'):
                    prediction['predicted_score'] = openai_prediction['exact_score']
                    prediction['predicted_total_goals'] = openai_prediction.get('total_goals', prediction.get('predicted_total_goals', 0))
            else:
                st.markdown(f"""
                    <div class="prediction-box">
                        <h2 style="margin: 0; font-size: 1.5rem;">Dự đoán tỷ số (ML Model)</h2>
                        <h1 style="margin: 1rem 0; font-size: 4rem;">{prediction['predicted_score']}</h1>
                    </div>
                """, unsafe_allow_html=True)
                
                st.info("""
                💡 **Kích hoạt OpenAI để có dự đoán chính xác hơn:**
                
                1. **Cách 1 - Biến môi trường (Khuyến nghị):**
                   ```powershell
                   $env:OPENAI_API_KEY="your-api-key-here"
                   streamlit run app.py
                   ```
                
                2. **Cách 2 - Streamlit Secrets:**
                   Tạo file `.streamlit/secrets.toml`:
                   ```toml
                   OPENAI_API_KEY = "your-api-key-here"
                   ```
                
                Sau khi thêm API key, làm mới trang để kích hoạt OpenAI.
                """)
        
        # Phân tích xG/xGA
        st.subheader("📊 Phân tích xG/xGA (Expected Goals)")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
                <div style="background: #e0f2fe; padding: 1.5rem; border-radius: 10px;">
                    <h4 style="margin: 0 0 1rem 0; color: #667eea;">🏠 {selected_match['home_team']}</h4>
                    <p><strong>xG (Expected Goals):</strong> {xg_data['home_xg']}</p>
                    <p><strong>xGA (Expected Goals Against):</strong> {xg_data['home_xga']}</p>
                    <p><strong>Sức mạnh tấn công:</strong> {strength_data['home_attack']}/100</p>
                    <p><strong>Sức mạnh phòng thủ:</strong> {strength_data['home_defense']}/100</p>
                    <p><strong>Sức mạnh tổng thể:</strong> {strength_data['home_strength']}/100</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div style="background: #fce7f3; padding: 1.5rem; border-radius: 10px;">
                    <h4 style="margin: 0 0 1rem 0; color: #f093fb;">✈️ {selected_match['away_team']}</h4>
                    <p><strong>xG (Expected Goals):</strong> {xg_data['away_xg']}</p>
                    <p><strong>xGA (Expected Goals Against):</strong> {xg_data['away_xga']}</p>
                    <p><strong>Sức mạnh tấn công:</strong> {strength_data['away_attack']}/100</p>
                    <p><strong>Sức mạnh phòng thủ:</strong> {strength_data['away_defense']}/100</p>
                    <p><strong>Sức mạnh tổng thể:</strong> {strength_data['away_strength']}/100</p>
                </div>
            """, unsafe_allow_html=True)
        
        # Xác suất từ ML Model và OpenAI
        st.subheader("🎯 Xác suất từ ML Model & OpenAI")
        
        # Cập nhật xác suất từ OpenAI nếu có
        if openai_prediction:
            ai_home_prob = openai_prediction.get('home_win_prob', ml_probs['home_win'])
            ai_draw_prob = openai_prediction.get('draw_prob', ml_probs['draw'])
            ai_away_prob = openai_prediction.get('away_win_prob', ml_probs['away_win'])
        else:
            ai_home_prob = ml_probs['home_win']
            ai_draw_prob = ml_probs['draw']
            ai_away_prob = ml_probs['away_win']
        
        prob_data = pd.DataFrame({
            'Kết quả': ['Thắng nhà', 'Hòa', 'Thắng khách'],
            'Xác suất ML (%)': [
                ml_probs['home_win'],
                ml_probs['draw'],
                ml_probs['away_win']
            ],
            'Xác suất OpenAI (%)': [
                ai_home_prob,
                ai_draw_prob,
                ai_away_prob
            ],
            'Xác suất cơ bản (%)': [
                prediction['home_win_prob'],
                prediction['draw_prob'],
                prediction['away_win_prob']
            ],
            'Đội': [selected_match['home_team'], 'Hòa', selected_match['away_team']]
        })
    
        col1, col2 = st.columns(2)
        
        with col1:
            if openai_prediction:
                fig_prob = px.bar(
                    prob_data,
                    x='Kết quả',
                    y=['Xác suất ML (%)', 'Xác suất OpenAI (%)', 'Xác suất cơ bản (%)'],
                    barmode='group',
                    color_discrete_map={
                        'Xác suất ML (%)': '#667eea',
                        'Xác suất OpenAI (%)': '#fbbf24',
                        'Xác suất cơ bản (%)': '#f093fb'
                    },
                    title='So sánh xác suất: ML vs OpenAI vs Cơ bản',
                    text='value'
                )
            else:
                fig_prob = px.bar(
                    prob_data,
                    x='Kết quả',
                    y=['Xác suất ML (%)', 'Xác suất cơ bản (%)'],
                    barmode='group',
                    color_discrete_map={
                        'Xác suất ML (%)': '#667eea',
                        'Xác suất cơ bản (%)': '#f093fb'
                    },
                    title='So sánh xác suất ML vs Cơ bản',
                    text='value'
                )
            fig_prob.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            st.plotly_chart(fig_prob, use_container_width=True)
        
        with col2:
            # Hiển thị xác suất ML và OpenAI
            if openai_prediction:
                st.metric("🏠 Thắng nhà (ML)", f"{ml_probs['home_win']}%", 
                         delta=f"OpenAI: {ai_home_prob}%")
                st.metric("🤝 Hòa (ML)", f"{ml_probs['draw']}%", 
                         delta=f"OpenAI: {ai_draw_prob}%")
                st.metric("✈️ Thắng khách (ML)", f"{ml_probs['away_win']}%", 
                         delta=f"OpenAI: {ai_away_prob}%")
                
                # Khuyến nghị dựa trên OpenAI (ưu tiên hơn)
                max_prob_ai = max(ai_home_prob, ai_draw_prob, ai_away_prob)
                if ai_home_prob == max_prob_ai and ai_home_prob >= 50:
                    recommendation = f"✅ OpenAI Khuyến nghị: {selected_match['home_team']} thắng ({ai_home_prob}%)"
                    st.success(recommendation)
                elif ai_away_prob == max_prob_ai and ai_away_prob >= 50:
                    recommendation = f"✅ OpenAI Khuyến nghị: {selected_match['away_team']} thắng ({ai_away_prob}%)"
                    st.info(recommendation)
                else:
                    st.warning(f"⚠️ OpenAI Khuyến nghị: Hòa hoặc không rõ ràng (Hòa: {ai_draw_prob}%)")
                
                if openai_prediction.get('reasoning'):
                    st.info(f"💡 **Lý do từ OpenAI:** {openai_prediction['reasoning']}")
            else:
                # Hiển thị xác suất ML
                st.metric("🏠 Thắng nhà (ML)", f"{ml_probs['home_win']}%", delta=f"{ml_probs['home_win'] - prediction['home_win_prob']:.1f}%")
                st.metric("🤝 Hòa (ML)", f"{ml_probs['draw']}%", delta=f"{ml_probs['draw'] - prediction['draw_prob']:.1f}%")
                st.metric("✈️ Thắng khách (ML)", f"{ml_probs['away_win']}%", delta=f"{ml_probs['away_win'] - prediction['away_win_prob']:.1f}%")
                
                # Khuyến nghị dựa trên ML
                max_prob_ml = max(ml_probs['home_win'], ml_probs['draw'], ml_probs['away_win'])
                if ml_probs['home_win'] == max_prob_ml and ml_probs['home_win'] >= 50:
                    recommendation = f"✅ ML Khuyến nghị: {selected_match['home_team']} thắng ({ml_probs['home_win']}%)"
                    st.success(recommendation)
                elif ml_probs['away_win'] == max_prob_ml and ml_probs['away_win'] >= 50:
                    recommendation = f"✅ ML Khuyến nghị: {selected_match['away_team']} thắng ({ml_probs['away_win']}%)"
                    st.info(recommendation)
                else:
                    st.warning(f"⚠️ ML Khuyến nghị: Hòa hoặc không rõ ràng (Hòa: {ml_probs['draw']}%)")
        
        # Value Bets
        st.divider()
        st.subheader("💰 Value Bets - Kèo có giá trị")
        
        if value_bets:
            st.success(f"🎯 Tìm thấy {len(value_bets)} Value Bet(s)!")
            
            for i, vb in enumerate(value_bets[:5], 1):
                st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); 
                                color: white; padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem;">
                        <h4 style="margin: 0 0 0.5rem 0;">#{i} {vb['type']}</h4>
                        <p style="margin: 0.5rem 0;"><strong>{vb['recommendation']}</strong></p>
                        <div style="display: flex; gap: 2rem; margin-top: 1rem;">
                            <div>
                                <p style="margin: 0; font-size: 0.9rem; opacity: 0.9;">Xác suất AI</p>
                                <p style="margin: 0; font-size: 1.2rem; font-weight: bold;">{vb['ai_prob']}%</p>
                            </div>
                            <div>
                                <p style="margin: 0; font-size: 0.9rem; opacity: 0.9;">Xác suất nhà cái</p>
                                <p style="margin: 0; font-size: 1.2rem; font-weight: bold;">{vb['implied_prob']}%</p>
                            </div>
                            <div>
                                <p style="margin: 0; font-size: 0.9rem; opacity: 0.9;">Value</p>
                                <p style="margin: 0; font-size: 1.2rem; font-weight: bold;">+{vb['value']}%</p>
                            </div>
                            <div>
                                <p style="margin: 0; font-size: 0.9rem; opacity: 0.9;">Odds</p>
                                <p style="margin: 0; font-size: 1.2rem; font-weight: bold;">{vb['odds']}</p>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Không tìm thấy Value Bet rõ ràng. Khuyến nghị: Chờ đợi hoặc phân tích kỹ hơn.")
        
        # ========== 5 PHẦN DỰ ĐOÁN CHI TIẾT ==========
        st.divider()
        st.subheader("🎯 5 Phần Dự Đoán Chi Tiết")
        
        # Cập nhật prediction với OpenAI nếu có
        if openai_prediction:
            # Cập nhật tỷ số
            if openai_prediction.get('exact_score'):
                prediction['predicted_score'] = openai_prediction['exact_score']
            # Cập nhật tổng bàn thắng
            if openai_prediction.get('total_goals'):
                prediction['predicted_total_goals'] = openai_prediction['total_goals']
        
        # 1. Tài/Xỉu Hiệp 1
        st.markdown("### 1️⃣ Tài/Xỉu Hiệp 1")
        first_half_ou = predict_first_half_over_under(selected_match, prediction)
        
        # Cập nhật với OpenAI nếu có
        if openai_prediction and openai_prediction.get('first_half_goals'):
            ai_first_half = openai_prediction['first_half_goals']
            # Cập nhật dự đoán tốt nhất
            for pred in first_half_ou['predictions']:
                if abs(pred['predicted_goals'] - ai_first_half) < 0.3:
                    pred['predicted_goals'] = round(ai_first_half, 2)
                    pred['confidence'] = min(80, pred['confidence'] + 10)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Dự đoán bàn thắng hiệp 1", f"{first_half_ou['predicted_first_half_goals']}")
        
        with col2:
            st.markdown("**Khuyến nghị theo mức kèo:**")
            for pred in first_half_ou['predictions']:
                st.write(f"- **{pred['recommendation']}** (Tự tin: {pred['confidence']}%)")
        
        # Hiển thị bảng chi tiết
        first_half_df = pd.DataFrame(first_half_ou['predictions'])
        st.dataframe(first_half_df[['line', 'predicted_goals', 'recommendation', 'confidence']], 
                    use_container_width=True, hide_index=True,
                    column_config={
                        'line': 'Mức kèo',
                        'predicted_goals': 'Dự đoán bàn',
                        'recommendation': 'Khuyến nghị',
                        'confidence': st.column_config.NumberColumn('Tự tin (%)', format='%.1f')
                    })
        
        st.divider()
        
        # 2. Tài/Xỉu Cả Trận
        st.markdown("### 2️⃣ Tài/Xỉu Cả Trận")
        full_match_ou = predict_full_match_over_under(selected_match, prediction)
        
        # Cập nhật với OpenAI nếu có
        if openai_prediction and openai_prediction.get('total_goals'):
            ai_total = openai_prediction['total_goals']
            full_match_ou['predicted_total_goals'] = round(ai_total, 2)
            # Cập nhật các dự đoán
            for pred in full_match_ou['predictions']:
                diff = ai_total - pred['line']
                if diff > 0.3:
                    pred['recommendation'] = f"Tài {pred['line']}"
                    pred['confidence'] = min(80, 55 + diff * 15)
                elif diff < -0.3:
                    pred['recommendation'] = f"Xỉu {pred['line']}"
                    pred['confidence'] = min(80, 55 + abs(diff) * 15)
                pred['predicted_total'] = round(ai_total, 2)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Dự đoán tổng bàn thắng", f"{full_match_ou['predicted_total_goals']}")
        
        with col2:
            st.markdown("**Khuyến nghị tốt nhất:**")
            best_ou = max(full_match_ou['predictions'], key=lambda x: x['confidence'])
            st.success(f"**{best_ou['recommendation']}** - Tự tin: {best_ou['confidence']}%")
        
        # Hiển thị bảng
        full_match_df = pd.DataFrame(full_match_ou['predictions'])
        st.dataframe(full_match_df[['line', 'predicted_total', 'recommendation', 'confidence', 'value']], 
                    use_container_width=True, hide_index=True,
                    column_config={
                        'line': 'Mức kèo',
                        'predicted_total': 'Dự đoán tổng',
                        'recommendation': 'Khuyến nghị',
                        'confidence': st.column_config.NumberColumn('Tự tin (%)', format='%.1f'),
                        'value': 'Giá trị'
                    })
        
        st.divider()
        
        # 3. Tỷ Số Cả Trận
        st.markdown("### 3️⃣ Tỷ Số Cả Trận")
        exact_score = predict_exact_score(selected_match, prediction)
        
        # Cập nhật với OpenAI nếu có
        if openai_prediction and openai_prediction.get('exact_score'):
            exact_score['main_prediction'] = openai_prediction['exact_score']
            # Tăng xác suất cho tỷ số từ OpenAI
            for score_info in exact_score['possible_scores']:
                if score_info['score'] == openai_prediction['exact_score']:
                    score_info['probability'] = 45  # Tăng từ 35% lên 45%
                    score_info['description'] = 'Tỷ số dự đoán chính (OpenAI)'
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            color: white; padding: 1.5rem; border-radius: 10px; text-align: center;">
                    <h3 style="margin: 0; font-size: 1rem;">Tỷ số chính</h3>
                    <h1 style="margin: 0.5rem 0; font-size: 2.5rem;">{exact_score['main_prediction']}</h1>
                    <p style="margin: 0; font-size: 0.9rem; opacity: 0.9;">Xác suất: 35%</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("**Các tỷ số có khả năng:**")
            for i, score_info in enumerate(exact_score['possible_scores'][:3], 1):
                st.write(f"{i}. **{score_info['score']}** - {score_info['probability']}%")
        
        with col3:
            st.markdown("**Lưu ý:**")
            st.info("Tỷ số chính xác khó đoán, nên kết hợp với các kèo khác")
        
        st.divider()
        
        # 4. Tài/Xỉu Phạt Góc
        st.markdown("### 4️⃣ Tài/Xỉu Phạt Góc")
        corners_ou = predict_corners_over_under(selected_match)
        
        # Cập nhật với OpenAI nếu có
        if openai_prediction and openai_prediction.get('total_corners'):
            ai_corners = openai_prediction['total_corners']
            corners_ou['predicted_total_corners'] = round(ai_corners, 1)
            # Cập nhật các dự đoán
            for pred in corners_ou['predictions']:
                diff = ai_corners - pred['line']
                if diff > 0.5:
                    pred['recommendation'] = f"Tài {pred['line']}"
                    pred['confidence'] = min(75, 50 + diff * 10)
                elif diff < -0.5:
                    pred['recommendation'] = f"Xỉu {pred['line']}"
                    pred['confidence'] = min(75, 50 + abs(diff) * 10)
                pred['predicted_corners'] = round(ai_corners, 1)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Dự đoán tổng phạt góc", f"{corners_ou['predicted_total_corners']}")
        
        with col2:
            st.markdown("**Khuyến nghị:**")
            best_corners = max(corners_ou['predictions'], key=lambda x: x['confidence'])
            st.success(f"**{best_corners['recommendation']}** - Tự tin: {best_corners['confidence']}%")
        
        # Hiển thị bảng
        corners_df = pd.DataFrame(corners_ou['predictions'])
        st.dataframe(corners_df[['line', 'predicted_corners', 'recommendation', 'confidence']], 
                    use_container_width=True, hide_index=True,
                    column_config={
                        'line': 'Mức kèo',
                        'predicted_corners': 'Dự đoán góc',
                        'recommendation': 'Khuyến nghị',
                        'confidence': st.column_config.NumberColumn('Tự tin (%)', format='%.1f')
                    })
        
        st.divider()
        
        # 5. Hướng dẫn cá dựa vào kèo chấp
        st.markdown("### 5️⃣ Hướng Dẫn Cá Dựa Vào Kèo Chấp")
        handicap_strategy = predict_handicap_betting_strategy(selected_match, prediction)
        
        # Cập nhật với OpenAI nếu có
        if openai_prediction and openai_prediction.get('handicap_recommendation'):
            # Thêm khuyến nghị từ OpenAI
            handicap_strategy['strategies'].insert(0, {
                'bet': openai_prediction['handicap_recommendation'],
                'reason': f"Khuyến nghị từ OpenAI AI (Tự tin: {openai_prediction.get('confidence', 75)}%)",
                'confidence': openai_prediction.get('confidence', 75),
                'odds': selected_match['asian_handicap']['home_odds'],
                'recommendation': '✅ OpenAI Khuyến nghị'
            })
        
        st.markdown(f"""
            <div style="background: #f0f9ff; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #667eea;">
                <h4 style="margin: 0 0 1rem 0;">📊 Phân tích kèo chấp</h4>
                <p><strong>Kèo chấp:</strong> {handicap_strategy['handicap_line']:+.1f}</p>
                <p><strong>Chênh lệch dự đoán:</strong> {handicap_strategy['predicted_diff']:+.1f} bàn</p>
                <p><strong>Sau khi áp dụng chấp:</strong> {handicap_strategy['result_after_handicap']:+.1f} bàn</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("**💡 Chiến lược cá:**")
        for strategy in handicap_strategy['strategies']:
            if '✅' in strategy['recommendation']:
                st.success(f"**{strategy['bet']}** - {strategy['reason']} (Tự tin: {strategy['confidence']}%, Odds: {strategy['odds']})")
            elif '⚠️' in strategy['recommendation']:
                st.warning(f"**{strategy['bet']}** - {strategy['reason']} (Tự tin: {strategy['confidence']}%)")
            else:
                st.info(f"**{strategy['bet']}** - {strategy['reason']} (Tự tin: {strategy['confidence']}%, Odds: {strategy['odds']})")
        
        if handicap_strategy['tips']:
            st.markdown("**⚠️ Lưu ý quan trọng:**")
            for tip in handicap_strategy['tips']:
                st.markdown(f"- {tip}")
        
        # ========== BẢNG TỔNG HỢP DỰ ĐOÁN ==========
        st.divider()
        st.subheader("📊 Bảng Tổng Hợp Dự Đoán")
        
        # Bảng 1: Tổng hợp tất cả các dự đoán chính
        st.markdown("### 📋 Bảng 1: Tổng Hợp Dự Đoán Chính")
        
        summary_data = {
            'Loại dự đoán': [
                'Tỷ số cả trận',
                'Kết quả 1X2',
                'Tài/Xỉu cả trận',
                'Tài/Xỉu hiệp 1',
                'Tài/Xỉu phạt góc',
                'Kèo chấp châu Á'
            ],
            'Dự đoán': [
                exact_score['main_prediction'],
                f"{selected_match['home_team']} thắng ({ml_probs['home_win']}%)" if ml_probs['home_win'] == max(ml_probs['home_win'], ml_probs['draw'], ml_probs['away_win']) else 
                f"Hòa ({ml_probs['draw']}%)" if ml_probs['draw'] == max(ml_probs['home_win'], ml_probs['draw'], ml_probs['away_win']) else 
                f"{selected_match['away_team']} thắng ({ml_probs['away_win']}%)",
                f"{full_match_ou['predictions'][1]['recommendation']} ({full_match_ou['predictions'][1]['confidence']}%)",
                f"{first_half_ou['predictions'][1]['recommendation']} ({first_half_ou['predictions'][1]['confidence']}%)",
                f"{corners_ou['predictions'][1]['recommendation']} ({corners_ou['predictions'][1]['confidence']}%)",
                handicap_strategy['strategies'][0]['bet'] if handicap_strategy['strategies'] else "Không rõ"
            ],
            'Mức tự tin': [
                '35%',
                f"{max(ml_probs['home_win'], ml_probs['draw'], ml_probs['away_win'])}%",
                f"{full_match_ou['predictions'][1]['confidence']}%",
                f"{first_half_ou['predictions'][1]['confidence']}%",
                f"{corners_ou['predictions'][1]['confidence']}%",
                f"{handicap_strategy['strategies'][0]['confidence']}%" if handicap_strategy['strategies'] else "N/A"
            ],
            'Khuyến nghị': [
                '✅ Nên cá',
                '✅ Nên cá' if max(ml_probs['home_win'], ml_probs['draw'], ml_probs['away_win']) >= 60 else '⚠️ Cân nhắc',
                '✅ Nên cá' if full_match_ou['predictions'][1]['confidence'] >= 60 else '⚠️ Cân nhắc',
                '✅ Nên cá' if first_half_ou['predictions'][1]['confidence'] >= 60 else '⚠️ Cân nhắc',
                '✅ Nên cá' if corners_ou['predictions'][1]['confidence'] >= 60 else '⚠️ Cân nhắc',
                handicap_strategy['strategies'][0]['recommendation'] if handicap_strategy['strategies'] else '⚠️ Cân nhắc'
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                'Loại dự đoán': st.column_config.TextColumn('Loại dự đoán', width='medium'),
                'Dự đoán': st.column_config.TextColumn('Dự đoán', width='large'),
                'Mức tự tin': st.column_config.TextColumn('Tự tin (%)', width='small'),
                'Nguồn': st.column_config.TextColumn('Nguồn', width='small'),
                'Khuyến nghị': st.column_config.TextColumn('Khuyến nghị', width='medium')
            }
        )
        
        # Bảng 2: So sánh các mức kèo Tài/Xỉu
        st.markdown("### 📈 Bảng 2: So Sánh Các Mức Kèo Tài/Xỉu")
        
        ou_comparison_data = []
        
        # Tài/Xỉu cả trận
        for pred in full_match_ou['predictions']:
            ou_comparison_data.append({
                'Loại': 'Tài/Xỉu cả trận',
                'Mức kèo': pred['line'],
                'Dự đoán': pred['predicted_total'],
                'Khuyến nghị': pred['recommendation'],
                'Tự tin': f"{pred['confidence']}%",
                'Giá trị': pred['value']
            })
        
        # Tài/Xỉu hiệp 1
        for pred in first_half_ou['predictions']:
            ou_comparison_data.append({
                'Loại': 'Tài/Xỉu hiệp 1',
                'Mức kèo': pred['line'],
                'Dự đoán': pred['predicted_goals'],
                'Khuyến nghị': pred['recommendation'],
                'Tự tin': f"{pred['confidence']}%",
                'Giá trị': 'N/A'
            })
        
        # Tài/Xỉu phạt góc
        for pred in corners_ou['predictions']:
            ou_comparison_data.append({
                'Loại': 'Tài/Xỉu phạt góc',
                'Mức kèo': pred['line'],
                'Dự đoán': pred['predicted_corners'],
                'Khuyến nghị': pred['recommendation'],
                'Tự tin': f"{pred['confidence']}%",
                'Giá trị': 'N/A'
            })
        
        ou_comparison_df = pd.DataFrame(ou_comparison_data)
        st.dataframe(
            ou_comparison_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                'Loại': st.column_config.TextColumn('Loại kèo', width='medium'),
                'Mức kèo': st.column_config.NumberColumn('Mức kèo', format='%.2f'),
                'Dự đoán': st.column_config.NumberColumn('Dự đoán', format='%.2f'),
                'Khuyến nghị': st.column_config.TextColumn('Khuyến nghị', width='medium'),
                'Tự tin': st.column_config.TextColumn('Tự tin', width='small'),
                'Giá trị': st.column_config.TextColumn('Giá trị', width='small')
            }
        )
        
        # Bảng 3: Tổng hợp Value Bets và Khuyến nghị
        st.markdown("### 💰 Bảng 3: Value Bets & Khuyến nghị Cá")
        
        betting_recommendations = []
        
        # Thêm Value Bets
        if value_bets:
            for vb in value_bets[:3]:
                betting_recommendations.append({
                    'Loại kèo': vb['type'],
                    'Khuyến nghị': vb['recommendation'],
                    'Odds': vb['odds'],
                    'Xác suất AI': f"{vb['ai_prob']}%",
                    'Xác suất nhà cái': f"{vb['implied_prob']}%",
                    'Value': f"+{vb['value']}%",
                    'Đánh giá': '✅ Rất tốt' if vb['value'] > 10 else '✅ Tốt'
                })
        
        # Thêm kèo chấp
        if handicap_strategy['strategies']:
            best_strategy = handicap_strategy['strategies'][0]
            betting_recommendations.append({
                'Loại kèo': 'Kèo chấp châu Á',
                'Khuyến nghị': best_strategy['bet'],
                'Odds': str(best_strategy['odds']),
                'Xác suất AI': f"{best_strategy['confidence']}%",
                'Xác suất nhà cái': 'N/A',
                'Value': 'N/A',
                'Đánh giá': best_strategy['recommendation']
            })
        
        # Thêm Tài/Xỉu tốt nhất
        best_full_ou = max(full_match_ou['predictions'], key=lambda x: x['confidence'])
        betting_recommendations.append({
            'Loại kèo': 'Tài/Xỉu cả trận',
            'Khuyến nghị': best_full_ou['recommendation'],
            'Odds': '1.90',
            'Xác suất AI': f"{best_full_ou['confidence']}%",
            'Xác suất nhà cái': 'N/A',
            'Value': 'N/A',
            'Đánh giá': '✅ Tốt' if best_full_ou['confidence'] >= 60 else '⚠️ Cân nhắc'
        })
        
        if betting_recommendations:
            betting_df = pd.DataFrame(betting_recommendations)
            st.dataframe(
                betting_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    'Loại kèo': st.column_config.TextColumn('Loại kèo', width='medium'),
                    'Khuyến nghị': st.column_config.TextColumn('Khuyến nghị', width='large'),
                    'Odds': st.column_config.TextColumn('Odds', width='small'),
                    'Xác suất AI': st.column_config.TextColumn('Xác suất AI', width='small'),
                    'Xác suất nhà cái': st.column_config.TextColumn('Xác suất NC', width='small'),
                    'Value': st.column_config.TextColumn('Value', width='small'),
                    'Đánh giá': st.column_config.TextColumn('Đánh giá', width='medium')
                }
            )
        else:
            st.info("Không có khuyến nghị Value Bet rõ ràng")
        
        # Bảng 4: Tổng hợp xác suất và dự đoán
        st.markdown("### 🎯 Bảng 4: Tổng Hợp Xác Suất & Dự Đoán")
        
        probability_summary = {
            'Kết quả': ['Thắng nhà', 'Hòa', 'Thắng khách'],
            'Xác suất ML (%)': [ml_probs['home_win'], ml_probs['draw'], ml_probs['away_win']],
            'Xác suất cơ bản (%)': [prediction['home_win_prob'], prediction['draw_prob'], prediction['away_win_prob']],
            'Chênh lệch': [
                f"{ml_probs['home_win'] - prediction['home_win_prob']:+.1f}%",
                f"{ml_probs['draw'] - prediction['draw_prob']:+.1f}%",
                f"{ml_probs['away_win'] - prediction['away_win_prob']:+.1f}%"
            ],
            'Đánh giá': [
                '✅ Cao' if ml_probs['home_win'] >= 50 else '⚠️ Thấp',
                '✅ Cao' if ml_probs['draw'] >= 40 else '⚠️ Thấp',
                '✅ Cao' if ml_probs['away_win'] >= 50 else '⚠️ Thấp'
            ]
        }
        
        prob_summary_df = pd.DataFrame(probability_summary)
        st.dataframe(
            prob_summary_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                'Kết quả': st.column_config.TextColumn('Kết quả', width='medium'),
                'Xác suất ML (%)': st.column_config.NumberColumn('Xác suất ML', format='%.1f'),
                'Xác suất cơ bản (%)': st.column_config.NumberColumn('Xác suất cơ bản', format='%.1f'),
                'Chênh lệch': st.column_config.TextColumn('Chênh lệch', width='small'),
                'Đánh giá': st.column_config.TextColumn('Đánh giá', width='small')
            }
        )
        
        # Bảng 5: Tổng hợp các tỷ số có khả năng
        st.markdown("### ⚽ Bảng 5: Các Tỷ Số Có Khả Năng")
        
        score_probability_data = []
        for score_info in exact_score['possible_scores']:
            score_probability_data.append({
                'Tỷ số': score_info['score'],
                'Xác suất (%)': score_info['probability'],
                'Mô tả': score_info['description']
            })
        
        score_prob_df = pd.DataFrame(score_probability_data)
        st.dataframe(
            score_prob_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                'Tỷ số': st.column_config.TextColumn('Tỷ số', width='small'),
                'Xác suất (%)': st.column_config.NumberColumn('Xác suất', format='%.0f'),
                'Mô tả': st.column_config.TextColumn('Mô tả', width='medium')
            }
        )
        
        # Bảng 6: Tổng hợp khuyến nghị cuối cùng
        st.markdown("### 🎯 Bảng 6: Khuyến Nghị Tổng Hợp")
        
        final_recommendations = []
        
        # Khuyến nghị tốt nhất cho từng loại
        final_recommendations.append({
            'Loại': 'Kết quả trận đấu',
            'Khuyến nghị': f"{selected_match['home_team']} thắng" if ml_probs['home_win'] == max(ml_probs['home_win'], ml_probs['draw'], ml_probs['away_win']) else 
                          f"Hòa" if ml_probs['draw'] == max(ml_probs['home_win'], ml_probs['draw'], ml_probs['away_win']) else 
                          f"{selected_match['away_team']} thắng",
            'Tỷ số dự đoán': exact_score['main_prediction'],
            'Xác suất': f"{max(ml_probs['home_win'], ml_probs['draw'], ml_probs['away_win'])}%",
            'Mức độ': '✅ Rất cao' if max(ml_probs['home_win'], ml_probs['draw'], ml_probs['away_win']) >= 60 else '✅ Cao'
        })
        
        if value_bets:
            best_value = value_bets[0]
            final_recommendations.append({
                'Loại': 'Value Bet tốt nhất',
                'Khuyến nghị': best_value['recommendation'],
                'Tỷ số dự đoán': 'N/A',
                'Xác suất': f"{best_value['ai_prob']}%",
                'Mức độ': f"✅ Value: +{best_value['value']}%"
            })
        
        best_ou = max(full_match_ou['predictions'], key=lambda x: x['confidence'])
        final_recommendations.append({
            'Loại': 'Tài/Xỉu cả trận',
            'Khuyến nghị': best_ou['recommendation'],
            'Tỷ số dự đoán': f"{best_ou['predicted_total']} bàn",
            'Xác suất': f"{best_ou['confidence']}%",
            'Mức độ': '✅ Rất cao' if best_ou['confidence'] >= 70 else '✅ Cao' if best_ou['confidence'] >= 60 else '⚠️ Trung bình'
        })
        
        best_first_half = max(first_half_ou['predictions'], key=lambda x: x['confidence'])
        final_recommendations.append({
            'Loại': 'Tài/Xỉu hiệp 1',
            'Khuyến nghị': best_first_half['recommendation'],
            'Tỷ số dự đoán': f"{best_first_half['predicted_goals']} bàn",
            'Xác suất': f"{best_first_half['confidence']}%",
            'Mức độ': '✅ Cao' if best_first_half['confidence'] >= 60 else '⚠️ Trung bình'
        })
        
        if handicap_strategy['strategies']:
            final_recommendations.append({
                'Loại': 'Kèo chấp châu Á',
                'Khuyến nghị': handicap_strategy['strategies'][0]['bet'],
                'Tỷ số dự đoán': f"Chênh lệch: {handicap_strategy['predicted_diff']:+.1f}",
                'Xác suất': f"{handicap_strategy['strategies'][0]['confidence']}%",
                'Mức độ': handicap_strategy['strategies'][0]['recommendation']
            })
        
        final_rec_df = pd.DataFrame(final_recommendations)
        st.dataframe(
            final_rec_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                'Loại': st.column_config.TextColumn('Loại', width='medium'),
                'Khuyến nghị': st.column_config.TextColumn('Khuyến nghị', width='large'),
                'Tỷ số dự đoán': st.column_config.TextColumn('Dự đoán', width='medium'),
                'Xác suất': st.column_config.TextColumn('Xác suất', width='small'),
                'Mức độ': st.column_config.TextColumn('Mức độ', width='medium')
            }
        )
        
        # Tóm tắt cuối cùng
        st.markdown("""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        color: white; padding: 2rem; border-radius: 15px; margin-top: 2rem;">
                <h3 style="margin: 0 0 1rem 0; text-align: center;">📌 Tóm Tắt Khuyến Nghị Cuối Cùng</h3>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                    <div>
                        <p style="margin: 0.5rem 0;"><strong>🎯 Kèo chắc chắn nhất:</strong></p>
                        <p style="margin: 0;">{}</p>
                    </div>
                    <div>
                        <p style="margin: 0.5rem 0;"><strong>💰 Value Bet tốt nhất:</strong></p>
                        <p style="margin: 0;">{}</p>
                    </div>
                </div>
                <p style="margin-top: 1.5rem; text-align: center; font-size: 0.9rem; opacity: 0.9;">
                    ⚠️ Lưu ý: Tất cả dự đoán dựa trên phân tích dữ liệu và ML. Kết quả thực tế có thể khác.
                </p>
            </div>
        """.format(
            f"{final_recommendations[0]['Khuyến nghị']} ({final_recommendations[0]['Xác suất']})" if final_recommendations else "N/A",
            f"{value_bets[0]['recommendation']} (Value: +{value_bets[0]['value']}%)" if value_bets else "Không có"
        ), unsafe_allow_html=True)
        
        st.divider()
    
    # Phần giải thích lý do dự đoán
    st.divider()
    st.subheader("📝 Lý do dự đoán")
    
    reasoning = generate_prediction_reasoning(selected_match, prediction)
    
    # Hiển thị từng lý do trong container riêng
    for i, reason in enumerate(reasoning, 1):
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); 
                        padding: 1rem; border-radius: 10px; border-left: 4px solid #667eea; 
                        margin-bottom: 0.75rem;">
                <p style="margin: 0; line-height: 1.6;">{reason}</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Tóm tắt
    st.markdown("""
        <div style="background: #f8f9fa; padding: 1rem; border-radius: 10px; margin-top: 1rem;">
            <p style="margin: 0; color: #666; font-style: italic;">
                <strong>Lưu ý:</strong> Dự đoán dựa trên phân tích thống kê, form gần đây và lịch sử đối đầu. 
                Kết quả thực tế có thể khác do nhiều yếu tố không lường trước được như chấn thương, 
                thời tiết, và phong độ ngày thi đấu.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Phân tích với OpenAI
    st.divider()
    st.subheader("🤖 Phân tích AI với OpenAI")
    
    # Kiểm tra API key
    api_key_available = get_openai_api_key()
    
    if not api_key_available:
        st.warning("""
        ⚠️ **Chưa có OpenAI API Key!**
        
        **Để sử dụng tính năng phân tích AI với OpenAI:**
        
        **Cách 1 - Biến môi trường (Khuyến nghị):**
        ```powershell
        # Windows PowerShell
        $env:OPENAI_API_KEY="sk-your-api-key-here"
        streamlit run app.py
        ```
        
        **Cách 2 - Streamlit Secrets:**
        1. Tạo thư mục `.streamlit` trong thư mục dự án (nếu chưa có)
        2. Tạo file `secrets.toml` trong thư mục `.streamlit`
        3. Thêm dòng sau vào file:
        ```toml
        OPENAI_API_KEY = "sk-your-api-key-here"
        ```
        4. Làm mới trang web
        
        **Lấy API Key:**
        - Truy cập: https://platform.openai.com/api-keys
        - Đăng nhập và tạo API key mới
        - Copy API key và dán vào biến môi trường hoặc secrets.toml
        
        **Sau khi thêm API key, nhấn nút "🤖 Phân tích với AI Agent" bên dưới để sử dụng.**
        """)
    else:
        if st.button("🤖 Phân tích với AI Agent", type="primary", use_container_width=True):
            with st.spinner("AI Agent đang phân tích chuyên sâu, vui lòng đợi..."):
                ai_analysis = analyze_with_openai(selected_match, prediction, xg_data, strength_data, ml_probs, value_bets)
                
                if ai_analysis:
                    st.markdown("""
                        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                    color: white; padding: 1.5rem; border-radius: 15px; margin-top: 1rem;">
                            <h3 style="margin: 0 0 1rem 0;">🤖 Phân tích từ OpenAI</h3>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                        <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 10px; 
                                    margin-top: 1rem; border-left: 4px solid #667eea;">
                            <div style="white-space: pre-wrap; line-height: 1.8;">
{ai_analysis}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error("Không thể lấy phân tích từ AI. Vui lòng kiểm tra lại API key.")

    with tab2:
        st.header("🎯 Phân tích kèo")
        
        prediction = calculate_prediction(selected_match)
        ah_analysis = analyze_asian_handicap(selected_match, prediction)
        ou_analysis = analyze_over_under(selected_match, prediction)
        
        # Kèo chấp châu Á
        st.subheader("📊 Kèo chấp châu Á (Asian Handicap)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        handicap_display = f"{ah_analysis['handicap_line']:+.1f}"
        if ah_analysis['handicap_line'] > 0:
            handicap_text = f"{selected_match['home_team']} chấp {ah_analysis['handicap_line']}"
        elif ah_analysis['handicap_line'] < 0:
            handicap_text = f"{selected_match['home_team']} nhận chấp {abs(ah_analysis['handicap_line'])}"
        else:
            handicap_text = "Hòa kèo"
        
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        color: white; padding: 1.5rem; border-radius: 10px; text-align: center;">
                <h3 style="margin: 0; font-size: 1.2rem;">Kèo chấp</h3>
                <h2 style="margin: 0.5rem 0; font-size: 2rem;">{handicap_display}</h2>
                <p style="margin: 0; font-size: 0.9rem; opacity: 0.9;">{handicap_text}</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 10px; text-align: center;">
                <h3 style="margin: 0; color: #667eea;">Tỷ lệ kèo</h3>
                <div style="margin-top: 1rem;">
                    <p style="margin: 0.5rem 0;"><strong>{selected_match['home_team']}</strong></p>
                    <h2 style="margin: 0; color: #667eea;">{ah_analysis['home_odds']}</h2>
                </div>
                <div style="margin-top: 1rem;">
                    <p style="margin: 0.5rem 0;"><strong>{selected_match['away_team']}</strong></p>
                    <h2 style="margin: 0; color: #f093fb;">{ah_analysis['away_odds']}</h2>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); 
                        color: white; padding: 1.5rem; border-radius: 10px; text-align: center;">
                <h3 style="margin: 0; font-size: 1.2rem;">Dự đoán</h3>
                <p style="margin: 0.5rem 0; font-size: 0.9rem;">{ah_analysis['recommendation']}</p>
                <h2 style="margin: 0.5rem 0; font-size: 1.8rem;">{ah_analysis['win_probability']}%</h2>
                <p style="margin: 0; font-size: 0.8rem; opacity: 0.9;">Xác suất thắng</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Giải thích kèo chấp
    st.info(f"""
    **Giải thích:** 
    - Chênh lệch dự đoán: {ah_analysis['predicted_diff']:+.1f} bàn
    - Sau khi áp dụng chấp {ah_analysis['handicap_line']:+.1f}: {ah_analysis['result_after_handicap']:+.1f}
    - **{ah_analysis['recommendation']}** với xác suất thắng {ah_analysis['win_probability']}%
    """)
    
    # Biểu đồ phân tích kèo chấp
    ah_data = pd.DataFrame({
        'Tình huống': ['Thắng kèo nhà', 'Hòa kèo', 'Thắng kèo khách'],
        'Xác suất (%)': [
            ah_analysis['win_probability'] if ah_analysis['result_after_handicap'] > 0 else 20,
            15 if abs(ah_analysis['result_after_handicap']) < 0.5 else 5,
            ah_analysis['win_probability'] if ah_analysis['result_after_handicap'] < 0 else 20
        ]
    })
    
    fig_ah = px.bar(
        ah_data,
        x='Tình huống',
        y='Xác suất (%)',
        color='Tình huống',
        color_discrete_map={
            'Thắng kèo nhà': '#667eea',
            'Hòa kèo': '#fbbf24',
            'Thắng kèo khách': '#f093fb'
        },
        title='Xác suất kết quả kèo chấp châu Á',
        text='Xác suất (%)'
    )
    fig_ah.update_traces(texttemplate='%{text:.0f}%', textposition='outside')
    st.plotly_chart(fig_ah, use_container_width=True)
    
    st.divider()
    
    # Kèo Tài Xỉu
    st.subheader("⚽ Kèo Tài/Xỉu (Over/Under)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                        color: white; padding: 1.5rem; border-radius: 10px; text-align: center;">
                <h3 style="margin: 0; font-size: 1.2rem;">Mức kèo</h3>
                <h2 style="margin: 0.5rem 0; font-size: 2rem;">{ou_analysis['over_under_line']}</h2>
                <p style="margin: 0; font-size: 0.9rem; opacity: 0.9;">Tổng bàn thắng</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 10px; text-align: center;">
                <h3 style="margin: 0; color: #667eea;">Tỷ lệ kèo</h3>
                <div style="margin-top: 1rem;">
                    <p style="margin: 0.5rem 0;"><strong>Tài</strong></p>
                    <h2 style="margin: 0; color: #10b981;">{ou_analysis['over_odds']}</h2>
                </div>
                <div style="margin-top: 1rem;">
                    <p style="margin: 0.5rem 0;"><strong>Xỉu</strong></p>
                    <h2 style="margin: 0; color: #ef4444;">{ou_analysis['under_odds']}</h2>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); 
                        color: white; padding: 1.5rem; border-radius: 10px; text-align: center;">
                <h3 style="margin: 0; font-size: 1.2rem;">Dự đoán</h3>
                <p style="margin: 0.5rem 0; font-size: 0.9rem;">{ou_analysis['recommendation']}</p>
                <h2 style="margin: 0.5rem 0; font-size: 1.8rem;">{ou_analysis['win_probability']}%</h2>
                <p style="margin: 0; font-size: 0.8rem; opacity: 0.9;">Xác suất thắng</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Giải thích tài xỉu
    predicted_total = ou_analysis['predicted_total']
    over_under_line = ou_analysis['over_under_line']
    difference = predicted_total - over_under_line
    
    if difference > 0:
        explanation = f"Dự đoán tổng bàn thắng ({predicted_total}) cao hơn mức kèo ({over_under_line}) {difference:.1f} bàn"
        st.success(f"**Phân tích:** {explanation}")
    elif difference < 0:
        explanation = f"Dự đoán tổng bàn thắng ({predicted_total}) thấp hơn mức kèo ({over_under_line}) {abs(difference):.1f} bàn"
        st.info(f"**Phân tích:** {explanation}")
    else:
        st.warning(f"**Phân tích:** Dự đoán tổng bàn thắng ({predicted_total}) gần bằng mức kèo ({over_under_line})")
    
    # Biểu đồ phân tích tài xỉu
    ou_data = pd.DataFrame({
        'Lựa chọn': ['Tài', 'Xỉu'],
        'Xác suất (%)': [
            ou_analysis['win_probability'] if predicted_total > over_under_line else 100 - ou_analysis['win_probability'],
            100 - ou_analysis['win_probability'] if predicted_total > over_under_line else ou_analysis['win_probability']
        ]
    })
    
    fig_ou = px.pie(
        ou_data,
        values='Xác suất (%)',
        names='Lựa chọn',
        color='Lựa chọn',
        color_discrete_map={'Tài': '#10b981', 'Xỉu': '#ef4444'},
        title=f'Xác suất Tài/Xỉu (Dự đoán: {predicted_total} bàn, Kèo: {over_under_line})'
    )
    fig_ou.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_ou, use_container_width=True)
    
    # So sánh tổng bàn thắng dự đoán với mức kèo
    comparison_goals = pd.DataFrame({
        'Loại': ['Dự đoán', 'Mức kèo'],
        'Số bàn': [predicted_total, over_under_line]
    })
    
    fig_goals_comp = px.bar(
        comparison_goals,
        x='Loại',
        y='Số bàn',
        color='Loại',
        color_discrete_map={'Dự đoán': '#667eea', 'Mức kèo': '#fbbf24'},
        title='So sánh tổng bàn thắng dự đoán vs mức kèo',
        text='Số bàn'
    )
    fig_goals_comp.update_traces(texttemplate='%{text:.1f}', textposition='outside')
    st.plotly_chart(fig_goals_comp, use_container_width=True)
    
    # Tổng hợp khuyến nghị
    st.divider()
    st.subheader("💡 Tổng hợp khuyến nghị")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
            <div style="background: #e0f2fe; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #667eea;">
                <h4 style="margin: 0 0 0.5rem 0; color: #667eea;">🎯 Kèo chấp châu Á</h4>
                <p style="margin: 0; font-size: 1.1rem; font-weight: 600;">{ah_analysis['recommendation']}</p>
                <p style="margin: 0.5rem 0 0 0; color: #666;">Xác suất: {ah_analysis['win_probability']}%</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div style="background: #f0fdf4; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #10b981;">
                <h4 style="margin: 0 0 0.5rem 0; color: #10b981;">⚽ Kèo Tài/Xỉu</h4>
                <p style="margin: 0; font-size: 1.1rem; font-weight: 600;">{ou_analysis['recommendation']}</p>
                <p style="margin: 0.5rem 0 0 0; color: #666;">Xác suất: {ou_analysis['win_probability']}%</p>
            </div>
        """, unsafe_allow_html=True)

with tab3:
    st.header("📈 So sánh đội bóng")
    
    # So sánh vị trí và điểm số
    comparison_data = pd.DataFrame({
        'Chỉ số': ['Vị trí', 'Điểm số', 'Bàn thắng TB', 'Bàn thua TB'],
        selected_match['home_team']: [
            selected_match['home_position'],
            selected_match['home_points'],
            round(selected_match['home_avg_goals'], 1),
            round(selected_match['home_avg_conceded'], 1)
        ],
        selected_match['away_team']: [
            selected_match['away_position'],
            selected_match['away_points'],
            round(selected_match['away_avg_goals'], 1),
            round(selected_match['away_avg_conceded'], 1)
        ]
    })
    
    st.dataframe(comparison_data, use_container_width=True, hide_index=True)
    
    # Biểu đồ radar
    categories = ['Vị trí (ngược)', 'Điểm số', 'Tấn công', 'Phòng thủ']
    
    # Chuyển đổi vị trí (vị trí càng thấp càng tốt)
    home_pos_score = (21 - selected_match['home_position']) * 5
    away_pos_score = (21 - selected_match['away_position']) * 5
    
    home_values = [
        home_pos_score,
        selected_match['home_points'] * 1.5,
        selected_match['home_avg_goals'] * 20,
        (3 - selected_match['home_avg_conceded']) * 20
    ]
    
    away_values = [
        away_pos_score,
        selected_match['away_points'] * 1.5,
        selected_match['away_avg_goals'] * 20,
        (3 - selected_match['away_avg_conceded']) * 20
    ]
    
    fig_radar = go.Figure()
    
    fig_radar.add_trace(go.Scatterpolar(
        r=home_values,
        theta=categories,
        fill='toself',
        name=selected_match['home_team'],
        line_color='#667eea'
    ))
    
    fig_radar.add_trace(go.Scatterpolar(
        r=away_values,
        theta=categories,
        fill='toself',
        name=selected_match['away_team'],
        line_color='#f093fb'
    ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title="Biểu đồ radar so sánh"
    )
    
    st.plotly_chart(fig_radar, use_container_width=True)

with tab4:
    st.header("📊 Form gần đây (5 trận)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"🏠 {selected_match['home_team']}")
        form_str = ' '.join([f"**{r}**" if r == 'W' else r for r in selected_match['home_form']])
        st.markdown(f"Form: {form_str}")
        
        # Đếm kết quả
        wins = selected_match['home_form'].count('W')
        draws = selected_match['home_form'].count('D')
        losses = selected_match['home_form'].count('L')
        
        form_data = pd.DataFrame({
            'Kết quả': ['Thắng', 'Hòa', 'Thua'],
            'Số trận': [wins, draws, losses]
        })
        
        fig_form_home = px.pie(
            form_data,
            values='Số trận',
            names='Kết quả',
            color='Kết quả',
            color_discrete_map={'Thắng': '#10b981', 'Hòa': '#fbbf24', 'Thua': '#ef4444'},
            title=f"Form {selected_match['home_team']}"
        )
        st.plotly_chart(fig_form_home, use_container_width=True)
    
    with col2:
        st.subheader(f"✈️ {selected_match['away_team']}")
        form_str = ' '.join([f"**{r}**" if r == 'W' else r for r in selected_match['away_form']])
        st.markdown(f"Form: {form_str}")
        
        # Đếm kết quả
        wins = selected_match['away_form'].count('W')
        draws = selected_match['away_form'].count('D')
        losses = selected_match['away_form'].count('L')
        
        form_data = pd.DataFrame({
            'Kết quả': ['Thắng', 'Hòa', 'Thua'],
            'Số trận': [wins, draws, losses]
        })
        
        fig_form_away = px.pie(
            form_data,
            values='Số trận',
            names='Kết quả',
            color='Kết quả',
            color_discrete_map={'Thắng': '#10b981', 'Hòa': '#fbbf24', 'Thua': '#ef4444'},
            title=f"Form {selected_match['away_team']}"
        )
        st.plotly_chart(fig_form_away, use_container_width=True)
    
    # So sánh form
    st.subheader("So sánh form")
    home_form_score = sum([3 if r == 'W' else 1 if r == 'D' else 0 for r in selected_match['home_form']])
    away_form_score = sum([3 if r == 'W' else 1 if r == 'D' else 0 for r in selected_match['away_form']])
    
    form_comparison = pd.DataFrame({
        'Đội': [selected_match['home_team'], selected_match['away_team']],
        'Điểm form': [home_form_score, away_form_score]
    })
    
    fig_form_comp = px.bar(
        form_comparison,
        x='Đội',
        y='Điểm form',
        color='Đội',
        color_discrete_map={
            selected_match['home_team']: '#667eea',
            selected_match['away_team']: '#f093fb'
        },
        title='So sánh điểm form (W=3, D=1, L=0)'
    )
    st.plotly_chart(fig_form_comp, use_container_width=True)

with tab5:
    st.header("⚔️ Lịch sử đối đầu")
    
    h2h = selected_match['head_to_head']
    total_matches = h2h['home_wins'] + h2h['draws'] + h2h['away_wins']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            f"🏠 {selected_match['home_team']} thắng",
            h2h['home_wins'],
            delta=f"{round(h2h['home_wins']/total_matches*100, 1)}%" if total_matches > 0 else "0%"
        )
    
    with col2:
        st.metric(
            "🤝 Hòa",
            h2h['draws'],
            delta=f"{round(h2h['draws']/total_matches*100, 1)}%" if total_matches > 0 else "0%"
        )
    
    with col3:
        st.metric(
            f"✈️ {selected_match['away_team']} thắng",
            h2h['away_wins'],
            delta=f"{round(h2h['away_wins']/total_matches*100, 1)}%" if total_matches > 0 else "0%"
        )
    
    # Biểu đồ lịch sử đối đầu
    h2h_data = pd.DataFrame({
        'Kết quả': [
            f"{selected_match['home_team']} thắng",
            'Hòa',
            f"{selected_match['away_team']} thắng"
        ],
        'Số trận': [h2h['home_wins'], h2h['draws'], h2h['away_wins']]
    })
    
    fig_h2h = px.bar(
        h2h_data,
        x='Kết quả',
        y='Số trận',
        color='Kết quả',
        color_discrete_map={
            f"{selected_match['home_team']} thắng": '#667eea',
            'Hòa': '#fbbf24',
            f"{selected_match['away_team']} thắng": '#f093fb'
        },
        title=f'Lịch sử đối đầu (Tổng: {total_matches} trận)',
        text='Số trận'
    )
    fig_h2h.update_traces(textposition='outside')
    st.plotly_chart(fig_h2h, use_container_width=True)

with tab6:
    st.header("📋 Thống kê đội bóng")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"🏠 {selected_match['home_team']}")
        
        stats_home = pd.DataFrame({
            'Chỉ số': ['Vị trí', 'Điểm', 'Bàn thắng TB/trận', 'Bàn thua TB/trận', 'Hiệu số'],
            'Giá trị': [
                f"#{selected_match['home_position']}",
                selected_match['home_points'],
                round(selected_match['home_avg_goals'], 1),
                round(selected_match['home_avg_conceded'], 1),
                round(selected_match['home_avg_goals'] - selected_match['home_avg_conceded'], 1)
            ]
        })
        
        st.dataframe(stats_home, use_container_width=True, hide_index=True)
        
        # Biểu đồ tấn công/phòng thủ
        attack_defense_home = pd.DataFrame({
            'Loại': ['Tấn công', 'Phòng thủ'],
            'Giá trị': [
                selected_match['home_avg_goals'] * 10,
                (3 - selected_match['home_avg_conceded']) * 10
            ]
        })
        
        fig_ad_home = px.bar(
            attack_defense_home,
            x='Loại',
            y='Giá trị',
            color='Loại',
            color_discrete_map={'Tấn công': '#10b981', 'Phòng thủ': '#3b82f6'},
            title='Tấn công vs Phòng thủ'
        )
        st.plotly_chart(fig_ad_home, use_container_width=True)
    
    with col2:
        st.subheader(f"✈️ {selected_match['away_team']}")
        
        stats_away = pd.DataFrame({
            'Chỉ số': ['Vị trí', 'Điểm', 'Bàn thắng TB/trận', 'Bàn thua TB/trận', 'Hiệu số'],
            'Giá trị': [
                f"#{selected_match['away_position']}",
                selected_match['away_points'],
                round(selected_match['away_avg_goals'], 1),
                round(selected_match['away_avg_conceded'], 1),
                round(selected_match['away_avg_goals'] - selected_match['away_avg_conceded'], 1)
            ]
        })
        
        st.dataframe(stats_away, use_container_width=True, hide_index=True)
        
        # Biểu đồ tấn công/phòng thủ
        attack_defense_away = pd.DataFrame({
            'Loại': ['Tấn công', 'Phòng thủ'],
            'Giá trị': [
                selected_match['away_avg_goals'] * 10,
                (3 - selected_match['away_avg_conceded']) * 10
            ]
        })
        
        fig_ad_away = px.bar(
            attack_defense_away,
            x='Loại',
            y='Giá trị',
            color='Loại',
            color_discrete_map={'Tấn công': '#10b981', 'Phòng thủ': '#3b82f6'},
            title='Tấn công vs Phòng thủ'
        )
        st.plotly_chart(fig_ad_away, use_container_width=True)
