import os
import sys
import yfinance as yf
import pandas as pd
import json
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompts.prediction_engine_prompts import (
    generate_prediction_prompt,
    generate_backtest_analysis_prompt,
    generate_performance_summary_prompt,
    format_prediction_output,
    format_backtest_results
)

####################################################
#               Prediction Data Classes            #
####################################################

@dataclass
class PredictionInput:
    """Structured input for a prediction analysis"""
    ticker: str
    analysis_date: datetime
    window_start: datetime
    window_end: datetime
    price_data: pd.DataFrame
    volume_data: pd.Series
    financial_context: Dict
    
@dataclass
class Prediction:
    """Structured prediction output"""
    ticker: str
    prediction_id: str
    analysis_date: datetime
    prediction_horizon: str  # "1w", "2w", "1m"
    direction: str  # "bullish", "bearish", "neutral"
    confidence: float  # 0-1 scale
    target_price: Optional[float]
    reasoning: str
    key_factors: List[str]
    risk_factors: List[str]

@dataclass
class PredictionOutcome:
    """Actual outcome vs prediction for backtesting"""
    prediction_id: str
    actual_direction: str
    actual_return: float
    target_hit: bool
    days_to_outcome: int
    outcome_date: datetime

####################################################
#               Analysis Engine                    #
####################################################

class PredictiveAnalysisEngine:
    def __init__(self):
        self.predictions_db = []  # In-memory storage for now
        self.outcomes_db = []
        
    def create_analysis_window(self, ticker: str, analysis_date: datetime, 
                             window_days: int = 30) -> PredictionInput:
        """Creates a structured analysis window for the given ticker and date"""
        
        window_start = analysis_date - timedelta(days=window_days)
        
        # Fetch price data
        stock = yf.Ticker(ticker)
        price_data = stock.history(start=window_start, end=analysis_date)
        
        if price_data.empty:
            raise ValueError(f"No price data available for {ticker}")
        
        # Get additional context
        try:
            info = stock.info
            financials = stock.quarterly_financials
            financial_context = {
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE'),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'recent_earnings': financials.iloc[:, 0].to_dict() if not financials.empty else {}
            }
        except:
            financial_context = {}
        
        return PredictionInput(
            ticker=ticker,
            analysis_date=analysis_date,
            window_start=window_start,
            window_end=analysis_date,
            price_data=price_data,
            volume_data=price_data['Volume'],
            financial_context=financial_context
        )
    
    def calculate_technical_features(self, price_data: pd.DataFrame) -> Dict:
        """Calculate key technical indicators for the analysis"""
        
        close = price_data['Close']
        high = price_data['High']
        low = price_data['Low']
        volume = price_data['Volume']
        
        # Price metrics
        current_price = close.iloc[-1]
        price_change_1w = (close.iloc[-1] / close.iloc[-5] - 1) * 100 if len(close) >= 5 else 0
        price_change_1m = (close.iloc[-1] / close.iloc[0] - 1) * 100
        
        # Volatility
        returns = close.pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) * 100  # Annualized vol
        
        # Volume analysis
        avg_volume_1w = volume.tail(5).mean() if len(volume) >= 5 else volume.mean()
        avg_volume_1m = volume.mean()
        volume_trend = (avg_volume_1w / avg_volume_1m - 1) * 100
        
        # Support/Resistance levels
        recent_high = high.tail(10).max()
        recent_low = low.tail(10).min()
        
        return {
            'current_price': current_price,
            'price_change_1w_pct': price_change_1w,
            'price_change_1m_pct': price_change_1m,
            'annualized_volatility': volatility,
            'volume_trend_pct': volume_trend,
            'recent_high': recent_high,
            'recent_low': recent_low,
            'distance_from_high_pct': (current_price / recent_high - 1) * 100,
            'distance_from_low_pct': (current_price / recent_low - 1) * 100
        }
    
    def make_prediction(self, ticker: str, analysis_date: datetime = None) -> Prediction:
        """Main method to create a prediction for a given ticker"""
        
        if analysis_date is None:
            analysis_date = datetime.now()
        
        # Create analysis input
        analysis_input = self.create_analysis_window(ticker, analysis_date)
        
        # Calculate technical features
        technical_features = self.calculate_technical_features(analysis_input.price_data)
        
        # Generate prediction using IMPORTED prompt function
        prompt = generate_prediction_prompt(
            ticker=ticker,
            analysis_date=analysis_date,
            technical_features=technical_features,
            financial_context=analysis_input.financial_context
        )
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        
        # ROBUST JSON PARSING (fixes your NVDA issue)
        raw_response = response.choices[0].message.content
        
        try:
            # First try direct JSON parsing
            prediction_data = json.loads(raw_response)
        except json.JSONDecodeError:
            # Extract JSON from markdown code blocks
            import re
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw_response, re.DOTALL)
            if json_match:
                try:
                    prediction_data = json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    raise ValueError("Could not parse extracted JSON")
            else:
                # Last resort: find any JSON-like content
                json_match = re.search(r'(\{.*?\})', raw_response, re.DOTALL)
                if json_match:
                    try:
                        prediction_data = json.loads(json_match.group(1))
                    except json.JSONDecodeError:
                        raise ValueError(f"No valid JSON found: {raw_response[:200]}...")
                else:
                    raise ValueError(f"No JSON content found: {raw_response[:200]}...")
        
        # Rest of the method stays the same...
        prediction_id = f"{ticker}_{analysis_date.strftime('%Y%m%d')}_{datetime.now().strftime('%H%M%S')}"
        
        prediction = Prediction(
            ticker=ticker,
            prediction_id=prediction_id,
            analysis_date=analysis_date,
            prediction_horizon=prediction_data['prediction_horizon'],
            direction=prediction_data['direction'],
            confidence=prediction_data['confidence'],
            target_price=prediction_data.get('target_price'),
            reasoning=prediction_data['reasoning'],
            key_factors=prediction_data['key_factors'],
            risk_factors=prediction_data['risk_factors']
        )
        
        self.predictions_db.append(prediction)
        return prediction
    
    def print_prediction_results(self, prediction: Prediction):
        """Print formatted prediction results"""
        print(format_prediction_output(prediction))
    
    def backtest_prediction(self, prediction: Prediction, 
                          outcome_date: datetime = None) -> PredictionOutcome:
        """Backtest a prediction against actual market outcomes"""
        
        if outcome_date is None:
            # Default to 2 weeks after prediction
            outcome_date = prediction.analysis_date + timedelta(days=14)
        
        # Fetch actual price data
        stock = yf.Ticker(prediction.ticker)
        outcome_data = stock.history(
            start=prediction.analysis_date, 
            end=outcome_date + timedelta(days=1)
        )
        
        if outcome_data.empty:
            raise ValueError(f"No outcome data available for {prediction.ticker}")
        
        # Calculate actual outcomes
        start_price = outcome_data['Close'].iloc[0]
        end_price = outcome_data['Close'].iloc[-1]
        actual_return = (end_price / start_price - 1) * 100
        
        # Determine actual direction
        if actual_return > 2:
            actual_direction = "bullish"
        elif actual_return < -2:
            actual_direction = "bearish"
        else:
            actual_direction = "neutral"
        
        # Check target hit
        target_hit = False
        if prediction.target_price:
            if prediction.direction == "bullish":
                target_hit = outcome_data['High'].max() >= prediction.target_price
            elif prediction.direction == "bearish":
                target_hit = outcome_data['Low'].min() <= prediction.target_price
        
        outcome = PredictionOutcome(
            prediction_id=prediction.prediction_id,
            actual_direction=actual_direction,
            actual_return=actual_return,
            target_hit=target_hit,
            days_to_outcome=len(outcome_data) - 1,
            outcome_date=outcome_date
        )
        
        self.outcomes_db.append(outcome)
        return outcome
    
    def analyze_prediction_performance(self, prediction: Prediction, outcome: PredictionOutcome):
        """Generate LLM analysis of prediction performance"""
        
        prediction_data = {
            'ticker': prediction.ticker,
            'direction': prediction.direction,
            'confidence': prediction.confidence,
            'target_price': prediction.target_price,
            'key_factors': prediction.key_factors,
            'risk_factors': prediction.risk_factors,
            'reasoning': prediction.reasoning
        }
        
        outcome_data = {
            'actual_direction': outcome.actual_direction,
            'actual_return': outcome.actual_return,
            'target_hit': outcome.target_hit,
            'days_to_outcome': outcome.days_to_outcome
        }
        
        prompt = generate_backtest_analysis_prompt(prediction_data, outcome_data)
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        
        return response.choices[0].message.content
    
    def get_performance_metrics(self) -> Dict:
        """Calculate performance metrics for all predictions"""
        
        if not self.outcomes_db:
            return {"message": "No outcomes available for analysis"}
        
        outcomes = self.outcomes_db
        predictions = {p.prediction_id: p for p in self.predictions_db}
        
        # Direction accuracy
        correct_directions = 0
        total_predictions = len(outcomes)
        
        confidence_weighted_score = 0
        total_confidence = 0
        
        for outcome in outcomes:
            pred = predictions.get(outcome.prediction_id)
            if pred and pred.direction == outcome.actual_direction:
                correct_directions += 1
                confidence_weighted_score += pred.confidence
            total_confidence += pred.confidence if pred else 0
        
        direction_accuracy = correct_directions / total_predictions if total_predictions > 0 else 0
        avg_confidence = total_confidence / total_predictions if total_predictions > 0 else 0
        
        # Target hit rate
        target_predictions = [o for o in outcomes if predictions.get(o.prediction_id, {}).target_price]
        target_hit_rate = sum(o.target_hit for o in target_predictions) / len(target_predictions) if target_predictions else 0
        
        # Return analysis
        avg_actual_return = np.mean([o.actual_return for o in outcomes])
        
        return {
            "total_predictions": total_predictions,
            "direction_accuracy": round(direction_accuracy, 3),
            "target_hit_rate": round(target_hit_rate, 3),
            "average_confidence": round(avg_confidence, 3),
            "average_actual_return": round(avg_actual_return, 2),
            "confidence_calibration": round(direction_accuracy - avg_confidence, 3)
        }

####################################################
#                    Demo Usage                    #
####################################################

def demo_prediction_system():
    """Demonstrate the prediction system with better formatting"""
    
    engine = PredictiveAnalysisEngine()
    
    # Get ticker from user
    ticker = input("Enter ticker symbol (e.g., AAPL): ").upper().strip()
    if not ticker:
        ticker = "AAPL"
    
    print(f"\n⏳ Making prediction for {ticker}...")
    try:
        prediction = engine.make_prediction(ticker)
        
        # Use the improved formatting
        print(format_prediction_output(prediction))
        
        # Optional: Show backtest capability
        print("\n💡 Tip: Run backtest after 2 weeks to evaluate performance!")
        
    except Exception as e:
        print(f"❌ Error making prediction: {e}")
        import traceback
        traceback.print_exc()  # For debugging