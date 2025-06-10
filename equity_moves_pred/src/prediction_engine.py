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
import warnings
warnings.filterwarnings('ignore')

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
#                     Utils                        #
####################################################



####################################################
#            NEW: Data Quality Classes             #
####################################################

@dataclass
class DataQualityReport:
    """Report on data quality for validation"""
    ticker: str
    is_valid: bool
    data_points: int
    avg_daily_volume: float
    price_gaps: int
    missing_days: int
    volatility_percentile: float
    issues: List[str]
    warnings: List[str]

class DataValidator:
    """Validates data quality before analysis"""
    
    MIN_DATA_POINTS = 20
    MIN_AVG_VOLUME = 50000  # Minimum average daily volume
    MAX_PRICE_GAP = 0.15    # 15% single-day move threshold
    MAX_MISSING_DAYS = 5    # Maximum missing trading days allowed
    
    @classmethod
    def validate_price_data(cls, ticker: str, price_data: pd.DataFrame, 
                          required_days: int = 30) -> DataQualityReport:
        """
        Comprehensive data quality validation
        
        Returns DataQualityReport with validation results and recommendations
        """
        issues = []
        warnings = []
        
        # Basic data availability checks
        data_points = len(price_data)
        if data_points < cls.MIN_DATA_POINTS:
            issues.append(f"Insufficient data: {data_points} points (need {cls.MIN_DATA_POINTS})")
        
        if data_points < required_days * 0.65:  # Expect ~65% of calendar days as trading days (Not a robust check)
            warnings.append(f"Limited historical data: {data_points} days vs {required_days} requested")
        
        # Volume analysis
        avg_volume = price_data['Volume'].mean()
        if avg_volume < cls.MIN_AVG_VOLUME:
            warnings.append(f"Low liquidity stock: {avg_volume:,.0f} avg volume")
        
        # Price gap analysis (potential data issues or extreme moves)
        daily_returns = price_data['Close'].pct_change().abs()
        large_gaps = (daily_returns > cls.MAX_PRICE_GAP).sum()
        if large_gaps > 2:
            warnings.append(f"Multiple large price gaps detected: {large_gaps} days >15%")
        
        # Missing trading days (weekends/holidays excluded)
        expected_trading_days = np.busday_count(
            price_data.index[0].date(), 
            price_data.index[-1].date()
        )
        actual_days = len(price_data)
        missing_days = max(0, expected_trading_days - actual_days)
        
        if missing_days > cls.MAX_MISSING_DAYS:
            issues.append(f"Too many missing trading days: {missing_days}")
        
        # Volatility analysis (for context)
        volatility = daily_returns.std() * np.sqrt(252)
        volatility_percentile = min(100, max(0, (volatility - 0.15) / 0.35 * 100))
        
        # Overall validation
        is_valid = len(issues) == 0
        
        return DataQualityReport(
            ticker=ticker,
            is_valid=is_valid,
            data_points=data_points,
            avg_daily_volume=avg_volume,
            price_gaps=large_gaps,
            missing_days=missing_days,
            volatility_percentile=volatility_percentile,
            issues=issues,
            warnings=warnings
        )

####################################################
#        ENHANCED: Prediction Data Classes         #
####################################################

@dataclass
class PredictionInput:
    """Structured input for a prediction analysis - ENHANCED"""
    ticker: str
    analysis_date: datetime
    window_start: datetime
    window_end: datetime
    price_data: pd.DataFrame
    volume_data: pd.Series
    financial_context: Dict
    data_quality: DataQualityReport  # NEW: Quality assessment
    
@dataclass
class Prediction:
    """Structured prediction output - ENHANCED"""
    ticker: str
    prediction_id: str
    analysis_date: datetime
    prediction_horizon: str
    direction: str
    confidence: float
    current_price: float
    target_price: Optional[float]
    reasoning: str
    key_factors: List[str]
    risk_factors: List[str]
    data_quality_score: float  # NEW: 0-1 score for data reliability


@dataclass
class PredictionOutcome:
    """Actual outcome vs prediction for backtesting - ENHANCED with prediction details"""
    prediction_id: str
    
    # Prediction details (for easy comparison)
    predicted_direction: str
    predicted_confidence: float
    predicted_target_price: Optional[float]
    predicted_move_pct: Optional[float]  # Expected % move if target was set
    
    # Actual outcome details
    actual_direction: str
    actual_return: float
    target_hit: bool
    
    # Timing and additional metrics
    days_to_outcome: int
    outcome_date: datetime
    max_favorable_return: float  # Best case during period
    max_adverse_return: float    # Worst case during period
    volatility_during_period: float  # Realized vol
    
    # Comparison metrics
    direction_correct: bool
    confidence_calibration: float  # How far off confidence was from reality
    
    def __str__(self) -> str:
        """Easy-to-read string representation"""
        outcome_symbol = "✅" if self.direction_correct else "❌"
        
        predicted_move_str = f" (Target: ${self.predicted_target_price:.2f})" if self.predicted_target_price else ""
        actual_move_str = f"{self.actual_return:+.1f}%"
        
        return f"""
{outcome_symbol} PREDICTION OUTCOME ({self.prediction_id})
   Predicted: {self.predicted_direction.upper()} @ {self.predicted_confidence:.0%} confidence{predicted_move_str}
   Actual:    {self.actual_direction.upper()} with {actual_move_str} return
   Target Hit: {'Yes' if self.target_hit else 'No'}
   Duration: {self.days_to_outcome} trading days
   Max Favorable: {self.max_favorable_return:+.1f}%
   Max Adverse: {self.max_adverse_return:+.1f}%
        """.strip()

####################################################
#           NEW: Advanced Metrics Calculator       #
####################################################

class AdvancedMetricsCalculator:
    """Calculate sophisticated performance metrics"""
    
    @staticmethod
    def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate annualized Sharpe ratio"""
        if not returns or len(returns) < 2:
            return 0.0
        
        returns_array = np.array(returns)
        excess_returns = returns_array - (risk_free_rate / 252)  # Daily risk-free rate
        
        if excess_returns.std() == 0:
            return 0.0
        
        return (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
    
    @staticmethod
    def calculate_max_drawdown(returns: List[float]) -> float:
        """Calculate maximum drawdown"""
        if not returns:
            return 0.0
        
        cumulative = np.cumprod(1 + np.array(returns))
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        
        return abs(drawdown.min())
    
    @staticmethod
    def calculate_win_streak_stats(outcomes: List[bool]) -> Dict[str, int]:
        """Calculate win/loss streak statistics"""
        if not outcomes:
            return {"max_win_streak": 0, "max_loss_streak": 0, "current_streak": 0}
        
        max_win = max_loss = current_win = current_loss = 0
        
        for outcome in outcomes:
            if outcome:  # Win
                current_win += 1
                current_loss = 0
            else:  # Loss
                current_loss += 1
                current_win = 0
            
            max_win = max(max_win, current_win)
            max_loss = max(max_loss, current_loss)
        
        current_streak = current_win if outcomes[-1] else -current_loss
        
        return {
            "max_win_streak": max_win,
            "max_loss_streak": max_loss,
            "current_streak": current_streak
        }
    
    @staticmethod
    def calculate_confidence_calibration(predictions: List[Prediction], 
                                       outcomes: List[PredictionOutcome]) -> Dict[str, float]:
        """Analyze how well confidence levels match actual outcomes"""
        if not predictions or not outcomes:
            return {"calibration_error": 0.0, "overconfidence_bias": 0.0}
        
        # Match predictions with outcomes
        outcome_dict = {o.prediction_id: o for o in outcomes}
        calibration_data = []
        
        for pred in predictions:
            outcome = outcome_dict.get(pred.prediction_id)
            if outcome:
                is_correct = pred.direction == outcome.actual_direction
                confidence = pred.confidence
                calibration_data.append((confidence, is_correct))
        
        if not calibration_data:
            return {"calibration_error": 0.0, "overconfidence_bias": 0.0}
        
        # Calculate calibration error (how far off confidence is from actual accuracy)
        confidences, correctness = zip(*calibration_data)
        avg_confidence = np.mean(confidences)
        actual_accuracy = np.mean(correctness)
        
        calibration_error = abs(avg_confidence - actual_accuracy)
        overconfidence_bias = avg_confidence - actual_accuracy  # Positive = overconfident
        
        return {
            "calibration_error": calibration_error,
            "overconfidence_bias": overconfidence_bias
        }

####################################################
#         ENHANCED: Analysis Engine                #
####################################################

class PredictiveAnalysisEngine:
    def __init__(self):
        self.predictions_db = []
        self.outcomes_db = []
        self.metrics_calculator = AdvancedMetricsCalculator()  # NEW

    def _parse_prediction_horizon_to_date(self, analysis_date: datetime, horizon: str) -> datetime:
        """
        Parse prediction horizon string and calculate outcome date
        
        Args:
            analysis_date: The date the prediction was made
            horizon: Horizon string like "2w", "30d", "3m", "1y"
        
        Returns:
            datetime: The calculated outcome date
            
        Examples:
            "2w" -> analysis_date + 14 days
            "30d" -> analysis_date + 30 days  
            "3m" -> analysis_date + ~90 days
            "1y" -> analysis_date + 365 days
        """
        import re
        
        # Clean the horizon string
        horizon = horizon.strip().lower()
        
        # Parse the horizon using regex
        match = re.match(r'(\d+)\s*([dwmy])', horizon)
        
        if not match:
            # Fallback: try to extract just numbers and assume days
            number_match = re.search(r'(\d+)', horizon)
            if number_match:
                days = int(number_match.group(1))
                print(f"⚠️  Could not parse horizon '{horizon}', assuming {days} days")
                return analysis_date + timedelta(days=days)
            else:
                # Ultimate fallback
                print(f"⚠️  Could not parse horizon '{horizon}', defaulting to 14 days")
                return analysis_date + timedelta(days=14)
        
        number = int(match.group(1))
        unit = match.group(2)
        
        if unit == 'd':  # days
            return analysis_date + timedelta(days=number)
        elif unit == 'w':  # weeks
            return analysis_date + timedelta(weeks=number)
        elif unit == 'm':  # months (approximate)
            return analysis_date + timedelta(days=number * 30)
        elif unit == 'y':  # years (approximate)
            return analysis_date + timedelta(days=number * 365)
        else:
            # Shouldn't reach here due to regex, but just in case
            print(f"⚠️  Unknown unit '{unit}' in horizon '{horizon}', defaulting to 14 days")
            return analysis_date + timedelta(days=14)
        
    def create_analysis_window(self, ticker: str, analysis_date: datetime, 
                             window_days: int = 30) -> PredictionInput:
        """Creates a structured analysis window - ENHANCED with validation"""
        
        window_start = analysis_date - timedelta(days=window_days)
        
        # Fetch price data
        stock = yf.Ticker(ticker)
        price_data = stock.history(start=window_start, end=analysis_date)
        
        if price_data.empty:
            raise ValueError(f"No price data available for {ticker}")
        
        # NEW: Validate data quality
        data_quality = DataValidator.validate_price_data(ticker, price_data, window_days)
        
        if not data_quality.is_valid:
            raise ValueError(f"Data quality issues for {ticker}: {', '.join(data_quality.issues)}")
        
        # Print warnings if any
        if data_quality.warnings:
            print(f"⚠️  Data quality warnings for {ticker}:")
            for warning in data_quality.warnings:
                print(f"   • {warning}")
        
        # Get financial context (unchanged)
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
            financial_context=financial_context,
            data_quality=data_quality  # NEW
        )
    
    def calculate_technical_features(self, price_data: pd.DataFrame) -> Dict:
        """Calculate key technical indicators - ENHANCED with more metrics"""
        
        close = price_data['Close']
        high = price_data['High']
        low = price_data['Low']
        volume = price_data['Volume']
        
        # Basic price metrics (unchanged)
        current_price = close.iloc[-1]
        price_change_1w = (close.iloc[-1] / close.iloc[-5] - 1) * 100 if len(close) >= 5 else 0
        price_change_1m = (close.iloc[-1] / close.iloc[0] - 1) * 100
        
        # Enhanced volatility analysis
        returns = close.pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) * 100
        recent_volatility = returns.tail(5).std() * np.sqrt(252) * 100  # NEW: Recent vol
        vol_regime = "high" if recent_volatility > volatility * 1.2 else "normal"  # NEW
        
        # Enhanced volume analysis
        avg_volume_1w = volume.tail(5).mean() if len(volume) >= 5 else volume.mean()
        avg_volume_1m = volume.mean()
        volume_trend = (avg_volume_1w / avg_volume_1m - 1) * 100
        
        # NEW: Volume spike detection
        volume_spikes = (volume > avg_volume_1m * 2).sum()  # Days with 2x+ volume
        
        # Support/Resistance levels (enhanced)
        recent_high = high.tail(10).max()
        recent_low = low.tail(10).min()
        
        # NEW: Momentum indicators
        rsi_period = min(14, len(close) - 1)
        if rsi_period >= 10:
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs)).iloc[-1]
        else:
            rsi = 50  # Neutral if insufficient data
        
        return {
            # Original metrics
            'current_price': current_price,
            'price_change_1w_pct': price_change_1w,
            'price_change_1m_pct': price_change_1m,
            'annualized_volatility': volatility,
            'volume_trend_pct': volume_trend,
            'recent_high': recent_high,
            'recent_low': recent_low,
            'distance_from_high_pct': (current_price / recent_high - 1) * 100,
            'distance_from_low_pct': (current_price / recent_low - 1) * 100,
            
            # NEW: Enhanced metrics
            'recent_volatility': recent_volatility,
            'volatility_regime': vol_regime,
            'volume_spikes': volume_spikes,
            'rsi': rsi,
            'momentum_score': (price_change_1w + price_change_1m) / 2  # Simple momentum
        }
    
    def make_prediction(self, ticker: str, analysis_date: datetime = None, prediction_horizon: str = "2w") -> Prediction:
        """Main method to create a prediction - ENHANCED with quality scoring"""
        
        if analysis_date is None:
            analysis_date = datetime.now()
        
        try:
            # Create analysis input (now includes validation)
            analysis_input = self.create_analysis_window(ticker, analysis_date)
            
            # Calculate data quality score (0-1)
            quality_score = self._calculate_data_quality_score(analysis_input.data_quality)
            
            # Calculate technical features
            technical_features = self.calculate_technical_features(analysis_input.price_data)
            
            # Get current price at end of analysis window
            current_price = analysis_input.price_data['Close'].iloc[-1]
            
            # Generate prediction using prompt
            prompt = generate_prediction_prompt(
                ticker=ticker,
                analysis_date=analysis_date,
                technical_features=technical_features,
                financial_context=analysis_input.financial_context,
                prediction_horizon=prediction_horizon
            )
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            
            # JSON parsing (unchanged)
            raw_response = response.choices[0].message.content
            
            try:
                prediction_data = json.loads(raw_response)
            except json.JSONDecodeError:
                import re
                json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw_response, re.DOTALL)
                if json_match:
                    try:
                        prediction_data = json.loads(json_match.group(1))
                    except json.JSONDecodeError:
                        raise ValueError("Could not parse extracted JSON")
                else:
                    json_match = re.search(r'(\{.*?\})', raw_response, re.DOTALL)
                    if json_match:
                        try:
                            prediction_data = json.loads(json_match.group(1))
                        except json.JSONDecodeError:
                            raise ValueError(f"No valid JSON found: {raw_response[:200]}...")
                    else:
                        raise ValueError(f"No JSON content found: {raw_response[:200]}...")
            
            # Create prediction with quality score and current price
            prediction_id = f"{ticker}_{analysis_date.strftime('%Y%m%d')}_{datetime.now().strftime('%H%M%S')}"
            print(raw_response)

            prediction = Prediction(
                ticker=ticker,
                prediction_id=prediction_id,
                analysis_date=analysis_date,
                prediction_horizon=prediction_data['prediction_horizon'],
                direction=prediction_data['direction'],
                confidence=prediction_data['confidence'],
                current_price=current_price,
                target_price=prediction_data.get('target_price'),
                reasoning=prediction_data['reasoning'],
                key_factors=prediction_data['key_factors'],
                risk_factors=prediction_data['risk_factors'],
                data_quality_score=quality_score
            )
            
            self.predictions_db.append(prediction)
            return prediction
            
        except Exception as e:
            print(f"❌ Error making prediction for {ticker}: {e}")
            raise
    
    def _calculate_data_quality_score(self, quality_report: DataQualityReport) -> float:
        """Calculate 0-1 data quality score"""
        score = 1.0
        
        # Penalize for issues and warnings
        score -= len(quality_report.issues) * 0.3
        score -= len(quality_report.warnings) * 0.1
        
        # Adjust for data sufficiency
        if quality_report.data_points < 25:
            score -= 0.1
        
        # Adjust for liquidity
        if quality_report.avg_daily_volume < 100000:
            score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    def backtest_prediction(self, prediction: Prediction, 
                            prediction_horizon: str = None,
                            outcome_date: datetime = None) -> PredictionOutcome:
        """
        Backtest a prediction - ENHANCED with prediction horizon parsing
        
        Args:
            prediction: The prediction to backtest
            prediction_horizon: Override horizon (e.g., "2w", "30d", "3m", "1y") 
                            If None, uses prediction.prediction_horizon
            outcome_date: Explicit outcome date (overrides horizon calculation)
        """
        
        # Use provided horizon or fall back to prediction's horizon
        horizon = prediction_horizon or prediction.prediction_horizon
        
        if outcome_date is None:
            outcome_date = self._parse_prediction_horizon_to_date(
                prediction.analysis_date, horizon
            )
        
        # Fetch actual price data
        stock = yf.Ticker(prediction.ticker)
        outcome_data = stock.history(
            start=prediction.analysis_date + timedelta(days=1),  # Day after prediction
            end=outcome_date + timedelta(days=1)
        )
        
        if outcome_data.empty:
            raise ValueError(f"No outcome data available for {prediction.ticker}")
        
        # Calculate outcomes
        start_price = outcome_data['Close'].iloc[0]
        end_price = outcome_data['Close'].iloc[-1]
        actual_return = (end_price / start_price - 1) * 100
        
        # Enhanced outcome analysis
        daily_returns = outcome_data['Close'].pct_change().dropna()
        max_favorable = daily_returns.max() * 100 if len(daily_returns) > 0 else 0
        max_adverse = daily_returns.min() * 100 if len(daily_returns) > 0 else 0
        period_volatility = daily_returns.std() * np.sqrt(252) * 100 if len(daily_returns) > 1 else 0
        
        # Direction determination
        if actual_return > 2:
            actual_direction = "bullish"
        elif actual_return < -2:
            actual_direction = "bearish"
        else:
            actual_direction = "neutral"
        
        # Target hit analysis
        target_hit = False
        if prediction.target_price:
            if prediction.direction == "bullish":
                target_hit = outcome_data['High'].max() >= prediction.target_price
            elif prediction.direction == "bearish":
                target_hit = outcome_data['Low'].min() <= prediction.target_price
        
        # Calculate predicted move percentage if target was set
        predicted_move_pct = None
        if prediction.target_price and prediction.current_price:
            predicted_move_pct = (prediction.target_price / prediction.current_price - 1) * 100
        
        # Direction correctness
        direction_correct = prediction.direction == actual_direction
        
        # Confidence calibration (positive = overconfident, negative = underconfident)
        confidence_calibration = prediction.confidence - (1.0 if direction_correct else 0.0)
        
        outcome = PredictionOutcome(
            prediction_id=prediction.prediction_id,
            
            # Prediction details
            predicted_direction=prediction.direction,
            predicted_confidence=prediction.confidence,
            predicted_target_price=prediction.target_price,
            predicted_move_pct=predicted_move_pct,
            
            # Actual outcome
            actual_direction=actual_direction,
            actual_return=actual_return,
            target_hit=target_hit,
            
            # Timing and metrics
            days_to_outcome=len(outcome_data),
            outcome_date=outcome_date,
            max_favorable_return=max_favorable,
            max_adverse_return=max_adverse,
            volatility_during_period=period_volatility,
            
            # Comparison
            direction_correct=direction_correct,
            confidence_calibration=confidence_calibration
        )
        
        self.outcomes_db.append(outcome)
        return outcome
    
    def get_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics - GREATLY ENHANCED"""
        
        if not self.outcomes_db:
            return {"message": "No outcomes available for analysis"}
        
        outcomes = self.outcomes_db
        predictions = {p.prediction_id: p for p in self.predictions_db}
        
        # Basic metrics (enhanced)
        total_predictions = len(outcomes)
        correct_directions = sum(1 for o in outcomes 
                               if predictions.get(o.prediction_id, {}).direction == o.actual_direction)
        direction_accuracy = correct_directions / total_predictions if total_predictions > 0 else 0
        
        # Confidence analysis
        valid_predictions = [p for p in self.predictions_db if p.prediction_id in [o.prediction_id for o in outcomes]]
        avg_confidence = np.mean([p.confidence for p in valid_predictions]) if valid_predictions else 0
        avg_data_quality = np.mean([p.data_quality_score for p in valid_predictions]) if valid_predictions else 0
        
        # Target analysis
        target_predictions = [o for o in outcomes if predictions.get(o.prediction_id, {}).target_price is not None]
        target_hit_rate = sum(o.target_hit for o in target_predictions) / len(target_predictions) if target_predictions else 0
        
        # Return analysis
        returns = [o.actual_return / 100 for o in outcomes]  # Convert to decimal
        avg_actual_return = np.mean(returns) * 100  # Back to percentage
        
        # NEW: Advanced metrics
        sharpe_ratio = self.metrics_calculator.calculate_sharpe_ratio(returns)
        max_drawdown = self.metrics_calculator.calculate_max_drawdown(returns)
        
        # Win/loss streaks
        correct_outcomes = [predictions.get(o.prediction_id, {}).direction == o.actual_direction for o in outcomes]
        streak_stats = self.metrics_calculator.calculate_win_streak_stats(correct_outcomes)
        
        # Confidence calibration
        calibration_stats = self.metrics_calculator.calculate_confidence_calibration(valid_predictions, outcomes)
        
        # Volatility analysis
        avg_volatility_during_predictions = np.mean([o.volatility_during_period for o in outcomes])
        
        return {
            # Basic metrics
            "total_predictions": total_predictions,
            "direction_accuracy": round(direction_accuracy, 3),
            "target_hit_rate": round(target_hit_rate, 3),
            "average_confidence": round(avg_confidence, 3),
            "average_data_quality": round(avg_data_quality, 3),
            "average_actual_return": round(avg_actual_return, 2),
            
            # NEW: Advanced performance metrics
            "sharpe_ratio": round(sharpe_ratio, 3),
            "max_drawdown": round(max_drawdown * 100, 2),  # As percentage
            "max_win_streak": streak_stats["max_win_streak"],
            "max_loss_streak": streak_stats["max_loss_streak"],
            "current_streak": streak_stats["current_streak"],
            
            # NEW: Calibration metrics
            "confidence_calibration_error": round(calibration_stats["calibration_error"], 3),
            "overconfidence_bias": round(calibration_stats["overconfidence_bias"], 3),
            
            # NEW: Market context
            "avg_market_volatility": round(avg_volatility_during_predictions, 1),
            
            # Legacy metric
            "confidence_calibration": round(direction_accuracy - avg_confidence, 3)
        }
    
    def print_prediction_results(self, prediction: Prediction):
        """Print formatted prediction results - ENHANCED"""
        
        # Add data quality info to the output
        quality_indicator = "🟢" if prediction.data_quality_score > 0.8 else "🟡" if prediction.data_quality_score > 0.6 else "🔴"
        
        formatted_output = format_prediction_output(prediction)
        
        # Add quality information
        quality_section = f"""
📊 DATA QUALITY: {quality_indicator} {prediction.data_quality_score:.2f}/1.00
"""
        
        # Insert quality section before reasoning
        parts = formatted_output.split("📝 REASONING:")
        if len(parts) == 2:
            enhanced_output = parts[0] + quality_section + "\n📝 REASONING:" + parts[1]
        else:
            enhanced_output = formatted_output + quality_section
        
        print(enhanced_output)

####################################################
#                Enhanced Demo                     #
####################################################

def demo_enhanced_prediction_system():
    """Demonstrate the enhanced prediction system"""
    
    engine = PredictiveAnalysisEngine()
    
    # Get ticker from user
    ticker = input("Enter ticker symbol (e.g., AAPL): ").upper().strip()
    if not ticker:
        ticker = "AAPL"
    
    print(f"\n⏳ Making enhanced prediction for {ticker}...")
    print("🔍 Validating data quality...")
    
    try:
        prediction = engine.make_prediction(ticker)
        
        # Enhanced formatting with data quality
        engine.print_prediction_results(prediction)
        
        # Show advanced metrics if we have historical predictions
        if len(engine.predictions_db) > 1:
            print("\n📈 PERFORMANCE METRICS:")
            metrics = engine.get_performance_metrics()
            for key, value in metrics.items():
                if key != "message":
                    print(f"   {key.replace('_', ' ').title()}: {value}")
        
        print("\n💡 Enhanced features:")
        print("   • Data quality validation")
        print("   • Advanced performance metrics")
        print("   • Confidence calibration analysis")
        print("   • Risk-adjusted returns (Sharpe ratio)")
        print("   • Drawdown analysis")
        
    except Exception as e:
        print(f"❌ Error making prediction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    demo_enhanced_prediction_system()