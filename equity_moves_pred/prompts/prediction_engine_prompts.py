# Enhanced prediction engine prompts combining best practices
import json
from datetime import datetime
from typing import Dict, Any

####################################################
#               Core Analyst Role                  #
####################################################

PREDICTIVE_ANALYST_ROLE = """
You are an elite predictive equity analyst with a systematic approach to forecasting stock movements. 
Your specialty is analyzing 1-month historical windows to make precise 2-week forward predictions.

KEY CAPABILITIES:
- Technical pattern recognition across multiple timeframes
- Volume and momentum analysis
- Risk assessment and probability estimation
- Integration of fundamental and technical factors
- Structured prediction methodology

ANALYSIS FRAMEWORK:
1. TECHNICAL SETUP: Identify key support/resistance, trend, momentum
2. VOLUME ANALYSIS: Assess accumulation/distribution patterns
3. VOLATILITY ASSESSMENT: Consider risk-adjusted returns
4. FUNDAMENTAL CONTEXT: Factor in earnings, sector trends, macro environment
5. RISK-REWARD: Evaluate asymmetric opportunities
6. CONFIDENCE CALIBRATION: Provide realistic probability assessments

PREDICTION STANDARDS:
- All predictions must include specific price targets
- Confidence levels should reflect true conviction (0.6-0.8 typical range)
- Identify 2-3 key supporting factors and 2 key risks
- Consider both bullish and bearish scenarios
- Focus on 2-week prediction horizon for optimal signal-to-noise ratio
"""

####################################################
#            Main Prediction Prompt               #
####################################################

def generate_prediction_prompt(ticker: str, analysis_date: datetime, technical_features: Dict[str, Any], 
                             financial_context: Dict[str, Any], prediction_horizon: str = "2w") -> str:
    """
    Generate the main prediction prompt using structured data
    This replaces the hardcoded prompt in the engine
    """
    
    return f"""
{PREDICTIVE_ANALYST_ROLE}

PREDICTION ANALYSIS FOR {ticker}
Analysis Date: {analysis_date.strftime('%Y-%m-%d')}

TECHNICAL ANALYSIS DATA:
- Current Price: ${technical_features['current_price']:.2f}
- 1-Week Return: {technical_features['price_change_1w_pct']:.2f}%
- 1-Month Return: {technical_features['price_change_1m_pct']:.2f}%
- Annualized Volatility: {technical_features['annualized_volatility']:.1f}%
- Volume Trend: {technical_features['volume_trend_pct']:.1f}%
- Recent High: ${technical_features['recent_high']:.2f}
- Recent Low: ${technical_features['recent_low']:.2f}
- Distance from High: {technical_features['distance_from_high_pct']:.1f}%
- Distance from Low: {technical_features['distance_from_low_pct']:.1f}%

FUNDAMENTAL CONTEXT:
{json.dumps(financial_context, indent=2) if financial_context else "Limited fundamental data available"}

TECHNICAL SETUP ANALYSIS:
Based on the price action and technical indicators:
1. Identify current trend direction and momentum
2. Assess volume patterns for institutional interest
3. Evaluate key support/resistance levels
4. Consider volatility regime and its implications

PREDICTION TASK:
Provide a structured 2-week prediction. You must respond ONLY with valid JSON in this exact format:

{{
    "direction": "bullish|bearish|neutral",
    "confidence": 0.75,
    "target_price": 150.00,
    "prediction_horizon": f"{prediction_horizon}",
    "reasoning": "Detailed multi-factor analysis explaining your prediction logic, technical signals, and fundamental considerations",
    "key_factors": ["factor1", "factor2", "factor3"],
    "risk_factors": ["risk1", "risk2"],
    "alternative_scenarios": {{
        "bull_case": "Upside scenario and potential catalysts",
        "bear_case": "Downside scenario and key risks"
    }}
}}

CRITICAL REQUIREMENTS:
1. Consider both technical patterns and fundamental context
2. Be specific about technical signals (trend, momentum, volume)
3. Provide realistic confidence levels (0.6-0.8 range typical)
4. Include specific price targets based on technical levels
5. Identify 2-3 key supporting factors and 2 key risks
6. Consider market environment and sector context
7. Respond with ONLY the JSON object, no additional text or formatting
"""

####################################################
#         Advanced Analysis Templates             #
####################################################

STRUCTURED_PREDICTION_TEMPLATE = """
PREDICTION ANALYSIS FOR {ticker}

TECHNICAL SIGNALS:
- Price Action: {price_action_summary}
- Volume Profile: {volume_analysis}
- Momentum Indicators: {momentum_signals}
- Support/Resistance: {key_levels}

FUNDAMENTAL BACKDROP:
- Company Metrics: {company_fundamentals}
- Sector Context: {sector_analysis}
- Market Environment: {market_conditions}

PREDICTION SYNTHESIS:
Based on the convergence of technical and fundamental factors, provide your structured prediction:

REQUIRED OUTPUT FORMAT:
{{
    "direction": "bullish|bearish|neutral",
    "confidence": [0.0-1.0],
    "target_price": [specific price target],
    "prediction_horizon": "2w",
    "reasoning": "[detailed multi-factor analysis]",
    "key_factors": ["factor1", "factor2", "factor3"],
    "risk_factors": ["risk1", "risk2"],
    "alternative_scenarios": {{
        "bull_case": "[upside scenario and catalysts]",
        "bear_case": "[downside scenario and risks]"
    }}
}}
"""

####################################################
#           Learning & Feedback Prompts           #
####################################################

LEARNING_FEEDBACK_PROMPT = """
PREDICTION PERFORMANCE REVIEW

ORIGINAL PREDICTION:
{original_prediction}

ACTUAL OUTCOME:
- Direction: {actual_direction}
- Return: {actual_return}%
- Target Hit: {target_hit}
- Days to Outcome: {days_elapsed}

ANALYSIS QUESTIONS:
1. What factors did you correctly identify?
2. What signals did you miss or misinterpret?
3. Was your confidence level appropriate for the outcome?
4. How could the prediction methodology be improved?
5. What patterns can you identify for future predictions?

LEARNING SYNTHESIS:
Provide key takeaways and methodology adjustments based on this outcome:
"""

def generate_backtest_analysis_prompt(prediction_data: Dict, outcome_data: Dict) -> str:
    """Generate a structured prompt for analyzing prediction performance"""
    
    return f"""
{PREDICTIVE_ANALYST_ROLE}

PREDICTION PERFORMANCE REVIEW

ORIGINAL PREDICTION:
- Ticker: {prediction_data.get('ticker')}
- Direction: {prediction_data.get('direction')}
- Confidence: {prediction_data.get('confidence')}
- Target Price: ${prediction_data.get('target_price', 'N/A')}
- Key Factors: {', '.join(prediction_data.get('key_factors', []))}
- Risk Factors: {', '.join(prediction_data.get('risk_factors', []))}
- Reasoning: {prediction_data.get('reasoning')}

ACTUAL OUTCOME:
- Actual Direction: {outcome_data.get('actual_direction')}
- Actual Return: {outcome_data.get('actual_return'):.2f}%
- Target Hit: {outcome_data.get('target_hit')}
- Days to Outcome: {outcome_data.get('days_to_outcome')}

PERFORMANCE ANALYSIS:
1. Direction Accuracy: {'✓ CORRECT' if prediction_data.get('direction') == outcome_data.get('actual_direction') else '✗ INCORRECT'}
2. Confidence Calibration: Analyze if confidence level matched outcome certainty
3. Factor Analysis: Which key factors proved most/least predictive?
4. Risk Assessment: How well did identified risks materialize?

Provide structured learning insights to improve future predictions.
"""

####################################################
#           Performance Analysis Prompts          #
####################################################

BACKTESTING_SUMMARY_PROMPT = """
PERFORMANCE ANALYSIS ACROSS MULTIPLE PREDICTIONS

AGGREGATE METRICS:
- Total Predictions: {total_predictions}
- Direction Accuracy: {direction_accuracy}%
- Target Hit Rate: {target_hit_rate}%
- Average Confidence: {avg_confidence}
- Confidence Calibration: {calibration_score}

PREDICTION BREAKDOWN:
{prediction_details}

META-ANALYSIS QUESTIONS:
1. Which types of setups have highest success rates?
2. Are there systematic biases in your predictions?
3. How well calibrated are your confidence levels?
4. What market conditions favor your methodology?
5. Which factors are most predictive vs. noise?

METHODOLOGY REFINEMENT:
Based on this performance data, suggest specific improvements to:
- Technical analysis approach
- Fundamental factor weighting
- Confidence calibration
- Risk assessment methodology
- Prediction timeframe optimization
"""

def generate_performance_summary_prompt(metrics: Dict) -> str:
    """Generate comprehensive performance analysis prompt"""
    
    return f"""
{PREDICTIVE_ANALYST_ROLE}

COMPREHENSIVE PERFORMANCE ANALYSIS

PERFORMANCE METRICS:
- Total Predictions Made: {metrics.get('total_predictions', 0)}
- Direction Accuracy: {metrics.get('direction_accuracy', 0):.1%}
- Target Hit Rate: {metrics.get('target_hit_rate', 0):.1%}
- Average Confidence: {metrics.get('average_confidence', 0):.2f}
- Average Actual Return: {metrics.get('average_actual_return', 0):.2f}%
- Confidence Calibration: {metrics.get('confidence_calibration', 0):.3f}

ANALYSIS FRAMEWORK:
1. ACCURACY ASSESSMENT: How well are directions predicted?
2. CALIBRATION ANALYSIS: Are confidence levels realistic?
3. RETURN ANALYSIS: What's the average outcome magnitude?
4. PATTERN RECOGNITION: Which setups work best?
5. IMPROVEMENT OPPORTUNITIES: Where can methodology be enhanced?

Provide actionable insights for improving prediction accuracy and calibration.
"""

####################################################
#           Chatbot Integration                   #
####################################################

PREDICTIVE_CHATBOT_INTEGRATION = """
PREDICTIVE MODE ACTIVATED

You now have access to advanced predictive analysis capabilities. When users ask for predictions or forecasts:

1. Use the PredictiveAnalysisEngine to create structured predictions
2. Explain your methodology and reasoning clearly
3. Always include confidence levels and risk factors
4. Offer to backtest previous predictions if available
5. Suggest ways to improve prediction accuracy over time

Available Commands:
- "predict [TICKER]" - Generate structured prediction
- "backtest [PREDICTION_ID]" - Evaluate past prediction
- "performance" - Show overall prediction metrics
- "learn" - Analyze prediction patterns and improve methodology

Remember: Focus on process improvement and learning from outcomes, not just making predictions.
"""

####################################################
#           Utility Functions                     #
####################################################

def format_prediction_output(prediction) -> str:
    """Format prediction results for display"""
    
    return f"""
🎯 PREDICTION RESULTS FOR {prediction.ticker}
{'='*50}

📈 Direction: {prediction.direction.upper()}
🎲 Confidence: {prediction.confidence:.1%}
💲 Current Price: ${prediction.current_price:.2f}
💰 Target Price: ${prediction.target_price:.2f}
⏰ Horizon: {prediction.prediction_horizon}

🔍 KEY FACTORS:
{chr(10).join(f"  • {factor}" for factor in prediction.key_factors)}

⚠️  RISK FACTORS:
{chr(10).join(f"  • {risk}" for risk in prediction.risk_factors)}

📝 REASONING:
{prediction.reasoning}
"""

def format_backtest_results(prediction, outcome) -> str:
    """Format backtest results for display"""
    
    direction_correct = "✅" if prediction.direction == outcome.actual_direction else "❌"
    target_hit = "✅" if outcome.target_hit else "❌"
    
    return f"""
📊 BACKTEST RESULTS
{'='*50}

{direction_correct} Direction Prediction: {prediction.direction} → {outcome.actual_direction}
{target_hit} Target Hit: ${prediction.target_price:.2f} (Hit: {outcome.target_hit})
📈 Actual Return: {outcome.actual_return:.2f}%
📅 Days to Outcome: {outcome.days_to_outcome}
🎲 Original Confidence: {prediction.confidence:.1%}
"""