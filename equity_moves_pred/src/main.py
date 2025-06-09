#!/usr/bin/env python3
"""
Enhanced Test script for the Predictive Analysis Engine
Tests all new features including data quality validation, advanced metrics, and enhanced analysis
"""

import os
import sys
from datetime import datetime, timedelta
from prediction_engine import PredictiveAnalysisEngine, AdvancedMetricsCalculator
import json
import pandas as pd

def print_separator():
    print("\n" + "="*60 + "\n")

def print_outcome_results(prediction, outcome):
    """Pretty print enhanced backtest results"""
    print(f"📈 ENHANCED BACKTEST RESULTS for {prediction.ticker}")
    print(f"   Prediction ID: {outcome.prediction_id}")
    print(f"   Predicted: {prediction.direction.upper()} (confidence: {prediction.confidence:.1%})")
    print(f"   Actual: {outcome.actual_direction.upper()}")
    print(f"   Actual Return: {outcome.actual_return:.2f}%")
    print(f"   Target Hit: {'✅ YES' if outcome.target_hit else '❌ NO'}")
    print(f"   Days Elapsed: {outcome.days_to_outcome}")
    
    # NEW: Enhanced outcome metrics
    print(f"   Max Favorable Move: {outcome.max_favorable_return:.2f}%")
    print(f"   Max Adverse Move: {outcome.max_adverse_return:.2f}%")
    print(f"   Period Volatility: {outcome.volatility_during_period:.1f}%")
    
    # Determine accuracy
    direction_correct = prediction.direction == outcome.actual_direction
    print(f"   Direction Accuracy: {'✅ CORRECT' if direction_correct else '❌ WRONG'}")

def test_data_quality_validation():
    """NEW: Test data quality validation system"""
    print("🔍 TESTING DATA QUALITY VALIDATION")
    
    engine = PredictiveAnalysisEngine()
    
    # Test with different ticker types
    test_tickers = [
        ("AAPL", "High-volume, reliable stock"),
        ("TSLA", "High-volatility stock"),
        ("SPY", "ETF - should be very reliable"),
    ]
    
    # Add a potentially problematic ticker if user wants
    low_volume_ticker = input("Enter a low-volume ticker to test quality validation (or press Enter to skip): ").upper().strip()
    if low_volume_ticker:
        test_tickers.append((low_volume_ticker, "User-provided low-volume test"))
    
    for ticker, description in test_tickers:
        try:
            print(f"\n📊 Testing data quality for {ticker} ({description})")
            
            # Create analysis window to trigger validation
            analysis_input = engine.create_analysis_window(ticker, datetime.now())
            quality_report = analysis_input.data_quality
            
            print(f"   ✅ Validation Result: {'PASS' if quality_report.is_valid else 'FAIL'}")
            print(f"   📈 Data Points: {quality_report.data_points}")
            print(f"   💰 Avg Daily Volume: {quality_report.avg_daily_volume:,.0f}")
            print(f"   📊 Volatility Percentile: {quality_report.volatility_percentile:.1f}")
            print(f"   🚨 Price Gaps (>15%): {quality_report.price_gaps}")
            print(f"   📅 Missing Days: {quality_report.missing_days}")
            
            if quality_report.issues:
                print(f"   ❌ Issues Found:")
                for issue in quality_report.issues:
                    print(f"      • {issue}")
            
            if quality_report.warnings:
                print(f"   ⚠️  Warnings:")
                for warning in quality_report.warnings:
                    print(f"      • {warning}")
            
            print(f"   🎯 Quality Score: {engine._calculate_data_quality_score(quality_report):.3f}/1.000")
            
        except Exception as e:
            print(f"   ❌ Error testing {ticker}: {e}")

def test_enhanced_technical_features():
    """NEW: Test enhanced technical analysis features"""
    print("📊 TESTING ENHANCED TECHNICAL FEATURES")
    
    engine = PredictiveAnalysisEngine()
    ticker = input("Enter ticker for technical analysis test (e.g., NVDA): ").upper().strip()
    if not ticker:
        ticker = "NVDA"
    
    try:
        print(f"\n⏳ Analyzing enhanced technical features for {ticker}...")
        
        # Get analysis input
        analysis_input = engine.create_analysis_window(ticker, datetime.now())
        technical_features = engine.calculate_technical_features(analysis_input.price_data)
        
        print(f"📈 ENHANCED TECHNICAL ANALYSIS for {ticker}:")
        print(f"   Current Price: ${technical_features['current_price']:.2f}")
        print(f"   1W Price Change: {technical_features['price_change_1w_pct']:.2f}%")
        print(f"   1M Price Change: {technical_features['price_change_1m_pct']:.2f}%")
        
        # NEW: Enhanced volatility metrics
        print(f"   Annualized Volatility: {technical_features['annualized_volatility']:.1f}%")
        print(f"   Recent Volatility: {technical_features['recent_volatility']:.1f}%")
        print(f"   Volatility Regime: {technical_features['volatility_regime'].upper()}")
        
        # NEW: Enhanced volume metrics
        print(f"   Volume Trend: {technical_features['volume_trend_pct']:.1f}%")
        print(f"   Volume Spikes (2x+): {technical_features['volume_spikes']} days")
        
        # NEW: Momentum indicators
        print(f"   RSI: {technical_features['rsi']:.1f}")
        print(f"   Momentum Score: {technical_features['momentum_score']:.2f}")
        
        # Support/Resistance
        print(f"   Distance from Recent High: {technical_features['distance_from_high_pct']:.2f}%")
        print(f"   Distance from Recent Low: {technical_features['distance_from_low_pct']:.2f}%")
        
    except Exception as e:
        print(f"❌ Error in technical analysis: {e}")

def test_advanced_metrics_calculator():
    """NEW: Test advanced metrics calculation"""
    print("🧮 TESTING ADVANCED METRICS CALCULATOR")
    
    calc = AdvancedMetricsCalculator()
    
    # Test with sample returns data
    sample_returns = [0.02, -0.01, 0.03, -0.02, 0.01, 0.04, -0.03, 0.02, -0.01, 0.01]
    sample_outcomes = [True, False, True, False, True, True, False, True, False, True]
    
    print("📊 Testing with sample data:")
    print(f"   Sample Returns: {[f'{r:.1%}' for r in sample_returns]}")
    print(f"   Sample Outcomes: {sample_outcomes}")
    
    # Test Sharpe ratio
    sharpe = calc.calculate_sharpe_ratio(sample_returns)
    print(f"   📈 Sharpe Ratio: {sharpe:.3f}")
    
    # Test maximum drawdown
    max_dd = calc.calculate_max_drawdown(sample_returns)
    print(f"   📉 Max Drawdown: {max_dd:.1%}")
    
    # Test win streak stats
    streak_stats = calc.calculate_win_streak_stats(sample_outcomes)
    print(f"   🏆 Max Win Streak: {streak_stats['max_win_streak']}")
    print(f"   📉 Max Loss Streak: {streak_stats['max_loss_streak']}")
    print(f"   🎯 Current Streak: {streak_stats['current_streak']}")
    
    print("\n💡 Testing edge cases:")
    
    # Test with empty data
    empty_sharpe = calc.calculate_sharpe_ratio([])
    print(f"   Empty data Sharpe: {empty_sharpe}")
    
    # Test with zero volatility
    zero_vol_returns = [0.01, 0.01, 0.01, 0.01]
    zero_vol_sharpe = calc.calculate_sharpe_ratio(zero_vol_returns)
    print(f"   Zero volatility Sharpe: {zero_vol_sharpe}")

def test_enhanced_prediction():
    """Test enhanced prediction with data quality scoring"""
    print("🎯 TESTING ENHANCED PREDICTION SYSTEM")
    
    engine = PredictiveAnalysisEngine()
    
    # Get ticker from user
    ticker = input("Enter ticker symbol (e.g., AAPL): ").upper().strip()
    if not ticker:
        ticker = "AAPL"
    
    try:
        print(f"\n⏳ Making enhanced prediction for {ticker}...")
        print("🔍 This will include data quality validation...")
        
        prediction = engine.make_prediction(ticker)
        engine.print_prediction_results(prediction)
        
        # Show the NEW data quality score
        # print(f"\n🎯 ENHANCED FEATURES DEMONSTRATED:")
        # print(f"   ✅ Data Quality Score: {prediction.data_quality_score:.3f}/1.000")
        # print(f"   🔍 Data validation performed automatically")
        # print(f"   📊 Enhanced technical indicators calculated")
        # print(f"   🎯 Confidence calibration ready for analysis")
        
        return engine, prediction
        
    except Exception as e:
        print(f"❌ Error making enhanced prediction: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_enhanced_backtest():
    """Test enhanced backtesting with additional outcome metrics"""
    print("📊 TESTING ENHANCED BACKTEST SYSTEM")
    
    engine = PredictiveAnalysisEngine()
    
    # Use a date from 6 months ago
    analysis_date = datetime.now() - timedelta(days=180)
    ticker = input("Enter ticker for enhanced backtest (e.g., MSFT): ").upper().strip()
    if not ticker:
        ticker = "MSFT"

    prediction_horizon = input("Enter prediction horizon (e.g., 2w, 1w, 10d): ").strip()
    if not prediction_horizon:
        prediction_horizon = "2w"
    
    try:
        print(f"⏳ Making historical prediction for {ticker} on {analysis_date.strftime('%Y-%m-%d')}...")
        print("🔍 Enhanced validation and analysis in progress...")
        
        prediction = engine.make_prediction(ticker, analysis_date, prediction_horizon)
        engine.print_prediction_results(prediction)
        
        print_separator()
        print("⏳ Running enhanced backtest...")
        
        # Backtest 2 weeks later
        outcome_date = analysis_date + timedelta(days=14)
        outcome = engine.backtest_prediction(prediction, outcome_date)
        print_outcome_results(prediction, outcome)
        
        return engine
        
    except Exception as e:
        print(f"❌ Error in enhanced backtest: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_enhanced_performance_metrics():
    """NEW: Test comprehensive enhanced performance metrics"""
    print("📈 TESTING ENHANCED PERFORMANCE METRICS")
    
    engine = PredictiveAnalysisEngine()
    tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
    analysis_date = datetime.now() - timedelta(days=21)
    
    print(f"⏳ Creating test dataset with {len(tickers)} predictions...")
    print("🔍 Each prediction includes enhanced data quality validation...")
    
    successful_predictions = 0
    for ticker in tickers:
        try:
            print(f"   Processing {ticker}...")
            prediction = engine.make_prediction(ticker, analysis_date)
            
            # Backtest it with enhanced metrics
            outcome_date = analysis_date + timedelta(days=14)
            outcome = engine.backtest_prediction(prediction, outcome_date)
            successful_predictions += 1
            
        except Exception as e:
            print(f"   ❌ Failed for {ticker}: {e}")
    
    print(f"\n✅ Completed {successful_predictions}/{len(tickers)} predictions")
    
    if successful_predictions > 0:
        print_separator()
        print("📊 ENHANCED PERFORMANCE METRICS")
        metrics = engine.get_performance_metrics()
        
        # Group metrics by category
        basic_metrics = [
            'total_predictions', 'direction_accuracy', 'target_hit_rate', 
            'average_confidence', 'average_data_quality', 'average_actual_return'
        ]
        
        advanced_metrics = [
            'sharpe_ratio', 'max_drawdown', 'max_win_streak', 
            'max_loss_streak', 'current_streak'
        ]
        
        calibration_metrics = [
            'confidence_calibration_error', 'overconfidence_bias'
        ]
        
        market_metrics = [
            'avg_market_volatility'
        ]
        
        print("🎯 BASIC PERFORMANCE:")
        for key in basic_metrics:
            if key in metrics:
                value = metrics[key]
                if isinstance(value, float) and ('accuracy' in key or 'rate' in key or 'quality' in key or 'confidence' in key):
                    print(f"   {key.replace('_', ' ').title()}: {value:.1%}")
                elif isinstance(value, float):
                    print(f"   {key.replace('_', ' ').title()}: {value:.2f}")
                else:
                    print(f"   {key.replace('_', ' ').title()}: {value}")
        
        print("\n🚀 ADVANCED RISK METRICS:")
        for key in advanced_metrics:
            if key in metrics:
                value = metrics[key]
                if 'drawdown' in key:
                    print(f"   {key.replace('_', ' ').title()}: {value:.2f}%")
                elif 'streak' in key:
                    print(f"   {key.replace('_', ' ').title()}: {value}")
                else:
                    print(f"   {key.replace('_', ' ').title()}: {value:.3f}")
        
        print("\n🎯 CALIBRATION ANALYSIS:")
        for key in calibration_metrics:
            if key in metrics:
                value = metrics[key]
                print(f"   {key.replace('_', ' ').title()}: {value:.1%}")
        
        print("\n📊 MARKET CONTEXT:")
        for key in market_metrics:
            if key in metrics:
                value = metrics[key]
                print(f"   {key.replace('_', ' ').title()}: {value:.1f}%")
        
        # NEW: Interpretation guide
        print("\n💡 METRIC INTERPRETATION:")
        if 'sharpe_ratio' in metrics:
            sharpe = metrics['sharpe_ratio']
            if sharpe > 1.0:
                sharpe_desc = "Excellent"
            elif sharpe > 0.5:
                sharpe_desc = "Good"
            elif sharpe > 0:
                sharpe_desc = "Acceptable"
            else:
                sharpe_desc = "Poor"
            print(f"   Sharpe Ratio ({sharpe:.3f}): {sharpe_desc} risk-adjusted performance")
        
        if 'confidence_calibration_error' in metrics:
            cal_error = metrics['confidence_calibration_error']
            if cal_error < 0.1:
                cal_desc = "Well-calibrated"
            elif cal_error < 0.2:
                cal_desc = "Moderately calibrated"
            else:
                cal_desc = "Poorly calibrated"
            print(f"   Calibration Error ({cal_error:.1%}): {cal_desc} confidence estimates")
    
    return engine

def enhanced_interactive_mode():
    """Enhanced interactive testing mode with new features"""
    print("🚀 ENHANCED INTERACTIVE PREDICTION MODE")
    print("Commands:")
    print("  predict <TICKER> - Make an enhanced prediction")
    print("  backtest <TICKER> - Make and backtest with enhanced metrics")
    print("  quality <TICKER> - Test data quality validation only")
    print("  technical <TICKER> - Show enhanced technical analysis only")
    print("  performance - Show enhanced performance metrics")
    print("  demo - Run the enhanced demo")
    print("  quit - Exit")
    
    engine = PredictiveAnalysisEngine()
    
    while True:
        try:
            command = input("\n> ").strip().lower()
            
            if command in ["quit", "exit"]:
                break
            
            elif command == "demo":
                print("\n🎯 Running Enhanced System Demo...")
                print_separator()
                
                # Quick demo of all enhanced features
                test_data_quality_validation()
                print_separator()
                test_enhanced_technical_features()
                print_separator()
                test_advanced_metrics_calculator()
                
            elif command == "performance":
                metrics = engine.get_performance_metrics()
                if "message" in metrics:
                    print(f"\n📊 {metrics['message']}")
                else:
                    print("\n📊 ENHANCED PERFORMANCE METRICS:")
                    for key, value in metrics.items():
                        if isinstance(value, float):
                            if 'accuracy' in key or 'rate' in key:
                                print(f"   {key.replace('_', ' ').title()}: {value:.1%}")
                            else:
                                print(f"   {key.replace('_', ' ').title()}: {value:.3f}")
                        else:
                            print(f"   {key.replace('_', ' ').title()}: {value}")
                    
            elif command.startswith("predict "):
                ticker = command.split()[1].upper()
                print(f"⏳ Making enhanced prediction for {ticker}...")
                prediction = engine.make_prediction(ticker)
                print_separator()
                engine.print_prediction_results(prediction)
                
            elif command.startswith("quality "):
                ticker = command.split()[1].upper()
                print(f"🔍 Testing data quality for {ticker}...")
                try:
                    analysis_input = engine.create_analysis_window(ticker, datetime.now())
                    quality_report = analysis_input.data_quality
                    quality_score = engine._calculate_data_quality_score(quality_report)
                    
                    print(f"\n📊 DATA QUALITY REPORT for {ticker}:")
                    print(f"   Overall Score: {quality_score:.3f}/1.000")
                    print(f"   Status: {'✅ PASS' if quality_report.is_valid else '❌ FAIL'}")
                    print(f"   Data Points: {quality_report.data_points}")
                    print(f"   Avg Volume: {quality_report.avg_daily_volume:,.0f}")
                    
                    if quality_report.issues:
                        print("   ❌ Issues:")
                        for issue in quality_report.issues:
                            print(f"      • {issue}")
                    
                    if quality_report.warnings:
                        print("   ⚠️  Warnings:")
                        for warning in quality_report.warnings:
                            print(f"      • {warning}")
                            
                except Exception as e:
                    print(f"❌ Error: {e}")
                
            elif command.startswith("technical "):
                ticker = command.split()[1].upper()
                print(f"📊 Enhanced technical analysis for {ticker}...")
                try:
                    analysis_input = engine.create_analysis_window(ticker, datetime.now())
                    features = engine.calculate_technical_features(analysis_input.price_data)
                    
                    print(f"\n📈 ENHANCED TECHNICAL FEATURES:")
                    for key, value in features.items():
                        if isinstance(value, (int, float)):
                            if 'pct' in key:
                                print(f"   {key.replace('_', ' ').title()}: {value:.2f}%")
                            elif 'price' in key:
                                print(f"   {key.replace('_', ' ').title()}: ${value:.2f}")
                            else:
                                print(f"   {key.replace('_', ' ').title()}: {value:.2f}")
                        else:
                            print(f"   {key.replace('_', ' ').title()}: {value}")
                            
                except Exception as e:
                    print(f"❌ Error: {e}")
                
            elif command.startswith("backtest "):
                ticker = command.split()[1].upper()
                analysis_date = datetime.now() - timedelta(days=21)
                print(f"⏳ Enhanced backtest for {ticker}...")
                
                try:
                    prediction = engine.make_prediction(ticker, analysis_date)
                    outcome_date = analysis_date + timedelta(days=14)
                    outcome = engine.backtest_prediction(prediction, outcome_date)
                    
                    print_separator()
                    engine.print_prediction_results(prediction)
                    print_separator()
                    print_outcome_results(prediction, outcome)
                    
                except Exception as e:
                    print(f"❌ Error: {e}")
                
            else:
                print("❌ Unknown command. Try 'predict AAPL', 'quality TSLA', or 'demo'")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Enhanced main testing function"""
    print("🎯 ENHANCED PREDICTIVE ANALYSIS ENGINE - TEST SUITE v2.0")
    print("="*60)
    print("NEW FEATURES: Data Quality Validation, Advanced Metrics, Enhanced Analysis")
    
    print("\nSelect test mode:")
    print("1. Enhanced single prediction test")
    print("2. Enhanced historical backtest test") 
    print("3. Enhanced performance metrics test")
    print("4. NEW: Data quality validation test")
    print("5. NEW: Advanced metrics calculator test")
    print("6. NEW: Enhanced technical features test")
    print("7. Enhanced interactive mode")
    print("8. Run all enhanced tests")
    
    choice = input("\nEnter choice (1-8): ").strip()
    
    print_separator()
    
    if choice == "1":
        test_enhanced_prediction()
        
    elif choice == "2":
        test_enhanced_backtest()
        
    elif choice == "3":
        test_enhanced_performance_metrics()
        
    elif choice == "4":
        test_data_quality_validation()
        
    elif choice == "5":
        test_advanced_metrics_calculator()
        
    elif choice == "6":
        test_enhanced_technical_features()
        
    elif choice == "7":
        enhanced_interactive_mode()
        
    elif choice == "8":
        print("🧪 RUNNING ALL ENHANCED TESTS")
        print_separator()
        
        test_enhanced_prediction()
        print_separator()
        
        test_enhanced_backtest()
        print_separator()
        
        test_data_quality_validation()
        print_separator()
        
        test_advanced_metrics_calculator()
        print_separator()
        
        test_enhanced_technical_features()
        print_separator()
        
        test_enhanced_performance_metrics()
        
    else:
        print("❌ Invalid choice")
        return
    
    print_separator()
    # print("✅ Enhanced testing complete!")
    # print("💡 New features tested:")
    # print("   • Data quality validation with scoring")
    # print("   • Enhanced technical indicators (RSI, volume spikes, volatility regimes)")
    # print("   • Advanced performance metrics (Sharpe ratio, max drawdown, streaks)")
    # print("   • Confidence calibration analysis")
    # print("   • Enhanced backtest outcomes (max favorable/adverse moves, period volatility)")

if __name__ == "__main__":
    main()