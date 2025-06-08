#!/usr/bin/env python3
"""
Test script for the Predictive Analysis Engine
Run this to test predictions and backtesting functionality
"""

import os
import sys
from datetime import datetime, timedelta
from prediction_engine import PredictiveAnalysisEngine
import json

def print_separator():
    print("\n" + "="*60 + "\n")

# def print_prediction_results(prediction):
#     """Pretty print prediction results"""
#     print(f"🎯 PREDICTION RESULTS for {prediction.ticker}")
#     print(f"   Analysis Date: {prediction.analysis_date.strftime('%Y-%m-%d')}")
#     print(f"   Direction: {prediction.direction.upper()}")
#     print(f"   Confidence: {prediction.confidence:.1%}")
#     print(f"   Target Price: ${prediction.target_price:.2f}" if prediction.target_price else "   Target Price: Not specified")
#     print(f"   Horizon: {prediction.prediction_horizon}")
    
#     print(f"\n📊 KEY FACTORS:")
#     for factor in prediction.key_factors:
#         print(f"   • {factor}")
    
#     print(f"\n!!! RISK FACTORS:")
#     for risk in prediction.risk_factors:
#         print(f"   • {risk}")
    
#     print(f"\n🧠 REASONING:")
#     print(f"   {prediction.reasoning}")

def print_outcome_results(prediction, outcome):
    """Pretty print backtest results"""
    print(f"📈 BACKTEST RESULTS for {prediction.ticker}")
    print(f"   Prediction ID: {outcome.prediction_id}")
    print(f"   Predicted: {prediction.direction.upper()} (confidence: {prediction.confidence:.1%})")
    print(f"   Actual: {outcome.actual_direction.upper()}")
    print(f"   Actual Return: {outcome.actual_return:.2f}%")
    print(f"   Target Hit: {'✅ YES' if outcome.target_hit else '❌ NO'}")
    print(f"   Days Elapsed: {outcome.days_to_outcome}")
    
    # Determine accuracy
    direction_correct = prediction.direction == outcome.actual_direction
    print(f"   Direction Accuracy: {'✅ CORRECT' if direction_correct else '❌ WRONG'}")

def test_single_prediction():
    """Test making a single prediction"""
    print("🔮 TESTING SINGLE PREDICTION")
    
    engine = PredictiveAnalysisEngine()
    
    # Get ticker from user
    ticker = input("Enter ticker symbol (e.g., AAPL): ").upper().strip()
    if not ticker:
        ticker = "AAPL"  # Default
    
    try:
        print(f"\n⏳ Making prediction for {ticker}...")
        prediction = engine.make_prediction(ticker)
        engine.print_prediction_results(prediction)
        
        return engine, prediction
        
    except Exception as e:
        print(f"❌ Error making prediction: {e}")
        return None, None

def test_historical_backtest():
    """Test backtesting with historical data"""
    print("📊 TESTING HISTORICAL BACKTEST")
    
    engine = PredictiveAnalysisEngine()
    
    # Use a date from 3 weeks ago so we can backtest
    analysis_date = datetime.now() - timedelta(days=21)
    ticker = input("Enter ticker for historical test (e.g., TSLA): ").upper().strip()
    if not ticker:
        ticker = "TSLA"
    
    try:
        print(f"⏳ Making historical prediction for {ticker} on {analysis_date.strftime('%Y-%m-%d')}...")
        prediction = engine.make_prediction(ticker, analysis_date)
        engine.print_prediction_results(prediction)
        
        print_separator()
        print("⏳ Running backtest...")
        
        # Backtest 2 weeks later
        outcome_date = analysis_date + timedelta(days=14)
        outcome = engine.backtest_prediction(prediction, outcome_date)
        print_outcome_results(prediction, outcome)
        
        return engine
        
    except Exception as e:
        print(f"❌ Error in historical backtest: {e}")
        return None

def test_multiple_predictions():
    """Test multiple predictions and performance metrics"""
    print("📈 TESTING MULTIPLE PREDICTIONS & PERFORMANCE")
    
    engine = PredictiveAnalysisEngine()
    tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
    analysis_date = datetime.now() - timedelta(days=21)  # 3 weeks ago
    
    print(f"⏳ Making predictions for {len(tickers)} stocks...")
    
    successful_predictions = 0
    for ticker in tickers:
        try:
            print(f"   Processing {ticker}...")
            prediction = engine.make_prediction(ticker, analysis_date)
            
            # Backtest it
            outcome_date = analysis_date + timedelta(days=14)
            outcome = engine.backtest_prediction(prediction, outcome_date)
            successful_predictions += 1
            
        except Exception as e:
            print(f"   ❌ Failed for {ticker}: {e}")
    
    print(f"\n✅ Completed {successful_predictions}/{len(tickers)} predictions")
    
    if successful_predictions > 0:
        print_separator()
        print("📊 PERFORMANCE METRICS")
        metrics = engine.get_performance_metrics()
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                if 'accuracy' in key or 'rate' in key:
                    print(f"   {key.replace('_', ' ').title()}: {value:.1%}")
                else:
                    print(f"   {key.replace('_', ' ').title()}: {value}")
            else:
                print(f"   {key.replace('_', ' ').title()}: {value}")
    
    return engine

def interactive_mode():
    """Interactive testing mode"""
    print("🚀 INTERACTIVE PREDICTION MODE")
    print("Commands:")
    print("  predict <TICKER> - Make a prediction")
    print("  backtest <TICKER> - Make and backtest a historical prediction")  
    print("  performance - Show performance metrics")
    print("  quit - Exit")
    
    engine = PredictiveAnalysisEngine()
    
    while True:
        try:
            command = input("\n> ").strip().lower()
            
            if command == "quit" or command == "exit":
                break
                
            elif command == "performance":
                metrics = engine.get_performance_metrics()
                print("\n📊 PERFORMANCE METRICS:")
                for key, value in metrics.items():
                    print(f"   {key}: {value}")
                    
            elif command.startswith("predict "):
                ticker = command.split()[1].upper()
                print(f"⏳ Making prediction for {ticker}...")
                prediction = engine.make_prediction(ticker)
                print_separator()
                engine.print_prediction_results(prediction)
                
            elif command.startswith("backtest "):
                ticker = command.split()[1].upper()
                analysis_date = datetime.now() - timedelta(days=21)
                print(f"⏳ Making historical prediction for {ticker}...")
                
                prediction = engine.make_prediction(ticker, analysis_date)
                outcome_date = analysis_date + timedelta(days=14)
                outcome = engine.backtest_prediction(prediction, outcome_date)
                
                print_separator()
                engine.print_prediction_results(prediction)
                print_separator()
                print_outcome_results(prediction, outcome)
                
            else:
                print("❌ Unknown command. Try 'predict AAPL' or 'performance'")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Main testing function"""
    print("🎯 PREDICTIVE ANALYSIS ENGINE - TEST SUITE")
    print("="*60)
    
    print("\nSelect test mode:")
    print("1. Single prediction test")
    print("2. Historical backtest test") 
    print("3. Multiple predictions test")
    print("4. Interactive mode")
    print("5. Run all tests")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    print_separator()
    
    if choice == "1":
        test_single_prediction()
        
    elif choice == "2":
        test_historical_backtest()
        
    elif choice == "3":
        test_multiple_predictions()
        
    elif choice == "4":
        interactive_mode()
        
    elif choice == "5":
        print("🧪 RUNNING ALL TESTS")
        print_separator()
        
        test_single_prediction()
        print_separator()
        
        test_historical_backtest()
        print_separator()
        
        test_multiple_predictions()
        
    else:
        print("❌ Invalid choice")
        return
    
    print_separator()
    print("✅ Testing complete!")

if __name__ == "__main__":
    main()