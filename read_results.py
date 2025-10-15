import json

print("📊 TESTING RESULTS ANALYSIS")
print("=" * 40)

# Read improved strategy results
try:
    with open('improved_strategy_results.json', 'r') as f:
        improved_results = json.load(f)
    
    print("\n✅ IMPROVED STRATEGY RESULTS:")
    print(f"Strategy Return: {improved_results.get('strategy_return', 0):.2f}%")
    print(f"SPY Return: {improved_results.get('spy_return', 0):.2f}%") 
    print(f"Outperformance: {improved_results.get('outperformance', 0):.2f}%")
    print(f"Total Switches: {improved_results.get('total_switches', 0)}")
    
except Exception as e:
    print(f"Error reading improved results: {e}")

# Read original strategy results  
try:
    with open('etf_switching_results.json', 'r') as f:
        original_results = json.load(f)
    
    print("\n📊 ORIGINAL STRATEGY RESULTS:")
    print(f"Strategy Return: {original_results.get('strategy_return', 0):.2f}%")
    print(f"SPY Return: {original_results.get('spy_return', 0):.2f}%")
    print(f"Outperformance: {original_results.get('outperformance', 0):.2f}%") 
    print(f"Total Switches: {original_results.get('total_switches', 0)}")
    
except Exception as e:
    print(f"Error reading original results: {e}")

# Check positive strategy status
import os
if os.path.exists('positive_strategy_results.json'):
    print("\n🎯 POSITIVE STRATEGY RESULTS:")
    try:
        with open('positive_strategy_results.json', 'r') as f:
            positive_results = json.load(f)
        
        print(f"Strategy Return: {positive_results.get('strategy_return', 0):.2f}%")
        print(f"SPY Return: {positive_results.get('spy_return', 0):.2f}%")
        print(f"Outperformance: {positive_results.get('outperformance', 0):.2f}%")
        print(f"Total Reward: {positive_results.get('total_reward', 0):.2f}")
        print(f"Total Switches: {positive_results.get('total_switches', 0)}")
        
        if positive_results.get('outperformance', 0) > 0:
            print("🏆 SUCCESS: Positive strategy BEAT SPY!")
        else:
            print("⚠️ Positive strategy underperformed SPY")
            
    except Exception as e:
        print(f"Error reading positive results: {e}")
else:
    print("\n⏳ POSITIVE STRATEGY: Still training/testing...")
    print("Results file not yet created")