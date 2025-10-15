"""
Quick check of training and testing results
"""
import os
import json
import tensorflow as tf

print("🔍 CHECKING TRAINING AND TESTING RESULTS")
print("=" * 50)

# Check if results file exists
if os.path.exists('positive_strategy_results.json'):
    print("✅ Testing completed - Results file found!")
    
    with open('positive_strategy_results.json', 'r') as f:
        results = json.load(f)
    
    print("\n📊 TESTING RESULTS:")
    print(f"Strategy Return: {results.get('strategy_return', 0):.2f}%")
    print(f"SPY Benchmark:   {results.get('spy_return', 0):.2f}%") 
    print(f"Outperformance:  {results.get('outperformance', 0):.2f}%")
    print(f"Total Reward:    {results.get('total_reward', 0):.2f}")
    print(f"Total Switches:  {results.get('total_switches', 0)}")
    
    if results.get('outperformance', 0) > 0:
        print("🏆 SUCCESS: Strategy BEAT SPY buy-and-hold!")
    else:
        print("⚠️ Strategy underperformed SPY")
        
else:
    print("⏳ Testing still in progress - Results file not yet created")

# Check model file
if os.path.exists('positive_etf_model.h5'):
    model_size = os.path.getsize('positive_etf_model.h5') / 1024
    print(f"\n🤖 Trained model: positive_etf_model.h5 ({model_size:.1f}KB)")
    
    try:
        model = tf.keras.models.load_model('positive_etf_model.h5')
        print("✅ Model loads successfully")
        print(f"Input shape: {model.input_shape}")
        print(f"Output shape: {model.output_shape}")
    except Exception as e:
        print(f"❌ Model loading error: {e}")
else:
    print("❌ No trained model found")

# Check other result files for comparison
other_results = ['improved_strategy_results.json', 'etf_switching_results.json']
for result_file in other_results:
    if os.path.exists(result_file):
        print(f"\n📋 {result_file} exists ({os.path.getsize(result_file)/1024:.1f}KB)")