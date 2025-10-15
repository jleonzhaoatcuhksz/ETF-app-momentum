"""
Main execution script for ETF Switching ML Strategy
"""

import sys
import os
import subprocess

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def check_data_availability():
    """Check if database and data are available"""
    if not os.path.exists('./etf_data.db'):
        print("❌ Database file 'etf_data.db' not found!")
        print("Please ensure the ETF data has been fetched and the database exists.")
        return False
    
    print("✅ Database file found!")
    return True

def run_training():
    """Run the ML model training"""
    print("\n" + "="*50)
    print("STARTING ETF SWITCHING STRATEGY TRAINING")
    print("="*50)
    
    try:
        # Import and run the ML model
        from ml_model import train_etf_switching_model, test_strategy
        
        print("Training Deep Q-Network for ETF switching...")
        agent, env = train_etf_switching_model(episodes=500)
        
        print("Testing trained strategy...")
        actions, values = test_strategy(agent, env)
        
        print("✅ Training and testing completed successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure all required packages are installed.")
        return False
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

def run_analysis():
    """Run the results analysis"""
    print("\n" + "="*50)
    print("ANALYZING STRATEGY RESULTS")
    print("="*50)
    
    try:
        from analyze_results import generate_report
        generate_report()
        print("✅ Analysis completed successfully!")
        return True
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return False

def main():
    """Main execution function"""
    print("ETF Switching Strategy with Machine Learning")
    print("=" * 60)
    print("This system uses Deep Q-Network (DQN) to learn optimal")
    print("ETF switching strategies based on Monthly_Trend data.")
    print("=" * 60)
    
    # Step 1: Check data availability
    if not check_data_availability():
        return False
    
    # Step 2: Install requirements
    print("\nStep 1: Installing requirements...")
    if not install_requirements():
        print("Please install the requirements manually:")
        print("pip install numpy pandas tensorflow scikit-learn matplotlib seaborn")
        return False
    
    # Step 3: Run training
    print("\nStep 2: Training ML model...")
    if not run_training():
        return False
    
    # Step 4: Run analysis
    print("\nStep 3: Analyzing results...")
    if not run_analysis():
        return False
    
    print("\n" + "="*60)
    print("🎉 ETF SWITCHING STRATEGY COMPLETE!")
    print("="*60)
    print("Files generated:")
    print("  - etf_switching_model.h5 (Trained ML model)")
    print("  - etf_switching_results.json (Trading results)")
    print("  - etf_switching_analysis.png (Performance charts)")
    print("\nThe ML model has learned to:")
    print("  ✅ Analyze Monthly_Trend data for all 14 ETFs")
    print("  ✅ Decide when to switch vs hold current position")
    print("  ✅ Select the most profitable ETF to switch to")
    print("  ✅ Minimize transaction costs while maximizing returns")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Execution failed. Please check the error messages above.")
        sys.exit(1)
    else:
        print("\n✅ All steps completed successfully!")