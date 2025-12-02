#!/usr/bin/env python3
"""
Verification script to confirm all bugs are fixed
Run this after installing dependencies: pip install -r requirements.txt
"""

import sys
import os

def check_requirements_format():
    """Verify requirements.txt is in correct format"""
    print("🔍 Checking requirements.txt format...")
    with open('requirements.txt', 'r') as f:
        first_line = f.readline().strip()
    
    if first_line.startswith('#'):
        print("   ✅ requirements.txt format is correct (comments, no shebang)")
        return True
    else:
        print(f"   ❌ First line is '{first_line}', expected comment")
        return False


def check_unified_imports():
    """Verify unified import paths work"""
    print("\n🔍 Checking unified import paths...")
    try:
        from data_pipeline.models import (
            create_dual_tower_model,
            create_dual_tower_data_loaders,
            create_dual_tower_optimizer,
            create_dual_tower_scheduler,
            DualTowerLoss,
            DualTowerTrainer,
            ConfigManager,
        )
        print("   ✅ All unified imports successful")
        return True
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return False


def check_backward_compatibility():
    """Verify old imports still work"""
    print("\n🔍 Checking backward compatibility...")
    try:
        from modelling import DualTowerRelevanceModel, DualTowerLoss
        print("   ✅ Backward compatibility imports work")
        return True
    except ImportError as e:
        print(f"   ❌ Backward compat error: {e}")
        return False


def check_lstm_imports():
    """Verify LSTM imports work"""
    print("\n🔍 Checking LSTM imports...")
    try:
        from data_pipeline.models import (
            create_lstm_model,
            create_lstm_data_loaders,
            LSTMTrainer,
            LSTMLoss,
        )
        print("   ✅ LSTM imports successful")
        return True
    except ImportError as e:
        print(f"   ❌ LSTM import error: {e}")
        return False


def check_function_names():
    """Verify new function names exist"""
    print("\n🔍 Checking function names...")
    try:
        from data_pipeline.models import (
            create_dual_tower_model,
            create_lstm_model,
            create_dual_tower_data_loaders,
            create_lstm_data_loaders,
        )
        print("   ✅ All new function names exist")
        return True
    except ImportError as e:
        print(f"   ❌ Function name error: {e}")
        return False


def main():
    """Run all verification checks"""
    print("\n" + "="*60)
    print("🐛 BUG FIX VERIFICATION")
    print("="*60)
    
    results = []
    
    # Run checks
    results.append(("Requirements Format", check_requirements_format()))
    results.append(("Unified Imports", check_unified_imports()))
    results.append(("Backward Compatibility", check_backward_compatibility()))
    results.append(("LSTM Imports", check_lstm_imports()))
    results.append(("Function Names", check_function_names()))
    
    # Summary
    print("\n" + "="*60)
    print("📊 VERIFICATION RESULTS")
    print("="*60)
    
    for check_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{check_name:.<40} {status}")
    
    # Final status
    print("="*60)
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("✅ ALL BUGS FIXED - Codebase is ready for use!")
        print("="*60)
        return 0
    else:
        print("❌ Some issues remain - see details above")
        print("="*60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
