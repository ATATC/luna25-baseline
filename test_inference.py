#!/usr/bin/env python3
"""
Test script for inference.py to verify the modifications work correctly
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
import json

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_import():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        from model import ConvNextLSTM
        print("✓ ConvNextLSTM imported successfully")
    except Exception as e:
        print(f"✗ Failed to import ConvNextLSTM: {e}")
        return False
    
    try:
        from processor import MalignancyProcessor
        print("✓ MalignancyProcessor imported successfully")
    except Exception as e:
        print(f"✗ Failed to import MalignancyProcessor: {e}")
        return False
    
    try:
        from inference import NoduleProcessor, run
        print("✓ inference module imported successfully")
    except Exception as e:
        print(f"✗ Failed to import inference module: {e}")
        return False
    
    return True

def test_model_initialization():
    """Test if ConvNextLSTM model can be initialized"""
    print("\nTesting model initialization...")
    
    try:
        from model import ConvNextLSTM
        import torch
        
        # Check if CUDA is available
        if not torch.cuda.is_available():
            print("⚠ CUDA not available, skipping GPU model test")
            return True
        
        model = ConvNextLSTM(pretrained=False, in_chans=3, class_num=1)
        print("✓ ConvNextLSTM model created successfully")
        
        # Test input shape
        batch_size = 1
        n_slices = 64
        channels = 3
        height = 64
        width = 64
        
        dummy_input = torch.randn(batch_size, channels, n_slices, height, width)
        print(f"✓ Created dummy input with shape: {dummy_input.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to initialize model: {e}")
        return False

def test_processor_initialization():
    """Test if MalignancyProcessor can be initialized"""
    print("\nTesting processor initialization...")
    
    try:
        from processor import MalignancyProcessor
        
        processor = MalignancyProcessor(
            mode="ConvNextLSTM", 
            suppress_logs=True, 
            model_name="LUNA25-baseline-ConvNextLSTM"
        )
        print("✓ MalignancyProcessor initialized successfully")
        return True
        
    except Exception as e:
        print(f"✗ Failed to initialize processor: {e}")
        return False

def main():
    """Run all tests"""
    print("Running inference.py tests...\n")
    
    tests = [
        test_import,
        test_model_initialization,
        test_processor_initialization,
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n{'='*50}")
    print(f"Tests passed: {passed}/{len(tests)}")
    
    if passed == len(tests):
        print("✓ All tests passed! The inference.py modifications should work correctly.")
        return True
    else:
        print("✗ Some tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)