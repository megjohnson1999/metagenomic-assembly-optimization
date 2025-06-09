#!/usr/bin/env python3
"""
Run comprehensive tests for the metagenomic assembly optimization toolkit.

This script runs both unit tests and integration tests, then generates
a comprehensive report.
"""

import sys
import time
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_quick_validation():
    """Run a quick validation to ensure the toolkit is working."""
    logger.info("Running quick validation test...")
    
    try:
        # Import and run the quick test we created earlier
        import quick_test_assembly
        quick_test_assembly.main()
        return True
    except Exception as e:
        logger.error(f"Quick validation failed: {e}")
        return False

def run_unit_tests():
    """Run unit tests."""
    logger.info("Running unit tests...")
    
    try:
        from test_unit import run_unit_tests
        return run_unit_tests()
    except Exception as e:
        logger.error(f"Unit tests failed: {e}")
        return False

def run_integration_tests():
    """Run integration tests with real datasets."""
    logger.info("Running integration tests...")
    
    try:
        from testing import AssemblyOptimizationTester
        
        # Run quick test (mock community only)
        tester = AssemblyOptimizationTester(output_dir="test_results")
        
        # Create mock community dataset
        mock_dataset = tester.downloader.create_mock_community()
        
        # Run tests
        results = []
        results.append(tester.run_grouping_test(mock_dataset))
        results.append(tester.run_bias_test(mock_dataset))
        results.append(tester.run_performance_test(mock_dataset))
        
        tester.test_results = results
        tester.generate_report()
        
        # Check if all tests passed
        all_passed = all(result.success for result in results)
        
        if all_passed:
            logger.info("✅ All integration tests passed")
        else:
            logger.warning("⚠️  Some integration tests failed")
        
        return all_passed
        
    except Exception as e:
        logger.error(f"Integration tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main testing function."""
    logger.info("Starting comprehensive testing of assembly optimization toolkit...")
    
    start_time = time.time()
    results = {
        'quick_validation': False,
        'unit_tests': False,
        'integration_tests': False
    }
    
    # Run tests in order
    print("="*60)
    print("ASSEMBLY OPTIMIZATION TOOLKIT - COMPREHENSIVE TESTING")
    print("="*60)
    
    # 1. Quick validation
    print("\n1. Quick Validation Test")
    print("-" * 30)
    results['quick_validation'] = run_quick_validation()
    
    # 2. Unit tests
    print("\n2. Unit Tests")
    print("-" * 30)
    results['unit_tests'] = run_unit_tests()
    
    # 3. Integration tests
    print("\n3. Integration Tests")
    print("-" * 30)
    results['integration_tests'] = run_integration_tests()
    
    # Summary
    total_time = time.time() - start_time
    
    print("\n" + "="*60)
    print("TESTING SUMMARY")
    print("="*60)
    
    for test_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    
    print(f"\nOverall: {total_passed}/{total_tests} test suites passed")
    print(f"Total runtime: {total_time:.2f} seconds")
    
    if total_passed == total_tests:
        print("\n🎉 ALL TESTS PASSED - Toolkit is ready for use!")
        
        print("\nGenerated files:")
        test_results_dir = Path("test_results")
        if test_results_dir.exists():
            for file in test_results_dir.glob("*"):
                print(f"  - {file}")
        
        print("\nNext steps:")
        print("  1. Review the testing report: test_results/testing_report.html")
        print("  2. Use the toolkit on your real gut microbiome data")
        print("  3. Refer to the examples in quick_test_assembly.py")
        
        return True
    else:
        print(f"\n⚠️  {total_tests - total_passed} test suite(s) failed")
        print("Please review the error messages above")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)