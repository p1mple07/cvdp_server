"""
Test script for Nemotron model to verify:
1. Model loads correctly
2. Basic generation works
3. API endpoint functionality
"""

import requests
import json
import time

# API endpoint
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test if the API is running and Nemotron model is available"""
    print("=" * 60)
    print("Test 1: Health Check")
    print("=" * 60)
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✓ API Status: {data['status']}")
            print(f"✓ Available Models: {data['available_models']}")
            
            if 'nemotron' in data['available_models']:
                print("✓ Nemotron model is loaded and available!")
                return True
            else:
                print("✗ Nemotron model is NOT available")
                print(f"Available models: {data['available_models']}")
                return False
        else:
            print(f"✗ Health check failed with status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Error connecting to API: {e}")
        print("\n⚠ Make sure the server is running:")
        print("  uvicorn main:app --host 0.0.0.0 --port 8000")
        return False

def test_model_info():
    """Get detailed model information"""
    print("\n" + "=" * 60)
    print("Test 2: Model Information")
    print("=" * 60)
    
    try:
        response = requests.get(f"{BASE_URL}/model_info")
        if response.status_code == 200:
            data = response.json()
            
            if 'nemotron' in data:
                nemotron_info = data['nemotron']
                print(f"✓ Model Name: {nemotron_info.get('model_name', 'N/A')}")
                print(f"✓ Model Type: {nemotron_info.get('model_type', 'N/A')}")
                print(f"✓ Device: {nemotron_info.get('device', 'N/A')}")
                print(f"✓ Status: {nemotron_info.get('status', 'N/A')}")
                return True
            else:
                print("✗ Nemotron model info not found")
                return False
        else:
            print(f"✗ Failed to get model info: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_basic_generation():
    """Test basic text generation"""
    print("\n" + "=" * 60)
    print("Test 3: Basic Text Generation")
    print("=" * 60)
    
    prompt = "Write a simple Python function to add two numbers."
    
    payload = {
        "prompt": prompt,
        "model": "nemotron",
        "max_tokens": 200,
        "temperature": 0.7
    }
    
    print(f"Prompt: {prompt}")
    print("\nGenerating response...")
    
    try:
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/generate", json=payload, timeout=60)
        end_time = time.time()
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n✓ Generation successful!")
            print(f"✓ Model used: {data.get('model', 'N/A')}")
            print(f"✓ Generation time: {data.get('generation_time', end_time - start_time):.2f}s")
            print(f"✓ Tokens generated: {data.get('tokens_generated', 'N/A')}")
            print(f"\nResponse:\n{'-' * 60}")
            print(data.get('response', 'No response'))
            print('-' * 60)
            return True
        else:
            print(f"✗ Generation failed with status code: {response.status_code}")
            print(f"Error: {response.text}")
            return False
    except requests.Timeout:
        print("✗ Request timed out (>60s). Model might be too slow or stuck.")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_code_generation():
    """Test code generation capability"""
    print("\n" + "=" * 60)
    print("Test 4: Code Generation (Nemotron Specialty)")
    print("=" * 60)
    
    prompt = "Write a Python function that implements binary search on a sorted array."
    
    payload = {
        "prompt": prompt,
        "model": "nemotron",
        "max_tokens": 300,
        "temperature": 0.5
    }
    
    print(f"Prompt: {prompt}")
    print("\nGenerating code...")
    
    try:
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/generate", json=payload, timeout=60)
        end_time = time.time()
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n✓ Code generation successful!")
            print(f"✓ Generation time: {end_time - start_time:.2f}s")
            print(f"\nGenerated Code:\n{'-' * 60}")
            print(data.get('response', 'No response'))
            print('-' * 60)
            return True
        else:
            print(f"✗ Code generation failed with status code: {response.status_code}")
            print(f"Error: {response.text}")
            return False
    except requests.Timeout:
        print("✗ Request timed out (>60s).")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_technical_question():
    """Test technical Q&A capability"""
    print("\n" + "=" * 60)
    print("Test 5: Technical Question Answering")
    print("=" * 60)
    
    prompt = "Explain what a barrel shifter is in hardware design."
    
    payload = {
        "prompt": prompt,
        "model": "nemotron",
        "max_tokens": 250,
        "temperature": 0.6
    }
    
    print(f"Prompt: {prompt}")
    print("\nGenerating answer...")
    
    try:
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/generate", json=payload, timeout=60)
        end_time = time.time()
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n✓ Answer generated successfully!")
            print(f"✓ Generation time: {end_time - start_time:.2f}s")
            print(f"\nAnswer:\n{'-' * 60}")
            print(data.get('response', 'No response'))
            print('-' * 60)
            return True
        else:
            print(f"✗ Failed with status code: {response.status_code}")
            print(f"Error: {response.text}")
            return False
    except requests.Timeout:
        print("✗ Request timed out (>60s).")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("NEMOTRON MODEL TEST SUITE")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Health check
    results['health_check'] = test_health_check()
    
    if not results['health_check']:
        print("\n⚠ API not responding or Nemotron model not loaded!")
        print("Make sure to start the server first with:")
        print("  cd /workspace/ckarfa/siang_btp/slm_api")
        print("  source flask_venv/bin/activate")
        print("  uvicorn main:app --host 0.0.0.0 --port 8000")
        return
    
    # Test 2: Model info
    results['model_info'] = test_model_info()
    
    # Test 3: Basic generation
    results['basic_generation'] = test_basic_generation()
    
    # Test 4: Code generation
    results['code_generation'] = test_code_generation()
    
    # Test 5: Technical Q&A
    results['technical_qa'] = test_technical_question()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Nemotron model is working correctly!")
    elif passed > 0:
        print("\n⚠ Some tests passed, but there are issues.")
    else:
        print("\n✗ All tests failed. Check server logs for errors.")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
