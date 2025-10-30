#!/usr/bin/env python3

"""
Test script for AI21 Jamba Reasoning 3B model integration
"""
import requests
import json

def test_jamba_api():
    """Test Jamba model via API"""
    base_url = "http://localhost:8000"
    
    # Test 1: Basic generation
    print("🧪 Testing Jamba Basic Generation...")
    try:
        response = requests.post(f"{base_url}/generate", json={
            "prompt": "What is machine learning?",
            "model": "jamba",
            "max_length": 500,
            "temperature": 0.7
        })
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success! Generated {result.get('tokens_generated', 0)} tokens")
            print(f"📝 Response: {result.get('response', '')[:300]}...")
            print(f"⏱️  Generation time: {result.get('generation_time', 0):.2f}s")
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Connection error: {e}")
    
    print("\n" + "="*60 + "\n")
    
    # Test 2: Reasoning task (Jamba's specialty)
    print("🧪 Testing Jamba Reasoning...")
    try:
        response = requests.post(f"{base_url}/generate", json={
            "prompt": "You are analyzing customer support tickets to decide which need escalation.\nTicket 1: 'App crashes when uploading files >50MB.'\nTicket 2: 'Forgot password, can't log in.'\nTicket 3: 'Billing page missing enterprise pricing.'\nClassify each ticket as Critical, Medium, or Low and explain your reasoning.",
            "model": "jamba",
            "max_length": 1000,
            "temperature": 0.6
        })
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success! Generated {result.get('tokens_generated', 0)} tokens")
            print(f"📝 Response: {result.get('response', '')}")
            print(f"⏱️  Generation time: {result.get('generation_time', 0):.2f}s")
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Connection error: {e}")
    
    print("\n" + "="*60 + "\n")
    
    # Test 3: Technical explanation
    print("🧪 Testing Jamba Technical Explanation...")
    try:
        response = requests.post(f"{base_url}/generate", json={
            "prompt": "Explain how a Linear Feedback Shift Register (LFSR) works and its applications in hardware design.",
            "model": "jamba",
            "max_length": 1500,
            "temperature": 0.5
        })
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success! Generated {result.get('tokens_generated', 0)} tokens")
            print(f"📝 Response length: {len(result.get('response', ''))} characters")
            print(f"📝 Response preview: {result.get('response', '')[:400]}...")
            print(f"⏱️  Generation time: {result.get('generation_time', 0):.2f}s")
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Connection error: {e}")
    
    print("\n" + "="*60 + "\n")
    
    # Test 4: Code generation
    print("🧪 Testing Jamba Code Generation...")
    try:
        response = requests.post(f"{base_url}/generate", json={
            "prompt": "Write a Python function to implement a simple barrel shifter that performs circular left and right shifts on a list.",
            "model": "jamba",
            "max_length": 1000,
            "temperature": 0.4
        })
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success! Generated {result.get('tokens_generated', 0)} tokens")
            print(f"📝 Response: {result.get('response', '')}")
            print(f"⏱️  Generation time: {result.get('generation_time', 0):.2f}s")
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Connection error: {e}")
    
    print("\n" + "="*60 + "\n")
    
    # Test 5: Model info
    print("🧪 Testing Model Information...")
    try:
        response = requests.get(f"{base_url}/model_info")
        
        if response.status_code == 200:
            result = response.json()
            jamba_info = result.get('jamba', {})
            print(f"✅ Jamba Status: {jamba_info.get('status', 'unknown')}")
            print(f"📋 Model Name: {jamba_info.get('model_name', 'N/A')}")
            print(f"🖥️  Device: {jamba_info.get('device', 'N/A')}")
            print(f"🏷️  Type: {jamba_info.get('model_type', 'N/A')}")
            print(f"📏 Context Length: {jamba_info.get('context_length', 'N/A')}")
            print(f"🏗️  Architecture: {jamba_info.get('architecture', 'N/A')}")
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Connection error: {e}")

def main():
    print("🚀 AI21 Jamba Reasoning 3B Model API Test Suite")
    print("="*60)
    
    # Check if server is running
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            available_models = health.get('available_models', [])
            if 'jamba' in available_models:
                print("✅ Jamba model is available!")
                print("Starting tests...\n")
                test_jamba_api()
            else:
                print("⚠️  Jamba model is not currently loaded")
                print("Available models:", available_models)
                print("\nTo enable Jamba model:")
                print("1. Edit main.py and uncomment the Jamba loading section in the lifespan function")
                print("2. Restart the server")
        else:
            print("❌ Server health check failed")
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        print("Make sure the server is running on http://localhost:8000")

if __name__ == "__main__":
    main()
