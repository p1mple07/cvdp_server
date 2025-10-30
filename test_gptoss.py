#!/usr/bin/env python3

"""
Test script for GPT OSS model integration
"""

import requests
import json
import time

def test_gptoss_api():
    """Test GPT OSS model via API"""
    base_url = "http://localhost:8000"
    
    # Test 1: Basic generation
    print("🧪 Testing GPT OSS Basic Generation...")
    try:
        response = requests.post(f"{base_url}/generate", json={
            "prompt": "What is machine learning?",
            "model": "gptoss",
            "max_length": 200,
            "temperature": 0.7
        })
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success! Generated {result.get('tokens_generated', 0)} tokens")
            print(f"📝 Response: {result.get('response', '')[:200]}...")
            print(f"⏱️  Generation time: {result.get('generation_time', 0):.2f}s")
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Connection error: {e}")
    
    print("\n" + "="*60 + "\n")
    
    # Test 2: JSON format response
    print("🧪 Testing GPT OSS JSON Format...")
    try:
        response = requests.post(f"{base_url}/generate", json={
            "prompt": 'Answer in JSON format: { "response": "your answer" } What are the benefits of deep learning?',
            "model": "gptoss",
            "max_length": 500,
            "temperature": 0.5
        })
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success! Generated {result.get('tokens_generated', 0)} tokens")
            print(f"📝 Response: {result.get('response', '')}")
            print(f"⏱️  Generation time: {result.get('generation_time', 0):.2f}s")
            
            # Try to parse response as JSON
            try:
                response_text = result.get('response', '')
                if response_text.startswith('{') and response_text.endswith('}'):
                    parsed = json.loads(response_text)
                    print(f"✅ Valid JSON response: {parsed.get('response', 'N/A')[:100]}...")
                else:
                    print(f"⚠️  Response not in JSON format")
            except json.JSONDecodeError:
                print(f"⚠️  Could not parse JSON response")
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Connection error: {e}")
    
    print("\n" + "="*60 + "\n")
    
    # Test 3: Long response (10x enhanced)
    print("🧪 Testing GPT OSS Long Response (10x Enhanced)...")
    try:
        response = requests.post(f"{base_url}/generate", json={
            "prompt": "Provide a comprehensive explanation of neural networks, including their architecture, training process, applications, and future prospects.",
            "model": "gptoss",
            "max_length": 2000,  # 10x enhanced capability
            "temperature": 0.6
        })
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success! Generated {result.get('tokens_generated', 0)} tokens")
            print(f"📝 Response length: {len(result.get('response', ''))} characters")
            print(f"📝 Response preview: {result.get('response', '')[:300]}...")
            print(f"⏱️  Generation time: {result.get('generation_time', 0):.2f}s")
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Connection error: {e}")
    
    print("\n" + "="*60 + "\n")
    
    # Test 4: Model info
    print("🧪 Testing Model Information...")
    try:
        response = requests.get(f"{base_url}/model_info")
        
        if response.status_code == 200:
            result = response.json()
            gptoss_info = result.get('gptoss', {})
            print(f"✅ GPT OSS Status: {gptoss_info.get('status', 'unknown')}")
            print(f"📋 Model Name: {gptoss_info.get('model_name', 'N/A')}")
            print(f"🖥️  Device: {gptoss_info.get('device', 'N/A')}")
            print(f"🏷️  Type: {gptoss_info.get('model_type', 'N/A')}")
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Connection error: {e}")

def main():
    print("🚀 GPT OSS Model API Test Suite")
    print("="*60)
    
    # Check if server is running
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            available_models = health.get('available_models', [])
            if 'gptoss' in available_models:
                print("✅ GPT OSS model is available!")
                print("Starting tests...\n")
                test_gptoss_api()
            else:
                print("❌ GPT OSS model is not available")
                print(f"Available models: {available_models}")
        else:
            print("❌ Server health check failed")
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        print("Make sure the server is running on http://localhost:8000")

if __name__ == "__main__":
    main()