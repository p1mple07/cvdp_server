"""
Test script for Phi model to verify:
1. Model loads correctly
2. Basic generation works
3. Context window size
4. API endpoint functionality
"""

import requests
import json
import time
from transformers import AutoTokenizer

# API endpoint
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test if the API is running and Phi model is available"""
    print("=" * 60)
    print("Test 1: Health Check")
    print("=" * 60)
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✓ API Status: {data['status']}")
            print(f"✓ Available Models: {data['available_models']}")
            
            if 'phi' in data['available_models']:
                print("✓ Phi model is loaded and available!")
                return True
            else:
                print("✗ Phi model is NOT available")
                return False
        else:
            print(f"✗ Health check failed with status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Error connecting to API: {e}")
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
            
            if 'phi' in data:
                phi_info = data['phi']
                print(f"✓ Model Name: {phi_info.get('model_name', 'N/A')}")
                print(f"✓ Model Type: {phi_info.get('model_type', 'N/A')}")
                print(f"✓ Device: {phi_info.get('device', 'N/A')}")
                print(f"✓ Status: {phi_info.get('status', 'N/A')}")
                print(f"✓ Context Length: {phi_info.get('context_length', 'Not specified')}")
                return True
            else:
                print("✗ Phi model info not found")
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
    
    prompt = "Explain what a barrel shifter is in one sentence."
    
    payload = {
        "prompt": prompt,
        "model": "phi",
        "max_tokens": 100,
        "temperature": 0.7
    }
    
    print(f"Prompt: {prompt}")
    print("\nGenerating response...")
    
    try:
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/generate", json=payload)
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
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_context_window_size():
    """Test and determine the actual context window size"""
    print("\n" + "=" * 60)
    print("Test 4: Context Window Size Detection")
    print("=" * 60)
    
    try:
        # Try to load the tokenizer to check the actual model config
        model_name = "microsoft/Phi-3-mini-128k-instruct"
        
        print(f"Loading tokenizer for: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Try to get model config
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        
        # Check various config attributes for context length
        max_position_embeddings = getattr(config, 'max_position_embeddings', None)
        n_positions = getattr(config, 'n_positions', None)
        max_sequence_length = getattr(config, 'max_sequence_length', None)
        
        print(f"\n✓ Tokenizer loaded successfully")
        print(f"✓ Model config attributes:")
        print(f"  - max_position_embeddings: {max_position_embeddings}")
        print(f"  - n_positions: {n_positions}")
        print(f"  - max_sequence_length: {max_sequence_length}")
        
        # Determine context window
        context_window = max_position_embeddings or n_positions or max_sequence_length or 4096
        
        print(f"\n✓ Detected Context Window: {context_window:,} tokens")
        
        if context_window >= 128000:
            print("✓ This is the 128K variant!")
        elif context_window >= 4000:
            print("✓ This is the 4K variant")
        else:
            print(f"? Unexpected context window size: {context_window}")
        
        return context_window
        
    except Exception as e:
        print(f"✗ Error detecting context window: {e}")
        print("\nNote: You may need to check the actual model being loaded in main.py")
        return None

def test_long_context():
    """Test with a longer prompt to see context handling"""
    print("\n" + "=" * 60)
    print("Test 5: Long Context Handling")
    print("=" * 60)
    
    # Create a moderately long prompt (around 500 tokens)
    long_text = """
    A barrel shifter is a digital circuit that can shift a data word by a specified number of bits 
    in a single clock cycle. It is commonly used in processors for arithmetic operations, bit manipulation,
    and data alignment. The barrel shifter gets its name from its structure, which resembles a cylindrical
    barrel that can rotate its contents.
    
    The key feature of a barrel shifter is its ability to perform shifts of any amount in constant time,
    unlike traditional shift registers which require multiple clock cycles for multi-bit shifts. This makes
    barrel shifters essential for high-performance computing applications.
    
    There are different types of barrel shifters:
    1. Logical shifters - shift bits left or right, filling with zeros
    2. Arithmetic shifters - preserve the sign bit during right shifts
    3. Rotate shifters - circular shift where bits wrap around
    4. Funnel shifters - combine two inputs for more complex operations
    
    In hardware design, barrel shifters are typically implemented using multiplexers arranged in a logarithmic
    structure. For an N-bit shifter, this requires log2(N) stages of multiplexers, each handling different
    shift amounts (1, 2, 4, 8, etc. bits).
    
    Testing a barrel shifter requires comprehensive verification:
    - Test all shift amounts from 0 to N-1
    - Verify edge cases like shifting by 0 or maximum amount
    - Check data integrity and no bit corruption
    - Validate timing and setup/hold requirements
    - Test corner cases like all zeros, all ones, alternating patterns
    """ * 3  # Repeat to make it longer
    
    prompt = f"{long_text}\n\nBased on the above text, explain in one sentence why testing edge cases is important for barrel shifters."
    
    payload = {
        "prompt": prompt,
        "model": "phi",
        "max_tokens": 200,
        "temperature": 0.5
    }
    
    # Count approximate tokens
    token_count = len(prompt.split())
    print(f"Approximate input token count: {token_count}")
    print("\nSending long context prompt...")
    
    try:
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/generate", json=payload)
        end_time = time.time()
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n✓ Long context generation successful!")
            print(f"✓ Generation time: {end_time - start_time:.2f}s")
            print(f"\nResponse:\n{'-' * 60}")
            print(data.get('response', 'No response'))
            print('-' * 60)
            return True
        else:
            print(f"✗ Generation failed with status code: {response.status_code}")
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def check_which_phi_variant():
    """Check which Phi variant is actually being used"""
    print("\n" + "=" * 60)
    print("Test 6: Which Phi Variant Is Loaded?")
    print("=" * 60)
    
    # Read the main.py file to see which model is specified
    try:
        with open('/workspace/ckarfa/siang_btp/slm_api/main.py', 'r') as f:
            content = f.read()
            
        # Look for PhiModel initialization
        if 'microsoft/Phi-3-mini-128k-instruct' in content:
            print("✓ Code specifies: Phi-3-mini-128k-instruct (128K context)")
            return "128k"
        elif 'microsoft/Phi-3-mini-4k-instruct' in content:
            print("✓ Code specifies: Phi-3-mini-4k-instruct (4K context)")
            return "4k"
        else:
            print("? Could not determine which Phi variant is specified")
            print("\nSearching for 'Phi' in model initialization...")
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if 'class PhiModel' in line or ('def __init__' in line and i > 0 and 'Phi' in lines[i-5:i]):
                    # Print surrounding lines
                    start = max(0, i - 2)
                    end = min(len(lines), i + 5)
                    print(f"\nFound at line {i}:")
                    print('\n'.join(lines[start:end]))
                    break
            return None
    except Exception as e:
        print(f"✗ Error reading main.py: {e}")
        return None

def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("PHI MODEL TEST SUITE")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Health check
    results['health_check'] = test_health_check()
    
    if not results['health_check']:
        print("\n⚠ API not responding or Phi model not loaded!")
        print("Make sure to start the server first with:")
        print("  uvicorn main:app --host 0.0.0.0 --port 8000")
        return
    
    # Test 2: Model info
    results['model_info'] = test_model_info()
    
    # Test 3: Basic generation
    results['basic_generation'] = test_basic_generation()
    
    # Test 4: Context window detection
    context_window = test_context_window_size()
    results['context_detection'] = context_window is not None
    
    # Test 5: Long context
    results['long_context'] = test_long_context()
    
    # Test 6: Which variant
    variant = check_which_phi_variant()
    results['variant_check'] = variant is not None
    
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
    
    if context_window:
        print(f"\n📊 Context Window: {context_window:,} tokens")
        if context_window >= 128000:
            print("   This is the 128K variant! ✓")
        elif context_window >= 4000:
            print("   This is the 4K variant")
    
    if variant:
        print(f"📝 Model variant in code: {variant}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
