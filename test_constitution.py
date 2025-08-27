#!/usr/bin/env python3
"""
Test script for HackRX RAG System
Tests the /hackrx/run endpoint with the correct format and authentication
"""

import asyncio
import httpx
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
BASE_URL = "http://localhost:8000"
BEARER_TOKEN = os.getenv("Authorization_Bearer", "dd3fcc07f8b77ec184de8bb29834c940272b01dae93a359d7d70d91b442764bc")

# Test data matching the HackRX format
TEST_REQUEST = {
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "Does this policy cover maternity expenses, and what are the conditions?",
        "What is the waiting period for cataract surgery?",
        "Are the medical expenses for an organ donor covered under this policy?",
        "What is the No Claim Discount (NCD) offered in this policy?",
        "Is there a benefit for preventive health check-ups?",
        "How does the policy define a 'Hospital'?",
        "What is the extent of coverage for AYUSH treatments?",
        "Are there any sub-limits on room rent and ICU charges for Plan A?"
    ]
}

async def test_hackrx_endpoint():
    """Test the HackRX endpoint"""
    print("🚀 Testing HackRX RAG System")
    print(f"📍 Base URL: {BASE_URL}")
    print(f"🔑 Using Bearer Token: {BEARER_TOKEN[:20]}...")
    print(f"📄 Document URL: {TEST_REQUEST['documents'][:50]}...")
    print(f"❓ Number of questions: {len(TEST_REQUEST['questions'])}")
    print("-" * 60)
    
    headers = {
        "Authorization": f"Bearer {BEARER_TOKEN}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            # Test health endpoint first
            print("🏥 Testing health endpoint...")
            health_response = await client.get(f"{BASE_URL}/health")
            print(f"Health Status: {health_response.status_code}")
            if health_response.status_code == 200:
                health_data = health_response.json()
                print(f"Models Loaded: {health_data.get('models_loaded', 'Unknown')}")
                print(f"Startup Complete: {health_data.get('startup_complete', 'Unknown')}")
            print()
            
            # Test the main endpoint
            print("🎯 Testing /hackrx/run endpoint...")
            response = await client.post(
                f"{BASE_URL}/hackrx/run",
                headers=headers,
                json=TEST_REQUEST
            )
            
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                answers = result.get("answers", [])
                
                print(f"✅ Success! Received {len(answers)} answers")
                print("-" * 60)
                
                # Display results
                for i, (question, answer) in enumerate(zip(TEST_REQUEST["questions"], answers), 1):
                    print(f"Q{i}: {question}")
                    print(f"A{i}: {answer}")
                    print()
                
                # Validate response format
                if isinstance(answers, list) and len(answers) == len(TEST_REQUEST["questions"]):
                    print("✅ Response format is correct!")
                    
                    # Save results
                    with open("hackrx_test_results.json", "w") as f:
                        json.dump({
                            "request": TEST_REQUEST,
                            "response": result,
                            "status": "success",
                            "timestamp": str(asyncio.get_event_loop().time())
                        }, f, indent=2)
                    print("💾 Results saved to hackrx_test_results.json")
                else:
                    print("❌ Response format validation failed!")
                    
            else:
                print(f"❌ Request failed with status {response.status_code}")
                print(f"Response: {response.text}")
                
        except Exception as e:
            print(f"❌ Error during testing: {str(e)}")
            return False
    
    return True

async def main():
    """Main test function"""
    print("=" * 60)
    print("🧪 HackRX RAG System Test")
    print("=" * 60)
    
    success = await test_hackrx_endpoint()
    
    print("=" * 60)
    if success:
        print("✅ Test completed successfully!")
    else:
        print("❌ Test failed!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
