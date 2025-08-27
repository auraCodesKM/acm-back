#!/usr/bin/env python3
"""
Focused Validation Test - Using Accessible Documents
Demonstrates system capabilities with known working document URLs
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime

# Test configuration
API_BASE_URL = "http://localhost:8000"
API_TOKEN = "dd3fcc07f8b77ec184de8bb29834c940272b01dae93a359d7d70d91b442764bc"

# Focused test cases with accessible documents
FOCUSED_TESTS = [
    {
        "name": "Bitcoin Whitepaper - Technical Analysis",
        "document": "https://bitcoin.org/bitcoin.pdf",
        "questions": [
            "What is the main problem Bitcoin solves according to the abstract?",
            "How many confirmations does the paper suggest for secure transactions?",
            "What happens if an attacker controls more than 50% of the network?"
        ]
    },
    {
        "name": "Bajaj Insurance Policy - Critical Queries",
        "document": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
        "questions": [
            "What is the waiting period for cataract surgery?",
            "What is the grace period for premium payment?",
            "Are AYUSH treatments covered under this policy?"
        ]
    }
]

async def run_focused_test():
    """Run focused validation test with accessible documents"""
    print("🎯 FOCUSED VALIDATION TEST - Production System Capabilities")
    print("=" * 70)
    
    results = []
    
    async with aiohttp.ClientSession() as session:
        for i, test in enumerate(FOCUSED_TESTS, 1):
            print(f"\n🧪 Test {i}: {test['name']}")
            print("-" * 50)
            
            start_time = time.time()
            
            try:
                payload = {
                    "documents": test["document"],
                    "questions": test["questions"]
                }
                
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {API_TOKEN}"
                }
                
                async with session.post(
                    f"{API_BASE_URL}/hackrx/run",
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    
                    processing_time = time.time() - start_time
                    
                    if response.status == 200:
                        result = await response.json()
                        answers = result.get("answers", [])
                        
                        print(f"✅ Status: SUCCESS ({processing_time:.2f}s)")
                        print(f"📊 Questions: {len(test['questions'])}")
                        print(f"📋 Answers received: {len(answers)}")
                        
                        # Display Q&A pairs
                        for j, (question, answer) in enumerate(zip(test["questions"], answers), 1):
                            print(f"\n❓ Q{j}: {question}")
                            print(f"💡 A{j}: {answer[:200]}{'...' if len(answer) > 200 else ''}")
                        
                        results.append({
                            "test_name": test["name"],
                            "status": "SUCCESS",
                            "processing_time": processing_time,
                            "questions": test["questions"],
                            "answers": answers,
                            "timestamp": datetime.now().isoformat()
                        })
                        
                    else:
                        error_text = await response.text()
                        print(f"❌ Status: FAILED ({response.status})")
                        print(f"🚨 Error: {error_text[:200]}...")
                        
                        results.append({
                            "test_name": test["name"],
                            "status": "FAILED",
                            "error": error_text,
                            "processing_time": processing_time,
                            "timestamp": datetime.now().isoformat()
                        })
                        
            except Exception as e:
                processing_time = time.time() - start_time
                print(f"❌ Exception: {str(e)}")
                
                results.append({
                    "test_name": test["name"],
                    "status": "EXCEPTION",
                    "error": str(e),
                    "processing_time": processing_time,
                    "timestamp": datetime.now().isoformat()
                })
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"focused_validation_results_{timestamp}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump({
            "test_summary": {
                "total_tests": len(FOCUSED_TESTS),
                "successful_tests": len([r for r in results if r["status"] == "SUCCESS"]),
                "timestamp": datetime.now().isoformat()
            },
            "results": results
        }, f, indent=2)
    
    print(f"\n📄 Results saved to: {filename}")
    
    # Summary
    successful = len([r for r in results if r["status"] == "SUCCESS"])
    print(f"\n🎯 FOCUSED TEST SUMMARY:")
    print(f"✅ Successful: {successful}/{len(FOCUSED_TESTS)}")
    print(f"⏱️  Total Time: {sum(r.get('processing_time', 0) for r in results):.2f}s")
    
    if successful == len(FOCUSED_TESTS):
        print("🎉 ALL FOCUSED TESTS PASSED! System demonstrates production capabilities!")
    else:
        print("⚠️  Some tests failed. Review results for details.")

if __name__ == "__main__":
    asyncio.run(run_focused_test())
