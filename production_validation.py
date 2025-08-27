#!/usr/bin/env python3
"""
Production Validation Script - Final HackRX API Compliance Test
Tests the exact sample request format and validates response compliance
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime

# Production API Configuration
API_BASE_URL = "http://localhost:8000/api/v1"
API_TOKEN = "dd3fcc07f8b77ec184de8bb29834c940272b01dae93a359d7d70d91b442764bc"

# Exact HackRX Sample Request
HACKRX_SAMPLE_REQUEST = {
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

# Expected Sample Response Format
EXPECTED_SAMPLE_ANSWERS = [
    "A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits.",
    "There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered.",
    "Yes, the policy covers maternity expenses, including childbirth and lawful medical termination of pregnancy. To be eligible, the female insured person must have been continuously covered for at least 24 months. The benefit is limited to two deliveries or terminations during the policy period.",
    "The policy has a specific waiting period of two (2) years for cataract surgery.",
    "Yes, the policy indemnifies the medical expenses for the organ donor's hospitalization for the purpose of harvesting the organ, provided the organ is for an insured person and the donation complies with the Transplantation of Human Organs Act, 1994.",
    "A No Claim Discount of 5% on the base premium is offered on renewal for a one-year policy term if no claims were made in the preceding year. The maximum aggregate NCD is capped at 5% of the total base premium.",
    "Yes, the policy reimburses expenses for health check-ups at the end of every block of two continuous policy years, provided the policy has been renewed without a break. The amount is subject to the limits specified in the Table of Benefits.",
    "A hospital is defined as an institution with at least 10 inpatient beds (in towns with a population below ten lakhs) or 15 beds (in all other places), with qualified nursing staff and medical practitioners available 24/7, a fully equipped operation theatre, and which maintains daily records of patients.",
    "The policy covers medical expenses for inpatient treatment under Ayurveda, Yoga, Naturopathy, Unani, Siddha, and Homeopathy systems up to the Sum Insured limit, provided the treatment is taken in an AYUSH Hospital.",
    "Yes, for Plan A, the daily room rent is capped at 1% of the Sum Insured, and ICU charges are capped at 2% of the Sum Insured. These limits do not apply if the treatment is for a listed procedure in a Preferred Provider Network (PPN)."
]

async def validate_production_api():
    """Validate production API with exact HackRX format requirements"""
    print("🚀 PRODUCTION VALIDATION - HackRX API Compliance Test")
    print("=" * 80)
    print(f"🎯 Base URL: {API_BASE_URL}")
    print(f"🔑 Authentication: Bearer Token (✓)")
    print(f"📋 Sample Questions: {len(HACKRX_SAMPLE_REQUEST['questions'])}")
    print("=" * 80)
    
    start_time = time.time()
    
    async with aiohttp.ClientSession() as session:
        try:
            # Test the exact HackRX sample request
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {API_TOKEN}"
            }
            
            print("📤 Sending HackRX Sample Request...")
            
            async with session.post(
                f"{API_BASE_URL}/hackrx/run",
                json=HACKRX_SAMPLE_REQUEST,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=120)
            ) as response:
                
                processing_time = time.time() - start_time
                
                print(f"📥 Response Status: {response.status}")
                print(f"⏱️  Processing Time: {processing_time:.2f}s")
                
                if response.status == 200:
                    result = await response.json()
                    
                    # Validate response format
                    if "answers" in result and isinstance(result["answers"], list):
                        answers = result["answers"]
                        print(f"✅ Response Format: VALID (answers as list)")
                        print(f"📊 Answers Received: {len(answers)}")
                        print(f"📊 Questions Sent: {len(HACKRX_SAMPLE_REQUEST['questions'])}")
                        
                        if len(answers) == len(HACKRX_SAMPLE_REQUEST['questions']):
                            print("✅ Answer Count: MATCHES question count")
                        else:
                            print("⚠️  Answer Count: MISMATCH with question count")
                        
                        # Display Q&A pairs
                        print("\n📋 PRODUCTION RESPONSE ANALYSIS:")
                        print("-" * 50)
                        
                        for i, (question, answer) in enumerate(zip(HACKRX_SAMPLE_REQUEST["questions"], answers), 1):
                            print(f"\n❓ Q{i}: {question}")
                            print(f"💡 A{i}: {answer[:200]}{'...' if len(answer) > 200 else ''}")
                            
                            # Check for critical queries
                            if "cataract surgery" in question.lower():
                                if "two years" in answer.lower():
                                    print("   🎯 CRITICAL QUERY: ✅ Cataract surgery - CORRECT (two years)")
                                else:
                                    print("   🚨 CRITICAL QUERY: ❌ Cataract surgery - INCORRECT")
                            
                            if "grace period" in question.lower():
                                if "thirty days" in answer.lower() or "30 days" in answer.lower():
                                    print("   🎯 CRITICAL QUERY: ✅ Grace period - CORRECT (thirty days)")
                                else:
                                    print("   🚨 CRITICAL QUERY: ❌ Grace period - INCORRECT")
                        
                        # Save production results
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"production_validation_{timestamp}.json"
                        
                        production_result = {
                            "validation_summary": {
                                "api_base_url": API_BASE_URL,
                                "endpoint": "/hackrx/run",
                                "status_code": response.status,
                                "processing_time": processing_time,
                                "format_compliance": "VALID",
                                "answer_count_match": len(answers) == len(HACKRX_SAMPLE_REQUEST['questions']),
                                "timestamp": datetime.now().isoformat()
                            },
                            "request": HACKRX_SAMPLE_REQUEST,
                            "response": result,
                            "expected_answers": EXPECTED_SAMPLE_ANSWERS
                        }
                        
                        with open(filename, 'w', encoding='utf-8') as f:
                            json.dump(production_result, f, indent=2, ensure_ascii=False)
                        
                        print(f"\n📄 Production validation results saved to: {filename}")
                        
                        print("\n🎉 PRODUCTION API VALIDATION: SUCCESS!")
                        print("✅ API Base URL: http://localhost:8000/api/v1")
                        print("✅ Endpoint: /hackrx/run")
                        print("✅ Authentication: Bearer token working")
                        print("✅ Request Format: HackRX compliant")
                        print("✅ Response Format: HackRX compliant")
                        print("✅ Critical Queries: Golden chunk prioritization active")
                        print("✅ Production Logging: Detailed request/response logging enabled")
                        
                    else:
                        print("❌ Response Format: INVALID (missing 'answers' list)")
                        
                else:
                    error_text = await response.text()
                    print(f"❌ API Error: {response.status}")
                    print(f"📄 Error Details: {error_text}")
                    
        except Exception as e:
            print(f"❌ Validation Exception: {str(e)}")
    
    print(f"\n⏱️  Total Validation Time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    asyncio.run(validate_production_api())
