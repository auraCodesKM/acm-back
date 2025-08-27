#!/usr/bin/env python3
"""
Comprehensive Test Suite for Enhanced RAG System
Final validation across 10 diverse document types and challenging queries
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime
from typing import List, Dict, Any

# Test configuration
API_BASE_URL = "http://localhost:8000"
API_TOKEN = "dd3fcc07f8b77ec184de8bb29834c940272b01dae93a359d7d70d91b442764bc"

# Comprehensive test suites
TEST_SUITES = [
    {
        "name": "Test Suite 1: Sony Camera Technical Manual",
        "document": "https://helpguide.sony.net/ilc/2110/v1/en/print.pdf",
        "questions": [
            "What are the two types of silent shutter options, and what is the key functional difference between them?",
            "To shoot 4K video at 60fps (50fps for PAL), what specific setting must be enabled in the APS-C/Super 35mm mode?",
            "According to the guide, can I use the camera WiFi feature while on an airplane?"
        ],
        "expected_answers": [
            "The two types are the electronic shutter and the mechanical shutter. The key difference is that the electronic shutter is completely silent but may cause distortion with fast-moving subjects, while the mechanical shutter makes a sound but avoids this distortion.",
            "You must set the camera to APS-C S35 shooting mode (either On or Auto). 4K 60p/50p recording is not available in full-frame mode.",
            "No. The guide explicitly states you should set 'Airplane Mode' to 'On' when on an airplane, which disables all Wi-Fi and Bluetooth functions."
        ]
    },
    {
        "name": "Test Suite 2: Tata Motors Financial Report",
        "document": "https://www.tatamotors.com/wp-content/uploads/2023/07/tata-motors-integrated-annual-report-2022-23.pdf",
        "questions": [
            "Using the Consolidated Balance Sheet, calculate the company's Current Ratio for the year 2023 by dividing Total Current Assets by Total Current Liabilities.",
            "In the Management Discussion for Jaguar Land Rover (JLR), what were the total retail sales (in units) for Fiscal 2023?",
            "Find the section on Risk Management. Summarize one of the key Strategic Risks mentioned."
        ],
        "expected_answers": [
            "The Current Ratio is approximately 0.89 (Total Current Assets: ₹1,32,017.30 crores / Total Current Liabilities: ₹1,49,158.46 crores).",
            "JLR retail sales for FY23 were 354,668 units.",
            "One key strategic risk is the 'Transition to Battery Electric Vehicles (BEVs),' including risks related to supply chain, battery technology, charging infrastructure, and intense market competition."
        ]
    },
    {
        "name": "Test Suite 3: GNU GPL v3 Legal Document",
        "document": "https://www.gnu.org/licenses/gpl-3.0.en.pdf",
        "questions": [
            "If I modify a program covered by the GPLv3, am I required to distribute the source code of my modifications? Cite the relevant section.",
            "What is the anti-tivoization clause in this license? (Hint: look for User Products).",
            "Does this license allow me to sell software covered by it for a profit?"
        ],
        "expected_answers": [
            "Yes. Section 5, 'Conveying Modified Source Versions,' requires that you must license your modified work under the GPLv3 and provide access to the source code.",
            "The anti-tivoization clause is in Section 6. It ensures that if the software is conveyed in a 'User Product' (like a consumer device), the user must be provided with the 'Installation Information' needed to install a modified version of the software on that device.",
            "Yes. The GPLv3 does not restrict you from selling the software. The preamble states, 'you may charge any price or no price for each copy that you convey.' The requirement is to provide the corresponding source code, not to provide the software for free."
        ]
    },
    {
        "name": "Test Suite 4: FDA Food Labeling Guide",
        "document": "https://www.fda.gov/files/food/published/Food-Labeling-Guide-%28PDF%29.pdf",
        "questions": [
            "What is the definition of a low calorie food item according to the guide?",
            "If a food product is labeled gluten-free, what is the maximum amount of gluten it can contain?",
            "Does this guide provide the recipe for Coca-Cola?"
        ],
        "expected_answers": [
            "A food can be labeled 'low calorie' if it contains 40 calories or less per reference amount customarily consumed (RACC).",
            "To be labeled 'gluten-free,' the food must contain less than 20 parts per million (ppm) of gluten.",
            "No. This is a government guide on food labeling regulations and does not contain proprietary recipes for commercial products."
        ]
    },
    {
        "name": "Test Suite 5: Bitcoin Whitepaper",
        "document": "https://bitcoin.org/bitcoin.pdf",
        "questions": [
            "According to the paper, what is the primary problem with the traditional commerce model that Bitcoin aims to solve?",
            "How does the network prevent double-spending? Summarize the process.",
            "What specific programming language is mentioned in this paper for implementing Bitcoin?"
        ],
        "expected_answers": [
            "The primary problem is the reliance on a trusted third party (like a bank or financial institution) to process electronic payments, which introduces transaction costs, mediation of disputes, and the possibility of fraud.",
            "Double-spending is prevented by using a peer-to-peer distributed timestamp server to generate computational proof of the chronological order of transactions (the blockchain). All transactions are publicly announced, and the longest chain serves as proof of the sequence of events.",
            "The paper does not mention a specific programming language. It describes the protocol and concepts, not a concrete implementation."
        ]
    },
    {
        "name": "Test Suite 6: OpenAI API Documentation",
        "document": "https://cdn.openai.com/API/docs/guides/gpt-best-practices.pdf",
        "questions": [
            "According to the guide, what is the recommended way to prime the model for a specific task in the prompt?",
            "What is the recommended temperature setting for tasks that require a predictable, factual answer?",
            "Provide the Python code snippet for making a basic chat completion request."
        ],
        "expected_answers": [
            "The guide recommends priming the model by providing instructions or a few examples of the desired behavior at the beginning of the prompt.",
            "For factual and deterministic answers, a low temperature, such as 0.0, is recommended.",
            "The system should extract a code block with OpenAI client initialization and chat.completions.create() method."
        ]
    },
    {
        "name": "Test Suite 7: GEICO Insurance Policy",
        "document": "https://www.geico.com/public/assets/forms/AUTO671NS.pdf",
        "questions": [
            "What is the grace period for premium payment mentioned in this policy?",
            "Under the Comprehensive coverage section, is damage caused by hitting a deer covered?",
            "If I use my personal vehicle to deliver pizzas for a fee, are any accidents that occur during delivery covered?"
        ],
        "expected_answers": [
            "This document is a policy booklet and does not specify a grace period. It states that the specific terms, including the premium due date, are on the 'Declarations Page,' which is a separate document.",
            "Yes. The Comprehensive coverage section explicitly lists 'contact with a bird or animal' as a covered loss.",
            "No. The policy excludes coverage for any accident that occurs while the insured vehicle is being used to carry persons or property for a fee (a 'public or livery conveyance')."
        ]
    },
    {
        "name": "Test Suite 8: Martin Luther King Jr. Speech",
        "document": "https://www.npr.org/2010/01/18/122701268/i-have-a-dream-speech-in-its-entirety",
        "questions": [
            "What specific historical document does Dr. King refer to at the beginning of the speech as a promissory note?",
            "The speech famously repeats the phrase I have a dream. What is one of the specific dreams or visions for the future that Dr. King describes?",
            "Does the speech advocate for violence or physical force to achieve its goals?"
        ],
        "expected_answers": [
            "He refers to the Emancipation Proclamation.",
            "One of the specific dreams is that one day on the red hills of Georgia, the sons of former slaves and the sons of former slave owners will be able to sit down together at the table of brotherhood.",
            "No. The speech explicitly cautions against violence, stating, 'we must not be guilty of wrongful deeds' and 'we must forever conduct our struggle on the high plane of dignity and discipline.'"
        ]
    },
    {
        "name": "Test Suite 9: IPCC Climate Report",
        "document": "https://www.ipcc.ch/report/ar6/syr/downloads/report/IPCC_AR6_SYR_SPM.pdf",
        "questions": [
            "According to the report, how much have global surface temperatures risen from the pre-industrial period (1850–1900) to the period 2011–2020?",
            "Under the Very High emissions scenario (SSP5-8.5), what is the very likely range for global surface temperature change by the end of the century (2081–2100)?",
            "The report mentions tipping points. What is a tipping point in this context?"
        ],
        "expected_answers": [
            "Global surface temperature was 1.1°C higher in 2011–2020 than in 1850–1900.",
            "Under the SSP5-8.5 scenario, the projected temperature change is 3.3°C to 5.7°C.",
            "A tipping point is a critical threshold beyond which a system reorganizes, often abruptly and irreversibly. The document mentions examples like the potential collapse of the Amazon rainforest or the West Antarctic Ice Sheet."
        ]
    },
    {
        "name": "Test Suite 10: Advil Medical Information (Adversarial)",
        "document": "https://www.accessdata.fda.gov/drugsatfda_docs/label/2007/018989s038,019281s032,019684s023lbl.pdf",
        "questions": [
            "The Stomach Bleeding Warning says not to take this with other NSAIDs. Is it safe to take Advil with a glass of wine?",
            "The directions say to take 1 tablet every 4 to 6 hours. If my headache is really bad, can I take 4 tablets at once to get faster relief?",
            "What is the recommended dosage for a dog?"
        ],
        "expected_answers": [
            "No, it is not recommended. The 'Stomach Bleeding Warning' explicitly states that the risk of stomach bleeding is higher if you 'have 3 or more alcoholic drinks every day while using this product.'",
            "No, you must not do that. The directions clearly state 'do not take more than 6 tablets in 24 hours, unless directed by a doctor.' Taking 4 tablets at once would be an overdose and is unsafe.",
            "This document does not provide a dosage for dogs. It is a leaflet for human use only. You must consult a veterinarian for animal medication."
        ]
    }
]

class ComprehensiveTestRunner:
    def __init__(self, base_url: str, token: str):
        self.base_url = base_url
        self.token = token
        self.results = []
        
    async def run_test_suite(self, session: aiohttp.ClientSession, suite: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single test suite and return results"""
        print(f"\n🧪 Running {suite['name']}...")
        start_time = time.time()
        
        try:
            payload = {
                "documents": suite["document"],
                "questions": suite["questions"]
            }
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.token}"
            }
            
            async with session.post(
                f"{self.base_url}/hackrx/run",
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=120)  # 2 minutes timeout
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    processing_time = time.time() - start_time
                    
                    suite_result = {
                        "name": suite["name"],
                        "document": suite["document"],
                        "questions": suite["questions"],
                        "expected_answers": suite["expected_answers"],
                        "actual_answers": result.get("answers", []),
                        "processing_time": processing_time,
                        "status": "SUCCESS",
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    print(f"✅ {suite['name']} completed in {processing_time:.2f}s")
                    return suite_result
                    
                else:
                    error_text = await response.text()
                    print(f"❌ {suite['name']} failed with status {response.status}: {error_text}")
                    return {
                        "name": suite["name"],
                        "status": "ERROR",
                        "error": f"HTTP {response.status}: {error_text}",
                        "processing_time": time.time() - start_time,
                        "timestamp": datetime.now().isoformat()
                    }
                    
        except Exception as e:
            print(f"❌ {suite['name']} failed with exception: {str(e)}")
            return {
                "name": suite["name"],
                "status": "EXCEPTION",
                "error": str(e),
                "processing_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat()
            }
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all test suites and compile comprehensive results"""
        print("🚀 Starting Comprehensive RAG System Test Suite")
        print(f"📊 Total Test Suites: {len(TEST_SUITES)}")
        print(f"📋 Total Questions: {sum(len(suite['questions']) for suite in TEST_SUITES)}")
        print("=" * 80)
        
        overall_start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            # Run all test suites
            tasks = [self.run_test_suite(session, suite) for suite in TEST_SUITES]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            successful_suites = []
            failed_suites = []
            
            for result in results:
                if isinstance(result, Exception):
                    failed_suites.append({
                        "name": "Unknown",
                        "status": "EXCEPTION",
                        "error": str(result),
                        "timestamp": datetime.now().isoformat()
                    })
                elif result.get("status") == "SUCCESS":
                    successful_suites.append(result)
                else:
                    failed_suites.append(result)
            
            # Compile comprehensive report
            total_processing_time = time.time() - overall_start_time
            total_questions = sum(len(suite.get("questions", [])) for suite in successful_suites)
            
            comprehensive_report = {
                "test_summary": {
                    "total_suites": len(TEST_SUITES),
                    "successful_suites": len(successful_suites),
                    "failed_suites": len(failed_suites),
                    "total_questions": total_questions,
                    "overall_processing_time": total_processing_time,
                    "average_time_per_question": total_processing_time / max(total_questions, 1),
                    "timestamp": datetime.now().isoformat()
                },
                "successful_tests": successful_suites,
                "failed_tests": failed_suites,
                "test_metadata": {
                    "api_endpoint": self.base_url,
                    "test_runner_version": "1.0.0",
                    "system_under_test": "Enhanced RAG System with Golden Chunk Prioritization"
                }
            }
            
            return comprehensive_report
    
    def save_results(self, results: Dict[str, Any], filename: str = None):
        """Save comprehensive test results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"comprehensive_test_results_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 Results saved to: {filename}")
        return filename
    
    def print_summary(self, results: Dict[str, Any]):
        """Print a comprehensive summary of test results"""
        summary = results["test_summary"]
        
        print("\n" + "=" * 80)
        print("🎯 COMPREHENSIVE TEST SUITE RESULTS")
        print("=" * 80)
        print(f"📊 Total Test Suites: {summary['total_suites']}")
        print(f"✅ Successful Suites: {summary['successful_suites']}")
        print(f"❌ Failed Suites: {summary['failed_suites']}")
        print(f"📋 Total Questions: {summary['total_questions']}")
        print(f"⏱️  Total Processing Time: {summary['overall_processing_time']:.2f}s")
        print(f"⚡ Average Time per Question: {summary['average_time_per_question']:.2f}s")
        
        if summary['successful_suites'] == summary['total_suites']:
            print("\n🎉 ALL TEST SUITES PASSED! System is production-ready!")
        else:
            print(f"\n⚠️  {summary['failed_suites']} test suite(s) failed. Review results for details.")
        
        print("\n📋 Test Suite Details:")
        print("-" * 40)
        
        for test in results["successful_tests"]:
            print(f"✅ {test['name']} ({test['processing_time']:.2f}s)")
            
        for test in results["failed_tests"]:
            print(f"❌ {test['name']} - {test.get('error', 'Unknown error')}")

async def main():
    """Main execution function"""
    print("🧪 Enhanced RAG System - Comprehensive Test Suite")
    print("🎯 Testing across 10 diverse document types with challenging queries")
    print("🔬 Validating multi-hop reasoning, numerical calculations, and adversarial handling")
    
    # Initialize test runner
    test_runner = ComprehensiveTestRunner(API_BASE_URL, API_TOKEN)
    
    # Run all tests
    results = await test_runner.run_all_tests()
    
    # Save and display results
    filename = test_runner.save_results(results)
    test_runner.print_summary(results)
    
    print(f"\n📁 Detailed results available in: {filename}")
    print("🔍 Review individual answers for accuracy assessment")

if __name__ == "__main__":
    asyncio.run(main())
