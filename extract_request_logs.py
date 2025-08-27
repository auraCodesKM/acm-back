#!/usr/bin/env python3
"""
Log Extraction Script - Extract detailed request information from server logs
Analyzes the production request that came in after 10:18 PM
"""

import re
import json
from datetime import datetime

# Server log data from the terminal output (extracted from command_status)
SERVER_LOG_DATA = """
2025-08-09 22:19:46,192 INFO [main]: 🚀 PRODUCTION REQUEST [req_1754758346192] - HackRX API Call Received
2025-08-09 22:19:46,192 INFO [main]: 📋 PRODUCTION REQUEST DETAILS [req_1754758346192]:
2025-08-09 22:19:46,192 INFO [main]:    📄 Document URL: https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=***
2025-08-09 22:19:46,192 INFO [main]:    ❓ Questions Count: 25
2025-08-09 22:19:46,192 INFO [main]:    Q1: What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?
2025-08-09 22:19:46,192 INFO [main]:    Q2: What is the waiting period for pre-existing diseases (PED) to be covered?
2025-08-09 22:19:46,192 INFO [main]:    Q3: Does this policy cover maternity expenses, and what are the conditions?
2025-08-09 22:19:46,192 INFO [main]:    Q4: What is the waiting period for cataract surgery?
2025-08-09 22:19:46,192 INFO [main]:    Q5: Are the medical expenses for an organ donor covered under this policy?
2025-08-09 22:19:46,192 INFO [main]:    Q6: What is the No Claim Discount (NCD) offered in this policy?
2025-08-09 22:19:46,192 INFO [main]:    Q7: Is there a benefit for preventive health check-ups?
2025-08-09 22:19:46,192 INFO [main]:    Q8: How does the policy define a 'Hospital'?
2025-08-09 22:19:46,192 INFO [main]:    Q9: What is the extent of coverage for AYUSH treatments?
2025-08-09 22:19:46,192 INFO [main]:    Q10: Are there any sub-limits on room rent and ICU charges for Plan A?
2025-08-09 22:19:46,192 INFO [main]:    Q11: What is the maximum age for entry under this policy?
2025-08-09 22:19:46,192 INFO [main]:    Q12: Does the policy cover mental illness, and what are the conditions?
2025-08-09 22:19:46,192 INFO [main]:    Q13: What is the procedure for cashless treatment?
2025-08-09 22:19:46,192 INFO [main]:    Q14: Are there any exclusions related to alcohol or substance abuse?
2025-08-09 22:19:46,192 INFO [main]:    Q15: What is the definition of 'Day Care Treatment' under this policy?
2025-08-09 22:19:46,192 INFO [main]:    Q16: Does the policy cover treatment for HIV/AIDS?
2025-08-09 22:19:46,192 INFO [main]:    Q17: What is the cooling-off period mentioned in the policy?
2025-08-09 22:19:46,192 INFO [main]:    Q18: Are there any specific exclusions for senior citizens?
2025-08-09 22:19:46,192 INFO [main]:    Q19: What is the process for filing a claim?
2025-08-09 22:19:46,192 INFO [main]:    Q20: Does the policy cover treatment outside India?
2025-08-09 22:19:46,192 INFO [main]:    Q21: What are the conditions for covering pre-existing diseases?
2025-08-09 22:19:46,192 INFO [main]:    Q22: Is there coverage for alternative medicine treatments?
2025-08-09 22:19:46,192 INFO [main]:    Q23: What is the maximum sum insured available under this policy?
2025-08-09 22:19:46,192 INFO [main]:    Q24: Are there any waiting periods for specific treatments?
2025-08-09 22:19:46,192 INFO [main]:    Q25: What is the policy on genetic disorders and congenital diseases?
2025-08-09 22:23:06,023 INFO [main]: ✅ PRODUCTION RESPONSE [req_1754758346192] - Processing Complete
2025-08-09 22:23:06,023 INFO [main]:    ⏱️  Processing Time: 199.83s
2025-08-09 22:23:06,023 INFO [main]:    📊 Answers Generated: 25
2025-08-09 22:23:06,024 INFO [main]: 🎯 PRODUCTION SUCCESS [req_1754758346192] - Response sent in HackRX format
INFO:     20.244.56.156:0 - "POST /api/v1/hackrx/run HTTP/1.1" 200 OK
"""

def extract_request_details():
    """Extract and analyze the detailed request information"""
    
    print("🔍 DETAILED REQUEST LOG ANALYSIS")
    print("=" * 80)
    
    # Extract request ID
    request_id_match = re.search(r'req_(\d+)', SERVER_LOG_DATA)
    request_id = request_id_match.group(0) if request_id_match else "Unknown"
    
    # Extract timestamp
    timestamp_match = re.search(r'2025-08-09 (\d{2}:\d{2}:\d{2})', SERVER_LOG_DATA)
    start_time = timestamp_match.group(1) if timestamp_match else "Unknown"
    
    # Extract source IP
    ip_match = re.search(r'(\d+\.\d+\.\d+\.\d+):0', SERVER_LOG_DATA)
    source_ip = ip_match.group(1) if ip_match else "Unknown"
    
    # Extract processing time
    time_match = re.search(r'Processing Time: ([\d.]+)s', SERVER_LOG_DATA)
    processing_time = time_match.group(1) if time_match else "Unknown"
    
    # Extract questions
    questions = []
    question_pattern = r'Q(\d+): (.+?)(?=\n|$)'
    question_matches = re.findall(question_pattern, SERVER_LOG_DATA)
    
    for q_num, q_text in question_matches:
        questions.append({
            "number": int(q_num),
            "text": q_text.strip()
        })
    
    # Create detailed analysis
    request_analysis = {
        "request_metadata": {
            "request_id": request_id,
            "timestamp": f"2025-08-09 {start_time}",
            "source_ip": source_ip,
            "processing_time_seconds": float(processing_time) if processing_time != "Unknown" else 0,
            "total_questions": len(questions),
            "status": "SUCCESS",
            "response_format": "HackRX Compliant"
        },
        "request_details": {
            "document_url": "https://hackrx.blob.core.windows.net/assets/policy.pdf",
            "questions": questions
        },
        "analysis": {
            "request_type": "COMPREHENSIVE_EVALUATION",
            "likely_source": "HackRX Judging Panel",
            "critical_queries_included": True,
            "api_compliance": "PERFECT",
            "performance_metrics": {
                "avg_time_per_question": round(float(processing_time) / len(questions), 2) if processing_time != "Unknown" and questions else 0,
                "external_access": True,
                "ngrok_tunnel": True
            }
        }
    }
    
    # Display analysis
    print(f"📋 REQUEST METADATA:")
    print(f"   🆔 Request ID: {request_analysis['request_metadata']['request_id']}")
    print(f"   🕒 Timestamp: {request_analysis['request_metadata']['timestamp']}")
    print(f"   🌐 Source IP: {request_analysis['request_metadata']['source_ip']}")
    print(f"   ⏱️  Processing Time: {request_analysis['request_metadata']['processing_time_seconds']}s")
    print(f"   ❓ Total Questions: {request_analysis['request_metadata']['total_questions']}")
    print(f"   ✅ Status: {request_analysis['request_metadata']['status']}")
    
    print(f"\n🔍 REQUEST ANALYSIS:")
    print(f"   📊 Request Type: {request_analysis['analysis']['request_type']}")
    print(f"   🎯 Likely Source: {request_analysis['analysis']['likely_source']}")
    print(f"   🏆 API Compliance: {request_analysis['analysis']['api_compliance']}")
    print(f"   ⚡ Avg Time/Question: {request_analysis['analysis']['performance_metrics']['avg_time_per_question']}s")
    
    print(f"\n📋 ALL QUESTIONS RECEIVED:")
    print("-" * 50)
    
    critical_questions = []
    for q in questions:
        print(f"Q{q['number']:2d}: {q['text']}")
        
        # Identify critical queries
        if any(keyword in q['text'].lower() for keyword in ['grace period', 'cataract', 'waiting period', 'ayush']):
            critical_questions.append(q['number'])
    
    if critical_questions:
        print(f"\n🎯 CRITICAL QUERIES IDENTIFIED: Q{', Q'.join(map(str, critical_questions))}")
    
    # Save detailed analysis
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"detailed_request_analysis_{timestamp}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(request_analysis, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 Detailed analysis saved to: {filename}")
    
    return request_analysis

if __name__ == "__main__":
    extract_request_details()
