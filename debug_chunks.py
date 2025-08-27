#!/usr/bin/env python3
"""
Final Diagnostic Script: Chunk Inspection
This script directly inspects document chunks to determine if the correct information
exists after chunking, bypassing the full RAG pipeline complexity.
"""

import asyncio
import sys
import os
from typing import List, Dict, Any

# Add the current directory to Python path to import from main.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import FastRAGSystem

# --- CONFIGURATION ---
DOCUMENT_URL = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"

# Target keywords for critical searches
TARGET_KEYWORDS_GRACE = ["grace period", "30 days", "thirty days", "section 2.21"]
TARGET_KEYWORDS_CATARACT = ["cataract", "two years", "2 years", "24 months"]

async def inspect_document_chunks():
    """
    This script ingests a document, breaks it into chunks, and then
    searches those chunks to see if the correct information is being
    processed correctly.
    """
    print("🔬 Final Diagnostic: Inspecting Document Chunks...")
    print("=" * 60)
    
    try:
        # Initialize the RAG system
        rag_system = FastRAGSystem()
        
        # 1. Fetch and chunk the document using exact pipeline logic
        print(f"📄 Processing document: {DOCUMENT_URL[:70]}...")
        
        # Process the document to get chunks
        result = await rag_system.process_documents_fast([DOCUMENT_URL])
        print(f"✅ Document processed: {result}")
        
        # Get the actual chunks from the system
        all_chunks = rag_system.documents  # All processed chunks
        quick_chunks = rag_system.quick_chunks  # Quick access chunks
        
        if not all_chunks:
            print("❌ ERROR: No chunks were generated. The ingestion process failed.")
            return

        print(f"✅ Document processed into {len(all_chunks)} total chunks.")
        print(f"📊 Quick chunks available: {len(quick_chunks)} chunks.")
        print("-" * 60)

        # 2. Inspect for "Grace Period" - Search all chunks
        print("🔍 Searching for 'Grace Period' chunks...")
        found_grace_period = []
        
        for i, chunk in enumerate(all_chunks):
            content = chunk.page_content.lower()
            
            # Check for any grace period mentions
            if "grace period" in content:
                print(f"\n--- 📋 GRACE PERIOD MENTION #{len(found_grace_period)+1} ---")
                print(f"Chunk Index: {i}")
                print(f"Chunk Type: {chunk.metadata.get('chunk_type', 'unknown')}")
                print(f"Content Preview: {chunk.page_content[:300]}...")
                
                # Check if it contains specific details
                has_30_days = any(keyword in content for keyword in ["30 days", "thirty days", "30-day"])
                has_section = "section 2.21" in content or "2.21" in content
                
                print(f"Contains 30 days: {has_30_days}")
                print(f"Contains Section 2.21: {has_section}")
                
                if has_30_days or has_section:
                    found_grace_period.append({
                        'index': i,
                        'chunk': chunk,
                        'has_30_days': has_30_days,
                        'has_section': has_section
                    })
                    print("✅ POTENTIAL MATCH!")
                
                print("-" * 40)
        
        if not found_grace_period:
            print("❌ CRITICAL FINDING: No chunks contain complete 'grace period' information.")
        else:
            print(f"✅ Found {len(found_grace_period)} potential grace period chunks.")

        print("-" * 60)

        # 3. Inspect for "Cataract Surgery" - Search all chunks
        print("🔍 Searching for 'Cataract Surgery' chunks...")
        found_cataract = []
        
        for i, chunk in enumerate(all_chunks):
            content = chunk.page_content.lower()
            
            # Check for cataract mentions
            if "cataract" in content:
                print(f"\n--- 👁️ CATARACT MENTION #{len(found_cataract)+1} ---")
                print(f"Chunk Index: {i}")
                print(f"Chunk Type: {chunk.metadata.get('chunk_type', 'unknown')}")
                print(f"Content Preview: {chunk.page_content[:300]}...")
                
                # Check for specific waiting periods
                has_two_years = any(keyword in content for keyword in ["two years", "2 years", "24 months"])
                has_three_years = any(keyword in content for keyword in ["three years", "3 years", "36 months"])
                has_waiting = "waiting" in content
                
                print(f"Contains 'two years': {has_two_years}")
                print(f"Contains 'three years': {has_three_years}")
                print(f"Contains 'waiting': {has_waiting}")
                
                if has_two_years or has_three_years:
                    found_cataract.append({
                        'index': i,
                        'chunk': chunk,
                        'has_two_years': has_two_years,
                        'has_three_years': has_three_years,
                        'has_waiting': has_waiting
                    })
                    print("✅ POTENTIAL MATCH!")
                
                print("-" * 40)

        if not found_cataract:
            print("❌ CRITICAL FINDING: No chunks contain 'cataract' with specific waiting periods.")
        else:
            print(f"✅ Found {len(found_cataract)} potential cataract chunks.")

        print("=" * 60)
        
        # 4. Summary and Diagnosis
        print("🔬 DIAGNOSTIC SUMMARY:")
        print("-" * 30)
        
        if found_grace_period:
            print("✅ Grace Period: Chunks with relevant content exist")
            for item in found_grace_period:
                if item['has_30_days'] and item['has_section']:
                    print("  🎯 PERFECT CHUNK: Contains both '30 days' and 'Section 2.21'")
                elif item['has_30_days']:
                    print("  📋 GOOD CHUNK: Contains '30 days' reference")
                elif item['has_section']:
                    print("  📋 GOOD CHUNK: Contains 'Section 2.21' reference")
        else:
            print("❌ Grace Period: NO relevant chunks found - CHUNKING ISSUE")
        
        if found_cataract:
            print("✅ Cataract Surgery: Chunks with relevant content exist")
            perfect_cataract = [item for item in found_cataract if item['has_two_years']]
            if perfect_cataract:
                print("  🎯 PERFECT CHUNK: Contains 'cataract' + 'two years'")
            else:
                print("  ⚠️ IMPERFECT CHUNKS: Contains 'cataract' but not 'two years'")
        else:
            print("❌ Cataract Surgery: NO relevant chunks found - CHUNKING ISSUE")
        
        print("\n🎯 FINAL DIAGNOSIS:")
        if found_grace_period and any(item['has_30_days'] for item in found_grace_period):
            print("Grace Period: RETRIEVAL/RANKING ISSUE (perfect chunks exist)")
        else:
            print("Grace Period: CHUNKING ISSUE (no perfect chunks)")
            
        if found_cataract and any(item['has_two_years'] for item in found_cataract):
            print("Cataract Surgery: RETRIEVAL/RANKING ISSUE (perfect chunks exist)")
        else:
            print("Cataract Surgery: CHUNKING ISSUE (no perfect chunks)")

        print("=" * 60)
        print("🔬 Diagnostic Complete.")

    except Exception as e:
        print(f"❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(inspect_document_chunks())
