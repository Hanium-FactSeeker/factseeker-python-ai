#!/usr/bin/env python3
"""
증거 전처리 함수 테스트 스크립트
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.lambdas import clean_evidence_content, clean_evidence_json

def test_evidence_cleaning():
    """증거 전처리 함수 테스트"""
    
    # 테스트 케이스들
    test_cases = [
        {
            "name": "기자명 제거 테스트",
            "input": "이번 사건은 매우 심각한 문제입니다. 김철수 기자가 현장에서 확인했습니다.",
            "expected_contains": "이번 사건은 매우 심각한 문제입니다."
        },
        {
            "name": "HTML 태그 제거 테스트",
            "input": "<p>이것은 <strong>중요한</strong> 뉴스입니다.</p>",
            "expected_contains": "이것은 중요한 뉴스입니다."
        },
        {
            "name": "Copyright 제거 테스트",
            "input": "경제 뉴스입니다. Copyright © 2024 한국경제. All rights reserved.",
            "expected_contains": "경제 뉴스입니다."
        },
        {
            "name": "언론사명 제거 테스트",
            "input": "[조선일보] 이번 사건은 매우 심각한 문제입니다.",
            "expected_contains": "이번 사건은 매우 심각한 문제입니다."
        },
        {
            "name": "복합 패턴 테스트",
            "input": "<p>[연합뉴스] 이번 사건은 매우 심각한 문제입니다. 김철수 기자. Copyright © 2024 연합뉴스.</p>",
            "expected_contains": "이번 사건은 매우 심각한 문제입니다."
        }
    ]
    
    print("🧪 증거 전처리 함수 테스트 시작\n")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"테스트 {i}: {test_case['name']}")
        print(f"입력: {test_case['input']}")
        
        result = clean_evidence_content(test_case['input'])
        print(f"결과: {result}")
        
        if test_case['expected_contains'] in result:
            print("✅ 통과")
        else:
            print("❌ 실패")
        print("-" * 50)
    
    # JSON 증거 리스트 테스트
    print("\n📋 JSON 증거 리스트 테스트")
    test_evidence = [
        {
            "url": "https://example.com/1",
            "snippet": "김철수 기자가 확인한 바에 따르면, 이번 사건은 매우 심각합니다.",
            "justification": "이번 사건은 매우 심각한 문제입니다. Copyright © 2024 연합뉴스."
        },
        {
            "url": "https://example.com/2", 
            "snippet": "<p>이것은 <strong>중요한</strong> 뉴스입니다.</p>",
            "justification": "[조선일보] 중요한 뉴스입니다."
        }
    ]
    
    cleaned_evidence = clean_evidence_json(test_evidence)
    
    print("원본 증거:")
    for i, evidence in enumerate(test_evidence):
        print(f"  {i+1}. snippet: {evidence['snippet']}")
        print(f"     justification: {evidence['justification']}")
    
    print("\n전처리된 증거:")
    for i, evidence in enumerate(cleaned_evidence):
        print(f"  {i+1}. snippet: {evidence['snippet']}")
        print(f"     justification: {evidence['justification']}")

if __name__ == "__main__":
    test_evidence_cleaning()

