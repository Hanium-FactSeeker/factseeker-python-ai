from langchain_core.prompts import PromptTemplate
from core.llm_chains import get_chat_llm


def build_article_claim_extractor():
    """LLM chain: Extract fact-checkable claims from an article body text.

    The output format intentionally mirrors the existing claim extractor style
    so downstream reducers and processors can be reused without changes.
    """
    prompt = PromptTemplate.from_template(
        """
[역할]
당신은 뉴스 기사 본문에서 **팩트체크 가능한 주장**만 선별하고, 각 주장을
간결하고 검증 가능한 문장으로 **정제**해 나열하는 전문가입니다.

[입력]
아래는 단일 뉴스 기사 본문입니다. 이 본문에서 검증 가능한 주장만 추출하세요.

[최상위 원칙]
1) 기사에 **명시된 정보만** 사용하세요. 없는 정보/추정 추가 금지.
2) **출력 형식**을 무조건 지키세요. 다른 설명/머리말/코멘트 금지.
3) 한 문장에 복수 주장이 있으면 **개별 항목으로 분리**하세요.

[팩트체크 가능 기준]
- 수치/금액/연도 등 **객관적 데이터**가 포함된 사실 주장
- 인물/기관의 **발언/행위/결정/조치** 등 검증 가능한 사건
- **공식 문서/보도/발표**로 확인 가능한 내용

[팩트체크 불가 기준]
- 감정/평가/의견/추측/전망 위주의 서술
- 주체/행위/시점이 불분명하거나 모호한 표현
- 광고/홍보성/단순 소개성 문구

[정제 지침]
- **주어+핵심 사실**으로 간단 명료하게.
- 필요 시 기사에서 나타난 **발언자/주체**를 명시하고, 불명확하면 "(발언자 미상)"으로 표기.
- 과도한 수사나 주관 표현 제거.

[출력 형식]
- [번호] 정제된 주장 문장. → (분류): 사유
- [번호] 정제된 주장 문장. → (분류): 사유

입력 기사 본문:
{article_text}

출력:
"""
    )
    return prompt | get_chat_llm()

