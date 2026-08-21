from __future__ import annotations

import re
from typing import Optional

import ollama

from rag.retriever import retrieve


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "qwen2.5:1.5b"

DEFAULT_TOP_K = 3

# Questions clearly related to resume/recruitment
RECRUITMENT_KEYWORDS = [
    "resume",
    "cv",
    "candidate",
    "applicant",
    "experience",
    "work experience",
    "skill",
    "skills",
    "education",
    "degree",
    "qualification",
    "project",
    "projects",
    "certification",
    "certifications",
    "internship",
    "internships",
    "job",
    "role",
    "position",
    "employment",
    "company",
    "companies",
    "sales",
    "marketing",
    "developer",
    "development",
    "programming",
    "python",
    "java",
    "javascript",
    "sql",
    "machine learning",
    "artificial intelligence",
    "ai",
    "team",
    "leadership",
    "manager",
    "management",
    "salary",
    "location",
    "strength",
    "weakness",
    "requirement",
    "requirements",
    "eligible",
    "eligibility",
    "qualified",
    "qualification",
    "match",
    "matches",
    "missing",
    "compare",
    "comparison",
    "better candidate",
    "stronger candidate",
    "shortlist",
    "selected",
    "reject",
    "rejected",
]


# ============================================================
# QUESTION CLASSIFICATION
# ============================================================

def normalize_question(question: str) -> str:
    """
    Normalize recruiter question for classification.
    """

    return re.sub(
        r"\s+",
        " ",
        question.strip().lower(),
    )


def is_comparison_question(question: str) -> bool:
    """
    Detect questions that require comparing candidates.
    """

    q = normalize_question(question)

    comparison_patterns = [
        "compare",
        "comparison",
        "compare candidate",
        "candidate a and candidate b",
        "candidate a vs candidate b",
        "candidate a versus candidate b",
        "who is better",
        "which candidate is better",
        "who is stronger",
        "which candidate is stronger",
        "best candidate",
        "better candidate",
        "stronger candidate",
        "rank candidates",
        "rank the candidates",
    ]

    return any(
        pattern in q
        for pattern in comparison_patterns
    )


def is_resume_related(question: str) -> bool:
    """
    Determine whether the question is related to
    resume screening/recruitment.

    This prevents unrelated questions from being
    passed to the LLM as if they were resume questions.
    """

    q = normalize_question(question)

    return any(
        keyword in q
        for keyword in RECRUITMENT_KEYWORDS
    )


# ============================================================
# CANDIDATE DETECTION
# ============================================================

def extract_candidates(results: list[dict]) -> list[str]:
    """
    Extract unique candidate names from retrieved chunks.
    """

    candidates = []

    for result in results:

        metadata = result.get(
            "metadata",
            {},
        )

        candidate = metadata.get(
            "candidate"
        )

        if candidate:

            candidate = str(
                candidate
            ).strip()

            if (
                candidate
                and candidate not in candidates
            ):
                candidates.append(
                    candidate
                )

    return candidates


# ============================================================
# CONTEXT BUILDING
# ============================================================

def build_context(
    results: list[dict],
) -> str:
    """
    Convert retrieved chunks into a strict context
    for the LLM.

    Each chunk contains source and page information.
    """

    if not results:
        return "NO RESUME EVIDENCE FOUND."

    context_parts = []

    for index, result in enumerate(
        results,
        start=1,
    ):

        metadata = result.get(
            "metadata",
            {},
        )

        candidate = metadata.get(
            "candidate",
            "Unknown Candidate",
        )

        source = metadata.get(
            "source",
            "Unknown Source",
        )

        page = metadata.get(
            "page",
            "Unknown",
        )

        text = result.get(
            "text",
            "",
        ).strip()

        if not text:
            continue

        context_parts.append(
            f"""
EVIDENCE {index}
Candidate: {candidate}
Source: {source}
Page: {page}

{text}
""".strip()
        )

    return "\n\n".join(
        context_parts
    )


# ============================================================
# OUT-OF-SCOPE RESPONSE
# ============================================================

def out_of_scope_response() -> str:
    """
    Response used when the recruiter asks something
    unrelated to resume screening.
    """

    return (
        "This question is outside the scope of the "
        "resume screening system. I can answer questions "
        "about the uploaded candidates, their skills, "
        "experience, education, projects, certifications, "
        "requirements, eligibility, and candidate comparison."
    )


# ============================================================
# SINGLE-CANDIDATE COMPARISON PROTECTION
# ============================================================

def comparison_error_response(
    candidates: list[str],
) -> str:
    """
    Prevent the LLM from inventing Candidate A/B
    when only one candidate exists.
    """

    if len(candidates) == 0:

        return (
            "Candidate comparison cannot be performed "
            "because no candidate evidence was found."
        )

    if len(candidates) == 1:

        return (
            "Candidate comparison cannot be performed "
            f"because only one candidate is available: "
            f"{candidates[0]}. Please upload at least "
            "two different candidate resumes for comparison."
        )

    return ""


# ============================================================
# PROMPT
# ============================================================

def build_prompt(
    question: str,
    context: str,
    candidates: list[str],
) -> str:
    """
    Create a strict recruiter-focused prompt.

    The model is explicitly instructed not to:
      - invent facts
      - use outside knowledge
      - create fake candidates
      - create fake evidence
      - answer unrelated questions
    """

    candidate_count = len(candidates)

    candidate_list = (
        ", ".join(candidates)
        if candidates
        else "None"
    )

    prompt = f"""
You are an AI Resume Screening Assistant.

Your ONLY job is to answer recruiter questions using
the resume evidence provided below.

IMPORTANT RULES:

1. Use ONLY the supplied resume evidence.
2. Do NOT use your general/world knowledge.
3. Do NOT invent information.
4. Do NOT assume that a missing skill exists.
5. If a skill, requirement, experience, education,
   project, certification, or other fact is not supported
   by the evidence, explicitly say:

   "Not demonstrated in the resume."

6. NEVER create fictional Candidate A, Candidate B,
   Candidate C, etc.
7. NEVER compare candidates unless at least TWO distinct
   candidates are present in the evidence.
8. If only one candidate exists and the user asks for
   comparison, clearly state that comparison is impossible
   with one candidate.
9. If the question is unrelated to recruitment or the
   uploaded resumes, do not answer it. State that the
   question is outside the scope of this system.
10. Do not repeat the same fact multiple times.
11. Give a concise recruiter-friendly answer.
12. Every factual statement must be supported by the
    supplied evidence.
13. Preserve numbers exactly when present in the resume.
14. Do not manufacture "Evidence 1", "Evidence 2", etc.
    Evidence labels below are internal source references.
15. When useful, mention the source file and page.
16. If the evidence is insufficient to answer the question,
    say so instead of guessing.

CURRENT CANDIDATES:
Number of candidates: {candidate_count}
Candidates: {candidate_list}

RECRUITER QUESTION:
{question}

RESUME EVIDENCE:
{context}

Now answer the recruiter question.

Return ONLY the final recruiter-facing answer.
Do not explain your reasoning.
"""

    return prompt.strip()


# ============================================================
# GENERATE WITH OLLAMA
# ============================================================

def generate_answer(
    question: str,
    results: list[dict],
    candidates: list[str],
) -> str:
    """
    Generate grounded answer using local Ollama model.
    """

    context = build_context(
        results
    )

    prompt = build_prompt(
        question=question,
        context=context,
        candidates=candidates,
    )

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict, factual resume "
                    "screening assistant. Never invent "
                    "candidate information."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        options={
            "temperature": 0.1,
            "top_p": 0.8,
            "num_predict": 300,
        },
    )

    answer = (
        response
        .get("message", {})
        .get("content", "")
        .strip()
    )

    if not answer:

        return (
            "I could not generate a reliable answer "
            "from the available resume evidence."
        )

    return answer


# ============================================================
# MAIN RAG PIPELINE
# ============================================================

def answer_recruiter_question(
    question: str,
    top_k: int = DEFAULT_TOP_K,
) -> dict:
    """
    Complete recruiter-question RAG pipeline.

    Flow:

        Question
            ↓
        Scope validation
            ↓
        Retrieval
            ↓
        Candidate detection
            ↓
        Comparison validation
            ↓
        Qwen generation
    """

    if not question or not question.strip():

        raise ValueError(
            "Recruiter question cannot be empty."
        )

    question = question.strip()

    # --------------------------------------------------------
    # 1. Scope check
    # --------------------------------------------------------

    if not is_resume_related(question):

        return {
            "answer": out_of_scope_response(),
            "results": [],
            "candidates": [],
            "comparison": False,
            "out_of_scope": True,
        }

    # --------------------------------------------------------
    # 2. Comparison detection
    # --------------------------------------------------------

    comparison_requested = (
        is_comparison_question(
            question
        )
    )

    # --------------------------------------------------------
    # 3. Retrieval
    # --------------------------------------------------------

    results = retrieve(
        query=question,
        top_k=top_k,
    )

    # --------------------------------------------------------
    # 4. Candidate detection
    # --------------------------------------------------------

    candidates = extract_candidates(
        results
    )

    # --------------------------------------------------------
    # 5. Comparison protection
    # --------------------------------------------------------

    if comparison_requested:

        error = comparison_error_response(
            candidates
        )

        if error:

            return {
                "answer": error,
                "results": results,
                "candidates": candidates,
                "comparison": True,
                "out_of_scope": False,
            }

    # --------------------------------------------------------
    # 6. Generate
    # --------------------------------------------------------

    answer = generate_answer(
        question=question,
        results=results,
        candidates=candidates,
    )

    return {
        "answer": answer,
        "results": results,
        "candidates": candidates,
        "comparison": comparison_requested,
        "out_of_scope": False,
    }


# ============================================================
# SOURCE DISPLAY
# ============================================================

def print_sources(
    results: list[dict],
):
    """
    Display source files and pages used by RAG.
    """

    if not results:

        print(
            "\nSOURCES\n• No resume evidence found."
        )

        return

    print("\n" + "-" * 70)
    print("SOURCES")
    print("-" * 70)

    seen = set()

    for result in results:

        metadata = result.get(
            "metadata",
            {},
        )

        source = metadata.get(
            "source",
            "Unknown",
        )

        page = metadata.get(
            "page",
            "Unknown",
        )

        key = (
            source,
            page,
        )

        if key in seen:
            continue

        seen.add(key)

        print(
            f"• {source} — Page {page}"
        )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("LOCAL RAG RECRUITER ASSISTANT")
    print("=" * 70)

    question = input(
        "\nRecruiter Question: "
    ).strip()

    if not question:

        print(
            "\n❌ Please enter a question."
        )

        raise SystemExit

    try:

        print(
            "\n[1/3] Processing recruiter question..."
        )

        result = answer_recruiter_question(
            question=question,
            top_k=3,
        )

        print(
            "\n[2/3] Candidates detected:"
        )

        candidates = result[
            "candidates"
        ]

        if candidates:

            for candidate in candidates:

                print(
                    f"  • {candidate}"
                )

        else:

            print(
                "  • None"
            )

        print(
            "\n[3/3] FINAL RAG ANSWER"
        )

        print("\n" + "=" * 70)

        print(
            result["answer"]
        )

        print("=" * 70)

        if result["results"]:

            print_sources(
                result["results"]
            )

        print(
            "\n" + "=" * 70
        )

        print(
            "RAG RECRUITER ASSISTANT SUCCESSFUL ✅"
        )

        print(
            "=" * 70
        )

    except Exception as error:

        print("\n❌ ERROR")
        print("-" * 70)
        print(error)