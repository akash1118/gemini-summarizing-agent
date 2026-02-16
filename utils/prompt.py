SUMMARY_PROMPT = """
You are a professional document summarizer.

TASK:
Summarize the following text in 150–180 words.

RULES:
- Maintain consistent professional tone.
- Do not exceed 180 words.
- Do not go below 150 words.
- Avoid repetition.
- Keep output structured.

FORMAT:

Overview:
<Concise overview paragraph>

Key Points:
- Bullet point 1
- Bullet point 2
- Bullet point 3

TEXT:
{text}
"""

COMBINE_PROMPT = """
You are given partial summaries of a document.

Your task:
Combine them into a final summary in 150–180 words.

Follow same formatting rules:

Overview:
...

Key Points:
- ...
- ...
- ...

Partial Summaries:
{combined_text}
"""

STRUCTURED_SUMMARY_PROMPT = """
You are a professional document summarizer.
{style_instruction}
Summarize the following text in 150–180 words.

TEXT:
{text}
"""

CONFIDENCE_PROMPT = """
Evaluate the quality of this summary.

Criteria:
- Coverage of main ideas
- Clarity
- Conciseness
- Structure compliance
- Word count between 150–180

Return ONLY a confidence score between 0 and 1.

Summary:
{summary}
"""
