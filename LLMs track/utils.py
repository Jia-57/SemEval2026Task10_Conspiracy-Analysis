import json

TASK_DESC = """You are an expert in psycholinguistics and conspiracy theory detection. 
Your task is to analyze Reddit comments to:
1. Identify if they indicate a belief in a conspiracy theory (Binary Classification).
2. Extract psycholinguistic markers (Actor, Action, Victim, Effect, Evidence) supporting this (Span Extraction)."""

UNIFIED_INSTRUCTIONS = """
### 1. Core Definition
A conspiracy theory is a causal narrative explaining a significant event as the secret result of powerful, deceptive actors working together for malevolent goals, rather than as a random occurrence.

### 2. Marker Definitions (Subtask 1)
You must extract the following narrative elements based on the logic above:
* **Actor**: The deceptive, coordinated individuals or groups perceived as working together to initiate the plot.
* **Action**: The secret, malevolent activities performed by the Actor to achieve their goal.
* **Victim**: The individual or population that is intentionally disenfranchised or harmed by the conspiracy.
* **Effect**: The negative consequences or the specific goal the actors aim to achieve.
* **Evidence**: Any arguments, documents, or data used to provide a closed-ended causal explanation.

### 3. Classification Guidance (Subtask 2)
* **Binary Decision**: You must provide a binary "Yes" or "No" label. 
* **The "Yes" Threshold**: Classify as "Yes" only if the text conveys a belief in a conspiracy narrative.
* **The "No" Default**: Default to "No" for debunking, neutral text, or lack of strong indicators.
"""

OUTPUT_FORMAT = """
### 4. Output Format
You must respond with a strict JSON object:
{
    "reasoning": "Brief analysis of the narrative logic.",
    "conspiracy": "Yes" or "No",
    "markers": [
        {"text": "exact phrase from text", "type": "Type"}
    ]
}
"""

def build_prompt(stage, text, few_shot_examples=None):
    """Assembles the unified prompt structure for different experimental stages (Zero-shot, Few-shot, or Fine-tuning)[cite: 91, 101]."""
    system_content = f"{TASK_DESC}\n{UNIFIED_INSTRUCTIONS}\n{OUTPUT_FORMAT}"
    user_content = ""

    # Dynamic insertion of reference samples for Stage B (Few-shot)
    if stage == "B" and few_shot_examples:
        user_content += "### Reference Examples (Learn from these):\n"
        for i, ex in enumerate(few_shot_examples[:5]):
            ex_output = {
                "reasoning": "Analysis of markers and narrative belief.",
                "conspiracy": ex.get('conspiracy', 'No'),
                "markers": [{"text": m['text'], "type": m['type']} for m in ex.get('markers', [])]
            }
            if ex_output["conspiracy"] == "Can't tell":
                ex_output["conspiracy"] = "No"

            user_content += f"Example {i + 1}:\nInput: {ex.get('text', '')}\nOutput: {json.dumps(ex_output, ensure_ascii=False)}\n\n"

    user_content += f"### Target Input Text to Analyze:\n{text}"

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]

def find_offsets(original_text, markers_pred):
    """Maps extracted marker text strings back to their precise start and end indices in the original text."""
    refined_markers = []

    for m in markers_pred:
        marker_text = m.get('text', '').strip()
        marker_type = m.get('type', 'Evidence')
        if not marker_text: continue

        # Priority 1: Exact match as provided by the LLM
        start = original_text.find(marker_text)

        # Priority 2: Match after stripping hallucinatory punctuation
        if start == -1:
            clean_marker = marker_text.strip('.,;!?"\'')
            if clean_marker:
                start = original_text.find(clean_marker)
                if start != -1:
                    marker_text = clean_marker

        # Priority 3: Case-insensitive match
        if start == -1:
            start = original_text.lower().find(marker_text.lower())

        if start != -1:
            end = start + len(marker_text)
            actual_span = original_text[start:end]

            refined_markers.append({
                "type": marker_type,
                "startIndex": start,
                "endIndex": end,
                "text": actual_span
            })

    return refined_markers