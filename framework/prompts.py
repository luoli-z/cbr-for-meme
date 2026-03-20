# -*- coding: utf-8 -*-
"""
Prompts for Retrieval-Augmented Multi-Tools Framework for Meme Detection

This module contains all prompt templates used by the framework components.
"""

# ===================== Knowledge Base Prompts =====================

KB_EXPLANATION_PROMPT = '''Analyze this meme and explain why it is classified as {label}.

The meme contains the text: "{text}"

Please provide a concise analysis (2-3 sentences) covering:
1. The relationship between image and text
2. Any cultural references or hidden meanings
3. Why this makes it {label}

Output ONLY the analysis, no introduction.'''


KB_IMAGE_DESCRIPTION_PROMPT = '''Describe this image concisely in English, focusing on:
1) Main subjects/objects
2) Actions/expressions
3) Text visible in image
4) Overall mood/tone

Keep under 50 words.'''


# ===================== Tool Prompts =====================

SENTIMENT_REVERSAL_PROMPT = '''Analyze the sentiment contrast between the image and text in this meme.

Text: "{text}"

Your task:
1. Identify the sentiment/emotion in the TEXT (positive, negative, neutral, sarcastic)
2. Identify the mood/atmosphere in the IMAGE (happy, sad, angry, neutral, dark, etc.)
3. Determine if there's a CONTRAST or REVERSAL between them

Output format (JSON):
{{
    "text_sentiment": "<sentiment>",
    "text_sentiment_score": <-1.0 to 1.0>,
    "image_mood": "<mood>",
    "image_mood_score": <-1.0 to 1.0>,
    "has_reversal": <true/false>,
    "reversal_type": "<type of reversal if any>",
    "confidence": <0.0 to 1.0>,
    "explanation": "<2-3 sentence explanation>"
}}'''


IMAGE_TEXT_ALIGNER_PROMPT = '''Analyze the alignment between image content and text in this meme.

Text: "{text}"

Your task:
1. List key ENTITIES mentioned in the text (objects, people, places, concepts)
2. Check if these entities EXIST in the image
3. Check if their ATTRIBUTES match (e.g., text says "expensive car" but image shows old bicycle)
4. Identify any FACTUAL INCONSISTENCIES

Output format (JSON):
{{
    "text_entities": ["<entity1>", "<entity2>", ...],
    "image_entities": ["<entity1>", "<entity2>", ...],
    "matched_entities": ["<entity>", ...],
    "mismatched_entities": [
        {{"entity": "<name>", "text_description": "<desc>", "image_reality": "<desc>"}}
    ],
    "missing_entities": ["<entity>", ...],
    "alignment_score": <0.0 to 1.0>,
    "has_inconsistency": <true/false>,
    "confidence": <0.0 to 1.0>,
    "explanation": "<2-3 sentence explanation>"
}}'''


VISUAL_RHETORIC_PROMPT = '''Analyze the visual rhetoric and stylistic devices in this meme.

Text: "{text}"

Identify any of these visual rhetoric techniques:
- Exaggeration: Overemphasis of features or situations
- Caricature: Distorted representations for satirical effect
- Juxtaposition: Contrasting elements placed together
- Symbolism: Objects representing abstract concepts
- Parody: Imitation of familiar styles for humor
- Ironic imagery: Visual elements contradicting their usual meaning

Output format (JSON):
{{
    "detected_techniques": [
        {{"technique": "<name>", "description": "<how it's used>", "element": "<which part>"}}
    ],
    "visual_manipulation_level": "<none/mild/moderate/strong>",
    "satirical_intent": <true/false>,
    "confidence": <0.0 to 1.0>,
    "explanation": "<2-3 sentence analysis>"
}}'''


MICRO_EXPRESSION_PROMPT = '''Analyze facial expressions in this meme and their relationship to the text.

Text: "{text}"

Your task:
1. Identify if there are FACES in the image
2. Analyze the EXPRESSIONS (happy, sad, angry, surprised, neutral, smirking, etc.)
3. Determine if the expression is APPROPRIATE or INAPPROPRIATE given the text context
4. Note any "inappropriate emotional response" (e.g., smiling at tragedy)

Output format (JSON):
{{
    "faces_detected": <true/false>,
    "num_faces": <number>,
    "expressions": [
        {{"face_id": <num>, "expression": "<type>", "intensity": "<mild/moderate/strong>"}}
    ],
    "text_context_emotion": "<what emotion text suggests>",
    "expression_appropriate": <true/false>,
    "incongruity_type": "<type if inappropriate>",
    "confidence": <0.0 to 1.0>,
    "explanation": "<2-3 sentence analysis>"
}}'''


CULTURE_RETRIEVER_PROMPT = '''Identify cultural references and contextual knowledge in this meme.

Text: "{text}"

Look for:
1. CELEBRITIES or PUBLIC FIGURES (identify who, their typical meme usage)
2. MOVIE/TV SCENES (identify source, original context)
3. INTERNET MEME TEMPLATES (identify template name, typical usage)
4. CULTURAL REFERENCES (historical events, social movements, stereotypes)
5. SLANG or DOG-WHISTLES (coded language with hidden meanings)

Output format (JSON):
{{
    "celebrities": [
        {{"name": "<name>", "typical_meme_usage": "<how usually used>", "confidence": <0-1>}}
    ],
    "scene_source": {{"source": "<movie/show>", "original_context": "<context>"}},
    "meme_template": {{"name": "<template>", "typical_meaning": "<meaning>"}},
    "cultural_references": [
        {{"reference": "<ref>", "meaning": "<meaning>", "potentially_offensive": <true/false>}}
    ],
    "hidden_meanings": ["<meaning1>", ...],
    "requires_context": <true/false>,
    "confidence": <0.0 to 1.0>,
    "explanation": "<2-3 sentence analysis>"
}}'''


PRAGMATIC_IRONY_PROMPT = '''Analyze linguistic markers of irony and sarcasm in this text.

Text: "{text}"

Identify:
1. RHETORICAL QUESTIONS (questions that imply the answer)
2. VERBAL IRONY (saying opposite of what's meant)
3. HYPERBOLE (extreme exaggeration for effect)
4. UNDERSTATEMENT (deliberately minimizing)
5. DOUBLE MEANINGS (puns, wordplay with hidden meanings)
6. OVER-POLITENESS (sarcastic politeness)
7. MOCK SINCERITY (fake earnestness)

Output format (JSON):
{{
    "irony_markers": [
        {{"type": "<marker_type>", "text_segment": "<relevant part>", "implied_meaning": "<what's really meant>"}}
    ],
    "overall_tone": "<sincere/ironic/sarcastic/satirical/ambiguous>",
    "irony_strength": "<none/mild/moderate/strong>",
    "literal_vs_intended": {{"literal": "<surface meaning>", "intended": "<actual meaning>"}},
    "confidence": <0.0 to 1.0>,
    "explanation": "<2-3 sentence analysis>"
}}'''


SCENE_TEXT_OCR_PROMPT = '''Extract and analyze any text embedded within the image itself.

Caption/Overlay Text: "{text}"

Your task:
1. Identify ALL text visible IN the image (signs, banners, labels, watermarks, etc.)
2. Distinguish between CAPTION TEXT and EMBEDDED IMAGE TEXT
3. Analyze if the EMBEDDED TEXT contradicts the CAPTION
4. Note any hidden messages or text manipulations

Output format (JSON):
{{
    "embedded_texts": [
        {{"text": "<text>", "location": "<where in image>", "type": "<sign/label/banner/other>"}}
    ],
    "caption_provided": "<the caption>",
    "text_relationship": "<consistent/contradictory/unrelated/complementary>",
    "contradictions": [
        {{"caption_says": "<x>", "image_says": "<y>", "significance": "<why this matters>"}}
    ],
    "hidden_text": ["<any hidden/subtle text>"],
    "confidence": <0.0 to 1.0>,
    "explanation": "<2-3 sentence analysis>"
}}'''


TARGET_IDENTIFIER_PROMPT = '''Identify the target of this meme and analyze the nature of the targeting.

Text: "{text}"

Determine:
1. WHO/WHAT is being targeted (self, individual, group, institution, concept)
2. The NATURE of targeting (humor, criticism, mockery, attack, support)
3. Whether the target is PROTECTED GROUP (race, gender, religion, disability, etc.)
4. The INTENT (self-deprecation, social commentary, hate speech, satire)

Output format (JSON):
{{
    "target_type": "<self/individual/group/institution/concept/none>",
    "specific_target": "<who or what specifically>",
    "protected_group": <true/false>,
    "protected_category": "<if true, which category>",
    "targeting_nature": "<humor/criticism/mockery/attack/support/ambiguous>",
    "intent": "<self-deprecation/social-commentary/satire/hate/joke/unknown>",
    "severity": "<harmless/mild/moderate/severe>",
    "confidence": <0.0 to 1.0>,
    "explanation": "<2-3 sentence analysis>"
}}'''


# ===================== Router Prompts =====================

ROUTER_PROMPT = '''You are an intelligent router for a meme analysis system. Your task is to select the most relevant cognitive tools to analyze a given meme.

## Available Tools:
{tool_descriptions}

## Current Meme:
Text: "{text}"
[Image attached]

## Reference Cases (similar memes from knowledge base):
{context}

## Your Task:
Based on the current meme and patterns observed in similar cases:
1. Analyze what aspects of this meme need investigation
2. Select {min_tools}-{max_tools} most relevant tools
3. Consider what tools were implicitly needed to analyze the reference cases
4. Prioritize tools based on likely relevance

## Output Format (JSON):
{{
    "analysis": "<brief analysis of what makes this meme potentially harmful/harmless>",
    "patterns_observed": ["<pattern 1 from references>", "<pattern 2>"],
    "selected_tools": ["<tool_key_1>", "<tool_key_2>", ...],
    "tool_justifications": {{
        "<tool_key>": "<why this tool is needed>"
    }},
    "priority_order": ["<most_important_tool>", "<second>", ...],
    "reasoning": "<overall reasoning for tool selection>",
    "confidence": <0.0 to 1.0>
}}'''


ROUTER_SIMPLE_PROMPT = '''Analyze this meme and select the most relevant analysis tools.

Text: "{text}"
[Image attached]

Available tools:
{tool_descriptions}

Select {min_tools}-{max_tools} tools that would be most useful.

Output as JSON:
{{
    "selected_tools": ["tool_key_1", "tool_key_2", ...],
    "reasoning": "<brief reasoning>",
    "confidence": <0.0-1.0>
}}'''


# ===================== Adjudicator Prompts =====================

ADJUDICATOR_PROMPT = '''You are the final adjudicator in a meme analysis system. Your task is to synthesize evidence from multiple cognitive tools and make a final judgment.

## Target Meme:
Text: "{text}"
[Image attached]

## Evidence from Cognitive Tools:
{tool_evidence}

## Reference Cases (similar memes):
{reference_context}

## Router's Reasoning:
{router_reasoning}

## Your Task - Chain of Consensus:

1. **NOISE FILTERING**: Identify which tool observations are NOT relevant or are potentially misleading for this specific meme.

2. **EVIDENCE SYNTHESIS**: Identify the KEY evidence that points toward {positive_label} or {negative_label}.

3. **CONTRADICTION RESOLUTION**: If tools disagree, determine which evidence is more compelling and why.

4. **CORE INCONGRUITY**: Identify the MAIN semantic incongruity or contradiction (if any) that makes this meme {positive_label}.

5. **FINAL VERDICT**: Based on all evidence, is this meme {positive_label} or {negative_label}?

## Output Format (JSON):
{{
    "noise_filtering": {{
        "filtered_evidence": ["<evidence deemed irrelevant>", ...],
        "filtering_reason": "<why these were filtered>"
    }},
    "evidence_synthesis": {{
        "for_{positive_label}": ["<evidence point 1>", "<evidence point 2>"],
        "for_{negative_label}": ["<evidence point 1>", "<evidence point 2>"]
    }},
    "contradiction_resolution": {{
        "conflicts": ["<conflict description>", ...],
        "resolution": "<how conflicts were resolved>"
    }},
    "core_contradiction": "<the main semantic/pragmatic incongruity>",
    "tool_contributions": {{
        "<tool_name>": <contribution_weight 0.0-1.0>,
        ...
    }},
    "prediction": "{positive_label}" or "{negative_label}",
    "confidence": <0.0 to 1.0>,
    "reasoning_summary": "<3-5 sentence summary explaining the final decision>",
    "key_evidence": ["<most important evidence point 1>", "<point 2>", ...]
}}'''


ADJUDICATOR_SIMPLE_PROMPT = '''Analyze this meme based on the tool observations and make a final judgment.

Text: "{text}"
[Image attached]

## Tool Observations:
{tool_evidence}

## Task:
Determine if this meme is {positive_label} or {negative_label}.

Output JSON:
{{
    "prediction": "{positive_label}" or "{negative_label}",
    "confidence": <0.0-1.0>,
    "reasoning_summary": "<2-3 sentences>",
    "key_evidence": ["<point 1>", "<point 2>"],
    "core_contradiction": "<main incongruity if any>"
}}'''


DEBATE_HARMFUL_PROMPT = '''Based on the evidence, argue that this meme IS {positive_label}.

Text: "{text}"
[Image attached]

Evidence:
{tool_evidence}

Reference cases:
{reference_context}

Make the strongest possible case for why this is {positive_label}. Output 3-5 bullet points.'''


DEBATE_HARMLESS_PROMPT = '''Based on the evidence, argue that this meme is NOT {positive_label} ({negative_label}).

Text: "{text}"
[Image attached]

Evidence:
{tool_evidence}

Reference cases:
{reference_context}

Make the strongest possible case for why this is {negative_label}. Output 3-5 bullet points.'''


DEBATE_JUDGMENT_PROMPT = '''As a neutral judge, review both arguments and make a final decision.

Text: "{text}"
[Image attached]

## Arguments for {positive_label}:
{harmful_args}

## Arguments for {negative_label}:
{harmless_args}

## Original Evidence:
{tool_evidence}

Based on weighing both arguments against the evidence, make your final judgment.

Output JSON:
{{
    "winning_side": "{positive_label}" or "{negative_label}",
    "prediction": "{positive_label}" or "{negative_label}",
    "confidence": <0.0-1.0>,
    "reasoning_summary": "<explanation of why one side's arguments were more compelling>",
    "key_evidence": ["<decisive evidence points>"],
    "core_contradiction": "<main issue if {positive_label}>"
}}'''


# ===================== Direct Prediction Prompts (Baselines) =====================

DIRECT_PREDICTION_PROMPT = '''Given this meme with the embedded text: "{text}"

Is this meme {positive_label} or {negative_label}?

Your output should strictly follow the format:
"Thought: [Your analysis of the meme]
Answer: [{positive_label}/{negative_label}]."'''


COT_PREDICTION_PROMPT = '''Given this meme with the embedded text: "{text}"

Analyze step by step:
1. What is shown in the image?
2. What does the text say?
3. Is there any contradiction or irony between image and text?
4. Are there any cultural references or hidden meanings?
5. Who or what is being targeted?
6. Is this {positive_label} or {negative_label}?

Your output should strictly follow the format:
"Step 1: [Image analysis]
Step 2: [Text analysis]
Step 3: [Contradiction analysis]
Step 4: [Cultural analysis]
Step 5: [Target analysis]
Final Answer: [{positive_label}/{negative_label}]."'''
