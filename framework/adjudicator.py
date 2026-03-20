# -*- coding: utf-8 -*-
"""

This module implements the final decision-making component that:
1. Receives observations from multiple cognitive tools
2. Filters noise and identifies core contradictions
3. Weighs evidence and generates final prediction
4. Produces interpretable reasoning summary
"""
import os
import json
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum

from openai import OpenAI

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from framework.config import (
    API_BASE_URL, AVAILABLE_MODELS, DEFAULT_MODEL, 
    DEFAULT_FRAMEWORK_CONFIG, DATASET_CONFIGS
)
from framework.tools import ToolObservation, ToolType
from framework.router import RoutingPlan
from framework.knowledge_base import RetrievalResult


@dataclass
class AdjudicationResult:
    """Final adjudication result"""
    prediction: int  # 0 = harmless, 1 = harmful
    confidence: float
    reasoning_summary: str
    key_evidence: List[str]  # Most important evidence points
    noise_filtered: List[str]  # Evidence deemed irrelevant
    core_contradiction: Optional[str]  # Main semantic incongruity found
    tool_contributions: Dict[str, float]  # How much each tool contributed
    
    def to_dict(self) -> Dict:
        return {
            "prediction": self.prediction,
            "confidence": self.confidence,
            "reasoning_summary": self.reasoning_summary,
            "key_evidence": self.key_evidence,
            "noise_filtered": self.noise_filtered,
            "core_contradiction": self.core_contradiction,
            "tool_contributions": self.tool_contributions
        }
    
    def get_prediction_label(self, dataset_name: str = "FHM") -> str:
        """Get human-readable prediction label"""
        config = DATASET_CONFIGS.get(dataset_name, DATASET_CONFIGS["FHM"])
        if self.prediction == 1:
            return config.get("positive_label", "harmful")
        return config.get("negative_label", "harmless")


class DialecticalAdjudicator:
    """
    Dialectical Consensus Adjudicator
    
    Simulates a "courtroom" process where evidence from multiple tools
    is weighed, contradictions are resolved, and a final verdict is reached.
    """
    
    def __init__(
        self,
        client: Optional[OpenAI] = None,
        model: str = DEFAULT_MODEL,
        config: Optional[Any] = None,
        dataset_name: str = "FHM"
    ):
        if client is None:
            self.client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=API_BASE_URL
            )
        else:
            self.client = client
            
        self.model_config = AVAILABLE_MODELS.get(model, AVAILABLE_MODELS[DEFAULT_MODEL])
        self.config = config or DEFAULT_FRAMEWORK_CONFIG
        self.dataset_name = dataset_name
        self.dataset_config = DATASET_CONFIGS.get(dataset_name, DATASET_CONFIGS["FHM"])
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64"""
        import base64
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    
    def _call_llm(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1500,
        max_retries: int = 3
    ) -> str:
        """Call LLM with optional image, with retry on connection errors"""
        import time as _time
        
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        }]
        
        if image_path and os.path.exists(image_path):
            image_b64 = self._encode_image(image_path)
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
            })
        
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_config.api_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    wait = 2 ** attempt
                    print(f"[Adjudicator] API call failed (attempt {attempt}/{max_retries}): {e}. Retrying in {wait}s...")
                    _time.sleep(wait)
                else:
                    print(f"[Adjudicator] API call failed after {max_retries} attempts: {e}")
        
        return f"Error: {str(last_error)}"
    
    def _format_tool_observations(self, observations: List[ToolObservation]) -> str:
        """Format tool observations for the adjudication prompt"""
        formatted = []
        for i, obs in enumerate(observations, 1):
            formatted.append(f"""
### Tool {i}: {obs.tool_name}
- **Observation**: {obs.observation}
- **Confidence**: {obs.confidence:.2f}
- **Initial Assessment**: {'Suggests harmful' if obs.suggests_harmful else 'Suggests harmless' if obs.suggests_harmful is False else 'Uncertain'}
""")
        return "\n".join(formatted)

    def _split_observations_by_stance(
        self, observations: List[ToolObservation]
    ) -> Tuple[List[Tuple[str, str, float]], List[Tuple[str, str, float]], List[Tuple[str, str, float]]]:
        """
        Split tools into pro (harmful), con (harmless), and neutral.
        Returns: (pro_evidence, con_evidence, neutral) where each item is (tool_name, observation, confidence).
        """
        pro_evidence: List[Tuple[str, str, float]] = []
        con_evidence: List[Tuple[str, str, float]] = []
        neutral: List[Tuple[str, str, float]] = []
        for obs in observations:
            entry = (obs.tool_name, obs.observation, obs.confidence)
            if obs.suggests_harmful is True:
                pro_evidence.append(entry)
            elif obs.suggests_harmful is False:
                con_evidence.append(entry)
            else:
                neutral.append(entry)
        return pro_evidence, con_evidence, neutral

    def _format_debator_evidence(
        self,
        pro_evidence: List[Tuple[str, str, float]],
        con_evidence: List[Tuple[str, str, float]],
        positive_label: str,
        negative_label: str,
        neutral_evidence: Optional[List[Tuple[str, str, float]]] = None
    ) -> str:
        """Format evidence as two debators: Pro (harmful) vs Con (harmless), plus neutral/uncertain observations."""
        lines = []
        lines.append(f"### positive_label}：")
        if pro_evidence:
            for name, obs, conf in pro_evidence:
                lines.append(f"- **{name}** (confidence {conf:.2f}): {obs}")
        else:
            lines.append("- NO TOOLS！")
        lines.append("")
        lines.append(f"### Con – {negative_label}) ：")
        if con_evidence:
            for name, obs, conf in con_evidence:
                lines.append(f"- **{name}** (confidence {conf:.2f}): {obs}")
        else:
            lines.append("- （NO TOOLS！）")
        if neutral_evidence:
            lines.append("")
            lines.append("### Neutral/Uncertain：")
            for name, obs, conf in neutral_evidence:
                lines.append(f"- **{name}** (confidence {conf:.2f}): {obs}")
        return "\n".join(lines)
    
    def _format_reference_context(self, retrieval_result: Optional[RetrievalResult]) -> str:
        """Format retrieved reference cases"""
        if not retrieval_result or not retrieval_result.retrieved_examples:
            return "No reference cases available."
        
        return retrieval_result.get_context_string()

    def _parse_prediction_from_text(self, text: str) -> int:
        """
        Parse prediction from free text without treating negations as positive.
        E.g. 'not misogynistic' / 'harmless' -> 0; 'misogynistic' / 'harmful' -> 1.
        """
        if not text or not text.strip():
            return 0
        lower = text.lower().strip()
        # Explicit negative (harmless) indicators first
        if any(phrase in lower for phrase in [
            "not misogynistic", "not harmful", "not hateful",
            "harmless", "benign", "safe"
        ]):
            return 0
        # Explicit positive (harmful) indicators
        if any(phrase in lower for phrase in [
            "harmful", "hateful", "sarcastic", "misogynistic",
            "1", "true", "yes"
        ]):
            return 1
        return 0

    def _parse_adjudication(self, response: str) -> Tuple[int, float, str, List[str], List[str], Optional[str], Dict[str, float]]:
        """Parse adjudication result from LLM response"""
        prediction = 0
        confidence = 0.5
        reasoning = response
        key_evidence = []
        noise_filtered = []
        core_contradiction = None
        tool_contributions = {}
        
        try:
            # Try to find JSON in response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                
                # Extract prediction: use explicit label string, respect "not misogynistic" / "harmless"
                pred_str = str(data.get("prediction", data.get("verdict", ""))).lower().strip()
                if not pred_str:
                    prediction = self._parse_prediction_from_text(response)
                elif any(phrase in pred_str for phrase in ["not misogynistic", "not harmful", "harmless", "not hateful"]):
                    prediction = 0
                elif any(word in pred_str for word in ["harmful", "1", "true", "yes", "sarcastic", "hateful", "misogynistic"]):
                    prediction = 1
                else:
                    prediction = 0
                
                confidence = float(data.get("confidence", 0.5))
                reasoning = data.get("reasoning_summary", data.get("reasoning", response))
                key_evidence = data.get("key_evidence", [])
                noise_filtered = data.get("noise_filtered", data.get("filtered_evidence", []))
                core_contradiction = data.get("core_contradiction", None)
                tool_contributions = data.get("tool_contributions", {})
                
        except json.JSONDecodeError:
            # Fallback: parse from full text; do NOT treat "not misogynistic" as harmful
            prediction = self._parse_prediction_from_text(response)
            reasoning = response
        
        return prediction, confidence, reasoning, key_evidence, noise_filtered, core_contradiction, tool_contributions
    
    def adjudicate(
        self,
        image_path: str,
        text: str,
        observations: List[ToolObservation],
        routing_plan: Optional[RoutingPlan] = None,
        retrieval_result: Optional[RetrievalResult] = None
    ) -> AdjudicationResult:
        """
        Main adjudication: tools are grouped into Pro (harmful) vs Con (harmless).
        If all tools support the same side, return that side directly; otherwise
        present both sides as debators and let the adjudicator decide.
        """
        positive_label = self.dataset_config.get("positive_label", "harmful")
        negative_label = self.dataset_config.get("negative_label", "harmless")

        pro_evidence, con_evidence, neutral = self._split_observations_by_stance(observations)
        num_pro = len(pro_evidence)
        num_con = len(con_evidence)
        num_clear = num_pro + num_con

        # Unanimous: all tools with clear stance support the same side -> return directly
        if num_clear > 0:
            if num_pro > 0 and num_con == 0:
                return AdjudicationResult(
                    prediction=1,
                    confidence=0.95,
                    reasoning_summary=f"All {num_pro} tool(s) with clear stance support {positive_label}. Unanimous verdict.",
                    key_evidence=[e[1] for e in pro_evidence[:3]],
                    noise_filtered=[],
                    core_contradiction=None,
                    tool_contributions={e[0]: e[2] for e in pro_evidence}
                )
            if num_con > 0 and num_pro == 0:
                return AdjudicationResult(
                    prediction=0,
                    confidence=0.95,
                    reasoning_summary=f"All {num_con} tool(s) with clear stance support {negative_label}. Unanimous verdict.",
                    key_evidence=[e[1] for e in con_evidence[:3]],
                    noise_filtered=[],
                    core_contradiction=None,
                    tool_contributions={e[0]: e[2] for e in con_evidence}
                )

        # Mixed or all neutral: present as two debators and call LLM to adjudicate
        debator_evidence = self._format_debator_evidence(
            pro_evidence, con_evidence, positive_label, negative_label,
            neutral_evidence=neutral
        )
        reference_context = self._format_reference_context(retrieval_result)
        router_reasoning = routing_plan.reasoning if routing_plan else "Not available"

        # Dataset-specific reasoning guidance (rules 6+)
        if self.dataset_name == "HarM":
            reasoning_suffix = f"""- **Dangerous Content & Trivialization**: Health misinformation, trivializing a pandemic, celebrating someone's illness/death, or incitement to violence are strong signals for {positive_label}. Even when wrapped in humor, satire, or critique, content that trivializes or normalizes dangerous attitudes should lean {positive_label}. Ask yourself: would sharing this meme widely normalize harmful attitudes?
- **Political Mockery in HarM**: In this dataset, mocking public figures is NOT automatically {negative_label}. Consider: Does the meme trivialize a serious crisis? Does it celebrate misfortune? Does it spread misinformation even as a joke? Does it dehumanize the target (e.g., comparing them to a virus)? If YES to any of these → lean {positive_label}. Only clearly benign, constructive commentary should be considered {negative_label}.
- **Schadenfreude**: Memes expressing joy, amusement, or satisfaction at someone's suffering (e.g., getting sick, failing) are {positive_label}, even when targeting public figures or non-protected individuals.
- **Weigh argument quality, not tool count**: A single compelling argument for {positive_label} (e.g., "the meme celebrates someone getting COVID") can outweigh multiple surface-level arguments for {negative_label} (e.g., "individual target", "mild severity", "satire"). Consider the deeper context beyond surface labels."""
        else:
            reasoning_suffix = f"""- **Dangerous Content Override**: Health misinformation (e.g., anti-vax, false cures) or incitement to violence is a **signal** for {positive_label}. However, use your judgment: consider broader context and weigh tool evidence; if it is clearly mocking the misinformation itself (rather than spreading it), it may be {negative_label}. If it spreads dangerous ideas under the guise of a joke, lean {positive_label}.
- **Political Critique**: Mocking public figures or government policies is typically legitimate critique ({negative_label}); do not flag it unless it contains hate speech or dangerous disinformation. Weigh tool evidence of intent and context when deciding."""

        prompt = f'''You are the final adjudicator. Multiple cognitive tools have been split into two sides; you must decide the final verdict.

## Target Meme:
Text: "{text}"
[Image attached]

## Evidence by side (Pro vs Con):
{debator_evidence}

## Reference Cases (similar memes):
{reference_context}

## Router's Reasoning:
{router_reasoning}

## Your Task:
- Treat the above as two debators: Pro – {positive_label}) vs Con – {negative_label}).
- Weigh the evidence from both sides. If only one side has evidence, that side usually wins unless the evidence is weak.
- Output your verdict as JSON with "prediction" exactly "{positive_label}" or "{negative_label}".

## Reasoning guidance (you weigh evidence and decide the verdict):
- **Promotion vs. critique**: Judge whether the meme **promotes or reinforces** the kind of content that counts as {positive_label} in this task, or instead **critiques, mocks, or satirizes** it. Content that critiques or mocks such views is often {negative_label}; content that spreads or normalizes them can be {positive_label}. **You** decide based on the evidence—do not apply a fixed rule such as "target is an individual therefore {negative_label}".
- **Who is targeted** is one signal among others; it does not by itself determine the verdict. Consider whether the message promotes/normalizes harmful content or criticizes it.
- **Reference vs. promotion**: Merely referencing a sensitive topic does not make the meme {positive_label}; only when the meme clearly **promotes or reinforces** that content should you lean {positive_label}. Critique or absurdist humor can reference topics and still be {negative_label}.
- **Severity and intent**: Evidence such as "severity: mild", "intent: joke", or "satire" supports {negative_label} only when the meme does **not** promote or normalize harmful content (e.g. health misinformation). Satire that **reinforces or spreads** false or dangerous ideas (e.g. fake cures, dismissing virus severity) should still lean {positive_label}.
- **Mockery of a person or their actions**: Criticizing or mocking a **person** (e.g. a public figure) or their **handling** of a situation is critique, not promotion—lean {negative_label} unless the meme **spreads false information** or **encourages the audience** to dismiss the issue.
{reasoning_suffix}

## Output Format (JSON only):
{{
    "noise_filtering": {{ "filtered_evidence": [], "filtering_reason": "" }},
    "evidence_synthesis": {{
        "for_{positive_label}": ["<key points from pro>"],
        "for_{negative_label}": ["<key points from con>"]
    }},
    "contradiction_resolution": {{ "conflicts": [], "resolution": "" }},
    "core_contradiction": "<main incongruity if {positive_label}, else null or N/A>",
    "tool_contributions": {{ "<tool_name>": <0.0-1.0>, ... }},
    "prediction": "{positive_label}" or "{negative_label}",
    "confidence": <0.0 to 1.0>,
    "reasoning_summary": "<3-5 sentence summary>",
    "key_evidence": ["<point 1>", "<point 2>"]
}}

Rules: Base decision ONLY on evidence. Use "prediction": "{negative_label}" or "prediction": "{positive_label}" (no "not" inside the value).'''

        response = self._call_llm(prompt, image_path)
        (prediction, confidence, reasoning, key_evidence,
         noise_filtered, core_contradiction, tool_contributions) = self._parse_adjudication(response)

        return AdjudicationResult(
            prediction=prediction,
            confidence=confidence,
            reasoning_summary=reasoning,
            key_evidence=key_evidence,
            noise_filtered=noise_filtered,
            core_contradiction=core_contradiction,
            tool_contributions=tool_contributions
        )
    
    def adjudicate_simple(
        self,
        image_path: str,
        text: str,
        observations: List[ToolObservation]
    ) -> AdjudicationResult:
        """
        Simplified adjudication without retrieval: same Pro/Con split and unanimous
        short-circuit as adjudicate(), then debator-style LLM call if needed.
        """
        positive_label = self.dataset_config.get("positive_label", "harmful")
        negative_label = self.dataset_config.get("negative_label", "harmless")
        pro_evidence, con_evidence, neutral = self._split_observations_by_stance(observations)
        num_pro, num_con = len(pro_evidence), len(con_evidence)

        if num_pro + num_con > 0:
            if num_pro > 0 and num_con == 0:
                return AdjudicationResult(
                    prediction=1,
                    confidence=0.95,
                    reasoning_summary=f"All {num_pro} tool(s) support {positive_label}. Unanimous.",
                    key_evidence=[e[1] for e in pro_evidence[:3]],
                    noise_filtered=[],
                    core_contradiction=None,
                    tool_contributions={e[0]: e[2] for e in pro_evidence}
                )
            if num_con > 0 and num_pro == 0:
                return AdjudicationResult(
                    prediction=0,
                    confidence=0.95,
                    reasoning_summary=f"All {num_con} tool(s) support {negative_label}. Unanimous.",
                    key_evidence=[e[1] for e in con_evidence[:3]],
                    noise_filtered=[],
                    core_contradiction=None,
                    tool_contributions={e[0]: e[2] for e in con_evidence}
                )

        debator_evidence = self._format_debator_evidence(
            pro_evidence, con_evidence, positive_label, negative_label
        )
        prompt = f'''You are the final adjudicator. Evidence is split into two sides; decide the verdict.

Text: "{text}"
[Image attached]

## Evidence by side:
{debator_evidence}

## Task:
Verdict: {positive_label} or {negative_label}. Output JSON:
{{ "prediction": "{positive_label}" or "{negative_label}", "confidence": <0.0-1.0>, "reasoning_summary": "<2-3 sentences>", "key_evidence": [], "core_contradiction": "" }}'''

        response = self._call_llm(prompt, image_path)
        (prediction, confidence, reasoning, key_evidence,
         noise_filtered, core_contradiction, tool_contributions) = self._parse_adjudication(response)
        return AdjudicationResult(
            prediction=prediction,
            confidence=confidence,
            reasoning_summary=reasoning,
            key_evidence=key_evidence,
            noise_filtered=[],
            core_contradiction=core_contradiction,
            tool_contributions=tool_contributions or {}
        )
    
    def adjudicate_with_debate(
        self,
        image_path: str,
        text: str,
        observations: List[ToolObservation],
        retrieval_result: Optional[RetrievalResult] = None
    ) -> AdjudicationResult:
        """
        Advanced adjudication with internal debate
        Generates arguments for both sides before making decision
        """
        positive_label = self.dataset_config.get("positive_label", "harmful")
        negative_label = self.dataset_config.get("negative_label", "harmless")
        
        tool_evidence = self._format_tool_observations(observations)
        reference_context = self._format_reference_context(retrieval_result)
        
        # Step 1: Generate arguments for harmful
        harmful_prompt = f'''Based on the evidence, argue that this meme IS {positive_label}.

Text: "{text}"
[Image attached]

Evidence:
{tool_evidence}

Reference cases:
{reference_context}

Make the strongest possible case for why this is {positive_label}. Output 3-5 bullet points.'''
        
        harmful_args = self._call_llm(harmful_prompt, image_path, max_tokens=500)
        
        # Step 2: Generate arguments for harmless
        harmless_prompt = f'''Based on the evidence, argue that this meme is NOT {positive_label} ({negative_label}).

Text: "{text}"
[Image attached]

Evidence:
{tool_evidence}

Reference cases:
{reference_context}

Make the strongest possible case for why this is {negative_label}. Output 3-5 bullet points.'''
        
        harmless_args = self._call_llm(harmless_prompt, image_path, max_tokens=500)
        
        # Step 3: Final judgment considering both sides
        judgment_prompt = f'''As a neutral judge, review both arguments and make a final decision.

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

        response = self._call_llm(judgment_prompt, image_path)
        
        (prediction, confidence, reasoning, key_evidence, 
         noise_filtered, core_contradiction, tool_contributions) = self._parse_adjudication(response)
        
        return AdjudicationResult(
            prediction=prediction,
            confidence=confidence,
            reasoning_summary=reasoning,
            key_evidence=key_evidence,
            noise_filtered=[],
            core_contradiction=core_contradiction,
            tool_contributions={}
        )


class EnsembleAdjudicator:
    """
    Ensemble adjudicator that combines multiple adjudication strategies
    """
    
    def __init__(
        self,
        client: Optional[OpenAI] = None,
        model: str = DEFAULT_MODEL,
        dataset_name: str = "FHM"
    ):
        self.adjudicator = DialecticalAdjudicator(
            client=client, 
            model=model, 
            dataset_name=dataset_name
        )
        self.dataset_name = dataset_name
    
    def adjudicate_ensemble(
        self,
        image_path: str,
        text: str,
        observations: List[ToolObservation],
        retrieval_result: Optional[RetrievalResult] = None
    ) -> AdjudicationResult:
        """
        Run multiple adjudication strategies and combine results
        """
        # Get results from different strategies
        simple_result = self.adjudicator.adjudicate_simple(image_path, text, observations)
        
        full_result = self.adjudicator.adjudicate(
            image_path, text, observations, 
            retrieval_result=retrieval_result
        )
        
        # Combine predictions based on confidence
        predictions = [
            (simple_result.prediction, simple_result.confidence),
            (full_result.prediction, full_result.confidence)
        ]
        
        # Weighted voting
        weighted_sum = sum(pred * conf for pred, conf in predictions)
        total_conf = sum(conf for _, conf in predictions)
        
        if total_conf > 0:
            avg_score = weighted_sum / total_conf
            final_prediction = 1 if avg_score >= 0.5 else 0
            final_confidence = max(simple_result.confidence, full_result.confidence)
        else:
            final_prediction = full_result.prediction
            final_confidence = full_result.confidence
        
        # Use the more detailed result's reasoning
        return AdjudicationResult(
            prediction=final_prediction,
            confidence=final_confidence,
            reasoning_summary=full_result.reasoning_summary,
            key_evidence=full_result.key_evidence,
            noise_filtered=full_result.noise_filtered,
            core_contradiction=full_result.core_contradiction,
            tool_contributions=full_result.tool_contributions
        )


if __name__ == "__main__":
    from framework.tools import CognitiveToolManager, ToolType
    
    # Test the adjudicator
    manager = CognitiveToolManager()
    adjudicator = DialecticalAdjudicator(dataset_name="FHM")
    
    test_image = "data/FHM/images/16395.png"
    test_text = "handjobs sold seperately"
    
    if os.path.exists(test_image):
        print("Getting tool observations...")
        observations = manager.execute_tools(
            [ToolType.SENTIMENT_REVERSAL, ToolType.IMAGE_TEXT_ALIGNER, ToolType.TARGET_IDENTIFIER],
            test_image,
            test_text
        )
        
        print(f"\nGot {len(observations)} observations")
        
        print("\nAdjudicating...")
        result = adjudicator.adjudicate_simple(test_image, test_text, observations)
        
        print(f"\nPrediction: {result.get_prediction_label('FHM')}")
        print(f"Confidence: {result.confidence}")
        print(f"Reasoning: {result.reasoning_summary}")
        print(f"Key Evidence: {result.key_evidence}")
        print(f"Core Contradiction: {result.core_contradiction}")
    else:
        print(f"Test image not found: {test_image}")
