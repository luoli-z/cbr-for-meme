# -*- coding: utf-8 -*-
"""
Module 2: Multi-View Cognitive Tools

This module implements 8 orthogonal cognitive tools for analyzing memes
from different perspectives:
1. Sentiment Reversal Detector 
2. Fine-grained Image-Text Aligner 
3. Visual Rhetoric Decoder 
4. Micro-Expression Analyzer 
5. Culture Knowledge Retriever 
6. Pragmatic Irony Identifier
7. Scene Text OCR Integrator
8. Target Identification Probe
"""
import os
import base64
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum

from openai import OpenAI

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from framework.config import API_BASE_URL, COGNITIVE_TOOLS, AVAILABLE_MODELS, DEFAULT_MODEL


class ToolType(Enum):
    """Enum for tool types"""
    SENTIMENT_REVERSAL = "sentiment_reversal"
    IMAGE_TEXT_ALIGNER = "image_text_aligner"
    VISUAL_RHETORIC = "visual_rhetoric"
    MICRO_EXPRESSION = "micro_expression"
    CULTURE_RETRIEVER = "culture_retriever"
    PRAGMATIC_IRONY = "pragmatic_irony"
    SCENE_TEXT_OCR = "scene_text_ocr"
    TARGET_IDENTIFIER = "target_identifier"


@dataclass
class ToolObservation:
    """Result from a cognitive tool analysis"""
    tool_name: str
    tool_type: ToolType
    observation: str  # Natural language description of findings
    confidence: float  # 0.0 to 1.0
    suggests_harmful: Optional[bool] = None  # Tool's suggestion if any
    raw_data: Optional[Dict] = None  # Any structured data from the tool
    
    def to_dict(self) -> Dict:
        return {
            "tool_name": self.tool_name,
            "tool_type": self.tool_type.value,
            "observation": self.observation,
            "confidence": self.confidence,
            "suggests_harmful": self.suggests_harmful,
            "raw_data": self.raw_data
        }


class BaseCognitiveTool(ABC):
    """Base class for all cognitive tools"""
    
    def __init__(
        self,
        client: Optional[OpenAI] = None,
        model: str = DEFAULT_MODEL,
        dataset_name: str = "FHM"
    ):
        if client is None:
            self.client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=API_BASE_URL
            )
        else:
            self.client = client
            
        self.dataset_name = dataset_name
        self.model_config = AVAILABLE_MODELS.get(model, AVAILABLE_MODELS[DEFAULT_MODEL])
        self.tool_config = COGNITIVE_TOOLS.get(self.tool_type.value)
    
    @property
    @abstractmethod
    def tool_type(self) -> ToolType:
        """Return the tool type"""
        pass
    
    @property
    def name(self) -> str:
        return self.tool_config.name if self.tool_config else self.tool_type.value
    
    @property
    def description(self) -> str:
        return self.tool_config.description if self.tool_config else ""
    
    @property
    def requires_vision(self) -> bool:
        return self.tool_config.requires_vision if self.tool_config else True
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    
    def _call_llm(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        max_retries: int = 3
    ) -> str:
        """Call LLM with optional image input, with retry on connection errors"""
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
                    wait = 2 ** attempt  # exponential backoff: 2s, 4s, 8s
                    print(f"[{self.name}] API call failed (attempt {attempt}/{max_retries}): {e}. Retrying in {wait}s...")
                    _time.sleep(wait)
                else:
                    print(f"[{self.name}] API call failed after {max_retries} attempts: {e}")
        
        return f"Error: {str(last_error)}"
    
    @abstractmethod
    def analyze(self, image_path: str, text: str) -> ToolObservation:
        """
        Analyze the meme and return observations
        
        Args:
            image_path: Path to the meme image
            text: Text content of the meme
            
        Returns:
            ToolObservation with analysis results
        """
        pass


class SentimentReversalDetector(BaseCognitiveTool):
    """
    Tool 1: Sentiment Reversal Detector
    
    Analyzes whether the sentiment expressed in text contradicts
    the emotional tone conveyed by the image.
    """
    
    @property
    def tool_type(self) -> ToolType:
        return ToolType.SENTIMENT_REVERSAL
    
    def analyze(self, image_path: str, text: str) -> ToolObservation:
        prompt = f'''Analyze the sentiment contrast between the image and text in this meme.

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

        response = self._call_llm(prompt, image_path)
        
        # Parse response
        try:
            # Try to extract JSON from response
            import json
            import re
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                has_reversal = data.get("has_reversal", False)
                confidence = float(data.get("confidence", 0.5))
                explanation = data.get("explanation", response)
                
                observation = f"Text sentiment: {data.get('text_sentiment', 'unknown')} ({data.get('text_sentiment_score', 0)}). "
                observation += f"Image mood: {data.get('image_mood', 'unknown')} ({data.get('image_mood_score', 0)}). "
                observation += explanation
                
                return ToolObservation(
                    tool_name=self.name,
                    tool_type=self.tool_type,
                    observation=observation,
                    confidence=confidence,
                    suggests_harmful=has_reversal,
                    raw_data=data
                )
        except:
            pass
        
        # Fallback if parsing fails
        return ToolObservation(
            tool_name=self.name,
            tool_type=self.tool_type,
            observation=response,
            confidence=0.5,
            suggests_harmful=None
        )


class ImageTextAligner(BaseCognitiveTool):
    """
    Tool 2: Fine-grained Image-Text Aligner 
    
    Checks whether entities mentioned in text exist in the image
    and whether their attributes match.
    """
    
    @property
    def tool_type(self) -> ToolType:
        return ToolType.IMAGE_TEXT_ALIGNER
    
    def analyze(self, image_path: str, text: str) -> ToolObservation:
        prompt = f'''Analyze the alignment between image content and text in this meme.

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

        response = self._call_llm(prompt, image_path)
        
        try:
            import json
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                has_inconsistency = data.get("has_inconsistency", False)
                confidence = float(data.get("confidence", 0.5))
                
                observation = f"Text mentions: {', '.join(data.get('text_entities', []))}. "
                if data.get("mismatched_entities"):
                    mismatches = [f"{m['entity']} (text: {m['text_description']}, image: {m['image_reality']})" 
                                 for m in data.get("mismatched_entities", [])]
                    observation += f"Mismatches found: {'; '.join(mismatches)}. "
                observation += data.get("explanation", "")
                
                return ToolObservation(
                    tool_name=self.name,
                    tool_type=self.tool_type,
                    observation=observation,
                    confidence=confidence,
                    suggests_harmful=has_inconsistency,
                    raw_data=data
                )
        except:
            pass
        
        return ToolObservation(
            tool_name=self.name,
            tool_type=self.tool_type,
            observation=response,
            confidence=0.5,
            suggests_harmful=None
        )


class VisualRhetoricDecoder(BaseCognitiveTool):
    """
    Tool 3: Visual Rhetoric Decoder
    
    Identifies visual rhetorical devices like exaggeration, caricature,
    and juxtaposition.
    """
    
    @property
    def tool_type(self) -> ToolType:
        return ToolType.VISUAL_RHETORIC
    
    def analyze(self, image_path: str, text: str) -> ToolObservation:
        prompt = f'''Analyze the visual rhetoric and stylistic devices in this meme.

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
    "explanation": "<2-3 sentence analysis>",
    "suggests_harmful": <true/false — based on your visual rhetoric analysis, does this meme suggest harmful content? Consider: do the visual devices dehumanize someone, trivialize a serious issue, promote stereotypes, celebrate suffering, or spread dangerous ideas? If YES → true. If the visual rhetoric is used for benign humor, constructive critique, or neutral commentary → false.>
}}'''

        response = self._call_llm(prompt, image_path)
        
        try:
            import json
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                techniques = data.get("detected_techniques", [])
                satirical = data.get("satirical_intent", False)
                confidence = float(data.get("confidence", 0.5))
                
                if techniques:
                    tech_list = [f"{t['technique']}: {t['description']}" for t in techniques]
                    observation = f"Detected rhetoric: {'; '.join(tech_list)}. "
                else:
                    observation = "No significant visual rhetoric detected. "
                observation += data.get("explanation", "")
                
                # Use LLM's own judgment for suggests_harmful
                suggests_harmful_llm = data.get("suggests_harmful")
                if suggests_harmful_llm is not None:
                    suggests_harmful_vr = bool(suggests_harmful_llm)
                else:
                    # Fallback: rule-based
                    suggests_harmful_vr = satirical and len(techniques) > 0
                
                return ToolObservation(
                    tool_name=self.name,
                    tool_type=self.tool_type,
                    observation=observation,
                    confidence=confidence,
                    suggests_harmful=suggests_harmful_vr,
                    raw_data=data
                )
        except:
            pass
        
        return ToolObservation(
            tool_name=self.name,
            tool_type=self.tool_type,
            observation=response,
            confidence=0.5,
            suggests_harmful=None
        )


class MicroExpressionAnalyzer(BaseCognitiveTool):
    """
    Tool 4: Micro-Expression Analyzer
    
    Analyzes facial expressions in relation to textual context
    to detect inappropriate emotional responses.
    """
    
    @property
    def tool_type(self) -> ToolType:
        return ToolType.MICRO_EXPRESSION
    
    def analyze(self, image_path: str, text: str) -> ToolObservation:
        prompt = f'''Analyze facial expressions in this meme and their relationship to the text.

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

        response = self._call_llm(prompt, image_path)
        
        try:
            import json
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                faces_detected = data.get("faces_detected", False)
                appropriate = data.get("expression_appropriate", True)
                confidence = float(data.get("confidence", 0.5))
                
                if faces_detected:
                    expressions = data.get("expressions", [])
                    expr_list = [f"{e['expression']} ({e['intensity']})" for e in expressions]
                    observation = f"Detected {data.get('num_faces', 0)} face(s) with expressions: {', '.join(expr_list)}. "
                    if not appropriate:
                        observation += f"Expression-context incongruity: {data.get('incongruity_type', 'detected')}. "
                else:
                    observation = "No faces detected in the image. "
                observation += data.get("explanation", "")
                
                return ToolObservation(
                    tool_name=self.name,
                    tool_type=self.tool_type,
                    observation=observation,
                    confidence=confidence if faces_detected else 0.3,
                    suggests_harmful=not appropriate if faces_detected else None,
                    raw_data=data
                )
        except:
            pass
        
        return ToolObservation(
            tool_name=self.name,
            tool_type=self.tool_type,
            observation=response,
            confidence=0.5,
            suggests_harmful=None
        )


class CultureRetriever(BaseCognitiveTool):
    """
    Tool 5: Culture Knowledge Retriever
    
    Identifies cultural references, celebrities, movie scenes,
    and their associated meanings.
    """
    
    @property
    def tool_type(self) -> ToolType:
        return ToolType.CULTURE_RETRIEVER
    
    def analyze(self, image_path: str, text: str) -> ToolObservation:
        # Build dataset-specific guidance
        if self.dataset_name == "HarM":
            harm_extra = """
IMPORTANT: Satire or critique does NOT automatically mean the meme is harmless.
Even satirical memes can be harmful when they:
- Trivialize serious issues (e.g., pandemic, illness, death)
- Celebrate or mock someone's misfortune or illness
- Spread misinformation even through humor or jokes
- Dehumanize individuals (e.g., comparing a person to a virus or disease)
Consider these aspects carefully in your analysis and explanation.

"""
        else:
            harm_extra = ""

        prompt = f'''Identify cultural references and contextual knowledge in this meme.

Text: "{text}"

Look for:
1. CELEBRITIES or PUBLIC FIGURES (identify who, their typical meme usage)
2. MOVIE/TV SCENES (identify source, original context)
3. INTERNET MEME TEMPLATES (identify template name, typical usage)
4. CULTURAL REFERENCES (historical events, social movements, stereotypes)
5. SLANG or DOG-WHISTLES (coded language with hidden meanings)

For each sensitive cultural reference (stereotypes, race, gender, religion, disability, etc.), say whether the meme REINFORCES that stereotype/topic or CRITIQUES/MOCKS it:
- reinforcing: the meme endorses or perpetuates the stereotype
- critiquing_mocking: the meme makes fun of the stereotype or the people who hold it (satire)
- neutral: factual or neutral reference
- unclear: cannot tell

{harm_extra}Output format (JSON):
{{
    "celebrities": [
        {{"name": "<name>", "typical_meme_usage": "<how usually used>", "confidence": <0-1>}}
    ],
    "scene_source": {{"source": "<movie/show>", "original_context": "<context>"}},
    "meme_template": {{"name": "<template>", "typical_meaning": "<meaning>"}},
    "cultural_references": [
        {{"reference": "<ref>", "meaning": "<meaning>", "potentially_offensive": <true/false>, "stereotype_role": "<reinforcing / critiquing_mocking / neutral / unclear>"}}
    ],
    "hidden_meanings": ["<meaning1>", ...],
    "requires_context": <true/false>,
    "confidence": <0.0 to 1.0>,
    "explanation": "<2-3 sentence analysis>",
    "suggests_harmful": <true/false — based on your cultural analysis, does this meme suggest harmful content? Consider: does the meme reinforce stereotypes, trivialize serious issues, celebrate misfortune, spread misinformation, or dehumanize individuals? If YES → true. If the meme critiques, mocks bigotry, or is benign commentary → false.>
}}'''

        response = self._call_llm(prompt, image_path)
        
        try:
            import json
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                confidence = float(data.get("confidence", 0.5))
                
                observations = []
                
                if data.get("celebrities"):
                    celeb_info = [f"{c['name']} (typically: {c['typical_meme_usage']})" for c in data["celebrities"]]
                    observations.append(f"Celebrities: {'; '.join(celeb_info)}")
                
                if data.get("meme_template", {}).get("name"):
                    observations.append(f"Meme template: {data['meme_template']['name']} - {data['meme_template'].get('typical_meaning', '')}")
                
                if data.get("cultural_references"):
                    refs = []
                    for r in data["cultural_references"]:
                        role = r.get("stereotype_role", "")
                        refs.append(f"{r['reference']}: {r['meaning']}" + (f" (role: {role})" if role else ""))
                    observations.append(f"Cultural refs: {'; '.join(refs)}")
                
                if data.get("hidden_meanings"):
                    observations.append(f"Hidden meanings: {', '.join(data['hidden_meanings'])}")
                
                observation = " | ".join(observations) if observations else "No significant cultural references detected."
                observation += " " + data.get("explanation", "")
                
                # Use LLM's own judgment for suggests_harmful
                suggests_harmful_llm = data.get("suggests_harmful")
                if suggests_harmful_llm is not None:
                    suggests_harmful = bool(suggests_harmful_llm)
                else:
                    # Fallback: rule-based determination
                    suggests_harmful = False
                    for r in data.get("cultural_references", []):
                        if not r.get("potentially_offensive", False):
                            continue
                        role = (r.get("stereotype_role") or "unclear").lower().replace(" ", "_")
                        if role in ("critiquing_mocking", "neutral"):
                            continue
                        suggests_harmful = True
                        break
                    if not suggests_harmful and data.get("hidden_meanings"):
                        suggests_harmful = True
                
                return ToolObservation(
                    tool_name=self.name,
                    tool_type=self.tool_type,
                    observation=observation,
                    confidence=confidence,
                    suggests_harmful=suggests_harmful,
                    raw_data=data
                )
        except:
            pass
        
        return ToolObservation(
            tool_name=self.name,
            tool_type=self.tool_type,
            observation=response,
            confidence=0.5,
            suggests_harmful=None
        )


class PragmaticIronyIdentifier(BaseCognitiveTool):
    """
    Tool 6: Pragmatic Irony Identifier
    
    Focuses on text-only analysis to detect linguistic irony markers
    like rhetorical questions, sarcasm, and double meanings.
    """
    
    @property
    def tool_type(self) -> ToolType:
        return ToolType.PRAGMATIC_IRONY
    
    @property
    def requires_vision(self) -> bool:
        return False  # This tool only analyzes text
    
    def analyze(self, image_path: str, text: str) -> ToolObservation:
        prompt = f'''Analyze linguistic markers of irony and sarcasm in this text.

Text: "{text}"

Identify:
1. RHETORICAL QUESTIONS (questions that imply the answer)
2. VERBAL IRONY (saying opposite of what's meant)
3. HYPERBOLE (extreme exaggeration for effect)
4. UNDERSTATEMENT (deliberately minimizing)
5. DOUBLE MEANINGS (puns, wordplay with hidden meanings)
6. OVER-POLITENESS (sarcastic politeness)
7. MOCK SINCERITY (fake earnestness)

Then determine the TARGET of the irony:
- Who or what is the irony directed AT? If the speaker is mocking a stereotype or hateful view itself (e.g. "I'm not racist but..."), that is mocking_hate. If the speaker is mocking a protected group or individuals, that is mocking_group.
- irony_target: "stereotype_or_hate_itself" | "specific_group_or_person" | "unclear"
- irony_direction: "mocking_hate" (critiquing bigotry) | "mocking_group" (attacking a group) | "unclear"

Output format (JSON):
{{
    "irony_markers": [
        {{"type": "<marker_type>", "text_segment": "<relevant part>", "implied_meaning": "<what's really meant>"}}
    ],
    "overall_tone": "<sincere/ironic/sarcastic/satirical/ambiguous>",
    "irony_strength": "<none/mild/moderate/strong>",
    "literal_vs_intended": {{"literal": "<surface meaning>", "intended": "<actual meaning>"}},
    "irony_target": "<stereotype_or_hate_itself / specific_group_or_person / unclear>",
    "irony_direction": "<mocking_hate / mocking_group / unclear>",
    "confidence": <0.0 to 1.0>,
    "explanation": "<2-3 sentence analysis>",
    "suggests_harmful": <true/false — based on your irony analysis, does this meme suggest harmful content? Consider: does the irony trivialize serious issues, celebrate misfortune, spread misinformation through humor, mock vulnerable groups, or dehumanize? If YES → true. If the irony critiques bigotry, mocks the powerful for their actions, or is benign wordplay → false.>
}}'''

        # Note: This tool doesn't use the image
        response = self._call_llm(prompt, image_path=None)
        
        try:
            import json
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                markers = data.get("irony_markers", [])
                tone = data.get("overall_tone", "ambiguous")
                strength = data.get("irony_strength", "none")
                irony_target = (data.get("irony_target") or "unclear").lower().replace(" ", "_")
                irony_direction = (data.get("irony_direction") or "unclear").lower().replace(" ", "_")
                confidence = float(data.get("confidence", 0.5))
                
                if markers:
                    marker_list = [f"{m['type']}: '{m['text_segment']}' → {m['implied_meaning']}" for m in markers]
                    observation = f"Irony markers: {'; '.join(marker_list)}. "
                else:
                    observation = "No strong irony markers detected. "
                
                observation += f"Overall tone: {tone} (strength: {strength}). "
                observation += f"Irony target: {data.get('irony_target', 'unclear')}; direction: {data.get('irony_direction', 'unclear')}. "
                
                if data.get("literal_vs_intended", {}).get("intended"):
                    lvi = data["literal_vs_intended"]
                    observation += f"Surface: '{lvi.get('literal', '')}' → Intent: '{lvi.get('intended', '')}'. "
                
                observation += data.get("explanation", "")
                
                # Use LLM's own judgment for suggests_harmful
                suggests_harmful_llm = data.get("suggests_harmful")
                if suggests_harmful_llm is not None:
                    suggests_harmful = bool(suggests_harmful_llm)
                else:
                    # Fallback: rule-based determination
                    tone_harmful = tone in ["ironic", "sarcastic", "satirical"] and strength in ["moderate", "strong"]
                    mocking_hate = (
                        "stereotype_or_hate_itself" in irony_target or "mocking_hate" in irony_direction
                    )
                    suggests_harmful = tone_harmful and not mocking_hate
                
                return ToolObservation(
                    tool_name=self.name,
                    tool_type=self.tool_type,
                    observation=observation,
                    confidence=confidence,
                    suggests_harmful=suggests_harmful,
                    raw_data=data
                )
        except:
            pass
        
        return ToolObservation(
            tool_name=self.name,
            tool_type=self.tool_type,
            observation=response,
            confidence=0.5,
            suggests_harmful=None
        )


class SceneTextOCR(BaseCognitiveTool):
    """
    Tool 7: Scene Text OCR Integrator
    
    Extracts embedded text within images and analyzes contradictions
    with the caption.
    """
    
    @property
    def tool_type(self) -> ToolType:
        return ToolType.SCENE_TEXT_OCR
    
    def analyze(self, image_path: str, text: str) -> ToolObservation:
        prompt = f'''Extract and analyze any text embedded within the image itself.

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

        response = self._call_llm(prompt, image_path)
        
        try:
            import json
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                embedded = data.get("embedded_texts", [])
                relationship = data.get("text_relationship", "unrelated")
                contradictions = data.get("contradictions", [])
                confidence = float(data.get("confidence", 0.5))
                
                if embedded:
                    text_list = [f"'{t['text']}' ({t.get('location', 'unknown')})" for t in embedded]
                    observation = f"Embedded text found: {'; '.join(text_list)}. "
                else:
                    observation = "No embedded text detected. "
                
                observation += f"Relationship to caption: {relationship}. "
                
                if contradictions:
                    contr_list = [f"Caption: '{c['caption_says']}' vs Image: '{c['image_says']}'" for c in contradictions]
                    observation += f"Contradictions: {'; '.join(contr_list)}. "
                
                observation += data.get("explanation", "")
                
                suggests_harmful = relationship == "contradictory" or len(contradictions) > 0
                
                return ToolObservation(
                    tool_name=self.name,
                    tool_type=self.tool_type,
                    observation=observation,
                    confidence=confidence if embedded else 0.3,
                    suggests_harmful=suggests_harmful if embedded else None,
                    raw_data=data
                )
        except:
            pass
        
        return ToolObservation(
            tool_name=self.name,
            tool_type=self.tool_type,
            observation=response,
            confidence=0.5,
            suggests_harmful=None
        )


class TargetIdentifier(BaseCognitiveTool):
    """
    Tool 8: Target Identification Probe
    
    Identifies who or what the meme is targeting and the nature
    of that targeting.
    """
    
    @property
    def tool_type(self) -> ToolType:
        return ToolType.TARGET_IDENTIFIER
    
    def analyze(self, image_path: str, text: str) -> ToolObservation:
        prompt = f'''Identify the target of this meme and analyze the nature of the targeting.

Text: "{text}"

Determine:
1. WHO/WHAT is being targeted (self, individual, group, institution, concept)
2. The NATURE of targeting (humor, criticism, mockery, attack, support)
3. Whether the target is a PROTECTED GROUP (race, gender, religion, disability, ethnicity, etc.) vs INDIVIDUAL/NON-PROTECTED (e.g. ex, friend, celebrity as person, concept)
4. The INTENT (self-deprecation, social commentary, hate speech, satire, joke)
5. Severity of potential harm

Output format (JSON):
{{
    "target_type": "<self/individual/group/institution/concept/none>",
    "specific_target": "<who or what specifically>",
    "protected_group": <true only if targeting a protected group as a group, false for individual or non-protected>,
    "target_class": "<protected_group / individual_or_non_protected>",
    "protected_category": "<if protected_group true: race/gender/religion/disability/etc.>",
    "targeting_nature": "<humor/criticism/mockery/attack/support/ambiguous>",
    "intent": "<self-deprecation/social-commentary/satire/hate/joke/unknown>",
    "severity": "<harmless/mild/moderate/severe>",
    "confidence": <0.0 to 1.0>,
    "explanation": "<2-3 sentence analysis>",
    "suggests_harmful": <true/false — based on your target analysis, does this meme suggest harmful content? Consider: does it attack protected groups, dehumanize individuals (e.g. comparing a person to a disease), celebrate someone's illness/death, trivialize serious crises, or spread hate? If YES → true. If the meme is benign humor, constructive critique, or mild political satire without dehumanization → false.>
}}'''

        response = self._call_llm(prompt, image_path)
        
        try:
            import json
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                target_type = data.get("target_type", "unknown")
                specific_target = data.get("specific_target", "unknown")
                protected = data.get("protected_group", False)
                target_class = data.get("target_class", "protected_group" if protected else "individual_or_non_protected")
                nature = data.get("targeting_nature", "ambiguous")
                severity = data.get("severity", "mild")
                intent = data.get("intent", "unknown")
                confidence = float(data.get("confidence", 0.5))
                
                observation = f"Target: {specific_target} ({target_type}). "
                observation += f"Target class: {target_class}. "
                observation += f"Nature: {nature}, Intent: {intent}. "
                
                if protected:
                    observation += f"⚠️ Targets protected category: {data.get('protected_category', 'unspecified')}. "
                
                observation += f"Severity: {severity}. "
                observation += data.get("explanation", "")
                
                # Use LLM's own judgment for suggests_harmful
                suggests_harmful_llm = data.get("suggests_harmful")
                if suggests_harmful_llm is not None:
                    suggests_harmful = bool(suggests_harmful_llm)
                else:
                    # Fallback: rule-based determination
                    if intent == "hate" or severity in ["moderate", "severe"]:
                        suggests_harmful = True
                    elif not protected:
                        suggests_harmful = False
                    else:
                        suggests_harmful = (
                            nature in ["mockery", "attack"] and
                            not (severity == "mild" and intent == "joke")
                        )
                
                return ToolObservation(
                    tool_name=self.name,
                    tool_type=self.tool_type,
                    observation=observation,
                    confidence=confidence,
                    suggests_harmful=suggests_harmful,
                    raw_data=data
                )
        except:
            pass
        
        return ToolObservation(
            tool_name=self.name,
            tool_type=self.tool_type,
            observation=response,
            confidence=0.5,
            suggests_harmful=None
        )


# ===================== Tool Manager =====================

class CognitiveToolManager:
    """
    Manager class for all cognitive tools
    Handles tool instantiation, selection, and parallel execution
    """
    
    # Map tool types to their classes
    TOOL_CLASSES = {
        ToolType.SENTIMENT_REVERSAL: SentimentReversalDetector,
        ToolType.IMAGE_TEXT_ALIGNER: ImageTextAligner,
        ToolType.VISUAL_RHETORIC: VisualRhetoricDecoder,
        ToolType.MICRO_EXPRESSION: MicroExpressionAnalyzer,
        ToolType.CULTURE_RETRIEVER: CultureRetriever,
        ToolType.PRAGMATIC_IRONY: PragmaticIronyIdentifier,
        ToolType.SCENE_TEXT_OCR: SceneTextOCR,
        ToolType.TARGET_IDENTIFIER: TargetIdentifier,
    }
    
    def __init__(
        self,
        client: Optional[OpenAI] = None,
        model: str = DEFAULT_MODEL,
        enabled_tools: Optional[List[ToolType]] = None,
        dataset_name: str = "FHM"
    ):
        if client is None:
            self.client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=API_BASE_URL
            )
        else:
            self.client = client
            
        self.model = model
        self.dataset_name = dataset_name
        
        # Initialize all tools
        self.tools: Dict[ToolType, BaseCognitiveTool] = {}
        
        enabled = enabled_tools if enabled_tools else list(ToolType)
        for tool_type in enabled:
            tool_class = self.TOOL_CLASSES.get(tool_type)
            if tool_class:
                self.tools[tool_type] = tool_class(client=self.client, model=model, dataset_name=dataset_name)
    
    def get_tool(self, tool_type: ToolType) -> Optional[BaseCognitiveTool]:
        """Get a specific tool by type"""
        return self.tools.get(tool_type)
    
    def get_all_tools(self) -> List[BaseCognitiveTool]:
        """Get all enabled tools"""
        return list(self.tools.values())
    
    def get_tool_descriptions(self) -> Dict[str, str]:
        """Get descriptions of all tools for router"""
        return {
            tool_type.value: tool.description 
            for tool_type, tool in self.tools.items()
        }
    
    def execute_tool(
        self,
        tool_type: ToolType,
        image_path: str,
        text: str
    ) -> Optional[ToolObservation]:
        """Execute a single tool"""
        tool = self.tools.get(tool_type)
        if tool:
            return tool.analyze(image_path, text)
        return None
    
    def execute_tools(
        self,
        tool_types: List[ToolType],
        image_path: str,
        text: str,
        parallel: bool = True,
        max_workers: int = 4
    ) -> List[ToolObservation]:
        """
        Execute multiple tools, optionally in parallel
        
        Args:
            tool_types: List of tool types to execute
            image_path: Path to meme image
            text: Meme text
            parallel: Whether to execute in parallel
            max_workers: Max parallel workers
            
        Returns:
            List of ToolObservations
        """
        if not parallel:
            # Sequential execution
            return [
                self.execute_tool(tt, image_path, text)
                for tt in tool_types
                if self.tools.get(tt)
            ]
        
        # Parallel execution
        observations = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_tool = {
                executor.submit(self.execute_tool, tt, image_path, text): tt
                for tt in tool_types
                if self.tools.get(tt)
            }
            
            for future in as_completed(future_to_tool):
                tool_type = future_to_tool[future]
                try:
                    observation = future.result()
                    if observation:
                        observations.append(observation)
                except Exception as e:
                    print(f"Tool {tool_type.value} failed: {e}")
        
        return observations
    
    def execute_all(
        self,
        image_path: str,
        text: str,
        parallel: bool = True,
        max_workers: int = 4
    ) -> List[ToolObservation]:
        """Execute all enabled tools"""
        return self.execute_tools(
            list(self.tools.keys()),
            image_path,
            text,
            parallel,
            max_workers
        )


if __name__ == "__main__":
    # Test the tools
    manager = CognitiveToolManager()
    
    test_image = "data/FHM/images/16395.png"
    test_text = "handjobs sold seperately"
    
    if os.path.exists(test_image):
        print("Testing individual tools...")
        
        # Test sentiment reversal
        obs = manager.execute_tool(ToolType.SENTIMENT_REVERSAL, test_image, test_text)
        print(f"\n{obs.tool_name}:")
        print(f"  Observation: {obs.observation}")
        print(f"  Confidence: {obs.confidence}")
        print(f"  Suggests harmful: {obs.suggests_harmful}")
    else:
        print(f"Test image not found: {test_image}")
