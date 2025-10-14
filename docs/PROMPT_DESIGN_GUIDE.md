# Prompt Design Guide

Complete guide for designing and optimizing prompts for the 3-model LLM ensemble (Qwen, Mistral, Llama).

## Table of Contents

1. [Overview](#overview)
2. [Model-Specific Templates](#model-specific-templates)
3. [Mode-Based Prompt Strategies](#mode-based-prompt-strategies)
4. [Prompt Engineering Best Practices](#prompt-engineering-best-practices)
5. [Korean/English Handling](#koreanenglish-handling)
6. [Examples](#examples)

---

## Overview

### 3-Model Pipeline

```
User Input → Router → Mode Selection → LLM Processing → Validation → Output
                          ↓
                  Fast / Balanced / Creative
                          ↓
              ┌───────────┼───────────┐
              │           │           │
           Qwen      Mistral      Llama
         (Primary)  (Validator)  (Creative)
```

### Model Roles

| Model | Primary Use | Mode | Strengths | Prompt Strategy |
|-------|------------|------|-----------|-----------------|
| **Qwen 2.5-7B** | Korean brief interpretation + JSON generation | All modes | Korean understanding, structured output | Detailed instructions, JSON schema |
| **Mistral 7B** | Schema/unit/IFRA validation | All modes | Rule-based validation, correction | Validation rules, error detection |
| **Llama 3-8B** | Creative hints generation | Creative only | Artistic expression, creativity | Open-ended prompts, inspiration |

---

## Model-Specific Templates

### 1. Qwen 2.5-7B Instruct (Primary LLM)

**Purpose**: Parse user's Korean/English description and generate structured `CreativeBrief` JSON

**System Prompt Template**:

```python
QWEN_SYSTEM_PROMPT = """You are an expert perfume designer assistant specializing in fragrance creation.

Your task is to analyze the user's description and extract structured information for fragrance formulation.

Output a JSON object with the following schema:
{
    "style": string,              # Main fragrance style: fresh/floral/woody/oriental/citrus/etc
    "intensity": float,           # Intensity level: 0.0-1.0 (0.3=light, 0.5=medium, 0.8=strong)
    "complexity": float,          # Complexity: 0.0-1.0 (0.3=simple, 0.5=balanced, 0.8=complex)
    "notes_preference": {         # Note family preferences
        "citrus": float,          # 0.0-1.0
        "floral": float,
        "woody": float,
        "oriental": float,
        "fresh": float,
        "fruity": float,
        "spicy": float
    },
    "product_category": string,   # EDP/EDT/Cologne/Body Spray
    "target_profile": string,     # daily_fresh/evening_elegant/sporty/etc
    "mood": list[string],         # ["energetic", "calm", "romantic", etc]
    "season": list[string],       # ["spring", "summer", "fall", "winter"]
    "budget_tier": string,        # entry/mid/premium/luxury
    "constraints": {
        "allergen_free": bool,
        "vegan": bool,
        "natural_only": bool,
        "max_allergens_ppm": float
    }
}

Guidelines:
1. Interpret Korean text naturally (존댓말/반말 both acceptable)
2. Extract emotional and sensory keywords
3. Infer missing fields with reasonable defaults
4. Use 0.0-1.0 scale for all numeric preferences
5. Output valid JSON only (no markdown, no explanations)

Korean keywords mapping:
- 상큼한/청량한/시원한 → fresh, citrus (high intensity)
- 꽃향기/플로럴/화사한 → floral (high intensity)
- 우디한/나무/따뜻한 → woody (high intensity)
- 달콤한/부드러운 → fruity, floral (medium-high)
- 강한/진한 → intensity: 0.7-0.9
- 은은한/약한 → intensity: 0.2-0.4
- 복잡한/다층적 → complexity: 0.7-0.9
- 단순한/미니멀 → complexity: 0.2-0.4
"""

QWEN_USER_PROMPT = """User description:
{user_text}

Extract the structured fragrance brief as JSON.
"""
```

**Example Qwen Prompt (Fast Mode)**:

```python
# Fast mode: Quick parsing, minimal inference
system = """You are a perfume assistant. Extract fragrance brief as JSON.

Output schema:
{"style": "string", "intensity": 0.0-1.0, "complexity": 0.0-1.0,
 "notes_preference": {"citrus": 0.0-1.0, "floral": 0.0-1.0, ...}}

Rules:
- Use obvious keywords only
- Default to 0.5 for ambiguous values
- Output JSON only"""

user = f"Description: {user_text}\n\nJSON output:"
```

**Example Qwen Prompt (Balanced Mode)**:

```python
# Balanced mode: More context, better inference
system = QWEN_SYSTEM_PROMPT  # Full system prompt above

user = f"""User description:
{user_text}

Analyze the description and generate a structured fragrance brief.
Consider:
1. Emotional keywords (energetic, calm, romantic)
2. Sensory descriptions (fresh, warm, sweet)
3. Context clues (season, occasion, time of day)
4. Cultural nuances (Korean beauty trends, K-pop style)

Output valid JSON."""
```

**Example Qwen Prompt (Creative Mode)**:

```python
# Creative mode: Deep interpretation, artistic inference
system = QWEN_SYSTEM_PROMPT + """

Additional creative guidelines:
- Interpret abstract and poetic language
- Infer mood from storytelling and imagery
- Map metaphors to fragrance notes (e.g., "ocean breeze" → fresh + citrus)
- Consider seasonal and emotional context deeply
- Generate sophisticated, nuanced preferences
"""

user = f"""User description (may be poetic or abstract):
{user_text}

Deeply analyze the description:
1. What emotions and images does it evoke?
2. What fragrance story would capture this feeling?
3. Which note families best represent this narrative?

Generate a creative fragrance brief as JSON."""
```

### 2. Mistral 7B Instruct (Validator)

**Purpose**: Validate and correct the brief for schema compliance, unit consistency, and IFRA regulations

**System Prompt Template**:

```python
MISTRAL_SYSTEM_PROMPT = """You are a fragrance regulation and quality control specialist.

Your task is to validate a fragrance brief for:
1. Schema compliance (all required fields present and valid)
2. Unit consistency (0.0-1.0 ranges, valid enums)
3. IFRA regulation compliance (allergen limits, restricted materials)
4. Logical consistency (e.g., "summer fresh" shouldn't have heavy oriental notes)

Input: CreativeBrief JSON
Output: ValidationReport JSON

Validation rules:
- intensity: 0.0-1.0 (0.0=none, 1.0=maximum)
- complexity: 0.0-1.0 (0.0=simple, 1.0=complex)
- notes_preference: Each note family 0.0-1.0
- product_category: Must be one of [EDP, EDT, Cologne, Body Spray]
- budget_tier: Must be one of [entry, mid, premium, luxury]
- max_allergens_ppm: Default 500 for EDP, 100 for Body Spray

IFRA restrictions (simplified):
- allergen_free=true → max_allergens_ppm must be 0
- natural_only=true → Restrict synthetic ingredients
- vegan=true → No animal-derived ingredients

Output schema:
{
    "valid": bool,
    "errors": list[string],      # Critical errors (must fix)
    "warnings": list[string],    # Warnings (should review)
    "corrections": dict,         # Suggested corrections
    "ifra_compliance": bool
}
"""

MISTRAL_USER_PROMPT = """Validate the following fragrance brief:

{brief_json}

Check for:
1. Schema validity
2. Unit ranges (0.0-1.0)
3. IFRA compliance
4. Logical consistency

Output validation report as JSON."""
```

**Example Mistral Validation Prompt**:

```python
system = MISTRAL_SYSTEM_PROMPT

user = f"""Validate this brief:

{json.dumps(brief, indent=2)}

Validation checklist:
✓ All required fields present?
✓ intensity in [0.0, 1.0]?
✓ complexity in [0.0, 1.0]?
✓ notes_preference values in [0.0, 1.0]?
✓ product_category valid enum?
✓ allergen_free + max_allergens_ppm consistent?
✓ Season/mood logical for style?

Output validation report."""
```

### 3. Llama 3-8B Instruct (Creative Hints)

**Purpose**: Generate creative hints and inspirational notes for creative mode (creative mode only)

**System Prompt Template**:

```python
LLAMA_SYSTEM_PROMPT = """You are a creative perfume artist and storyteller.

Your task is to generate evocative, poetic hints that inspire fragrance creation.

Given a fragrance brief, generate 3-8 creative hints:
- Artistic descriptions
- Sensory metaphors
- Emotional associations
- Storytelling elements
- Cultural references
- Natural imagery

Guidelines:
1. Be poetic and evocative, not technical
2. Use vivid sensory language
3. Create emotional connections
4. Inspire creativity in formulation
5. Keep hints concise (5-15 words each)

Output as JSON list:
{
    "creative_hints": [
        "Hint 1",
        "Hint 2",
        ...
    ]
}

Example hints:
- "Morning dew on jasmine petals, kissed by first light"
- "A walk through an ancient cedar forest after rain"
- "The warmth of sun-baked amber on Mediterranean shores"
- "Whispers of vanilla and spice in a moonlit garden"
"""

LLAMA_USER_PROMPT = """Fragrance brief:
Style: {style}
Mood: {mood}
Season: {season}
Notes: {notes}

Generate 5-8 creative, poetic hints that capture the essence of this fragrance."""
```

**Example Llama Creative Prompt**:

```python
system = LLAMA_SYSTEM_PROMPT

user = f"""Brief:
Style: {brief.style}
Intensity: {brief.intensity}
Mood: {', '.join(brief.mood)}
Season: {', '.join(brief.season)}
Top notes: {brief.notes_preference['citrus']:.1f} citrus, {brief.notes_preference['fresh']:.1f} fresh
Heart notes: {brief.notes_preference['floral']:.1f} floral
Base notes: {brief.notes_preference['woody']:.1f} woody

Generate creative hints that evoke:
1. The emotional journey of this fragrance
2. Natural scenes and imagery
3. Sensory metaphors
4. Poetic descriptions

Output as JSON list of 5-8 hints."""
```

---

## Mode-Based Prompt Strategies

### Fast Mode (Qwen Only)

**Goal**: Quick turnaround, literal interpretation

**Prompt Characteristics**:
- Short, direct system prompt
- Minimal context
- Clear schema definition
- No creative inference
- Default to obvious mappings

**Example**:
```python
system = "Extract fragrance brief as JSON. Use obvious keywords only."
user = f"Description: {text}\nJSON:"
```

**Expected Latency**: 2-5 seconds

### Balanced Mode (Qwen + Mistral)

**Goal**: Accurate interpretation + validation

**Prompt Characteristics**:
- Full system prompt with context
- Moderate inference depth
- Schema validation
- Logical consistency checks
- Error correction

**Example**:
```python
# Qwen: Full interpretation
system = QWEN_SYSTEM_PROMPT
user = f"Description: {text}\n\nAnalyze and generate JSON."

# Mistral: Validation
system = MISTRAL_SYSTEM_PROMPT
user = f"Validate: {qwen_output}\n\nCheck schema and IFRA."
```

**Expected Latency**: 5-10 seconds

### Creative Mode (Qwen + Mistral + Llama)

**Goal**: Deep interpretation + validation + creative hints

**Prompt Characteristics**:
- Enhanced creative system prompt
- Deep emotional inference
- Abstract language handling
- Artistic metaphor mapping
- Creative hints generation

**Example**:
```python
# Qwen: Creative interpretation
system = QWEN_SYSTEM_PROMPT + CREATIVE_ENHANCEMENT
user = f"Description (poetic): {text}\n\nDeeply analyze and generate JSON."

# Mistral: Validation (same as balanced)
system = MISTRAL_SYSTEM_PROMPT
user = f"Validate: {qwen_output}"

# Llama: Creative hints
system = LLAMA_SYSTEM_PROMPT
user = f"Brief: {validated_brief}\n\nGenerate creative hints."
```

**Expected Latency**: 10-20 seconds

---

## Prompt Engineering Best Practices

### 1. Schema Definition

**Always provide clear JSON schema**:

✅ Good:
```python
"""Output JSON:
{
    "style": string,        # One of: fresh/floral/woody/oriental
    "intensity": float,     # Range: 0.0-1.0
    "complexity": float     # Range: 0.0-1.0
}"""
```

❌ Bad:
```python
"""Give me fragrance info"""
```

### 2. Korean Language Handling

**Provide explicit Korean keyword mappings**:

✅ Good:
```python
"""Korean keywords:
- 상큼한 → citrus, fresh
- 꽃향기 → floral
- 우디한 → woody
- 달콤한 → fruity, sweet
"""
```

### 3. Default Values

**Specify defaults for ambiguous cases**:

✅ Good:
```python
"""If intensity unclear, default to 0.5 (medium).
If category not specified, default to EDP."""
```

### 4. Output Format Control

**Strictly enforce JSON-only output**:

✅ Good:
```python
"""Output valid JSON only. No markdown, no explanations, no code blocks.
Start with { and end with }."""
```

### 5. Context Preservation

**Include relevant context without overloading**:

✅ Good (Balanced):
```python
"""Consider:
1. Emotional keywords
2. Sensory descriptions
3. Seasonal context
(3-5 focused points)
"""
```

❌ Bad (Overload):
```python
"""Consider 20 different aspects... (overwhelming)"""
```

### 6. Temperature and Top-P Settings

**Mode-specific generation parameters**:

| Mode | Qwen Temp | Qwen Top-P | Llama Temp | Llama Top-P |
|------|-----------|------------|------------|-------------|
| **Fast** | 0.3 | 0.8 | N/A | N/A |
| **Balanced** | 0.7 | 0.9 | N/A | N/A |
| **Creative** | 0.8 | 0.95 | 0.9 | 0.95 |

**Rationale**:
- **Low temp (0.3)**: Deterministic, literal interpretation (fast mode)
- **Medium temp (0.7)**: Balanced creativity and accuracy (balanced mode)
- **High temp (0.9)**: Maximum creativity (creative mode, Llama)

---

## Korean/English Handling

### Bilingual Prompt Strategy

**System prompt should support both languages**:

```python
BILINGUAL_SYSTEM = """You are a perfume assistant (Korean/English).

한국어와 영어 모두 지원합니다.
Supports both Korean and English input.

Korean examples:
- "상큼한 여름 향기" → fresh, citrus, summer
- "은은한 꽃향기" → floral, low intensity

English examples:
- "fresh summer scent" → fresh, citrus, summer
- "subtle floral notes" → floral, low intensity
"""
```

### Korean Keyword Dictionary

**Comprehensive mapping**:

| Korean | English | Note Families | Intensity |
|--------|---------|---------------|-----------|
| 상큼한 | Fresh, crisp | citrus, fresh | 0.7-0.9 |
| 청량한 | Refreshing | citrus, fresh | 0.8-1.0 |
| 꽃향기 | Floral | floral | 0.6-0.8 |
| 화사한 | Bright, flowery | floral, fresh | 0.7-0.9 |
| 우디한 | Woody | woody | 0.6-0.8 |
| 따뜻한 | Warm | woody, oriental | 0.5-0.7 |
| 달콤한 | Sweet | fruity, floral | 0.6-0.8 |
| 부드러운 | Soft, gentle | floral, fruity | 0.4-0.6 |
| 강한 | Strong | (any) | 0.8-1.0 |
| 진한 | Intense | (any) | 0.7-0.9 |
| 은은한 | Subtle | (any) | 0.2-0.4 |
| 약한 | Light, weak | (any) | 0.1-0.3 |
| 복잡한 | Complex | (any) | 0.7-0.9 |
| 단순한 | Simple | (any) | 0.2-0.4 |
| 시원한 | Cool, refreshing | fresh, citrus | 0.7-0.9 |
| 스파이시 | Spicy | spicy, oriental | 0.6-0.8 |
| 나무 | Wood | woody | 0.6-0.8 |
| 동양적 | Oriental | oriental | 0.6-0.8 |
| 시트러스 | Citrus | citrus | 0.7-0.9 |
| 과일 | Fruity | fruity | 0.6-0.8 |

---

## Examples

### Example 1: Fast Mode (Korean)

**Input**:
```
상큼한 레몬 향기
```

**Qwen Prompt**:
```python
system = """Extract fragrance brief as JSON.
Korean mapping: 상큼한→citrus/fresh, 레몬→lemon/citrus
Output JSON only."""

user = "Description: 상큼한 레몬 향기\nJSON:"
```

**Expected Output**:
```json
{
    "style": "fresh",
    "intensity": 0.7,
    "complexity": 0.3,
    "notes_preference": {
        "citrus": 0.9,
        "fresh": 0.8,
        "floral": 0.2,
        "woody": 0.1
    }
}
```

### Example 2: Balanced Mode (English)

**Input**:
```
I want a warm, woody fragrance for winter evenings. Something sophisticated and calming.
```

**Qwen Prompt**:
```python
system = QWEN_SYSTEM_PROMPT  # Full system prompt

user = """Description:
I want a warm, woody fragrance for winter evenings. Something sophisticated and calming.

Analyze:
1. Keywords: warm, woody, winter, sophisticated, calming
2. Mood: calm, elegant
3. Season: winter
4. Intensity: moderate-high (evening fragrance)

Generate JSON."""
```

**Expected Output**:
```json
{
    "style": "woody",
    "intensity": 0.7,
    "complexity": 0.6,
    "notes_preference": {
        "woody": 0.9,
        "oriental": 0.6,
        "spicy": 0.4,
        "citrus": 0.2
    },
    "mood": ["calm", "sophisticated"],
    "season": ["winter"],
    "target_profile": "evening_elegant"
}
```

**Mistral Validation**:
```python
system = MISTRAL_SYSTEM_PROMPT

user = f"""Validate: {qwen_output}

Checks:
✓ All fields valid
✓ intensity 0.7 reasonable for "warm" description
✓ woody 0.9 matches user request
✓ Season winter appropriate

Output: validation report"""
```

### Example 3: Creative Mode (Poetic Korean)

**Input**:
```
봄날 아침, 햇살에 반짝이는 이슬 맺힌 하얀 꽃잎.
상쾌하면서도 우아한, 마치 발레리나의 첫 스텝처럼.
```

**Translation**:
```
Spring morning, white petals glistening with dew in the sunlight.
Refreshing yet elegant, like a ballerina's first step.
```

**Qwen Creative Prompt**:
```python
system = QWEN_SYSTEM_PROMPT + """
Creative mode: Interpret poetic and abstract language.
Map imagery to fragrance notes:
- 이슬 (dew) → fresh, aquatic
- 하얀 꽃잎 (white petals) → floral, clean
- 발레리나 (ballerina) → elegant, graceful, sophisticated
"""

user = """Description (poetic):
봄날 아침, 햇살에 반짝이는 이슬 맺힌 하얀 꽃잎.
상쾌하면서도 우아한, 마치 발레리나의 첫 스텝처럼.

Imagery analysis:
- Spring morning: fresh, new beginnings
- Dew on petals: aquatic, fresh, delicate
- White petals: floral, clean, pure
- Ballerina: elegant, graceful, sophisticated, light

Generate creative fragrance brief."""
```

**Expected Output**:
```json
{
    "style": "floral",
    "intensity": 0.5,
    "complexity": 0.6,
    "notes_preference": {
        "floral": 0.9,
        "fresh": 0.8,
        "citrus": 0.4,
        "woody": 0.2
    },
    "mood": ["elegant", "graceful", "energetic"],
    "season": ["spring"],
    "target_profile": "daily_elegant"
}
```

**Llama Creative Hints**:
```python
system = LLAMA_SYSTEM_PROMPT

user = f"""Brief: {validated_brief}

Imagery: Spring morning, dew on white petals, ballerina's grace

Generate creative hints."""
```

**Expected Llama Output**:
```json
{
    "creative_hints": [
        "첫 햇살에 깨어나는 하얀 목련의 속삭임",
        "Morning dew dancing on lily of the valley",
        "발레리나의 투투처럼 가볍고 우아한 첫인상",
        "Spring breeze carrying whispers of jasmine and neroli",
        "순백의 꽃잎이 공중에서 그리는 우아한 곡선",
        "The delicate strength of a ballerina's first arabesque",
        "Crystalline drops of freshness on silk-soft petals"
    ]
}
```

---

## Prompt Versioning

**Track prompt versions for reproducibility**:

```python
PROMPT_VERSION = "v2.1"

QWEN_SYSTEM_PROMPT_V2_1 = """..."""  # Version 2.1
QWEN_SYSTEM_PROMPT_V2_0 = """..."""  # Previous version

# Log prompt version with each request
logger.info(f"Using prompt version: {PROMPT_VERSION}")
```

---

## Testing Prompts

**Prompt evaluation metrics**:

1. **Accuracy**: Correct field extraction rate
2. **Consistency**: Same input → same output (low temp)
3. **Latency**: Response time
4. **JSON Validity**: Parseable output rate

**Test cases**:

```python
test_cases = [
    {
        "input": "상큼한 레몬향",
        "expected": {"style": "fresh", "notes_preference": {"citrus": ">0.7"}},
        "mode": "fast"
    },
    {
        "input": "warm woody winter fragrance",
        "expected": {"style": "woody", "season": ["winter"]},
        "mode": "balanced"
    },
    # ... more cases
]

for case in test_cases:
    output = llm.generate(case["input"], mode=case["mode"])
    assert output["style"] == case["expected"]["style"]
```

---

## Summary

**Key Takeaways**:

1. **Model Specialization**: Qwen (parsing), Mistral (validation), Llama (creativity)
2. **Mode Adaptation**: Fast (short prompts), Balanced (full context), Creative (deep inference)
3. **Bilingual Support**: Explicit Korean keyword mapping
4. **Schema Enforcement**: Always provide clear JSON schema
5. **Temperature Control**: Mode-specific generation parameters
6. **Version Control**: Track prompt versions for reproducibility

**Prompt Template Files**: All templates available in `fragrance_ai/llm/prompts/`

For implementation details, see:
- `fragrance_ai/llm/qwen_model.py`: Qwen prompt implementation
- `fragrance_ai/llm/mistral_validator.py`: Mistral validation prompts
- `fragrance_ai/llm/llama_creative.py`: Llama creative hints

For operational guidance, see:
- `docs/FAILURE_SCENARIOS.md`: Handling model failures
- `docs/TUNING_GUIDE.md`: Parameter optimization
