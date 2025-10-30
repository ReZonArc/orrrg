# Persona-Based Training Guide

## Overview

The ESM-2 Hypergredient Framework now supports **persona-based training**, enabling the model to learn and adapt to different user types and preferences. This allows for personalized formulation recommendations based on individual skin characteristics, concerns, and preferences.

## What are Personas?

Personas are user profiles that capture specific characteristics, preferences, and behaviors. Each persona includes:

- **Skin Type**: oily, dry, sensitive, normal, combination
- **Primary Concerns**: wrinkles, acne, sensitivity, hyperpigmentation, etc.
- **Sensitivity Level**: How sensitive the user is to active ingredients (0.0-1.0)
- **Budget Preference**: budget, mid-range, premium
- **Ingredient Preferences**: Preferred ingredient types
- **Ingredient Aversions**: Ingredients to avoid
- **Safety Priority**: How much they prioritize safety over efficacy (0.0-1.0)

## Default Personas

The system comes with 4 pre-configured personas:

### 1. Sensitive Skin Specialist
- **Focus**: Gentle, hypoallergenic formulations
- **Skin Type**: sensitive
- **Safety Priority**: 95% (very high)
- **Prefers**: ceramides, niacinamide, hyaluronic acid
- **Avoids**: alcohol, fragrances, essential oils

### 2. Anti-Aging Enthusiast
- **Focus**: Powerful anti-aging ingredients
- **Skin Type**: normal
- **Safety Priority**: 60% (moderate)
- **Prefers**: retinoids, peptides, vitamin C
- **Avoids**: parabens

### 3. Acne-Prone Specialist
- **Focus**: Oil control and acne treatment
- **Skin Type**: oily
- **Safety Priority**: 70% (moderate-high)
- **Prefers**: salicylic acid, niacinamide, zinc
- **Avoids**: comedogenic oils, heavy moisturizers

### 4. Natural Beauty Advocate
- **Focus**: Natural and organic ingredients
- **Skin Type**: normal
- **Safety Priority**: 80% (high)
- **Prefers**: plant extracts, oils, botanicals
- **Avoids**: sulfates, parabens, synthetic fragrances

## How to Use Persona Training

### 1. Basic Persona Usage

```python
from hypergredient_framework import PersonaTrainingSystem, HypergredientAI, FormulationRequest

# Initialize systems
persona_system = PersonaTrainingSystem()
ai_system = HypergredientAI(persona_system)

# Set active persona
ai_system.persona_system.set_active_persona('sensitive_skin')

# Make persona-aware predictions
request = FormulationRequest(
    target_concerns=['sensitivity', 'redness'],
    skin_type='sensitive',
    budget=500.0
)

prediction = ai_system.predict_optimal_combination(request)
print(f"Active persona: {prediction['active_persona']}")
print(f"Persona adjustments: {prediction['persona_adjustments']}")
```

### 2. Training a Persona

```python
# Create training data
training_requests = [
    FormulationRequest(['sensitivity'], skin_type='sensitive', budget=400),
    FormulationRequest(['redness'], skin_type='sensitive', budget=500)
]

training_results = [
    # FormulationResult objects with actual outcomes
]

training_feedback = [
    {'efficacy': 8.0, 'safety': 9.5, 'user_satisfaction': 8.5},
    {'efficacy': 7.5, 'safety': 9.8, 'user_satisfaction': 9.0}
]

# Train the persona
ai_system.train_with_persona('sensitive_skin', training_requests, training_results, training_feedback)
```

### 3. Creating Custom Personas

```python
from hypergredient_framework import PersonaProfile

# Create a custom persona
custom_persona = PersonaProfile(
    persona_id="mature_skin",
    name="Mature Skin Expert",
    description="Focuses on age-related skin concerns",
    skin_type="mature",
    primary_concerns=["fine_lines", "loss_of_elasticity", "age_spots"],
    sensitivity_level=0.6,
    budget_preference="premium",
    ingredient_preferences=["retinol", "peptides", "antioxidants"],
    ingredient_aversions=["harsh_acids"],
    safety_priority=0.75,
    natural_preference=0.4
)

# Add to system
persona_system.add_persona(custom_persona)
```

## Command Line Interface

### View Available Personas
```bash
python3 hypergraph_query.py --query persona
```

This shows:
- All available personas with their characteristics
- Demo predictions comparing different personas
- Training status for each persona

### Train a Persona
```bash
python3 hypergraph_query.py --query persona_train
```

This demonstrates:
- Simulated training process
- Training data ingestion
- Model adaptation for the persona
- Training performance metrics

## How Persona Training Works

### 1. Feature Adjustment
When a persona is active, the AI system adjusts prediction features based on persona characteristics:

```python
# Example adjustments
adjusted_features['persona_sensitivity'] = persona.sensitivity_level
adjusted_features['persona_safety_priority'] = persona.safety_priority
adjusted_features['persona_natural_preference'] = persona.natural_preference
```

### 2. Confidence Scoring
Personas influence ingredient confidence scores:

- **Safety-conscious personas**: Reduce confidence in strong actives
- **Concern-specific personas**: Boost confidence in relevant ingredient classes
- **Budget-aware personas**: Adjust based on cost considerations

### 3. Ingredient Preferences
Personas have ingredient preference scores:

```python
preferences = {
    'preferred_ingredient': 1.5,  # 50% boost
    'avoided_ingredient': 0.1     # 90% penalty
}
```

### 4. Training Data Integration
Persona training accumulates:

- **Formulation Requests**: What users asked for
- **Formulation Results**: What was recommended
- **Feedback Scores**: How well it worked (efficacy, safety, satisfaction)

## Model Retraining

The system tracks persona-specific feedback and incorporates it during model retraining:

```python
# After 100 feedback samples, model retrains
# Persona-specific feedback is weighted appropriately
# Model version increments (v1.0 → v1.1)
```

## Integration with Existing Framework

Persona training seamlessly integrates with existing capabilities:

- **Hypergredient Database**: All ingredients remain available
- **Compatibility Analysis**: Safety checks still apply
- **Cost Optimization**: Budget constraints are respected
- **Evolutionary Formulation**: Can evolve persona-specific formulas

## Best Practices

### 1. Persona Design
- Keep personas distinct and focused
- Base on real user research when possible
- Include clear safety boundaries
- Consider regulatory constraints

### 2. Training Data Quality
- Use diverse formulation scenarios
- Include both positive and negative feedback
- Balance efficacy and safety outcomes
- Document persona-specific successes/failures

### 3. Model Management
- Monitor persona performance separately
- Regular retraining with fresh data
- A/B test persona recommendations
- Maintain persona-specific metrics

## Examples and Use Cases

### Beauty Brand Application
```python
# Different personas for brand's customer segments
personas = [
    'teen_acne',      # Teenage acne sufferers
    'pregnant_safe',  # Pregnancy-safe formulations  
    'men_grooming',   # Male skincare preferences
    'luxury_anti_aging'  # High-end anti-aging market
]
```

### Dermatology Clinic Integration
```python
# Personas based on clinical conditions
clinical_personas = [
    'rosacea_management',
    'post_procedure_care',
    'sensitive_reactive',
    'melasma_treatment'
]
```

### Regulatory Compliance
```python
# Region-specific personas
regulatory_personas = [
    'eu_natural_organic',  # EU organic standards
    'fda_otc_compliant',   # US OTC drug requirements
    'k_beauty_trends',     # Korean beauty preferences
    'ayurveda_traditional' # Traditional Indian formulations
]
```

## Advanced Features

### Persona Clustering
Analyze persona similarities and differences:

```python
persona_analysis = persona_system.analyze_persona_clusters()
# Groups similar personas
# Identifies unique characteristics
# Suggests new persona opportunities
```

### Cross-Persona Learning
Transfer learning between related personas:

```python
# Anti-aging knowledge can inform mature skin persona
# Sensitive skin learnings apply to post-procedure care
```

### Dynamic Persona Evolution
Personas can evolve based on aggregate user feedback:

```python
# Trending ingredients update persona preferences
# New research findings adjust safety priorities
# Market changes influence budget considerations
```

## Future Enhancements

1. **Temporal Personas**: Seasonal or life-stage based personas
2. **Contextual Adaptation**: Occasion-specific formulations
3. **Multi-Modal Learning**: Integration with user images/selfies
4. **Real-Time Feedback**: Continuous learning from user interactions
5. **Federated Learning**: Privacy-preserving cross-brand persona insights

---

For technical implementation details, see the source code in `hypergredient_framework.py` and tests in `test_persona_training.py`.