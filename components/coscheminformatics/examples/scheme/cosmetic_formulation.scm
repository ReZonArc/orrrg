;
; cosmetic_formulation.scm
;
; Advanced example demonstrating cosmetic chemistry applications using
; the OpenCog cheminformatics framework. This example shows how to:
; 1. Define cosmetic ingredients with their functional properties
; 2. Create complex formulations with multiple ingredient types
; 3. Check ingredient compatibility and interactions
; 4. Assess formulation properties and stability
;
; To run this example, start the guile shell in the `examples/scheme` 
; directory and then:
;   `(load "cosmetic_formulation.scm")`
; It will create formulations and analyze their properties.

(use-modules (opencog) (opencog cheminformatics))
(use-modules (opencog exec))

; =================================================================
; Define Common Cosmetic Ingredients with Properties
; =================================================================

; Active Ingredients
(define hyaluronic-acid
    (Active_ingredient
        (name "hyaluronic_acid")
        (inci_name "Sodium Hyaluronate")
        (function "humectant")
        (typical_usage_range "0.1-2.0")
        (ph_stability "4.0-8.0")))

(define retinol
    (Active_ingredient
        (name "retinol")
        (inci_name "Retinol")
        (function "anti_aging")
        (typical_usage_range "0.01-1.0")
        (ph_stability "5.5-7.0")
        (light_sensitive "true")
        (oxygen_sensitive "true")))

(define vitamin-c
    (Active_ingredient
        (name "vitamin_c")
        (inci_name "Ascorbic Acid")
        (function "antioxidant")
        (typical_usage_range "5.0-20.0")
        (ph_stability "3.0-6.0")
        (oxygen_sensitive "true")))

(define niacinamide
    (Active_ingredient
        (name "niacinamide")
        (inci_name "Niacinamide")
        (function "skin_conditioning")
        (typical_usage_range "2.0-10.0")
        (ph_stability "4.0-7.0")))

; Preservatives
(define phenoxyethanol
    (Preservative
        (name "phenoxyethanol")
        (inci_name "Phenoxyethanol")
        (antimicrobial_spectrum "broad")
        (typical_usage_range "0.3-1.0")
        (ph_effectiveness "3.0-10.0")))

(define potassium-sorbate
    (Preservative
        (name "potassium_sorbate")
        (inci_name "Potassium Sorbate")
        (antimicrobial_spectrum "mold_yeast")
        (typical_usage_range "0.1-0.3")
        (ph_effectiveness "3.0-6.5")))

; Emulsifiers
(define cetyl-alcohol
    (Emulsifier
        (name "cetyl_alcohol")
        (inci_name "Cetyl Alcohol")
        (emulsion_type "oil_in_water")
        (typical_usage_range "1.0-5.0")
        (melting_point "49-51")))

(define lecithin
    (Emulsifier
        (name "lecithin")
        (inci_name "Lecithin")
        (emulsion_type "water_in_oil")
        (typical_usage_range "0.2-2.0")
        (natural_origin "true")))

; Humectants
(define glycerin
    (Humectant
        (name "glycerin")
        (inci_name "Glycerin")
        (hygroscopic "true")
        (typical_usage_range "3.0-15.0")
        (solubility "water")))

; UV Filters
(define zinc-oxide
    (Uv_filter
        (name "zinc_oxide")
        (inci_name "Zinc Oxide")
        (filter_type "physical")
        (protection_range "UVA_UVB")
        (typical_usage_range "5.0-25.0")
        (white_cast "moderate")))

; =================================================================
; Define Ingredient Interactions and Compatibility Rules
; =================================================================

; Vitamin C and Retinol are incompatible (different pH requirements)
(define vitamin-c-retinol-incompatibility
    (Incompatibility_link
        (ingredient1 vitamin-c)
        (ingredient2 retinol)
        (reason "pH_incompatible")
        (recommendation "use_separately")))

; Vitamin C and Vitamin E show synergy
(define antioxidant-synergy
    (Synergy_link
        (ingredient1 vitamin-c)
        (ingredient2 "vitamin_e")
        (effect "enhanced_antioxidant_activity")
        (mechanism "regeneration")))

; Hyaluronic acid and Niacinamide are compatible
(define ha-niacinamide-compatibility
    (Compatibility_link
        (ingredient1 hyaluronic-acid)
        (ingredient2 niacinamide)
        (formulation_benefit "enhanced_hydration")))

; =================================================================
; Create Advanced Cosmetic Formulations
; =================================================================

; Anti-Aging Serum Formulation
(define anti-aging-serum
    (Skincare_formulation
        (product_type "serum")
        (target_skin_concern "aging")
        (phase_structure "single_phase")
        
        ; Water phase (85%)
        (water_phase
            (Molecule (H "water1") (O "water1")) ; Aqua
            (humectant glycerin "5.0%")
            (active hyaluronic-acid "1.0%")
            (active niacinamide "5.0%"))
        
        ; Oil phase (10%)
        (oil_phase
            (emollient "squalane" "8.0%")
            (antioxidant "vitamin_e" "2.0%"))
        
        ; Preservative system (1%)
        (preservative_system
            (preservative phenoxyethanol "0.8%")
            (preservative potassium-sorbate "0.2%"))
        
        ; Properties
        (target_pH "5.5-6.0")
        (viscosity "low")
        (texture "lightweight")
        (application "morning_evening")))

; Moisturizing Cream Formulation
(define moisturizing-cream
    (Skincare_formulation
        (product_type "cream")
        (target_skin_concern "dryness")
        (phase_structure "oil_in_water_emulsion")
        
        ; Water phase (70%)
        (water_phase
            (Molecule (H "water1") (O "water1"))
            (humectant glycerin "10.0%")
            (active hyaluronic-acid "2.0%"))
        
        ; Oil phase (25%)
        (oil_phase
            (emollient "shea_butter" "8.0%")
            (emollient "jojoba_oil" "5.0%")
            (emulsifier cetyl-alcohol "3.0%")
            (antioxidant "vitamin_e" "1.0%"))
        
        ; Thickening system (4%)
        (thickening_system
            (thickener "carbomer" "0.2%")
            (ph_adjuster "triethanolamine" "0.1%"))
        
        ; Preservative system (1%)
        (preservative_system
            (preservative phenoxyethanol "0.7%")
            (preservative potassium-sorbate "0.3%"))
        
        ; Properties
        (target_pH "6.0-6.5")
        (viscosity "high")
        (texture "rich")
        (application "evening")))

; Sunscreen Formulation
(define broad-spectrum-sunscreen
    (Skincare_formulation
        (product_type "sunscreen")
        (target_skin_concern "UV_protection")
        (phase_structure "oil_in_water_emulsion")
        (spf_target "30")
        
        ; Water phase (55%)
        (water_phase
            (Molecule (H "water1") (O "water1"))
            (humectant glycerin "5.0%"))
        
        ; UV filter system (25%)
        (uv_protection
            (uv_filter zinc-oxide "15.0%")
            (uv_filter "titanium_dioxide" "5.0%")
            (uv_filter "avobenzone" "3.0%")
            (uv_filter "octinoxate" "2.0%"))
        
        ; Oil phase (15%)
        (oil_phase
            (emollient "caprylic_triglyceride" "8.0%")
            (emulsifier lecithin "2.0%")
            (antioxidant "vitamin_e" "1.0%"))
        
        ; Sensory modifiers (4%)
        (sensory_system
            (texture_modifier "dimethicone" "3.0%")
            (slip_agent "cyclopentasiloxane" "1.0%"))
        
        ; Preservative system (1%)
        (preservative_system
            (preservative phenoxyethanol "0.8%")
            (preservative "ethylhexylglycerin" "0.2%"))
        
        ; Properties
        (target_pH "6.5-7.0")
        (viscosity "medium")
        (texture "non_greasy")
        (water_resistance "40_minutes")
        (application "morning")))

; =================================================================
; Formulation Analysis and Optimization
; =================================================================

; Define a rule to check pH compatibility in formulations
(define ph-compatibility-check
    (BindLink
        ; Variables for ingredients and their pH ranges
        (VariableList
            (TypedVariable (Variable "$ingredient1") (Type 'Active_ingredient))
            (TypedVariable (Variable "$ingredient2") (Type 'Active_ingredient))
            (TypedVariable (Variable "$formulation") (Type 'Skincare_formulation)))
        
        ; Pattern: Look for formulations containing multiple actives
        (AndLink
            (MemberLink (Variable "$ingredient1") (Variable "$formulation"))
            (MemberLink (Variable "$ingredient2") (Variable "$formulation"))
            (NotLink (EqualLink (Variable "$ingredient1") (Variable "$ingredient2"))))
        
        ; Create compatibility assessment
        (ExecutionOutputLink
            (GroundedSchemaNode "scm: assess-ph-compatibility")
            (ListLink (Variable "$ingredient1") (Variable "$ingredient2") (Variable "$formulation")))))

; Define a rule to identify potential synergies
(define synergy-identification
    (BindLink
        (VariableList
            (TypedVariable (Variable "$ingredient1") (Type 'Cosmetic_ingredient))
            (TypedVariable (Variable "$ingredient2") (Type 'Cosmetic_ingredient)))
        
        ; Pattern: Look for known synergistic combinations
        (AndLink
            (Synergy_link (Variable "$ingredient1") (Variable "$ingredient2"))
            (NotLink (EqualLink (Variable "$ingredient1") (Variable "$ingredient2"))))
        
        ; Output the synergistic pair
        (ListLink (Variable "$ingredient1") (Variable "$ingredient2"))))

; =================================================================
; Execute Analysis
; =================================================================

; Check for ingredient synergies in our formulations
(define synergies (cog-execute! synergy-identification))

; Create a comprehensive ingredient database
(define cosmetic-ingredient-database
    (ConceptNode "cosmetic_ingredients_db"
        (active_ingredients (ListLink hyaluronic-acid retinol vitamin-c niacinamide))
        (preservatives (ListLink phenoxyethanol potassium-sorbate))
        (emulsifiers (ListLink cetyl-alcohol lecithin))
        (humectants (ListLink glycerin))
        (uv_filters (ListLink zinc-oxide))))

; Output results
(format #t "=== Cosmetic Formulation Analysis ===~n")
(format #t "Created formulations:~n")
(format #t "1. Anti-aging serum: ~A~n" anti-aging-serum)
(format #t "2. Moisturizing cream: ~A~n" moisturizing-cream)
(format #t "3. Broad-spectrum sunscreen: ~A~n" broad-spectrum-sunscreen)
(format #t "~n")
(format #t "Ingredient synergies found: ~A~n" synergies)
(format #t "~n")
(format #t "Formulation database created: ~A~n" cosmetic-ingredient-database)

; =================================================================
; Advanced Compatibility Matrix
; =================================================================

; Create a comprehensive compatibility matrix
(define compatibility-matrix
    (list
        ; Vitamin C compatibility
        (list vitamin-c niacinamide "compatible" "pH_managed")
        (list vitamin-c retinol "incompatible" "pH_conflict")
        (list vitamin-c hyaluronic-acid "compatible" "enhanced_penetration")
        
        ; Retinol compatibility  
        (list retinol niacinamide "compatible" "irritation_reduction")
        (list retinol hyaluronic-acid "compatible" "dryness_mitigation")
        
        ; Niacinamide compatibility
        (list niacinamide hyaluronic-acid "compatible" "hydration_boost")))

(format #t "~n=== Ingredient Compatibility Matrix ===~n")
(for-each
    (lambda (combination)
        (format #t "~A + ~A: ~A (~A)~n"
            (car combination)
            (cadr combination)  
            (caddr combination)
            (cadddr combination)))
    compatibility-matrix)

; ------------------------------------------------
; The end.
; That's all, folks!