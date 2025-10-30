;; Complex Cosmetic Formulation Modeling Example
;; 
;; This Scheme script demonstrates advanced cosmetic formulation modeling
;; using the OpenCog cheminformatics framework with specialized atom types
;; for cosmetic chemistry applications.

;; Load cosmetic chemistry atom types
(use-modules (opencog)
             (opencog query)
             (opencog exec))

;; =============================================================================
;; INGREDIENT DEFINITIONS
;; =============================================================================

;; Define active ingredients with detailed properties
(define hyaluronic-acid
  (ConceptNode "hyaluronic_acid"))

(define niacinamide  
  (ConceptNode "niacinamide"))

(define vitamin-c
  (ConceptNode "vitamin_c"))

(define retinol
  (ConceptNode "retinol"))

(define peptide-complex
  (ConceptNode "palmitoyl_pentapeptide"))

;; Define supporting ingredients
(define glycerin
  (ConceptNode "glycerin"))

(define cetyl-alcohol
  (ConceptNode "cetyl_alcohol"))

(define xanthan-gum
  (ConceptNode "xanthan_gum"))

(define phenoxyethanol
  (ConceptNode "phenoxyethanol"))

(define vitamin-e
  (ConceptNode "vitamin_e"))

(define citric-acid
  (ConceptNode "citric_acid"))

;; =============================================================================
;; INGREDIENT CLASSIFICATIONS
;; =============================================================================

;; Classify ingredients by functional categories
(InheritanceLink hyaluronic-acid (ConceptNode "ACTIVE_INGREDIENT"))
(InheritanceLink hyaluronic-acid (ConceptNode "HUMECTANT"))

(InheritanceLink niacinamide (ConceptNode "ACTIVE_INGREDIENT"))
(InheritanceLink niacinamide (ConceptNode "VITAMIN"))

(InheritanceLink vitamin-c (ConceptNode "ACTIVE_INGREDIENT"))
(InheritanceLink vitamin-c (ConceptNode "ANTIOXIDANT"))

(InheritanceLink retinol (ConceptNode "ACTIVE_INGREDIENT"))
(InheritanceLink retinol (ConceptNode "VITAMIN"))

(InheritanceLink peptide-complex (ConceptNode "ACTIVE_INGREDIENT"))
(InheritanceLink peptide-complex (ConceptNode "PEPTIDE"))

(InheritanceLink glycerin (ConceptNode "HUMECTANT"))
(InheritanceLink cetyl-alcohol (ConceptNode "EMULSIFIER"))
(InheritanceLink xanthan-gum (ConceptNode "THICKENER"))
(InheritanceLink phenoxyethanol (ConceptNode "PRESERVATIVE"))
(InheritanceLink vitamin-e (ConceptNode "ANTIOXIDANT"))
(InheritanceLink citric-acid (ConceptNode "PH_ADJUSTER"))

;; =============================================================================
;; INGREDIENT PROPERTIES
;; =============================================================================

;; Define ingredient properties using property links
(EvaluationLink
  (PredicateNode "molecular_weight")
  (ListLink hyaluronic-acid (NumberNode "1000000")))

(EvaluationLink
  (PredicateNode "solubility")
  (ListLink hyaluronic-acid (ConceptNode "water_soluble")))

(EvaluationLink
  (PredicateNode "ph_stability_range")
  (ListLink hyaluronic-acid 
            (ListLink (NumberNode "3.0") (NumberNode "8.0"))))

(EvaluationLink
  (PredicateNode "max_concentration")
  (ListLink hyaluronic-acid (NumberNode "2.0")))

;; Niacinamide properties
(EvaluationLink
  (PredicateNode "molecular_weight")
  (ListLink niacinamide (NumberNode "122.12")))

(EvaluationLink
  (PredicateNode "solubility")
  (ListLink niacinamide (ConceptNode "water_soluble")))

(EvaluationLink
  (PredicateNode "ph_stability_range")
  (ListLink niacinamide 
            (ListLink (NumberNode "5.0") (NumberNode "7.0"))))

(EvaluationLink
  (PredicateNode "max_concentration")
  (ListLink niacinamide (NumberNode "10.0")))

;; Vitamin C properties
(EvaluationLink
  (PredicateNode "solubility")
  (ListLink vitamin-c (ConceptNode "water_soluble")))

(EvaluationLink
  (PredicateNode "ph_stability_range")
  (ListLink vitamin-c 
            (ListLink (NumberNode "2.0") (NumberNode "3.5"))))

(EvaluationLink
  (PredicateNode "oxidation_sensitive")
  (ListLink vitamin-c (ConceptNode "true")))

;; Retinol properties
(EvaluationLink
  (PredicateNode "solubility")
  (ListLink retinol (ConceptNode "oil_soluble")))

(EvaluationLink
  (PredicateNode "light_sensitive")
  (ListLink retinol (ConceptNode "true")))

(EvaluationLink
  (PredicateNode "oxygen_sensitive")
  (ListLink retinol (ConceptNode "true")))

(EvaluationLink
  (PredicateNode "max_concentration")
  (ListLink retinol (NumberNode "1.0")))

;; =============================================================================
;; FORMULATION CREATION
;; =============================================================================

;; Create advanced anti-aging serum formulation
(define advanced-serum
  (ConceptNode "advanced_anti_aging_serum"))

(InheritanceLink advanced-serum (ConceptNode "SKINCARE_FORMULATION"))

;; Add formulation ingredients with concentrations
(MemberLink hyaluronic-acid advanced-serum)
(ExecutionLink
  (SchemaNode "concentration")
  (ListLink advanced-serum hyaluronic-acid)
  (NumberNode "1.5"))

(MemberLink niacinamide advanced-serum)
(ExecutionLink
  (SchemaNode "concentration")
  (ListLink advanced-serum niacinamide)
  (NumberNode "5.0"))

(MemberLink peptide-complex advanced-serum)
(ExecutionLink
  (SchemaNode "concentration")
  (ListLink advanced-serum peptide-complex)
  (NumberNode "3.0"))

(MemberLink glycerin advanced-serum)
(ExecutionLink
  (SchemaNode "concentration")
  (ListLink advanced-serum glycerin)
  (NumberNode "10.0"))

(MemberLink phenoxyethanol advanced-serum)
(ExecutionLink
  (SchemaNode "concentration")
  (ListLink advanced-serum phenoxyethanol)
  (NumberNode "0.5"))

(MemberLink xanthan-gum advanced-serum)
(ExecutionLink
  (SchemaNode "concentration")
  (ListLink advanced-serum xanthan-gum)
  (NumberNode "0.3"))

;; Create vitamin C brightening serum (separate formulation due to pH incompatibility)
(define vitamin-c-serum
  (ConceptNode "vitamin_c_brightening_serum"))

(InheritanceLink vitamin-c-serum (ConceptNode "SKINCARE_FORMULATION"))

(MemberLink vitamin-c vitamin-c-serum)
(ExecutionLink
  (SchemaNode "concentration")
  (ListLink vitamin-c-serum vitamin-c)
  (NumberNode "15.0"))

(MemberLink vitamin-e vitamin-c-serum)
(ExecutionLink
  (SchemaNode "concentration")
  (ListLink vitamin-c-serum vitamin-e)
  (NumberNode "0.5"))

(MemberLink citric-acid vitamin-c-serum)
(ExecutionLink
  (SchemaNode "concentration")
  (ListLink vitamin-c-serum citric-acid)
  (NumberNode "0.1"))

;; =============================================================================
;; COMPATIBILITY ANALYSIS
;; =============================================================================

;; Define ingredient compatibility relationships
(EvaluationLink
  (PredicateNode "compatible")
  (ListLink hyaluronic-acid niacinamide))

(EvaluationLink
  (PredicateNode "compatible")
  (ListLink hyaluronic-acid peptide-complex))

(EvaluationLink
  (PredicateNode "compatible")
  (ListLink niacinamide glycerin))

(EvaluationLink
  (PredicateNode "compatible")
  (ListLink peptide-complex glycerin))

;; Define incompatible combinations
(EvaluationLink
  (PredicateNode "incompatible")
  (ListLink vitamin-c retinol))

(EvaluationLink
  (PredicateNode "incompatible_reason")
  (ListLink 
    (ListLink vitamin-c retinol)
    (ConceptNode "pH_incompatibility")))

(EvaluationLink
  (PredicateNode "incompatible")
  (ListLink niacinamide vitamin-c))

(EvaluationLink
  (PredicateNode "incompatible_reason")
  (ListLink 
    (ListLink niacinamide vitamin-c)
    (ConceptNode "pH_difference")))

;; Define synergistic relationships
(EvaluationLink
  (PredicateNode "synergistic")
  (ListLink vitamin-c vitamin-e))

(EvaluationLink
  (PredicateNode "synergy_mechanism")
  (ListLink 
    (ListLink vitamin-c vitamin-e)
    (ConceptNode "antioxidant_network")))

(EvaluationLink
  (PredicateNode "synergistic")
  (ListLink hyaluronic-acid glycerin))

(EvaluationLink
  (PredicateNode "synergy_mechanism")
  (ListLink 
    (ListLink hyaluronic-acid glycerin)
    (ConceptNode "enhanced_hydration")))

;; =============================================================================
;; FORMULATION PROPERTIES
;; =============================================================================

;; Define target properties for advanced serum
(EvaluationLink
  (PredicateNode "target_benefit")
  (ListLink advanced-serum (ConceptNode "anti_aging")))

(EvaluationLink
  (PredicateNode "target_benefit")
  (ListLink advanced-serum (ConceptNode "deep_hydration")))

(EvaluationLink
  (PredicateNode "target_benefit")
  (ListLink advanced-serum (ConceptNode "skin_repair")))

(EvaluationLink
  (PredicateNode "target_benefit")
  (ListLink advanced-serum (ConceptNode "collagen_stimulation")))

;; Define physical properties
(EvaluationLink
  (PredicateNode "texture")
  (ListLink advanced-serum (ConceptNode "lightweight_gel")))

(EvaluationLink
  (PredicateNode "viscosity")
  (ListLink advanced-serum (ConceptNode "medium")))

(EvaluationLink
  (PredicateNode "absorption_rate")
  (ListLink advanced-serum (ConceptNode "fast")))

(EvaluationLink
  (PredicateNode "target_ph")
  (ListLink advanced-serum (NumberNode "5.5")))

;; Define target properties for vitamin C serum
(EvaluationLink
  (PredicateNode "target_benefit")
  (ListLink vitamin-c-serum (ConceptNode "brightening")))

(EvaluationLink
  (PredicateNode "target_benefit")
  (ListLink vitamin-c-serum (ConceptNode "antioxidant_protection")))

(EvaluationLink
  (PredicateNode "target_ph")
  (ListLink vitamin-c-serum (NumberNode "3.0")))

;; =============================================================================
;; SAFETY AND REGULATORY INFORMATION
;; =============================================================================

;; Define concentration limits (EU regulations)
(EvaluationLink
  (PredicateNode "eu_concentration_limit")
  (ListLink retinol (NumberNode "0.3")))

(EvaluationLink
  (PredicateNode "fda_gras_status")
  (ListLink vitamin-c (ConceptNode "approved")))

(EvaluationLink
  (PredicateNode "pregnancy_safe")
  (ListLink hyaluronic-acid (ConceptNode "yes")))

(EvaluationLink
  (PredicateNode "pregnancy_safe")
  (ListLink niacinamide (ConceptNode "yes")))

(EvaluationLink
  (PredicateNode "pregnancy_safe")
  (ListLink retinol (ConceptNode "no")))

;; Define allergen information
(EvaluationLink
  (PredicateNode "allergenicity_rating")
  (ListLink hyaluronic-acid (ConceptNode "very_low")))

(EvaluationLink
  (PredicateNode "allergenicity_rating")
  (ListLink niacinamide (ConceptNode "very_low")))

(EvaluationLink
  (PredicateNode "allergenicity_rating")
  (ListLink vitamin-c (ConceptNode "low")))

;; =============================================================================
;; QUERY FUNCTIONS
;; =============================================================================

;; Query to find all active ingredients in a formulation
(define find-active-ingredients
  (lambda (formulation)
    (cog-execute!
      (GetLink
        (VariableNode "$ingredient")
        (AndLink
          (MemberLink (VariableNode "$ingredient") formulation)
          (InheritanceLink (VariableNode "$ingredient") 
                          (ConceptNode "ACTIVE_INGREDIENT")))))))

;; Query to find compatible ingredient pairs
(define find-compatible-pairs
  (GetLink
    (VariableList
      (VariableNode "$ingredient1")
      (VariableNode "$ingredient2"))
    (EvaluationLink
      (PredicateNode "compatible")
      (ListLink (VariableNode "$ingredient1") 
                (VariableNode "$ingredient2")))))

;; Query to find ingredients with specific properties
(define find-ingredients-by-property
  (lambda (property value)
    (cog-execute!
      (GetLink
        (VariableNode "$ingredient")
        (EvaluationLink
          (PredicateNode property)
          (ListLink (VariableNode "$ingredient") value))))))

;; Query to check formulation pH compatibility
(define check-ph-compatibility
  (lambda (formulation)
    (let* ((ingredients (cog-outgoing-set 
                        (find-active-ingredients formulation)))
           (ph-ranges (map get-ph-range ingredients)))
      (if (all-ranges-overlap? ph-ranges)
          (ConceptNode "pH_compatible")
          (ConceptNode "pH_incompatible")))))

;; Helper function to get pH range for ingredient
(define get-ph-range
  (lambda (ingredient)
    (cog-execute!
      (GetLink
        (VariableNode "$range")
        (EvaluationLink
          (PredicateNode "ph_stability_range")
          (ListLink ingredient (VariableNode "$range")))))))

;; =============================================================================
;; ANALYSIS FUNCTIONS
;; =============================================================================

;; Function to calculate total active ingredient concentration
(define calculate-total-actives
  (lambda (formulation)
    (let* ((active-ingredients (cog-outgoing-set 
                               (find-active-ingredients formulation)))
           (concentrations (map (lambda (ing) 
                                 (get-concentration formulation ing))
                               active-ingredients)))
      (apply + (filter number? concentrations)))))

;; Helper function to get ingredient concentration
(define get-concentration
  (lambda (formulation ingredient)
    (let ((result (cog-execute!
                   (GetLink
                     (VariableNode "$conc")
                     (ExecutionLink
                       (SchemaNode "concentration")
                       (ListLink formulation ingredient)
                       (VariableNode "$conc"))))))
      (if (null? (cog-outgoing-set result))
          0
          (string->number (cog-name (car (cog-outgoing-set result))))))))

;; Function to analyze formulation stability
(define analyze-stability
  (lambda (formulation)
    (let* ((ingredients (get-formulation-ingredients formulation))
           (ph-stable (check-ph-compatibility formulation))
           (oxidation-risk (any-oxidation-sensitive? ingredients))
           (light-sensitive (any-light-sensitive? ingredients)))
      (ListLink
        (EvaluationLink (PredicateNode "ph_stable") 
                       (ListLink formulation ph-stable))
        (EvaluationLink (PredicateNode "oxidation_risk") 
                       (ListLink formulation 
                                (if oxidation-risk 
                                    (ConceptNode "high") 
                                    (ConceptNode "low"))))
        (EvaluationLink (PredicateNode "light_protection_needed") 
                       (ListLink formulation 
                                (if light-sensitive 
                                    (ConceptNode "yes") 
                                    (ConceptNode "no"))))))))

;; Helper functions for stability analysis
(define get-formulation-ingredients
  (lambda (formulation)
    (cog-outgoing-set
      (cog-execute!
        (GetLink
          (VariableNode "$ingredient")
          (MemberLink (VariableNode "$ingredient") formulation))))))

(define any-oxidation-sensitive?
  (lambda (ingredients)
    (any (lambda (ing) (has-property? ing "oxidation_sensitive")) ingredients)))

(define any-light-sensitive?
  (lambda (ingredients)
    (any (lambda (ing) (has-property? ing "light_sensitive")) ingredients)))

(define has-property?
  (lambda (ingredient property)
    (not (null? (cog-outgoing-set
                 (cog-execute!
                   (GetLink
                     (VariableNode "$value")
                     (EvaluationLink
                       (PredicateNode property)
                       (ListLink ingredient (VariableNode "$value"))))))))))

;; =============================================================================
;; DEMONSTRATION AND TESTING
;; =============================================================================

;; Display formulation information
(display "=== Advanced Anti-Aging Serum Analysis ===\n")

;; Find active ingredients
(display "Active ingredients in advanced serum:\n")
(let ((actives (find-active-ingredients advanced-serum)))
  (for-each (lambda (ing) 
              (display (string-append "  - " (cog-name ing) "\n")))
            (cog-outgoing-set actives)))

;; Calculate total active concentration
(display "\nTotal active ingredient concentration: ")
(display (calculate-total-actives advanced-serum))
(display "%\n")

;; Find compatible pairs
(display "\nCompatible ingredient pairs:\n")
(let ((pairs (cog-execute! find-compatible-pairs)))
  (for-each (lambda (pair)
              (let ((pair-list (cog-outgoing-set pair)))
                (display (string-append "  - " 
                                       (cog-name (car pair-list)) 
                                       " + " 
                                       (cog-name (cadr pair-list)) 
                                       "\n"))))
            (cog-outgoing-set pairs)))

;; Analyze stability
(display "\nStability Analysis:\n")
(let ((stability (analyze-stability advanced-serum)))
  (for-each (lambda (result)
              (display "  ")
              (display result)
              (display "\n"))
            (cog-outgoing-set stability)))

;; Display vitamin C serum information
(display "\n=== Vitamin C Brightening Serum Analysis ===\n")

(display "Active ingredients in vitamin C serum:\n")
(let ((actives (find-active-ingredients vitamin-c-serum)))
  (for-each (lambda (ing) 
              (display (string-append "  - " (cog-name ing) "\n")))
            (cog-outgoing-set actives)))

(display "\nTotal active ingredient concentration: ")
(display (calculate-total-actives vitamin-c-serum))
(display "%\n")

;; Find water-soluble ingredients
(display "\nWater-soluble ingredients:\n")
(let ((water-soluble (find-ingredients-by-property "solubility" 
                                                   (ConceptNode "water_soluble"))))
  (for-each (lambda (ing) 
              (display (string-append "  - " (cog-name ing) "\n")))
            (cog-outgoing-set water-soluble)))

(display "\n=== Formulation Recommendations ===\n")
(display "1. Use advanced serum and vitamin C serum at different times\n")
(display "2. Apply vitamin C serum in morning, advanced serum at night\n")
(display "3. Store vitamin C serum in dark, cool place\n")
(display "4. Consider adding vitamin E to advanced serum for antioxidant protection\n")
(display "5. pH buffering may be needed for optimal ingredient stability\n")

(display "\n=== Analysis Complete ===\n")