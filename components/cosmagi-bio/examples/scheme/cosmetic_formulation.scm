;;; cosmetic_formulation.scm
;;; 
;;; Complex Cosmetic Formulation Modeling with Compatibility Analysis
;;; 
;;; This Scheme example demonstrates advanced cosmetic formulation modeling
;;; using OpenCog's knowledge representation capabilities. It shows how to
;;; create complex formulations, analyze ingredient interactions, and perform
;;; formulation optimization through reasoning.
;;;
;;; Requirements:
;;; - OpenCog AtomSpace with bioscience extensions loaded
;;; - Guile Scheme interpreter
;;;
;;; Usage:
;;;   guile -l cosmetic_formulation.scm
;;;
;;; Author: OpenCog Cosmetic Chemistry Framework
;;; License: AGPL-3.0

(use-modules (opencog)
             (opencog bioscience))

;;; Load cosmetic chemistry atom types
(load "opencog/bioscience/types/bioscience_types.scm")

;;; =============================================================================
;;; INGREDIENT DEFINITION FUNCTIONS
;;; =============================================================================

(define (create-cosmetic-ingredient name type functions properties)
  "Create a comprehensive cosmetic ingredient with all properties"
  (let ((ingredient (ConceptNode name)))
    
    ;; Set primary type
    (InheritanceLink ingredient (ConceptNode type))
    
    ;; Set additional functions
    (for-each 
      (lambda (function)
        (InheritanceLink ingredient (ConceptNode function)))
      functions)
    
    ;; Set properties
    (for-each
      (lambda (prop-pair)
        (let ((property (car prop-pair))
              (value (cdr prop-pair)))
          (EvaluationLink
            (PredicateNode property)
            (ListLink ingredient 
                      (if (number? value)
                          (NumberNode (number->string value))
                          (ConceptNode (symbol->string value)))))))
      properties)
    
    ingredient))

;;; =============================================================================
;;; COSMETIC INGREDIENT DATABASE
;;; =============================================================================

(display "🧴 Creating cosmetic ingredient database...\n")

;; Define active ingredients
(define hyaluronic-acid
  (create-cosmetic-ingredient
    "hyaluronic_acid"
    "ACTIVE_INGREDIENT"
    '("HUMECTANT")
    '((max_concentration . 5.0)
      (optimal_ph_min . 4.0)
      (optimal_ph_max . 7.0)
      (cost_per_kg . 2500.0)
      (allergen_risk . low)
      (water_soluble . yes)
      (stability . temperature_sensitive))))

(define niacinamide
  (create-cosmetic-ingredient
    "niacinamide"
    "ACTIVE_INGREDIENT"
    '()
    '((max_concentration . 10.0)
      (optimal_ph_min . 5.0)
      (optimal_ph_max . 7.0)
      (cost_per_kg . 45.0)
      (allergen_risk . low)
      (water_soluble . yes)
      (stability . stable))))

(define retinol
  (create-cosmetic-ingredient
    "retinol"
    "ACTIVE_INGREDIENT"
    '()
    '((max_concentration . 1.0)
      (optimal_ph_min . 5.5)
      (optimal_ph_max . 6.5)
      (cost_per_kg . 8000.0)
      (allergen_risk . medium)
      (water_soluble . no)
      (stability . very_sensitive))))

(define vitamin-c
  (create-cosmetic-ingredient
    "vitamin_c"
    "ACTIVE_INGREDIENT" 
    '("ANTIOXIDANT")
    '((max_concentration . 20.0)
      (optimal_ph_min . 3.0)
      (optimal_ph_max . 4.0)
      (cost_per_kg . 120.0)
      (allergen_risk . low)
      (water_soluble . yes)
      (stability . sensitive))))

;; Define functional ingredients
(define glycerin
  (create-cosmetic-ingredient
    "glycerin"
    "HUMECTANT"
    '()
    '((max_concentration . 20.0)
      (optimal_ph_min . 4.0)
      (optimal_ph_max . 8.0)
      (cost_per_kg . 3.0)
      (allergen_risk . low)
      (water_soluble . yes)
      (stability . stable))))

(define cetyl-alcohol
  (create-cosmetic-ingredient
    "cetyl_alcohol"
    "EMULSIFIER"
    '("EMOLLIENT")
    '((max_concentration . 10.0)
      (optimal_ph_min . 4.0)
      (optimal_ph_max . 8.0)
      (cost_per_kg . 8.0)
      (allergen_risk . low)
      (water_soluble . no)
      (stability . stable))))

(define phenoxyethanol
  (create-cosmetic-ingredient
    "phenoxyethanol"
    "PRESERVATIVE"
    '()
    '((max_concentration . 1.0)
      (optimal_ph_min . 4.0)
      (optimal_ph_max . 8.0)
      (cost_per_kg . 12.0)
      (allergen_risk . low)
      (water_soluble . yes)
      (stability . stable))))

(define xanthan-gum
  (create-cosmetic-ingredient
    "xanthan_gum"
    "THICKENER"
    '()
    '((max_concentration . 2.0)
      (optimal_ph_min . 4.0)
      (optimal_ph_max . 10.0)
      (cost_per_kg . 15.0)
      (allergen_risk . low)
      (water_soluble . yes)
      (stability . stable))))

(define squalane
  (create-cosmetic-ingredient
    "squalane"
    "EMOLLIENT"
    '()
    '((max_concentration . 30.0)
      (optimal_ph_min . 4.0)
      (optimal_ph_max . 8.0)
      (cost_per_kg . 35.0)
      (allergen_risk . low)
      (water_soluble . no)
      (stability . very_stable))))

(define vitamin-e
  (create-cosmetic-ingredient
    "vitamin_e"
    "ANTIOXIDANT"
    '()
    '((max_concentration . 1.0)
      (optimal_ph_min . 4.0)
      (optimal_ph_max . 8.0)
      (cost_per_kg . 18.0)
      (allergen_risk . low)
      (water_soluble . no)
      (stability . light_sensitive))))

(display "✓ Ingredient database created with 10 ingredients\n")

;;; =============================================================================
;;; INGREDIENT INTERACTION RULES
;;; =============================================================================

(display "🔗 Defining ingredient interaction rules...\n")

;; Define compatibility relationships
(define (create-compatibility ingredient1 ingredient2 description)
  "Create a compatibility relationship between two ingredients"
  (let ((compatibility-link
          (EvaluationLink
            (PredicateNode "compatible_with")
            (ListLink ingredient1 ingredient2))))
    
    ;; Add description
    (EvaluationLink
      (PredicateNode "interaction_description")
      (ListLink compatibility-link (ConceptNode description)))
    
    compatibility-link))

;; Define incompatibility relationships  
(define (create-incompatibility ingredient1 ingredient2 reason)
  "Create an incompatibility relationship between two ingredients"
  (let ((incompatibility-link
          (EvaluationLink
            (PredicateNode "incompatible_with")
            (ListLink ingredient1 ingredient2))))
    
    ;; Add reason
    (EvaluationLink
      (PredicateNode "incompatibility_reason")
      (ListLink incompatibility-link (ConceptNode reason)))
    
    incompatibility-link))

;; Define synergy relationships
(define (create-synergy ingredient1 ingredient2 benefit)
  "Create a synergistic relationship between two ingredients"
  (let ((synergy-link
          (EvaluationLink
            (PredicateNode "synergistic_with")
            (ListLink ingredient1 ingredient2))))
    
    ;; Add benefit description
    (EvaluationLink
      (PredicateNode "synergy_benefit")
      (ListLink synergy-link (ConceptNode benefit)))
    
    synergy-link))

;; Create compatibility rules
(create-compatibility hyaluronic-acid niacinamide 
                     "Enhanced hydration and barrier function")
(create-compatibility niacinamide glycerin
                     "Complementary moisturizing effects")
(create-compatibility cetyl-alcohol glycerin
                     "Improved emulsion stability")
(create-compatibility retinol squalane
                     "Reduced irritation from emollient cushioning")

;; Create incompatibility rules
(create-incompatibility vitamin-c retinol
                       "pH incompatibility and instability")
(create-incompatibility vitamin-c niacinamide
                       "Potential interaction reducing efficacy")

;; Create synergy rules
(create-synergy vitamin-c vitamin-e
               "Enhanced antioxidant activity and stability")
(create-synergy hyaluronic-acid glycerin
               "Superior moisture retention")

(display "✓ Interaction rules established\n")

;;; =============================================================================
;;; FORMULATION CREATION FUNCTIONS
;;; =============================================================================

(define (create-formulation name type ingredients-with-concentrations)
  "Create a complete formulation with ingredient concentrations"
  (let ((formulation (ConceptNode name)))
    
    ;; Set formulation type
    (InheritanceLink formulation (ConceptNode type))
    
    ;; Add ingredients with concentrations
    (for-each
      (lambda (ingredient-data)
        (let ((ingredient (car ingredient-data))
              (concentration (cadr ingredient-data))
              (role (if (> (length ingredient-data) 2)
                        (caddr ingredient-data)
                        "ingredient")))
          
          ;; Add concentration
          (EvaluationLink
            (PredicateNode "concentration")
            (ListLink formulation ingredient 
                      (NumberNode (number->string concentration))))
          
          ;; Add role
          (EvaluationLink
            (PredicateNode "ingredient_role")
            (ListLink formulation ingredient (ConceptNode role)))))
      ingredients-with-concentrations)
    
    formulation))

;;; =============================================================================
;;; COMPLEX FORMULATION EXAMPLES
;;; =============================================================================

(display "🧪 Creating complex formulations...\n")

;; Create an advanced anti-aging serum
(define anti-aging-serum
  (create-formulation
    "advanced_anti_aging_serum"
    "SKINCARE_FORMULATION"
    `((,hyaluronic-acid 2.0 "hydrating_active")
      (,niacinamide 5.0 "barrier_active")
      (,retinol 0.5 "anti_aging_active")
      (,glycerin 8.0 "humectant")
      (,squalane 10.0 "emollient")
      (,vitamin-e 0.5 "antioxidant_stabilizer")
      (,phenoxyethanol 0.8 "preservative")
      (,xanthan-gum 0.3 "thickener"))))

;; Create a vitamin C brightening serum
(define vitamin-c-serum
  (create-formulation
    "vitamin_c_brightening_serum"
    "SKINCARE_FORMULATION"
    `((,vitamin-c 15.0 "brightening_active")
      (,vitamin-e 0.8 "antioxidant_synergist")
      (,hyaluronic-acid 1.5 "hydrating_active")
      (,glycerin 12.0 "humectant")
      (,phenoxyethanol 0.9 "preservative")
      (,xanthan-gum 0.4 "thickener"))))

;; Create a gentle hydrating moisturizer
(define hydrating-moisturizer
  (create-formulation
    "gentle_hydrating_moisturizer"
    "SKINCARE_FORMULATION"
    `((,hyaluronic-acid 3.0 "primary_hydrator")
      (,niacinamide 3.0 "barrier_support")
      (,glycerin 15.0 "humectant")
      (,cetyl-alcohol 4.0 "emulsifier")
      (,squalane 8.0 "emollient")
      (,phenoxyethanol 0.7 "preservative")
      (,xanthan-gum 0.2 "thickener"))))

(display "✓ Created 3 complex formulations\n")

;;; =============================================================================
;;; FORMULATION ANALYSIS FUNCTIONS
;;; =============================================================================

(define (analyze-formulation-compatibility formulation)
  "Analyze compatibility issues within a formulation"
  (display (format #f "\n🔍 Analyzing compatibility for ~a:\n" 
                   (cog-name formulation)))
  
  ;; Get all ingredients in the formulation
  (define formulation-ingredients
    (filter-map
      (lambda (atom)
        (and (eq? (cog-type atom) 'EvaluationLink)
             (eq? (cog-name (gar atom)) "concentration")
             (eq? (gadr atom) formulation)
             (gaddr atom)))
      (cog-get-atoms 'EvaluationLink)))
  
  (display (format #f "  Ingredients: ~a\n" 
                   (length formulation-ingredients)))
  
  ;; Check all pairs for compatibility issues
  (define compatibility-issues '())
  (define synergies '())
  
  (for-each
    (lambda (ing1)
      (for-each
        (lambda (ing2)
          (when (not (eq? ing1 ing2))
            ;; Check for incompatibilities
            (let ((incompatibility
                    (find
                      (lambda (atom)
                        (and (eq? (cog-type atom) 'EvaluationLink)
                             (eq? (cog-name (gar atom)) "incompatible_with")
                             (or (and (eq? (gadr atom) ing1)
                                      (eq? (gaddr atom) ing2))
                                 (and (eq? (gadr atom) ing2)
                                      (eq? (gaddr atom) ing1)))))
                      (cog-get-atoms 'EvaluationLink))))
              
              (when incompatibility
                (set! compatibility-issues 
                      (cons (list ing1 ing2) compatibility-issues))))
            
            ;; Check for synergies
            (let ((synergy
                    (find
                      (lambda (atom)
                        (and (eq? (cog-type atom) 'EvaluationLink)
                             (eq? (cog-name (gar atom)) "synergistic_with")
                             (or (and (eq? (gadr atom) ing1)
                                      (eq? (gaddr atom) ing2))
                                 (and (eq? (gadr atom) ing2)
                                      (eq? (gaddr atom) ing1)))))
                      (cog-get-atoms 'EvaluationLink))))
              
              (when synergy
                (set! synergies 
                      (cons (list ing1 ing2) synergies))))))
        formulation-ingredients))
    formulation-ingredients)
  
  ;; Report results
  (if (null? compatibility-issues)
      (display "  ✓ No compatibility issues detected\n")
      (begin
        (display "  ⚠ Compatibility issues found:\n")
        (for-each
          (lambda (issue)
            (display (format #f "    - ~a + ~a\n" 
                             (cog-name (car issue))
                             (cog-name (cadr issue)))))
          compatibility-issues)))
  
  (if (null? synergies)
      (display "  • No synergies detected\n")
      (begin
        (display "  ⚡ Synergistic combinations:\n")
        (for-each
          (lambda (synergy)
            (display (format #f "    + ~a + ~a\n"
                             (cog-name (car synergy))
                             (cog-name (cadr synergy)))))
          synergies)))
  
  (list compatibility-issues synergies))

(define (calculate-formulation-properties formulation)
  "Calculate key properties of a formulation"
  (display (format #f "\n📊 Calculating properties for ~a:\n"
                   (cog-name formulation)))
  
  ;; Get ingredient concentrations
  (define ingredient-concentrations
    (filter-map
      (lambda (atom)
        (and (eq? (cog-type atom) 'EvaluationLink)
             (eq? (cog-name (gar atom)) "concentration")
             (eq? (gadr atom) formulation)
             (cons (gaddr atom) 
                   (string->number (cog-name (gadddr atom))))))
      (cog-get-atoms 'EvaluationLink)))
  
  ;; Calculate total active concentration
  (define total-active-concentration
    (fold
      (lambda (ing-conc acc)
        (let ((ingredient (car ing-conc))
              (concentration (cdr ing-conc)))
          (if (find
                (lambda (atom)
                  (and (eq? (cog-type atom) 'InheritanceLink)
                       (eq? (gar atom) ingredient)
                       (eq? (cog-name (gdr atom)) "ACTIVE_INGREDIENT")))
                (cog-get-atoms 'InheritanceLink))
              (+ acc concentration)
              acc)))
      0
      ingredient-concentrations))
  
  ;; Calculate estimated cost
  (define estimated-cost-per-100g
    (fold
      (lambda (ing-conc acc)
        (let* ((ingredient (car ing-conc))
               (concentration (cdr ing-conc))
               ;; Get cost per kg (simplified lookup)
               (cost-per-kg
                 (let ((cost-atom
                         (find
                           (lambda (atom)
                             (and (eq? (cog-type atom) 'EvaluationLink)
                                  (eq? (cog-name (gar atom)) "cost_per_kg")
                                  (eq? (gadr atom) ingredient)))
                           (cog-get-atoms 'EvaluationLink))))
                   (if cost-atom
                       (string->number (cog-name (gaddr cost-atom)))
                       10.0))) ; default cost
               (ingredient-cost (* (/ concentration 100) (/ cost-per-kg 1000))))
          (+ acc ingredient-cost)))
      0
      ingredient-concentrations))
  
  ;; Display results
  (display (format #f "  • Total ingredients: ~a\n" 
                   (length ingredient-concentrations)))
  (display (format #f "  • Total active concentration: ~a%\n"
                   total-active-concentration))
  (display (format #f "  • Estimated cost per 100g: $~a\n"
                   (exact->inexact estimated-cost-per-100g)))
  
  ;; Set calculated properties
  (EvaluationLink
    (PredicateNode "total_active_concentration")
    (ListLink formulation 
              (NumberNode (number->string total-active-concentration))))
  
  (EvaluationLink
    (PredicateNode "estimated_cost_per_100g")
    (ListLink formulation 
              (NumberNode (number->string estimated-cost-per-100g))))
  
  (list total-active-concentration estimated-cost-per-100g))

;;; =============================================================================
;;; FORMULATION OPTIMIZATION RULES
;;; =============================================================================

(define (suggest-formulation-improvements formulation)
  "Suggest improvements for a formulation based on analysis"
  (display (format #f "\n💡 Improvement suggestions for ~a:\n"
                   (cog-name formulation)))
  
  (let ((compatibility-analysis (analyze-formulation-compatibility formulation))
        (properties (calculate-formulation-properties formulation)))
    
    (define compatibility-issues (car compatibility-analysis))
    (define synergies (cadr compatibility-analysis))
    (define total-actives (car properties))
    (define cost (cadr properties))
    
    ;; Suggestions based on analysis
    (when (not (null? compatibility-issues))
      (display "  📝 Compatibility Improvements:\n")
      (display "    - Consider separating incompatible ingredients\n")
      (display "    - Use stabilizing agents or adjust pH\n")
      (display "    - Implement time-release or layered delivery\n"))
    
    (when (> total-actives 15.0)
      (display "  📝 Active Concentration:\n")
      (display "    - High active concentration may cause irritation\n")
      (display "    - Consider reducing concentrations or phased introduction\n"))
    
    (when (> cost 10.0)
      (display "  📝 Cost Optimization:\n")
      (display "    - Formulation cost is high\n")
      (display "    - Consider ingredient substitutions\n")
      (display "    - Optimize concentrations for cost-efficiency\n"))
    
    (when (null? synergies)
      (display "  📝 Efficacy Enhancement:\n")
      (display "    - No synergistic combinations detected\n")
      (display "    - Consider adding complementary ingredients\n"))
    
    (when (and (null? compatibility-issues)
               (<= total-actives 15.0)
               (<= cost 10.0)
               (not (null? synergies)))
      (display "  ✅ Formulation appears well-optimized!\n"))))

;;; =============================================================================
;;; MAIN ANALYSIS EXECUTION
;;; =============================================================================

(display "\n🚀 Starting comprehensive formulation analysis...\n")

;; Analyze each formulation
(suggest-formulation-improvements anti-aging-serum)
(suggest-formulation-improvements vitamin-c-serum)
(suggest-formulation-improvements hydrating-moisturizer)

;;; =============================================================================
;;; KNOWLEDGE QUERY EXAMPLES
;;; =============================================================================

(display "\n🔍 Knowledge Query Examples:\n")

;; Query 1: Find all active ingredients
(display "\nQuery: Find all active ingredients\n")
(define active-ingredients
  (filter
    (lambda (atom)
      (and (eq? (cog-type atom) 'InheritanceLink)
           (eq? (cog-name (gdr atom)) "ACTIVE_INGREDIENT")))
    (cog-get-atoms 'InheritanceLink)))

(display (format #f "Found ~a active ingredients:\n" (length active-ingredients)))
(for-each
  (lambda (link)
    (display (format #f "  • ~a\n" (cog-name (gar link)))))
  active-ingredients)

;; Query 2: Find compatible ingredient pairs
(display "\nQuery: Find all compatible ingredient pairs\n")
(define compatible-pairs
  (filter
    (lambda (atom)
      (and (eq? (cog-type atom) 'EvaluationLink)
           (eq? (cog-name (gar atom)) "compatible_with")))
    (cog-get-atoms 'EvaluationLink)))

(display (format #f "Found ~a compatible pairs:\n" (length compatible-pairs)))
(for-each
  (lambda (link)
    (display (format #f "  • ~a + ~a\n" 
                     (cog-name (gadr link))
                     (cog-name (gaddr link)))))
  compatible-pairs)

;;; =============================================================================
;;; SUMMARY
;;; =============================================================================

(display "\n📋 === Formulation Analysis Summary ===\n")
(display (format #f "Total atoms in atomspace: ~a\n" (length (cog-get-atoms 'Atom))))
(display (format #f "Formulations created: 3\n"))
(display (format #f "Ingredients modeled: 10\n"))
(display (format #f "Interaction rules: ~a\n" 
                 (+ (length compatible-pairs) 
                    (length (filter
                              (lambda (atom)
                                (and (eq? (cog-type atom) 'EvaluationLink)
                                     (or (eq? (cog-name (gar atom)) "incompatible_with")
                                         (eq? (cog-name (gar atom)) "synergistic_with"))))
                              (cog-get-atoms 'EvaluationLink))))))

(display "\n✅ Complex formulation analysis completed!\n")
(display "Next steps:\n")
(display "  • Load cosmetic_compatibility.scm for interaction analysis\n")  
(display "  • Explore Python examples for numerical calculations\n")
(display "  • Implement custom reasoning rules for optimization\n")