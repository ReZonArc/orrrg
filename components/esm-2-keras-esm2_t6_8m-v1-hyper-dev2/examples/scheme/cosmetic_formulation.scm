#!/usr/bin/opencog/scm
;;;
;;; Cosmetic Formulation Analysis - Complex Formulation Modeling
;;;
;;; This example demonstrates complex formulation modeling with compatibility
;;; analysis using the OpenCog Cheminformatics Framework atom types for
;;; cosmetic chemistry applications.
;;;
;;; Usage: Load this file in OpenCog and execute the examples
;;;

(use-modules (opencog))
(use-modules (opencog atom-types))

;; Load cosmetic chemistry atom types
(load "../../cheminformatics/types/atom_types.script")

;;;
;;; INGREDIENT DEFINITIONS
;;; Define cosmetic ingredients with their properties
;;;

;; Active Ingredients
(define hyaluronic-acid
  (ACTIVE_INGREDIENT
    (ConceptNode "hyaluronic_acid")
    (ListLink
      (ConceptNode "hydration")
      (ConceptNode "plumping")
      (ConceptNode "surface_moisturization"))))

(define retinol
  (ACTIVE_INGREDIENT
    (ConceptNode "retinol")
    (ListLink
      (ConceptNode "anti_aging")
      (ConceptNode "cellular_turnover")
      (ConceptNode "collagen_stimulation"))))

(define niacinamide
  (ACTIVE_INGREDIENT
    (ConceptNode "niacinamide")
    (ListLink
      (ConceptNode "pore_minimizing")
      (ConceptNode "oil_control")
      (ConceptNode "barrier_strengthening"))))

(define vitamin-c
  (ACTIVE_INGREDIENT
    (ConceptNode "vitamin_c")
    (ListLink
      (ConceptNode "antioxidant")
      (ConceptNode "brightening")
      (ConceptNode "collagen_synthesis"))))

(define bakuchiol
  (ACTIVE_INGREDIENT
    (ConceptNode "bakuchiol")
    (ListLink
      (ConceptNode "retinol_alternative")
      (ConceptNode "gentle_anti_aging")
      (ConceptNode "pregnancy_safe"))))

;; Supporting Ingredients
(define glycerin
  (HUMECTANT
    (ConceptNode "glycerin")
    (ListLink
      (ConceptNode "moisture_retention")
      (ConceptNode "skin_conditioning")
      (ConceptNode "humectancy"))))

(define cetyl-alcohol
  (EMULSIFIER
    (ConceptNode "cetyl_alcohol")
    (ListLink
      (ConceptNode "emulsification")
      (ConceptNode "thickening")
      (ConceptNode "stability"))))

(define phenoxyethanol
  (PRESERVATIVE
    (ConceptNode "phenoxyethanol")
    (ListLink
      (ConceptNode "broad_spectrum")
      (ConceptNode "antimicrobial")
      (ConceptNode "globally_accepted"))))

(define vitamin-e
  (ANTIOXIDANT
    (ConceptNode "vitamin_e")
    (ListLink
      (ConceptNode "antioxidant")
      (ConceptNode "stabilizer")
      (ConceptNode "lipid_protection"))))

;;;
;;; PROPERTY DEFINITIONS
;;; Define physical and chemical properties
;;;

;; pH Properties
(PH_PROPERTY
  (ConceptNode "hyaluronic_acid")
  (NumberNode 4.0)
  (NumberNode 7.0))

(PH_PROPERTY
  (ConceptNode "retinol")
  (NumberNode 5.5)
  (NumberNode 6.5))

(PH_PROPERTY
  (ConceptNode "vitamin_c")
  (NumberNode 3.5)
  (NumberNode 4.0))

(PH_PROPERTY
  (ConceptNode "niacinamide")
  (NumberNode 5.0)
  (NumberNode 7.0))

;; Concentration Limits
(CONCENTRATION_LIMIT
  (ConceptNode "retinol")
  (NumberNode 1.0)) ; 1.0% maximum

(CONCENTRATION_LIMIT
  (ConceptNode "vitamin_c")
  (NumberNode 20.0)) ; 20% maximum

(CONCENTRATION_LIMIT
  (ConceptNode "niacinamide")
  (NumberNode 10.0)) ; 10% maximum

(CONCENTRATION_LIMIT
  (ConceptNode "phenoxyethanol")
  (NumberNode 1.0)) ; 1.0% maximum

;;;
;;; INTERACTION DEFINITIONS
;;; Define ingredient interactions and compatibility
;;;

;; Compatible Combinations
(COMPATIBILITY_LINK
  (ConceptNode "hyaluronic_acid")
  (ConceptNode "niacinamide"))

(COMPATIBILITY_LINK
  (ConceptNode "hyaluronic_acid")
  (ConceptNode "vitamin_e"))

(COMPATIBILITY_LINK
  (ConceptNode "niacinamide")
  (ConceptNode "glycerin"))

(COMPATIBILITY_LINK
  (ConceptNode "retinol")
  (ConceptNode "hyaluronic_acid"))

(COMPATIBILITY_LINK
  (ConceptNode "bakuchiol")
  (ConceptNode "hyaluronic_acid"))

;; Incompatible Combinations
(INCOMPATIBILITY_LINK
  (ConceptNode "vitamin_c")
  (ConceptNode "retinol"))

(INCOMPATIBILITY_LINK
  (ConceptNode "vitamin_c")
  (ConceptNode "niacinamide"))

;; Synergistic Combinations
(SYNERGY_LINK
  (ConceptNode "vitamin_c")
  (ConceptNode "vitamin_e"))

(SYNERGY_LINK
  (ConceptNode "hyaluronic_acid")
  (ConceptNode "glycerin"))

(SYNERGY_LINK
  (ConceptNode "retinol")
  (ConceptNode "vitamin_e"))

;;;
;;; FORMULATION DEFINITIONS
;;; Create complete cosmetic formulations
;;;

;; Anti-Aging Night Serum (Premium)
(define anti-aging-serum
  (SKINCARE_FORMULATION
    (ConceptNode "anti_aging_night_serum")
    (ListLink
      retinol
      hyaluronic-acid
      vitamin-e
      phenoxyethanol)))

;; Vitamin C Morning Serum
(define vitamin-c-serum
  (SKINCARE_FORMULATION
    (ConceptNode "vitamin_c_morning_serum")
    (ListLink
      vitamin-c
      vitamin-e
      hyaluronic-acid
      phenoxyethanol)))

;; Gentle Daily Moisturizer
(define gentle-moisturizer
  (SKINCARE_FORMULATION
    (ConceptNode "gentle_daily_moisturizer")
    (ListLink
      bakuchiol
      hyaluronic-acid
      niacinamide
      glycerin
      cetyl-alcohol
      phenoxyethanol)))

;; Problem Formulation (for testing)
(define problem-formulation
  (SKINCARE_FORMULATION
    (ConceptNode "problem_formulation_test")
    (ListLink
      vitamin-c
      retinol
      niacinamide)))

;;;
;;; FORMULATION ANALYSIS FUNCTIONS
;;; Analyze formulations for compatibility and optimization
;;;

;; Check if two ingredients are compatible
(define (ingredients-compatible? ingredient1 ingredient2)
  (or
    ;; Check for explicit compatibility
    (cog-atom-at-key
      (COMPATIBILITY_LINK
        ingredient1
        ingredient2))
    (cog-atom-at-key
      (COMPATIBILITY_LINK
        ingredient2
        ingredient1))
    ;; Check for synergy (implies compatibility)
    (cog-atom-at-key
      (SYNERGY_LINK
        ingredient1
        ingredient2))
    (cog-atom-at-key
      (SYNERGY_LINK
        ingredient2
        ingredient1))))

;; Check if two ingredients are incompatible
(define (ingredients-incompatible? ingredient1 ingredient2)
  (or
    (cog-atom-at-key
      (INCOMPATIBILITY_LINK
        ingredient1
        ingredient2))
    (cog-atom-at-key
      (INCOMPATIBILITY_LINK
        ingredient2
        ingredient1))))

;; Check if two ingredients are synergistic
(define (ingredients-synergistic? ingredient1 ingredient2)
  (or
    (cog-atom-at-key
      (SYNERGY_LINK
        ingredient1
        ingredient2))
    (cog-atom-at-key
      (SYNERGY_LINK
        ingredient2
        ingredient1))))

;; Extract ingredients from formulation
(define (get-formulation-ingredients formulation)
  (cog-outgoing-set 
    (cog-outgoing-atom formulation 1)))

;; Validate formulation compatibility
(define (validate-formulation formulation)
  (let* ((ingredients (get-formulation-ingredients formulation))
         (ingredient-concepts (map (lambda (ing) 
                                    (cog-outgoing-atom ing 0))
                                  ingredients))
         (incompatibilities '())
         (synergies '())
         (warnings '()))
    
    ;; Check all ingredient pairs
    (for-each
      (lambda (ing1)
        (for-each
          (lambda (ing2)
            (when (not (equal? ing1 ing2))
              (cond
                ;; Check for incompatibilities
                ((ingredients-incompatible? ing1 ing2)
                 (set! incompatibilities
                   (cons (list ing1 ing2) incompatibilities)))
                ;; Check for synergies
                ((ingredients-synergistic? ing1 ing2)
                 (set! synergies
                   (cons (list ing1 ing2) synergies))))))
          ingredient-concepts))
      ingredient-concepts)
    
    ;; Return validation results
    (list
      (cons 'formulation formulation)
      (cons 'valid (null? incompatibilities))
      (cons 'incompatibilities incompatibilities)
      (cons 'synergies synergies)
      (cons 'warnings warnings))))

;; Find optimal pH range for formulation
(define (find-optimal-ph-range formulation)
  (let* ((ingredients (get-formulation-ingredients formulation))
         (ingredient-concepts (map (lambda (ing) 
                                    (cog-outgoing-atom ing 0))
                                  ingredients))
         (ph-ranges '()))
    
    ;; Collect pH ranges for all ingredients
    (for-each
      (lambda (ingredient)
        (let ((ph-property (cog-atom-at-key
                           (PH_PROPERTY
                             ingredient
                             (VariableNode "$ph_min")
                             (VariableNode "$ph_max")))))
          (when ph-property
            (let ((ph-min (cog-number (cog-outgoing-atom ph-property 1)))
                  (ph-max (cog-number (cog-outgoing-atom ph-property 2))))
              (set! ph-ranges (cons (list ph-min ph-max) ph-ranges))))))
      ingredient-concepts)
    
    ;; Calculate optimal range
    (if (null? ph-ranges)
        '(4.5 7.0) ; Default range
        (let ((min-ph (apply max (map car ph-ranges)))
              (max-ph (apply min (map cadr ph-ranges))))
          (if (> min-ph max-ph)
              '() ; No compatible pH range
              (list min-ph max-ph))))))

;; Suggest formulation improvements
(define (suggest-improvements formulation)
  (let* ((validation (validate-formulation formulation))
         (incompatibilities (cdr (assoc 'incompatibilities validation)))
         (suggestions '()))
    
    ;; Suggest alternatives for incompatible ingredients
    (for-each
      (lambda (incompatible-pair)
        (let ((ing1 (car incompatible-pair))
              (ing2 (cadr incompatible-pair)))
          (cond
            ;; Vitamin C + Retinol -> suggest separate application
            ((and (equal? (cog-name ing1) "vitamin_c")
                  (equal? (cog-name ing2) "retinol"))
             (set! suggestions
               (cons "Separate Vitamin C (morning) and Retinol (evening) application"
                     suggestions)))
            ;; Retinol sensitivity -> suggest bakuchiol
            ((equal? (cog-name ing1) "retinol")
             (set! suggestions
               (cons "Consider replacing Retinol with Bakuchiol for sensitive skin"
                     suggestions)))
            ;; General incompatibility
            (else
             (set! suggestions
               (cons (string-append "Avoid combining "
                                   (cog-name ing1)
                                   " with "
                                   (cog-name ing2))
                     suggestions))))))
      incompatibilities)
    
    ;; Add general suggestions
    (when (null? incompatibilities)
      (set! suggestions
        (cons "Formulation shows good ingredient compatibility"
              suggestions)))
    
    suggestions))

;;;
;;; ADVANCED ANALYSIS FUNCTIONS
;;; More sophisticated formulation analysis
;;;

;; Calculate formulation complexity score
(define (calculate-complexity-score formulation)
  (let* ((ingredients (get-formulation-ingredients formulation))
         (num-ingredients (length ingredients))
         (validation (validate-formulation formulation))
         (num-incompatibilities (length (cdr (assoc 'incompatibilities validation))))
         (num-synergies (length (cdr (assoc 'synergies validation)))))
    
    ;; Base complexity from number of ingredients
    (let ((base-score (* num-ingredients 0.1))
          (incompatibility-penalty (* num-incompatibilities 0.3))
          (synergy-bonus (* num-synergies 0.1)))
      (+ base-score incompatibility-penalty (- synergy-bonus)))))

;; Generate formulation report
(define (generate-formulation-report formulation)
  (let* ((validation (validate-formulation formulation))
         (ph-range (find-optimal-ph-range formulation))
         (improvements (suggest-improvements formulation))
         (complexity (calculate-complexity-score formulation)))
    
    (display "=== FORMULATION ANALYSIS REPORT ===\n")
    (display (string-append "Formulation: " 
                            (cog-name (cog-outgoing-atom formulation 0)) "\n"))
    (display (string-append "Valid: " 
                            (if (cdr (assoc 'valid validation)) "YES" "NO") "\n"))
    (display (string-append "Complexity Score: " 
                            (number->string complexity) "\n"))
    
    ;; pH Range
    (if (null? ph-range)
        (display "pH Range: INCOMPATIBLE - No suitable pH range found\n")
        (display (string-append "Optimal pH Range: " 
                                (number->string (car ph-range)) 
                                " - " 
                                (number->string (cadr ph-range)) "\n")))
    
    ;; Incompatibilities
    (let ((incompatibilities (cdr (assoc 'incompatibilities validation))))
      (when (not (null? incompatibilities))
        (display "\nINCOMPATIBILITIES:\n")
        (for-each
          (lambda (pair)
            (display (string-append "  ❌ " 
                                   (cog-name (car pair)) 
                                   " + " 
                                   (cog-name (cadr pair)) "\n")))
          incompatibilities)))
    
    ;; Synergies
    (let ((synergies (cdr (assoc 'synergies validation))))
      (when (not (null? synergies))
        (display "\nSYNERGIES:\n")
        (for-each
          (lambda (pair)
            (display (string-append "  ✅ " 
                                   (cog-name (car pair)) 
                                   " + " 
                                   (cog-name (cadr pair)) "\n")))
          synergies)))
    
    ;; Suggestions
    (when (not (null? improvements))
      (display "\nSUGGESTIONS:\n")
      (for-each
        (lambda (suggestion)
          (display (string-append "  💡 " suggestion "\n")))
        improvements))
    
    (display "\n")))

;;;
;;; EXAMPLE DEMONSTRATIONS
;;; Demonstrate the cosmetic formulation analysis system
;;;

(define (run-formulation-examples)
  (display "🧴 COSMETIC FORMULATION ANALYSIS EXAMPLES\n")
  (display "=========================================\n\n")
  
  ;; Example 1: Analyze anti-aging serum
  (display "1. Anti-Aging Night Serum Analysis\n")
  (display "-----------------------------------\n")
  (generate-formulation-report anti-aging-serum)
  
  ;; Example 2: Analyze vitamin C serum
  (display "2. Vitamin C Morning Serum Analysis\n")
  (display "-----------------------------------\n")
  (generate-formulation-report vitamin-c-serum)
  
  ;; Example 3: Analyze gentle moisturizer
  (display "3. Gentle Daily Moisturizer Analysis\n")
  (display "------------------------------------\n")
  (generate-formulation-report gentle-moisturizer)
  
  ;; Example 4: Analyze problematic formulation
  (display "4. Problem Formulation Analysis\n")
  (display "-------------------------------\n")
  (generate-formulation-report problem-formulation)
  
  ;; Example 5: Ingredient compatibility matrix
  (display "5. Ingredient Compatibility Matrix\n")
  (display "----------------------------------\n")
  (let ((test-ingredients (list 
                          (ConceptNode "hyaluronic_acid")
                          (ConceptNode "retinol")
                          (ConceptNode "vitamin_c")
                          (ConceptNode "niacinamide")
                          (ConceptNode "bakuchiol"))))
    (display "     ")
    (for-each (lambda (ing) 
                (display (string-append (substring (cog-name ing) 0 8) " ")))
              test-ingredients)
    (display "\n")
    (for-each
      (lambda (ing1)
        (display (string-append (substring (cog-name ing1) 0 8) " "))
        (for-each
          (lambda (ing2)
            (cond
              ((equal? ing1 ing2) (display "   -    "))
              ((ingredients-synergistic? ing1 ing2) (display "   +    "))
              ((ingredients-incompatible? ing1 ing2) (display "   X    "))
              ((ingredients-compatible? ing1 ing2) (display "   ✓    "))
              (else (display "   ?    "))))
          test-ingredients)
        (display "\n"))
      test-ingredients)
    (display "\nLegend: + = Synergistic, ✓ = Compatible, X = Incompatible, ? = Unknown\n\n"))
  
  (display "✅ All formulation examples completed!\n"))

;;;
;;; QUERY EXAMPLES
;;; Demonstrate OpenCog querying capabilities
;;;

;; Find all compatible ingredient pairs
(define find-compatible-pairs
  (GetLink
    (VariableList
      (VariableNode "$ingredient1")
      (VariableNode "$ingredient2"))
    (COMPATIBILITY_LINK
      (VariableNode "$ingredient1")
      (VariableNode "$ingredient2"))))

;; Find all synergistic combinations
(define find-synergistic-pairs
  (GetLink
    (VariableList
      (VariableNode "$ingredient1")
      (VariableNode "$ingredient2"))
    (SYNERGY_LINK
      (VariableNode "$ingredient1")
      (VariableNode "$ingredient2"))))

;; Find all incompatible combinations
(define find-incompatible-pairs
  (GetLink
    (VariableList
      (VariableNode "$ingredient1")
      (VariableNode "$ingredient2"))
    (INCOMPATIBILITY_LINK
      (VariableNode "$ingredient1")
      (VariableNode "$ingredient2"))))

;; Find ingredients compatible with hyaluronic acid
(define find-hyaluronic-compatible
  (GetLink
    (VariableNode "$ingredient")
    (COMPATIBILITY_LINK
      (ConceptNode "hyaluronic_acid")
      (VariableNode "$ingredient"))))

(define (run-query-examples)
  (display "🔍 OPENCOG QUERY EXAMPLES\n")
  (display "========================\n\n")
  
  (display "Compatible ingredient pairs:\n")
  (let ((compatible-pairs (cog-execute find-compatible-pairs)))
    (for-each
      (lambda (pair)
        (display (string-append "  • " 
                                (cog-name (cog-outgoing-atom pair 0))
                                " + "
                                (cog-name (cog-outgoing-atom pair 1))
                                "\n")))
      (cog-outgoing-set compatible-pairs)))
  
  (display "\nSynergistic combinations:\n")
  (let ((synergistic-pairs (cog-execute find-synergistic-pairs)))
    (for-each
      (lambda (pair)
        (display (string-append "  • " 
                                (cog-name (cog-outgoing-atom pair 0))
                                " + "
                                (cog-name (cog-outgoing-atom pair 1))
                                "\n")))
      (cog-outgoing-set synergistic-pairs)))
  
  (display "\nIncompatible combinations:\n")
  (let ((incompatible-pairs (cog-execute find-incompatible-pairs)))
    (for-each
      (lambda (pair)
        (display (string-append "  • " 
                                (cog-name (cog-outgoing-atom pair 0))
                                " + "
                                (cog-name (cog-outgoing-atom pair 1))
                                "\n")))
      (cog-outgoing-set incompatible-pairs)))
  
  (display "\n✅ Query examples completed!\n"))

;;;
;;; MAIN EXECUTION
;;; Run all examples when this file is loaded
;;;

(display "Loading cosmetic formulation analysis system...\n")
(run-formulation-examples)
(run-query-examples)
(display "\n📋 System ready for interactive analysis!\n")
(display "Try: (generate-formulation-report <your-formulation>)\n")