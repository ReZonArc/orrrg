;;; cosmetic_compatibility.scm
;;;
;;; Simple Ingredient Interaction Checking
;;;
;;; This Scheme example demonstrates basic cosmetic ingredient compatibility
;;; checking using OpenCog's pattern matching and reasoning capabilities.
;;; It provides simple functions to check ingredient interactions and
;;; validate formulation safety.
;;;
;;; Requirements:
;;; - OpenCog AtomSpace with bioscience extensions loaded
;;; - Guile Scheme interpreter
;;;
;;; Usage:
;;;   guile -l cosmetic_compatibility.scm
;;;
;;; Author: OpenCog Cosmetic Chemistry Framework
;;; License: AGPL-3.0

(use-modules (opencog)
             (opencog bioscience)
             (opencog query))

;;; Load cosmetic chemistry atom types
(load "opencog/bioscience/types/bioscience_types.scm")

;;; =============================================================================
;;; BASIC INGREDIENT DEFINITIONS
;;; =============================================================================

(display "🧪 Simple Cosmetic Compatibility Checker\n")
(display "========================================\n")

;; Define common cosmetic ingredients
(define (define-ingredient name type)
  "Define a basic cosmetic ingredient with its functional type"
  (let ((ingredient (ConceptNode name)))
    (InheritanceLink ingredient (ConceptNode type))
    (display (format #f "Defined ~a as ~a\n" name type))
    ingredient))

(display "\n📋 Defining common cosmetic ingredients:\n")

;; Active ingredients
(define hyaluronic-acid (define-ingredient "hyaluronic_acid" "ACTIVE_INGREDIENT"))
(define niacinamide (define-ingredient "niacinamide" "ACTIVE_INGREDIENT"))
(define retinol (define-ingredient "retinol" "ACTIVE_INGREDIENT"))
(define vitamin-c (define-ingredient "vitamin_c" "ACTIVE_INGREDIENT"))
(define salicylic-acid (define-ingredient "salicylic_acid" "ACTIVE_INGREDIENT"))

;; Supporting ingredients
(define glycerin (define-ingredient "glycerin" "HUMECTANT"))
(define cetyl-alcohol (define-ingredient "cetyl_alcohol" "EMULSIFIER"))
(define phenoxyethanol (define-ingredient "phenoxyethanol" "PRESERVATIVE"))
(define vitamin-e (define-ingredient "vitamin_e" "ANTIOXIDANT"))

;;; =============================================================================
;;; INTERACTION RULE DEFINITIONS
;;; =============================================================================

(display "\n🔗 Defining ingredient interaction rules:\n")

(define (create-compatibility-rule ingredient1 ingredient2 description)
  "Create a compatibility rule between two ingredients"
  (let ((compatibility-link
          (EvaluationLink
            (PredicateNode "compatible_with")
            (ListLink ingredient1 ingredient2))))
    
    ;; Add description
    (EvaluationLink
      (PredicateNode "compatibility_reason")
      (ListLink compatibility-link (ConceptNode description)))
    
    (display (format #f "✓ Compatible: ~a + ~a (~a)\n" 
                     (cog-name ingredient1) 
                     (cog-name ingredient2)
                     description))
    compatibility-link))

(define (create-incompatibility-rule ingredient1 ingredient2 reason)
  "Create an incompatibility rule between two ingredients"
  (let ((incompatibility-link
          (EvaluationLink
            (PredicateNode "incompatible_with")
            (ListLink ingredient1 ingredient2))))
    
    ;; Add reason
    (EvaluationLink
      (PredicateNode "incompatibility_reason")
      (ListLink incompatibility-link (ConceptNode reason)))
    
    (display (format #f "⚠ Incompatible: ~a + ~a (~a)\n"
                     (cog-name ingredient1)
                     (cog-name ingredient2)
                     reason))
    incompatibility-link))

(define (create-synergy-rule ingredient1 ingredient2 benefit)
  "Create a synergy rule between two ingredients"
  (let ((synergy-link
          (EvaluationLink
            (PredicateNode "synergistic_with")
            (ListLink ingredient1 ingredient2))))
    
    ;; Add benefit
    (EvaluationLink
      (PredicateNode "synergy_benefit")
      (ListLink synergy-link (ConceptNode benefit)))
    
    (display (format #f "⚡ Synergistic: ~a + ~a (~a)\n"
                     (cog-name ingredient1)
                     (cog-name ingredient2)
                     benefit))
    synergy-link))

;; Define compatibility rules
(create-compatibility-rule hyaluronic-acid niacinamide
                          "both_gentle_and_hydrating")
(create-compatibility-rule niacinamide glycerin
                          "complementary_barrier_support")
(create-compatibility-rule retinol vitamin-e
                          "antioxidant_stabilization")
(create-compatibility-rule hyaluronic-acid glycerin
                          "enhanced_moisture_retention")

;; Define incompatibility rules
(create-incompatibility-rule vitamin-c retinol
                            "pH_incompatibility_and_instability")
(create-incompatibility-rule vitamin-c niacinamide
                            "potential_efficacy_reduction")
(create-incompatibility-rule salicylic-acid retinol
                            "increased_irritation_risk")

;; Define synergy rules
(create-synergy-rule vitamin-c vitamin-e
                    "enhanced_antioxidant_stability")
(create-synergy-rule hyaluronic-acid glycerin
                    "superior_hydration_effect")

;;; =============================================================================
;;; COMPATIBILITY CHECKING FUNCTIONS
;;; =============================================================================

(define (check-ingredient-compatibility ingredient1 ingredient2)
  "Check if two ingredients are compatible, incompatible, or synergistic"
  (let ((compatible (find-compatibility ingredient1 ingredient2))
        (incompatible (find-incompatibility ingredient1 ingredient2))
        (synergistic (find-synergy ingredient1 ingredient2)))
    
    (cond
      (incompatible
        (let ((reason (get-incompatibility-reason incompatible)))
          (list 'incompatible reason)))
      (synergistic
        (let ((benefit (get-synergy-benefit synergistic)))
          (list 'synergistic benefit)))
      (compatible
        (let ((reason (get-compatibility-reason compatible)))
          (list 'compatible reason)))
      (else
        (list 'unknown "no_specific_interaction_data")))))

(define (find-compatibility ingredient1 ingredient2)
  "Find compatibility link between two ingredients"
  (find
    (lambda (atom)
      (and (eq? (cog-type atom) 'EvaluationLink)
           (eq? (cog-name (gar atom)) "compatible_with")
           (or (and (eq? (gadr atom) ingredient1)
                    (eq? (gaddr atom) ingredient2))
               (and (eq? (gadr atom) ingredient2)
                    (eq? (gaddr atom) ingredient1)))))
    (cog-get-atoms 'EvaluationLink)))

(define (find-incompatibility ingredient1 ingredient2)
  "Find incompatibility link between two ingredients"
  (find
    (lambda (atom)
      (and (eq? (cog-type atom) 'EvaluationLink)
           (eq? (cog-name (gar atom)) "incompatible_with")
           (or (and (eq? (gadr atom) ingredient1)
                    (eq? (gaddr atom) ingredient2))
               (and (eq? (gadr atom) ingredient2)
                    (eq? (gaddr atom) ingredient1)))))
    (cog-get-atoms 'EvaluationLink)))

(define (find-synergy ingredient1 ingredient2)
  "Find synergy link between two ingredients"
  (find
    (lambda (atom)
      (and (eq? (cog-type atom) 'EvaluationLink)
           (eq? (cog-name (gar atom)) "synergistic_with")
           (or (and (eq? (gadr atom) ingredient1)
                    (eq? (gaddr atom) ingredient2))
               (and (eq? (gadr atom) ingredient2)
                    (eq? (gaddr atom) ingredient1)))))
    (cog-get-atoms 'EvaluationLink)))

(define (get-compatibility-reason compatibility-link)
  "Get the reason for compatibility"
  (let ((reason-link
          (find
            (lambda (atom)
              (and (eq? (cog-type atom) 'EvaluationLink)
                   (eq? (cog-name (gar atom)) "compatibility_reason")
                   (eq? (gadr atom) compatibility-link)))
            (cog-get-atoms 'EvaluationLink))))
    (if reason-link
        (cog-name (gaddr reason-link))
        "unknown_reason")))

(define (get-incompatibility-reason incompatibility-link)
  "Get the reason for incompatibility"
  (let ((reason-link
          (find
            (lambda (atom)
              (and (eq? (cog-type atom) 'EvaluationLink)
                   (eq? (cog-name (gar atom)) "incompatibility_reason")
                   (eq? (gadr atom) incompatibility-link)))
            (cog-get-atoms 'EvaluationLink))))
    (if reason-link
        (cog-name (gaddr reason-link))
        "unknown_reason")))

(define (get-synergy-benefit synergy-link)
  "Get the benefit of synergy"
  (let ((benefit-link
          (find
            (lambda (atom)
              (and (eq? (cog-type atom) 'EvaluationLink)
                   (eq? (cog-name (gar atom)) "synergy_benefit")
                   (eq? (gadr atom) synergy-link)))
            (cog-get-atoms 'EvaluationLink))))
    (if benefit-link
        (cog-name (gaddr benefit-link))
        "unknown_benefit")))

;;; =============================================================================
;;; FORMULATION VALIDATION FUNCTIONS
;;; =============================================================================

(define (validate-ingredient-list ingredients)
  "Validate a list of ingredients for compatibility issues"
  (display (format #f "\n🔍 Validating ingredient list: ~a\n"
                   (map cog-name ingredients)))
  
  (define issues '())
  (define synergies '())
  (define compatibilities '())
  
  ;; Check all pairs
  (for-each
    (lambda (ingredient1)
      (for-each
        (lambda (ingredient2)
          (when (not (eq? ingredient1 ingredient2))
            (let ((result (check-ingredient-compatibility ingredient1 ingredient2)))
              (case (car result)
                ((incompatible)
                 (set! issues (cons (list ingredient1 ingredient2 (cadr result)) issues)))
                ((synergistic)
                 (set! synergies (cons (list ingredient1 ingredient2 (cadr result)) synergies)))
                ((compatible)
                 (set! compatibilities (cons (list ingredient1 ingredient2 (cadr result)) compatibilities)))))))
        ingredients))
    ingredients)
  
  ;; Report results
  (if (null? issues)
      (display "  ✅ No incompatibility issues found\n")
      (begin
        (display "  ⚠️ Incompatibility issues detected:\n")
        (for-each
          (lambda (issue)
            (display (format #f "    • ~a + ~a: ~a\n"
                             (cog-name (car issue))
                             (cog-name (cadr issue))
                             (caddr issue))))
          (delete-duplicates issues))))
  
  (if (null? synergies)
      (display "  💡 No synergies detected\n")
      (begin
        (display "  ⚡ Synergistic combinations found:\n")
        (for-each
          (lambda (synergy)
            (display (format #f "    • ~a + ~a: ~a\n"
                             (cog-name (car synergy))
                             (cog-name (cadr synergy))
                             (caddr synergy))))
          (delete-duplicates synergies))))
  
  (if (null? compatibilities)
      (display "  📝 No specific compatibilities noted\n")
      (begin
        (display "  ✅ Compatible combinations:\n")
        (for-each
          (lambda (compat)
            (display (format #f "    • ~a + ~a: ~a\n"
                             (cog-name (car compat))
                             (cog-name (cadr compat))
                             (caddr compat))))
          (delete-duplicates compatibilities))))
  
  (list issues synergies compatibilities))

;;; =============================================================================
;;; EXAMPLE USAGE AND DEMONSTRATIONS
;;; =============================================================================

(display "\n🧪 === Compatibility Check Examples ===\n")

;; Example 1: Check individual ingredient pairs
(display "\n1. Individual Ingredient Pair Checks:\n")

(define test-pairs
  `((,hyaluronic-acid ,niacinamide)
    (,vitamin-c ,retinol)
    (,vitamin-c ,vitamin-e)
    (,retinol ,salicylic-acid)
    (,niacinamide ,glycerin)))

(for-each
  (lambda (pair)
    (let* ((ing1 (car pair))
           (ing2 (cadr pair))
           (result (check-ingredient-compatibility ing1 ing2))
           (status (car result))
           (reason (cadr result)))
      (case status
        ((compatible)
         (display (format #f "   ✅ ~a + ~a: Compatible (~a)\n"
                          (cog-name ing1) (cog-name ing2) reason)))
        ((incompatible)
         (display (format #f "   ❌ ~a + ~a: Incompatible (~a)\n"
                          (cog-name ing1) (cog-name ing2) reason)))
        ((synergistic)
         (display (format #f "   ⚡ ~a + ~a: Synergistic (~a)\n"
                          (cog-name ing1) (cog-name ing2) reason)))
        ((unknown)
         (display (format #f "   ❓ ~a + ~a: Unknown interaction\n"
                          (cog-name ing1) (cog-name ing2)))))))
  test-pairs)

;; Example 2: Validate complete formulations
(display "\n2. Complete Formulation Validation:\n")

;; Good formulation example
(define good-formulation
  (list hyaluronic-acid niacinamide glycerin cetyl-alcohol phenoxyethanol))

(display "\nGood Formulation Example:")
(validate-ingredient-list good-formulation)

;; Problematic formulation example
(define problematic-formulation
  (list vitamin-c retinol niacinamide salicylic-acid phenoxyethanol))

(display "\nProblematic Formulation Example:")
(validate-ingredient-list problematic-formulation)

;; Synergistic formulation example
(define synergistic-formulation
  (list vitamin-c vitamin-e hyaluronic-acid glycerin phenoxyethanol))

(display "\nSynergistic Formulation Example:")
(validate-ingredient-list synergistic-formulation)

;;; =============================================================================
;;; UTILITY FUNCTIONS FOR RECOMMENDATIONS
;;; =============================================================================

(define (recommend-ingredient-alternatives problematic-ingredient formulation)
  "Recommend alternatives for a problematic ingredient"
  (display (format #f "\n💡 Alternatives for ~a in current formulation:\n"
                   (cog-name problematic-ingredient)))
  
  ;; Find ingredients that cause issues
  (define problematic-pairs '())
  (for-each
    (lambda (ingredient)
      (when (not (eq? ingredient problematic-ingredient))
        (let ((result (check-ingredient-compatibility problematic-ingredient ingredient)))
          (when (eq? (car result) 'incompatible)
            (set! problematic-pairs (cons ingredient problematic-pairs))))))
    formulation)
  
  (if (null? problematic-pairs)
      (display "  ✅ No issues with current formulation\n")
      (begin
        (display "  Issues with:\n")
        (for-each
          (lambda (ing)
            (display (format #f "    • ~a\n" (cog-name ing))))
          problematic-pairs)
        
        ;; Simple recommendations based on ingredient type
        (let ((ingredient-type (get-ingredient-type problematic-ingredient)))
          (case ingredient-type
            ((ACTIVE_INGREDIENT)
             (display "  Suggested alternatives:\n")
             (display "    • Consider gentler actives\n")
             (display "    • Use time-release formulations\n")
             (display "    • Implement alternating use protocols\n"))
            (else
             (display "  Consider reformulation without this ingredient\n")))))))

(define (get-ingredient-type ingredient)
  "Get the primary type of an ingredient"
  (let ((type-link
          (find
            (lambda (atom)
              (and (eq? (cog-type atom) 'InheritanceLink)
                   (eq? (gar atom) ingredient)))
            (cog-get-atoms 'InheritanceLink))))
    (if type-link
        (string->symbol (cog-name (gdr type-link)))
        'UNKNOWN)))

;;; =============================================================================
;;; INTERACTIVE QUERY FUNCTIONS
;;; =============================================================================

(define (query-compatible-with ingredient)
  "Query all ingredients compatible with the given ingredient"
  (display (format #f "\n🔍 Ingredients compatible with ~a:\n"
                   (cog-name ingredient)))
  
  (define compatible-ingredients
    (filter-map
      (lambda (atom)
        (and (eq? (cog-type atom) 'EvaluationLink)
             (eq? (cog-name (gar atom)) "compatible_with")
             (cond
               ((eq? (gadr atom) ingredient) (gaddr atom))
               ((eq? (gaddr atom) ingredient) (gadr atom))
               (else #f))))
      (cog-get-atoms 'EvaluationLink)))
  
  (if (null? compatible-ingredients)
      (display "  No specific compatibility data available\n")
      (for-each
        (lambda (compatible-ing)
          (display (format #f "  ✅ ~a\n" (cog-name compatible-ing))))
        compatible-ingredients))
  
  compatible-ingredients)

(define (query-incompatible-with ingredient)
  "Query all ingredients incompatible with the given ingredient"
  (display (format #f "\n⚠️ Ingredients incompatible with ~a:\n"
                   (cog-name ingredient)))
  
  (define incompatible-ingredients
    (filter-map
      (lambda (atom)
        (and (eq? (cog-type atom) 'EvaluationLink)
             (eq? (cog-name (gar atom)) "incompatible_with")
             (cond
               ((eq? (gadr atom) ingredient) (gaddr atom))
               ((eq? (gaddr atom) ingredient) (gadr atom))
               (else #f))))
      (cog-get-atoms 'EvaluationLink)))
  
  (if (null? incompatible-ingredients)
      (display "  No incompatibility data available\n")
      (for-each
        (lambda (incompatible-ing)
          (display (format #f "  ❌ ~a\n" (cog-name incompatible-ing))))
        incompatible-ingredients))
  
  incompatible-ingredients)

;;; =============================================================================
;;; DEMONSTRATION OF QUERY FUNCTIONS
;;; =============================================================================

(display "\n🔍 === Interactive Query Examples ===\n")

;; Query compatibility for specific ingredients
(query-compatible-with vitamin-c)
(query-incompatible-with vitamin-c)
(query-compatible-with retinol)
(query-incompatible-with retinol)

;; Demonstrate recommendation system
(recommend-ingredient-alternatives vitamin-c problematic-formulation)

;;; =============================================================================
;;; SUMMARY AND STATISTICS
;;; =============================================================================

(display "\n📊 === Compatibility Database Summary ===\n")

(define total-atoms (length (cog-get-atoms 'Atom)))
(define total-ingredients (length (filter
                                    (lambda (atom)
                                      (eq? (cog-type atom) 'ConceptNode))
                                    (cog-get-atoms 'ConceptNode))))
(define total-compatibility-rules (length (filter
                                            (lambda (atom)
                                              (and (eq? (cog-type atom) 'EvaluationLink)
                                                   (member (cog-name (gar atom))
                                                          '("compatible_with" "incompatible_with" "synergistic_with"))))
                                            (cog-get-atoms 'EvaluationLink))))

(display (format #f "Total atoms in knowledge base: ~a\n" total-atoms))
(display (format #f "Cosmetic ingredients defined: ~a\n" total-ingredients))
(display (format #f "Interaction rules established: ~a\n" total-compatibility-rules))

(display "\nInteraction rule breakdown:\n")
(for-each
  (lambda (rule-type)
    (let ((count (length (filter
                           (lambda (atom)
                             (and (eq? (cog-type atom) 'EvaluationLink)
                                  (eq? (cog-name (gar atom)) rule-type)))
                           (cog-get-atoms 'EvaluationLink)))))
      (display (format #f "  • ~a: ~a rules\n" rule-type count))))
  '("compatible_with" "incompatible_with" "synergistic_with"))

(display "\n✅ Simple compatibility checking system ready!\n")
(display "\nUsage examples:\n")
(display "  • (check-ingredient-compatibility ingredient1 ingredient2)\n")
(display "  • (validate-ingredient-list '(ing1 ing2 ing3 ...))\n")
(display "  • (query-compatible-with ingredient)\n")
(display "  • (query-incompatible-with ingredient)\n")
(display "\nNext steps:\n")
(display "  • Load cosmetic_formulation.scm for advanced analysis\n")
(display "  • Add more ingredients and interaction rules\n")
(display "  • Implement reasoning for formulation optimization\n")