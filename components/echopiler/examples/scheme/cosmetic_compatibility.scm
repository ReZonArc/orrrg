;; Simple Ingredient Compatibility Checking Example
;; 
;; This Scheme script demonstrates basic ingredient interaction checking
;; for cosmetic chemistry applications using the OpenCog framework.

;; Load required modules
(use-modules (opencog)
             (opencog query)
             (opencog exec))

;; =============================================================================
;; BASIC INGREDIENT SETUP
;; =============================================================================

;; Define common cosmetic ingredients
(define ingredients
  (list
    (ConceptNode "hyaluronic_acid")
    (ConceptNode "niacinamide")
    (ConceptNode "vitamin_c")
    (ConceptNode "retinol")
    (ConceptNode "vitamin_e")
    (ConceptNode "glycerin")
    (ConceptNode "salicylic_acid")
    (ConceptNode "benzoyl_peroxide")
    (ConceptNode "alpha_arbutin")
    (ConceptNode "peptides")))

;; Classify ingredients by type
(InheritanceLink (ConceptNode "hyaluronic_acid") (ConceptNode "HUMECTANT"))
(InheritanceLink (ConceptNode "niacinamide") (ConceptNode "ACTIVE_INGREDIENT"))
(InheritanceLink (ConceptNode "vitamin_c") (ConceptNode "ANTIOXIDANT"))
(InheritanceLink (ConceptNode "retinol") (ConceptNode "ACTIVE_INGREDIENT"))
(InheritanceLink (ConceptNode "vitamin_e") (ConceptNode "ANTIOXIDANT"))
(InheritanceLink (ConceptNode "glycerin") (ConceptNode "HUMECTANT"))
(InheritanceLink (ConceptNode "salicylic_acid") (ConceptNode "BHA_EXFOLIANT"))
(InheritanceLink (ConceptNode "benzoyl_peroxide") (ConceptNode "ANTIMICROBIAL"))
(InheritanceLink (ConceptNode "alpha_arbutin") (ConceptNode "BRIGHTENING_AGENT"))
(InheritanceLink (ConceptNode "peptides") (ConceptNode "ACTIVE_INGREDIENT"))

;; =============================================================================
;; COMPATIBILITY RELATIONSHIPS
;; =============================================================================

;; Define compatible pairs (work well together)
(define compatible-pairs
  '((hyaluronic_acid niacinamide)
    (hyaluronic_acid glycerin)  
    (hyaluronic_acid alpha_arbutin)
    (hyaluronic_acid peptides)
    (niacinamide glycerin)
    (niacinamide alpha_arbutin)
    (niacinamide peptides)
    (vitamin_c vitamin_e)
    (vitamin_e alpha_arbutin)
    (glycerin alpha_arbutin)
    (glycerin peptides)
    (alpha_arbutin peptides)))

;; Create compatibility links
(for-each
  (lambda (pair)
    (EvaluationLink
      (PredicateNode "compatible")
      (ListLink (ConceptNode (symbol->string (car pair)))
                (ConceptNode (symbol->string (cadr pair))))))
  compatible-pairs)

;; Define incompatible pairs (should not be used together)
(define incompatible-pairs
  '((vitamin_c retinol "pH_incompatibility")
    (vitamin_c niacinamide "potential_pH_conflict")
    (retinol salicylic_acid "irritation_risk")
    (retinol benzoyl_peroxide "excessive_irritation")
    (salicylic_acid benzoyl_peroxide "over_exfoliation")))

;; Create incompatibility links with reasons
(for-each
  (lambda (pair)
    (EvaluationLink
      (PredicateNode "incompatible")
      (ListLink (ConceptNode (symbol->string (car pair)))
                (ConceptNode (symbol->string (cadr pair)))))
    (EvaluationLink
      (PredicateNode "incompatible_reason")
      (ListLink 
        (ListLink (ConceptNode (symbol->string (car pair)))
                  (ConceptNode (symbol->string (cadr pair))))
        (ConceptNode (symbol->string (caddr pair))))))
  incompatible-pairs)

;; Define synergistic pairs (enhance each other's effects)
(define synergistic-pairs
  '((vitamin_c vitamin_e "antioxidant_network")
    (hyaluronic_acid glycerin "enhanced_hydration")
    (niacinamide alpha_arbutin "brightening_synergy")))

;; Create synergy links with mechanisms
(for-each
  (lambda (pair)
    (EvaluationLink
      (PredicateNode "synergistic")
      (ListLink (ConceptNode (symbol->string (car pair)))
                (ConceptNode (symbol->string (cadr pair)))))
    (EvaluationLink
      (PredicateNode "synergy_mechanism")
      (ListLink 
        (ListLink (ConceptNode (symbol->string (car pair)))
                  (ConceptNode (symbol->string (cadr pair))))
        (ConceptNode (symbol->string (caddr pair))))))
  synergistic-pairs)

;; =============================================================================
;; QUERY FUNCTIONS
;; =============================================================================

;; Query to find all compatible partners for a given ingredient
(define find-compatible-with
  (lambda (ingredient)
    (cog-execute!
      (GetLink
        (VariableNode "$partner")
        (OrLink
          (EvaluationLink
            (PredicateNode "compatible")
            (ListLink ingredient (VariableNode "$partner")))
          (EvaluationLink
            (PredicateNode "compatible")
            (ListLink (VariableNode "$partner") ingredient)))))))

;; Query to find all incompatible partners for a given ingredient
(define find-incompatible-with
  (lambda (ingredient)
    (cog-execute!
      (GetLink
        (VariableNode "$partner")
        (OrLink
          (EvaluationLink
            (PredicateNode "incompatible")
            (ListLink ingredient (VariableNode "$partner")))
          (EvaluationLink
            (PredicateNode "incompatible")
            (ListLink (VariableNode "$partner") ingredient)))))))

;; Query to find synergistic partners
(define find-synergistic-with
  (lambda (ingredient)
    (cog-execute!
      (GetLink
        (VariableNode "$partner")
        (OrLink
          (EvaluationLink
            (PredicateNode "synergistic")
            (ListLink ingredient (VariableNode "$partner")))
          (EvaluationLink
            (PredicateNode "synergistic")
            (ListLink (VariableNode "$partner") ingredient)))))))

;; Function to check if two specific ingredients are compatible
(define check-compatibility
  (lambda (ingredient1 ingredient2)
    (let ((compatible-result
           (cog-execute!
             (GetLink
               (VariableNode "$x")
               (OrLink
                 (EvaluationLink
                   (PredicateNode "compatible")
                   (ListLink ingredient1 ingredient2))
                 (EvaluationLink
                   (PredicateNode "compatible")
                   (ListLink ingredient2 ingredient1))))))
          (incompatible-result
           (cog-execute!
             (GetLink
               (VariableNode "$x")
               (OrLink
                 (EvaluationLink
                   (PredicateNode "incompatible")
                   (ListLink ingredient1 ingredient2))
                 (EvaluationLink
                   (PredicateNode "incompatible")
                   (ListLink ingredient2 ingredient1))))))
          (synergistic-result
           (cog-execute!
             (GetLink
               (VariableNode "$x")
               (OrLink
                 (EvaluationLink
                   (PredicateNode "synergistic")
                   (ListLink ingredient1 ingredient2))
                 (EvaluationLink
                   (PredicateNode "synergistic")
                   (ListLink ingredient2 ingredient1)))))))
      (cond
        ((not (null? (cog-outgoing-set synergistic-result))) "synergistic")
        ((not (null? (cog-outgoing-set compatible-result))) "compatible")
        ((not (null? (cog-outgoing-set incompatible-result))) "incompatible")
        (else "unknown")))))

;; Function to get incompatibility reason
(define get-incompatibility-reason
  (lambda (ingredient1 ingredient2)
    (cog-execute!
      (GetLink
        (VariableNode "$reason")
        (OrLink
          (EvaluationLink
            (PredicateNode "incompatible_reason")
            (ListLink 
              (ListLink ingredient1 ingredient2)
              (VariableNode "$reason")))
          (EvaluationLink
            (PredicateNode "incompatible_reason")
            (ListLink 
              (ListLink ingredient2 ingredient1)
              (VariableNode "$reason"))))))))

;; =============================================================================
;; FORMULATION COMPATIBILITY CHECKER
;; =============================================================================

;; Function to check compatibility of an entire ingredient list
(define check-formulation-compatibility
  (lambda (ingredient-list)
    (define issues '())
    (define synergies '())
    
    ;; Check all pairs in the formulation
    (for-each
      (lambda (i)
        (for-each
          (lambda (j)
            (when (< i j)  ; Avoid checking the same pair twice
              (let* ((ing1 (list-ref ingredient-list i))
                     (ing2 (list-ref ingredient-list j))
                     (compatibility (check-compatibility ing1 ing2)))
                (cond
                  ((string=? compatibility "incompatible")
                   (let ((reason-result (get-incompatibility-reason ing1 ing2)))
                     (if (not (null? (cog-outgoing-set reason-result)))
                         (set! issues 
                               (cons (list (cog-name ing1) 
                                          (cog-name ing2)
                                          (cog-name (car (cog-outgoing-set reason-result))))
                                     issues))
                         (set! issues 
                               (cons (list (cog-name ing1) 
                                          (cog-name ing2)
                                          "unknown_reason")
                                     issues)))))
                  ((string=? compatibility "synergistic")
                   (set! synergies 
                         (cons (list (cog-name ing1) (cog-name ing2))
                               synergies)))))))
          (iota (length ingredient-list))))
      (iota (length ingredient-list)))
    
    (list issues synergies)))

;; =============================================================================
;; DEMONSTRATION FUNCTIONS
;; =============================================================================

;; Function to display ingredient compatibility information
(define display-ingredient-info
  (lambda (ingredient)
    (display (string-append "\n=== " (cog-name ingredient) " Compatibility Analysis ===\n"))
    
    ;; Compatible ingredients
    (display "Compatible with:\n")
    (let ((compatible (find-compatible-with ingredient)))
      (if (null? (cog-outgoing-set compatible))
          (display "  No specific compatible ingredients defined\n")
          (for-each (lambda (partner)
                      (display (string-append "  ✓ " (cog-name partner) "\n")))
                    (cog-outgoing-set compatible))))
    
    ;; Synergistic ingredients
    (display "Synergistic with:\n")
    (let ((synergistic (find-synergistic-with ingredient)))
      (if (null? (cog-outgoing-set synergistic))
          (display "  No synergistic partners defined\n")
          (for-each (lambda (partner)
                      (display (string-append "  ⚡ " (cog-name partner) "\n")))
                    (cog-outgoing-set synergistic))))
    
    ;; Incompatible ingredients
    (display "Incompatible with:\n")
    (let ((incompatible (find-incompatible-with ingredient)))
      (if (null? (cog-outgoing-set incompatible))
          (display "  No known incompatibilities\n")
          (for-each (lambda (partner)
                      (display (string-append "  ✗ " (cog-name partner)))
                      (let ((reason (get-incompatibility-reason ingredient partner)))
                        (if (not (null? (cog-outgoing-set reason)))
                            (display (string-append " (" (cog-name (car (cog-outgoing-set reason))) ")\n"))
                            (display "\n"))))
                    (cog-outgoing-set incompatible))))))

;; Function to test a sample formulation
(define test-sample-formulation
  (lambda (formulation-name ingredient-names)
    (display (string-append "\n=== Testing " formulation-name " ===\n"))
    (display "Ingredients:\n")
    (for-each (lambda (name)
                (display (string-append "  - " name "\n")))
              ingredient-names)
    
    (let* ((ingredient-atoms (map (lambda (name) (ConceptNode name)) ingredient-names))
           (compatibility-results (check-formulation-compatibility ingredient-atoms))
           (issues (car compatibility-results))
           (synergies (cadr compatibility-results)))
      
      (display "\nCompatibility Issues:\n")
      (if (null? issues)
          (display "  ✓ No compatibility issues found!\n")
          (for-each (lambda (issue)
                      (display (string-append "  ✗ " 
                                              (car issue) " + " (cadr issue)
                                              " (" (caddr issue) ")\n")))
                    issues))
      
      (display "\nSynergistic Combinations:\n")
      (if (null? synergies)
          (display "  No specific synergies identified\n")
          (for-each (lambda (synergy)
                      (display (string-append "  ⚡ " 
                                              (car synergy) " + " (cadr synergy) "\n")))
                    synergies))
      
      (display (string-append "\nFormulation Status: " 
                              (if (null? issues) "✓ COMPATIBLE" "⚠ ISSUES DETECTED")
                              "\n")))))

;; =============================================================================
;; DEMONSTRATION AND TESTING
;; =============================================================================

;; Display compatibility information for key ingredients
(display "COSMETIC INGREDIENT COMPATIBILITY ANALYSIS")
(display "\n" (make-string 50 #\=) "\n")

;; Analyze individual ingredients
(display-ingredient-info (ConceptNode "vitamin_c"))
(display-ingredient-info (ConceptNode "retinol"))
(display-ingredient-info (ConceptNode "niacinamide"))
(display-ingredient-info (ConceptNode "hyaluronic_acid"))

;; Test sample formulations
(test-sample-formulation "Morning Vitamin C Serum"
                        '("vitamin_c" "vitamin_e" "hyaluronic_acid" "glycerin"))

(test-sample-formulation "Evening Anti-Aging Serum"
                        '("retinol" "hyaluronic_acid" "peptides" "glycerin"))

(test-sample-formulation "Brightening Treatment"
                        '("niacinamide" "alpha_arbutin" "hyaluronic_acid"))

(test-sample-formulation "Problematic Combination (Demo)"
                        '("vitamin_c" "retinol" "salicylic_acid"))

;; Quick compatibility checks
(display "\n=== Quick Compatibility Checks ===\n")

(define test-pairs
  '(("vitamin_c" "vitamin_e")
    ("vitamin_c" "retinol")
    ("niacinamide" "hyaluronic_acid")
    ("retinol" "salicylic_acid")))

(for-each
  (lambda (pair)
    (let* ((ing1 (ConceptNode (car pair)))
           (ing2 (ConceptNode (cadr pair)))
           (result (check-compatibility ing1 ing2)))
      (display (string-append (car pair) " + " (cadr pair) ": "))
      (cond
        ((string=? result "synergistic") (display "⚡ SYNERGISTIC"))
        ((string=? result "compatible") (display "✓ COMPATIBLE"))
        ((string=? result "incompatible") (display "✗ INCOMPATIBLE"))
        (else (display "? UNKNOWN")))
      (display "\n")))
  test-pairs)

;; Summary
(display "\n=== Summary ===\n")
(display "✓ Ingredient compatibility database loaded\n")
(display "✓ Compatibility analysis functions defined\n")
(display "✓ Sample formulations tested\n")
(display "✓ Individual ingredient profiles analyzed\n")

(display "\nKey Insights:\n")
(display "• Vitamin C and Vitamin E work synergistically\n")
(display "• Vitamin C and Retinol should not be combined\n")
(display "• Hyaluronic Acid is compatible with most ingredients\n")
(display "• Multiple actives (retinol + BHA + benzoyl peroxide) can cause irritation\n")

(display "\nRecommendations:\n")
(display "• Use incompatible ingredients at different times (AM/PM)\n")
(display "• Start with lower concentrations when combining actives\n")
(display "• Always patch test new combinations\n")
(display "• Consider pH requirements when formulating\n")

(display "\n=== Compatibility Analysis Complete ===\n")