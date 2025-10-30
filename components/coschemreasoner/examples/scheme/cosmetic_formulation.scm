;; Cosmetic Formulation Modeling Example
;; 
;; This Scheme example demonstrates complex formulation modeling with 
;; compatibility analysis using the OpenCog cheminformatics framework's
;; cosmetic chemistry specializations.
;;
;; Author: OpenCog Cheminformatics Team
;; License: MIT

(use-modules (opencog)
             (opencog atom-types)
             (opencog exec))

;; ============================================================================
;; INGREDIENT DEFINITIONS
;; ============================================================================

;; Active Ingredients
(define hyaluronic-acid (ACTIVE_INGREDIENT "hyaluronic_acid"))
(define niacinamide (ACTIVE_INGREDIENT "niacinamide"))
(define vitamin-c (ACTIVE_INGREDIENT "vitamin_c"))
(define retinol (ACTIVE_INGREDIENT "retinol"))
(define alpha-arbutin (ACTIVE_INGREDIENT "alpha_arbutin"))
(define peptides (ACTIVE_INGREDIENT "peptides"))

;; Humectants
(define glycerin (HUMECTANT "glycerin"))
(define propylene-glycol (HUMECTANT "propylene_glycol"))
(define sodium-pca (HUMECTANT "sodium_pca"))
(define betaine (HUMECTANT "betaine"))

;; Emulsifiers
(define cetyl-alcohol (EMULSIFIER "cetyl_alcohol"))
(define stearic-acid (EMULSIFIER "stearic_acid"))
(define lecithin (EMULSIFIER "lecithin"))
(define polysorbate-80 (EMULSIFIER "polysorbate_80"))

;; Emollients
(define squalane (EMOLLIENT "squalane"))
(define jojoba-oil (EMOLLIENT "jojoba_oil"))
(define shea-butter (EMOLLIENT "shea_butter"))
(define dimethicone (EMOLLIENT "dimethicone"))

;; Antioxidants
(define vitamin-e (ANTIOXIDANT "vitamin_e"))
(define ferulic-acid (ANTIOXIDANT "ferulic_acid"))
(define green-tea-extract (ANTIOXIDANT "green_tea_extract"))

;; Preservatives
(define phenoxyethanol (PRESERVATIVE "phenoxyethanol"))
(define ethylhexylglycerin (PRESERVATIVE "ethylhexylglycerin"))
(define potassium-sorbate (PRESERVATIVE "potassium_sorbate"))

;; pH Adjusters
(define citric-acid (PH_ADJUSTER "citric_acid"))
(define sodium-hydroxide (PH_ADJUSTER "sodium_hydroxide"))
(define triethanolamine (PH_ADJUSTER "triethanolamine"))

;; Thickeners
(define xanthan-gum (THICKENER "xanthan_gum"))
(define carbomer (THICKENER "carbomer"))
(define hydroxyethylcellulose (THICKENER "hydroxyethylcellulose"))

;; ============================================================================
;; PROPERTY DEFINITIONS
;; ============================================================================

;; pH Properties for ingredients
(define (create-ph-property ingredient min-ph max-ph)
  (EvaluationLink
    (PredicateNode "has-ph-range")
    (ListLink ingredient (NumberNode min-ph) (NumberNode max-ph))))

;; Define pH ranges for ingredients
(create-ph-property hyaluronic-acid 5.0 7.0)
(create-ph-property niacinamide 5.0 7.0)
(create-ph-property vitamin-c 3.0 4.0)
(create-ph-property retinol 5.5 6.5)
(create-ph-property glycerin 4.0 8.0)
(create-ph-property phenoxyethanol 4.0 8.0)

;; Concentration limits
(define (create-concentration-limit ingredient max-concentration)
  (EvaluationLink
    (PredicateNode "max-concentration")
    (ListLink ingredient (NumberNode max-concentration))))

(create-concentration-limit hyaluronic-acid 2.0)
(create-concentration-limit niacinamide 10.0)
(create-concentration-limit vitamin-c 20.0)
(create-concentration-limit retinol 1.0)
(create-concentration-limit phenoxyethanol 1.0)

;; ============================================================================
;; COMPATIBILITY RELATIONSHIPS
;; ============================================================================

;; Compatible ingredient pairs
(COMPATIBILITY_LINK hyaluronic-acid niacinamide)
(COMPATIBILITY_LINK hyaluronic-acid glycerin)
(COMPATIBILITY_LINK hyaluronic-acid alpha-arbutin)
(COMPATIBILITY_LINK niacinamide alpha-arbutin)
(COMPATIBILITY_LINK niacinamide peptides)
(COMPATIBILITY_LINK glycerin propylene-glycol)
(COMPATIBILITY_LINK squalane jojoba-oil)
(COMPATIBILITY_LINK cetyl-alcohol stearic-acid)

;; Incompatible ingredient pairs
(INCOMPATIBILITY_LINK vitamin-c retinol)
(INCOMPATIBILITY_LINK vitamin-c niacinamide) ;; at certain pH levels
(INCOMPATIBILITY_LINK retinol alpha-arbutin)

;; Synergistic ingredient pairs
(SYNERGY_LINK vitamin-c vitamin-e)
(SYNERGY_LINK vitamin-c ferulic-acid)
(SYNERGY_LINK hyaluronic-acid sodium-pca)
(SYNERGY_LINK peptides niacinamide)

;; Antagonistic ingredient pairs
(ANTAGONISM_LINK retinol vitamin-c)

;; ============================================================================
;; FORMULATION CREATION FUNCTIONS
;; ============================================================================

;; Function to create a formulation with ingredients and concentrations
(define (create-formulation formulation-type ingredients-with-concentrations)
  (let ((formulation (cog-new-node formulation-type (gensym "formulation"))))
    (for-each
      (lambda (ingredient-conc-pair)
        (let ((ingredient (car ingredient-conc-pair))
              (concentration (cadr ingredient-conc-pair)))
          (EvaluationLink
            (PredicateNode "contains-ingredient")
            (ListLink formulation ingredient (NumberNode concentration)))))
      ingredients-with-concentrations)
    formulation))

;; Function to check if two ingredients are compatible
(define (ingredients-compatible? ing1 ing2)
  (not (null? (cog-link 'COMPATIBILITY_LINK ing1 ing2))))

;; Function to check if two ingredients are incompatible
(define (ingredients-incompatible? ing1 ing2)
  (not (null? (cog-link 'INCOMPATIBILITY_LINK ing1 ing2))))

;; Function to get all ingredients in a formulation
(define (get-formulation-ingredients formulation)
  (let ((contains-links (cog-get-pred formulation (PredicateNode "contains-ingredient"))))
    (map (lambda (link)
           (let ((list-link (gar (gdr link))))
             (car (cog-outgoing-set list-link))))
         contains-links)))

;; Function to validate formulation compatibility
(define (validate-formulation-compatibility formulation)
  (let ((ingredients (get-formulation-ingredients formulation))
        (incompatible-pairs '())
        (compatible-pairs '())
        (synergistic-pairs '()))
    
    ;; Check all ingredient pairs
    (for-each
      (lambda (ing1)
        (for-each
          (lambda (ing2)
            (cond
              ((ingredients-incompatible? ing1 ing2)
               (set! incompatible-pairs (cons (list ing1 ing2) incompatible-pairs)))
              ((ingredients-compatible? ing1 ing2)
               (set! compatible-pairs (cons (list ing1 ing2) compatible-pairs)))
              ((not (null? (cog-link 'SYNERGY_LINK ing1 ing2)))
               (set! synergistic-pairs (cons (list ing1 ing2) synergistic-pairs)))))
          ingredients))
      ingredients)
    
    (list 
      (cons 'incompatible incompatible-pairs)
      (cons 'compatible compatible-pairs)
      (cons 'synergistic synergistic-pairs))))

;; ============================================================================
;; EXAMPLE FORMULATIONS
;; ============================================================================

;; Anti-Aging Serum Formulation
(define anti-aging-serum
  (create-formulation 'SKINCARE_FORMULATION
    (list
      (list niacinamide 5.0)           ;; 5% Niacinamide
      (list hyaluronic-acid 1.0)       ;; 1% Hyaluronic Acid
      (list alpha-arbutin 2.0)         ;; 2% Alpha Arbutin
      (list peptides 3.0)              ;; 3% Peptides
      (list glycerin 8.0)              ;; 8% Glycerin
      (list propylene-glycol 2.0)      ;; 2% Propylene Glycol
      (list phenoxyethanol 0.8)        ;; 0.8% Phenoxyethanol
      (list ethylhexylglycerin 0.2))))  ;; 0.2% Ethylhexylglycerin

;; Hydrating Moisturizer Formulation
(define hydrating-moisturizer
  (create-formulation 'SKINCARE_FORMULATION
    (list
      (list hyaluronic-acid 1.5)       ;; 1.5% Hyaluronic Acid
      (list niacinamide 3.0)           ;; 3% Niacinamide
      (list glycerin 10.0)             ;; 10% Glycerin
      (list sodium-pca 1.0)            ;; 1% Sodium PCA
      (list cetyl-alcohol 3.0)         ;; 3% Cetyl Alcohol
      (list stearic-acid 2.0)          ;; 2% Stearic Acid
      (list squalane 4.0)              ;; 4% Squalane
      (list jojoba-oil 2.0)            ;; 2% Jojoba Oil
      (list vitamin-e 0.5)             ;; 0.5% Vitamin E
      (list phenoxyethanol 0.9))))     ;; 0.9% Phenoxyethanol

;; Vitamin C Brightening Serum (potentially problematic)
(define vitamin-c-serum
  (create-formulation 'SKINCARE_FORMULATION
    (list
      (list vitamin-c 15.0)            ;; 15% Vitamin C
      (list ferulic-acid 0.5)          ;; 0.5% Ferulic Acid
      (list vitamin-e 1.0)             ;; 1% Vitamin E
      (list glycerin 5.0)              ;; 5% Glycerin
      (list citric-acid 0.1)           ;; 0.1% Citric Acid (pH adjuster)
      (list phenoxyethanol 0.8))))     ;; 0.8% Phenoxyethanol

;; ============================================================================
;; FORMULATION ANALYSIS FUNCTIONS
;; ============================================================================

;; Function to calculate formulation pH (simplified)
(define (calculate-formulation-ph formulation)
  (let ((ingredients (get-formulation-ingredients formulation))
        (total-ph 0)
        (count 0))
    (for-each
      (lambda (ingredient)
        (let ((ph-range (cog-get-pred ingredient (PredicateNode "has-ph-range"))))
          (when (not (null? ph-range))
            (let* ((ph-values (cog-outgoing-set (gdr (car ph-range))))
                   (min-ph (cog-number (cadr ph-values)))
                   (max-ph (cog-number (caddr ph-values)))
                   (avg-ph (/ (+ min-ph max-ph) 2)))
              (set! total-ph (+ total-ph avg-ph))
              (set! count (+ count 1))))))
      ingredients)
    (if (> count 0)
        (/ total-ph count)
        7.0))) ;; Default neutral pH

;; Function to check concentration limits
(define (check-concentration-limits formulation)
  (let ((violations '()))
    (let ((contains-links (cog-get-pred formulation (PredicateNode "contains-ingredient"))))
      (for-each
        (lambda (link)
          (let* ((list-link (gar (gdr link)))
                 (ingredient (car (cog-outgoing-set list-link)))
                 (concentration (cog-number (cadr (cog-outgoing-set list-link))))
                 (limit-link (cog-get-pred ingredient (PredicateNode "max-concentration"))))
            (when (not (null? limit-link))
              (let ((max-conc (cog-number (cadr (cog-outgoing-set (gar (gdr (car limit-link))))))))
                (when (> concentration max-conc)
                  (set! violations (cons (list ingredient concentration max-conc) violations)))))))
        contains-links))
    violations))

;; Function to analyze formulation safety
(define (analyze-formulation-safety formulation)
  (let ((compatibility-analysis (validate-formulation-compatibility formulation))
        (ph-value (calculate-formulation-ph formulation))
        (concentration-violations (check-concentration-limits formulation)))
    
    (display "=== Formulation Safety Analysis ===\n")
    (display (string-append "Formulation: " (cog-name formulation) "\n"))
    (display (string-append "Calculated pH: " (number->string ph-value) "\n\n"))
    
    ;; Display concentration violations
    (if (null? concentration-violations)
        (display "✓ All ingredients within concentration limits\n")
        (begin
          (display "⚠ Concentration Limit Violations:\n")
          (for-each
            (lambda (violation)
              (let ((ingredient (car violation))
                    (actual (cadr violation))
                    (limit (caddr violation)))
                (display (string-append "  - " (cog-name ingredient) 
                                      ": " (number->string actual) 
                                      "% (limit: " (number->string limit) "%)\n"))))
            concentration-violations)))
    
    ;; Display compatibility analysis
    (let ((incompatible (cdr (assoc 'incompatible compatibility-analysis)))
          (compatible (cdr (assoc 'compatible compatibility-analysis)))
          (synergistic (cdr (assoc 'synergistic compatibility-analysis))))
      
      (if (null? incompatible)
          (display "\n✓ No incompatible ingredient pairs found\n")
          (begin
            (display "\n⚠ Incompatible ingredient pairs:\n")
            (for-each
              (lambda (pair)
                (display (string-append "  - " (cog-name (car pair)) 
                                      " + " (cog-name (cadr pair)) "\n")))
              incompatible)))
      
      (unless (null? synergistic)
        (display "\n✓ Synergistic ingredient pairs:\n")
        (for-each
          (lambda (pair)
            (display (string-append "  + " (cog-name (car pair)) 
                                  " + " (cog-name (cadr pair)) "\n")))
          synergistic)))
    
    (display "\n")))

;; ============================================================================
;; FORMULATION OPTIMIZATION FUNCTIONS
;; ============================================================================

;; Function to suggest formulation improvements
(define (suggest-formulation-improvements formulation)
  (let ((compatibility-analysis (validate-formulation-compatibility formulation))
        (ph-value (calculate-formulation-ph formulation)))
    
    (display "=== Formulation Improvement Suggestions ===\n")
    
    ;; pH optimization suggestions
    (cond
      ((< ph-value 4.5)
       (display "• Consider adding pH adjusters to increase pH to skin-compatible range (5.5-6.5)\n"))
      ((> ph-value 7.5)
       (display "• Consider adding acidic pH adjusters to lower pH to skin-compatible range\n"))
      (else
       (display "• pH is within acceptable range for topical application\n")))
    
    ;; Compatibility improvement suggestions
    (let ((incompatible (cdr (assoc 'incompatible compatibility-analysis))))
      (unless (null? incompatible)
        (display "• Consider the following to resolve incompatibilities:\n")
        (for-each
          (lambda (pair)
            (let ((ing1 (cog-name (car pair)))
                  (ing2 (cog-name (cadr pair))))
              (cond
                ((and (string=? ing1 "vitamin_c") (string=? ing2 "retinol"))
                 (display "  - Use Vitamin C in AM routine and Retinol in PM routine\n"))
                ((and (string=? ing1 "retinol") (string=? ing2 "alpha_arbutin"))
                 (display "  - Consider alternating days for retinol and alpha arbutin\n"))
                (else
                 (display (string-append "  - Separate " ing1 " and " ing2 " into different products\n"))))))
          incompatible)))
    
    (display "\n")))

;; ============================================================================
;; DEMO EXECUTION
;; ============================================================================

;; Run analysis on example formulations
(display "COSMETIC FORMULATION MODELING DEMONSTRATION\n")
(display "============================================\n\n")

;; Analyze anti-aging serum
(analyze-formulation-safety anti-aging-serum)
(suggest-formulation-improvements anti-aging-serum)

;; Analyze hydrating moisturizer
(analyze-formulation-safety hydrating-moisturizer)
(suggest-formulation-improvements hydrating-moisturizer)

;; Analyze vitamin C serum
(analyze-formulation-safety vitamin-c-serum)
(suggest-formulation-improvements vitamin-c-serum)

;; ============================================================================
;; ADVANCED FEATURES DEMONSTRATION
;; ============================================================================

;; Function to find compatible ingredient substitutes
(define (find-compatible-substitutes ingredient existing-ingredients)
  (let ((all-ingredients (list hyaluronic-acid niacinamide vitamin-c retinol
                              alpha-arbutin peptides glycerin propylene-glycol
                              sodium-pca betaine squalane jojoba-oil))
        (compatible-substitutes '()))
    
    ;; Find ingredients with same functional type that are compatible
    (for-each
      (lambda (candidate)
        (when (and (not (equal? candidate ingredient))
                   (string=? (cog-type candidate) (cog-type ingredient))
                   (not (member candidate existing-ingredients)))
          ;; Check if compatible with all existing ingredients
          (let ((all-compatible #t))
            (for-each
              (lambda (existing)
                (when (ingredients-incompatible? candidate existing)
                  (set! all-compatible #f)))
              existing-ingredients)
            (when all-compatible
              (set! compatible-substitutes (cons candidate compatible-substitutes))))))
      all-ingredients)
    
    compatible-substitutes))

;; Demonstrate substitute finding
(display "=== Compatible Ingredient Substitution ===\n")
(let ((moisturizer-ingredients (get-formulation-ingredients hydrating-moisturizer)))
  (display "Finding substitutes for hyaluronic acid in moisturizer:\n")
  (let ((substitutes (find-compatible-substitutes hyaluronic-acid moisturizer-ingredients)))
    (if (null? substitutes)
        (display "No compatible substitutes found\n")
        (for-each
          (lambda (substitute)
            (display (string-append "• " (cog-name substitute) "\n")))
          substitutes))))

(display "\nFormulation modeling demonstration complete!\n")