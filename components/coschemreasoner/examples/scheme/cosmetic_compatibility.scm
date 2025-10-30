;; Simple Cosmetic Ingredient Compatibility Checker
;;
;; This Scheme example demonstrates simple ingredient interaction checking
;; using the OpenCog cheminformatics framework's cosmetic chemistry atom types.
;; It provides basic compatibility analysis for common cosmetic ingredients.
;;
;; Author: OpenCog Cheminformatics Team
;; License: MIT

(use-modules (opencog)
             (opencog atom-types)
             (opencog exec))

;; ============================================================================
;; BASIC INGREDIENT DEFINITIONS
;; ============================================================================

;; Common Active Ingredients
(define hyaluronic-acid (ACTIVE_INGREDIENT "hyaluronic_acid"))
(define niacinamide (ACTIVE_INGREDIENT "niacinamide"))
(define vitamin-c (ACTIVE_INGREDIENT "vitamin_c"))
(define retinol (ACTIVE_INGREDIENT "retinol"))
(define salicylic-acid (ACTIVE_INGREDIENT "salicylic_acid"))
(define glycolic-acid (ACTIVE_INGREDIENT "glycolic_acid"))
(define lactic-acid (ACTIVE_INGREDIENT "lactic_acid"))
(define alpha-arbutin (ACTIVE_INGREDIENT "alpha_arbutin"))
(define kojic-acid (ACTIVE_INGREDIENT "kojic_acid"))

;; Supporting Ingredients
(define glycerin (HUMECTANT "glycerin"))
(define aloe-vera (HUMECTANT "aloe_vera"))
(define vitamin-e (ANTIOXIDANT "vitamin_e"))
(define green-tea (ANTIOXIDANT "green_tea_extract"))
(define phenoxyethanol (PRESERVATIVE "phenoxyethanol"))

;; ============================================================================
;; COMPATIBILITY KNOWLEDGE BASE
;; ============================================================================

;; Well-established compatible pairs
(COMPATIBILITY_LINK hyaluronic-acid niacinamide)
(COMPATIBILITY_LINK hyaluronic-acid glycerin)
(COMPATIBILITY_LINK hyaluronic-acid aloe-vera)
(COMPATIBILITY_LINK hyaluronic-acid alpha-arbutin)
(COMPATIBILITY_LINK niacinamide alpha-arbutin)
(COMPATIBILITY_LINK glycerin aloe-vera)
(COMPATIBILITY_LINK vitamin-e vitamin-c)

;; Known incompatible pairs
(INCOMPATIBILITY_LINK vitamin-c retinol)
(INCOMPATIBILITY_LINK vitamin-c niacinamide) ;; at low pH
(INCOMPATIBILITY_LINK retinol salicylic-acid)
(INCOMPATIBILITY_LINK retinol glycolic-acid)
(INCOMPATIBILITY_LINK retinol lactic-acid)
(INCOMPATIBILITY_LINK salicylic-acid glycolic-acid) ;; over-exfoliation risk

;; Synergistic pairs that work better together
(SYNERGY_LINK vitamin-c vitamin-e)
(SYNERGY_LINK hyaluronic-acid glycerin)
(SYNERGY_LINK niacinamide alpha-arbutin)

;; ============================================================================
;; COMPATIBILITY CHECKING FUNCTIONS
;; ============================================================================

;; Function to check if two ingredients are explicitly compatible
(define (check-compatibility ingredient1 ingredient2)
  (let ((comp-link (cog-link 'COMPATIBILITY_LINK ingredient1 ingredient2)))
    (not (null? comp-link))))

;; Function to check if two ingredients are explicitly incompatible
(define (check-incompatibility ingredient1 ingredient2)
  (let ((incomp-link (cog-link 'INCOMPATIBILITY_LINK ingredient1 ingredient2)))
    (not (null? incomp-link))))

;; Function to check if two ingredients have synergy
(define (check-synergy ingredient1 ingredient2)
  (let ((synergy-link (cog-link 'SYNERGY_LINK ingredient1 ingredient2)))
    (not (null? synergy-link))))

;; Function to get compatibility status between two ingredients
(define (get-compatibility-status ingredient1 ingredient2)
  (cond
    ((check-incompatibility ingredient1 ingredient2) 'incompatible)
    ((check-synergy ingredient1 ingredient2) 'synergistic)
    ((check-compatibility ingredient1 ingredient2) 'compatible)
    (else 'unknown)))

;; Function to display compatibility result
(define (display-compatibility ingredient1 ingredient2)
  (let ((status (get-compatibility-status ingredient1 ingredient2))
        (name1 (cog-name ingredient1))
        (name2 (cog-name ingredient2)))
    (display (string-append "\n" name1 " + " name2 ": "))
    (case status
      ((incompatible) 
       (display "❌ INCOMPATIBLE - Avoid combining"))
      ((synergistic) 
       (display "✨ SYNERGISTIC - Work better together"))
      ((compatible) 
       (display "✅ COMPATIBLE - Safe to combine"))
      ((unknown) 
       (display "❓ UNKNOWN - No specific data, use caution")))))

;; ============================================================================
;; BATCH COMPATIBILITY ANALYSIS
;; ============================================================================

;; Function to analyze compatibility of multiple ingredients
(define (analyze-ingredient-list ingredients)
  (display "\n=== INGREDIENT COMPATIBILITY ANALYSIS ===\n")
  (display "Ingredients to analyze:\n")
  (for-each
    (lambda (ingredient)
      (display (string-append "• " (cog-name ingredient) "\n")))
    ingredients)
  
  (display "\nPairwise Compatibility Results:\n")
  (display "================================\n")
  
  (let ((incompatible-pairs '())
        (synergistic-pairs '())
        (compatible-pairs '())
        (unknown-pairs '()))
    
    ;; Check all pairs
    (for-each
      (lambda (ing1)
        (for-each
          (lambda (ing2)
            (when (not (equal? ing1 ing2))
              (let ((status (get-compatibility-status ing1 ing2)))
                (case status
                  ((incompatible) 
                   (set! incompatible-pairs (cons (list ing1 ing2) incompatible-pairs)))
                  ((synergistic) 
                   (set! synergistic-pairs (cons (list ing1 ing2) synergistic-pairs)))
                  ((compatible) 
                   (set! compatible-pairs (cons (list ing1 ing2) compatible-pairs)))
                  ((unknown) 
                   (set! unknown-pairs (cons (list ing1 ing2) unknown-pairs)))))))
          ingredients))
      ingredients)
    
    ;; Display results by category
    (unless (null? incompatible-pairs)
      (display "\n❌ INCOMPATIBLE PAIRS (AVOID):\n")
      (for-each
        (lambda (pair)
          (display (string-append "   " (cog-name (car pair)) 
                                " + " (cog-name (cadr pair)) "\n")))
        incompatible-pairs))
    
    (unless (null? synergistic-pairs)
      (display "\n✨ SYNERGISTIC PAIRS (ENHANCED BENEFITS):\n")
      (for-each
        (lambda (pair)
          (display (string-append "   " (cog-name (car pair)) 
                                " + " (cog-name (cadr pair)) "\n")))
        synergistic-pairs))
    
    (unless (null? compatible-pairs)
      (display "\n✅ COMPATIBLE PAIRS (SAFE TO COMBINE):\n")
      (for-each
        (lambda (pair)
          (display (string-append "   " (cog-name (car pair)) 
                                " + " (cog-name (cadr pair)) "\n")))
        compatible-pairs))
    
    ;; Provide overall safety assessment
    (display "\n=== OVERALL ASSESSMENT ===\n")
    (if (null? incompatible-pairs)
        (display "✅ No known incompatibilities detected - formulation appears safe\n")
        (display "⚠️  CAUTION: Incompatible ingredients detected - review formulation\n"))
    
    ;; Return summary
    (list 
      (cons 'incompatible (length incompatible-pairs))
      (cons 'synergistic (length synergistic-pairs))
      (cons 'compatible (length compatible-pairs))
      (cons 'unknown (length unknown-pairs)))))

;; ============================================================================
;; INGREDIENT RECOMMENDATION FUNCTIONS
;; ============================================================================

;; Function to suggest compatible ingredients for a base ingredient
(define (suggest-compatible-ingredients base-ingredient)
  (let ((all-ingredients (list hyaluronic-acid niacinamide vitamin-c retinol
                              salicylic-acid glycolic-acid lactic-acid
                              alpha-arbutin kojic-acid glycerin aloe-vera
                              vitamin-e green-tea phenoxyethanol))
        (compatible-list '())
        (synergistic-list '()))
    
    (for-each
      (lambda (ingredient)
        (when (not (equal? ingredient base-ingredient))
          (let ((status (get-compatibility-status base-ingredient ingredient)))
            (case status
              ((compatible) 
               (set! compatible-list (cons ingredient compatible-list)))
              ((synergistic) 
               (set! synergistic-list (cons ingredient synergistic-list)))))))
      all-ingredients)
    
    (display (string-append "\n=== RECOMMENDATIONS FOR " 
                          (string-upcase (cog-name base-ingredient)) " ===\n"))
    
    (unless (null? synergistic-list)
      (display "\n✨ HIGHLY RECOMMENDED (Synergistic):\n")
      (for-each
        (lambda (ingredient)
          (display (string-append "   • " (cog-name ingredient) "\n")))
        synergistic-list))
    
    (unless (null? compatible-list)
      (display "\n✅ SAFE TO COMBINE:\n")
      (for-each
        (lambda (ingredient)
          (display (string-append "   • " (cog-name ingredient) "\n")))
        compatible-list))))

;; ============================================================================
;; COMMON SKINCARE ROUTINE ANALYSIS
;; ============================================================================

;; Function to analyze common skincare routines
(define (analyze-skincare-routine routine-name ingredients)
  (display (string-append "\n=== ANALYZING " routine-name " ===\n"))
  (analyze-ingredient-list ingredients))

;; ============================================================================
;; DEMONSTRATION EXAMPLES
;; ============================================================================

(display "COSMETIC INGREDIENT COMPATIBILITY CHECKER\n")
(display "=========================================\n")

;; Example 1: Simple pairwise compatibility checks
(display "\n1. PAIRWISE COMPATIBILITY EXAMPLES:")
(display-compatibility hyaluronic-acid niacinamide)
(display-compatibility vitamin-c retinol)
(display-compatibility vitamin-c vitamin-e)
(display-compatibility retinol salicylic-acid)
(display-compatibility hyaluronic-acid glycerin)

;; Example 2: Analyze a hydrating serum formula
(display "\n\n2. HYDRATING SERUM ANALYSIS:")
(let ((hydrating-serum (list hyaluronic-acid niacinamide glycerin aloe-vera phenoxyethanol)))
  (analyze-ingredient-list hydrating-serum))

;; Example 3: Analyze a potentially problematic anti-aging routine
(display "\n\n3. ANTI-AGING ROUTINE ANALYSIS:")
(let ((anti-aging-routine (list vitamin-c retinol niacinamide hyaluronic-acid)))
  (analyze-ingredient-list anti-aging-routine))

;; Example 4: Analyze an exfoliating treatment
(display "\n\n4. EXFOLIATING TREATMENT ANALYSIS:")
(let ((exfoliating-treatment (list salicylic-acid glycolic-acid niacinamide hyaluronic-acid)))
  (analyze-ingredient-list exfoliating-treatment))

;; Example 5: Get recommendations for specific ingredients
(suggest-compatible-ingredients vitamin-c)
(suggest-compatible-ingredients retinol)
(suggest-compatible-ingredients hyaluronic-acid)

;; ============================================================================
;; QUICK REFERENCE GUIDE
;; ============================================================================

(display "\n\n=== QUICK REFERENCE GUIDE ===\n")
(display "\nSAFE COMBINATIONS:\n")
(display "• Hyaluronic Acid + Niacinamide + Glycerin (hydrating powerhouse)\n")
(display "• Vitamin C + Vitamin E (antioxidant protection)\n")
(display "• Niacinamide + Alpha Arbutin (brightening combination)\n")

(display "\nAVOID COMBINATIONS:\n")
(display "• Vitamin C + Retinol (use at different times)\n")
(display "• Retinol + AHA/BHA acids (over-exfoliation risk)\n")
(display "• Multiple strong acids together (irritation risk)\n")

(display "\nUSAGE TIPS:\n")
(display "• Start with lower concentrations when combining actives\n")
(display "• Use incompatible ingredients at different times (AM vs PM)\n")
(display "• Always patch test new combinations\n")
(display "• Wait 15-30 minutes between applying different actives\n")

(display "\nCompatibility analysis complete!\n")