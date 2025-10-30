#!/usr/bin/opencog/scm
;;;
;;; Cosmetic Compatibility Analysis - Simple Ingredient Interaction Checking
;;;
;;; This example demonstrates simple ingredient compatibility checking using
;;; the OpenCog Cheminformatics Framework atom types for cosmetic chemistry.
;;; 
;;; This is a beginner-friendly introduction to the system focusing on
;;; basic ingredient interactions and safety considerations.
;;;

(use-modules (opencog))
(use-modules (opencog atom-types))

;; Load cosmetic chemistry atom types
(load "../../cheminformatics/types/atom_types.script")

;;;
;;; BASIC INGREDIENT DEFINITIONS
;;; Define common cosmetic ingredients
;;;

;; Popular Active Ingredients
(ACTIVE_INGREDIENT
  (ConceptNode "retinol")
  (ListLink
    (ConceptNode "anti_aging")
    (ConceptNode "wrinkle_reduction")))

(ACTIVE_INGREDIENT
  (ConceptNode "vitamin_c")
  (ListLink
    (ConceptNode "antioxidant")
    (ConceptNode "brightening")))

(ACTIVE_INGREDIENT
  (ConceptNode "niacinamide")
  (ListLink
    (ConceptNode "pore_refining")
    (ConceptNode "oil_control")))

(ACTIVE_INGREDIENT
  (ConceptNode "hyaluronic_acid")
  (ListLink
    (ConceptNode "hydration")
    (ConceptNode "plumping")))

(ACTIVE_INGREDIENT
  (ConceptNode "salicylic_acid")
  (ListLink
    (ConceptNode "exfoliation")
    (ConceptNode "acne_treatment")))

(ACTIVE_INGREDIENT
  (ConceptNode "glycolic_acid")
  (ListLink
    (ConceptNode "exfoliation")
    (ConceptNode "skin_renewal")))

(ACTIVE_INGREDIENT
  (ConceptNode "benzoyl_peroxide")
  (ListLink
    (ConceptNode "acne_treatment")
    (ConceptNode "antibacterial")))

;;;
;;; COMPATIBILITY RULES
;;; Define which ingredients work well together
;;;

;; SAFE COMBINATIONS - These ingredients are compatible
(COMPATIBILITY_LINK
  (ConceptNode "hyaluronic_acid")
  (ConceptNode "niacinamide"))

(COMPATIBILITY_LINK
  (ConceptNode "hyaluronic_acid")
  (ConceptNode "retinol"))

(COMPATIBILITY_LINK
  (ConceptNode "hyaluronic_acid")
  (ConceptNode "vitamin_c"))

(COMPATIBILITY_LINK
  (ConceptNode "hyaluronic_acid")
  (ConceptNode "salicylic_acid"))

(COMPATIBILITY_LINK
  (ConceptNode "niacinamide")
  (ConceptNode "salicylic_acid"))

(COMPATIBILITY_LINK
  (ConceptNode "niacinamide")
  (ConceptNode "retinol"))

;;;
;;; INCOMPATIBILITY RULES
;;; Define which ingredients should NOT be combined
;;;

;; AVOID COMBINATIONS - These can cause irritation or reduced efficacy
(INCOMPATIBILITY_LINK
  (ConceptNode "retinol")
  (ConceptNode "vitamin_c"))

(INCOMPATIBILITY_LINK
  (ConceptNode "retinol")
  (ConceptNode "salicylic_acid"))

(INCOMPATIBILITY_LINK
  (ConceptNode "retinol")
  (ConceptNode "glycolic_acid"))

(INCOMPATIBILITY_LINK
  (ConceptNode "retinol")
  (ConceptNode "benzoyl_peroxide"))

(INCOMPATIBILITY_LINK
  (ConceptNode "vitamin_c")
  (ConceptNode "salicylic_acid"))

(INCOMPATIBILITY_LINK
  (ConceptNode "vitamin_c")
  (ConceptNode "glycolic_acid"))

(INCOMPATIBILITY_LINK
  (ConceptNode "vitamin_c")
  (ConceptNode "niacinamide"))

(INCOMPATIBILITY_LINK
  (ConceptNode "benzoyl_peroxide")
  (ConceptNode "salicylic_acid"))

;;;
;;; SYNERGISTIC COMBINATIONS
;;; Define ingredients that work better together
;;;

;; POWER COUPLES - These combinations enhance each other's benefits
(SYNERGY_LINK
  (ConceptNode "hyaluronic_acid")
  (ConceptNode "niacinamide"))

(SYNERGY_LINK
  (ConceptNode "vitamin_c")
  (ConceptNode "vitamin_e"))

;;;
;;; SIMPLE COMPATIBILITY FUNCTIONS
;;; Easy-to-use functions for checking ingredient interactions
;;;

;; Check if two ingredients are explicitly compatible
(define (safe-to-combine? ingredient1 ingredient2)
  "Check if two ingredients are safe to combine"
  (or
    ;; Check forward compatibility
    (cog-link-exists? 'COMPATIBILITY_LINK
      (ConceptNode ingredient1)
      (ConceptNode ingredient2))
    ;; Check reverse compatibility
    (cog-link-exists? 'COMPATIBILITY_LINK
      (ConceptNode ingredient2)
      (ConceptNode ingredient1))
    ;; Check for synergy (implies safety)
    (cog-link-exists? 'SYNERGY_LINK
      (ConceptNode ingredient1)
      (ConceptNode ingredient2))
    (cog-link-exists? 'SYNERGY_LINK
      (ConceptNode ingredient2)
      (ConceptNode ingredient1))))

;; Check if two ingredients are incompatible
(define (avoid-combination? ingredient1 ingredient2)
  "Check if two ingredients should be avoided together"
  (or
    (cog-link-exists? 'INCOMPATIBILITY_LINK
      (ConceptNode ingredient1)
      (ConceptNode ingredient2))
    (cog-link-exists? 'INCOMPATIBILITY_LINK
      (ConceptNode ingredient2)
      (ConceptNode ingredient1))))

;; Check if two ingredients are synergistic
(define (synergistic-pair? ingredient1 ingredient2)
  "Check if two ingredients enhance each other"
  (or
    (cog-link-exists? 'SYNERGY_LINK
      (ConceptNode ingredient1)
      (ConceptNode ingredient2))
    (cog-link-exists? 'SYNERGY_LINK
      (ConceptNode ingredient2)
      (ConceptNode ingredient1))))

;; Get interaction status between two ingredients
(define (check-interaction ingredient1 ingredient2)
  "Get the interaction status between two ingredients"
  (cond
    ((avoid-combination? ingredient1 ingredient2) 'incompatible)
    ((synergistic-pair? ingredient1 ingredient2) 'synergistic)
    ((safe-to-combine? ingredient1 ingredient2) 'compatible)
    (else 'unknown)))

;; Print compatibility result with emoji and advice
(define (print-compatibility-result ingredient1 ingredient2)
  "Print user-friendly compatibility result"
  (let ((status (check-interaction ingredient1 ingredient2)))
    (display (string-append ingredient1 " + " ingredient2 ": "))
    (case status
      ((incompatible)
       (display "❌ AVOID - Can cause irritation or reduce effectiveness\n")
       (display "   💡 Tip: Use these ingredients at different times (AM/PM)\n"))
      ((synergistic)
       (display "✨ POWER COUPLE - Enhanced benefits when combined!\n")
       (display "   💡 Tip: Great combination for maximum results\n"))
      ((compatible)
       (display "✅ SAFE - Can be used together without issues\n")
       (display "   💡 Tip: Good basic combination\n"))
      ((unknown)
       (display "❓ UNKNOWN - No specific interaction data available\n")
       (display "   💡 Tip: Start with patch testing if unsure\n")))
    (display "\n")))

;;;
;;; SKINCARE ROUTINE ANALYSIS
;;; Analyze complete skincare routines for compatibility
;;;

;; Define sample routines
(define morning-routine '("vitamin_c" "hyaluronic_acid" "niacinamide"))
(define evening-routine '("retinol" "hyaluronic_acid"))
(define problem-routine '("retinol" "vitamin_c" "salicylic_acid"))

;; Analyze routine for compatibility issues
(define (analyze-routine routine-name ingredients)
  "Analyze a skincare routine for compatibility issues"
  (display (string-append "🔍 ANALYZING " (string-upcase routine-name) " ROUTINE\n"))
  (display "=====================================\n")
  (display "Ingredients: ")
  (for-each (lambda (ing) (display (string-append ing " "))) ingredients)
  (display "\n\n")
  
  (let ((issues 0)
        (synergies 0)
        (safe-pairs 0))
    
    ;; Check all ingredient pairs
    (let loop ((remaining ingredients))
      (when (not (null? remaining))
        (let ((current (car remaining))
              (rest (cdr remaining)))
          (for-each
            (lambda (other)
              (let ((status (check-interaction current other)))
                (case status
                  ((incompatible)
                   (set! issues (+ issues 1))
                   (display (string-append "❌ ISSUE: " current " + " other "\n")))
                  ((synergistic)
                   (set! synergies (+ synergies 1))
                   (display (string-append "✨ SYNERGY: " current " + " other "\n")))
                  ((compatible)
                   (set! safe-pairs (+ safe-pairs 1))
                   (display (string-append "✅ SAFE: " current " + " other "\n"))))))
            rest)
          (loop rest))))
    
    ;; Summary
    (display "\n📊 ROUTINE SUMMARY:\n")
    (display (string-append "   Issues: " (number->string issues) "\n"))
    (display (string-append "   Synergies: " (number->string synergies) "\n"))
    (display (string-append "   Safe pairs: " (number->string safe-pairs) "\n"))
    
    ;; Overall assessment
    (cond
      ((> issues 0)
       (display "\n🚨 RECOMMENDATION: Routine needs adjustment!\n")
       (display "   Consider separating incompatible ingredients\n"))
      ((> synergies 0)
       (display "\n🌟 EXCELLENT: Great synergistic routine!\n"))
      (else
       (display "\n👍 GOOD: Safe routine with no major issues\n")))
    
    (display "\n")))

;;;
;;; INGREDIENT SAFETY DATABASE
;;; Simple safety information for common ingredients
;;;

;; Safety assessments
(SAFETY_ASSESSMENT
  (ConceptNode "retinol")
  (ListLink
    (ConceptNode "photosensitizing")
    (ConceptNode "pregnancy_avoid")
    (ConceptNode "start_low_concentration")))

(SAFETY_ASSESSMENT
  (ConceptNode "vitamin_c")
  (ListLink
    (ConceptNode "generally_safe")
    (ConceptNode "can_cause_tingling")
    (ConceptNode "patch_test_recommended")))

(SAFETY_ASSESSMENT
  (ConceptNode "salicylic_acid")
  (ListLink
    (ConceptNode "photosensitizing")
    (ConceptNode "avoid_sensitive_skin")
    (ConceptNode "start_low_frequency")))

;; Print safety information
(define (print-safety-info ingredient)
  "Print safety information for an ingredient"
  (display (string-append "🛡️ SAFETY INFO: " (string-upcase ingredient) "\n"))
  (let ((safety-link (cog-link 'SAFETY_ASSESSMENT (ConceptNode ingredient))))
    (if safety-link
        (let ((safety-info (cog-outgoing-atom safety-link 1)))
          (for-each
            (lambda (info)
              (let ((info-text (cog-name info)))
                (display (string-append "   • " 
                                       (string-join (string-split info-text #\_) " ")
                                       "\n"))))
            (cog-outgoing-set safety-info)))
        (display "   No specific safety information available\n")))
  (display "\n"))

;;;
;;; INTERACTIVE EXAMPLES
;;; Demonstrate the compatibility checking system
;;;

(define (run-compatibility-examples)
  "Run interactive compatibility checking examples"
  (display "🧴 COSMETIC INGREDIENT COMPATIBILITY CHECKER\n")
  (display "===========================================\n\n")
  
  ;; Individual ingredient pair checks
  (display "1. INDIVIDUAL INGREDIENT COMPATIBILITY\n")
  (display "-------------------------------------\n")
  (print-compatibility-result "retinol" "vitamin_c")
  (print-compatibility-result "hyaluronic_acid" "niacinamide")
  (print-compatibility-result "vitamin_c" "salicylic_acid")
  (print-compatibility-result "retinol" "hyaluronic_acid")
  (print-compatibility-result "niacinamide" "salicylic_acid")
  
  ;; Routine analysis
  (display "2. SKINCARE ROUTINE ANALYSIS\n")
  (display "----------------------------\n")
  (analyze-routine "MORNING" morning-routine)
  (analyze-routine "EVENING" evening-routine)
  (analyze-routine "PROBLEMATIC" problem-routine)
  
  ;; Safety information
  (display "3. INGREDIENT SAFETY INFORMATION\n")
  (display "--------------------------------\n")
  (print-safety-info "retinol")
  (print-safety-info "vitamin_c")
  (print-safety-info "salicylic_acid")
  
  ;; Quick reference guide
  (display "4. QUICK REFERENCE GUIDE\n")
  (display "------------------------\n")
  (display "✅ GENERALLY SAFE COMBINATIONS:\n")
  (display "   • Hyaluronic Acid + almost anything\n")
  (display "   • Niacinamide + most ingredients\n")
  (display "   • Ceramides + most ingredients\n\n")
  
  (display "❌ AVOID THESE COMBINATIONS:\n")
  (display "   • Retinol + Vitamin C (use AM/PM)\n")
  (display "   • Retinol + AHA/BHA (too harsh)\n")
  (display "   • Vitamin C + AHA/BHA (pH conflict)\n")
  (display "   • Multiple strong actives together\n\n")
  
  (display "✨ POWER COUPLES (SYNERGISTIC):\n")
  (display "   • Vitamin C + Vitamin E\n")
  (display "   • Hyaluronic Acid + Niacinamide\n")
  (display "   • Retinol + Hyaluronic Acid\n\n")
  
  (display "💡 GENERAL TIPS:\n")
  (display "   • Start with one active ingredient\n")
  (display "   • Patch test new combinations\n")
  (display "   • Use strong actives on alternate nights\n")
  (display "   • Always use sunscreen with photosensitizing ingredients\n\n")
  
  (display "✅ Compatibility checking examples completed!\n"))

;;;
;;; HELPER FUNCTIONS
;;; Utility functions for interactive use
;;;

;; Quick compatibility check function
(define (quick-check ing1 ing2)
  "Quick compatibility check between two ingredients"
  (print-compatibility-result ing1 ing2))

;; List all ingredients in the database
(define (list-ingredients)
  "List all available ingredients"
  (display "📋 AVAILABLE INGREDIENTS:\n")
  (let ((ingredients '("retinol" "vitamin_c" "niacinamide" "hyaluronic_acid" 
                      "salicylic_acid" "glycolic_acid" "benzoyl_peroxide")))
    (for-each
      (lambda (ing)
        (display (string-append "   • " ing "\n")))
      ingredients))
  (display "\nUsage: (quick-check \"ingredient1\" \"ingredient2\")\n\n"))

;;;
;;; AUTOMATIC EXECUTION
;;; Run examples when file is loaded
;;;

(display "Loading cosmetic compatibility checker...\n\n")
(run-compatibility-examples)

(display "🎯 INTERACTIVE FUNCTIONS READY:\n")
(display "------------------------------\n")
(display "(quick-check \"ingredient1\" \"ingredient2\") - Check compatibility\n")
(display "(list-ingredients) - Show available ingredients\n")
(display "(print-safety-info \"ingredient\") - Get safety information\n")
(display "(analyze-routine \"name\" '(\"ing1\" \"ing2\" ...)) - Analyze routine\n\n")

(display "Example: (quick-check \"retinol\" \"niacinamide\")\n\n")