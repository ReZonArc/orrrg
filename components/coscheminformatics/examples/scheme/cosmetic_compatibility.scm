;
; cosmetic_compatibility.scm
;
; Simple example demonstrating cosmetic ingredient compatibility
; checking using the OpenCog cheminformatics framework.
; This shows basic usage of the cosmetic chemistry atom types
; for ingredient interaction analysis.
;
; To run this example: `(load "cosmetic_compatibility.scm")`

(use-modules (opencog) (opencog cheminformatics))
(use-modules (opencog exec))

; Define some common cosmetic ingredients
(define vitamin-c (Active_ingredient "vitamin_c"))
(define retinol (Active_ingredient "retinol"))
(define niacinamide (Active_ingredient "niacinamide"))
(define hyaluronic-acid (Active_ingredient "hyaluronic_acid"))

; Define compatibility relationships
(define vitamin-c-retinol-conflict
    (Incompatibility_link vitamin-c retinol))

(define niacinamide-retinol-compatible  
    (Compatibility_link niacinamide retinol))

(define ha-niacinamide-synergy
    (Synergy_link hyaluronic-acid niacinamide))

; Create a simple compatibility checker
(define compatibility-checker
    (BindLink
        (VariableList
            (TypedVariable (Variable "$ingredient1") (Type 'Active_ingredient))
            (TypedVariable (Variable "$ingredient2") (Type 'Active_ingredient)))
        
        ; Pattern: Find any compatibility relationship
        (OrLink
            (Compatibility_link (Variable "$ingredient1") (Variable "$ingredient2"))
            (Incompatibility_link (Variable "$ingredient1") (Variable "$ingredient2"))
            (Synergy_link (Variable "$ingredient1") (Variable "$ingredient2")))
        
        ; Output the relationship
        (ListLink (Variable "$ingredient1") (Variable "$ingredient2"))))

; Execute the compatibility check
(define found-relationships (cog-execute! compatibility-checker))

(format #t "=== Cosmetic Ingredient Compatibility Analysis ===~n")
(format #t "Defined ingredients:~n")
(format #t "  - Vitamin C: ~A~n" vitamin-c)
(format #t "  - Retinol: ~A~n" retinol)
(format #t "  - Niacinamide: ~A~n" niacinamide)
(format #t "  - Hyaluronic Acid: ~A~n" hyaluronic-acid)
(format #t "~n")

(format #t "Ingredient relationships:~n")
(format #t "  - Vitamin C + Retinol: INCOMPATIBLE (pH conflict)~n")
(format #t "  - Niacinamide + Retinol: COMPATIBLE (reduces irritation)~n")
(format #t "  - Hyaluronic Acid + Niacinamide: SYNERGISTIC (enhanced hydration)~n")
(format #t "~n")

(format #t "Found relationships: ~A~n" found-relationships)

; Create a safe formulation based on compatibility
(define safe-anti-aging-serum
    (Skincare_formulation
        retinol              ; Anti-aging active
        niacinamide         ; Compatible skin conditioner  
        hyaluronic-acid     ; Synergistic hydrator
        (Preservative "phenoxyethanol"))) ; Preservation

(format #t "~nSafe anti-aging formulation: ~A~n" safe-anti-aging-serum)
(format #t "This formulation avoids the Vitamin C + Retinol conflict.~n")

; Example of a problematic formulation (for educational purposes)
(define problematic-serum
    (Skincare_formulation
        vitamin-c           ; Antioxidant active
        retinol            ; Anti-aging active - CONFLICT!
        (Preservative "phenoxyethanol")))

(format #t "~nProblematic formulation: ~A~n" problematic-serum)
(format #t "⚠ This formulation contains incompatible actives!~n")

(format #t "~n=== Cosmetic Compatibility Check Complete ===~n")