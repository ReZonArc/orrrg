"""
Dynamic Hypergredient Database

This module contains detailed hypergredient databases for each class
with comprehensive performance metrics and properties.
"""

from typing import Dict, List
from .core import Hypergredient, HypergredientDatabase, HypergredientMetrics


class HypergredientDB:
    """Static database of hypergredients with detailed properties"""
    
    @staticmethod
    def create_cellular_turnover_agents() -> List[Hypergredient]:
        """H.CT - Cellular Turnover Agents"""
        return [
            Hypergredient(
                name="tretinoin",
                inci_name="Tretinoin",
                hypergredient_class="H.CT",
                primary_function="Cellular turnover acceleration",
                secondary_functions=["Anti-aging", "Acne treatment"],
                potency=10.0,
                ph_range=(5.5, 6.5),
                stability="uv-sensitive",
                interactions={"benzoyl_peroxide": "incompatible"},
                cost_per_gram=15.00,
                bioavailability=85.0,
                safety_score=6.0,
                molecular_weight=300.44,
                clinical_evidence="strong"
            ),
            Hypergredient(
                name="bakuchiol",
                inci_name="Bakuchiol",
                hypergredient_class="H.CT",
                primary_function="Retinol alternative",
                secondary_functions=["Anti-aging", "Antioxidant"],
                potency=7.0,
                ph_range=(4.0, 9.0),
                stability="stable",
                interactions={},  # Compatible with all
                cost_per_gram=240.00,
                bioavailability=70.0,
                safety_score=9.0,
                molecular_weight=256.34,
                clinical_evidence="moderate"
            ),
            Hypergredient(
                name="retinol",
                inci_name="Retinol",
                hypergredient_class="H.CT",
                primary_function="Cellular turnover",
                secondary_functions=["Anti-aging", "Wrinkle reduction"],
                potency=8.0,
                ph_range=(5.5, 6.5),
                stability="o2-sensitive",
                interactions={"aha": "incompatible", "bha": "incompatible"},
                cost_per_gram=180.00,
                bioavailability=60.0,
                safety_score=7.0,
                molecular_weight=286.45,
                clinical_evidence="strong"
            ),
            Hypergredient(
                name="retinyl_palmitate",
                inci_name="Retinyl Palmitate",
                hypergredient_class="H.CT",
                primary_function="Gentle retinoid",
                secondary_functions=["Anti-aging"],
                potency=5.0,
                ph_range=(5.0, 7.0),
                stability="moderate",
                interactions={},
                cost_per_gram=150.00,
                bioavailability=40.0,
                safety_score=9.0,
                molecular_weight=524.86,
                clinical_evidence="moderate"
            ),
            Hypergredient(
                name="hydroxypinacolone_retinoate",
                inci_name="Hydroxypinacolone Retinoate",
                hypergredient_class="H.CT",
                primary_function="Advanced retinoid",
                secondary_functions=["Anti-aging", "Skin brightening"],
                potency=9.0,
                ph_range=(5.0, 8.0),
                stability="stable",
                interactions={},
                cost_per_gram=450.00,
                bioavailability=75.0,
                safety_score=8.0,
                molecular_weight=358.48,
                clinical_evidence="strong"
            ),
            Hypergredient(
                name="glycolic_acid",
                inci_name="Glycolic Acid",
                hypergredient_class="H.CT",
                primary_function="Chemical exfoliation",
                secondary_functions=["Brightening", "Texture improvement"],
                potency=6.0,
                ph_range=(3.5, 4.5),
                stability="stable",
                interactions={"retinoids": "incompatible"},
                cost_per_gram=45.00,
                bioavailability=90.0,
                safety_score=7.0,
                molecular_weight=76.05,
                clinical_evidence="strong"
            ),
            Hypergredient(
                name="lactic_acid",
                inci_name="Lactic Acid",
                hypergredient_class="H.CT",
                primary_function="Gentle exfoliation",
                secondary_functions=["Hydration", "Brightening"],
                potency=5.0,
                ph_range=(3.5, 5.0),
                stability="stable",
                interactions={},
                cost_per_gram=35.00,
                bioavailability=85.0,
                safety_score=8.0,
                molecular_weight=90.08,
                clinical_evidence="strong"
            )
        ]
    
    @staticmethod
    def create_collagen_synthesis_promoters() -> List[Hypergredient]:
        """H.CS - Collagen Synthesis Promoters"""
        return [
            Hypergredient(
                name="matrixyl_3000",
                inci_name="Palmitoyl Tripeptide-1, Palmitoyl Tetrapeptide-7",
                hypergredient_class="H.CS",
                primary_function="Collagen stimulation",
                secondary_functions=["Anti-aging", "Wrinkle reduction"],
                potency=9.0,
                ph_range=(5.0, 7.0),
                stability="stable",
                interactions={"vitamin_c": "synergy"},
                cost_per_gram=120.00,
                bioavailability=75.0,
                safety_score=9.0,
                molecular_weight=1000.0,  # Approximate for peptide complex
                clinical_evidence="strong"
            ),
            Hypergredient(
                name="argireline",
                inci_name="Acetyl Hexapeptide-8",
                hypergredient_class="H.CS",
                primary_function="Neurotransmitter modulation",
                secondary_functions=["Expression line reduction"],
                potency=7.0,
                ph_range=(5.0, 7.0),
                stability="stable",
                interactions={"peptides": "synergy"},
                cost_per_gram=150.00,
                bioavailability=60.0,
                safety_score=9.0,
                molecular_weight=888.99,
                clinical_evidence="moderate"
            ),
            Hypergredient(
                name="copper_peptides",
                inci_name="Copper Tripeptide-1",
                hypergredient_class="H.CS",
                primary_function="Collagen remodeling",
                secondary_functions=["Wound healing", "Anti-aging"],
                potency=8.0,
                ph_range=(5.5, 7.0),
                stability="moderate",
                interactions={"vitamin_c": "incompatible"},
                cost_per_gram=390.00,
                bioavailability=70.0,
                safety_score=8.0,
                molecular_weight=340.85,
                clinical_evidence="strong"
            ),
            Hypergredient(
                name="vitamin_c_laa",
                inci_name="L-Ascorbic Acid",
                hypergredient_class="H.CS",
                primary_function="Collagen cofactor",
                secondary_functions=["Antioxidant", "Brightening"],
                potency=8.0,
                ph_range=(3.0, 4.0),
                stability="unstable",
                interactions={"copper": "incompatible", "peptides": "synergy"},
                cost_per_gram=85.00,
                bioavailability=85.0,
                safety_score=7.0,
                molecular_weight=176.12,
                clinical_evidence="strong"
            ),
            Hypergredient(
                name="vitamin_c_sap",
                inci_name="Sodium Ascorbyl Phosphate",
                hypergredient_class="H.CS",
                primary_function="Stable vitamin C",
                secondary_functions=["Antioxidant", "Acne treatment"],
                potency=6.0,
                ph_range=(6.0, 7.0),
                stability="stable",
                interactions={},
                cost_per_gram=70.00,
                bioavailability=65.0,
                safety_score=9.0,
                molecular_weight=322.05,
                clinical_evidence="moderate"
            ),
            Hypergredient(
                name="centella_asiatica",
                inci_name="Centella Asiatica Extract",
                hypergredient_class="H.CS",
                primary_function="Collagen synthesis",
                secondary_functions=["Anti-inflammatory", "Healing"],
                potency=7.0,
                ph_range=(5.0, 7.0),
                stability="stable",
                interactions={},
                cost_per_gram=55.00,
                bioavailability=70.0,
                safety_score=9.0,
                molecular_weight=488.0,  # Asiaticoside
                clinical_evidence="strong"
            ),
            Hypergredient(
                name="egf",
                inci_name="sh-Oligopeptide-1",
                hypergredient_class="H.CS",
                primary_function="Growth factor",
                secondary_functions=["Cell regeneration", "Wound healing"],
                potency=9.0,
                ph_range=(6.0, 7.0),
                stability="unstable",
                interactions={"peptides": "synergy"},
                cost_per_gram=500.00,
                bioavailability=60.0,
                safety_score=8.0,
                molecular_weight=6200.0,
                clinical_evidence="moderate"
            )
        ]
    
    @staticmethod
    def create_antioxidant_systems() -> List[Hypergredient]:
        """H.AO - Antioxidant Systems"""
        return [
            Hypergredient(
                name="astaxanthin",
                inci_name="Astaxanthin",
                hypergredient_class="H.AO",
                primary_function="Super antioxidant",
                secondary_functions=["Anti-inflammatory", "UV protection"],
                potency=9.0,
                ph_range=(5.0, 7.0),
                stability="light-sensitive",
                interactions={"vitamin_e": "synergy"},
                cost_per_gram=360.00,
                bioavailability=60.0,
                safety_score=9.0,
                molecular_weight=596.84,
                clinical_evidence="strong"
            ),
            Hypergredient(
                name="resveratrol",
                inci_name="Resveratrol",
                hypergredient_class="H.AO",
                primary_function="Antioxidant",
                secondary_functions=["Anti-aging", "Anti-inflammatory"],
                potency=8.0,
                ph_range=(5.0, 7.0),
                stability="moderate",
                interactions={"ferulic_acid": "synergy"},
                cost_per_gram=190.00,
                bioavailability=70.0,
                safety_score=8.0,
                molecular_weight=228.24,
                clinical_evidence="strong"
            ),
            Hypergredient(
                name="vitamin_e",
                inci_name="Tocopherol",
                hypergredient_class="H.AO",
                primary_function="Lipid antioxidant",
                secondary_functions=["Moisturizing", "Stabilizing"],
                potency=7.0,
                ph_range=(5.0, 8.0),
                stability="stable",
                interactions={"vitamin_c": "synergy"},
                cost_per_gram=50.00,
                bioavailability=85.0,
                safety_score=9.0,
                molecular_weight=430.71,
                clinical_evidence="strong"
            ),
            Hypergredient(
                name="ferulic_acid",
                inci_name="Ferulic Acid",
                hypergredient_class="H.AO",
                primary_function="Antioxidant booster",
                secondary_functions=["UV protection", "Stabilizing"],
                potency=7.0,
                ph_range=(4.0, 6.0),
                stability="ph-dependent",
                interactions={"vitamin_c": "strong_synergy", "vitamin_e": "strong_synergy"},
                cost_per_gram=125.00,
                bioavailability=75.0,
                safety_score=9.0,
                molecular_weight=194.18,
                clinical_evidence="strong"
            ),
            Hypergredient(
                name="coq10",
                inci_name="Ubiquinone",
                hypergredient_class="H.AO",
                primary_function="Cellular energy antioxidant",
                secondary_functions=["Anti-aging", "Energy metabolism"],
                potency=8.0,
                ph_range=(5.0, 7.0),
                stability="stable",
                interactions={"vitamin_e": "synergy"},
                cost_per_gram=190.00,
                bioavailability=65.0,
                safety_score=9.0,
                molecular_weight=863.34,
                clinical_evidence="moderate"
            ),
            Hypergredient(
                name="ergothioneine",
                inci_name="Ergothioneine",
                hypergredient_class="H.AO",
                primary_function="Master antioxidant",
                secondary_functions=["DNA protection", "Anti-aging"],
                potency=9.0,
                ph_range=(5.0, 8.0),
                stability="very_stable",
                interactions={},
                cost_per_gram=370.00,
                bioavailability=80.0,
                safety_score=10.0,
                molecular_weight=229.30,
                clinical_evidence="strong"
            ),
            Hypergredient(
                name="green_tea_egcg",
                inci_name="Epigallocatechin Gallate",
                hypergredient_class="H.AO",
                primary_function="Polyphenol antioxidant",
                secondary_functions=["Anti-inflammatory", "Anti-aging"],
                potency=8.0,
                ph_range=(5.0, 7.0),
                stability="unstable",
                interactions={"vitamin_c": "synergy"},
                cost_per_gram=65.00,
                bioavailability=45.0,
                safety_score=8.0,
                molecular_weight=458.37,
                clinical_evidence="strong"
            )
        ]
    
    @staticmethod
    def create_hydration_systems() -> List[Hypergredient]:
        """H.HY - Hydration Systems"""
        return [
            Hypergredient(
                name="hyaluronic_acid_hmw",
                inci_name="Sodium Hyaluronate (High MW)",
                hypergredient_class="H.HY",
                primary_function="Surface hydration",
                secondary_functions=["Film formation", "Smoothing"],
                potency=8.0,
                ph_range=(5.0, 7.0),
                stability="stable",
                interactions={"glycerin": "synergy"},
                cost_per_gram=180.00,
                bioavailability=95.0,
                safety_score=10.0,
                molecular_weight=1000000.0,
                clinical_evidence="strong"
            ),
            Hypergredient(
                name="hyaluronic_acid_lmw",
                inci_name="Sodium Hyaluronate (Low MW)",
                hypergredient_class="H.HY",
                primary_function="Deep hydration",
                secondary_functions=["Penetration", "Plumping"],
                potency=9.0,
                ph_range=(5.0, 7.0),
                stability="stable",
                interactions={"ceramides": "synergy"},
                cost_per_gram=220.00,
                bioavailability=85.0,
                safety_score=10.0,
                molecular_weight=50000.0,
                clinical_evidence="strong"
            ),
            Hypergredient(
                name="glycerin",
                inci_name="Glycerin",
                hypergredient_class="H.HY",
                primary_function="Humectant",
                secondary_functions=["Moisturizing", "Texture enhancement"],
                potency=7.0,
                ph_range=(4.0, 8.0),
                stability="stable",
                interactions={"hyaluronic_acid": "synergy"},
                cost_per_gram=2.50,
                bioavailability=95.0,
                safety_score=10.0,
                molecular_weight=92.09,
                clinical_evidence="strong"
            ),
            Hypergredient(
                name="beta_glucan",
                inci_name="Beta-Glucan",
                hypergredient_class="H.HY",
                primary_function="Moisturizing and soothing",
                secondary_functions=["Anti-inflammatory", "Barrier support"],
                potency=7.0,
                ph_range=(5.0, 7.0),
                stability="stable",
                interactions={},
                cost_per_gram=85.00,
                bioavailability=75.0,
                safety_score=10.0,
                molecular_weight=100000.0,
                clinical_evidence="moderate"
            )
        ]


def create_hypergredient_database() -> HypergredientDatabase:
    """Create complete hypergredient database with all classes"""
    db = HypergredientDatabase()
    
    # Add all hypergredient classes
    for hypergredient in HypergredientDB.create_cellular_turnover_agents():
        db.add_hypergredient(hypergredient)
    
    for hypergredient in HypergredientDB.create_collagen_synthesis_promoters():
        db.add_hypergredient(hypergredient)
    
    for hypergredient in HypergredientDB.create_antioxidant_systems():
        db.add_hypergredient(hypergredient)
    
    for hypergredient in HypergredientDB.create_hydration_systems():
        db.add_hypergredient(hypergredient)
    
    # Add remaining classes with basic entries
    _add_barrier_repair_complex(db)
    _add_melanin_modulators(db)
    _add_anti_inflammatory_agents(db)
    _add_microbiome_balancers(db)
    _add_sebum_regulators(db)
    _add_penetration_enhancers(db)
    
    return db


def _add_barrier_repair_complex(db: HypergredientDatabase):
    """Add H.BR - Barrier Repair Complex hypergredients"""
    hypergredients = [
        Hypergredient(
            name="ceramide_np",
            inci_name="Ceramide NP",
            hypergredient_class="H.BR",
            primary_function="Barrier restoration",
            secondary_functions=["Moisturizing", "Anti-aging"],
            potency=8.0,
            cost_per_gram=180.00,
            safety_score=10.0,
            clinical_evidence="strong"
        ),
        Hypergredient(
            name="cholesterol",
            inci_name="Cholesterol",
            hypergredient_class="H.BR",
            primary_function="Lipid barrier support",
            secondary_functions=["Moisturizing"],
            potency=7.0,
            cost_per_gram=45.00,
            safety_score=10.0,
            clinical_evidence="strong"
        ),
        Hypergredient(
            name="niacinamide",
            inci_name="Niacinamide",
            hypergredient_class="H.BR",
            primary_function="Barrier strengthening",
            secondary_functions=["Sebum regulation", "Brightening"],
            potency=8.0,
            cost_per_gram=25.00,
            safety_score=10.0,
            clinical_evidence="strong"
        )
    ]
    
    for h in hypergredients:
        db.add_hypergredient(h)


def _add_melanin_modulators(db: HypergredientDatabase):
    """Add H.ML - Melanin Modulators hypergredients"""
    hypergredients = [
        Hypergredient(
            name="alpha_arbutin",
            inci_name="Alpha Arbutin",
            hypergredient_class="H.ML",
            primary_function="Tyrosinase inhibition",
            secondary_functions=["Brightening", "Spot reduction"],
            potency=8.0,
            cost_per_gram=120.00,
            safety_score=9.0,
            clinical_evidence="strong"
        ),
        Hypergredient(
            name="tranexamic_acid",
            inci_name="Tranexamic Acid",
            hypergredient_class="H.ML",
            primary_function="Melanin pathway inhibition",
            secondary_functions=["Anti-inflammatory", "Brightening"],
            potency=8.0,
            cost_per_gram=85.00,
            safety_score=9.0,
            clinical_evidence="strong"
        ),
        Hypergredient(
            name="kojic_acid",
            inci_name="Kojic Acid",
            hypergredient_class="H.ML",
            primary_function="Tyrosinase inhibition",
            secondary_functions=["Brightening"],
            potency=7.0,
            cost_per_gram=65.00,
            safety_score=6.0,
            clinical_evidence="moderate"
        )
    ]
    
    for h in hypergredients:
        db.add_hypergredient(h)


def _add_anti_inflammatory_agents(db: HypergredientDatabase):
    """Add H.AI - Anti-Inflammatory Agents hypergredients"""
    hypergredients = [
        Hypergredient(
            name="allantoin",
            inci_name="Allantoin",
            hypergredient_class="H.AI",
            primary_function="Anti-inflammatory",
            secondary_functions=["Soothing", "Healing"],
            potency=7.0,
            cost_per_gram=35.00,
            safety_score=10.0,
            clinical_evidence="strong"
        ),
        Hypergredient(
            name="bisabolol",
            inci_name="Bisabolol",
            hypergredient_class="H.AI",
            primary_function="Anti-inflammatory",
            secondary_functions=["Soothing", "Antimicrobial"],
            potency=8.0,
            cost_per_gram=85.00,
            safety_score=10.0,
            clinical_evidence="strong"
        )
    ]
    
    for h in hypergredients:
        db.add_hypergredient(h)


def _add_microbiome_balancers(db: HypergredientDatabase):
    """Add H.MB - Microbiome Balancers hypergredients"""
    hypergredients = [
        Hypergredient(
            name="lactobacillus_ferment",
            inci_name="Lactobacillus Ferment",
            hypergredient_class="H.MB",
            primary_function="Microbiome support",
            secondary_functions=["Barrier strengthening", "pH balancing"],
            potency=7.0,
            cost_per_gram=95.00,
            safety_score=9.0,
            clinical_evidence="moderate"
        )
    ]
    
    for h in hypergredients:
        db.add_hypergredient(h)


def _add_sebum_regulators(db: HypergredientDatabase):
    """Add H.SE - Sebum Regulators hypergredients"""
    hypergredients = [
        Hypergredient(
            name="salicylic_acid",
            inci_name="Salicylic Acid",
            hypergredient_class="H.SE",
            primary_function="Sebum regulation",
            secondary_functions=["Exfoliation", "Pore cleansing"],
            potency=8.0,
            cost_per_gram=25.00,
            safety_score=7.0,
            clinical_evidence="strong"
        ),
        Hypergredient(
            name="zinc_pca",
            inci_name="Zinc PCA",
            hypergredient_class="H.SE",
            primary_function="Sebum control",
            secondary_functions=["Antimicrobial", "Astringent"],  
            potency=7.0,
            cost_per_gram=45.00,
            safety_score=9.0,
            clinical_evidence="moderate"
        )
    ]
    
    for h in hypergredients:
        db.add_hypergredient(h)


def _add_penetration_enhancers(db: HypergredientDatabase):
    """Add H.PD - Penetration/Delivery Enhancers hypergredients"""
    hypergredients = [
        Hypergredient(
            name="dimethyl_isosorbide",
            inci_name="Dimethyl Isosorbide",
            hypergredient_class="H.PD",
            primary_function="Penetration enhancement",
            secondary_functions=["Solubilizer"],
            potency=8.0,
            cost_per_gram=150.00,
            safety_score=8.0,
            clinical_evidence="moderate"
        ),
        Hypergredient(
            name="propanediol",
            inci_name="Propanediol",
            hypergredient_class="H.PD",
            primary_function="Penetration enhancement",
            secondary_functions=["Humectant", "Antimicrobial"],
            potency=6.0,
            cost_per_gram=25.00,
            safety_score=9.0,
            clinical_evidence="moderate"
        )
    ]
    
    for h in hypergredients:
        db.add_hypergredient(h)