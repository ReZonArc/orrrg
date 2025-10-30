#!/usr/bin/env python3
"""
ESM-2 Structure Analysis Module

Implements structure prediction analysis capabilities based on the ESM-2 paper:
- Attention pattern analysis for contact map prediction
- Perplexity-based structure quality metrics
- Scaling analysis tools
- Structure emergence visualization
"""

import json
import random
import math
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class ContactPrediction:
    """Represents a predicted contact between amino acid residues"""
    i: int  # First residue position
    j: int  # Second residue position  
    probability: float  # Contact probability
    distance_threshold: float = 8.0  # Angstrom threshold for contact


@dataclass
class StructurePredictionMetrics:
    """Metrics for evaluating structure prediction quality"""
    tm_score: float  # Template Modeling score
    gdt_score: float  # Global Distance Test score
    contact_precision_l: float  # Long-range contact precision at L contacts
    contact_precision_l2: float  # Long-range contact precision at L/2 contacts
    contact_precision_l5: float  # Long-range contact precision at L/5 contacts
    perplexity: float  # Language model perplexity
    plddt_score: float  # Predicted LDDT confidence score


class ESM2StructureAnalyzer:
    """
    Structure analysis engine for ESM-2 hypergraph models
    
    Implements the key structure prediction capabilities described in the ESM-2 paper:
    - Attention-based contact map extraction
    - Perplexity correlation analysis  
    - Scaling behavior analysis
    - Structure emergence detection
    """
    
    def __init__(self, hypergraph):
        """Initialize with ESM2 hypergraph"""
        self.hypergraph = hypergraph
        self.config = hypergraph.config
        self.num_heads = self.config["num_heads"]
        self.num_layers = self.config["num_layers"]
        
    def extract_attention_contacts(self, sequence_length: int, 
                                 attention_matrices: Optional[List] = None) -> List[ContactPrediction]:
        """
        Extract contact predictions from attention patterns
        
        Based on ESM-2 paper findings that attention patterns correspond to 
        residue-residue contact maps in protein structures.
        
        Args:
            sequence_length: Length of protein sequence
            attention_matrices: Optional pre-computed attention matrices
            
        Returns:
            List of predicted contacts sorted by probability
        """
        if attention_matrices is None:
            # Simulate attention patterns for demonstration
            attention_matrices = self._simulate_attention_patterns(sequence_length)
            
        contacts = []
        
        for layer_idx, layer_attention in enumerate(attention_matrices):
            # Extract contacts from each attention head
            for head_idx in range(self.num_heads):
                head_attention = layer_attention[head_idx]
                
                # Find high attention pairs (potential contacts)
                for i in range(sequence_length):
                    for j in range(i + 6, sequence_length):  # Long-range contacts only
                        attention_score = head_attention[i][j]
                        
                        # Convert attention to contact probability
                        contact_prob = self._attention_to_contact_probability(
                            attention_score, layer_idx, head_idx
                        )
                        
                        if contact_prob > 0.1:  # Threshold for significant contacts
                            contacts.append(ContactPrediction(i, j, contact_prob))
        
        # Sort by probability and remove duplicates
        contacts = sorted(contacts, key=lambda x: x.probability, reverse=True)
        return self._deduplicate_contacts(contacts)
    
    def calculate_contact_precision(self, predicted_contacts: List[ContactPrediction],
                                  true_contacts: List[Tuple[int, int]], 
                                  sequence_length: int) -> Dict[str, float]:
        """
        Calculate contact prediction precision metrics
        
        Following ESM-2 paper methodology for evaluating contact prediction accuracy.
        
        Args:
            predicted_contacts: List of predicted contacts
            true_contacts: List of true contact pairs
            sequence_length: Length of protein sequence
            
        Returns:
            Dictionary of precision metrics at different top-K levels
        """
        L = sequence_length
        true_contact_set = set(true_contacts)
        
        metrics = {}
        
        # Precision at top L, L/2, L/5 predictions (standard evaluation)
        for k, suffix in [(L, "_L"), (L//2, "_L2"), (L//5, "_L5")]:
            if k > len(predicted_contacts):
                k = len(predicted_contacts)
                
            top_k_predicted = predicted_contacts[:k]
            correct_predictions = sum(1 for contact in top_k_predicted 
                                    if (contact.i, contact.j) in true_contact_set or 
                                       (contact.j, contact.i) in true_contact_set)
            
            precision = correct_predictions / k if k > 0 else 0.0
            metrics[f"contact_precision{suffix}"] = precision
            
        return metrics
    
    def analyze_perplexity_correlation(self, sequences: List[str], 
                                     structure_metrics: List[StructurePredictionMetrics]) -> Dict[str, float]:
        """
        Analyze correlation between language model perplexity and structure prediction accuracy
        
        Based on ESM-2 paper finding that perplexity strongly correlates with structure quality.
        
        Args:
            sequences: List of protein sequences
            structure_metrics: List of structure prediction metrics
            
        Returns:
            Correlation coefficients between perplexity and various structure metrics
        """
        if len(sequences) != len(structure_metrics):
            raise ValueError("Sequences and metrics lists must have same length")
            
        # Extract perplexity and structure quality metrics
        perplexities = [m.perplexity for m in structure_metrics]
        tm_scores = [m.tm_score for m in structure_metrics]
        contact_precisions = [m.contact_precision_l for m in structure_metrics]
        plddt_scores = [m.plddt_score for m in structure_metrics]
        
        correlations = {}
        correlations["perplexity_tm_score"] = self._calculate_correlation(perplexities, tm_scores)
        correlations["perplexity_contact_precision"] = self._calculate_correlation(perplexities, contact_precisions)  
        correlations["perplexity_plddt"] = self._calculate_correlation(perplexities, plddt_scores)
        
        return correlations
    
    def analyze_scaling_behavior(self, model_sizes: List[int], 
                               performance_metrics: List[Dict]) -> Dict[str, Any]:
        """
        Analyze how structure prediction improves with model scale
        
        Based on ESM-2 paper analysis of 8M to 15B parameter scaling.
        
        Args:
            model_sizes: List of model parameter counts
            performance_metrics: List of performance metrics for each model size
            
        Returns:
            Scaling analysis results and trends
        """
        scaling_analysis = {
            "model_sizes": model_sizes,
            "scaling_trends": {},
            "emergence_thresholds": {},
            "performance_improvements": {}
        }
        
        # Analyze scaling trends for different metrics
        metrics_to_analyze = ["tm_score", "contact_precision_l", "perplexity"]
        
        for metric in metrics_to_analyze:
            values = [m.get(metric, 0) for m in performance_metrics]
            
            # Calculate scaling trend (log-linear fit)
            log_sizes = [math.log10(size) for size in model_sizes]
            trend_coeff = self._fit_scaling_trend(log_sizes, values)
            scaling_analysis["scaling_trends"][metric] = trend_coeff
            
            # Identify emergence thresholds (sharp improvements)
            thresholds = self._identify_emergence_thresholds(model_sizes, values)
            scaling_analysis["emergence_thresholds"][metric] = thresholds
            
            # Calculate relative improvements
            if len(values) > 1:
                improvement = (values[-1] - values[0]) / values[0] * 100
                scaling_analysis["performance_improvements"][metric] = improvement
        
        return scaling_analysis
    
    def predict_structure_quality(self, sequence: str, 
                                 attention_patterns: Optional[List] = None) -> StructurePredictionMetrics:
        """
        Predict structure quality metrics for a protein sequence
        
        Simulates the ESM-2 structure prediction pipeline.
        
        Args:
            sequence: Protein sequence
            attention_patterns: Optional pre-computed attention patterns
            
        Returns:
            Predicted structure quality metrics
        """
        sequence_length = len(sequence)
        
        # Simulate perplexity calculation
        perplexity = self._calculate_perplexity(sequence)
        
        # Extract contacts from attention
        contacts = self.extract_attention_contacts(sequence_length, attention_patterns)
        
        # Simulate structure quality metrics based on contacts and perplexity
        # In real implementation, this would use actual structure prediction
        tm_score = self._predict_tm_score_from_perplexity(perplexity)
        gdt_score = tm_score * 0.8  # Rough approximation
        
        # Contact precision (simulated)
        contact_precision_l = min(0.9, 0.6 + (6.0 - perplexity) * 0.1)
        contact_precision_l = max(0.1, contact_precision_l)
        
        # pLDDT confidence score
        plddt_score = self._predict_plddt_from_contacts(contacts, sequence_length)
        
        return StructurePredictionMetrics(
            tm_score=tm_score,
            gdt_score=gdt_score,
            contact_precision_l=contact_precision_l,
            contact_precision_l2=contact_precision_l * 0.9,
            contact_precision_l5=contact_precision_l * 0.8,
            perplexity=perplexity,
            plddt_score=plddt_score
        )
    
    def generate_structure_report(self, sequences: List[str]) -> Dict[str, Any]:
        """
        Generate comprehensive structure analysis report
        
        Args:
            sequences: List of protein sequences to analyze
            
        Returns:
            Complete structure analysis report
        """
        report = {
            "summary": {
                "num_sequences": len(sequences),
                "model_config": self.config,
                "analysis_timestamp": "2024-01-01T00:00:00Z"  # Would be actual timestamp
            },
            "individual_predictions": [],
            "aggregate_statistics": {},
            "scaling_analysis": {},
            "correlations": {}
        }
        
        # Analyze each sequence
        all_metrics = []
        for i, sequence in enumerate(sequences):
            metrics = self.predict_structure_quality(sequence)
            
            prediction_result = {
                "sequence_id": i,
                "sequence_length": len(sequence),
                "metrics": metrics.__dict__
            }
            report["individual_predictions"].append(prediction_result)
            all_metrics.append(metrics)
        
        # Calculate aggregate statistics
        if all_metrics:
            report["aggregate_statistics"] = {
                "mean_tm_score": sum(m.tm_score for m in all_metrics) / len(all_metrics),
                "mean_contact_precision": sum(m.contact_precision_l for m in all_metrics) / len(all_metrics),
                "mean_perplexity": sum(m.perplexity for m in all_metrics) / len(all_metrics),
                "mean_plddt": sum(m.plddt_score for m in all_metrics) / len(all_metrics)
            }
            
            # Analyze correlations
            report["correlations"] = self.analyze_perplexity_correlation(sequences, all_metrics)
        
        return report
    
    # Helper methods
    
    def _simulate_attention_patterns(self, sequence_length: int) -> List[List[List[List[float]]]]:
        """Simulate attention patterns for demonstration"""
        patterns = []
        for layer in range(self.num_layers):
            layer_patterns = []
            for head in range(self.num_heads):
                # Create realistic attention matrix with higher values for nearby residues
                head_pattern = []
                for i in range(sequence_length):
                    row = []
                    for j in range(sequence_length):
                        if i == j:
                            attention = 0.5
                        else:
                            # Decay with distance, with some long-range contacts
                            distance = abs(i - j)
                            attention = max(0.01, 0.3 * math.exp(-distance / 10.0))
                            # Add some random long-range contacts
                            if distance > 20 and random.random() < 0.05:
                                attention += 0.2
                        row.append(attention)
                    head_pattern.append(row)
                layer_patterns.append(head_pattern)
            patterns.append(layer_patterns)
        return patterns
    
    def _attention_to_contact_probability(self, attention_score: float, 
                                        layer_idx: int, head_idx: int) -> float:
        """Convert attention score to contact probability"""
        # Simple transformation - in practice this would be learned
        base_prob = min(0.95, attention_score * 2.0)
        
        # Later layers and certain heads might be more informative
        layer_weight = 1.0 + layer_idx * 0.1
        head_weight = 1.0 if head_idx < 5 else 0.8  # First few heads more structural
        
        return min(0.95, base_prob * layer_weight * head_weight)
    
    def _deduplicate_contacts(self, contacts: List[ContactPrediction]) -> List[ContactPrediction]:
        """Remove duplicate contacts and keep highest probability"""
        contact_dict = {}
        for contact in contacts:
            key = (min(contact.i, contact.j), max(contact.i, contact.j))
            if key not in contact_dict or contact.probability > contact_dict[key].probability:
                contact_dict[key] = contact
        return list(contact_dict.values())
    
    def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation coefficient"""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
            
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(xi * xi for xi in x)
        sum_y2 = sum(yi * yi for yi in y)
        
        numerator = n * sum_xy - sum_x * sum_y
        denominator = math.sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y))
        
        return numerator / denominator if denominator != 0 else 0.0
    
    def _fit_scaling_trend(self, log_sizes: List[float], values: List[float]) -> float:
        """Fit linear trend to log-scale data"""
        if len(log_sizes) < 2:
            return 0.0
        return self._calculate_correlation(log_sizes, values)
    
    def _identify_emergence_thresholds(self, sizes: List[int], values: List[float]) -> List[int]:
        """Identify model sizes where sharp improvements occur"""
        thresholds = []
        for i in range(1, len(values)):
            improvement = (values[i] - values[i-1]) / values[i-1] if values[i-1] > 0 else 0
            if improvement > 0.1:  # 10% improvement threshold
                thresholds.append(sizes[i])
        return thresholds
    
    def _calculate_perplexity(self, sequence: str) -> float:
        """Simulate perplexity calculation for a sequence"""
        # Simple heuristic based on sequence properties
        # In real implementation, this would use the actual language model
        
        # Factors that might affect perplexity:
        # - Sequence length (longer sequences might be harder)
        # - Amino acid diversity (more diverse = potentially lower perplexity)
        # - Presence of common motifs
        
        length_factor = min(2.0, len(sequence) / 100.0)
        diversity = len(set(sequence)) / 20.0  # 20 standard amino acids
        
        # Base perplexity with some randomness
        base_perplexity = 6.0 + length_factor * 0.5 - diversity * 1.0
        base_perplexity += random.normalvariate(0, 0.5)  # Add noise
        
        return max(3.0, min(12.0, base_perplexity))  # Clamp to reasonable range
    
    def _predict_tm_score_from_perplexity(self, perplexity: float) -> float:
        """Predict TM-score from perplexity (based on paper correlations)"""
        # ESM-2 paper shows strong negative correlation between perplexity and TM-score
        # TM-score ranges from 0 to 1, with 0.5 being the fold recognition threshold
        
        # Simple inverse relationship with some noise
        base_tm = max(0.2, min(0.9, 1.1 - perplexity / 8.0))
        base_tm += random.normalvariate(0, 0.05)  # Add some noise
        
        return max(0.1, min(0.95, base_tm))
    
    def _predict_plddt_from_contacts(self, contacts: List[ContactPrediction], 
                                   sequence_length: int) -> float:
        """Predict pLDDT confidence from contact predictions"""
        if not contacts:
            return 50.0
            
        # Higher confidence with more high-probability contacts
        high_conf_contacts = [c for c in contacts if c.probability > 0.5]
        contact_density = len(high_conf_contacts) / sequence_length
        
        base_plddt = 60.0 + contact_density * 30.0
        base_plddt += random.normalvariate(0, 5.0)  # Add noise
        
        return max(20.0, min(95.0, base_plddt))


def main():
    """Demo of structure analysis capabilities"""
    from esm2_hypergraph import create_esm2_hypergraph
    
    # Create hypergraph
    config = {
        "name": "esm_backbone",
        "trainable": True,
        "vocabulary_size": 33,
        "num_layers": 6,
        "num_heads": 20,
        "hidden_dim": 320,
        "intermediate_dim": 1280,
        "dropout": 0,
        "max_wavelength": 10000,
        "use_bias": True,
        "activation": "gelu",
        "layer_norm_eps": 0.00001,
        "use_pre_layer_norm": False,
        "position_embedding_type": "rotary",
        "max_sequence_length": 1026,
        "pad_token_id": 1
    }
    
    hypergraph = create_esm2_hypergraph(config)
    analyzer = ESM2StructureAnalyzer(hypergraph)
    
    # Example protein sequences
    sequences = [
        "MKLLVLGLGGTAAMAAAQPQPAPQPSAPQPLPQLPLQAQPQPQPQPQQLQQM",  # Example sequence 1
        "MKLLVLGLGGTAAMAGGGGPSPQPLPQLPLQAQPQPQPQPQQLQQMKLLVL",  # Example sequence 2
        "MEEGLLAAGGGPSPQPLPQLPLQAQPQPQPQPQQLQQMKLLVLGLGGTAAM"   # Example sequence 3
    ]
    
    print("ESM-2 Structure Analysis Demo")
    print("=" * 50)
    
    # Generate structure report
    report = analyzer.generate_structure_report(sequences)
    
    print(f"Analyzed {report['summary']['num_sequences']} sequences")
    print(f"Model: {report['summary']['model_config']['name']}")
    print()
    
    # Print aggregate statistics
    stats = report["aggregate_statistics"]
    print("Aggregate Statistics:")
    print(f"  Mean TM-score: {stats['mean_tm_score']:.3f}")
    print(f"  Mean Contact Precision: {stats['mean_contact_precision']:.3f}")
    print(f"  Mean Perplexity: {stats['mean_perplexity']:.3f}")
    print(f"  Mean pLDDT: {stats['mean_plddt']:.3f}")
    print()
    
    # Print correlations
    corr = report["correlations"]
    print("Perplexity Correlations:")
    print(f"  Perplexity vs TM-score: {corr['perplexity_tm_score']:.3f}")
    print(f"  Perplexity vs Contact Precision: {corr['perplexity_contact_precision']:.3f}")
    print(f"  Perplexity vs pLDDT: {corr['perplexity_plddt']:.3f}")
    
    # Save detailed report
    with open("structure_analysis_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\nDetailed report saved to structure_analysis_report.json")


if __name__ == "__main__":
    main()