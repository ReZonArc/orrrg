#!/usr/bin/env python3
"""
Inference Reliability and Accuracy Testing Suite

Comprehensive testing framework for evaluating:
1. Inference speed across different sequence lengths and batch sizes
2. Reliability metrics (consistency, error rates, stability)
3. Accuracy measurements (TM-score, contact precision, perplexity)
4. Memory usage and efficiency analysis
5. Comparison with standard benchmarks
"""

import json
import time
import math
import random
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict
import statistics

# Import existing modules
from esm2_hypergraph import create_esm2_hypergraph
from gpt2_hypergraph import create_gpt2_hypergraph
from scaling_analysis import ESM2ScalingAnalyzer, ModelConfiguration, ScalingMetrics
from folding_speed_analysis import ESMFoldSpeedAnalyzer


@dataclass
class InferenceTask:
    """Defines an inference task for testing"""
    task_id: str
    model_type: str
    sequence_length: int
    batch_size: int
    num_repeats: int
    expected_accuracy_min: float
    timeout_ms: float


@dataclass
class ReliabilityMetrics:
    """Metrics for reliability assessment"""
    consistency_score: float       # How consistent are results across runs
    error_rate: float             # Percentage of failed inferences
    stability_score: float        # How stable is performance over time
    memory_efficiency: float      # Memory usage efficiency
    throughput: float             # Sequences processed per second
    latency_p95: float           # 95th percentile latency


@dataclass
class AccuracyMetrics:
    """Metrics for accuracy assessment"""
    tm_score: float
    contact_precision_l: float
    contact_precision_l2: float
    contact_precision_l5: float
    perplexity: float
    structure_similarity: float
    prediction_confidence: float


@dataclass
class InferenceResult:
    """Result of a single inference test"""
    task_id: str
    success: bool
    inference_time_ms: float
    memory_usage_mb: float
    accuracy_metrics: Optional[AccuracyMetrics]
    error_message: Optional[str]
    timestamp: float


class SequenceGenerator:
    """Generate test protein sequences for benchmarking"""
    
    def __init__(self):
        # Standard amino acid alphabet
        self.amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        
        # Common protein motifs for realistic sequences
        self.motifs = [
            "MKLLVLGLGGTAAM",      # Signal peptide
            "GGGPSPQPLPQLPL",      # Flexible linker
            "HEAAHEAAHEAAHEAA",    # Helical repeat
            "YIGYGPYGPYGPYG",      # Beta sheet motif
            "PFGPFGPFGPFG",        # Turn motif
        ]
    
    def generate_random_sequence(self, length: int, seed: Optional[int] = None) -> str:
        """Generate a random protein sequence"""
        if seed is not None:
            random.seed(seed)
        
        return ''.join(random.choices(self.amino_acids, k=length))
    
    def generate_realistic_sequence(self, length: int, seed: Optional[int] = None) -> str:
        """Generate a more realistic protein sequence with motifs"""
        if seed is not None:
            random.seed(seed)
        
        sequence = ""
        remaining = length
        
        # Add some motifs
        while remaining > 20 and len(sequence) < length * 0.7:
            motif = random.choice(self.motifs)
            if remaining >= len(motif):
                sequence += motif
                remaining -= len(motif)
        
        # Fill remainder with random amino acids
        if remaining > 0:
            sequence += self.generate_random_sequence(remaining, seed)
        
        return sequence[:length]
    
    def generate_test_suite(self, num_sequences: int = 100) -> List[Tuple[int, str]]:
        """Generate a comprehensive test suite of sequences"""
        test_sequences = []
        
        # Different length categories
        length_categories = [
            (50, 10),    # Short sequences
            (100, 20),   # Medium sequences  
            (200, 30),   # Long sequences
            (500, 25),   # Very long sequences
            (1000, 15)   # Maximum length sequences
        ]
        
        seq_id = 0
        for target_length, count in length_categories:
            for i in range(min(count, num_sequences)):
                # Mix of random and realistic sequences
                if i % 2 == 0:
                    seq = self.generate_realistic_sequence(target_length, seed=seq_id)
                else:
                    seq = self.generate_random_sequence(target_length, seed=seq_id)
                
                test_sequences.append((target_length, seq))
                seq_id += 1
                
                if len(test_sequences) >= num_sequences:
                    break
            
            if len(test_sequences) >= num_sequences:
                break
        
        return test_sequences


class InferenceSimulator:
    """Simulate inference operations for testing"""
    
    def __init__(self):
        self.scaling_analyzer = ESM2ScalingAnalyzer()
        self.speed_analyzer = ESMFoldSpeedAnalyzer()
    
    def simulate_esm2_inference(self, sequence: str, model_config: ModelConfiguration) -> Tuple[float, AccuracyMetrics]:
        """Simulate ESM-2 inference with realistic metrics"""
        seq_length = len(sequence)
        
        # Base inference time (ms) - scales with sequence length and model size
        base_time = 10.0  # Base 10ms
        length_factor = seq_length / 100  # Linear with length
        model_factor = math.log10(model_config.num_parameters) / 6.0  # Log with model size
        
        inference_time = base_time * length_factor * model_factor
        
        # Add some realistic variance
        variance = random.gauss(1.0, 0.1)
        inference_time *= max(0.5, variance)
        
        # Generate accuracy metrics based on model and sequence
        metrics = self.scaling_analyzer.simulate_scaling_performance(model_config, [sequence])
        
        # Add sequence-specific factors
        gc_content = (sequence.count('G') + sequence.count('C')) / len(sequence)
        hydrophobic_ratio = sum(1 for aa in sequence if aa in 'AILMFPWYV') / len(sequence)
        
        # Adjust metrics based on sequence properties
        tm_score_adj = metrics.tm_score * (0.9 + 0.2 * hydrophobic_ratio)
        contact_precision_adj = metrics.contact_precision_l * (0.8 + 0.4 * gc_content)
        
        accuracy = AccuracyMetrics(
            tm_score=min(1.0, tm_score_adj),
            contact_precision_l=min(1.0, contact_precision_adj),
            contact_precision_l2=min(1.0, contact_precision_adj * 0.85),
            contact_precision_l5=min(1.0, contact_precision_adj * 0.70),
            perplexity=metrics.perplexity,
            structure_similarity=min(1.0, tm_score_adj * 0.9),
            prediction_confidence=min(1.0, (tm_score_adj + contact_precision_adj) / 2)
        )
        
        return inference_time, accuracy
    
    def simulate_memory_usage(self, sequence: str, model_config: ModelConfiguration, batch_size: int = 1) -> float:
        """Simulate memory usage during inference"""
        seq_length = len(sequence)
        
        # Base memory usage
        base_memory = 128  # MB
        
        # Scale with model parameters
        param_factor = model_config.num_parameters / 1e6  # Per million parameters
        model_memory = param_factor * 4  # ~4MB per million parameters
        
        # Scale with sequence length (attention matrices)
        attention_memory = (seq_length ** 2) * model_config.num_heads * 4 / (1024 * 1024)  # MB
        
        # Batch scaling
        batch_memory = attention_memory * batch_size
        
        total_memory = base_memory + model_memory + batch_memory
        
        return total_memory
    
    def calculate_reliability_score(self, results: List[InferenceResult]) -> ReliabilityMetrics:
        """Calculate reliability metrics from inference results"""
        if not results:
            return ReliabilityMetrics(0, 1.0, 0, 0, 0, 0)
        
        successful_results = [r for r in results if r.success]
        total_results = len(results)
        
        # Error rate
        error_rate = (total_results - len(successful_results)) / total_results
        
        if not successful_results:
            return ReliabilityMetrics(0, error_rate, 0, 0, 0, 0)
        
        # Inference times for analysis
        inference_times = [r.inference_time_ms for r in successful_results]
        memory_usages = [r.memory_usage_mb for r in successful_results]
        
        # Consistency score (based on coefficient of variation)
        if len(inference_times) > 1:
            time_std = statistics.stdev(inference_times)
            time_mean = statistics.mean(inference_times)
            consistency_score = 1.0 - min(1.0, time_std / time_mean)
        else:
            consistency_score = 1.0
        
        # Stability score (how performance changes over time)
        if len(inference_times) >= 10:
            # Compare first half vs second half
            mid = len(inference_times) // 2
            first_half_mean = statistics.mean(inference_times[:mid])
            second_half_mean = statistics.mean(inference_times[mid:])
            stability_score = 1.0 - min(1.0, abs(second_half_mean - first_half_mean) / first_half_mean)
        else:
            stability_score = consistency_score
        
        # Memory efficiency (lower is better, normalize to 0-1)
        avg_memory = statistics.mean(memory_usages)
        memory_efficiency = 1.0 / (1.0 + avg_memory / 1000)  # Normalize around 1GB
        
        # Throughput (sequences per second)
        avg_time_s = statistics.mean(inference_times) / 1000
        throughput = 1.0 / avg_time_s if avg_time_s > 0 else 0
        
        # 95th percentile latency
        latency_p95 = sorted(inference_times)[int(0.95 * len(inference_times))] if inference_times else 0
        
        return ReliabilityMetrics(
            consistency_score=consistency_score,
            error_rate=error_rate,
            stability_score=stability_score,
            memory_efficiency=memory_efficiency,
            throughput=throughput,
            latency_p95=latency_p95
        )


class InferenceReliabilityTester:
    """Main testing framework for inference reliability"""
    
    def __init__(self):
        self.sequence_generator = SequenceGenerator()
        self.inference_simulator = InferenceSimulator()
        self.test_results = []
    
    def create_test_tasks(self) -> List[InferenceTask]:
        """Create comprehensive test task suite"""
        tasks = []
        
        # Different model configurations
        model_configs = [
            ("esm2_8m", 8_000_000, 6, 20, 320, 1280),
            ("esm2_35m", 35_000_000, 12, 20, 480, 1920),
            ("esm2_150m", 150_000_000, 30, 20, 640, 2560)
        ]
        
        # Different sequence lengths and batch sizes
        test_scenarios = [
            (50, 1, 10, 0.3, 5000),     # Short sequences, single
            (100, 1, 10, 0.35, 8000),   # Medium sequences, single  
            (200, 1, 8, 0.4, 15000),    # Long sequences, single
            (500, 1, 5, 0.45, 30000),   # Very long sequences
            (50, 4, 8, 0.3, 10000),     # Short sequences, batch
            (100, 4, 8, 0.35, 20000),   # Medium sequences, batch
            (200, 2, 5, 0.4, 25000),    # Long sequences, small batch
        ]
        
        task_id = 0
        for model_name, params, layers, heads, hidden, intermediate in model_configs:
            for seq_len, batch_size, repeats, min_acc, timeout in test_scenarios:
                task = InferenceTask(
                    task_id=f"{model_name}_{seq_len}_{batch_size}_{task_id}",
                    model_type=model_name,
                    sequence_length=seq_len,
                    batch_size=batch_size,
                    num_repeats=repeats,
                    expected_accuracy_min=min_acc,
                    timeout_ms=timeout
                )
                tasks.append(task)
                task_id += 1
        
        return tasks
    
    def run_inference_task(self, task: InferenceTask) -> List[InferenceResult]:
        """Run a single inference task multiple times"""
        results = []
        
        # Generate test sequence
        test_sequence = self.sequence_generator.generate_realistic_sequence(
            task.sequence_length, 
            seed=hash(task.task_id) % 1000
        )
        
        # Create model configuration
        model_params = {
            "esm2_8m": (8_000_000, 6, 20, 320, 1280),
            "esm2_35m": (35_000_000, 12, 20, 480, 1920),
            "esm2_150m": (150_000_000, 30, 20, 640, 2560)
        }
        
        if task.model_type in model_params:
            params, layers, heads, hidden, intermediate = model_params[task.model_type]
            model_config = ModelConfiguration(
                name=task.model_type,
                num_parameters=params,
                num_layers=layers,
                num_heads=heads,
                hidden_dim=hidden,
                intermediate_dim=intermediate
            )
        else:
            # Default configuration
            model_config = ModelConfiguration(
                name=task.model_type,
                num_parameters=35_000_000,
                num_layers=12,
                num_heads=20,
                hidden_dim=480,
                intermediate_dim=1920
            )
        
        # Run inference multiple times
        for repeat in range(task.num_repeats):
            try:
                start_time = time.time()
                
                # Simulate inference
                inference_time, accuracy = self.inference_simulator.simulate_esm2_inference(
                    test_sequence, model_config
                )
                
                memory_usage = self.inference_simulator.simulate_memory_usage(
                    test_sequence, model_config, task.batch_size
                )
                
                # Check timeout
                if inference_time > task.timeout_ms:
                    result = InferenceResult(
                        task_id=task.task_id,
                        success=False,
                        inference_time_ms=inference_time,
                        memory_usage_mb=memory_usage,
                        accuracy_metrics=None,
                        error_message="Timeout exceeded",
                        timestamp=start_time
                    )
                else:
                    # Check minimum accuracy
                    success = accuracy.tm_score >= task.expected_accuracy_min
                    
                    result = InferenceResult(
                        task_id=task.task_id,
                        success=success,
                        inference_time_ms=inference_time,
                        memory_usage_mb=memory_usage,
                        accuracy_metrics=accuracy,
                        error_message=None if success else "Below minimum accuracy",
                        timestamp=start_time
                    )
                
                results.append(result)
                
            except Exception as e:
                result = InferenceResult(
                    task_id=task.task_id,
                    success=False,
                    inference_time_ms=0,
                    memory_usage_mb=0,
                    accuracy_metrics=None,
                    error_message=str(e),
                    timestamp=time.time()
                )
                results.append(result)
        
        return results
    
    def run_comprehensive_test_suite(self) -> Dict[str, Any]:
        """Run the complete test suite and generate report"""
        print("Running Comprehensive Inference Reliability Test Suite")
        print("=" * 60)
        
        # Create test tasks
        tasks = self.create_test_tasks()
        print(f"Created {len(tasks)} test tasks")
        
        # Run all tasks
        all_results = {}
        task_summaries = {}
        
        for i, task in enumerate(tasks):
            print(f"Running task {i+1}/{len(tasks)}: {task.task_id}")
            
            results = self.run_inference_task(task)
            all_results[task.task_id] = results
            
            # Calculate reliability metrics for this task
            reliability = self.inference_simulator.calculate_reliability_score(results)
            task_summaries[task.task_id] = {
                "task": asdict(task),
                "reliability": asdict(reliability),
                "num_results": len(results),
                "successful_runs": sum(1 for r in results if r.success)
            }
        
        # Calculate overall statistics
        all_inference_results = []
        for results in all_results.values():
            all_inference_results.extend(results)
        
        overall_reliability = self.inference_simulator.calculate_reliability_score(all_inference_results)
        
        # Generate comprehensive report
        report = {
            "test_suite_summary": {
                "timestamp": time.time(),
                "total_tasks": len(tasks),
                "total_inference_runs": len(all_inference_results),
                "overall_success_rate": sum(1 for r in all_inference_results if r.success) / len(all_inference_results),
                "test_duration_estimate": sum(len(results) for results in all_results.values()) * 0.1  # seconds
            },
            "overall_reliability": asdict(overall_reliability),
            "task_results": task_summaries,
            "performance_analysis": self.analyze_performance_trends(all_results),
            "recommendations": self.generate_recommendations(task_summaries, overall_reliability)
        }
        
        return report
    
    def analyze_performance_trends(self, all_results: Dict[str, List[InferenceResult]]) -> Dict[str, Any]:
        """Analyze performance trends across different conditions"""
        trends = {
            "sequence_length_impact": {},
            "batch_size_impact": {},
            "model_size_impact": {},
            "memory_scaling": {}
        }
        
        # Group results by different factors
        by_seq_length = defaultdict(list)
        by_batch_size = defaultdict(list)
        by_model_type = defaultdict(list)
        
        for task_id, results in all_results.items():
            # Parse task parameters from task_id
            parts = task_id.split('_')
            if len(parts) >= 4:
                model_type = f"{parts[0]}_{parts[1]}"
                seq_length = int(parts[2])
                batch_size = int(parts[3])
                
                successful_results = [r for r in results if r.success]
                if successful_results:
                    by_seq_length[seq_length].extend(successful_results)
                    by_batch_size[batch_size].extend(successful_results)
                    by_model_type[model_type].extend(successful_results)
        
        # Analyze sequence length impact
        for seq_len, results in by_seq_length.items():
            if results:
                avg_time = statistics.mean(r.inference_time_ms for r in results)
                avg_memory = statistics.mean(r.memory_usage_mb for r in results)
                avg_accuracy = statistics.mean(r.accuracy_metrics.tm_score for r in results if r.accuracy_metrics)
                
                trends["sequence_length_impact"][seq_len] = {
                    "avg_inference_time_ms": avg_time,
                    "avg_memory_mb": avg_memory,
                    "avg_tm_score": avg_accuracy,
                    "num_samples": len(results)
                }
        
        # Analyze batch size impact
        for batch_size, results in by_batch_size.items():
            if results:
                avg_time = statistics.mean(r.inference_time_ms for r in results)
                avg_memory = statistics.mean(r.memory_usage_mb for r in results)
                
                trends["batch_size_impact"][batch_size] = {
                    "avg_inference_time_ms": avg_time,
                    "avg_memory_mb": avg_memory,
                    "throughput_seq_per_sec": batch_size * 1000 / avg_time if avg_time > 0 else 0,
                    "num_samples": len(results)
                }
        
        # Analyze model size impact
        for model_type, results in by_model_type.items():
            if results:
                avg_time = statistics.mean(r.inference_time_ms for r in results)
                avg_memory = statistics.mean(r.memory_usage_mb for r in results)
                avg_accuracy = statistics.mean(r.accuracy_metrics.tm_score for r in results if r.accuracy_metrics)
                
                trends["model_size_impact"][model_type] = {
                    "avg_inference_time_ms": avg_time,
                    "avg_memory_mb": avg_memory,
                    "avg_tm_score": avg_accuracy,
                    "num_samples": len(results)
                }
        
        return trends
    
    def generate_recommendations(self, task_summaries: Dict[str, Any], 
                               overall_reliability: ReliabilityMetrics) -> Dict[str, Any]:
        """Generate recommendations based on test results"""
        
        # Find best performing configurations
        best_reliability = max(task_summaries.items(), 
                             key=lambda x: x[1]["reliability"]["consistency_score"])
        
        best_throughput = max(task_summaries.items(),
                            key=lambda x: x[1]["reliability"]["throughput"])
        
        lowest_memory = min(task_summaries.items(),
                          key=lambda x: x[1]["reliability"]["memory_efficiency"])
        
        recommendations = {
            "overall_assessment": {
                "reliability_grade": self.grade_reliability(overall_reliability),
                "key_strengths": [],
                "areas_for_improvement": []
            },
            "best_configurations": {
                "most_reliable": {
                    "task_id": best_reliability[0],
                    "consistency_score": best_reliability[1]["reliability"]["consistency_score"]
                },
                "highest_throughput": {
                    "task_id": best_throughput[0],
                    "throughput": best_throughput[1]["reliability"]["throughput"]
                },
                "memory_efficient": {
                    "task_id": lowest_memory[0],
                    "memory_efficiency": lowest_memory[1]["reliability"]["memory_efficiency"]
                }
            },
            "optimization_suggestions": [],
            "deployment_recommendations": []
        }
        
        # Add strengths and improvements based on metrics
        if overall_reliability.consistency_score > 0.8:
            recommendations["overall_assessment"]["key_strengths"].append("High consistency across runs")
        
        if overall_reliability.error_rate < 0.05:
            recommendations["overall_assessment"]["key_strengths"].append("Low error rate")
        
        if overall_reliability.throughput > 10:
            recommendations["overall_assessment"]["key_strengths"].append("Good throughput performance")
        
        # Areas for improvement
        if overall_reliability.consistency_score < 0.7:
            recommendations["overall_assessment"]["areas_for_improvement"].append("Consistency could be improved")
            recommendations["optimization_suggestions"].append("Investigate sources of inference time variation")
        
        if overall_reliability.error_rate > 0.1:
            recommendations["overall_assessment"]["areas_for_improvement"].append("Error rate is concerning")
            recommendations["optimization_suggestions"].append("Implement better error handling and recovery")
        
        if overall_reliability.memory_efficiency < 0.5:
            recommendations["overall_assessment"]["areas_for_improvement"].append("Memory usage could be optimized")
            recommendations["optimization_suggestions"].append("Consider memory optimization techniques")
        
        # Deployment recommendations
        if overall_reliability.latency_p95 < 1000:  # Under 1 second
            recommendations["deployment_recommendations"].append("Suitable for real-time applications")
        
        if overall_reliability.throughput > 5:
            recommendations["deployment_recommendations"].append("Good for batch processing workloads")
        
        if overall_reliability.error_rate < 0.02:
            recommendations["deployment_recommendations"].append("Reliable enough for production deployment")
        
        return recommendations
    
    def grade_reliability(self, reliability: ReliabilityMetrics) -> str:
        """Grade overall reliability on A-F scale"""
        score = (
            reliability.consistency_score * 0.3 +
            (1 - reliability.error_rate) * 0.3 +
            reliability.stability_score * 0.2 +
            reliability.memory_efficiency * 0.1 +
            min(1.0, reliability.throughput / 10) * 0.1
        )
        
        if score >= 0.9:
            return "A"
        elif score >= 0.8:
            return "B"
        elif score >= 0.7:
            return "C"
        elif score >= 0.6:
            return "D"
        else:
            return "F"
    
    def save_test_report(self, report: Dict[str, Any], filename: str = "inference_reliability_report.json") -> str:
        """Save test report to file"""
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        return filename


def main():
    """Run inference reliability testing suite"""
    print("Inference Reliability and Accuracy Testing Suite")
    print("=" * 50)
    
    tester = InferenceReliabilityTester()
    
    # Run comprehensive test suite
    report = tester.run_comprehensive_test_suite()
    
    # Save report
    report_file = tester.save_test_report(report)
    print(f"\n✓ Test report saved to {report_file}")
    
    # Display summary
    summary = report["test_suite_summary"]
    reliability = report["overall_reliability"]
    recommendations = report["recommendations"]
    
    print(f"\nTest Suite Summary:")
    print(f"  Total tasks: {summary['total_tasks']}")
    print(f"  Total inference runs: {summary['total_inference_runs']}")
    print(f"  Overall success rate: {summary['overall_success_rate']:.1%}")
    
    print(f"\nReliability Assessment:")
    print(f"  Grade: {recommendations['overall_assessment']['reliability_grade']}")
    print(f"  Consistency score: {reliability['consistency_score']:.3f}")
    print(f"  Error rate: {reliability['error_rate']:.1%}")
    print(f"  Throughput: {reliability['throughput']:.2f} seq/s")
    print(f"  95th percentile latency: {reliability['latency_p95']:.1f} ms")
    
    print(f"\nKey Strengths:")
    for strength in recommendations["overall_assessment"]["key_strengths"]:
        print(f"  • {strength}")
    
    print(f"\nAreas for Improvement:")
    for improvement in recommendations["overall_assessment"]["areas_for_improvement"]:
        print(f"  • {improvement}")


if __name__ == "__main__":
    main()