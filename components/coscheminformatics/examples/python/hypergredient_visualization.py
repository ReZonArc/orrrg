#!/usr/bin/env python3
"""
hypergredient_visualization.py

🧬 Hypergredient Framework Visualization System
Advanced visualization and reporting capabilities for formulation analysis

This module implements:
1. Formulation radar charts and property visualization
2. Interaction network diagrams
3. Cost breakdown pie charts  
4. Performance timeline predictions
5. Comparative analysis charts
6. Comprehensive reporting dashboard
"""

import time
import json
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict

# Import hypergredient framework
try:
    from hypergredient_framework import (
        HypergredientClass, HypergredientDatabase, HypergredientFormulation,
        HypergredientOptimizer, HypergredientInteractionMatrix, HYPERGREDIENT_DATABASE
    )
    from hypergredient_evolution import FormulationEvolution, HypergredientAI, PerformanceFeedback
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from hypergredient_framework import (
        HypergredientClass, HypergredientDatabase, HypergredientFormulation,
        HypergredientOptimizer, HypergredientInteractionMatrix, HYPERGREDIENT_DATABASE
    )
    from hypergredient_evolution import FormulationEvolution, HypergredientAI, PerformanceFeedback

@dataclass
class VisualizationData:
    """Data structure for visualization components"""
    chart_type: str
    title: str
    data: Dict[str, Any]
    description: str
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

class HypergredientVisualizer:
    """Comprehensive visualization system for hypergredient formulations"""
    
    def __init__(self):
        self.database = HypergredientDatabase()
        self.interaction_matrix = HypergredientInteractionMatrix()
        
    def generate_formulation_report(self, formulation: HypergredientFormulation) -> Dict[str, VisualizationData]:
        """Generate comprehensive visual report for a formulation"""
        
        report = {}
        
        # 1. Radar chart - Efficacy profile
        report['radar_chart'] = self._create_efficacy_radar_chart(formulation)
        
        # 2. Network diagram - Ingredient interactions
        report['interaction_network'] = self._create_interaction_network(formulation)
        
        # 3. Pie chart - Cost breakdown
        report['cost_breakdown'] = self._create_cost_pie_chart(formulation)
        
        # 4. Line chart - Performance prediction timeline
        report['performance_timeline'] = self._create_performance_timeline(formulation)
        
        # 5. Bar chart - Hypergredient class distribution
        report['class_distribution'] = self._create_class_distribution_chart(formulation)
        
        # 6. Compatibility matrix
        report['compatibility_matrix'] = self._create_compatibility_matrix(formulation)
        
        return report
    
    def _create_efficacy_radar_chart(self, formulation: HypergredientFormulation) -> VisualizationData:
        """Create radar chart showing formulation efficacy across different parameters"""
        
        # Calculate scores for different parameters
        parameters = {
            'Anti-Aging': 0.0,
            'Hydration': 0.0,
            'Brightening': 0.0,
            'Safety': 0.0,
            'Stability': 0.0,
            'Cost Efficiency': 0.0,
            'Synergy': formulation.synergy_score / 3.0,  # Normalize to 0-1
            'Bioavailability': 0.0
        }
        
        total_weight = 0.0
        
        for hg_class, data in formulation.hypergredients.items():
            for ing_data in data['ingredients']:
                ingredient = ing_data['ingredient']
                weight = ing_data['percentage'] / 100.0
                total_weight += weight
                
                # Map ingredient properties to parameters
                if hg_class in [HypergredientClass.CT, HypergredientClass.CS]:
                    parameters['Anti-Aging'] += ingredient.potency * weight / 10.0
                
                if hg_class == HypergredientClass.HY:
                    parameters['Hydration'] += ingredient.potency * weight / 10.0
                
                if hg_class == HypergredientClass.ML:
                    parameters['Brightening'] += ingredient.potency * weight / 10.0
                
                parameters['Safety'] += ingredient.safety_score * weight / 10.0
                parameters['Bioavailability'] += ingredient.bioavailability * weight / 100.0
                
                # Cost efficiency (inverse of cost)
                cost_efficiency = max(0.1, 1.0 - (ingredient.cost_per_gram / 500.0))  # Normalize to R500 max
                parameters['Cost Efficiency'] += cost_efficiency * weight
                
                # Stability
                stability_score = 1.0 if ingredient.stability == "stable" else 0.7 if ingredient.stability == "moderate" else 0.4
                parameters['Stability'] += stability_score * weight
        
        # Normalize by total weight
        if total_weight > 0:
            for param in parameters:
                if param != 'Synergy':  # Synergy already normalized
                    parameters[param] = min(1.0, parameters[param] / total_weight)
        
        radar_data = {
            'parameters': list(parameters.keys()),
            'values': list(parameters.values()),
            'max_value': 1.0,
            'formulation_id': formulation.id
        }
        
        return VisualizationData(
            chart_type='radar',
            title='Formulation Efficacy Profile',
            data=radar_data,
            description='Radar chart showing formulation performance across key parameters'
        )
    
    def _create_interaction_network(self, formulation: HypergredientFormulation) -> VisualizationData:
        """Create network diagram showing ingredient interactions and synergies"""
        
        nodes = []
        edges = []
        
        # Create nodes for each ingredient
        ingredient_list = []
        for hg_class, data in formulation.hypergredients.items():
            for ing_data in data['ingredients']:
                ingredient = ing_data['ingredient']
                percentage = ing_data['percentage']
                
                node = {
                    'id': ingredient.name,
                    'label': ingredient.name,
                    'class': hg_class.value,
                    'size': percentage * 2,  # Scale node size by concentration
                    'potency': ingredient.potency,
                    'safety': ingredient.safety_score,
                    'color': self._get_class_color(hg_class)
                }
                nodes.append(node)
                ingredient_list.append(ingredient)
        
        # Create edges for interactions
        for i, ingredient1 in enumerate(ingredient_list):
            for j, ingredient2 in enumerate(ingredient_list[i+1:], i+1):
                
                # Get interaction coefficient
                coefficient = self.interaction_matrix.get_interaction_coefficient(
                    ingredient1.hypergredient_class, ingredient2.hypergredient_class
                )
                
                # Only show significant interactions
                if abs(coefficient - 1.0) > 0.2:
                    edge = {
                        'source': ingredient1.name,
                        'target': ingredient2.name,
                        'weight': coefficient,
                        'color': 'green' if coefficient > 1.0 else 'red' if coefficient < 0.8 else 'gray',
                        'width': abs(coefficient - 1.0) * 5,  # Scale edge width by interaction strength
                        'interaction_type': 'synergy' if coefficient > 1.2 else 'antagonism' if coefficient < 0.8 else 'neutral'
                    }
                    edges.append(edge)
        
        network_data = {
            'nodes': nodes,
            'edges': edges,
            'total_nodes': len(nodes),
            'total_edges': len(edges),
            'network_density': len(edges) / (len(nodes) * (len(nodes) - 1) / 2) if len(nodes) > 1 else 0
        }
        
        return VisualizationData(
            chart_type='network',
            title='Ingredient Interaction Network',
            data=network_data,
            description='Network diagram showing synergies and antagonisms between ingredients'
        )
    
    def _create_cost_pie_chart(self, formulation: HypergredientFormulation) -> VisualizationData:
        """Create pie chart showing cost breakdown by ingredient"""
        
        cost_breakdown = []
        total_cost = 0.0
        
        for hg_class, data in formulation.hypergredients.items():
            for ing_data in data['ingredients']:
                ingredient = ing_data['ingredient']
                percentage = ing_data['percentage']
                
                # Calculate cost for 50g product
                ingredient_cost = ingredient.cost_per_gram * (percentage / 100.0) * 50
                total_cost += ingredient_cost
                
                cost_item = {
                    'ingredient': ingredient.name,
                    'cost': ingredient_cost,
                    'percentage_of_total': 0.0,  # Will be calculated after total is known
                    'class': hg_class.value,
                    'color': self._get_class_color(hg_class)
                }
                cost_breakdown.append(cost_item)
        
        # Calculate percentages
        for item in cost_breakdown:
            item['percentage_of_total'] = (item['cost'] / total_cost * 100) if total_cost > 0 else 0
        
        # Sort by cost
        cost_breakdown.sort(key=lambda x: x['cost'], reverse=True)
        
        pie_data = {
            'items': cost_breakdown,
            'total_cost': total_cost,
            'currency': 'ZAR',
            'product_size': '50g'
        }
        
        return VisualizationData(
            chart_type='pie',
            title='Cost Breakdown by Ingredient',
            data=pie_data,
            description='Pie chart showing the cost contribution of each ingredient'
        )
    
    def _create_performance_timeline(self, formulation: HypergredientFormulation) -> VisualizationData:
        """Create timeline showing predicted performance over time"""
        
        # Simulate performance timeline based on ingredient properties
        timeline_points = []
        time_periods = [
            ('1 week', 1),
            ('2 weeks', 2),
            ('4 weeks', 4),
            ('8 weeks', 8),
            ('12 weeks', 12),
            ('16 weeks', 16),
            ('24 weeks', 24)
        ]
        
        for period_name, weeks in time_periods:
            performance_score = self._calculate_performance_at_time(formulation, weeks)
            timeline_points.append({
                'period': period_name,
                'weeks': weeks,
                'performance': performance_score,
                'confidence': self._calculate_prediction_confidence(formulation, weeks)
            })
        
        timeline_data = {
            'points': timeline_points,
            'formulation_id': formulation.id,
            'max_performance': max(point['performance'] for point in timeline_points),
            'plateau_week': self._estimate_plateau_time(timeline_points)
        }
        
        return VisualizationData(
            chart_type='line',
            title='Predicted Performance Timeline',
            data=timeline_data,
            description='Timeline showing expected efficacy improvement over time'
        )
    
    def _create_class_distribution_chart(self, formulation: HypergredientFormulation) -> VisualizationData:
        """Create bar chart showing distribution of hypergredient classes"""
        
        class_distribution = defaultdict(float)
        class_counts = defaultdict(int)
        
        for hg_class, data in formulation.hypergredients.items():
            total_percentage = data['total_percentage']
            ingredient_count = len(data['ingredients'])
            
            class_distribution[hg_class.value] = total_percentage
            class_counts[hg_class.value] = ingredient_count
        
        distribution_items = []
        for class_name, percentage in class_distribution.items():
            distribution_items.append({
                'class': class_name,
                'class_full_name': HYPERGREDIENT_DATABASE.get(
                    next(hc for hc in HypergredientClass if hc.value == class_name),
                    class_name
                ),
                'percentage': percentage,
                'ingredient_count': class_counts[class_name],
                'color': self._get_class_color_by_name(class_name)
            })
        
        # Sort by percentage
        distribution_items.sort(key=lambda x: x['percentage'], reverse=True)
        
        bar_data = {
            'items': distribution_items,
            'total_classes': len(distribution_items),
            'total_percentage': sum(item['percentage'] for item in distribution_items)
        }
        
        return VisualizationData(
            chart_type='bar',
            title='Hypergredient Class Distribution',
            data=bar_data,
            description='Bar chart showing concentration distribution across hypergredient classes'
        )
    
    def _create_compatibility_matrix(self, formulation: HypergredientFormulation) -> VisualizationData:
        """Create compatibility matrix showing ingredient interactions"""
        
        ingredients = []
        for hg_class, data in formulation.hypergredients.items():
            for ing_data in data['ingredients']:
                ingredients.append(ing_data['ingredient'])
        
        matrix_data = []
        for i, ingredient1 in enumerate(ingredients):
            row = []
            for j, ingredient2 in enumerate(ingredients):
                if i == j:
                    compatibility = 1.0  # Perfect self-compatibility
                else:
                    compatibility = self.interaction_matrix.get_interaction_coefficient(
                        ingredient1.hypergredient_class, ingredient2.hypergredient_class
                    )
                
                row.append({
                    'value': compatibility,
                    'color_intensity': min(1.0, max(0.0, (compatibility - 0.5) / 1.5)),
                    'ingredient1': ingredient1.name,
                    'ingredient2': ingredient2.name
                })
            matrix_data.append(row)
        
        compatibility_data = {
            'matrix': matrix_data,
            'ingredients': [ing.name for ing in ingredients],
            'size': len(ingredients)
        }
        
        return VisualizationData(
            chart_type='matrix',
            title='Ingredient Compatibility Matrix',
            data=compatibility_data,
            description='Matrix showing compatibility scores between all ingredient pairs'
        )
    
    def _get_class_color(self, hg_class: HypergredientClass) -> str:
        """Get color for hypergredient class"""
        color_map = {
            HypergredientClass.CT: '#FF6B6B',  # Red - Cellular Turnover
            HypergredientClass.CS: '#4ECDC4',  # Teal - Collagen Synthesis
            HypergredientClass.AO: '#45B7D1',  # Blue - Antioxidants
            HypergredientClass.BR: '#96CEB4',  # Green - Barrier Repair
            HypergredientClass.ML: '#FECA57',  # Yellow - Melanin Modulators
            HypergredientClass.HY: '#48CAE4',  # Light Blue - Hydration
            HypergredientClass.AI: '#F38BA8',  # Pink - Anti-Inflammatory
            HypergredientClass.MB: '#A8DADC',  # Light Gray - Microbiome
            HypergredientClass.SE: '#F1C0E8',  # Light Pink - Sebum Regulators
            HypergredientClass.PD: '#CFBAF0'   # Lavender - Penetration Enhancers
        }
        return color_map.get(hg_class, '#CCCCCC')
    
    def _get_class_color_by_name(self, class_name: str) -> str:
        """Get color for hypergredient class by name"""
        for hg_class in HypergredientClass:
            if hg_class.value == class_name:
                return self._get_class_color(hg_class)
        return '#CCCCCC'
    
    def _calculate_performance_at_time(self, formulation: HypergredientFormulation, weeks: int) -> float:
        """Calculate predicted performance at specific time point"""
        
        base_performance = formulation.efficacy_prediction / 100.0
        
        # Performance curve based on ingredient onset times
        time_factor = 1.0
        total_weight = 0.0
        
        for hg_class, data in formulation.hypergredients.items():
            for ing_data in data['ingredients']:
                ingredient = ing_data['ingredient']
                weight = ing_data['percentage'] / 100.0
                total_weight += weight
                
                # Estimate onset time in weeks
                onset_weeks = self._parse_onset_time(ingredient.onset_time)
                
                # S-curve for performance buildup
                if weeks <= onset_weeks:
                    ingredient_factor = 0.3 + 0.7 * (weeks / onset_weeks)  # Build up to full effect
                else:
                    ingredient_factor = 1.0  # Full effect
                
                time_factor += ingredient_factor * weight
        
        if total_weight > 0:
            time_factor = time_factor / total_weight
        
        # Apply plateau effect (performance plateaus around 16-20 weeks)
        plateau_factor = min(1.0, 1.0 - max(0.0, (weeks - 16) / 20))
        
        return min(1.0, base_performance * time_factor * plateau_factor)
    
    def _parse_onset_time(self, onset_time: Optional[str]) -> int:
        """Parse onset time string to weeks"""
        if not onset_time:
            return 4  # Default 4 weeks
        
        onset_time = onset_time.lower()
        
        if 'immediate' in onset_time:
            return 0.5
        elif 'week' in onset_time:
            try:
                return int(''.join(filter(str.isdigit, onset_time)))
            except ValueError:
                return 4
        elif 'day' in onset_time:
            try:
                days = int(''.join(filter(str.isdigit, onset_time)))
                return max(1, days // 7)
            except ValueError:
                return 2
        else:
            return 4  # Default
    
    def _calculate_prediction_confidence(self, formulation: HypergredientFormulation, weeks: int) -> float:
        """Calculate confidence in performance prediction"""
        
        confidence_factors = []
        
        for hg_class, data in formulation.hypergredients.items():
            for ing_data in data['ingredients']:
                ingredient = ing_data['ingredient']
                
                # Evidence level confidence
                if ingredient.evidence_level == "Strong":
                    confidence_factors.append(0.9)
                elif ingredient.evidence_level == "Moderate":
                    confidence_factors.append(0.7)
                else:
                    confidence_factors.append(0.5)
        
        base_confidence = sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5
        
        # Confidence decreases with longer time predictions
        time_penalty = max(0.3, 1.0 - (weeks / 52))  # Confidence decreases over a year
        
        return base_confidence * time_penalty
    
    def _estimate_plateau_time(self, timeline_points: List[Dict]) -> int:
        """Estimate when performance plateaus"""
        
        # Find the point where performance improvement slows significantly
        for i in range(1, len(timeline_points)):
            current_improvement = timeline_points[i]['performance'] - timeline_points[i-1]['performance']
            if current_improvement < 0.05:  # Less than 5% improvement
                return timeline_points[i]['weeks']
        
        return 16  # Default plateau at 16 weeks
    
    def compare_formulations(self, formulations: List[HypergredientFormulation]) -> Dict[str, VisualizationData]:
        """Create comparative analysis between multiple formulations"""
        
        comparison_report = {}
        
        # 1. Comparative radar chart
        comparison_report['comparative_radar'] = self._create_comparative_radar(formulations)
        
        # 2. Cost comparison
        comparison_report['cost_comparison'] = self._create_cost_comparison(formulations)
        
        # 3. Ingredient overlap analysis
        comparison_report['ingredient_overlap'] = self._create_ingredient_overlap(formulations)
        
        return comparison_report
    
    def _create_comparative_radar(self, formulations: List[HypergredientFormulation]) -> VisualizationData:
        """Create comparative radar chart for multiple formulations"""
        
        formulation_data = []
        
        for formulation in formulations:
            radar_data = self._create_efficacy_radar_chart(formulation)
            formulation_data.append({
                'id': formulation.id,
                'name': f"Formulation {formulation.id[-6:]}",  # Short ID
                'values': radar_data.data['values'],
                'color': f"#{hash(formulation.id) % 16777215:06x}"  # Generate color from ID
            })
        
        comparative_data = {
            'parameters': ['Anti-Aging', 'Hydration', 'Brightening', 'Safety', 'Stability', 
                          'Cost Efficiency', 'Synergy', 'Bioavailability'],
            'formulations': formulation_data,
            'max_value': 1.0
        }
        
        return VisualizationData(
            chart_type='comparative_radar',
            title='Formulation Comparison - Efficacy Profiles',
            data=comparative_data,
            description='Comparative radar chart showing performance across multiple formulations'
        )
    
    def _create_cost_comparison(self, formulations: List[HypergredientFormulation]) -> VisualizationData:
        """Create cost comparison chart"""
        
        cost_data = []
        
        for formulation in formulations:
            cost_data.append({
                'id': formulation.id,
                'name': f"Formulation {formulation.id[-6:]}",
                'cost': formulation.cost_total,
                'efficacy': formulation.efficacy_prediction,
                'cost_efficiency': formulation.efficacy_prediction / formulation.cost_total if formulation.cost_total > 0 else 0
            })
        
        # Sort by cost efficiency
        cost_data.sort(key=lambda x: x['cost_efficiency'], reverse=True)
        
        comparison_data = {
            'formulations': cost_data,
            'currency': 'ZAR',
            'best_value': cost_data[0] if cost_data else None
        }
        
        return VisualizationData(
            chart_type='cost_comparison',
            title='Cost-Efficacy Comparison',
            data=comparison_data,
            description='Comparison of cost vs efficacy across formulations'
        )
    
    def _create_ingredient_overlap(self, formulations: List[HypergredientFormulation]) -> VisualizationData:
        """Create ingredient overlap analysis"""
        
        all_ingredients = set()
        formulation_ingredients = {}
        
        # Collect all ingredients
        for formulation in formulations:
            ingredients = set()
            for hg_class, data in formulation.hypergredients.items():
                for ing_data in data['ingredients']:
                    ingredient_name = ing_data['ingredient'].name
                    ingredients.add(ingredient_name)
                    all_ingredients.add(ingredient_name)
            formulation_ingredients[formulation.id] = ingredients
        
        # Calculate overlap matrix
        overlap_matrix = []
        formulation_ids = list(formulation_ingredients.keys())
        
        for i, id1 in enumerate(formulation_ids):
            row = []
            for j, id2 in enumerate(formulation_ids):
                if i == j:
                    overlap = 1.0
                else:
                    set1 = formulation_ingredients[id1]
                    set2 = formulation_ingredients[id2]
                    overlap = len(set1.intersection(set2)) / len(set1.union(set2)) if set1.union(set2) else 0
                row.append(overlap)
            overlap_matrix.append(row)
        
        overlap_data = {
            'matrix': overlap_matrix,
            'formulation_ids': formulation_ids,
            'formulation_names': [f"Form {fid[-6:]}" for fid in formulation_ids],
            'total_unique_ingredients': len(all_ingredients),
            'common_ingredients': list(set.intersection(*formulation_ingredients.values()) if len(formulation_ingredients) > 1 else set())
        }
        
        return VisualizationData(
            chart_type='overlap_matrix',
            title='Ingredient Overlap Analysis',
            data=overlap_data,
            description='Analysis of ingredient overlap between formulations'
        )

def generate_visualization_html(report: Dict[str, VisualizationData]) -> str:
    """Generate HTML report with all visualizations"""
    
    html_template = """<!DOCTYPE html>
<html>
<head>
    <title>Hypergredient Formulation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                 color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
        .section {{ background: white; padding: 20px; margin: 20px 0; border-radius: 10px; 
                  box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .chart-container {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; 
                          border-radius: 5px; background: #fafafa; }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f2f2f2; }}
        .metric {{ display: inline-block; margin: 10px; padding: 15px; 
                 background: #e3f2fd; border-radius: 5px; min-width: 150px; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #1976d2; }}
        .metric-label {{ font-size: 14px; color: #666; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🧬 Hypergredient Formulation Report</h1>
        <p>Comprehensive analysis generated at {timestamp}</p>
    </div>
    
    {content}
    
    <div class="section">
        <h3>Report Summary</h3>
        <p>This report was generated using the Hypergredient Framework, providing advanced 
           cosmeceutical formulation analysis and optimization capabilities.</p>
    </div>
</body>
</html>"""
    
    content_sections = []
    
    for chart_name, viz_data in report.items():
        section_html = f"""
        <div class="section">
            <h2>{viz_data.title}</h2>
            <p>{viz_data.description}</p>
            <div class="chart-container">
                <h4>Chart Type: {viz_data.chart_type.title()}</h4>
                <pre>{json.dumps(viz_data.data, indent=2)}</pre>
            </div>
        </div>
        """
        content_sections.append(section_html)
    
    return html_template.format(
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        content="\n".join(content_sections)
    )

def demonstrate_visualization_system():
    """Demonstrate the visualization system capabilities"""
    
    print("=== HYPERGREDIENT VISUALIZATION SYSTEM DEMO ===\n")
    
    # Create test formulations
    optimizer = HypergredientOptimizer()
    
    # Anti-aging formulation
    anti_aging = optimizer.optimize_formulation(
        target_concerns=['wrinkles', 'firmness'],
        skin_type='mature',
        budget=1200,
        preferences=['gentle']
    )
    
    # Brightening formulation
    brightening = optimizer.optimize_formulation(
        target_concerns=['brightness', 'dullness'],
        skin_type='normal',
        budget=800,
        preferences=['proven']
    )
    
    # Hydrating formulation
    hydrating = optimizer.optimize_formulation(
        target_concerns=['dryness', 'barrier_damage'],
        skin_type='dry',
        budget=600,
        preferences=['gentle']
    )
    
    # Initialize visualizer
    visualizer = HypergredientVisualizer()
    
    print("1. Individual Formulation Report:")
    report = visualizer.generate_formulation_report(anti_aging)
    
    print(f"   Generated {len(report)} visualization components:")
    for viz_name, viz_data in report.items():
        print(f"     • {viz_data.title} ({viz_data.chart_type})")
    
    print(f"\n2. Radar Chart Data (Anti-Aging):")
    radar_data = report['radar_chart'].data
    for param, value in zip(radar_data['parameters'], radar_data['values']):
        print(f"     {param}: {value:.2f}")
    
    print(f"\n3. Cost Breakdown (Anti-Aging):")
    cost_data = report['cost_breakdown'].data
    print(f"     Total Cost: {cost_data['currency']} {cost_data['total_cost']:.2f}")
    for item in cost_data['items'][:3]:  # Top 3 most expensive
        print(f"     • {item['ingredient']}: {item['percentage_of_total']:.1f}% (R{item['cost']:.2f})")
    
    print(f"\n4. Interaction Network (Anti-Aging):")
    network_data = report['interaction_network'].data
    print(f"     Nodes: {network_data['total_nodes']}, Edges: {network_data['total_edges']}")
    print(f"     Network Density: {network_data['network_density']:.2f}")
    
    # Comparative analysis
    print(f"\n5. Comparative Analysis:")
    formulations = [anti_aging, brightening, hydrating]
    comparison_report = visualizer.compare_formulations(formulations)
    
    print(f"   Generated {len(comparison_report)} comparison charts:")
    for comp_name, comp_data in comparison_report.items():
        print(f"     • {comp_data.title}")
    
    # Cost comparison
    cost_comp = comparison_report['cost_comparison'].data
    print(f"\n6. Cost-Efficacy Ranking:")
    for i, form in enumerate(cost_comp['formulations']):
        print(f"     {i+1}. {form['name']}: R{form['cost']:.2f} ({form['efficacy']:.0f}% efficacy)")
        print(f"        Cost Efficiency: {form['cost_efficiency']:.2f} efficacy per rand")
    
    # Performance timeline
    print(f"\n7. Performance Timeline (Anti-Aging):")
    timeline_data = report['performance_timeline'].data
    print(f"   Predicted plateau at week {timeline_data['plateau_week']}")
    for point in timeline_data['points'][::2]:  # Every other point
        print(f"     {point['period']}: {point['performance']*100:.0f}% efficacy "
              f"(confidence: {point['confidence']*100:.0f}%)")
    
    # Generate HTML report
    print(f"\n8. HTML Report Generation:")
    html_report = generate_visualization_html(report)
    
    # Save to file
    with open('/tmp/hypergredient_report.html', 'w') as f:
        f.write(html_report)
    
    print(f"   HTML report saved to /tmp/hypergredient_report.html")
    print(f"   Report size: {len(html_report)} characters")
    
    # Save visualization data as JSON
    json_data = {}
    for viz_name, viz_data in report.items():
        json_data[viz_name] = {
            'chart_type': viz_data.chart_type,
            'title': viz_data.title,
            'data': viz_data.data,
            'description': viz_data.description,
            'timestamp': viz_data.timestamp
        }
    
    with open('/tmp/hypergredient_visualizations.json', 'w') as f:
        json.dump(json_data, f, indent=2, default=str)
    
    print(f"   Visualization data saved to /tmp/hypergredient_visualizations.json")

if __name__ == "__main__":
    demonstrate_visualization_system()