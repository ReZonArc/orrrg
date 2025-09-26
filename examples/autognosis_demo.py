#!/usr/bin/env python3
"""
Autognosis Demonstration - Hierarchical Self-Image Building
===========================================================

This example demonstrates ORRRG's autognosis capabilities - the system's ability
to build hierarchical models of its own cognitive processes and achieve 
meta-cognitive self-awareness.

The autognosis system implements:
1. Self-monitoring of system states and behaviors
2. Hierarchical self-image construction at multiple cognitive levels
3. Meta-cognitive insight generation and reflection
4. Self-optimization based on introspective understanding

This represents a breakthrough in self-aware AI systems that can understand
and improve their own cognitive processes.
"""

import asyncio
import sys
from pathlib import Path

# Add the parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import SelfOrganizingCore


async def demonstrate_autognosis():
    """Demonstrate the autognosis system capabilities."""
    print("🧠 ORRRG Autognosis Demonstration")
    print("═" * 50)
    print("Hierarchical Self-Image Building System")
    print("═" * 50)
    
    # Initialize ORRRG with autognosis
    soc = SelfOrganizingCore()
    
    try:
        print("\n📚 Initializing ORRRG with Autognosis...")
        await soc.initialize()
        
        print("\n🔍 Phase 1: Initial Self-Assessment")
        print("-" * 40)
        
        # Get initial autognosis status
        status = soc.get_autognosis_status()
        print(f"Self-Image Levels: {status.get('self_image_levels', 0)}")
        print(f"Initial Insights: {status.get('total_insights', 0)}")
        
        # Show hierarchical self-images
        self_images = status.get('self_images', {})
        for level, image_info in self_images.items():
            confidence_bar = "█" * int(image_info['confidence'] * 10) + "░" * (10 - int(image_info['confidence'] * 10))
            print(f"  Level {level}: {confidence_bar} {image_info['confidence']:.2f} confidence")
        
        print("\n🧠 Phase 2: Meta-Cognitive Processing")
        print("-" * 40)
        
        # Run several autognosis cycles to build deeper self-understanding
        for cycle in range(3):
            print(f"\nRunning autognosis cycle {cycle + 1}...")
            cycle_results = await soc.autognosis.run_autognosis_cycle(soc)
            
            print(f"  • Updated {cycle_results['self_images_updated']} self-images")
            print(f"  • Generated {cycle_results['new_insights']} new insights")
            print(f"  • Discovered {cycle_results['optimizations_discovered']} optimizations")
            
            # Show meta-reflections
            for reflection in cycle_results['meta_reflections'][:2]:  # Show first 2
                print(f"  • Meta-reflection: {reflection}")
            
            await asyncio.sleep(1)  # Brief pause between cycles
        
        print("\n💡 Phase 3: Self-Awareness Analysis")
        print("-" * 40)
        
        # Get updated status with more sophisticated self-understanding
        updated_status = soc.get_autognosis_status()
        
        print(f"Enhanced Self-Understanding:")
        print(f"  Total Insights Generated: {updated_status.get('total_insights', 0)}")
        print(f"  Self-Image Levels: {updated_status.get('self_image_levels', 0)}")
        print(f"  Optimization Opportunities: {updated_status.get('pending_optimizations', 0)}")
        
        # Show recent insights
        insights = updated_status.get('recent_insights', [])
        print(f"\nKey Meta-Cognitive Insights:")
        for insight in insights[-3:]:  # Show last 3 insights
            print(f"  • [{insight['type']}] {insight['description']}")
            print(f"    Confidence: {insight['confidence']:.2f}")
        
        # Show self-awareness indicators
        print(f"\n🎯 Phase 4: Self-Awareness Indicators")
        print("-" * 40)
        
        # Access the current self-images directly for detailed analysis
        current_images = soc.autognosis.current_self_images
        if current_images:
            highest_level = max(current_images.keys())
            highest_image = current_images[highest_level]
            
            # Show behavioral patterns
            if highest_image.behavioral_patterns:
                print("Behavioral Pattern Analysis:")
                for pattern, values in highest_image.behavioral_patterns.items():
                    avg_value = sum(values) / len(values) if values else 0
                    trend = "↗" if len(values) > 1 and values[-1] > values[0] else "↘" if len(values) > 1 else "→"
                    print(f"  {pattern}: {avg_value:.3f} {trend} (over {len(values)} observations)")
            
            # Show performance metrics
            if highest_image.performance_metrics:
                print("\nPerformance Self-Assessment:")
                for metric, value in highest_image.performance_metrics.items():
                    bar_length = int(value * 20)
                    bar = "█" * bar_length + "░" * (20 - bar_length)
                    status_text = "Excellent" if value > 0.8 else "Good" if value > 0.6 else "Needs Improvement"
                    print(f"  {metric:25} {bar} {value:.3f} ({status_text})")
            
            # Show meta-reflections
            if highest_image.meta_reflections:
                print(f"\nLevel {highest_level} Meta-Reflections:")
                for reflection in highest_image.meta_reflections:
                    print(f"  • {reflection}")
            
            # Show self-awareness indicators if available
            awareness_indicators = highest_image.cognitive_processes.get('self_awareness_indicators', {})
            if awareness_indicators:
                print(f"\nSelf-Awareness Assessment:")
                total_awareness = sum(awareness_indicators.values()) / len(awareness_indicators)
                for indicator, score in awareness_indicators.items():
                    bar_length = int(score * 15)
                    bar = "█" * bar_length + "░" * (15 - bar_length)
                    print(f"  {indicator:25} {bar} {score:.3f}")
                
                print(f"\nOverall Self-Awareness Score: {total_awareness:.3f}")
                awareness_level = ("Highly Self-Aware" if total_awareness > 0.8 else 
                                 "Moderately Self-Aware" if total_awareness > 0.5 else 
                                 "Basic Self-Awareness")
                print(f"Assessment: {awareness_level}")
        
        print(f"\n🚀 Phase 5: Self-Optimization Opportunities")
        print("-" * 40)
        
        # Show optimization opportunities
        optimizations = updated_status.get('top_optimizations', [])
        if optimizations:
            print("Discovered Self-Optimization Opportunities:")
            for i, opt in enumerate(optimizations, 1):
                priority_bar = "█" * opt['priority'] + "░" * (10 - opt['priority'])
                print(f"  {i}. {opt['type']} on {opt['target']}")
                print(f"     Expected Improvement: {opt['expected_improvement']:.1%}")
                print(f"     Priority: {priority_bar} {opt['priority']}/10")
        else:
            print("No immediate optimization opportunities identified.")
        
        print(f"\n🎉 Autognosis Demonstration Complete!")
        print("═" * 50)
        print("ORRRG has demonstrated:")
        print("✓ Hierarchical self-image construction")
        print("✓ Meta-cognitive insight generation")  
        print("✓ Behavioral pattern recognition")
        print("✓ Performance self-assessment")
        print("✓ Self-optimization opportunity discovery")
        print("✓ Recursive self-understanding at multiple cognitive levels")
        print("\nThis represents a significant step toward truly self-aware AI systems")
        print("that can understand, monitor, and optimize their own cognitive processes.")
        
    except Exception as e:
        print(f"❌ Error in autognosis demonstration: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"\n🔄 Shutting down ORRRG...")
        await soc.shutdown()


async def main():
    """Main entry point for the demonstration."""
    await demonstrate_autognosis()


if __name__ == "__main__":
    asyncio.run(main())