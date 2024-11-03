import random
from collections import defaultdict


def sample_diverse_combinations(data, N):
    # Step 1: Initialize variables to track diversity and selected samples
    selected_samples = []
    combination_persona_count = defaultdict(set)  # Track selected personas for each combination
    style_count = defaultdict(int)
    length_count = defaultdict(int)
    
    # Step 2: Flatten out all combinations to sample from them
    all_possible_samples = []
    
    for entry in data:
        combination = tuple(entry['combination'])  # Using tuple for hashable key
        nodes = entry['nodes']
        
        for persona in entry['personas']:
            for style in entry['styles']:
                for length in entry['lengths']:
                    all_possible_samples.append({
                        'combination': combination,
                        'persona': persona,
                        'nodes': nodes,
                        'style': style,
                        'length': length
                    })
                    
    # Step 3: Shuffle to ensure random selection
    random.shuffle(all_possible_samples)
    
    # Step 4: Sample with priority on combination, ensuring unique persona within combination
    for sample in all_possible_samples:
        if len(selected_samples) >= N:
            break
        
        combination = sample['combination']
        persona = sample['persona']
        style = sample['style']
        length = sample['length']
        
        # Check for priority diversity conditions
        # 1. Ensure unique persona within the same combination
        if persona not in combination_persona_count[combination]:
            selected_samples.append(sample)
            combination_persona_count[combination].add(persona)
        # 2. Next, check for style diversity if combination and persona already diverse enough
        elif style_count[style] < max(style_count.values(), default=0) + 1:
            selected_samples.append(sample)
            style_count[style] += 1
        # 3. Finally, check for length diversity
        elif length_count[length] < max(length_count.values(), default=0) + 1:
            selected_samples.append(sample)
            length_count[length] += 1
    
    return selected_samples