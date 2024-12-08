from sklearn.metrics.pairwise import euclidean_distances
import numpy as np

def calculate_sammon_error(original_data, selected_data):
    """Calculate Sammon error between original and selected feature spaces"""
    original_distances = euclidean_distances(original_data)
    selected_distances = euclidean_distances(selected_data)
    
    # Extract upper triangular indices (i < j)
    triu_indices = np.triu_indices_from(original_distances, k=1)
    
    original_dist = original_distances[triu_indices]
    selected_dist = selected_distances[triu_indices]
    
    # Handle cases where original_dist is zero to avoid division by zero
    epsilon = 1e-10
    original_dist = np.where(original_dist == 0, epsilon, original_dist)
    
    # Normalization constant
    c = 1.0 / np.sum(original_dist)
    
    # Calculate Sammon's error
    error = np.sum(((original_dist - selected_dist) ** 2) / original_dist)
    
    return c * error


from sklearn.metrics.pairwise import euclidean_distances
import numpy as np

def calculate_kruskal_stress(original_data, selected_data):
    """Calculate Kruskal stress between original and selected feature spaces"""
    original_distances = euclidean_distances(original_data)
    selected_distances = euclidean_distances(selected_data)
    
    # Extract upper triangular indices (i < j)
    triu_indices = np.triu_indices_from(original_distances, k=1)
    
    original_dist = original_distances[triu_indices]
    selected_dist = selected_distances[triu_indices]
    
    numerator = np.sum((original_dist - selected_dist) ** 2)
    denominator = np.sum(original_dist ** 2)
    
    return np.sqrt(numerator / denominator)
