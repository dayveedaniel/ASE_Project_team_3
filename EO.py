from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
import warnings
import utils

warnings.filterwarnings('ignore')

class MetricSelector:
    def __init__(self, fitness_func, dimension, max_features = None, omega=0.8, pool_size=4, max_iter=50, part_count=10, min_features=1,tolerance=1e-4, patience=5):
        self.omega = omega          # Weight between accuracy and number of features
        self.pool_size = pool_size  # Size of equilibrium pool
        self.max_iter = max_iter    # Maximum iterations
        self.part_count = part_count # Number of particles(solutions) in population
        self.scaler = StandardScaler()
        self.fitness_func = fitness_func
        self.min_features = min_features
        self.dimension = dimension
        self.max_features = int(0.5 * dimension)  if max_features is None else max_features
        self.tolerance = tolerance   # Minimum fitness improvement to continue
        self.patience = patience     # Number of iterations to tolerate without improvement
        self.no_improvement_count = 0  # Counter for patience

        # EO parameters
        self.a1 = 2
        self.a2 = 1
        self.GP = 0.5

        # save fitness for each iteration
        self.fitness_list:list = []

        assert self.min_features > 0 ,"Minimun number of features must be > 0"
        assert self.max_features >= self.min_features and self.max_features <= self.dimension,f"Max number of features must be in range {min_features} - {dimension}"
        
    
    def calculate_time(self, iteration):
        """Calculate time variable t for EO algorithm"""
        return (1 - iteration / self.max_iter) ** (self.a2 * iteration / self.max_iter)
    
    def preprocess_data(self, df: pd.DataFrame):
        """Preprocess the metrics data"""
        scaled_data = self.scaler.fit_transform(df)
        return scaled_data, df.columns
    
    def calculate_metric_importance(self, feature_selection_counts, feature_names):
        """Calculate importance scores for metrics based on selection frequency"""
        total_runs = len(feature_selection_counts)
        importance_scores = {}
        
        for i, feature in enumerate(feature_names):
            # selection_count = sum(1 for selection in feature_selection_counts if selection[i] == 1)
            selection_count = sum(selection[i] for selection in feature_selection_counts)
            importance_scores[feature] = selection_count / total_runs
            
        return importance_scores
    
    def select_features(self, data):
        """Main feature selection process"""
        feature_selection_counts = []
        
        # Initialize population with random feature subsets
        population = self.initialize_population()
        
        # Run EO algorithm multiple times
        for i in range(self.max_iter):
            eqPool, population, stop_EO = self.run_EO(population, data, i)
            # Store the selected features from this run
            feature_selection_counts.append(eqPool[0])
            if(stop_EO):
                break
        return feature_selection_counts
    
    def initialize_population(self):
        """Initialize population with random feature subsets"""
        population = np.zeros((self.part_count, self.dimension))
        # max_features for each solution
        
        for i in range(self.part_count):
            num_features = np.random.randint(self.min_features, self.max_features + 1)
            selected_features = np.random.choice(self.dimension, num_features, replace=False)
            population[i, selected_features] = 1
            
        return population
    
    def run_EO(self, population, data, current_iteration):
        """Run one iteration of the EO algorithm with early stopping"""
        # Early stop triggered or not
        stop_EO = False
        eqPool = np.zeros((self.pool_size, data.shape[1]))
        fitness_pool = np.full(self.pool_size, float('inf'))
        
        current_fitness = np.array([self.calculate_fitness(p, data) for p in population])
        
        # Update equilibrium pool
        all_solutions = np.vstack((population, eqPool))
        all_fitness = np.concatenate((current_fitness, fitness_pool))
        
        sorted_indices = np.argsort(all_fitness)
        eqPool = all_solutions[sorted_indices[:self.pool_size]]
        fitness_pool = all_fitness[sorted_indices[:self.pool_size]]
        
        # Calculate time
        t = self.calculate_time(current_iteration)
        new_population = np.zeros_like(population)
        
        # Update each particle
        for i in range(self.part_count):
            eq_idx = np.random.randint(0, self.pool_size)
            eq_candidate = eqPool[eq_idx]
            
            r1 = np.random.rand()
            r2 = np.random.rand()
            r3 = np.random.rand(data.shape[1])
            
            F = self.a1 * np.sign(r1 - 0.5) * (1 - np.exp(-2 * t))
            if r2 < 0.5:
                new_position = eq_candidate + F * (r3 * (eqPool[0] - population[i]))
            else:
                lambda_val = (1 - r2) * r3
                new_position = eq_candidate + F * (lambda_val * (eq_candidate - population[i]))
            
            prob = 1 / (1 + np.exp(-new_position))
            new_position = (prob > 0.5).astype(int)
            
            if np.sum(new_position) == 0:
                random_feature = np.random.randint(0, len(new_position))
                new_position[random_feature] = 1
            
            new_fitness = self.calculate_fitness(new_position, data)
            if new_fitness < current_fitness[i]:
                new_population[i] = new_position
            else:
                new_population[i] = population[i]
        
        best_fitness = min(fitness_pool[0], min(current_fitness))
        self.fitness_list.append(best_fitness)
        
        # Early Stopping Logic
        if len(self.fitness_list) > 1:
            improvement = abs(self.fitness_list[-2] - best_fitness)
            if improvement < self.tolerance:
                self.no_improvement_count += 1
            else:
                self.no_improvement_count = 0
            
            if self.no_improvement_count >= self.patience:
                stop_EO = True
                print(f"Early stopping triggered at iteration {current_iteration} with fitness: {best_fitness:.4f}")

        print(f'Iteration {current_iteration}, Best {self.fitness_func.__name__} fitness: {best_fitness:.4f}')
        return eqPool, new_population, stop_EO

    
    def calculate_fitness(self, particle, data):
        """Calculate fitness using fitness func"""
        if sum(particle) == 0:
            return float('inf')
            
        selected_features = data[:, particle == 1]
        error = self.fitness_func(data, selected_features)
            
        # Add penalty for number of features (normalized)
        feature_penalty = (sum(particle) / len(particle)) * (1 - self.omega)
        
        return error + feature_penalty
    
    def analyze_metrics(self, df):
        """Main method to analyze and select metrics"""
        scaled_data, feature_names = self.preprocess_data(df)
        feature_selections = self.select_features(scaled_data)
        '''Number of times selected/total number of iteration runs'''
        importance_scores = self.calculate_metric_importance(feature_selections, feature_names)
        return importance_scores


df = pd.read_csv("data/data_all.csv", index_col=0)
"""
result sample
{
sammons_error:[(0,1,2),(3,4,5)]
kruskal:[(0,1,2),(3,4,5)]
}
list of tuple containing (min,max,mean) for each iteration 
"""
#
fitness_result  = defaultdict(list)
range_ = range(2, df.shape[1], 2)

# Run analysis with both fitness types
for fitness in [utils.calculate_sammon_error, utils.calculate_kruskal_stress]:
    for max_feats in range_:
        print(f"\nUsing {fitness.__name__} and Max features = {max_feats}")
        selector = MetricSelector(
            omega=0.9,
            pool_size=4,
            max_iter=50,
            part_count=10,
            dimension=df.shape[1],
            fitness_func=fitness,
            max_features=max_feats
        )
        results = selector.analyze_metrics(df)
        results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
        fitness_result[fitness.__name__].append((np.min(selector.fitness_list), np.max(selector.fitness_list), np.mean(selector.fitness_list)))


        # Print results
        print("\nMetric Importance Scores:")
        print("-" * 40)
        print(f"{'Metric':<30} Score")
        print("-" * 40)
        for metric, score in results.items():
            print(f"{metric:<30} {score:.3f}")