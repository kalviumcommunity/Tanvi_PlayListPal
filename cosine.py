import numpy as np


## CONCEPT: Cosine Similarity ##

def cosine_similarity(vec1, vec2):
    """Calculates cosine similarity between two vectors."""
    
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
        
    return dot_product / (norm_vec1 * norm_vec2)


def run_similarity_calculation_demo():
    print("--- Running Cosine Similarity Calculation Demo ---")

    
    vec_workout = [0.1, 0.8, 0.2, 0.3]
    vec_gym = [0.15, 0.75, 0.22, 0.31]
    vec_sad = [0.9, -0.5, 0.1, -0.4]    

    print("Comparing three sample vectors:")
    print(f"  A (Workout Music): {vec_workout}")
    print(f"  B (Gym Songs):     {vec_gym}")
    print(f"  C (Sad Music):     {vec_sad}")
    
    
    similarity_A_B = cosine_similarity(vec_workout, vec_gym)
    similarity_A_C = cosine_similarity(vec_workout, vec_sad)

    print("\n--- Results ---")
    print(f"Similarity between A (Workout) and B (Gym): {similarity_A_B:.4f}")
    print(f"Similarity between A (Workout) and C (Sad): {similarity_A_C:.4f}")
    
    print("\nA score close to 1.0 means 'very similar'.")
    print("A score close to 0.0 means 'not very similar'.")

if __name__ == '__main__':
    run_similarity_calculation_demo()