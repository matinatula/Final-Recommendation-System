import numpy as np
from ndcg import evaluate_model_on_users

# Define weight ranges (0.0 to 1.0 in steps of 0.1)
content_weights = np.arange(0.0, 1.1, 0.1)
collab_weights = np.arange(0.0, 1.1, 0.1)
emotion_weights = np.arange(0.0, 1.1, 0.1)

best_ndcg = -1
best_metrics = None
best_weights = None

# Iterate over all weight combinations
for cw in content_weights:
    for aw in collab_weights:
        for ew in emotion_weights:
            # Skip combinations that don't sum to 1
            if abs(cw + aw + ew - 1.0) > 1e-6:
                continue

            # Evaluate the hybrid model with these weights
            metrics = evaluate_model_on_users(
                model='Hybrid',
                content_weight=cw,
                collab_weight=aw,
                emotion_weight=ew
            )

            ndcg10 = metrics.get('NDCG@10', 0)

            if ndcg10 > best_ndcg:
                best_ndcg = ndcg10
                best_metrics = metrics
                best_weights = (cw, aw, ew)
                print(
                    f"New best NDCG@10: {ndcg10:.4f} with weights C:{cw} A:{aw} E:{ew}")

print("\n=== Optimal Hybrid Weights ===")
print(
    f"Content: {best_weights[0]}, Collaborative: {best_weights[1]}, Emotion: {best_weights[2]}")
print("Metrics at optimal weights:")
for metric, value in best_metrics.items():
    print(f"{metric}: {value:.4f}")
