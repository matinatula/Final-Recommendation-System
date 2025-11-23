from ndcg import compare_models, make_recommend_fn_from_model
from hybrid import HybridRecommender
from collaborative_als import CollaborativeRecommender
from content_based import ContentBasedRecommender
from database import SessionLocal, UserListeningHistory


def sample_user_ids(limit=50):
    db = SessionLocal()
    try:
        rows = (
            db.query(UserListeningHistory.user_id)
              .distinct()
              .limit(limit)
              .all()
        )
        return [r[0] for r in rows]
    finally:
        db.close()


# ---- Load and prepare your models ----
print("Loading models...")

hybrid = HybridRecommender(
    content_weight=0.3, collaborative_weight=0.5, emotion_weight=0.2)
hybrid.load_and_train()

collab = CollaborativeRecommender()
collab.load_data()
collab.train()

content = ContentBasedRecommender()
content.load_features()

# ---- Create adapter functions ----
hybrid_fn = make_recommend_fn_from_model(hybrid, model_type="hybrid")
collab_fn = make_recommend_fn_from_model(collab, model_type="collaborative")
content_fn = make_recommend_fn_from_model(content, model_type="content")

# ---- Pick users to evaluate ----
users = sample_user_ids(limit=50)

# ---- Compare models ----
model_fns = {
    "Hybrid": hybrid_fn,
    "ALS Collaborative": collab_fn,
    "Content-Based": content_fn
}

compare_models(model_fns, users, k=10)
