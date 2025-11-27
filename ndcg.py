import numpy as np
from database import SessionLocal, UserListeningHistory, Track, SongFeature
from content_based import ContentBasedRecommender
from collaborative_als import CollaborativeRecommender
from hybrid import HybridRecommender


class NDCGEvaluator:
    """Evaluates recommendation quality using NDCG@k metric."""

    def __init__(self):
        self.db = SessionLocal()

    def calculate_dcg(self, relevance_scores, k):
        """Calculate Discounted Cumulative Gain at position k."""
        relevance_scores = relevance_scores[:k]
        dcg = 0.0
        for i, rel in enumerate(relevance_scores):
            position = i + 1
            dcg += rel / np.log2(position + 1)
        return dcg

    def calculate_idcg(self, relevance_scores, k):
        """Calculate Ideal DCG (best possible ranking)."""
        ideal_relevance = sorted(relevance_scores, reverse=True)
        return self.calculate_dcg(ideal_relevance, k)

    def calculate_ndcg(self, relevance_scores, k):
        """
        Calculate Normalized Discounted Cumulative Gain.
        
        Returns:
            NDCG score between 0 and 1 (1 = perfect ranking)
        """
        dcg = self.calculate_dcg(relevance_scores, k)
        idcg = self.calculate_idcg(relevance_scores, k)

        if idcg == 0:
            return 0.0

        return dcg / idcg

    def get_relevance_scores(self, user_id, recommended_song_ids):
        """
        Calculate relevance scores for recommended songs.
        
        A song is relevant if:
        1. User has listened to it before (high relevance)
        2. It's similar to songs the user has listened to (medium relevance)
        
        Args:
            user_id: User identifier
            recommended_song_ids: List of recommended song IDs
            
        Returns:
            List of relevance scores (0 to 1)
        """
        history = self.db.query(UserListeningHistory)\
            .filter(UserListeningHistory.user_id == user_id)\
            .all()

        if not history:
            return [0.0] * len(recommended_song_ids)

        user_listens = {}
        for h in history:
            user_listens[h.song_id] = h.listen_count

        all_features = {}
        features_query = self.db.query(SongFeature).all()
        for f in features_query:
            all_features[f.song_id] = {
                'tempo': f.tempo,
                'energy': f.rms_energy
            }

        relevance_scores = []
        for song_id in recommended_song_ids:
            if song_id in user_listens:
                listen_count = user_listens[song_id]
                relevance = min(1.0, np.log1p(listen_count) / 3.0)
                relevance_scores.append(relevance)
            else:
                if song_id not in all_features:
                    relevance_scores.append(0.0)
                    continue

                rec_features = all_features[song_id]
                similarities = []

                for listened_song_id in user_listens.keys():
                    if listened_song_id in all_features:
                        listened_features = all_features[listened_song_id]

                        tempo_diff = abs(
                            rec_features['tempo'] - listened_features['tempo']) / 100
                        energy_diff = abs(
                            rec_features['energy'] - listened_features['energy']) / 0.2

                        distance = tempo_diff + energy_diff
                        similarity = 1.0 / (1.0 + distance)

                        weight = np.log1p(user_listens[listened_song_id])
                        similarities.append(similarity * weight)

                if similarities:
                    avg_similarity = np.mean(similarities)
                    relevance = avg_similarity * 0.7
                    relevance_scores.append(relevance)
                else:
                    relevance_scores.append(0.0)

        return relevance_scores

    def evaluate_recommender(self, recommender, user_ids, k=10, emotion='neutral'):
        """
        Evaluate a recommender system using NDCG@k.
        
        Args:
            recommender: Recommender instance to evaluate
            user_ids: List of user IDs for evaluation
            k: Number of top recommendations to evaluate
            emotion: Target emotion (for hybrid system)
            
        Returns:
            Dictionary containing evaluation metrics
        """
        print(f"\nEvaluating on {len(user_ids)} users (NDCG@{k})...")

        ndcg_scores = []
        successful_evaluations = 0

        for user_id in user_ids:
            try:
                if isinstance(recommender, HybridRecommender):
                    recs = recommender.recommend(
                        user_id=user_id, target_emotion=emotion, top_n=k)
                    recommended_song_ids = [r['song_id'] for r in recs]

                elif isinstance(recommender, CollaborativeRecommender):
                    recs = recommender.recommend_for_user(
                        user_id=user_id, top_n=k)
                    recommended_song_ids = [r['song_id'] for r in recs]

                elif isinstance(recommender, ContentBasedRecommender):
                    history = self.db.query(UserListeningHistory)\
                        .filter(UserListeningHistory.user_id == user_id)\
                        .limit(20)\
                        .all()

                    if not history:
                        continue

                    user_song_ids = [h.song_id for h in history]
                    recs = recommender.recommend_for_user_history(
                        user_song_ids=user_song_ids, top_n=k)
                    recommended_song_ids = [r['song_id'] for r in recs]

                else:
                    raise ValueError("Unknown recommender type")

                relevance_scores = self.get_relevance_scores(
                    user_id, recommended_song_ids)
                ndcg = self.calculate_ndcg(relevance_scores, k)
                ndcg_scores.append(ndcg)
                successful_evaluations += 1

            except Exception:
                continue

        if ndcg_scores:
            mean_ndcg = np.mean(ndcg_scores)
            std_ndcg = np.std(ndcg_scores)
            min_ndcg = np.min(ndcg_scores)
            max_ndcg = np.max(ndcg_scores)
        else:
            mean_ndcg = std_ndcg = min_ndcg = max_ndcg = 0.0

        print(
            f"Evaluated {successful_evaluations}/{len(user_ids)} users successfully")

        return {
            'mean_ndcg': mean_ndcg,
            'std_ndcg': std_ndcg,
            'min_ndcg': min_ndcg,
            'max_ndcg': max_ndcg,
            'num_users': successful_evaluations,
            'all_scores': ndcg_scores
        }

    def compare_recommenders(self, user_ids, k=10):
        """
        Compare content-based, collaborative, and hybrid recommenders.
        
        Args:
            user_ids: List of user IDs for evaluation
            k: Number of top recommendations to evaluate
            
        Returns:
            Dictionary containing results for all systems
        """
        print("\n" + "="*60)
        print("COMPARING RECOMMENDATION SYSTEMS")
        print("="*60)

        results = {}

        print("\n[1/3] Content-Based Recommender:")
        content_rec = ContentBasedRecommender()
        content_rec.load_features()
        results['content'] = self.evaluate_recommender(
            content_rec, user_ids, k)
        content_rec.close()

        print("\n[2/3] Collaborative Filtering:")
        collab_rec = CollaborativeRecommender()
        collab_rec.load_data()
        collab_rec.train()
        results['collaborative'] = self.evaluate_recommender(
            collab_rec, user_ids, k)
        collab_rec.close()

        print("\n[3/3] Hybrid System:")
        hybrid_rec = HybridRecommender()
        hybrid_rec.load_and_train()
        results['hybrid'] = self.evaluate_recommender(
            hybrid_rec, user_ids, k, emotion='neutral')
        hybrid_rec.close()

        return results

    def print_comparison_table(self, results):
        """Print evaluation results in table format."""
        print("\n" + "="*60)
        print("NDCG EVALUATION RESULTS")
        print("="*60)

        print(
            f"\n{'System':<20} {'Mean NDCG':<12} {'Std Dev':<12} {'Min':<8} {'Max':<8}")
        print("-" * 60)

        for system_name, metrics in results.items():
            print(f"{system_name.capitalize():<20} "
                  f"{metrics['mean_ndcg']:.4f}       "
                  f"{metrics['std_ndcg']:.4f}       "
                  f"{metrics['min_ndcg']:.4f}   "
                  f"{metrics['max_ndcg']:.4f}")

        print("\n" + "="*60)
        best_system = max(
            results.keys(), key=lambda x: results[x]['mean_ndcg'])
        best_score = results[best_system]['mean_ndcg']

        print(f"BEST SYSTEM: {best_system.upper()}")
        print(f"NDCG Score: {best_score:.4f}")

        if best_score > 0.8:
            print("Evaluation: Excellent recommendation quality")
        elif best_score > 0.6:
            print("Evaluation: Good recommendation quality")
        elif best_score > 0.4:
            print("Evaluation: Moderate recommendation quality")
        else:
            print("Evaluation: Needs improvement")

    def close(self):
        """Close database connection."""
        self.db.close()


if __name__ == "__main__":
    print("="*60)
    print("NDCG EVALUATION")
    print("="*60)

    evaluator = NDCGEvaluator()

    print("\nLoading test users...")
    db = SessionLocal()
    all_users = db.query(UserListeningHistory.user_id)\
        .distinct()\
        .limit(50)\
        .all()
    test_user_ids = [u[0] for u in all_users]
    print(f"Testing on {len(test_user_ids)} users")
    db.close()

    results = evaluator.compare_recommenders(user_ids=test_user_ids, k=10)
    evaluator.print_comparison_table(results)
    evaluator.close()

    print("\n" + "="*60)
    print("Evaluation complete")
    print("="*60)
