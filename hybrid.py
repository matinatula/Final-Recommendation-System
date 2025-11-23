# hybrid.py
import numpy as np
from content_based import ContentBasedRecommender
from collaborative_als import CollaborativeRecommender
from emotion_based import EmotionBasedFilter
from database import SessionLocal, Track, UserListeningHistory


class HybridRecommender:
    """
    Hybrid recommendation system that combines:
    1. Content-based filtering (audio similarity)
    2. Collaborative filtering (user behavior patterns)
    3. Emotion-aware filtering (mood matching)
    
    This is the MAIN recommender your app will use!
    """

    def __init__(self,
                 content_weight=0.3,
                 collaborative_weight=0.5,
                 emotion_weight=0.2):
        """
        Initialize the hybrid recommender.
        
        Args:
            content_weight: Weight for content-based scores (default: 0.3)
            collaborative_weight: Weight for collaborative scores (default: 0.5)
            emotion_weight: Weight for emotion matching (default: 0.2)
        """
        # Store the weights
        self.w1 = content_weight
        self.w2 = collaborative_weight
        self.w3 = emotion_weight

        # Make sure weights sum to 1.0 (for fair comparison)
        total = self.w1 + self.w2 + self.w3
        if not np.isclose(total, 1.0):
            print(f"⚠️  Warning: Weights sum to {total}, normalizing...")
            self.w1 /= total
            self.w2 /= total
            self.w3 /= total

        # Initialize all three recommenders
        self.content_recommender = ContentBasedRecommender()
        self.collaborative_recommender = CollaborativeRecommender()
        self.emotion_filter = EmotionBasedFilter()

        # Database connection
        self.db = SessionLocal()

        print("✅ Hybrid Recommender initialized")
        print(
            f"   Weights: Content={self.w1:.2f}, Collaborative={self.w2:.2f}, Emotion={self.w3:.2f}")

    def load_and_train(self):
        """
        Load data and train all recommender modules.
        This must be called before making recommendations!
        """
        print("\n" + "="*60)
        print("🚀 LOADING AND TRAINING HYBRID SYSTEM")
        print("="*60)

        # Step 1: Load content-based features
        print("\n[1/3] Content-Based Recommender:")
        self.content_recommender.load_features()

        # Step 2: Load and train collaborative filtering
        print("\n[2/3] Collaborative Filtering (ALS):")
        self.collaborative_recommender.load_data()
        self.collaborative_recommender.train()

        # Step 3: Load emotion labels
        print("\n[3/3] Emotion-Based Filter:")
        self.emotion_filter.load_emotion_labels()

        print("\n" + "="*60)
        print("✅ HYBRID SYSTEM READY!")
        print("="*60)

    def _normalize_scores(self, recommendations):
        """
        Normalize scores to 0-1 range using min-max normalization.
        
        Why we need this:
        - Content-based scores are usually 0.5-1.0 (cosine similarity)
        - ALS scores can be -10 to +50 (latent factors)
        - Emotion scores are 0 or 1 (binary match)
        
        We need to normalize so all scores are comparable!
        
        Args:
            recommendations: List of {'song_id': ..., 'score': ...}
        
        Returns:
            Same list with normalized scores (0-1 range)
        """
        if not recommendations:
            return []

        # Extract all scores
        scores = [rec['score'] for rec in recommendations]

        # Find min and max
        min_score = min(scores)
        max_score = max(scores)

        # Avoid division by zero
        if max_score - min_score == 0:
            # All scores are the same - set them all to 0.5
            for rec in recommendations:
                rec['score'] = 0.5
            return recommendations

        # Apply min-max normalization: (x - min) / (max - min)
        normalized = []
        for rec in recommendations:
            original_score = rec['score']
            normalized_score = (original_score - min_score) / \
                (max_score - min_score)

            normalized.append({
                'song_id': rec['song_id'],
                'score': normalized_score
            })

        return normalized

    def recommend(self, user_id, target_emotion='neutral', top_n=10):
        """
        Generate hybrid recommendations for a user.
        
        This is THE main function your app will call!
        
        Args:
            user_id: ID of the user to recommend for
            target_emotion: User's current emotion 
                           ('sad', 'neutral', 'happy', 'fear', 'angry')
            top_n: Number of recommendations to return
        
        Returns:
            List of recommended songs with scores and details
        """
        print(f"\n🎵 Generating recommendations for {user_id}")
        print(f"   Target emotion: {target_emotion}")
        print(f"   Requested: top {top_n} songs")

        # ========================================
        # STEP 1: Get Collaborative Recommendations
        # ========================================
        print("\n[1/4] Getting collaborative filtering recommendations...")
        try:
            collaborative_recs = self.collaborative_recommender.recommend_for_user(
                user_id=user_id,
                top_n=50  # Get more than needed for better merging
            )
            print(
                f"   ✅ Got {len(collaborative_recs)} collaborative recommendations")
        except Exception as e:
            print(f"   ⚠️  Collaborative filtering failed: {e}")
            collaborative_recs = []

        # ========================================
        # STEP 2: Get Content-Based Recommendations
        # ========================================
        print("\n[2/4] Getting content-based recommendations...")
        try:
            # Get user's listening history to build their taste profile
            user_history = self.db.query(UserListeningHistory)\
                .filter(UserListeningHistory.user_id == user_id)\
                .order_by(UserListeningHistory.listen_count.desc())\
                .limit(20)\
                .all()

            if user_history:
                # Get songs user has listened to
                user_song_ids = [h.song_id for h in user_history]

                # Find similar songs
                content_recs = self.content_recommender.recommend_for_user_history(
                    user_song_ids=user_song_ids,
                    top_n=50
                )
                print(
                    f"   ✅ Got {len(content_recs)} content-based recommendations")
            else:
                print("   ⚠️  No listening history - skipping content-based")
                content_recs = []

        except Exception as e:
            print(f"   ⚠️  Content-based filtering failed: {e}")
            content_recs = []

        # ========================================
        # STEP 3: Normalize and Merge Scores
        # ========================================
        print("\n[3/4] Normalizing and merging scores...")

        # Normalize both score ranges to 0-1
        collaborative_recs = self._normalize_scores(collaborative_recs)
        content_recs = self._normalize_scores(content_recs)

        # Build a dictionary to merge scores by song_id
        merged_scores = {}

        # Add collaborative scores
        for rec in collaborative_recs:
            song_id = rec['song_id']
            merged_scores[song_id] = {
                'content_score': 0.0,
                'collaborative_score': rec['score'],
                'song_id': song_id
            }

        # Add content scores
        for rec in content_recs:
            song_id = rec['song_id']
            if song_id in merged_scores:
                # Song appears in both - update content score
                merged_scores[song_id]['content_score'] = rec['score']
            else:
                # Song only in content-based
                merged_scores[song_id] = {
                    'content_score': rec['score'],
                    'collaborative_score': 0.0,
                    'song_id': song_id
                }

        # Calculate hybrid score for each song (without emotion yet)
        hybrid_recs = []
        for song_id, scores in merged_scores.items():
            # Apply the hybrid formula: w1*content + w2*collaborative
            # Note: We save w3 for emotion scoring
            base_score = (
                self.w1 * scores['content_score'] +
                self.w2 * scores['collaborative_score']
            )

            hybrid_recs.append({
                'song_id': song_id,
                'score': base_score,
                'content_score': scores['content_score'],
                'collaborative_score': scores['collaborative_score']
            })

        # Sort by hybrid score (highest first)
        hybrid_recs.sort(key=lambda x: x['score'], reverse=True)

        print(f"   ✅ Merged into {len(hybrid_recs)} unique songs")

        # Debug: Show score distribution
        print(f"   📊 Score ranges:")
        if hybrid_recs:
            content_scores = [r['content_score']
                              for r in hybrid_recs if r['content_score'] > 0]
            collab_scores = [r['collaborative_score']
                             for r in hybrid_recs if r['collaborative_score'] > 0]
            if content_scores:
                print(
                    f"      Content: {min(content_scores):.3f} - {max(content_scores):.3f}")
            if collab_scores:
                print(
                    f"      Collaborative: {min(collab_scores):.3f} - {max(collab_scores):.3f}")

        # ========================================
        # STEP 4: Apply Emotion Scoring
        # ========================================
        print(
            f"\n[4/4] Applying emotion scoring (target: {target_emotion})...")

        # Add emotion scores to each recommendation
        emotion_matches = 0
        for rec in hybrid_recs:
            song_id = rec['song_id']
            song_emotion = self.emotion_filter.get_song_emotion(song_id)

            # Calculate emotion match score
            if song_emotion == target_emotion:
                emotion_score = 1.0  # Perfect match
                emotion_matches += 1
            else:
                emotion_score = 0.0  # No match

            # Store emotion info
            rec['emotion'] = song_emotion if song_emotion else 'unknown'
            rec['emotion_match'] = (song_emotion == target_emotion)

            # Add emotion component to final score
            # Final score = base_score + (w3 * emotion_match)
            rec['score'] = rec['score'] + (self.w3 * emotion_score)

        # Re-sort by updated scores (with emotion)
        hybrid_recs.sort(key=lambda x: x['score'], reverse=True)

        # Take final top N
        final_recs = hybrid_recs[:top_n]

        print(
            f"   ✅ Found {emotion_matches} songs matching emotion '{target_emotion}'")
        print(f"   ✅ Returning top {len(final_recs)} recommendations")

        # ========================================
        # STEP 5: Add Track Details
        # ========================================
        print("\n📋 Adding track details...")
        detailed_recs = []
        for rec in final_recs:
            track = self.db.query(Track).filter(
                Track.id == rec['song_id']).first()
            if track:
                detailed_recs.append({
                    'song_id': rec['song_id'],
                    'song_name': track.name,
                    'popularity': track.popularity,
                    'duration_ms': track.duration_ms,
                    'final_score': rec['score'],
                    'content_score': rec['content_score'],
                    'collaborative_score': rec['collaborative_score'],
                    'emotion': rec['emotion'],
                    'emotion_match': rec['emotion_match']
                })

        return detailed_recs

    def get_song_details(self, song_id):
        """Get track details from database."""
        track = self.db.query(Track).filter(Track.id == song_id).first()
        return track

    def update_weights(self, content_weight, collaborative_weight, emotion_weight):
        """
        Update the weights for the hybrid model.
        Useful for tuning based on NDCG results!
        
        Args:
            content_weight: New weight for content-based
            collaborative_weight: New weight for collaborative
            emotion_weight: New weight for emotion
        """
        self.w1 = content_weight
        self.w2 = collaborative_weight
        self.w3 = emotion_weight

        # Normalize
        total = self.w1 + self.w2 + self.w3
        self.w1 /= total
        self.w2 /= total
        self.w3 /= total

        print(
            f"✅ Updated weights: Content={self.w1:.2f}, Collaborative={self.w2:.2f}, Emotion={self.w3:.2f}")

    def close(self):
        """Close all database connections."""
        self.content_recommender.close()
        self.collaborative_recommender.close()
        self.emotion_filter.close()
        self.db.close()


# ============================================================
# 📝 USAGE EXAMPLE / TEST
# ============================================================

if __name__ == "__main__":
    """
    Test the hybrid recommender!
    Run this file to see it in action.
    """

    print("="*60)
    print("🎯 TESTING HYBRID RECOMMENDATION SYSTEM")
    print("="*60)

    # Initialize hybrid recommender
    hybrid = HybridRecommender(
        content_weight=0.3,
        collaborative_weight=0.5,
        emotion_weight=0.2
    )

    # Load and train all models
    hybrid.load_and_train()

    # Test with a user
    test_user_id = 'user_0001'

    # Test different emotions
    emotions_to_test = ['happy', 'sad', 'neutral']

    for emotion in emotions_to_test:
        print("\n" + "="*60)
        print(f"🎭 RECOMMENDATIONS FOR {test_user_id} (Emotion: {emotion})")
        print("="*60)

        recommendations = hybrid.recommend(
            user_id=test_user_id,
            target_emotion=emotion,
            top_n=10
        )

        print(f"\n📊 Top 10 Recommendations:")
        print(f"{'#':<4} {'Song Name':<35} {'Score':<8} {'Content':<8} {'Collab':<8} {'Emotion':<8} {'Match':<6}")
        print("-" * 95)

        for i, rec in enumerate(recommendations, 1):
            match_icon = "✓" if rec['emotion_match'] else "✗"
            print(f"{i:<4} {rec['song_name'][:34]:<35} "
                  f"{rec['final_score']:.3f}    "
                  f"{rec['content_score']:.3f}    "
                  f"{rec['collaborative_score']:.3f}    "
                  f"{rec['emotion']:<8} {match_icon:<6}")

    # Cleanup
    hybrid.close()

    print("\n" + "="*60)
    print("✅ HYBRID SYSTEM WORKING PERFECTLY!")
    print("="*60)
    print("\n💡 Next: Chapter 7 - NDCG Evaluation!")
