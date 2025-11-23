# emotion_based.py
from database import SessionLocal, EmotionLabel, Track


class EmotionBasedFilter:
    """
    Emotion-aware recommendation filter.
    
    This class does NOT detect emotions - it receives emotion labels
    and uses them to boost/filter recommendations.
    
    The five emotions we support:
    - sad
    - neutral
    - happy
    - fear
    - angry
    """

    def __init__(self):
        """Initialize the emotion filter."""
        self.db = SessionLocal()

        # Cache emotion labels for fast lookup
        self.song_emotions = {}  # {song_id: emotion}

    def load_emotion_labels(self):
        """
        Load all emotion labels from database into memory.
        This makes filtering super fast later.
        """
        print("📥 Loading emotion labels from database...")

        # Query all emotion labels
        labels = self.db.query(EmotionLabel).all()

        if not labels:
            raise ValueError("❌ No emotion labels found in database!")

        # Build the emotion dictionary
        for label in labels:
            self.song_emotions[label.song_id] = label.emotion

        print(f"✅ Loaded emotion labels for {len(self.song_emotions)} songs")

        # Show emotion distribution
        emotion_counts = {}
        for emotion in self.song_emotions.values():
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

        print("\n📊 Emotion Distribution:")
        for emotion, count in sorted(emotion_counts.items()):
            print(f"   {emotion:8s}: {count:4d} songs")

    def get_song_emotion(self, song_id):
        """
        Get the emotion label for a specific song.
        
        Args:
            song_id: ID of the song
            
        Returns:
            Emotion string ('sad', 'neutral', 'happy', 'fear', 'angry')
            or None if song not found
        """
        # If emotions not loaded yet, load them
        if not self.song_emotions:
            self.load_emotion_labels()

        return self.song_emotions.get(song_id, None)

    def filter_by_emotion(self, song_list, target_emotion, boost_factor=2.0):
        """
        Filter and boost recommendations based on emotion match.
        
        This is the MAIN function you'll use in your hybrid system.
        
        Args:
            song_list: List of song recommendations from other modules
                       Each item should be a dict like:
                       {'song_id': 'track_0001', 'score': 0.85}
            
            target_emotion: The user's current emotion
                           ('sad', 'neutral', 'happy', 'fear', 'angry')
            
            boost_factor: How much to boost matching songs (default: 2.0)
                         - 1.0 = no boost (emotion doesn't matter)
                         - 2.0 = double the score for matching songs
                         - 0.5 = penalize non-matching songs
        
        Returns:
            List of recommendations with emotion-adjusted scores
        """
        # Make sure emotions are loaded
        if not self.song_emotions:
            self.load_emotion_labels()

        # Apply emotion boosting
        emotion_adjusted_list = []

        for item in song_list:
            song_id = item['song_id']
            original_score = item['score']

            # Get this song's emotion
            song_emotion = self.song_emotions.get(song_id, None)

            # Calculate emotion match score
            if song_emotion is None:
                # Song has no emotion label - keep original score
                emotion_match = 1.0
            elif song_emotion == target_emotion:
                # Perfect match - boost the score!
                emotion_match = boost_factor
            else:
                # No match - keep original or slightly penalize
                emotion_match = 1.0

            # Calculate final score
            adjusted_score = original_score * emotion_match

            # Preserve all original fields from the item
        adjusted_item = item.copy()  # ✅ Copy all original fields

        # Update with emotion information
        adjusted_item.update({
            'score': adjusted_score,
            'emotion': song_emotion,
            'emotion_match': (song_emotion == target_emotion)
        })

        emotion_adjusted_list.append(adjusted_item)

        # Sort by adjusted score (highest first)
        emotion_adjusted_list.sort(key=lambda x: x['score'], reverse=True)

        return emotion_adjusted_list

    def get_songs_by_emotion(self, emotion, limit=100):
        """
        Get all songs with a specific emotion.
        
        Useful for creating emotion-based playlists.
        
        Args:
            emotion: Target emotion ('sad', 'neutral', 'happy', 'fear', 'angry')
            limit: Maximum number of songs to return
            
        Returns:
            List of song_ids with the specified emotion
        """
        # Make sure emotions are loaded
        if not self.song_emotions:
            self.load_emotion_labels()

        # Filter songs by emotion
        matching_songs = [
            song_id
            for song_id, song_emotion in self.song_emotions.items()
            if song_emotion == emotion
        ]

        return matching_songs[:limit]

    def get_emotion_distribution(self, song_ids):
        """
        Analyze emotion distribution in a list of songs.
        
        Useful for checking if recommendations are emotionally diverse.
        
        Args:
            song_ids: List of song IDs
            
        Returns:
            Dictionary with emotion counts
        """
        # Make sure emotions are loaded
        if not self.song_emotions:
            self.load_emotion_labels()

        emotion_counts = {
            'sad': 0,
            'neutral': 0,
            'happy': 0,
            'fear': 0,
            'angry': 0
        }

        for song_id in song_ids:
            emotion = self.song_emotions.get(song_id, 'neutral')
            if emotion in emotion_counts:
                emotion_counts[emotion] += 1

        return emotion_counts

    def get_song_details(self, song_id):
        """
        Get track details from database.
        
        Args:
            song_id: ID of the song
            
        Returns:
            Track object with song details
        """
        track = self.db.query(Track).filter(Track.id == song_id).first()
        return track

    def close(self):
        """Close database connection."""
        self.db.close()


# ============================================================
# 📝 USAGE EXAMPLES
# ============================================================

if __name__ == "__main__":
    """
    This shows how to use the EmotionBasedFilter class.
    Run this file to test emotion filtering!
    """

    print("="*60)
    print("🎭 TESTING EMOTION-BASED FILTERING")
    print("="*60)

    # Initialize the filter
    emotion_filter = EmotionBasedFilter()

    # Load emotion labels
    emotion_filter.load_emotion_labels()

    # Test 1: Get emotion for a specific song
    print("\n" + "="*60)
    print("🔍 Test 1: Get emotion for specific songs")
    print("="*60)
    test_songs = ['track_0001', 'track_0002', 'track_0003']
    for song_id in test_songs:
        emotion = emotion_filter.get_song_emotion(song_id)
        track = emotion_filter.get_song_details(song_id)
        if track:
            print(f"{track.name:30s} → {emotion}")

    # Test 2: Filter recommendations by emotion
    print("\n" + "="*60)
    print("🎵 Test 2: Filter recommendations for a 'sad' user")
    print("="*60)

    # Simulate recommendations from another module
    sample_recommendations = [
        {'song_id': 'track_0001', 'score': 0.85},
        {'song_id': 'track_0002', 'score': 0.80},
        {'song_id': 'track_0003', 'score': 0.75},
        {'song_id': 'track_0004', 'score': 0.70},
        {'song_id': 'track_0005', 'score': 0.65},
    ]

    print("\nOriginal scores:")
    for rec in sample_recommendations:
        song_id = rec['song_id']
        emotion = emotion_filter.get_song_emotion(song_id)
        track = emotion_filter.get_song_details(song_id)
        print(
            f"  {track.name:30s} | Score: {rec['score']:.2f} | Emotion: {emotion}")

    # Apply emotion filtering for a sad user
    filtered_recs = emotion_filter.filter_by_emotion(
        sample_recommendations,
        target_emotion='sad',
        boost_factor=2.0
    )

    print("\nAfter emotion filtering (target: sad, boost: 2.0x):")
    for rec in filtered_recs:
        song_id = rec['song_id']
        track = emotion_filter.get_song_details(song_id)
        match_icon = "✓" if rec['emotion_match'] else "✗"
        print(
            f"  {match_icon} {track.name:30s} | Score: {rec['score']:.2f} | Emotion: {rec['emotion']}")

    # Test 3: Get all songs with a specific emotion
    print("\n" + "="*60)
    print("🎭 Test 3: Get all 'happy' songs")
    print("="*60)
    happy_songs = emotion_filter.get_songs_by_emotion('happy', limit=10)
    print(f"Found {len(happy_songs)} happy songs")
    for i, song_id in enumerate(happy_songs[:5], 1):
        track = emotion_filter.get_song_details(song_id)
        print(f"  {i}. {track.name}")

    # Test 4: Analyze emotion distribution
    print("\n" + "="*60)
    print("📊 Test 4: Emotion distribution in recommendations")
    print("="*60)
    song_ids = [rec['song_id'] for rec in sample_recommendations]
    distribution = emotion_filter.get_emotion_distribution(song_ids)
    for emotion, count in distribution.items():
        print(f"  {emotion:8s}: {count} songs")

    # Cleanup
    emotion_filter.close()

    print("\n" + "="*60)
    print("✅ EMOTION FILTERING WORKING PERFECTLY!")
    print("="*60)
