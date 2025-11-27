from database import SessionLocal, EmotionLabel, Track


class EmotionBasedFilter:
    """
    Emotion-aware filtering for recommendations.
    Boosts songs matching the user's current emotional state.
    
    Supported emotions: sad, neutral, happy, fear, angry
    """

    def __init__(self):
        self.db = SessionLocal()
        self.song_emotions = {}

    def load_emotion_labels(self):
        """Load emotion labels from database into memory for fast lookup."""
        print("Loading emotion labels from database...")

        labels = self.db.query(EmotionLabel).all()
        if not labels:
            raise ValueError("No emotion labels found in database")

        for label in labels:
            self.song_emotions[label.song_id] = label.emotion

        print(f"Loaded emotion labels for {len(self.song_emotions)} songs")

        emotion_counts = {}
        for emotion in self.song_emotions.values():
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

        print("\nEmotion distribution:")
        for emotion, count in sorted(emotion_counts.items()):
            print(f"  {emotion}: {count} songs")

    def get_song_emotion(self, song_id):
        """
        Get emotion label for a song.
        
        Args:
            song_id: Song identifier
            
        Returns:
            Emotion string or None if not found
        """
        if not self.song_emotions:
            self.load_emotion_labels()

        return self.song_emotions.get(song_id, None)

    def filter_by_emotion(self, song_list, target_emotion, boost_factor=2.0):
        """
        Adjust recommendation scores based on emotion matching.
        
        Args:
            song_list: List of recommendations [{'song_id': ..., 'score': ...}, ...]
            target_emotion: User's current emotion (sad/neutral/happy/fear/angry)
            boost_factor: Multiplier for matching emotions (default: 2.0)
            
        Returns:
            List of recommendations with adjusted scores
        """
        if not self.song_emotions:
            self.load_emotion_labels()

        emotion_adjusted_list = []

        for item in song_list:
            song_id = item['song_id']
            original_score = item['score']
            song_emotion = self.song_emotions.get(song_id, None)

            if song_emotion is None:
                emotion_match_multiplier = 1.0
            elif song_emotion == target_emotion:
                emotion_match_multiplier = boost_factor
            else:
                emotion_match_multiplier = 1.0

            adjusted_score = original_score * emotion_match_multiplier

            adjusted_item = item.copy()
            adjusted_item.update({
                'score': adjusted_score,
                'emotion': song_emotion,
                'emotion_match': (song_emotion == target_emotion)
            })

            emotion_adjusted_list.append(adjusted_item)

        emotion_adjusted_list.sort(key=lambda x: x['score'], reverse=True)

        return emotion_adjusted_list

    def get_songs_by_emotion(self, emotion, limit=100):
        """
        Retrieve songs with a specific emotion.
        
        Args:
            emotion: Target emotion
            limit: Maximum number of songs to return
            
        Returns:
            List of song IDs
        """
        if not self.song_emotions:
            self.load_emotion_labels()

        matching_songs = [
            song_id
            for song_id, song_emotion in self.song_emotions.items()
            if song_emotion == emotion
        ]

        return matching_songs[:limit]

    def get_emotion_distribution(self, song_ids):
        """
        Calculate emotion distribution for a list of songs.
        
        Args:
            song_ids: List of song IDs
            
        Returns:
            Dictionary with emotion counts
        """
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

    def close(self):
        """Close database connection."""
        self.db.close()
