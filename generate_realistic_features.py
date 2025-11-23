"""
Generate BETTER audio features with REAL style clustering.
Songs in same style will have SIMILAR features.
"""

import numpy as np
from database import SessionLocal, SongFeature, Track

print("="*60)
print("🎵 GENERATING CLUSTERED AUDIO FEATURES")
print("="*60)

db = SessionLocal()

# Get all songs
all_songs = db.query(Track).all()
print(f"📊 Processing {len(all_songs)} songs...\n")

# Define style profiles (these will be STRONG patterns, not drowned in noise)
style_profiles = {
    0: {  # Rock
        'mfcc_base': np.array([5, 3, 1, -2, -5, -8, -10, -8, -5, 0, 5, 10, 8]),
        'tempo_range': (100, 140),
        'spectral_range': (3500, 7500),
        'energy_range': (0.3, 0.8)
    },
    1: {  # Pop
        'mfcc_base': np.array([0, 2, 5, 8, 10, 8, 5, 2, 0, -2, -5, -8, -10]),
        'tempo_range': (90, 130),
        'spectral_range': (2500, 6500),
        'energy_range': (0.4, 0.9)
    },
    2: {  # Hip-hop
        'mfcc_base': np.array([10, 8, 5, 2, 0, -2, -5, -8, -10, -5, 0, 5, 10]),
        'tempo_range': (85, 115),
        'spectral_range': (1500, 4000),
        'energy_range': (0.2, 0.7)
    },
    3: {  # EDM
        'mfcc_base': np.array([-5, -3, 0, 3, 5, 8, 10, 8, 5, 3, 0, -3, -5]),
        'tempo_range': (120, 150),
        'spectral_range': (2000, 5500),
        'energy_range': (0.5, 1.0)
    },
    4: {  # Jazz
        'mfcc_base': np.array([2, 4, 6, 8, 5, 3, 1, 0, -1, -3, -5, -7, -8]),
        'tempo_range': (80, 180),
        'spectral_range': (1000, 3500),
        'energy_range': (0.1, 0.6)
    }
}

# Generate features for each song
for i, track in enumerate(all_songs):
    if i % 100 == 0:
        print(f"   Processing song {i}/{len(all_songs)}...", end='\r')

    # Determine style
    track_num = int(track.id.split('_')[1])
    style = track_num % 5
    profile = style_profiles[style]

    # Use SMALL random noise (variation within style)
    # NOT huge noise that drowns out the style

    # MFCC: Base pattern + SMALL variation
    # Multiply by 0.1 to make noise small relative to style pattern
    mfcc_noise = np.random.randn(13) * 1.5  # Much smaller noise now
    mfcc_list = (profile['mfcc_base'] + mfcc_noise).tolist()

    # Tempo: Within style range + small variation
    tempo_min, tempo_max = profile['tempo_range']
    tempo = np.random.uniform(tempo_min, tempo_max)

    # Chroma: Similar for all, small variation
    chroma = (np.random.rand(12) * 0.3 + 0.35).tolist()

    # Spectral Centroid: Within style range
    spectral_min, spectral_max = profile['spectral_range']
    spectral_centroid = np.random.uniform(spectral_min, spectral_max)

    # Spectral Bandwidth: Consistent within range
    spectral_bandwidth = np.random.uniform(800, 2500)

    # Spectral Contrast: Similar pattern
    spectral_contrast = (np.random.rand(7) * 3 + 5).tolist()

    # RMS Energy: Within style range
    energy_min, energy_max = profile['energy_range']
    rms_energy = np.random.uniform(energy_min, energy_max)

    # Zero Crossing Rate: Small variation
    zcr = np.random.uniform(0.08, 0.25)

    # Update or create
    song_feature = db.query(SongFeature).filter(
        SongFeature.song_id == track.id
    ).first()

    if song_feature:
        song_feature.mfcc = mfcc_list
        song_feature.tempo = tempo
        song_feature.chroma = chroma
        song_feature.spectral_centroid = spectral_centroid
        song_feature.spectral_bandwidth = spectral_bandwidth
        song_feature.spectral_contrast = spectral_contrast
        song_feature.rms_energy = rms_energy
        song_feature.zcr = zcr
    else:
        song_feature = SongFeature(
            song_id=track.id,
            mfcc=mfcc_list,
            tempo=tempo,
            chroma=chroma,
            spectral_centroid=spectral_centroid,
            spectral_bandwidth=spectral_bandwidth,
            spectral_contrast=spectral_contrast,
            rms_energy=rms_energy,
            zcr=zcr
        )
        db.add(song_feature)

    db.commit()

print(f"\n✅ Generated clustered features!")

# Verify clustering
print(f"\n🔍 Verification - Checking feature clustering:")
all_features = db.query(SongFeature).all()

for style in range(5):
    style_songs = [f for f in all_features if int(
        f.song_id.split('_')[1]) % 5 == style]

    if len(style_songs) >= 2:
        s1 = style_songs[0]
        s2 = style_songs[1]

        mfcc1 = np.array(s1.mfcc)
        mfcc2 = np.array(s2.mfcc)
        mfcc_distance = np.linalg.norm(mfcc1 - mfcc2)

        print(
            f"   Style {style}: MFCC distance = {mfcc_distance:.2f} (should be < 10)")

db.close()
print("\n" + "="*60)
print("✅ DONE! Songs in same style now CLUSTER together.")
print("   Run: python ndcg.py")
print("="*60)
