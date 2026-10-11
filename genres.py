"""Genre tags used for conditional text generation."""
import json
from pathlib import Path
GENRES = ('ACTION','ADVENTURE','FANTASY','ROMANCE','MYSTERY','THRILLER','HORROR','SCI_FI','COMEDY','DRAMA','SLICE_OF_LIFE','SUPERNATURAL','HISTORICAL','TRAGEDY')
def normalize_genres(items):
    genres = []
    for item in items:
        genre = item.strip().upper().replace('-', '_').replace(' ', '_')
        if genre not in GENRES:
            raise ValueError(f'Unsupported genre: {genre}. Allowed: {GENRES}')
        if genre not in genres: genres.append(genre)
    if not genres: raise ValueError('At least one genre is required')
    return genres
def load_genres(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    return {key: normalize_genres(values) for key, values in data.items()}
def genre_prefix(genres):
    return ''.join(f'<GENRE_{g}>' for g in normalize_genres(genres)) + '\n<STORY>\n'
