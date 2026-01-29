"""MPD (Million Playlist Dataset) data utilities."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Dict, List, Any
from collections import Counter


@dataclass
class Track:
    """Represents a track in a playlist."""
    track_uri: str
    artist_uri: str
    album_uri: str
    track_name: str
    artist_name: str
    album_name: str
    duration_ms: int
    pos: int


@dataclass
class Playlist:
    """Represents a playlist from the MPD."""
    pid: int
    name: str
    tracks: List[Track]
    num_followers: int
    num_edits: int = 0
    modified_at: int = 0
    collaborative: bool = False


def iter_mpd_slices(mpd_dir: Path | str) -> Iterator[Dict[str, Any]]:
    """
    Memory-efficient generator that yields raw JSON data from MPD slices.

    Args:
        mpd_dir: Path to the MPD directory containing slice files.

    Yields:
        Dictionary containing the raw JSON data from each slice.
    """
    mpd_dir = Path(mpd_dir)
    slice_files = sorted(mpd_dir.glob("mpd.slice.*.json"))

    if not slice_files:
        raise FileNotFoundError(f"No MPD slice files found in {mpd_dir}")

    for slice_path in slice_files:
        with open(slice_path, "r", encoding="utf-8") as f:
            yield json.load(f)


def parse_track(raw: Dict[str, Any], pos: int) -> Track:
    """Parse a raw track dictionary into a Track object."""
    return Track(
        track_uri=raw["track_uri"],
        artist_uri=raw["artist_uri"],
        album_uri=raw["album_uri"],
        track_name=raw["track_name"],
        artist_name=raw["artist_name"],
        album_name=raw["album_name"],
        duration_ms=raw["duration_ms"],
        pos=pos,
    )


def parse_playlist(raw: Dict[str, Any]) -> Playlist:
    """
    Parse a raw playlist dictionary into a Playlist object.

    Args:
        raw: Raw playlist dictionary from MPD JSON.

    Returns:
        Parsed Playlist object.
    """
    tracks = [
        parse_track(t, i) for i, t in enumerate(raw.get("tracks", []))
    ]

    return Playlist(
        pid=raw["pid"],
        name=raw.get("name", ""),
        tracks=tracks,
        num_followers=raw.get("num_followers", 0),
        num_edits=raw.get("num_edits", 0),
        modified_at=raw.get("modified_at", 0),
        collaborative=raw.get("collaborative", "false") == "true",
    )


def iter_playlists(mpd_dir: Path | str) -> Iterator[Playlist]:
    """
    Memory-efficient generator that yields Playlist objects.

    Args:
        mpd_dir: Path to the MPD directory containing slice files.

    Yields:
        Parsed Playlist objects.
    """
    for slice_data in iter_mpd_slices(mpd_dir):
        for raw_playlist in slice_data.get("playlists", []):
            yield parse_playlist(raw_playlist)


def compute_dataset_statistics(mpd_dir: Path | str) -> Dict[str, Any]:
    """
    Compute statistics about the MPD dataset.

    Args:
        mpd_dir: Path to the MPD directory.

    Returns:
        Dictionary containing dataset statistics.
    """
    from tqdm import tqdm

    track_counts: Counter = Counter()
    artist_counts: Counter = Counter()
    album_counts: Counter = Counter()
    playlist_lengths: List[int] = []
    total_playlists = 0
    total_tracks = 0

    for playlist in tqdm(iter_playlists(mpd_dir), desc="Computing statistics"):
        total_playlists += 1
        playlist_lengths.append(len(playlist.tracks))

        for track in playlist.tracks:
            total_tracks += 1
            track_counts[track.track_uri] += 1
            artist_counts[track.artist_uri] += 1
            album_counts[track.album_uri] += 1

    import numpy as np
    lengths_array = np.array(playlist_lengths)

    return {
        "total_playlists": total_playlists,
        "total_track_occurrences": total_tracks,
        "unique_tracks": len(track_counts),
        "unique_artists": len(artist_counts),
        "unique_albums": len(album_counts),
        "playlist_length": {
            "mean": float(lengths_array.mean()),
            "std": float(lengths_array.std()),
            "min": int(lengths_array.min()),
            "max": int(lengths_array.max()),
            "median": float(np.median(lengths_array)),
            "p95": float(np.percentile(lengths_array, 95)),
        },
        "track_frequency": {
            "mean": float(np.mean(list(track_counts.values()))),
            "max": max(track_counts.values()),
            "min": min(track_counts.values()),
        },
    }


def get_track_uris_from_playlist(playlist: Playlist) -> List[str]:
    """Extract track URIs from a playlist in order."""
    return [track.track_uri for track in playlist.tracks]


def filter_playlists(
    mpd_dir: Path | str,
    min_length: int = 5,
    max_length: int = 500,
) -> Iterator[Playlist]:
    """
    Yield playlists that meet length criteria.

    Args:
        mpd_dir: Path to MPD directory.
        min_length: Minimum number of tracks.
        max_length: Maximum number of tracks.

    Yields:
        Filtered Playlist objects.
    """
    for playlist in iter_playlists(mpd_dir):
        if min_length <= len(playlist.tracks) <= max_length:
            yield playlist
