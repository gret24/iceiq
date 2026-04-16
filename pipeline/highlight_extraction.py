import json
import os
from collections import defaultdict

class HighlightExtractor:
    def __init__(self, config_path=os.path.expanduser('~/iceiq-dev/configs/highlight_extraction_config.json')):
        self.config = self._load_config(config_path)
        self.highlight_clips = []

    def _load_config(self, config_path):
        """Loads the highlight extraction configuration from a JSON file."""
        if not os.path.exists(config_path):
            print(f"Warning: Config file not found at {config_path}. Using default empty config.")
            return {} # Return empty config if not found
        with open(config_path, 'r') as f:
            return json.load(f)

    def _score_event(self, event):
        """
        Scores an individual event based on the configuration.
        Events are expected to have a 'type' and potentially other details.
        """
        event_type = event.get('type')
        if not event_type:
            return 0

        score_config = self.config.get('event_scores', {})
        base_score = score_config.get(event_type, 0)

        # Add logic for dynamic scoring based on event details (e.g., shot quality, hit severity)
        # For now, a simple base score.
        return base_score

    def _identify_key_moments(self, all_events):
        """
        Identifies key moments by scoring all events and applying a threshold.
        Returns a list of dictionaries, each representing a potential highlight.
        """
        scored_moments = []
        for event in all_events:
            score = self._score_event(event)
            if score > 0: # Only consider events with a positive score
                scored_moments.append({
                    'event': event,
                    'score': score,
                    'start_frame': event.get('frame_number', event.get('timestamp', 0)), # Use frame or timestamp
                    'end_frame': event.get('end_frame', event.get('frame_number', event.get('timestamp', 0)) + self.config.get('default_highlight_duration', 100)), # Default duration in frames
                    'description': event.get('description', f"{event.get('type')} event")
                })
        
        # Sort by score and apply a threshold if configured
        scored_moments = sorted(scored_moments, key=lambda x: x['score'], reverse=True)
        
        # Optionally, filter by a minimum score threshold
        min_score_threshold = self.config.get('min_score_threshold', 0)
        filtered_moments = [m for m in scored_moments if m['score'] >= min_score_threshold]

        # Further refinement: group proximate events, ensure unique highlights, etc.
        # This could involve merging highlights that are very close in time.
        return filtered_moments

    def extract_highlights(self, processed_data):
        """
        Main method to extract highlights from processed video data.
        :param processed_data: A dictionary containing various analysis results.
                               Expected keys: 'events', 'puck_possession_data', etc.
        """
        print("Extracting highlights...")
        all_events = processed_data.get('events', [])
        
        # You might combine events from different modules here (e.g., 'event_detection' and 'puck_possession_data')
        # For simplicity, let's assume 'events' contains all relevant detected events.
        
        key_moments = self._identify_key_moments(all_events)

        self.highlight_clips = []
        for moment in key_moments:
            # Determine actual clip start/end frames, potentially expanding around the event
            # based on configured pre/post event duration.
            pre_event_frames = self.config.get('pre_event_duration_frames', 60)
            post_event_frames = self.config.get('post_event_duration_frames', 120)
            
            start_frame = max(0, moment['start_frame'] - pre_event_frames)
            end_frame = moment['end_frame'] + post_event_frames

            self.highlight_clips.append({
                'start_frame': start_frame,
                'end_frame': end_frame,
                'description': moment['description'],
                'score': moment['score'],
                'event_details': moment['event'] # Keep original event details
            })

        print(f"Extracted {len(self.highlight_clips)} potential highlight clips.")
        return self.highlight_clips

    def get_highlights(self):
        """Returns the list of extracted highlight clips."""
        return self.highlight_clips

# Example Usage (for testing purposes, not part of the class)
if __name__ == '__main__':
    # Create a dummy config file for testing
    config_dir = os.path.expanduser('~/iceiq-dev/configs/')
    os.makedirs(config_dir, exist_ok=True)
    test_config_path = os.path.join(config_dir, 'highlight_extraction_config.json')
    
    test_config = {
        "event_scores": {
            "goal": 100,
            "shot_on_goal": 50,
            "big_hit": 70,
            "good_pass": 30,
            "save": 60,
            "turnover": -10 # Negative score for bad events
        },
        "min_score_threshold": 20,
        "default_highlight_duration": 100, # frames
        "pre_event_duration_frames": 90, # 3 seconds @ 30fps
        "post_event_duration_frames": 150 # 5 seconds @ 30fps
    }

    with open(test_config_path, 'w') as f:
        json.dump(test_config, f, indent=4)

    # Simulate processed data
    dummy_events = [
        {"type": "goal", "frame_number": 300, "player_id": 5, "team": "home", "description": "Player 5 scores!"},
        {"type": "shot_on_goal", "frame_number": 250, "player_id": 10, "team": "away", "description": "Player 10 shot saved."},
        {"type": "big_hit", "frame_number": 450, "player_id": 8, "target_player_id": 12, "description": "Big hit by Player 8."},
        {"type": "good_pass", "frame_number": 280, "player_id": 7, "target_player_id": 5, "description": "Great pass to Player 5."},
        {"type": "turnover", "frame_number": 100, "player_id": 3, "description": "Bad turnover."},
        {"type": "goal", "frame_number": 700, "player_id": 15, "team": "away", "description": "Player 15 snipes it!"}
    ]

    processed_data = {
        'events': dummy_events,
        'metadata': {'fps': 30, 'video_duration_frames': 1000}
    }

    extractor = HighlightExtractor(config_path=test_config_path)
    highlights = extractor.extract_highlights(processed_data)

    print("\n--- Extracted Highlights ---")
    for h in highlights:
        print(f"Description: {h['description']}, Score: {h['score']}, "
              f"Start Frame: {h['start_frame']}, End Frame: {h['end_frame']}")

    # Clean up dummy config
    # os.remove(test_config_path)
