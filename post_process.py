# post_process.py
import pandas as pd
import os
import orjson
from tqdm import tqdm

def consolidate_run(capture_dir: str):
    """
    Reads all data from a capture run, consolidates it into a single
    master Parquet file, and generates a queryable interval topology map.
    """
    print(f"Consolidating data for run: {capture_dir}")
    output_file = os.path.join(capture_dir, 'events_complete.parquet')
    topology_file = os.path.join(capture_dir, 'topology.json')

    # --- Part 1: Consolidate all events into a single DataFrame ---
    # (This logic is the same as before, recovering from chunks and the journal)
    chunk_files = sorted([os.path.join(capture_dir, f) for f in os.listdir(capture_dir) if f.startswith('events_chunk_') and f.endswith('.parquet')])
    journal_file = os.path.join(capture_dir, 'events_stream.jsonl')
    
    all_dfs = []
    if chunk_files:
        for chunk_file in tqdm(chunk_files, desc="Loading Parquet chunks"):
            all_dfs.append(pd.read_parquet(chunk_file))

    num_events_in_chunks = sum(len(df) for df in all_dfs)
    
    if os.path.exists(journal_file):
        journal_events = []
        with open(journal_file, 'rb') as f:
            for line in f: journal_events.append(orjson.loads(line))
        un_chunked_events = journal_events[num_events_in_chunks:]
        if un_chunked_events:
            all_dfs.append(pd.DataFrame(un_chunked_events))

    if not all_dfs:
        print("No data found to consolidate.")
        return

    complete_df = pd.concat(all_dfs, ignore_index=True)
    complete_df.to_parquet(output_file, index=False)
    print(f"Saved consolidated file to {output_file} with {len(complete_df)} total events.")

    # --- Part 2: UNSTUBBED - Generate the Interval Topology Map ---
    print("Generating interval topology map...")
    if complete_df.empty:
        print("DataFrame is empty, cannot generate topology.")
        return

    # Calculate end times for all durational events
    complete_df['end_timestamp'] = complete_df['start_timestamp'] + complete_df['delta_timestamp']

    # Collect all unique timestamps (starts and ends) that define intervals
    breakpoints = pd.concat([
        complete_df['start_timestamp'],
        complete_df['end_timestamp']
    ]).unique()
    breakpoints.sort()

    topology = {}
    if len(breakpoints) < 2:
        print("Not enough breakpoints to create intervals.")
        return

    for i in tqdm(range(len(breakpoints) - 1), desc="Building Topology"):
        start, end = breakpoints[i], breakpoints[i+1]
        
        # Don't create zero-duration intervals
        if start >= end:
            continue
            
        # Find the midpoint of the interval to check for active events
        midpoint = start + (end - start) / 2
        
        # Query the DataFrame to find all events that contain this midpoint
        # An event is active if: start_timestamp <= midpoint < end_timestamp
        active_events_df = complete_df[
            (complete_df['start_timestamp'] <= midpoint) &
            (complete_df['end_timestamp'] > midpoint)
        ]
        
        if not active_events_df.empty:
            active_event_ids = active_events_df['event_id'].tolist()
            interval_key = f"{start:.4f}-{end:.4f}"
            topology[interval_key] = active_event_ids

    # Save the final topology map to a JSON file
    with open(topology_file, "wb") as f:
        f.write(orjson.dumps(topology))
    print(f"Saved interval topology map to {topology_file}")
    print("Consolidation and post-processing complete.")

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        consolidate_run(sys.argv[1])
    else:
        print("Usage: python post_process.py <path_to_capture_directory>")