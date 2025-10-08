# recording_stuff.py

import mss
import pynput
import torch
import pyarrow as pa
import pyarrow.parquet as pq
import orjson # Faster than the standard json library
import time
import uuid
from multiprocessing import Process, Queue
from threading import Thread, Lock

class MemoryAndDispatchManager:
    def __init__(self, buffer_shape, buffer_dtype):
        # Create a shared memory block for the circular buffer
        self.shm = shared_memory.SharedMemory(create=True, size=np.prod(buffer_shape) * np.dtype(buffer_dtype).itemsize)
        self.buffer = np.ndarray(buffer_shape, dtype=buffer_dtype, buffer=self.shm.buf)
        
        self.ref_counts = {}
        self.dispatch_times = {}
        self.results = {}
        self.write_head = 0 # Index for the circular buffer

    def run(self):
        # Start capture thread (writes to self.buffer)
        # Start analysis workers
        self.start_workers()

        # Main loop for dispatching and finalizing
        while True:
            # 1. Check for newly completed chunks to dispatch
            if new_chunk_is_ready:
                self.dispatch_chunk(chunk_id, start_index, num_frames)

            # 2. Check the return queue for results from workers
            if not return_queue.empty():
                result = return_queue.get()
                chunk_id = result['chunk_id']
                
                if chunk_id not in self.results: self.results[chunk_id] = []
                self.results[chunk_id].append(result)
                self.ref_counts[chunk_id] -= 1

            # 3. Check for finalized chunks (refcount zero or timeout)
            for chunk_id in list(self.ref_counts.keys()):
                is_timed_out = (time.time() - self.dispatch_times.get(chunk_id, time.time())) > 5.0 # 5s timeout
                if self.ref_counts[chunk_id] <= 0 or is_timed_out:
                    self.finalize_chunk(chunk_id)

    def dispatch_chunk(self, chunk_id, start, num):
        task = {"chunk_id": chunk_id, "start": start, "num": num, "shm_name": self.shm.name, "shape": self.buffer.shape, "dtype": self.buffer.dtype}
        num_workers = 0
        for queue in task_queues.values():
            queue.put(task)
            num_workers += 1
        
        self.ref_counts[chunk_id] = num_workers
        self.dispatch_times[chunk_id] = time.time()
        print(f"Dispatched {chunk_id} with refcount {num_workers}")

    def finalize_chunk(self, chunk_id):
        print(f"Finalizing {chunk_id}...")
        collected_results = self.results.pop(chunk_id, [])
        self.slice_and_encode_function(chunk_id, collected_results)
        
        # Cleanup
        del self.ref_counts[chunk_id]
        del self.dispatch_times[chunk_id]
        # The buffer space is implicitly freed by the moving write_head

    def slice_and_encode_function(self, chunk_id, results):
        # Logic to decide encoding quality based on collected results
        # e.g., if any result contains "encode_quality": "HIGH", use high bitrate.
        # Call ffmpeg subprocess here.
        pass
        
    def start_workers(self):
        Process(target=analysis_worker_loop, args=(task_queues['ocr'], return_queue, "ocr_analyzer_function")).start()
        Process(target=analysis_worker_loop, args=(task_queues['visual'], return_queue, "visual_analyzer_function")).start()

        
# --- Inside MemoryAndDispatchManager class ---

    def slice_and_encode_function(self, chunk_id, collected_results):
        """Decides what to save based on a collection of analysis results for a chunk."""
        
        # --- Default Encoding Parameters ---
        encoding_instructions = {
            "video_quality": "VERY_LOW", # e.g., 240p, 1 frame per 8 seconds
            "video_time_range": None,    # None means encode the whole chunk
            "save_stills": []            # List of (timestamp, quality) tuples
        }

        # --- Rule Application Loop ---
        # The rules are additive and escalate the quality requirements.
        for result_group in collected_results:
            # result_group is from one worker (e.g., ocr_analyzer)
            # and may contain multiple events (e.g., multiple keyframes)
            for event in result_group['data']:
                
                # Rule 1: OCR Saliency Detected
                if event['type'] == 'OCR_KEYFRAME':
                    print(f"Applying OCR Rule for chunk {chunk_id}")
                    encoding_instructions["save_stills"].append({
                        "timestamp": event['timestamp'],
                        "quality": event['image_quality_requirement'], # "LOSSLESS"
                        "reason": "ocr_event",
                        "metadata": {
                            "text": event['ocr_text'],
                            "latents_pointer": self.save_latents(event['latents'])
                        }
                    })

                # Rule 2: Visual Keyframe Saliency Detected
                if event['type'] == 'VISUAL_KEYFRAME':
                    print(f"Applying Visual Keyframe Rule for chunk {chunk_id}")
                    # Escalate video quality to HIGH
                    encoding_instructions["video_quality"] = "HIGH" # e.g., 720p, 30fps
                    
                    # Define the 4-second high-quality window centered on the event
                    # This may span across chunk boundaries, which the encoder must handle.
                    start_time = event['timestamp'] - 2.0
                    end_time = event['timestamp'] + 2.0
                    encoding_instructions["video_time_range"] = (start_time, end_time)

        # --- Final Action ---
        # After evaluating all results, execute the final encoding instructions.
        
        if encoding_instructions["video_quality"] == "HIGH":
            # Call ffmpeg to encode the specific time range at high quality.
            # This is an async call; it returns a future/promise.
            encoding_future = self.video_encoder.encode_slice(
                time_range=encoding_instructions["video_time_range"],
                quality="HIGH"
            )
            # The success of this future will be the final signal to release memory.
        else:
            # Encode the default low-quality video for archival purposes.
            encoding_future = self.video_encoder.encode_slice(
                time_range=get_chunk_time_range(chunk_id),
                quality="VERY_LOW"
            )

        for still_job in encoding_instructions["save_stills"]:
            # Save the high-quality still image. This is fast and can be done synchronously.
            self.frame_archiver.save_frame(
                timestamp=still_job['timestamp'],
                quality=still_job['quality']
            )
            # Log the metadata (text, latents pointer) in our main event list.

        # IMPORTANT: The memory for this chunk is not truly free until 'encoding_future' completes.
        # The manager must track this future and only mark the buffer space as writable
        # upon its successful completion.


# --- Process B, C...: Analysis Worker ---

def analysis_worker_loop(task_queue, return_queue, analysis_function_name):
    # This loop runs in a separate process
    existing_shm = None
    buffer = None

    while True:
        task = task_queue.get()
        
        # Attach to shared memory only once or if it changes
        if existing_shm is None or existing_shm.name != task['shm_name']:
            existing_shm = shared_memory.SharedMemory(name=task['shm_name'])
            buffer = np.ndarray(task['shape'], dtype=task['dtype'], buffer=existing_shm.buf)
            
        # Get a zero-copy view of the relevant frames
        frames_to_process = buffer[task['start'] : task['start'] + task['num']]
        
        # --- Run the actual analysis ---
        # result_data = globals()[analysis_function_name](frames_to_process)
        result_data = {"example": "data"} # Placeholder
        
        return_message = {"chunk_id": task['chunk_id'], "source": analysis_function_name, "data": result_data}
        return_queue.put(return_message)
# --- Entry Point ---

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    import os
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # --- Configuration ---
    RECORDING_WINDOW_NAME = "My Netgame 2015" # The title of the game window
    OUTPUT_PATH = "./game_capture_run_1"

    # --- Main Application Class: The Orchestrator ---

    # --- Data Structures & Queues ---
    task_queues = {"ocr": Queue(), "visual": Queue()}
    return_queue = Queue()

    # --- Process A: MemoryAndDispatchManager ---

    app = GameCaptureApp()
    try:
        app.run()
    except KeyboardInterrupt:
        app.shutdown()
    finally:
        if app.is_running: # Ensure shutdown runs even on other errors
            app.shutdown()