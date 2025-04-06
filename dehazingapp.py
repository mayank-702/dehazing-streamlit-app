import threading

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = None
        self.ready = False
        self.output_frame = None
        self.input_frame = None
        self.lock = threading.Lock()
        self.thread = None
        self.frame_count = 0

    def update_model(self, model):
        self.model = model
        self.ready = True
        # Start background thread
        if self.thread is None:
            self.thread = threading.Thread(target=self.process_loop, daemon=True)
            self.thread.start()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        with self.lock:
            self.input_frame = img.copy()
            display_frame = self.output_frame if self.output_frame is not None else img

        return av.VideoFrame.from_ndarray(display_frame, format="bgr24")

    def process_loop(self):
        while True:
            if not self.ready:
                continue

            with self.lock:
                frame = self.input_frame.copy() if self.input_frame is not None else None

            if frame is not None:
                try:
                    # Process only every 5 frames to save compute
                    self.frame_count += 1
                    if self.frame_count % 5 == 0:
                        input_tensor = preprocess_frame(frame)
                        output_tensor = self.model.predict(input_tensor, verbose=0)
                        result = postprocess_frame(output_tensor)
                        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

                        with self.lock:
                            self.output_frame = result
                except Exception as e:
                    print("Error in processing:", e)
