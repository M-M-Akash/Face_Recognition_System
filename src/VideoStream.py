import cv2
from threading import Thread, Lock


class VideoStream(object):
    def __init__(self, src=0):
        """_Initializes the class instance with the video source. 
        By default, the source is set to 0, which means the default camera (usually the built-in webcam)._

        Args:
            src (str, optional): _camera source link_. Defaults to 0.
        """
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.started = False
        self.read_lock = Lock()

    def start(self):
        """_Starts a thread that continuously reads frames from the video source and updates the frame buffer. 
        The update method is called by this thread._

        Returns:
            _object_: _description_
        """
        if self.started:
            print("already started!!")
            return None
        self.started = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        """_Continuously reads frames from the video source and updates the frame buffer. 
        This method is called by the thread started by the start method._
        """
        while self.started:
            (grabbed, frame) = self.stream.read()
            self.read_lock.acquire()
            self.grabbed, self.frame = grabbed, frame
            self.read_lock.release()

    def read(self):
        """_Returns the latest frame from the frame buffer._

        Returns:
            _type_: _description_
        """
        self.read_lock.acquire()
        frame = self.frame
        self.read_lock.release()
        return frame

    def stop(self):
        """_Stops the thread that is reading frames from the video source._
        """
        self.started = False
        self.thread.join()

    def exit(self):
        """_Releases the video source._
        """
        self.stream.release()