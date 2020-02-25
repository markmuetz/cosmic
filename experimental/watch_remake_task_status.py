import sys
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class RemakeTaskStatusEventHandler(FileSystemEventHandler):
    def on_any_event(self, event):
        super().on_any_event(event)
        print(event)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else '.'
    event_handler = RemakeTaskStatusEventHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()
    try:
        while observer.is_alive():
            observer.join(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

