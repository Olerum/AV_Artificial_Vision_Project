import sys
import cv2
from gui import Application
from homography import run_localization


if __name__ == "__main__":
    app = Application()
    app.mainloop()

def main():
    # If two args provided, run CLI localization
    if len(sys.argv) == 3:
        image_path, homography_path = sys.argv[1], sys.argv[2]
        run_localization(image_path, homography_path)
    else:
        # Otherwise launch GUI
        app = Application()
        app.mainloop()


if __name__ == "__main__":
    main()
