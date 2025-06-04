import cv2
import sys

# Open a file to write output
with open('opencv_test_output.txt', 'w') as f:
    f.write(f"Python version: {sys.version}\n")
    f.write(f"OpenCV version: {cv2.__version__}\n")

    # Try to open a simple image or create one
    try:
        # Create a simple black image
        img = cv2.imread('reference_image_left.jpg')
        if img is None:
            # If image doesn't exist, create a blank one
            f.write("Could not load reference image, creating a blank one\n")
            img = cv2.imread('nonexistent.jpg')
            if img is None:
                img = cv2.cvtColor(cv2.imread('nonexistent.jpg'), cv2.COLOR_BGR2GRAY)
            
        # Show image dimensions if available
        if img is not None:
            f.write(f"Image dimensions: {img.shape}\n")
        else:
            f.write("Could not load or create an image\n")
            
        # Try to save a test image if it exists
        if img is not None:
            # Save a test image
            cv2.imwrite('opencv_test_output.jpg', img)
            f.write("Successfully wrote test image\n")
            
    except Exception as e:
        f.write(f"Error: {e}\n")

    f.write("OpenCV test completed\n") 