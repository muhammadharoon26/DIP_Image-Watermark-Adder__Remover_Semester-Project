Of course! Here is a professional `README.md` for your project.

I've structured it based on best practices, including a project overview, feature list, technical stack, detailed setup instructions (including a generated `requirements.txt`), and API usage examples. This will make your project easy for others to understand, use, and contribute to.

---

# Advanced Image Watermark Tool

[![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Django](https://img.shields.io/badge/Django-5.2-green?style=for-the-badge&logo=django)](https://www.djangoproject.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-blue?style=for-the-badge&logo=opencv)](https://opencv.org/)
[![LaMa Inpainting](https://img.shields.io/badge/AI%20Model-LaMa%20Inpainting-orange?style=for-the-badge)](https://github.com/saic-mdal/lama)

A powerful web application for adding and removing watermarks from images, developed as a semester project for Digital Image Processing (DIP). This tool leverages both classic computer vision techniques and state-of-the-art AI inpainting models to provide high-quality results through a simple and intuitive web interface.

## 🌟 Key Features

-   **AI-Powered Watermark Removal**: Utilizes the **LaMa (Large Mask Inpainting)** model to intelligently detect and erase watermarks, reconstructing the underlying image content with impressive accuracy.
-   **Automatic Mask Generation**: Employs image processing techniques like Canny edge detection and morphological operations to automatically create a mask of the watermark, eliminating the need for manual selection.
-   **Versatile Watermark Addition**:
    -   Add customizable **text watermarks**.
    -   Add **image-based watermarks** (e.g., logos).
    -   Properly handles transparency (alpha channels) in watermark images.
-   **Multiple Input Methods**:
    -   Upload images directly from your device.
    -   Process images directly from a URL.
-   **Web Interface & API**:
    -   An easy-to-use, responsive UI built with Bootstrap.
    -   Simple RESTful endpoints for programmatic access.

## 📸 Screenshot


*(A placeholder image representing the application's UI)*

## 🛠️ Technology Stack

-   **Backend**: Django
-   **Image Processing**:
    -   **OpenCV**: For core image manipulation, text rendering, and image blending.
    -   **Simple-Lama-Inpainting**: A user-friendly wrapper for the LaMa inpainting model.
    -   **Pillow (PIL)**: For image format conversions.
    -   **NumPy**: For efficient numerical operations on image data.
-   **Frontend**: HTML, Bootstrap 5, Vanilla JavaScript (Fetch API)
-   **Database**: SQLite (default)

---

## 🚀 Getting Started

Follow these instructions to set up and run the project locally.

### Prerequisites

-   Python 3.9+ and `pip`
-   Git

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/DIP_Image-Watermark-Adder-Remover.git
cd DIP_Image-Watermark-Adder-Remover/image_watermark_adder_remover
```

### 2. Create and Activate a Virtual Environment

It is highly recommended to use a virtual environment to manage project dependencies.

-   **On macOS/Linux:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
-   **On Windows:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

### 3. Install Dependencies

The `simple-lama-inpainting` library requires PyTorch. The installation can be large, so be prepared for a download.

Create a `requirements.txt` file in the `image_watermark_adder_remover` directory with the following content:

**`requirements.txt`**
```
Django
numpy
opencv-python
requests
scikit-image
Pillow
# The following are required for AI-based inpainting
simple-lama-inpainting
torch
torchvision
```

Now, install the packages using pip:
```bash
pip install -r requirements.txt
```

### 4. Apply Database Migrations

This will set up the initial database schema (using SQLite by default).

```bash
python manage.py migrate
```

### 5. Run the Development Server

Start the Django development server.

```bash
python manage.py runserver
```

The application will be available at `http://127.0.0.1:8000/`.

---

## 🕹️ Usage

### Web Interface

Navigate to `http://127.0.0.1:8000/` in your web browser.

-   **Remove Watermark Tab**: Upload an image or provide a URL to remove a watermark. The processed image will be displayed with a download link.
-   **Add Watermark Tab**: Upload a main image, specify optional text, and/or upload a watermark image. The final result can be downloaded.

### API Endpoints

You can also interact with the service programmatically.

#### 1. Remove Watermark

-   **`POST /remove_watermark/`**

    Upload an image file to have its watermark removed.

    **Example using `curl`:**
    ```bash
    curl -X POST -F "image=@/path/to/your/image.jpg" http://127.0.0.1:8000/remove_watermark/ --output result_removed.jpg
    ```

-   **`GET /remove_watermark/`**

    Process an image from a public URL.

    **Example URL:**
    ```
    http://127.0.0.1:8000/remove_watermark/?image_url=https://example.com/image_with_watermark.png
    ```

#### 2. Add Watermark

-   **`POST /add_watermark/`**

    Upload a main image and optionally a watermark image and/or text.

    **Example using `curl` (Text and Image Watermark):**
    ```bash
    curl -X POST \
      -F "main_image=@/path/to/main.jpg" \
      -F "watermark_image=@/path/to/logo.png" \
      -F "text=My Copyright" \
      http://127.0.0.1:8000/add_watermark/ --output result_watermarked.jpg
    ```

---

## 🧠 How It Works

### Watermark Removal

The watermark removal process is a two-step pipeline:

1.  **Mask Generation (`_generate_watermark_mask`)**:
    -   The input image is converted to grayscale to focus on luminance information.
    -   **Canny Edge Detection** is applied to identify high-contrast regions, which often correspond to the edges of a watermark.
    -   **Morphological operations** (dilation and closing) are used to connect the detected edges and form a solid binary mask that covers the entire watermark.

2.  **AI Inpainting (`_remove_watermark_with_lama`)**:
    -   The original image and the generated mask are passed to the **LaMa (Large Mask Inpainting)** model.
    -   LaMa is a deep learning model specifically trained to fill in large missing regions (the masked watermark) in an image in a semantically consistent and visually plausible way. It excels at regenerating textures and structures that were previously obscured.

### Watermark Addition

Watermark addition is handled by OpenCV:

-   **Text Watermark**: The `cv2.putText()` function renders the specified text onto a copy of the image. `cv2.addWeighted()` is then used to blend this overlay with the original image, creating a semi-transparent effect.
-   **Image Watermark**: The watermark image is resized relative to the main image. If the watermark has an alpha channel (for transparency), it is used to blend the watermark smoothly onto the target region. Otherwise, `cv2.addWeighted()` is used for a simple opacity-based blend.

## 📜 License

This project is open-source. Please feel free to use, modify, and distribute it. If you plan to release it publicly, consider adding an open-source license like MIT.

---

**Note**: This tool is provided for educational purposes. Please respect copyright laws and only process images that you have the legal right to modify.