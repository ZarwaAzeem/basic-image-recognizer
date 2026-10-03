# Basic Image Recognition 🔢

A basic image recognition project in Python that recognizes handwritten digits (0 to 9) using scikit-learn.

## Goal

Implement a basic image recognition task using available libraries.

## What It Does

1. **Loads** the digits dataset that ships with scikit-learn (1,797 handwritten digit images, 8x8 pixels each)
2. **Trains** a Support Vector Machine (SVM) on 80% of the images
3. **Evaluates** the model on 20% of the images it has never seen
4. **Recognizes** 10 sample images and prints the predicted digit with a confidence score
5. **Displays** the results clearly: a results table, a text-art view of one image, and a saved picture of all predictions

## Skills Demonstrated

- Using AI libraries
- Understanding model outputs (predictions and confidence scores)

## Requirements

- Python 3.x
- scikit-learn
- matplotlib

Install everything with:

```bash
pip install -r requirements.txt
```

## How to Run

```bash
python recognizer.py
```

The script prints the results in the terminal and saves `predictions.png` in the same folder.

## Sample Output

```
Model accuracy on 360 unseen images: 98.89%

=== Sample Recognition ===
#   Actual  Predicted  Confidence  Result
1   5       5          96.7%       OK
2   2       2          95.3%       OK
3   8       8          95.6%       OK
4   1       1          71.8%       OK
5   7       7          94.5%       OK
...
```

Exact numbers may vary slightly with different library versions.

## How It Works

Each image is an 8x8 grid of pixel intensities, flattened into 64 numbers. The SVM learns which pixel patterns belong to each digit during training. For a new image, it predicts the most likely digit, and the confidence score shows how strongly the model favors that choice. Lower confidence (like the 71.8% above) usually means the digit looks ambiguous.

## Project Structure

```
image-recognition/
├── recognizer.py
├── requirements.txt
└── README.md
```

## Future Improvements

- Let users draw or upload their own digit image
- Use a pre-trained deep learning model (for example, MobileNet) to recognize everyday objects
- Try text recognition (OCR) with `pytesseract`
- Compare SVM with other models such as KNN or Random Forest

## License

This project is open source and available for learning purposes.
