import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

SHADES = " .:-=+*#%@"  # light to dark, used for terminal display


def ascii_digit(image):
    """Draw an 8x8 digit image in the terminal using text characters."""
    lines = []
    for row in image:
        lines.append("".join(SHADES[int(v / 16 * (len(SHADES) - 1))] * 2 for v in row))
    return "\n".join(lines)


def main():
    # Step 1: Load a small image dataset (1,797 handwritten digits, 8x8 pixels)
    digits = load_digits()
    print("=== Dataset ===")
    print("Images:", len(digits.images))
    print("Image size:", digits.images[0].shape, "pixels")
    print("Classes:", [int(c) for c in digits.target_names])

    # Step 2: Split the data and train a Support Vector Machine (SVM)
    X_train, X_test, y_train, y_test, img_train, img_test = train_test_split(
        digits.data, digits.target, digits.images,
        test_size=0.2, random_state=42, stratify=digits.target,
    )
    model = SVC(gamma=0.001, probability=True)
    model.fit(X_train, y_train)

    # Step 3: Measure how well the model recognizes unseen images
    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"\nModel accuracy on {len(y_test)} unseen images: {accuracy:.2%}")

    # Step 4: Recognize sample images and show the results clearly
    sample_count = 10
    samples = X_test[:sample_count]
    predictions = model.predict(samples)
    confidences = model.predict_proba(samples).max(axis=1)

    print("\n=== Sample Recognition ===")
    print(f"{'#':<4}{'Actual':<8}{'Predicted':<11}{'Confidence':<12}Result")
    for i in range(sample_count):
        correct = "OK" if predictions[i] == y_test[i] else "WRONG"
        print(f"{i + 1:<4}{y_test[i]:<8}{predictions[i]:<11}{confidences[i]:<12.1%}{correct}")

    # Show the first sample as text art
    print("\nFirst sample image (as the model sees it):")
    print(ascii_digit(img_test[0]))
    print(f"-> Model says: {predictions[0]} ({confidences[0]:.1%} confident)")

    # Save a picture of all sample predictions
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    for i, ax in enumerate(axes.ravel()):
        ax.imshow(img_test[i], cmap="gray_r")
        color = "green" if predictions[i] == y_test[i] else "red"
        ax.set_title(f"Pred: {predictions[i]} ({confidences[i]:.0%})", color=color)
        ax.axis("off")
    plt.suptitle("Handwritten Digit Recognition")
    plt.tight_layout()
    plt.savefig("predictions.png")
    print("\nSaved predictions to predictions.png")


if __name__ == "__main__":
    main()
