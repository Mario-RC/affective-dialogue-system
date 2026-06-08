from affective_dialogue_system.emotion import EmotionClassifier


def main() -> None:
    classifier = EmotionClassifier()
    prediction = classifier.predict("I cannot believe this happened again.")
    print(prediction)


if __name__ == "__main__":
    main()

