from affective_dialogue_system.interest import score_interest


def main() -> None:
    text = "Siento pinchazos en el brazo y un poco de mareo."
    print(score_interest(text))


if __name__ == "__main__":
    main()

