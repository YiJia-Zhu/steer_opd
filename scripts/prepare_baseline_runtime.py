"""Select the unsteered baseline variant of the training runtime."""
from prepare_training_runtime import prepare as prepare_training_runtime


def prepare():
    return prepare_training_runtime(baseline=True)


if __name__ == '__main__':
    print(prepare())
