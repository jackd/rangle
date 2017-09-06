from estimator import get_estimator
from data import get_input_fn


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='training script')
    parser.add_argument('model_name', nargs='?', type=str, default='base')
    parser.add_argument('--batch_size', '-b', nargs='?', type=int, default=64)
    parser.add_argument('--max_steps', '-s', nargs='?', type=int, default=1e6)
    args = parser.parse_args()

    estimator = get_estimator(args.model_name)
    input_fn = get_input_fn(args.batch_size, shuffle=True)
    estimator.train(input_fn, max_steps=args.max_steps)
