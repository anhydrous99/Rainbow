import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os


def directory_check(directory):
    if not os.path.isdir(directory):
        os.mkdir(directory)


def save(data, save_path):
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)


def plot(data, save_path, name, runs=1):
    df = pd.DataFrame(data)
    print(df)
    sns.set_style('darkgrid')
    fix, ax = plt.subplots()
    ax.set_title(name)
    clrs = sns.color_palette('hls', runs)
    sns.lineplot(x='step', y='reward', data=df, c=clrs[0])
    plt.savefig(save_path)


def main():
    parser = argparse.ArgumentParser(description='Deep Q-network (DQN) Plotter')
    parser.add_argument('--data', type=str, default='data/CartPole-v1.csv')
    parser.add_argument('--png', type=str, default='plot.png')
    parser.add_argument('--name', type=str, default='CartPole-v1')
    args = parser.parse_args()

    assert os.path.exists(args.data)
    df = pd.read_csv(args.data)
    plot(df, args.png, args.name)


if __name__ == '__main__':
    main()
