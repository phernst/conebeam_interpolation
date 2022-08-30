import json

from matplotlib import pyplot as plt


def main():
    wdlr = {}
    for wd in range(2, 8):
        with open(f'lr_finder_wd1e-{wd}.json', 'r', encoding='utf-8') as lrfile:
            wdlr[wd] = json.load(lrfile)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for wd, dct in wdlr.items():
        ax.plot(dct['lr'], dct['loss'], label=f'1e-{wd}')

    ax.set_xscale('log')
    ax.legend()
    plt.show()


if __name__ == '__main__':
    main()
