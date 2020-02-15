"""Code from https://gist.github.com/ihsgnef/f13c35cd46624c8f458a4d23589ac768"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def colorize(words, color_array):
    # words is a list of words
    # color_array is an array of numbers between 0 and 1 of length equal to words
    cmap = matplotlib.cm.get_cmap('jet')
    template = '<span class="barcode"; style="color: black; background-color: {}">{}</span>'
    colored_string = ''
    for word, color in zip(words, color_array):
        color = matplotlib.colors.rgb2hex(cmap(color)[:3])
        colored_string += template.format(color, '&nbsp' + word + '&nbsp')
    return colored_string

def colorize_with_label(idx, words, color_array, label, pred):
    # words is a list of words
    # color_array is an array of numbers between 0 and 1 of length equal to words
    cmap = matplotlib.cm.get_cmap('RdBu')
    template = '<span class="barcode"; style="color: black; background-color: {}">{}</span>'
    colored_string = ''
    for word, color in zip(words, color_array):
        color = matplotlib.colors.rgb2hex(cmap(color)[:3])
        colored_string += template.format(color, '&nbsp' + word + '&nbsp')
    return "<p>Idx {} Label: {} Pred: {} Grad Norm: ".format(idx,label, pred) + colored_string + "</p>"

if __name__ == "__main__":
    words = 'The quick brown fox jumps over the lazy dog'.split()
    color_array = np.random.rand(len(words))
    s = colorize(words, color_array)

    # to display in ipython notebook
    from IPython.display import display, HTML

    display(HTML(s))

    # or simply save in an html file and open in browser
    with open('colorize.html', 'w') as f:
        f.write(s)