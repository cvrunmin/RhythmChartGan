import numpy as np
import matplotlib.pyplot as plt
import argparse
import os.path
from matplotlib.colors import ListedColormap


def second_to_tick(sec, bpm):
    from sus.sus_score import TICK_PER_BEAT

    return sec * TICK_PER_BEAT * bpm / 60


def draw_alpha(chart, bpm=None):
    assert chart.ndim == 2
    fig = plt.figure('chart_alpha')

    t = np.arange(0, chart.shape[0]) / 50
    if bpm is not None:
        t = second_to_tick(t, bpm)

    mesh = plt.pcolormesh(np.arange(1, 13), t, chart, shading='nearest', figure=fig)
    mesh.axes.set_xlabel('Note Location')
    if bpm is not None:
        mesh.axes.set_ylabel('time (tick)')
    else:
        mesh.axes.set_ylabel('time (s)')
    mesh.axes.set_title('Chart Alpha')
    colorbar = fig.colorbar(mesh)
    colorbar.ax.get_yaxis().labelpad = 15
    return fig


def draw_group(chart, bpm=None):
    assert chart.ndim == 2
    mappings = {
        'None': 'k',
        'Group 1':'r',
        'Group 2': 'b',
        'Group 1&2': 'm'
    }
    chart = chart.astype(int)

    fig = plt.figure('chart_group')
    max_color = chart.max() + 1

    t = np.arange(0, chart.shape[0]) / 50
    if bpm is not None:
        t = second_to_tick(t, bpm)

    mesh = plt.pcolormesh(np.arange(1, 13), t, chart, shading='nearest', cmap=ListedColormap(mappings.values(), N=max_color), figure=fig)
    mesh.axes.set_xlabel('Note Location')
    if bpm is not None:
        mesh.axes.set_ylabel('time (tick)')
    else:
        mesh.axes.set_ylabel('time (s)')
    mesh.axes.set_title('Chart Group')
    colorbar = fig.colorbar(mesh)
    colorbar.ax.get_yaxis().set_ticks([])
    for i, lbl in enumerate(list(mappings.keys())[:max_color]):
        colorbar.ax.text(1.5, (2 * i + 1) / (2 * max_color), lbl, ha='left', va='center', transform=colorbar.ax.transAxes)
    colorbar.ax.get_yaxis().labelpad = 15
    return fig


def draw_color(chart, bpm=None):
    assert chart.ndim == 2

    mappings = {
        'None': 'k',
        'Click/Slide':'b',
        'Slide Checkpoint': 'g',
        'Flick Up': 'salmon', 
        'Flick Left': 'orangered', 
        'Flick Right': 'r'
    }
    chart = chart.astype(int)

    fig = plt.figure('chart_color')
    max_color = chart.max() + 1
    t = np.arange(0, chart.shape[0]) / 50
    if bpm is not None:
        t = second_to_tick(t, bpm)

    mesh = plt.pcolormesh(np.arange(1, 13), t, chart, shading='nearest', cmap=ListedColormap(mappings.values(), N=max_color), figure=fig)
    mesh.axes.set_xlabel('Note Location')
    if bpm is not None:
        mesh.axes.set_ylabel('time (tick)')
    else:
        mesh.axes.set_ylabel('time (s)')
    mesh.axes.set_title('Chart Color')
    colorbar = fig.colorbar(mesh)
    colorbar.ax.get_yaxis().set_ticks([])
    for i, lbl in enumerate(list(mappings.keys())[:max_color]):
        colorbar.ax.text(1.5, (2 * i + 1) / (2 * max_color), lbl, ha='left', va='center', transform=colorbar.ax.transAxes)
    colorbar.ax.get_yaxis().labelpad = 15
    return fig


def draw(path, *, is_prsk=True, threshold_alpha=True):
    import chart_renderer

    if is_prsk:
        sus_chart = chart_renderer.read_prsk_sus(path)
    else:
        sus_chart = chart_renderer.read_sus(path)
    notes = chart_renderer.bake_note_data(sus_chart.to_raw_note())
    return np.stack(chart_renderer.draw(notes, frame_length=20, threshold_alpha=threshold_alpha))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath', type=str)
    parser.add_argument('-c', '--note-type', choices=['alpha', 'a', 'group', 'g', 'color', 'c'])
    parser.add_argument('-D', dest='threshold_alpha', action='store_false')
    parser.add_argument('--bpm', type=float)
    parser.add_argument('--no_filter', dest='is_prsk', action='store_false')
    args = parser.parse_args()

    if not os.path.isfile(args.filepath):
        print('ERROR: file not exists')
        exit(1)

    if os.path.splitext(args.filepath)[1].casefold() == '.sus'.casefold():
        charts = draw(args.filepath, is_prsk=args.is_prsk, threshold_alpha=args.threshold_alpha)
    else:
        charts = np.load(args.filepath)
    
    if charts.ndim not in [2, 3]:
        print(f'ERROR: loaded chart has unknown dimension. wanted: 2/3, actual: {charts.ndim}')
    if charts.ndim == 2 or (charts.ndim == 3 and charts.shape[0] == 1):
        if charts.ndim == 3 and charts.shape[0] == 1:
            charts = charts[0]
        if args.note_type is None:
            print('ERROR: loaded chart is mono channel, but channel type is not given.')
        if args.note_type in ['alpha', 'a']:
            draw_alpha(charts, bpm=args.bpm)
        elif args.note_type in ['group', 'g']:
            draw_group(charts, bpm=args.bpm)
        else:
            draw_color(charts, bpm=args.bpm)
        plt.show()
    if charts.ndim == 3:
        if charts.shape[0] == 3:
            if args.note_type in ['alpha', 'a']:
                draw_alpha(charts[0], bpm=args.bpm)
            elif args.note_type in ['group', 'g']:
                draw_group(charts[1], bpm=args.bpm)
            elif args.note_type in ['color', 'c']:
                draw_color(charts[2], bpm=args.bpm)
            else:
                draw_alpha(charts[0], bpm=args.bpm)
                draw_group(charts[1], bpm=args.bpm)
                draw_color(charts[2], bpm=args.bpm)
            plt.show()
        elif charts.shape[0] == 2:
            if args.note_type in ['alpha', 'a']:
                print('given chart has no alpha channel')
            elif args.note_type in ['group', 'g']:
                draw_group(charts[0], bpm=args.bpm)
                plt.show()
            elif args.note_type in ['color', 'c']:
                draw_color(charts[1], bpm=args.bpm)
                plt.show()
            else:
                draw_group(charts[0], bpm=args.bpm)
                draw_color(charts[1], bpm=args.bpm)
                plt.show()
        else:
            print(f'ERROR: unknown channel size {charts.shape[0]}')



if __name__ == '__main__':
    main()