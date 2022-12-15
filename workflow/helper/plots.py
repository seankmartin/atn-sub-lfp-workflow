import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
import numpy as np


def plot_pos_over_time(x, y, rate=2, skip_rate=27, save=False):
    fig, ax = plt.subplots(figsize=(8, 8))
    (scatter,) = ax.plot([], [], "ko")
    (scatter2,) = ax.plot([], [], "bx", markersize=14, mew=10)
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)
    frame_time = 0
    n_pressed = 0
    do_skip = False
    n_quit = []
    saved_times = []

    print(
        "Press space to save time, d to delete last time, "
        + "q to quit, c to exit program."
    )

    def init():
        global frame_time
        global n_pressed
        global do_skip
        global n_quit
        frame_time = 0
        n_pressed = 0
        do_skip = False
        n_quit = []
        ax.set_xlim(0, max(x) + 1)
        ax.set_ylim(max(y) + 1, 0)
        time_text.set_text("")
        return scatter, time_text

    def update(frame):
        global frame_time
        global do_skip
        if do_skip:
            # skip 27 seconds ahead
            frame += skip_rate * 40
        scatter.set_data(x[0:frame], y[0:frame])

        other_data_x = []
        other_data_y = []
        for t in saved_times:
            other_data_x.append(x[int(t * 50)])
            other_data_y.append(y[int(t * 50)])
        scatter2.set_data(other_data_x, other_data_y)

        # 50 is the sampling rate
        frame_time = frame / 50
        time_text.set_text(f"time = {frame_time}")
        return scatter, time_text

    def on_keyboard(event):
        global frame_time
        global n_pressed
        global do_skip
        if event.key == " ":
            t_save = frame_time - 0.2
            print("Saving {:.2f}".format(t_save))
            saved_times.append(t_save)
            n_pressed += 1
            if n_pressed == 3:
                do_skip = True
        if event.key == "d":
            if len(saved_times) > 0:
                ft = saved_times.pop()
                print("Deleting {:.2f}".format(ft))
                n_pressed -= 1
        if event.key == "c":
            print("Quitting program")
            n_quit.append("False")
            plt.close()

    num_samples = int(len(x) // rate)
    interval = int(20 // rate)

    frames = np.linspace(0, len(x), num=num_samples, dtype=np.uint32)

    ani = FuncAnimation(
        fig, update, frames=frames, interval=interval, init_func=init, repeat=True
    )

    if save:
        Writer = writers["ffmpeg"]
        writer = Writer(fps=15, metadata=dict(artist="Me"), bitrate=1800)
        ani.save("animated.mp4", writer)

    else:
        plt.gcf().canvas.mpl_connect("key_press_event", on_keyboard)
        plt.show()

    if len(n_quit) == 0:
        return saved_times
    else:
        return "QUIT"
