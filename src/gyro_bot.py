from disturbances import disturbance

def main():
    disturbances = disturbance(averages = [1, .5, 0], spreads = [.25, .25, 1], max = 1, jerk_mean = 3, jerk_spread = 1)

    events = disturbances.generate_events(10)
    disturbances.plot_events(events, .05)

if __name__ == '__main__':
    main()