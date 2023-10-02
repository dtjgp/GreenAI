'''
This is a test python file that used to practice the codecarbon library.
'''

from codecarbon import EmissionsTracker

tracker = EmissionsTracker()
tracker.start()
try:
    for i in range(10000000):
        print(i)
    _ = 1 + 1
finally:
    tracker.stop()