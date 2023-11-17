import numpy as np
import matplotlib.pyplot as plt

class Street:
    def __init__(self, x_start, y_start, x_end, y_end, n_lanes=2, lane_width=5):
        self.x_start = x_start
        self.y_start = y_start
        self.x_end = x_end
        self.y_end = y_end
        self.n_lanes = n_lanes
        self.lane_width = lane_width
        self.length = np.sqrt((x_end - x_start)**2 + (y_end - y_start)**2)
        self.angle = np.arctan2(y_end - y_start, x_end - x_start)
        self.R01 = np.matrix([[np.cos(self.angle), -np.sin(self.angle)], [np.sin(self.angle), np.cos(self.angle)]])
        self.R10 = self.R01.T

        self.init_lanes()

    def init_lanes(self):
        self.lanes = [] 
        if self.n_lanes%2 == 0:
            self.lanes.append(Lane(self.x_start, self.y_start - self.lane_width/2 * self.n_lanes/2, self.x_end, self.y_end - self.lane_width/2 * self.n_lanes/2))
            for i in range(1, self.n_lanes):
                self.lanes.append(Lane(self.x_start, self.lanes[0].y_start + i * self.lane_width, self.x_end, self.lanes[0].y_end + i * self.lane_width))
    
    def xy_to_s(self, x, y):
        return (x - self.x_start) * np.cos(self.angle) + (y - self.y_start) * np.sin(self.angle)\

    def plot_street(self):
        for i in range(self.n_lanes):
            plt.plot([self.lanes[i].x_start, self.lanes[i].x_end], [self.lanes[i].y_start, self.lanes[i].y_end], 'b--')
            print(self.lanes[i].length)
        plt.plot([self.x_start, self.x_end], [self.y_start, self.y_end], 'r--')
        
class Lane:
    def __init__(self, x_start, y_start, x_end, y_end):
        self.x_start = x_start
        self.y_start = y_start
        self.x_end = x_end
        self.y_end = y_end
        self.length = np.sqrt((x_end - x_start)**2 + (y_end - y_start)**2)
        self.angle = np.arctan2(y_end - y_start, x_end - x_start)

    def s_to_xy(self, s):
        return self.x_start + s * np.cos(self.angle), self.y_start + s * np.sin(self.angle)
        

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    street = Street(0, 0, 100, 0)
    for i in range(street.n_lanes):
        plt.plot([street.lanes[i].x_start, street.lanes[i].x_end], [street.lanes[i].y_start, street.lanes[i].y_end], 'b--')
        print(street.lanes[i].length)
    plt.plot([street.x_start, street.x_end], [street.y_start, street.y_end], 'r--')
    plt.show()