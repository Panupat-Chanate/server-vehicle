import math
import csv

side = ['top-l', 'top-c', 'top-r', 'bottom-l', 'bottom-c', 'bottom-r', 'left', 'right']

def distance_veh(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def process_distances(data) :
    for obj1 in data:
        for obj2 in data:
            if obj1['id'] != obj2['id']:
                obj1.setdefault('distances_to_other_ids', {})
                for s1 in side:
                    for s2 in side:
                        newKey = s1+"-"+s2
                        if 'distances_to_other_ids' not in obj1:
                            obj1['distances_to_other_ids'] = {}
                        if obj2['id'] not in obj1['distances_to_other_ids']:
                            obj1['distances_to_other_ids'][obj2['id']] = {}
                        obj1['distances_to_other_ids'][obj2['id']][newKey] = round(distance_veh(obj1[s1], obj2[s2]), 3)

    for i in range(len(data)):
        min_distances = {}
        for id, distances in data[i]['distances_to_other_ids'].items():
            min_distance = min(distances.values())
            min_distances[id] = min_distance

        data[i] = {
            "id" : data[i]['id'],
            "distances_to_other_ids" : min_distances
        }
    # print(data[0])

    return data

# process_distances(data)